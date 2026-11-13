#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tq_codec.cu.h"

using namespace tensorflow;
using GPUDevice = Eigen::GpuDevice;

namespace {

#define RETURN_IF_CUDA_ERROR(call)                                              \
  do {                                                                          \
    const cudaError_t status_ = (call);                                         \
    if (status_ != cudaSuccess)                                                 \
      return errors::Internal(#call, " failed: ", cudaGetErrorString(status_)); \
  } while (0)

// Whether a slot tensor holds a frame (frames are never empty; a default
// Tensor is an empty one, which Tensor::IsInitialized counts as initialized).
bool Held(const Tensor& tensor) { return tensor.NumElements() > 0; }

// How a store keeps its float32 frames: exact ("raw"), or encoded where the
// link needs it with our temporal codec TQ-pack ("tq", tq_codec.cu.h: on the
// grid 2^-16, error <= 2^-17 per finite value).
enum class Codec { kRaw, kTq };

const char* CodecName(Codec codec) { return codec == Codec::kTq ? "tq" : "raw"; }

// Copy groups submitted to the D2H stream and not yet finished: one copying
// and one queued behind it, so the copy engine does not idle between groups.
constexpr size_t kInFlightGroups = 2;
// TQ: the ring arena's size in worst-case encoded frames (default; the
// environment variable V1_FRAME_STORE_RING_FRAMES overrides it), and the
// encodes whose placement the host has not read yet (more stay raw).
constexpr double kRingFrames = 6.0;
constexpr int64_t kRecords = 4096;
constexpr int kMaxOrder = tq::kOrder;
// The adaptive codec: the share of the link time left in the forward the
// projection counts on, the decay of the link-rate and encoded-length
// averages per completed group, and the encoded length assumed before any.
constexpr double kLinkMargin = 0.9;
constexpr double kDecay = 0.9;
constexpr double kInitialRatio = 0.55;

// The frames of one (GPU, key). Frame i is in one of these states:
// - on the GPU: `resident` holds it while it waits for a full group
//   (`staged_`), for admission to the copy stream (`queued_`), while its copy
//   out runs (`pending_`; these hold their own reference) or for good (`keep`:
//   at or above resident_from, chosen in place by a prefetch, or not admitted
//   before the forward ended). Once its copy out is done the frame is
//   `on_host` and, unless kept, released on the GPU;
// - encoded (float32 frames, codec "tq"): each offloaded frame
//   goes out raw as above or, when the link would not take every frame by the
//   end of the forward (see Decide), encoded: on `encode_` (`encoding_`) into
//   its `buffer` (a record in the ring arena), after which the raw frame is
//   released and the
//   buffer stands for the frame on the GPU, through admission and the copy
//   out, which reads the buffer's encoded bytes directly. Only an encoded
//   frame that stays on the GPU (not admitted before the forward ended) is
//   copied into a `packet` of its encoded length, on `encode_`.
//   A take decodes a buffer or packet directly; a frame back from the host is
//   decoded from a staging buffer (raw ones are copied back as such);
// - arriving: an H2D copy (or decode) into `arriving` was started; `arrived`
//   (shared by the frames of one group) marks its end;
// - taken / never recorded: neither tensor.
// TQ predicts frame s from frames s + 1 and s + 2 (`order` of them), so a
// frame chosen for encoding waits (`waiting_`) until those are recorded and
// is encoded from their raw values (kept alive by its entry, `recent_` and
// `resident`). Decoding needs their reconstructions, taken in reverse order:
// a reference still arriving, resident, or `reference`, the tensor a take
// handed out, kept while an encoded frame that predicts from it has not been
// decoded in the current reverse pass (for good, until the next rollout, when
// the frame is not on the host: an anchor). A reference that is gone but
// stored (host, packet or ring) is rebuilt by walking up the chain into three
// rotating scratch frames (a second pass over the host copies).
// Copies out are admitted by completion, not by the host loop: groups enter
// the D2H stream, earliest frames first, only while fewer than
// kInFlightGroups are unfinished (the store's thread admits the next one as
// each finishes), so frames the copy engine cannot take yet stay on the GPU
// rather than in an unbounded CUDA queue. When the GPU has passed the
// forward's last frame, the frames not yet admitted stay on the GPU for good
// (encoded ones as packets): the forward's memory peak is behind, and the
// reverse pass, which reaches the latest frames first, would only fetch them
// straight back. The forward never waits for the copies or the encoder: a
// frame the ring cannot take stays raw.
// Every GPU tensor here was allocated for the compute stream; it is released
// only after the copy reading or writing it is known complete (D2H, encode,
// pack, decode: event query; H2D: the compute stream waits on its event
// first), so the allocator can hand its memory to later compute work.
class FrameStore {
 public:
  Status Offload(OpKernelContext* c, int64_t index, int64_t capacity, int64_t resident_from,
                 int64_t group, Codec codec, const Tensor& frame) {
    std::unique_lock<std::mutex> lock(mu_);
    const cudaStream_t compute = c->eigen_device<GPUDevice>().stream();
    TF_RETURN_IF_ERROR(Initialize());
    TF_RETURN_IF_ERROR(Poll());
    TF_RETURN_IF_ERROR(Configure(c, compute, frame, capacity, codec));
    if (index == 0) TF_RETURN_IF_ERROR(StartRollout(compute));
    capacity_ = capacity;
    const bool offload = index < resident_from;
    const bool encode = offload && compressed() && Decide(c, compute, index, capacity, resident_from);
    if (encode) TF_RETURN_IF_ERROR(EnsureRing(c, compute));
    Slot& slot = slots_[index];
    TF_RETURN_IF_ERROR(Drop(index, compute));
    slot.generation = ++generation_;
    slot.resident = frame;
    latest_ = index;
    if (compressed()) {
      recent_.push_back({index, slot.generation, frame});
      if (recent_.size() > static_cast<size_t>(kMaxOrder)) recent_.pop_front();
    }
    if (offload) {
      Entry entry{index, slot.generation, frame, encode};
      if (encode) {
        slot.encoding = true;
        waiting_.push_back(std::move(entry));
      } else {
        staged_.push_back(std::move(entry));
        if (static_cast<int64_t>(staged_.size()) >= group) TF_RETURN_IF_ERROR(Flush(compute, false));
      }
    } else {
      slot.keep = true;
    }
    if (compressed()) TF_RETURN_IF_ERROR(Promote(compute, false, index));
    if (index == capacity - 1) {  // the forward's last frame
      TF_RETURN_IF_ERROR(Flush(compute));
      RETURN_IF_CUDA_ERROR(cudaEventRecord(forward_end_, compute));
      forward_end_recorded_ = true;
      recent_.clear();  // every waiting frame was encoded (its entry holds them)
    }
    return absl::OkStatus();
  }

  Status Take(OpKernelContext* c, int64_t index, int64_t prefetch, int64_t group, DataType dtype,
              const TensorShape& shape) {
    std::unique_lock<std::mutex> lock(mu_);
    TF_RETURN_IF_ERROR(Ready(index));
    if (dtype != dtype_ || shape != shape_)
      return errors::InvalidArgument("V1FrameStoreTake expects ", DataTypeString(dtype_), " ",
                                     shape_.DebugString(), " frames, not ",
                                     DataTypeString(dtype), " ", shape.DebugString());
    const cudaStream_t compute = c->eigen_device<GPUDevice>().stream();
    // A take at or above the previous one starts a new reverse pass.
    if (index >= last_take_) ++pass_;
    last_take_ = index;
    TF_RETURN_IF_ERROR(Flush(compute));  // frames still waiting for a full group
    TF_RETURN_IF_ERROR(Poll());
    // The frame itself (now, alone, if no prefetch started it), then the
    // frames ahead of it.
    TF_RETURN_IF_ERROR(Fetch(c, compute, index, index, 1));
    if (prefetch > 0) TF_RETURN_IF_ERROR(Fetch(c, compute, index - 1, index - prefetch, group));
    Slot& slot = slots_[index];
    const Tensor* output = nullptr;
    if (Held(slot.arriving)) {
      TF_RETURN_IF_ERROR(Await(compute, slot.arrived));
      output = &slot.arriving;
      ++stats_.fetched;
    } else if (Held(slot.resident)) {
      output = &slot.resident;
      ++stats_.in_place;
    } else {
      // Taken before (a second reverse pass) while its raw copy out still
      // runs (an encoded frame in transit stays in the slot and was decoded
      // above).
      for (const Group& pending : pending_)
        for (const Entry& entry : pending.entries)
          if (entry.index == index && entry.generation == slot.generation && Held(entry.frame))
            output = &entry.frame;
      if (output == nullptr)
        return errors::FailedPrecondition("frame ", index, " was never offloaded");
    }
    TF_RETURN_IF_ERROR(AwaitReader(compute, slot));
    c->set_output(0, *output);
    // TQ: frames below that predict from this one and are not decoded yet
    // need it as it was handed out.
    if (compressed() && NeedsReference(index)) {
      slot.reference = *output;
      referenced_.insert(index);
    }
    slot.arriving = Tensor();
    slot.arrived.reset();
    slot.resident = Tensor();
    slot.keep = false;
    // An encoded frame not on its way to the host leaves the store with it.
    if (!slot.in_transit) {
      slot.packet = Tensor();
      slot.buffer = -1;
    }
    TF_RETURN_IF_ERROR(ReleaseReferences(compute));
    if (index == 0 && getenv("V1_FRAME_STORE_STATS")) PrintStats();
    return absl::OkStatus();
  }

 private:
  // A copy event, returned to the pool when the last frame using it lets go
  // (always under mu_). Not copyable: a copy's destructor would return the
  // event while the original still uses it.
  struct Arrival {
    Arrival(FrameStore* store, cudaEvent_t event) : store(store), event(event) {}
    Arrival(const Arrival&) = delete;
    Arrival& operator=(const Arrival&) = delete;
    ~Arrival() { store->free_events_.push_back(event); }
    FrameStore* store;
    cudaEvent_t event;
  };
  struct Slot {
    uint64_t generation = 0;
    Tensor resident;
    int64_t buffer = -1;      // encoded: the ring record holding it
    Tensor packet;            // encoded: the frame packed on the GPU
    size_t packet_bytes = 0;  // encoded: its encoded length (0: stored raw)
    bool in_transit = false;  // its encoded copy out is on the D2H stream
    bool on_host = false;
    bool keep = false;
    Tensor arriving;
    std::shared_ptr<Arrival> arrived;
    // TQ: chosen for encoding and not encoded yet; the references (count and
    // generations of frames index + 1, index + 2) it was encoded from; the
    // taken tensor kept as a reference; the reverse pass that last fetched it.
    bool encoding = false;
    int order = 0;
    std::array<uint64_t, kMaxOrder> refs{};
    Tensor reference;
    uint64_t pass = 0;
    // TQ: the latest decode reading this frame as a reference. The compute
    // stream waits for it before the store hands the frame out or drops it,
    // so the frame's memory is reused only after that read (as the
    // allocator's stream order needs); decodes do not hold their references.
    std::shared_ptr<Arrival> reader;
  };
  struct Entry {
    int64_t index;
    uint64_t generation;
    Tensor frame;           // the raw frame (until encoded, if encoded)
    bool encode = false;    // chosen for encoding
    int64_t buffer = -1;    // its ring record, until released
    Tensor packet;          // its packet, if packed
    size_t bytes = 0;       // its encoded length, once encoded
    int order = 0;          // TQ: references it is predicted from
    std::array<Tensor, kMaxOrder> refs;  // TQ: their raw frames, until encoded
    std::array<uint64_t, kMaxOrder> ref_generations{};
    bool refused = false;   // TQ: the ring was full; it stays raw
  };
  // A group of frames moving through one stage together. `produced` (compute
  // stream, raw frames) marks that its frames are written; `done` that the
  // stage's work on them (encode or copy out) ended; with a codec, `start`
  // (timed, like `done`) that its copy out started.
  struct Group {
    std::vector<Entry> entries;
    cudaEvent_t produced = nullptr;
    cudaEvent_t done = nullptr;
    cudaEvent_t start = nullptr;
    size_t bytes = 0;  // copied out
  };
  // A decode's inputs (packets, scratch frames on the GPU), held until the
  // decode ended (references are ordered through Slot::reader instead).
  struct ReadLease {
    std::vector<Tensor> tensors;
    std::shared_ptr<Arrival> arrived;
  };
  // A flushed group's last frame and a timed compute-stream event behind it,
  // for the forward's frame rate on the GPU.
  struct Mark {
    int64_t index = -1;
    cudaEvent_t event = nullptr;
  };
  // TQ: the latest raw frames handed over (references of the waiting ones).
  struct Recent {
    int64_t index;
    uint64_t generation;
    Tensor frame;
  };
  // TQ: one encode's place in the ring (virtual byte position: the arena
  // offset is position % ring_bytes_), live while encoding, held by its entry
  // or read by a decode.
  struct Record {
    int64_t id;
    unsigned long long position = 0;
    size_t length = 0;  // 0: ring full (or still encoding)
    bool encoding = true;
    bool held = true;
    bool launched = true;  // false: never encoded (no room), stays raw
    std::shared_ptr<Arrival> reader;
  };

  bool compressed() const { return codec_ == Codec::kTq; }

  Status Ready(int64_t index) {
    TF_RETURN_IF_ERROR(admission_status_);
    if (host_ == nullptr || index >= static_cast<int64_t>(slots_.size()))
      return errors::FailedPrecondition("frame ", index, " was never offloaded");
    return absl::OkStatus();
  }

  Status Initialize() {
    TF_RETURN_IF_ERROR(admission_status_);
    if (d2h_ != nullptr) return absl::OkStatus();
    RETURN_IF_CUDA_ERROR(cudaGetDevice(&device_));
    RETURN_IF_CUDA_ERROR(cudaStreamCreateWithFlags(&d2h_, cudaStreamNonBlocking));
    RETURN_IF_CUDA_ERROR(cudaStreamCreateWithFlags(&h2d_, cudaStreamNonBlocking));
    RETURN_IF_CUDA_ERROR(cudaEventCreateWithFlags(&forward_end_, cudaEventDisableTiming));
    RETURN_IF_CUDA_ERROR(cudaEventCreateWithFlags(&h2d_marker_, cudaEventDisableTiming));
    RETURN_IF_CUDA_ERROR(cudaEventCreateWithFlags(&pack_marker_, cudaEventDisableTiming));
    RETURN_IF_CUDA_ERROR(cudaEventCreateWithFlags(&packed_, cudaEventDisableTiming));
    // Stores live for the process, and so does their admission thread.
    std::thread([this] { Admission(); }).detach();
    return absl::OkStatus();
  }

  // Whether the store's thread has work: a stage whose GPU work it awaits.
  bool Busy() const {
    return !queued_.empty() || !pending_.empty() || !encoding_.empty() || !reads_.empty() ||
           !marks_.empty() || Held(ring_);
  }

  // The admission thread: moves frames through encode and admission and
  // releases the copied frames as the GPU finishes each stage, whatever the
  // host's op calls are doing.
  void Admission() {
    const cudaError_t status = cudaSetDevice(device_);
    std::unique_lock<std::mutex> lock(mu_);
    if (status != cudaSuccess) {
      admission_status_ = errors::Internal("cudaSetDevice failed: ", cudaGetErrorString(status));
      return;
    }
    for (;;) {
      if (Busy()) {
        wake_.wait_for(lock, std::chrono::microseconds(100));
      } else {
        wake_.wait(lock);
      }
      Status polled = Poll();
      if (polled.ok()) polled = Admit();
      if (polled.ok()) polled = MaybeDropRing();
      if (!polled.ok()) {
        admission_status_ = polled;
        return;
      }
    }
  }

  // The offload of frame 0 starts a rollout: fresh statistics and frame-rate
  // marks (the link rate and the encoded-length average carry over); TQ lets
  // go of the previous rollout's references.
  Status StartRollout(cudaStream_t compute) {
    stats_ = Stats();
    forward_end_recorded_ = false;
    for (const Mark& mark : marks_) free_timed_.push_back(mark.event);
    marks_.clear();
    if (first_mark_.event != nullptr) free_timed_.push_back(first_mark_.event);
    first_mark_ = Mark();
    produced_index_ = -1;
    share_ = 0.0;
    ring_peak_ = 0;
    latest_ = -1;
    const char* link = getenv("V1_FRAME_STORE_LINK_GBPS");  // a simulated link (tests)
    link_override_ = link != nullptr ? std::max(0.0, atof(link)) : -1.0;
    for (int64_t index : referenced_) {
      TF_RETURN_IF_ERROR(AwaitReader(compute, slots_[index]));
      slots_[index].reference = Tensor();
    }
    referenced_.clear();
    recent_.clear();
    ++pass_;
    last_take_ = std::numeric_limits<int64_t>::max();
    return absl::OkStatus();
  }

  // Whether to encode the forward's frame `index`. The store projects the
  // bytes still on the GPU when the forward ends if every frame from here
  // went out raw: the backlog (frames handed over and not yet copied) plus
  // the frames still to come, minus what the link moves in the forward's
  // remaining time (link rate measured from this store's recent copies; the
  // time from the GPU's latest completed frame and the forward's measured
  // frame rate). Nothing to fit: raw. Otherwise it encodes the share of the
  // frames to come that makes the mixed stream fit, spread evenly by error
  // diffusion (no switching between modes); if even encoding them all cannot
  // fit, it encodes them all and also the frames already queued raw,
  // earliest first. Without measurements yet (the first frames of the first
  // rollout) frames go out raw.
  bool Decide(OpKernelContext* c, cudaStream_t compute, int64_t index, int64_t capacity,
              int64_t resident_from) {
    const double raw = static_cast<double>(frame_bytes_);
    const double saved = raw * (1.0 - ratio_);  // per encoded frame
    const double to_come = static_cast<double>(std::min(capacity, resident_from) - index);
    const double rate = link_override_ >= 0.0 ? link_override_ * 1e6 : LinkRate();  // bytes/ms
    double link = 0.0;  // the bytes the link takes before the forward ends
    if (rate > 0.0) {
      if (frame_rate_ <= 0.0) return false;
      link = kLinkMargin * rate * static_cast<double>(capacity - 1 - produced_index_) /
             frame_rate_;
    } else if (rate < 0.0) {
      return false;
    }
    const double excess = Backlog() + to_come * raw - link;
    if (excess <= 0.0) {
      share_ = 0.0;
      return false;
    }
    const double share = excess / (to_come * saved);
    if (share >= 1.0) {
      share_ = 0.0;
      EncodeQueued(c, compute, excess - to_come * saved, saved);
      return true;
    }
    share_ += share;
    if (share_ < 1.0) return false;
    share_ -= 1.0;
    return true;
  }

  // Bytes per ms the D2H stream moved over its recent groups (-1: unknown).
  double LinkRate() const { return link_ms_ > 0.0 ? link_bytes_ / link_ms_ : -1.0; }

  // The bytes handed to the store and not yet copied out.
  double Backlog() const {
    const double raw = static_cast<double>(frame_bytes_), encoded = raw * ratio_;
    double bytes = encoded * waiting_.size();
    for (const Entry& entry : staged_) bytes += entry.encode ? encoded : raw;
    for (const Group& group : encoding_) bytes += encoded * group.entries.size();
    for (const Group& group : queued_)
      for (const Entry& entry : group.entries)
        bytes += entry.encode ? static_cast<double>(entry.bytes) : raw;
    for (const Group& group : pending_) bytes += static_cast<double>(group.bytes);
    return bytes;
  }

  // Encodes queued raw groups, earliest first, until `excess` bytes are
  // saved (never waits), each predicted from the raw frames after it still on
  // the GPU.
  void EncodeQueued(OpKernelContext* c, cudaStream_t compute, double excess, double saved) {
    const Status ring = EnsureRing(c, compute);
    if (!ring.ok()) {
      admission_status_ = ring;
      return;
    }
    for (auto it = queued_.begin(); it != queued_.end() && excess > 0.0;) {
      bool raw = true;
      for (const Entry& entry : it->entries) raw &= !entry.encode && !entry.refused;
      if (!raw) {
        ++it;
        continue;
      }
      Group group = std::move(*it);
      it = queued_.erase(it);
      // Its references may be later frames: the encode starts behind the
      // compute stream's work so far.
      if (group.produced != nullptr) free_events_.push_back(group.produced);
      Status status = NewEvent(&group.produced);
      if (status.ok() && cudaEventRecord(group.produced, compute) != cudaSuccess)
        status = errors::Internal("cudaEventRecord failed");
      if (!status.ok()) {
        admission_status_ = status;
        return;
      }
      for (Entry& entry : group.entries) {
        entry.encode = true;
        AttachReferences(entry, true);
        slots_[entry.index].encoding = true;
      }
      excess -= saved * group.entries.size();
      const Status encoded = Encode(std::move(group));
      if (!encoded.ok()) {
        admission_status_ = encoded;
        return;
      }
    }
  }

  // TQ: the ring arena (allocated as the codec first needs it in a rollout,
  // released once the forward ended and every frame left it). Its encodes
  // start after the compute stream's work so far (the allocator may hand out
  // memory that compute work enqueued before this point still uses).
  Status EnsureRing(OpKernelContext* c, cudaStream_t compute) {
    if (Held(ring_)) return absl::OkStatus();
    TF_RETURN_IF_ERROR(
        c->allocate_temp(DT_UINT8, TensorShape({static_cast<int64_t>(ring_bytes_)}), &ring_));
    RETURN_IF_CUDA_ERROR(cudaMemsetAsync(scratch_.data(), 0, sizeof(tq::Scratch), encode_));
    RETURN_IF_CUDA_ERROR(cudaEventRecord(pack_marker_, compute));
    RETURN_IF_CUDA_ERROR(cudaStreamWaitEvent(encode_, pack_marker_, 0));
    host_head_ = 0;
    return absl::OkStatus();
  }

  // TQ: lets go of the ring once the forward ended and no frame uses it.
  Status MaybeDropRing() {
    if (!Held(ring_) || !forward_end_recorded_ || !waiting_.empty() || !encoding_.empty())
      return absl::OkStatus();
    RingTail();
    if (!records_.empty()) return absl::OkStatus();
    for (cudaEvent_t event : {forward_end_, packed_}) {
      const cudaError_t status = cudaEventQuery(event);
      if (status == cudaErrorNotReady) return absl::OkStatus();
      RETURN_IF_CUDA_ERROR(status);
    }
    ring_ = Tensor();
    return absl::OkStatus();
  }

  // (Re)allocates the host buffer (and the codec state) when the frame
  // layout, capacity or codec changes; only then does the host wait for the
  // store's copies.
  Status Configure(OpKernelContext* c, cudaStream_t compute, const Tensor& frame,
                   int64_t capacity, Codec codec) {
    if (frame.dtype() != DT_FLOAT) codec = Codec::kRaw;
    if (host_ != nullptr && frame.dtype() == dtype_ && frame.shape() == shape_ &&
        capacity <= static_cast<int64_t>(slots_.size()) && codec == codec_)
      return absl::OkStatus();
    for (int64_t index = 0; index < static_cast<int64_t>(slots_.size()); ++index)
      TF_RETURN_IF_ERROR(Drop(index, compute));
    // Every stream's work so far ends, then every stage's frames are dropped.
    for (cudaStream_t stream : {encode_, d2h_, h2d_})
      if (stream != nullptr) RETURN_IF_CUDA_ERROR(cudaStreamSynchronize(stream));
    for (std::deque<Group>* stage : {&queued_, &encoding_, &pending_}) {
      for (const Group& group : *stage) {
        for (cudaEvent_t event : {group.produced, group.done})
          if (event != nullptr) free_events_.push_back(event);
        if (group.start != nullptr) free_timed_.push_back(group.start);
      }
      stage->clear();
    }
    // (With a codec `done` of the copy stage is timed; it only goes back to
    // the wrong pool here, which is harmless: both pools hold events.)
    staged_.clear();
    waiting_.clear();
    recent_.clear();
    reads_.clear();
    last_awaited_.reset();
    if (host_ != nullptr) RETURN_IF_CUDA_ERROR(cudaFreeHost(host_));
    host_ = nullptr;
    slots_.clear();
    referenced_.clear();
    records_.clear();
    ring_ = Tensor();
    decode_buffer_ = Tensor();
    link_bytes_ = link_ms_ = frame_rate_ = 0.0;
    ratio_ = kInitialRatio;
    const size_t bytes = frame.TotalBytes();
    codec_ = codec;
    host_stride_ = bytes;
    if (compressed() && encode_ == nullptr) {
      RETURN_IF_CUDA_ERROR(cudaStreamCreateWithFlags(&encode_, cudaStreamNonBlocking));
      RETURN_IF_CUDA_ERROR(cudaHostAlloc(reinterpret_cast<void**>(&host_results_),
                                         sizeof(tq::Result) * kRecords,
                                         cudaHostAllocPortable | cudaHostAllocMapped));
      RETURN_IF_CUDA_ERROR(cudaHostGetDevicePointer(reinterpret_cast<void**>(&device_results_),
                                                    host_results_, 0));
    }
    // The packets, ring and scratch come from this GPU's allocator itself
    // (never a step's tracking wrapper), and their first writes wait for the
    // compute stream's work so far (pack_marker_).
    allocator_ = c->device()->GetAllocator(AllocatorAttributes());
    compute_ = compute;
    if (compressed()) {
      layout_ = tq::MakeLayout(frame.NumElements());
      // Payload offsets are 32-bit words.
      if (layout_.blocks * 32 >= (1ll << 32))
        return errors::InvalidArgument("a TQ frame of ", frame.NumElements(),
                                       " values is too large");
      encoded_capacity_ = layout_.max_bytes;
      const char* frames = getenv("V1_FRAME_STORE_RING_FRAMES");
      const double ring_frames = frames != nullptr ? std::max(1.0, atof(frames)) : kRingFrames;
      ring_bytes_ = tq::AlignUp(static_cast<size_t>(ring_frames * encoded_capacity_) +
                                tq::kAlign);
      scratch_ = Tensor(allocator_, DT_UINT8,
                        TensorShape({static_cast<int64_t>(sizeof(tq::Scratch) + 8 * layout_.tiles)}));
      if (!scratch_.IsInitialized())
        return errors::ResourceExhausted("no GPU memory for the TQ codec state");
      RETURN_IF_CUDA_ERROR(cudaEventRecord(pack_marker_, compute));
      RETURN_IF_CUDA_ERROR(cudaStreamWaitEvent(encode_, pack_marker_, 0));
    }
    if (compressed()) host_stride_ = std::max(bytes, encoded_capacity_);
    void* host = nullptr;
    RETURN_IF_CUDA_ERROR(cudaHostAlloc(&host, host_stride_ * capacity, cudaHostAllocPortable));
    host_ = static_cast<char*>(host);
    frame_bytes_ = bytes;
    dtype_ = frame.dtype();
    shape_ = frame.shape();
    slots_.resize(capacity);
    return absl::OkStatus();
  }

  // Hands the staged frames on, after everything the compute stream holds so
  // far (which includes the kernels that wrote them), to admission as one
  // group (frames chosen for encoding wait in `waiting_` instead). With
  // `drain`, TQ frames still waiting for their references are encoded from
  // those recorded (a lower order).
  Status Flush(cudaStream_t compute, bool drain = true) {
    if (drain && !waiting_.empty()) TF_RETURN_IF_ERROR(Promote(compute, true, -1));
    if (staged_.empty()) return absl::OkStatus();
    Group group{std::move(staged_)};
    staged_.clear();
    TF_RETURN_IF_ERROR(NewEvent(&group.produced));
    RETURN_IF_CUDA_ERROR(cudaEventRecord(group.produced, compute));
    if (compressed()) TF_RETURN_IF_ERROR(RecordMark(compute, group.entries.back().index));
    queued_.push_back(std::move(group));
    TF_RETURN_IF_ERROR(Admit());
    wake_.notify_one();
    return absl::OkStatus();
  }

  // A timed compute-stream event behind frame `index` (the forward's rate).
  Status RecordMark(cudaStream_t compute, int64_t index) {
    Mark mark{index};
    TF_RETURN_IF_ERROR(NewTimedEvent(&mark.event));
    RETURN_IF_CUDA_ERROR(cudaEventRecord(mark.event, compute));
    marks_.push_back(mark);
    return absl::OkStatus();
  }

  // TQ: the raw frame `index` of this rollout, if still held.
  const Tensor* RawFrame(int64_t index) const {
    if (index > latest_) return nullptr;
    const Slot& slot = slots_[index];
    if (Held(slot.resident)) return &slot.resident;
    for (const Recent& recent : recent_)
      if (recent.index == index && recent.generation == slot.generation) return &recent.frame;
    return nullptr;
  }

  // TQ: gives the entry the raw frames after it (as many as the order and the
  // forward's length ask for); false if one is not recorded yet. With
  // `partial`, it takes those available (a lower order).
  bool AttachReferences(Entry& entry, bool partial) {
    const int64_t wanted = std::min<int64_t>(kMaxOrder, capacity_ - 1 - entry.index);
    int order = 0;
    for (; order < wanted; ++order) {
      const Tensor* frame = RawFrame(entry.index + order + 1);
      if (frame == nullptr) break;
      entry.refs[order] = *frame;
      entry.ref_generations[order] = slots_[entry.index + order + 1].generation;
    }
    if (order < wanted && !partial) {
      entry.refs = {};
      return false;
    }
    entry.order = order;
    return true;
  }

  // TQ: encodes the waiting frames whose references are recorded (with
  // `drain`, all of them), after the compute stream's work so far; `index`
  // >= 0 is the frame just handed over (a frame-rate mark).
  Status Promote(cudaStream_t compute, bool drain, int64_t index) {
    Group group;
    for (auto it = waiting_.begin(); it != waiting_.end();) {
      if (!AttachReferences(*it, drain)) break;  // later frames wait too
      group.entries.push_back(std::move(*it));
      it = waiting_.erase(it);
    }
    if (group.entries.empty()) return absl::OkStatus();
    TF_RETURN_IF_ERROR(NewEvent(&group.produced));
    RETURN_IF_CUDA_ERROR(cudaEventRecord(group.produced, compute));
    if (index >= 0) TF_RETURN_IF_ERROR(RecordMark(compute, index));
    TF_RETURN_IF_ERROR(Encode(std::move(group)));
    wake_.notify_one();
    return absl::OkStatus();
  }

  // The oldest byte of the ring still in use (virtual position); frees the
  // records no longer in use at its front.
  unsigned long long RingTail() {
    while (!records_.empty()) {
      Record& front = records_.front();
      if (front.encoding || front.held) break;
      if (front.reader) {
        if (cudaEventQuery(front.reader->event) == cudaErrorNotReady) break;
        front.reader.reset();
      }
      records_.pop_front();
    }
    for (const Record& record : records_) {
      if (record.encoding) return host_head_;  // it and later ones start at the head
      if (record.length > 0) return record.position;
    }
    return host_head_;
  }

  Record& RecordOf(int64_t id) { return records_[id - records_.front().id]; }

  // Encodes a group's frames into ring records on `encode_`, after the
  // compute-stream event `produced` (returned to the pool: the wait captured
  // its record); the ring tail moves only past finished readers.
  Status Encode(Group group) {
    RETURN_IF_CUDA_ERROR(cudaStreamWaitEvent(encode_, group.produced, 0));
    free_events_.push_back(group.produced);
    group.produced = nullptr;
    for (Entry& entry : group.entries) {
      const unsigned long long tail = RingTail();
      // Live records have distinct result slots; with no room to track one
      // more (or no ring), the frame stays raw.
      const bool room = Held(ring_) && records_.size() + 1 < static_cast<size_t>(kRecords);
      const int64_t id = next_record_++;
      records_.push_back(Record{id});
      records_.back().launched = room;
      entry.buffer = id;
      if (!room) continue;
      tq::Result* result = device_results_ + id % kRecords;
      auto data = [](const Tensor& tensor) {
        return Held(tensor) ? tensor.flat<float>().data() : nullptr;
      };
      RETURN_IF_CUDA_ERROR(tq::Encode(
          entry.frame.flat<float>().data(), data(entry.refs[0]), data(entry.refs[1]),
          entry.order, layout_, RingData(), ring_bytes_, tail,
          reinterpret_cast<tq::Scratch*>(scratch_.data()),
          reinterpret_cast<unsigned long long*>(
              static_cast<char*>(scratch_.data()) + sizeof(tq::Scratch)),
          result, encode_));
    }
    TF_RETURN_IF_ERROR(NewEvent(&group.done));
    RETURN_IF_CUDA_ERROR(cudaEventRecord(group.done, encode_));
    encoding_.push_back(std::move(group));
    return absl::OkStatus();
  }

  unsigned char* RingData() const {
    return reinterpret_cast<unsigned char*>(const_cast<char*>(ring_.tensor_data().data()));
  }
  // The encoded bytes of a frame in its ring record.
  const char* BufferData(int64_t buffer) {
    return reinterpret_cast<const char*>(RingData()) + RecordOf(buffer).position % ring_bytes_;
  }

  // Whether a frame still wants its raw copy out or its encoding: it is the
  // slot's current frame, not taken, and not chosen to stay for the reverse
  // pass.
  bool WantsRaw(const Entry& entry) const {
    const Slot& slot = slots_[entry.index];
    return slot.generation == entry.generation && Held(slot.resident) && !slot.keep;
  }
  // Whether an encoded frame still wants its copy out: the slot's current
  // buffer or packet, neither taken nor already being decoded for the
  // reverse pass.
  bool Wanted(const Entry& entry) const {
    if (!entry.encode) return WantsRaw(entry);
    const Slot& slot = slots_[entry.index];
    if (slot.generation != entry.generation || Held(slot.arriving)) return false;
    return entry.buffer >= 0 ? slot.buffer == entry.buffer : Held(slot.packet);
  }
  // An entry leaves the pipeline: its ring record, if any, is free again
  // (behind any decode reading it).
  void Release(Entry& entry) {
    if (entry.buffer >= 0) RecordOf(entry.buffer).held = false;
    entry.buffer = -1;
  }

  // Copies an encoded frame from its buffer into a packet of its encoded
  // length on `encode_` (so the next encode into the buffer runs after it),
  // after the compute stream's work so far (the allocator may hand out
  // memory that compute work enqueued before this point still uses), and
  // frees the buffer. Copies out and decodes of a packet wait on `packed_`,
  // recorded behind the latest pack.
  Status PackEntry(Entry& entry) {
    entry.packet = Tensor(allocator_, DT_UINT8, TensorShape({static_cast<int64_t>(entry.bytes)}));
    if (!entry.packet.IsInitialized())
      return errors::ResourceExhausted("no GPU memory for an encoded packet of ", entry.bytes,
                                       " bytes");
    RETURN_IF_CUDA_ERROR(cudaEventRecord(pack_marker_, compute_));
    RETURN_IF_CUDA_ERROR(cudaStreamWaitEvent(encode_, pack_marker_, 0));
    RETURN_IF_CUDA_ERROR(cudaMemcpyAsync(const_cast<char*>(entry.packet.tensor_data().data()),
                                         BufferData(entry.buffer), entry.bytes,
                                         cudaMemcpyDeviceToDevice, encode_));
    RETURN_IF_CUDA_ERROR(cudaEventRecord(packed_, encode_));
    Slot& slot = slots_[entry.index];
    slot.packet = entry.packet;
    slot.buffer = -1;
    Release(entry);
    return absl::OkStatus();
  }

  // Submits queued groups to the D2H stream while fewer than kInFlightGroups
  // are unfinished; once the GPU has passed the forward's last frame, keeps
  // the queued frames on the GPU instead (see the class comment), packing
  // the encoded ones out of their buffers.
  Status Admit() {
    if (forward_end_recorded_ && !queued_.empty()) {
      const cudaError_t status = cudaEventQuery(forward_end_);
      if (status != cudaErrorNotReady) {
        RETURN_IF_CUDA_ERROR(status);
        for (Group& group : queued_) {
          for (Entry& entry : group.entries) {
            const bool wanted = Wanted(entry);
            stats_.not_admitted += wanted;
            if (wanted && entry.buffer >= 0) {
              ++stats_.kept_packed;
              TF_RETURN_IF_ERROR(PackEntry(entry));
            }
            Release(entry);
          }
          if (group.produced != nullptr) free_events_.push_back(group.produced);
        }
        queued_.clear();
        return absl::OkStatus();
      }
    }
    while (!queued_.empty() && pending_.size() < kInFlightGroups) {
      Group group = std::move(queued_.front());
      queued_.pop_front();
      std::vector<Entry> wanted;
      for (Entry& entry : group.entries) {
        if (Wanted(entry)) {
          wanted.push_back(std::move(entry));
        } else {
          Release(entry);
        }
      }
      group.entries = std::move(wanted);
      if (!group.entries.empty()) {
        if (group.produced != nullptr)
          RETURN_IF_CUDA_ERROR(cudaStreamWaitEvent(d2h_, group.produced, 0));
        bool packets = false;
        for (const Entry& entry : group.entries) packets |= Held(entry.packet);
        if (packets) RETURN_IF_CUDA_ERROR(cudaStreamWaitEvent(d2h_, packed_, 0));
        if (compressed()) {
          TF_RETURN_IF_ERROR(NewTimedEvent(&group.start));
          RETURN_IF_CUDA_ERROR(cudaEventRecord(group.start, d2h_));
        }
        for (const Entry& entry : group.entries) {
          const char* source;
          size_t bytes;
          if (!entry.encode) {
            source = entry.frame.tensor_data().data();
            bytes = frame_bytes_;
            ++stats_.sent_raw;
          } else {
            source = Held(entry.packet) ? entry.packet.tensor_data().data()
                                        : BufferData(entry.buffer);
            bytes = entry.bytes;
            ++stats_.sent_encoded;
          }
          RETURN_IF_CUDA_ERROR(cudaMemcpyAsync(host_ + entry.index * host_stride_, source, bytes,
                                               cudaMemcpyDeviceToHost, d2h_));
          group.bytes += bytes;
          slots_[entry.index].in_transit = true;
        }
        if (compressed()) {
          TF_RETURN_IF_ERROR(NewTimedEvent(&group.done));
        } else {
          TF_RETURN_IF_ERROR(NewEvent(&group.done));
        }
        RETURN_IF_CUDA_ERROR(cudaEventRecord(group.done, d2h_));
        stats_.copied_out += group.entries.size();
      }
      // The wait above captured the event's record, so it can be reused.
      if (group.produced != nullptr) free_events_.push_back(group.produced);
      group.produced = nullptr;
      if (!group.entries.empty()) pending_.push_back(std::move(group));
    }
    return absl::OkStatus();
  }

  // Into the admission queue at its frames' place (raw groups may have
  // overtaken it).
  void Enqueue(Group group) {
    const int64_t first = group.entries.front().index;
    auto at = std::find_if(queued_.begin(), queued_.end(), [first](const Group& queued) {
      return queued.entries.front().index > first;
    });
    queued_.insert(at, std::move(group));
  }

  // Moves every group whose stage has finished on to the next one, in issue
  // order: finished decodes release their inputs; the forward's frame marks
  // update its frame rate; encoded frames release their raw frame and wait
  // for admission, in the queue's frame order (TQ frames the ring could not
  // take go on raw); copied-out frames are on the host and update the link
  // rate.
  Status Poll() {
    while (!reads_.empty()) {
      const cudaError_t status = cudaEventQuery(reads_.front().arrived->event);
      if (status == cudaErrorNotReady) break;
      RETURN_IF_CUDA_ERROR(status);
      reads_.pop_front();
    }
    while (!marks_.empty()) {
      const Mark mark = marks_.front();
      const cudaError_t status = cudaEventQuery(mark.event);
      if (status == cudaErrorNotReady) break;
      RETURN_IF_CUDA_ERROR(status);
      marks_.pop_front();
      produced_index_ = std::max(produced_index_, mark.index);
      if (first_mark_.event == nullptr) {
        first_mark_ = mark;
        continue;
      }
      float ms = 0.0f;
      RETURN_IF_CUDA_ERROR(cudaEventElapsedTime(&ms, first_mark_.event, mark.event));
      if (ms > 0.0f && mark.index > first_mark_.index)
        frame_rate_ = static_cast<double>(mark.index - first_mark_.index) / ms;
      free_timed_.push_back(mark.event);
    }
    bool admit = false;
    while (!encoding_.empty()) {
      Group& front = encoding_.front();
      const cudaError_t status = cudaEventQuery(front.done);
      if (status == cudaErrorNotReady) break;
      RETURN_IF_CUDA_ERROR(status);
      free_events_.push_back(front.done);
      front.done = nullptr;
      std::vector<Entry> wanted, raw;
      for (Entry& entry : front.entries) {
        Slot& slot = slots_[entry.index];
        if (slot.generation == entry.generation) slot.encoding = false;
        Record& record = RecordOf(entry.buffer);
        const tq::Result result =
            record.launched ? host_results_[entry.buffer % kRecords] : tq::Result{0, 0};
        record.encoding = false;
        entry.refs = {};
        const size_t bytes = result.length;
        if (bytes == 0) {  // the ring was full: it goes on raw
          record.held = false;
          entry.buffer = -1;
          ++stats_.ring_full;
          if (WantsRaw(entry)) {
            entry.encode = false;
            entry.refused = true;
            raw.push_back(std::move(entry));
          }
          continue;
        }
        record.position = result.position;
        record.length = bytes;
        host_head_ = result.position + tq::AlignUp(bytes);
        if (bytes > encoded_capacity_)
          return errors::Internal("invalid encoded length ", bytes);
        if (!WantsRaw(entry)) {  // taken or kept meanwhile: it stays raw
          Release(entry);
          continue;
        }
        ++stats_.encoded;
        stats_.raw_bytes += frame_bytes_;
        stats_.packet_bytes += bytes;
        ratio_ = kDecay * ratio_ + (1.0 - kDecay) * static_cast<double>(bytes) / frame_bytes_;
        slot.buffer = entry.buffer;
        slot.packet_bytes = bytes;
        slot.order = entry.order;
        slot.refs = entry.ref_generations;
        slot.resident = Tensor();  // the early release of the raw frame
        entry.frame = Tensor();
        entry.bytes = bytes;
        wanted.push_back(std::move(entry));
      }
      for (std::vector<Entry>* entries : {&wanted, &raw}) {
        if (entries->empty()) continue;
        Enqueue(Group{std::move(*entries)});
        admit = true;
      }
      encoding_.pop_front();
    }
    if (Held(ring_))
      ring_peak_ = std::max<unsigned long long>(ring_peak_, host_head_ - RingTail());
    while (!pending_.empty()) {
      Group& front = pending_.front();
      const cudaError_t status = cudaEventQuery(front.done);
      if (status == cudaErrorNotReady) break;
      RETURN_IF_CUDA_ERROR(status);
      for (Entry& entry : front.entries) {
        const int64_t buffer = entry.buffer;
        Release(entry);
        Slot& slot = slots_[entry.index];
        if (slot.generation != entry.generation) continue;
        slot.on_host = true;
        slot.in_transit = false;
        slot.packet = Tensor();
        if (buffer >= 0 && slot.buffer == buffer) slot.buffer = -1;
        if (!slot.keep) slot.resident = Tensor();
      }
      if (front.start != nullptr) {
        float ms = 0.0f;
        RETURN_IF_CUDA_ERROR(cudaEventElapsedTime(&ms, front.start, front.done));
        if (ms > 0.0f) {
          link_bytes_ = kDecay * link_bytes_ + static_cast<double>(front.bytes);
          link_ms_ = kDecay * link_ms_ + ms;
        }
        free_timed_.push_back(front.start);
        free_timed_.push_back(front.done);
      } else {
        free_events_.push_back(front.done);
      }
      pending_.pop_front();
    }
    return admit ? Admit() : absl::OkStatus();
  }

  // TQ: the reconstruction of frame `index` on the GPU, if at hand.
  const Tensor* Reconstructed(int64_t index) const {
    const Slot& slot = slots_[index];
    if (Held(slot.arriving)) return &slot.arriving;
    if (Held(slot.resident)) return &slot.resident;
    if (Held(slot.reference)) return &slot.reference;
    return nullptr;
  }

  // TQ: whether frame r is a reference some stored encoded frame below it
  // still needs: one not decoded in the current reverse pass or, if r is not
  // on the host (it cannot be fetched again: an anchor), any, until the next
  // rollout. Frames chosen for encoding and not encoded yet count too.
  bool NeedsReference(int64_t r) const {
    const bool anchor = !slots_[r].on_host;
    for (int64_t d = std::max<int64_t>(0, r - kMaxOrder); d < r; ++d) {
      const Slot& dependent = slots_[d];
      if (!dependent.encoding) {
        const int k = static_cast<int>(r - d);
        if (dependent.packet_bytes == 0 || dependent.order < k ||
            dependent.refs[k - 1] != slots_[r].generation)
          continue;
        if (!(dependent.on_host || dependent.in_transit || Held(dependent.packet) ||
              dependent.buffer >= 0))
          continue;  // its encoded frame is gone
      }
      if (anchor || dependent.pass != pass_) return true;
    }
    return false;
  }

  // TQ: lets go of the kept references no frame needs any more (behind
  // their readers on the compute stream).
  Status ReleaseReferences(cudaStream_t compute) {
    for (auto it = referenced_.begin(); it != referenced_.end();) {
      if (NeedsReference(*it)) {
        ++it;
      } else {
        TF_RETURN_IF_ERROR(AwaitReader(compute, slots_[*it]));
        slots_[*it].reference = Tensor();
        it = referenced_.erase(it);
      }
    }
    return absl::OkStatus();
  }

  // TQ: frame `index` is read as a reference by the decode being issued; a
  // resident one stays (it is handed out by its take, which waits for this
  // read).
  void ReadReference(int64_t index) {
    slots_[index].keep |= Held(slots_[index].resident);
    reading_.push_back(index);
  }

  // The compute stream waits for the latest decode reading the slot's frame.
  Status AwaitReader(cudaStream_t compute, Slot& slot) {
    if (slot.reader) TF_RETURN_IF_ERROR(Await(compute, slot.reader));
    slot.reader.reset();
    return absl::OkStatus();
  }

  // TQ: decodes encoded frame `index` into `output` on h2d_, reading its
  // buffer or packet directly or its host copy through the staging buffer.
  // `references` are the reconstructions of the frames after it.
  Status DecodeTq(OpKernelContext* c, int64_t index, char* output,
                  const std::array<const float*, kMaxOrder>& references,
                  std::vector<Tensor>* hold, std::vector<int64_t>* records, bool* packed) {
    const Slot& slot = slots_[index];
    const char* input;
    if (slot.buffer >= 0) {  // still on the GPU, possibly while copied out
      records->push_back(slot.buffer);
      input = BufferData(slot.buffer);
      ++stats_.direct_decodes;
    } else if (Held(slot.packet)) {
      if (!*packed) RETURN_IF_CUDA_ERROR(cudaStreamWaitEvent(h2d_, packed_, 0));
      *packed = true;
      hold->push_back(slot.packet);
      input = slot.packet.tensor_data().data();
      ++stats_.direct_decodes;
    } else {
      TF_RETURN_IF_ERROR(StagingBuffer(c));
      input = decode_buffer_.tensor_data().data();
      RETURN_IF_CUDA_ERROR(cudaMemcpyAsync(const_cast<char*>(input), host_ + index * host_stride_,
                                           slot.packet_bytes, cudaMemcpyHostToDevice, h2d_));
    }
    RETURN_IF_CUDA_ERROR(tq::Decode(reinterpret_cast<float*>(output), references[0],
                                    references[1], slot.order, layout_,
                                    reinterpret_cast<const unsigned char*>(input), h2d_));
    return absl::OkStatus();
  }

  // The staging buffer of encoded frames back from the host (decoded on
  // h2d_; allocated when first needed, before the h2d_ marker of its Fetch).
  Status StagingBuffer(OpKernelContext* c) {
    if (Held(decode_buffer_)) return absl::OkStatus();
    TF_RETURN_IF_ERROR(c->allocate_temp(
        DT_UINT8, TensorShape({static_cast<int64_t>(encoded_capacity_)}), &decode_buffer_));
    return Behind(c->eigen_device<GPUDevice>().stream());
  }

  // h2d_ waits for the compute stream's work so far (new buffers are free
  // for compute work after this point).
  Status Behind(cudaStream_t compute) {
    RETURN_IF_CUDA_ERROR(cudaEventRecord(h2d_marker_, compute));
    RETURN_IF_CUDA_ERROR(cudaStreamWaitEvent(h2d_, h2d_marker_, 0));
    return absl::OkStatus();
  }

  // TQ: the reconstructions of encoded frame `index`'s references, held in
  // `hold`. A reference not at hand is rebuilt from the store, walking up the
  // chain of the frames it predicts from into three rotating scratch frames.
  Status References(OpKernelContext* c, int64_t index,
                    std::array<const float*, kMaxOrder>* references, std::vector<Tensor>* hold,
                    std::vector<int64_t>* records, bool* packed) {
    *references = {nullptr, nullptr};
    std::set<int64_t> need;
    auto check = [this](int64_t frame) -> Status {
      const Slot& slot = slots_[frame];
      for (int k = 0; k < slot.order; ++k)
        if (slot.refs[k] != slots_[frame + k + 1].generation)
          return errors::FailedPrecondition("frame ", frame, " was encoded from frame ",
                                            frame + k + 1, ", which was offloaded again");
      return absl::OkStatus();
    };
    TF_RETURN_IF_ERROR(check(index));
    for (int k = 0; k < slots_[index].order; ++k) {
      const Tensor* reference = Reconstructed(index + k + 1);
      if (reference != nullptr) {
        ReadReference(index + k + 1);
        (*references)[k] = reference->flat<float>().data();
      } else {
        need.insert(index + k + 1);
      }
    }
    if (need.empty()) return absl::OkStatus();
    // The frames to rebuild, ascending; each encoded one needs its own.
    std::vector<int64_t> walk;
    for (auto it = need.begin(); it != need.end(); it = need.upper_bound(*it)) {
      const int64_t frame = *it;
      if (Reconstructed(frame) != nullptr) continue;
      const Slot& slot = slots_[frame];
      const bool encoded = slot.packet_bytes > 0 &&
                           (slot.buffer >= 0 || Held(slot.packet) || slot.on_host);
      if (!encoded && !(slot.packet_bytes == 0 && slot.on_host))
        return errors::FailedPrecondition(
            "frame ", index, " cannot be decoded: frame ", frame,
            ", which it is predicted from, is no longer stored (take frames in reverse order)");
      if (encoded) {
        TF_RETURN_IF_ERROR(check(frame));
        for (int k = 0; k < slot.order; ++k) need.insert(frame + k + 1);
      }
      walk.push_back(frame);
    }
    ++stats_.chain_walks;
    std::array<Tensor, 3> scratch;
    for (Tensor& buffer : scratch) TF_RETURN_IF_ERROR(c->allocate_temp(dtype_, shape_, &buffer));
    TF_RETURN_IF_ERROR(Behind(c->eigen_device<GPUDevice>().stream()));
    std::map<int64_t, const float*> rebuilt;
    auto lookup = [&](int64_t frame) -> const float* {
      auto found = rebuilt.find(frame);
      if (found != rebuilt.end()) return found->second;
      const Tensor* reference = Reconstructed(frame);
      ReadReference(frame);
      return reference->flat<float>().data();
    };
    for (auto it = walk.rbegin(); it != walk.rend(); ++it) {
      const int64_t frame = *it;
      const Slot& slot = slots_[frame];
      char* output = const_cast<char*>(scratch[frame % 3].tensor_data().data());
      if (slot.packet_bytes == 0) {
        RETURN_IF_CUDA_ERROR(cudaMemcpyAsync(output, host_ + frame * host_stride_, frame_bytes_,
                                             cudaMemcpyHostToDevice, h2d_));
      } else {
        std::array<const float*, kMaxOrder> inputs{nullptr, nullptr};
        for (int k = 0; k < slot.order; ++k) inputs[k] = lookup(frame + k + 1);
        TF_RETURN_IF_ERROR(DecodeTq(c, frame, output, inputs, hold, records, packed));
      }
      rebuilt[frame] = reinterpret_cast<const float*>(output);
    }
    for (int k = 0; k < slots_[index].order; ++k)
      if ((*references)[k] == nullptr) (*references)[k] = lookup(index + k + 1);
    for (Tensor& buffer : scratch) hold->push_back(buffer);
    return absl::OkStatus();
  }

  // Starts bringing back the frames in [lowest, highest] that are only on
  // the host (or only encoded on the GPU, decoded directly), highest first,
  // in groups of `group` frames with one arrival event each (the compute
  // stream later waits once per group). A partial group waits for more
  // frames unless its highest frame is due within two takes (>= highest -
  // 1). Raw frames still on the GPU are kept there and used in place.
  Status Fetch(OpKernelContext* c, cudaStream_t compute, int64_t highest, int64_t lowest,
               int64_t group) {
    lowest = std::max<int64_t>(lowest, 0);
    std::vector<int64_t> needed;
    bool staging = false;  // an encoded frame comes back from the host
    for (int64_t index = highest; index >= lowest; --index) {
      Slot& slot = slots_[index];
      if (Held(slot.arriving)) continue;
      if (Held(slot.resident)) {
        slot.keep = true;
      } else if (slot.on_host || Held(slot.packet) || slot.buffer >= 0) {
        needed.push_back(index);
      }
    }
    size_t issued = needed.size() - needed.size() % group;
    if (issued < needed.size() && needed[issued] >= highest - 1) issued = needed.size();
    if (issued == 0) return absl::OkStatus();
    for (size_t k = 0; k < issued; ++k) {
      const Slot& slot = slots_[needed[k]];
      staging |= slot.packet_bytes > 0 && slot.buffer < 0 && !Held(slot.packet);
    }
    std::vector<Tensor> buffers(issued);
    for (Tensor& buffer : buffers) TF_RETURN_IF_ERROR(c->allocate_temp(dtype_, shape_, &buffer));
    if (staging && !Held(decode_buffer_))
      TF_RETURN_IF_ERROR(c->allocate_temp(
          DT_UINT8, TensorShape({static_cast<int64_t>(encoded_capacity_)}), &decode_buffer_));
    // The new buffers are free for work the compute stream holds after this
    // point (allocated before it is recorded); the H2D stream starts there
    // too (one compute-stream event per call).
    TF_RETURN_IF_ERROR(Behind(compute));
    bool packed = false;  // h2d_ waited for the packs so far
    for (size_t first = 0; first < issued; first += group) {
      const size_t last = std::min<size_t>(first + group, issued);
      std::vector<Tensor> hold;
      std::vector<int64_t> buffers_read;
      for (size_t k = first; k < last; ++k) {
        Slot& slot = slots_[needed[k]];
        char* output = const_cast<char*>(buffers[k].tensor_data().data());
        const char* host = host_ + needed[k] * host_stride_;
        if (slot.packet_bytes == 0) {  // stored raw
          RETURN_IF_CUDA_ERROR(
              cudaMemcpyAsync(output, host, frame_bytes_, cudaMemcpyHostToDevice, h2d_));
        } else {
          std::array<const float*, kMaxOrder> references;
          TF_RETURN_IF_ERROR(
              References(c, needed[k], &references, &hold, &buffers_read, &packed));
          TF_RETURN_IF_ERROR(
              DecodeTq(c, needed[k], output, references, &hold, &buffers_read, &packed));
        }
        // Arriving from here on: a reference for the frames below.
        slot.arriving = buffers[k];
        slot.pass = pass_;
      }
      cudaEvent_t event;
      TF_RETURN_IF_ERROR(NewEvent(&event));
      RETURN_IF_CUDA_ERROR(cudaEventRecord(event, h2d_));
      auto arrived = std::make_shared<Arrival>(this, event);
      if (!hold.empty()) reads_.push_back({std::move(hold), arrived});
      // A buffer read here is written again only after this decode.
      for (int64_t buffer : buffers_read) RecordOf(buffer).reader = arrived;
      for (size_t k = first; k < last; ++k) slots_[needed[k]].arrived = arrived;
      for (int64_t index : reading_) slots_[index].reader = arrived;
      reading_.clear();
    }
    wake_.notify_one();  // a read lease to release
    return absl::OkStatus();
  }

  // The compute stream waits for a group's copy, once per group.
  Status Await(cudaStream_t compute, const std::shared_ptr<Arrival>& arrived) {
    if (arrived == last_awaited_) return absl::OkStatus();
    RETURN_IF_CUDA_ERROR(cudaStreamWaitEvent(compute, arrived->event, 0));
    last_awaited_ = arrived;
    return absl::OkStatus();
  }

  // Forgets a slot's GPU frames; an arriving one is released only behind its
  // copy on the compute stream. A copy out (or encoding) of the old frame
  // keeps its own reference and no longer updates the slot (a queued one is
  // skipped).
  Status Drop(int64_t index, cudaStream_t compute) {
    Slot& slot = slots_[index];
    if (Held(slot.arriving)) {
      TF_RETURN_IF_ERROR(Await(compute, slot.arrived));
      slot.arriving = Tensor();
    }
    slot.arrived.reset();
    slot.resident = Tensor();
    slot.buffer = -1;
    slot.packet = Tensor();
    slot.packet_bytes = 0;
    slot.in_transit = false;
    slot.on_host = false;
    slot.keep = false;
    slot.encoding = false;
    slot.order = 0;
    TF_RETURN_IF_ERROR(AwaitReader(compute, slot));
    slot.reference = Tensor();
    referenced_.erase(index);
    return absl::OkStatus();
  }

  Status NewEvent(cudaEvent_t* event) {
    if (!free_events_.empty()) {
      *event = free_events_.back();
      free_events_.pop_back();
      return absl::OkStatus();
    }
    RETURN_IF_CUDA_ERROR(cudaEventCreateWithFlags(event, cudaEventDisableTiming));
    return absl::OkStatus();
  }
  // An event that records a time (frame-rate marks, copy groups).
  Status NewTimedEvent(cudaEvent_t* event) {
    if (!free_timed_.empty()) {
      *event = free_timed_.back();
      free_timed_.pop_back();
      return absl::OkStatus();
    }
    RETURN_IF_CUDA_ERROR(cudaEventCreate(event));
    return absl::OkStatus();
  }

  void PrintStats() const {
    int device = -1;
    cudaGetDevice(&device);
    const double rate = LinkRate();
    fprintf(stderr,
            "V1_FRAME_STORE_STATS device=%d copied_out=%ld fetched_back=%ld taken_on_gpu=%ld "
            "not_admitted=%ld codec=%s raw_bytes=%zu packet_bytes=%zu direct_decodes=%ld "
            "sent_raw=%ld sent_encoded=%ld encoded=%ld kept_packed=%ld link_gbps=%.2f "
            "frames_per_s=%.1f ring_mb=%.1f ring_peak_mb=%.1f ring_full=%ld chain_walks=%ld "
            "references=%zu leases=%zu\n",
            device, (long)stats_.copied_out, (long)stats_.fetched, (long)stats_.in_place,
            (long)stats_.not_admitted, CodecName(codec_), stats_.raw_bytes,
            stats_.packet_bytes, (long)stats_.direct_decodes, (long)stats_.sent_raw,
            (long)stats_.sent_encoded, (long)stats_.encoded, (long)stats_.kept_packed,
            rate > 0.0 ? rate * 1e-6 : 0.0, frame_rate_ * 1e3,
            compressed() ? ring_bytes_ * 1e-6 : 0.0, ring_peak_ * 1e-6,
            (long)stats_.ring_full, (long)stats_.chain_walks, referenced_.size(), reads_.size());
  }

  std::mutex mu_;
  std::condition_variable wake_;  // work for the admission thread
  Status admission_status_;       // the admission thread's error, if any
  int device_ = -1;
  cudaStream_t d2h_ = nullptr, h2d_ = nullptr, encode_ = nullptr;
  cudaStream_t compute_ = nullptr;  // the compute stream of this store's GPU
  cudaEvent_t forward_end_ = nullptr, h2d_marker_ = nullptr, pack_marker_ = nullptr;
  cudaEvent_t packed_ = nullptr;  // behind the latest pack (encode_)
  bool forward_end_recorded_ = false;  // forward_end_ marks this rollout's last frame
  char* host_ = nullptr;
  size_t frame_bytes_ = 0, host_stride_ = 0;
  int64_t capacity_ = 0;  // the current rollout's frames
  int64_t latest_ = -1;   // the latest frame handed over in this rollout
  DataType dtype_ = DT_INVALID;
  TensorShape shape_;
  // The codec state (a codec only for float32 frames; raw otherwise).
  Codec codec_ = Codec::kRaw;
  size_t encoded_capacity_ = 0;  // the largest encoded frame
  // TQ: the frame layout, the ring arena and its records (encode order), the
  // device state (cursor, look-back status) and the mapped encode results.
  tq::Layout layout_;
  Tensor ring_, scratch_;
  size_t ring_bytes_ = 0;
  std::deque<Record> records_;
  int64_t next_record_ = 0;
  unsigned long long host_head_ = 0;  // end of the latest frame known placed
  unsigned long long ring_peak_ = 0;  // largest ring bytes in use, this rollout
  tq::Result* host_results_ = nullptr;
  tq::Result* device_results_ = nullptr;
  std::vector<Entry> waiting_;  // chosen for encoding, waiting for references
  std::deque<Recent> recent_;
  std::set<int64_t> referenced_;  // slots holding a reference
  std::vector<int64_t> reading_;  // references read by the decodes being issued
  uint64_t pass_ = 0;
  int64_t last_take_ = std::numeric_limits<int64_t>::max();
  Tensor decode_buffer_;  // an encoded frame back from the host, decoded on h2d_
  Allocator* allocator_ = nullptr;
  // The adaptive codec (see Decide): link bytes and ms (decayed sums), the
  // forward's frames per ms on the GPU, the encoded length per raw byte, the
  // GPU's latest completed frame, the error-diffusion accumulator and a
  // simulated link rate in GB/s (V1_FRAME_STORE_LINK_GBPS; -1: measured).
  double link_bytes_ = 0.0, link_ms_ = 0.0, frame_rate_ = 0.0, ratio_ = kInitialRatio;
  int64_t produced_index_ = -1;
  double share_ = 0.0, link_override_ = -1.0;
  std::deque<Mark> marks_;
  Mark first_mark_;
  std::vector<Slot> slots_;
  std::vector<Entry> staged_;
  std::deque<Group> encoding_, queued_, pending_;
  std::deque<ReadLease> reads_;
  std::vector<cudaEvent_t> free_events_, free_timed_;
  std::shared_ptr<Arrival> last_awaited_;
  uint64_t generation_ = 0;
  // Per rollout (reset by the offload of frame 0), printed by the take of
  // frame 0 when V1_FRAME_STORE_STATS is set.
  struct Stats {
    int64_t copied_out = 0, fetched = 0, in_place = 0, not_admitted = 0;
    int64_t direct_decodes = 0;
    int64_t sent_raw = 0, sent_encoded = 0, encoded = 0, kept_packed = 0;
    int64_t ring_full = 0, chain_walks = 0;
    size_t raw_bytes = 0, packet_bytes = 0;
  } stats_;
};

// The store of (current GPU, key). Stores live for the process: their pinned
// buffers are reused by every rollout, and freeing them at exit would race
// the CUDA runtime's own teardown.
Status LookupStore(const std::string& key, FrameStore** store) {
  static std::mutex registry_mu;
  static auto* registry = new std::map<std::pair<int, std::string>, std::unique_ptr<FrameStore>>;
  int device;
  RETURN_IF_CUDA_ERROR(cudaGetDevice(&device));
  std::lock_guard<std::mutex> lock(registry_mu);
  auto& entry = (*registry)[{device, key}];
  if (!entry) entry = std::make_unique<FrameStore>();
  *store = entry.get();
  return absl::OkStatus();
}

Status ScalarIndex(const Tensor& index, int64_t* value, bool negative_allowed = false) {
  if (!TensorShapeUtils::IsScalar(index.shape()))
    return errors::InvalidArgument("index must be a scalar");
  *value = index.scalar<int32_t>()();
  if (*value < 0 && !negative_allowed)
    return errors::InvalidArgument("index must be non-negative");
  return absl::OkStatus();
}

class OffloadOp : public OpKernel {
 public:
  explicit OffloadOp(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("key", &key_));
    OP_REQUIRES_OK(c, c->GetAttr("capacity", &capacity_));
    OP_REQUIRES_OK(c, c->GetAttr("resident_from", &resident_from_));
    OP_REQUIRES_OK(c, c->GetAttr("group", &group_));
    std::string codec;
    OP_REQUIRES_OK(c, c->GetAttr("codec", &codec));
    codec_ = codec == "tq" ? Codec::kTq : Codec::kRaw;
    OP_REQUIRES_OK(c, c->GetAttr("drop_outside", &drop_outside_));
  }
  void Compute(OpKernelContext* c) override {
    int64_t i;
    OP_REQUIRES_OK(c, ScalarIndex(c->input(1), &i, drop_outside_));
    Tensor* token;
    OP_REQUIRES_OK(c, c->allocate_output(0, TensorShape({0}), &token));
    if (drop_outside_ && (i < 0 || i >= capacity_)) return;
    OP_REQUIRES(c, i < capacity_,
                errors::InvalidArgument("index ", i, " is outside [0, ", capacity_, ")"));
    OP_REQUIRES(c, c->input(0).NumElements() > 0,
                errors::InvalidArgument("frames must not be empty"));
    FrameStore* store;
    OP_REQUIRES_OK(c, LookupStore(key_, &store));
    OP_REQUIRES_OK(c,
                   store->Offload(c, i, capacity_, resident_from_, group_, codec_, c->input(0)));
  }

 private:
  std::string key_;
  int64_t capacity_, resident_from_, group_;
  Codec codec_;
  bool drop_outside_;
};

class TakeOp : public OpKernel {
 public:
  explicit TakeOp(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("key", &key_));
    OP_REQUIRES_OK(c, c->GetAttr("T", &dtype_));
    OP_REQUIRES_OK(c, c->GetAttr("prefetch", &prefetch_));
    OP_REQUIRES_OK(c, c->GetAttr("group", &group_));
    PartialTensorShape shape;
    OP_REQUIRES_OK(c, c->GetAttr("shape", &shape));
    OP_REQUIRES(c, shape.AsTensorShape(&shape_),
                errors::InvalidArgument("shape must be fully defined"));
    OP_REQUIRES_OK(c, c->GetAttr("zero_before_start", &zero_before_start_));
  }
  void Compute(OpKernelContext* c) override {
    int64_t i;
    OP_REQUIRES_OK(c, ScalarIndex(c->input(0), &i, zero_before_start_));
    if (i < 0) {  // before the record's start: a zero frame, on the compute stream
      Tensor* frame;
      OP_REQUIRES_OK(c, c->allocate_output(0, shape_, &frame));
      const cudaError_t status =
          cudaMemsetAsync(const_cast<char*>(frame->tensor_data().data()), 0,
                          frame->TotalBytes(), c->eigen_device<GPUDevice>().stream());
      OP_REQUIRES(c, status == cudaSuccess,
                  errors::Internal("cudaMemsetAsync failed: ", cudaGetErrorString(status)));
      return;
    }
    FrameStore* store;
    OP_REQUIRES_OK(c, LookupStore(key_, &store));
    OP_REQUIRES_OK(c, store->Take(c, i, prefetch_, group_, dtype_, shape_));
  }

 private:
  std::string key_;
  DataType dtype_;
  int64_t prefetch_, group_;
  TensorShape shape_;
  bool zero_before_start_;
};

}  // namespace

#define REG(T)                                                                    \
  REGISTER_KERNEL_BUILDER(Name("V1FrameStoreOffload")                             \
                              .Device(DEVICE_GPU)                                 \
                              .HostMemory("index")                                \
                              .TypeConstraint<T>("T"),                            \
                          OffloadOp);                                             \
  REGISTER_KERNEL_BUILDER(Name("V1FrameStoreTake")                                \
                              .Device(DEVICE_GPU)                                 \
                              .HostMemory("index")                                \
                              .TypeConstraint<T>("T"),                            \
                          TakeOp);
REG(float); REG(Eigen::half); REG(int32_t);
#undef REG
#endif

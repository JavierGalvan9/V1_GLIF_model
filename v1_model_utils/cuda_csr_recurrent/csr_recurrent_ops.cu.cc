#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include <cub/block/block_scan.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_select.cuh>
#include <cub/iterator/counting_input_iterator.cuh>
#include <cub/iterator/transform_input_iterator.cuh>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <limits>
#include <memory>
#include <type_traits>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_var.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

using namespace tensorflow;
using GPUDevice = Eigen::GpuDevice;

#ifndef V1_FORWARD_THREADS
#define V1_FORWARD_THREADS 128
#endif
#ifndef V1_FORWARD_FLOAT_ACCUM
#define V1_FORWARD_FLOAT_ACCUM 0
#endif
#ifndef V1_DIRECT_CSR
#define V1_DIRECT_CSR 0
#endif

// With direct-CSR weights the caller keeps `weights` and `weight_grad` in CSR
// edge order, so the CSR position is the weight index and the effectively
// random `edge_ids[csr]` gather disappears from every inner loop. The choice is
// global: forward and backward must agree.
#if V1_DIRECT_CSR
#define V1_EDGE_INDEX(csr) (csr)
#else
#define V1_EDGE_INDEX(csr) (edge_ids[csr])
#endif

#include "event_weight_grad.cuh"

// The forward's device-built active-row queue holds each active (batch,
// presynaptic row) slot as its flat index batch * n_pre + row, so it needs
// B * n_pre < 2^31 (cub's 32-bit item count); the consumer divides once per
// active slot. The entries are plain `unsigned int`, not `uint32`: the
// resource library compiles this file with `uint32` redefined as a signed type.
constexpr int64_t kQueueMaxSlots = std::numeric_limits<int>::max();

// A synapse type's four FP32 basis values in one vector load.
__device__ __forceinline__ ::float4 LoadBasis4(const float* basis, int type) {
  return *reinterpret_cast<const ::float4*>(basis + type * 4);
}

template <typename T>
__device__ __forceinline__ float AsFloat(T value) {
  return static_cast<float>(value);
}

template <typename T>
__device__ __forceinline__ T FromFloat(float value) {
  return static_cast<T>(value);
}

template <typename T>
__device__ __forceinline__ void AtomicAddValue(T* address, float value);

template <>
__device__ __forceinline__ void AtomicAddValue<float>(float* address,
                                                       float value) {
  atomicAdd(address, value);
}

template <>
__device__ __forceinline__ void AtomicAddValue<Eigen::half>(
    Eigen::half* address, float value) {
  atomicAdd(reinterpret_cast<__half*>(address), __float2half(value));
}

// Two adjacent receptors in one atomic where the element type allows it.
template <typename O>
__device__ __forceinline__ void AtomicAddPair(O* address, float first, float second) {
  if constexpr (std::is_same<O, Eigen::half>::value) {
    atomicAdd(reinterpret_cast<__half2*>(address), __floats2half2_rn(first, second));
  } else {
    atomicAdd(address, first);
    atomicAdd(address + 1, second);
  }
}

// All four receptors of one target in one atomic. On sm_90+ that is a single
// vector reduction: 8-byte `red.v2.f16x2` for FP16 (SASS REDG.ADD.F16x4), each
// element still rounded on its own exactly as two f16x2 atomics round it, and
// 16-byte `red.v4.f32` for FP32. The L2 atomic units see half (a quarter) of
// the operations. `address` is aligned to the vector because every output row
// holds four elements. No "memory" clobber: no forward kernel reads the
// currents it accumulates, and the kernel boundary orders the reductions
// before any reader, so the compiler may schedule loads across them.
template <typename O>
__device__ __forceinline__ void AtomicAddQuad(O* address, float first, float second,
                                              float third, float fourth) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if constexpr (std::is_same<O, Eigen::half>::value) {
    const __half2 low = __floats2half2_rn(first, second);
    const __half2 high = __floats2half2_rn(third, fourth);
    asm volatile("red.global.add.noftz.v2.f16x2 [%0], {%1, %2};"
                 :: "l"(address), "r"(*reinterpret_cast<const unsigned int*>(&low)),
                    "r"(*reinterpret_cast<const unsigned int*>(&high)));
    return;
  } else if constexpr (std::is_same<O, float>::value) {
    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};"
                 :: "l"(address), "f"(first), "f"(second), "f"(third), "f"(fourth));
    return;
  }
#endif
  AtomicAddPair(address, first, second);
  AtomicAddPair(address + 2, third, fourth);
}

// ---------------------------------------------------------------------------
// Forward
// ---------------------------------------------------------------------------

// The device-built list of active (batch, presynaptic row) slots the scatter
// forward visits, as an ordered stream compaction: `tf.where` computed the same
// list, but it copies the count to the host to size its output and blocks every
// forward on that round trip. Here the queue is sized for every slot and the
// count stays on the device. The order is the row-major slot order, which is
// what makes concurrent blocks scatter into the same few samples' currents and,
// with the Morton layout, into neighbouring postsynaptic neurons.
//
// The compaction selects the slot index (a 32-bit counting iterator) under a
// flag read straight from the spike array. Packing (batch, row) into the entry
// instead cost an int64 division and modulo for every one of the B * n_pre
// slots, active or not, and int64 item offsets throughout cub's sweep.
template <typename T>
struct SlotIsActive {
  __host__ __device__ bool operator()(const T& spike) const {
    return static_cast<float>(spike) != 0.0f;
  }
};

// sample_starts[b] = the first queue position of sample b (the queue is
// ordered by slot), for b in [0, batch]; one thread per queue entry.
__global__ void QueueSampleStartsKernel(int64_t n_pre, int batch, const unsigned int* queue,
                                        const unsigned int* queue_count,
                                        unsigned int* sample_starts) {
  const unsigned int total = *queue_count;
  const unsigned int stride = gridDim.x * blockDim.x;
  if (blockIdx.x == 0 && threadIdx.x == 0 && total == 0) {
    for (int sample = 0; sample <= batch; ++sample) sample_starts[sample] = 0;
  }
  for (unsigned int index = blockIdx.x * blockDim.x + threadIdx.x; index < total;
       index += stride) {
    const int sample = static_cast<int>(queue[index] / static_cast<unsigned int>(n_pre));
    const int before =
        index == 0 ? -1 : static_cast<int>(queue[index - 1] / static_cast<unsigned int>(n_pre));
    for (int covered = before + 1; covered <= sample; ++covered) sample_starts[covered] = index;
    if (index == total - 1) {
      for (int covered = sample + 1; covered <= batch; ++covered) {
        sample_starts[covered] = total;
      }
    }
  }
}

// A slot's queue record: the flat indices batch * width + column of its active
// entries in order (room for all batch * width), then their count, then the
// batch + 1 sample starts. A slot's record depends only on the slot tensor, so
// the recurrent spike history, whose slot k at one step is slot k - 1 at the
// step before, carries its records along instead of sweeping every slot again
// (models.V1Column): each step computes only the newest slot's.
inline int64_t QueueRecordWords(int64_t batch, int64_t width) {
  return batch * width + batch + 2;
}

struct QueueRecords {
  const unsigned int* record[kMaxSpikeSlots];
};

// Several slots: each slot's record was built on its own. Entry j of slot d, in
// sample b, goes to the position it has in the queue of the concatenated
// matrix, which is ordered by sample, then slot, then column:
//   sum_d' starts[d'][b] + sum_{d' < d} (starts[d'][b + 1] - starts[d'][b]) + j - starts[d][b],
// as the value b * n_pre + d * width + column.
__global__ void MergeSlotQueuesKernel(int slots, unsigned int width, FastDivider by_width,
                                      QueueRecords records, int64_t capacity,
                                      unsigned int* queue, unsigned int* queue_count) {
  const unsigned int stride = gridDim.x * blockDim.x;
  const unsigned int n_pre = width * slots;
  unsigned int total = 0;
  for (int slot = 0; slot < slots; ++slot) {
    const unsigned int* entries = records.record[slot];
    const unsigned int count = entries[capacity];
    const unsigned int* starts = entries + capacity + 1;
    total += count;
    for (unsigned int j = blockIdx.x * blockDim.x + threadIdx.x; j < count; j += stride) {
      const unsigned int entry = entries[j];
      const unsigned int sample = by_width.Div(entry);
      unsigned int position = j - starts[sample];
      for (int other = 0; other < slots; ++other) {
        const unsigned int* other_starts = records.record[other] + capacity + 1;
        position += other < slot ? other_starts[sample + 1] : other_starts[sample];
      }
      queue[position] = sample * n_pre + slot * width + (entry - sample * width);
    }
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) *queue_count = total;
}

// Selects the flat indices of the non-zero entries of `count` values.
template <typename T>
Status SelectNonzero(OpKernelContext* context, const T* values, int64_t count,
                     unsigned int* selected, unsigned int* selected_count) {
  const cub::CountingInputIterator<unsigned int> indices(0);
  const cub::TransformInputIterator<bool, SlotIsActive<T>, const T*> active(
      values, SlotIsActive<T>{});
  const int items = static_cast<int>(count);
  auto stream = context->eigen_device<GPUDevice>().stream();
  size_t scratch_bytes = 0;
  cub::DeviceSelect::Flagged(nullptr, scratch_bytes, indices, active, selected, selected_count,
                             items, stream);
  Tensor scratch;
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DT_INT8, TensorShape({static_cast<int64_t>(scratch_bytes)}), &scratch));
  if (cub::DeviceSelect::Flagged(scratch.flat<int8>().data(), scratch_bytes, indices, active,
                                 selected, selected_count, items, stream) != cudaSuccess) {
    return errors::Internal("building the active-row queue failed");
  }
  return OkStatus();
}

// Builds the queue record (see QueueRecordWords) of one [batch, width] slot.
template <typename T>
Status BuildQueueRecord(OpKernelContext* context, const T* slot, int64_t batch,
                        int64_t width, unsigned int* record) {
  const int64_t capacity = batch * width;
  TF_RETURN_IF_ERROR(SelectNonzero<T>(context, slot, capacity, record, record + capacity));
  return GpuLaunchKernel(QueueSampleStartsKernel, 64, 256, 0,
                         context->eigen_device<GPUDevice>().stream(), width,
                         static_cast<int>(batch), record, record + capacity,
                         record + capacity + 1);
}

// Every slot's queue record: `carried.record[d]` is slot d's when the caller
// has it (null otherwise), `newest` receives slot 0's (null: a temporary), and
// the other missing ones are built into `scratch`, which must outlive their use.
template <typename T>
Status BuildMissingRecords(OpKernelContext* context, const SpikeMatrix<const T*>& spikes,
                           const QueueRecords& carried, unsigned int* newest,
                           QueueRecords* records, Tensor* scratch) {
  const SpikeSlots<const T*>& slots = spikes.slots;
  const int64_t words = QueueRecordWords(spikes.batch, slots.width);
  int missing = 0;
  for (int slot = 0; slot < slots.count; ++slot) {
    missing += carried.record[slot] == nullptr && !(slot == 0 && newest != nullptr);
  }
  unsigned int* next = nullptr;
  if (missing) TF_RETURN_IF_ERROR(AllocateQueueWords(context, missing * words, scratch, &next));
  *records = carried;
  for (int slot = 0; slot < slots.count; ++slot) {
    if (records->record[slot] != nullptr) continue;
    unsigned int* record = slot == 0 && newest != nullptr ? newest : next;
    if (record == next) next += words;
    TF_RETURN_IF_ERROR(BuildQueueRecord<T>(context, slots.slot[slot], spikes.batch,
                                           slots.width, record));
    records->record[slot] = record;
  }
  return OkStatus();
}

// The queue of active (batch, row) slots of the concatenated [batch, n_pre]
// matrix, in its row-major order. With several slot tensors, a single
// compaction over the concatenated index would locate every element's tensor
// with two divisions, which made the sweep compute-bound (132 us against 50 at
// B = 64); each tensor is compacted with the plain streaming select instead,
// and the few active entries are merged into the concatenated order.
template <typename T>
Status BuildActiveQueue(OpKernelContext* context, const SpikeMatrix<const T*>& spikes,
                        unsigned int* queue, unsigned int* queue_count,
                        const QueueRecords& carried, unsigned int* newest) {
  const SpikeSlots<const T*>& slots = spikes.slots;
  if (slots.count == 1) {
    return SelectNonzero<T>(context, slots.slot[0], spikes.NumElements(), queue, queue_count);
  }
  QueueRecords records;
  Tensor scratch;
  TF_RETURN_IF_ERROR(BuildMissingRecords<T>(context, spikes, carried, newest, &records, &scratch));
  return GpuLaunchKernel(MergeSlotQueuesKernel, 64, 256, 0,
                         context->eigen_device<GPUDevice>().stream(), slots.count,
                         slots.width, slots.by_width, records, spikes.batch * slots.width,
                         queue, queue_count);
}

// Sums each of `values` over the lanes of this lane's run (the lanes `group`
// names, which are contiguous) that sit at or above it: a segmented
// Hillis-Steele suffix scan, so the run's lowest lane ends up holding the run's
// totals. Membership of `lane + offset` is tested once per step for all values.
template <int kValues>
__device__ __forceinline__ void RunSuffixSums(float (&values)[kValues], unsigned int group,
                                              int lane) {
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    const int source = lane + offset;
    const bool take = source < 32 && ((group >> source) & 1u);
#pragma unroll
    for (int index = 0; index < kValues; ++index) {
      const float other = __shfl_down_sync(0xffffffffu, values[index], offset);
      if (take) values[index] += other;
    }
  }
}

// Scatter forward over the ordered device-built queue. A fixed grid consumes a
// queue whose length only the device knows, each block taking the next slots
// from an atomic ticket, so consumption follows queue order: the atomics at any
// moment land in a few samples' currents and stay in L2. (A static stride over
// the same ordered queue lets blocks drift apart; once the currents outgrow L2
// it measured 1.5x slower at batch 128 and 3.2x at 512 on the 203,816-neuron
// network, for at most 0.02 ms gained at batch <= 32.)
// The ticket also balances the heavy-tailed row lengths. A ticket covers
// `slots_per_ticket` consecutive slots, about 1,024 edges of work: one long LGN
// row (~5,500 edges), or seven short recurrent ones (~140), which keeps the
// ticket from becoming the bottleneck when rows are short and numerous. Each
// slot takes the whole block, so concurrent atomics stay on neighbouring
// targets; giving each warp its own slot measured 1.4-2x slower.
//
// With kAggregate, contributions from edges of the same row that land on the
// same postsynaptic neuron are summed in FP32 registers and committed by a
// single lane: fewer atomics, and the sum happens in FP32 rather than through
// repeated narrow read-modify-writes, so it is also more accurate. LGN rows
// average 1.42 edges per target (one per synapse type), which is where this
// pays. Recurrent rows never repeat a target (the connectivity says so, see
// csr_order.repeats_targets), so they need no warp collectives: with four basis
// columns they take TiledForwardKernel below, otherwise this kernel's plain
// one-thread-per-edge scatter.
//
// Aggregation relies on the CSR invariant that postsynaptic ids ascend within a
// row, so a run occupies consecutive lanes. Its loop is warp-uniform (every
// lane of a warp shares `base`) because __match_any_sync requires all named
// lanes to arrive together. kBasis == 4 unrolls the receptors; 0 loops over
// `n_basis`.
template <typename T, typename W, typename O, int kBasis, bool kAggregate>
__global__ void ForwardKernel(
    int64_t n_pre, int n_post, int n_basis, SpikeSlots<const T*> spikes,
    const unsigned int* queue, const unsigned int* queue_count, unsigned int* ticket,
    unsigned int slots_per_ticket,
    const W* weights, const uint32* post_ids, const uint8* synapse_types,
    const uint32* row_splits, const uint32* edge_ids, const float* basis,
    O* currents) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int warps = blockDim.x >> 5;
  const unsigned int total = *queue_count;
  const int basis_width = kBasis == 4 ? 4 : n_basis;
  __shared__ unsigned int next_slot;

  for (;;) {
    if (threadIdx.x == 0) next_slot = atomicAdd(ticket, slots_per_ticket);
    __syncthreads();
    const unsigned int first_slot = next_slot;
    __syncthreads();   // every thread has read it before thread 0 overwrites it
    if (first_slot >= total) break;
    const unsigned int last_slot = min(first_slot + slots_per_ticket, total);
  for (unsigned int active_id = first_slot; active_id < last_slot; ++active_id) {
    const unsigned int slot = queue[active_id];
    const unsigned int batch = slot / static_cast<unsigned int>(n_pre);
    const unsigned int pre = slot - batch * static_cast<unsigned int>(n_pre);
    const float spike = AsFloat(spikes.At(batch, pre));
    const uint32 start = row_splits[pre];
    const uint32 end = row_splits[pre + 1];

    if constexpr (!kAggregate) {
      for (uint32 csr = start + threadIdx.x; csr < end; csr += blockDim.x) {
        const float weighted = spike * AsFloat(weights[V1_EDGE_INDEX(csr)]);
        const int type = synapse_types[csr];
        O* output =
            currents + (batch * static_cast<int64_t>(n_post) + post_ids[csr]) * basis_width;
        if constexpr (kBasis == 4) {
          const ::float4 bv = LoadBasis4(basis, type);
          AtomicAddQuad(output, weighted * bv.x, weighted * bv.y, weighted * bv.z,
                        weighted * bv.w);
        } else if ((n_basis & 1) == 0) {
          for (int receptor = 0; receptor < n_basis; receptor += 2) {
            AtomicAddPair(output + receptor, weighted * basis[type * n_basis + receptor],
                          weighted * basis[type * n_basis + receptor + 1]);
          }
        } else {
          for (int receptor = 0; receptor < n_basis; ++receptor) {
            AtomicAddValue(output + receptor, weighted * basis[type * n_basis + receptor]);
          }
        }
      }
      continue;
    }

    for (uint32 base = start + warp * 32; base < end; base += warps * 32) {
      const uint32 csr = base + lane;
      const bool valid = csr < end;
      // Invalid lanes take a post no real edge can have, so they never merge
      // with a live run (they share one run of their own, whose leader the
      // `valid` test below excludes).
      const unsigned int post = valid ? post_ids[csr] : 0xffffffffu;
      const float weighted = valid ? spike * AsFloat(weights[V1_EDGE_INDEX(csr)]) : 0.0f;
      const int type = valid ? synapse_types[csr] : 0;
      const unsigned int group = __match_any_sync(0xffffffffu, post);
      // When every run in the warp is a singleton the scan cannot move anything,
      // and one vote is far cheaper than the shuffles.
      const bool any_run = !__all_sync(0xffffffffu, __popc(group) == 1);
      const bool leader = valid && lane == __ffs(group) - 1;
      O* output = currents + (batch * static_cast<int64_t>(n_post) + post) * basis_width;
      if constexpr (kBasis == 4) {
        const ::float4 bv = LoadBasis4(basis, type);
        float value[4] = {weighted * bv.x, weighted * bv.y, weighted * bv.z, weighted * bv.w};
        if (any_run) RunSuffixSums(value, group, lane);
        if (leader) {
          AtomicAddQuad(output, value[0], value[1], value[2], value[3]);
        }
      } else {
        // A half2 atomic needs a 4-byte aligned address, which every receptor
        // pair has only when the row width is even.
        const bool paired = (n_basis & 1) == 0;
        for (int receptor = 0; receptor < n_basis; receptor += paired ? 2 : 1) {
          float value[2] = {weighted * basis[type * n_basis + receptor],
                            paired ? weighted * basis[type * n_basis + receptor + 1] : 0.0f};
          if (any_run) RunSuffixSums(value, group, lane);
          if (!leader) continue;
          if (paired) {
            AtomicAddPair(output + receptor, value[0], value[1]);
          } else {
            AtomicAddValue(output + receptor, value[0]);
          }
        }
      }
    }
  }
  }
}

// The inverse of CastForwardOutputKernel. With V1_FORWARD_FLOAT_ACCUM the
// scatter lands in an fp32 side buffer, so that buffer -- not `output` -- is
// what the kernel accumulates onto. `output` has already been seeded by the op
// with the `initial` tensor (the LGN -> BKG -> recurrent accumulator chain) or
// with zeros, so the fp32 buffer must start from it. Memsetting it to zero
// instead silently drops `initial`.
template <typename T>
__global__ void SeedForwardAccumKernel(int64_t elements, const T* input,
                                       float* output) {
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < elements; index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    output[index] = AsFloat(input[index]);
  }
}

template <typename T>
__global__ void CastForwardOutputKernel(int64_t elements, const float* input,
                                        T* output) {
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < elements; index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    output[index] = FromFloat<T>(input[index]);
  }
}

// ---------------------------------------------------------------------------
// Tiled forward (rows that never repeat a target, four basis columns)
//
// The postsynaptic neurons are cut into tiles of kTileTargets. One block owns
// one (sample, tile): it zeroes an FP32 [kTileTargets, 4] accumulator in
// shared memory, walks the sample's active rows (the ordered queue), takes the
// segment of each row that lands in its tile, accumulates those edges with
// shared-memory atomics, and writes the tile's currents once, rounded once:
// `currents = T(initial + sum)`.
//
// Against the global-atomic scatter this removes the memset of the currents
// and every global read-modify-write: at batch 64 the recurrent atomics touch
// about 2.3 M 32-byte sectors of the 104 MB currents, nearly all missing L2.
// The output is instead written once, coalesced. It is also more accurate:
// each current is an FP32 sum rounded once, instead of one FP16 rounding per
// contribution. The order of the FP32 additions within a tile still follows
// the shared atomics, so the result is not bitwise reproducible, but its
// run-to-run spread is FP32-sized.
//
// Rows' postsynaptic ids ascend, so a row's edges into one tile are one
// contiguous segment. ForwardTileSegments lists them per row, sorted by tile;
// the op builds that table on the device on first use and keeps it.
// ---------------------------------------------------------------------------

#ifndef V1_TILE_TARGETS
#define V1_TILE_TARGETS 5120
#endif
// 5,120 targets x 4 FP32 = 80 KiB of shared accumulators; sm_120 allows at
// most 99 KiB per block. Recurrent rows spread over many Morton tiles (25
// segments of ~7 edges per row at 2,048 targets, 17 of ~11 at 5,120), and
// every tile block checks every active row of its sample, so the tile is as
// wide as shared memory allows: 2,560 targets measured 193 us against 176.
constexpr int kTileTargets = V1_TILE_TARGETS;
#ifndef V1_TILE_THREADS
#define V1_TILE_THREADS 1024
#endif
constexpr int kTileThreads = V1_TILE_THREADS;

// Per-row tile segments of one CSR: row r's segments are
// [segment_splits[r], segment_splits[r + 1]), segment s starting at CSR
// position segment_starts[s] and landing in tile segment_tiles[s]. Immutable
// tables are shared by operators using the same CSR buffers on the same device.
// Keeping the source tensors prevents their addresses being reused while any
// operator still holds the table.
struct ForwardTileSegments {
  const DeviceBase* device = nullptr;
  Tensor post_ids;
  Tensor row_splits;
  int n_post = -1;
  Tensor segment_splits;
  Tensor segment_starts;
  Tensor segment_tiles;
  // Bit t of row r's mask is set when the row has a segment in tile t (only
  // built when there are at most 64 tiles; otherwise the kernel searches).
  Tensor segment_masks;
};

struct ForwardTileCache {
  mutex mu;
  std::shared_ptr<const ForwardTileSegments> segments;
  // Whether this op's TiledForwardKernel may use kTileTargets of dynamic
  // shared memory. Kept per op, not in a function-local static: the resource
  // library compiles these sources too, and GCC makes template statics unique
  // process-wide, so one library would configure only its own copy of the
  // kernel and leave the other's launch invalid.
  bool kernel_checked = false;
  bool kernel_available = false;
  // The same for each tile shape of CsrTileSweepKernel (SweepShape), which
  // keep their own tables because their tile sizes need not be kTileTargets.
  struct Sweep {
    std::shared_ptr<const ForwardTileSegments> segments;
    bool checked = false;
    bool available = false;
  } sweep[2];
};

// Warp per row: count (kFill = false) or write (kFill = true) the row's tile
// segments. A segment starts at the row's first edge and wherever the tile of
// the postsynaptic id changes.
template <bool kFill, int kTargets>
__global__ void TileSegmentsKernel(int64_t n_pre, const uint32* row_splits,
                                   const uint32* post_ids, unsigned int* segment_splits,
                                   unsigned int* segment_starts,
                                   unsigned int* segment_tiles,
                                   unsigned long long* segment_masks) {
  const int lane = threadIdx.x & 31;
  const int64_t warps = static_cast<int64_t>(gridDim.x) * (blockDim.x >> 5);
  for (int64_t row = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
       row < n_pre; row += warps) {
    const uint32 start = row_splits[row];
    const uint32 end = row_splits[row + 1];
    unsigned int written = kFill ? segment_splits[row] : 0;
    unsigned int previous = 0xffffffffu;
    for (uint32 base = start; base < end; base += 32) {
      const uint32 csr = base + lane;
      const unsigned int tile =
          csr < end ? static_cast<unsigned int>(post_ids[csr]) / kTargets : 0xfffffffeu;
      unsigned int before = __shfl_up_sync(0xffffffffu, tile, 1);
      if (lane == 0) before = previous;
      const bool starts = csr < end && tile != before;
      const unsigned int flags = __ballot_sync(0xffffffffu, starts);
      if (kFill && starts) {
        const unsigned int position = written + __popc(flags & ((1u << lane) - 1));
        segment_starts[position] = csr;
        segment_tiles[position] = tile;
        if (segment_masks != nullptr) atomicOr(segment_masks + row, 1ull << tile);
      }
      written += __popc(flags);
      previous = __shfl_sync(0xffffffffu, tile, 31);
    }
    if (!kFill && lane == 0) segment_splits[row] = written;
  }
}

// Build once, completing the fill before another operator/stream can read it.
template <int kTargets>
inline Status BuildTileSegments(OpKernelContext* context, const Tensor& post_ids,
                                const Tensor& row_splits, int n_post,
                                ForwardTileSegments* segments) {
  const int64_t n_pre = row_splits.NumElements() - 1;
  auto stream = context->eigen_device<GPUDevice>().stream();
  unsigned int* splits;
  TF_RETURN_IF_ERROR(
      AllocateQueueWords(context, n_pre + 1, &segments->segment_splits, &splits));
  constexpr int kThreads = 256;
  const int blocks = static_cast<int>(std::min<int64_t>((n_pre * 32 + kThreads - 1) / kThreads,
                                                        65535));
  cudaMemsetAsync(splits + n_pre, 0, sizeof(unsigned int), stream);
  TF_RETURN_IF_ERROR(GpuLaunchKernel(TileSegmentsKernel<false, kTargets>, blocks, kThreads, 0, stream,
                                     n_pre, row_splits.flat<uint32>().data(),
                                     post_ids.flat<uint32>().data(), splits, nullptr,
                                     nullptr, nullptr));
  size_t scratch_bytes = 0;
  cub::DeviceScan::ExclusiveSum(nullptr, scratch_bytes, splits, splits,
                                static_cast<int>(n_pre + 1), stream);
  Tensor scratch;
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DT_INT8, TensorShape({static_cast<int64_t>(scratch_bytes)}), &scratch));
  cub::DeviceScan::ExclusiveSum(scratch.flat<int8>().data(), scratch_bytes, splits, splits,
                                static_cast<int>(n_pre + 1), stream);
  unsigned int total = 0;
  cudaMemcpyAsync(&total, splits + n_pre, sizeof(unsigned int), cudaMemcpyDeviceToHost,
                  stream);
  if (cudaStreamSynchronize(stream) != cudaSuccess) {
    return errors::Internal("building the forward tile segments failed");
  }
  unsigned int* starts;
  unsigned int* tiles;
  TF_RETURN_IF_ERROR(AllocateQueueWords(context, std::max<int64_t>(total, 1),
                                        &segments->segment_starts, &starts));
  TF_RETURN_IF_ERROR(AllocateQueueWords(context, std::max<int64_t>(total, 1),
                                        &segments->segment_tiles, &tiles));
  unsigned int* masks = nullptr;
  segments->segment_masks = Tensor();
  if ((n_post + kTargets - 1) / kTargets <= 64) {
    TF_RETURN_IF_ERROR(
        AllocateQueueWords(context, 2 * n_pre, &segments->segment_masks, &masks));
    cudaMemsetAsync(masks, 0, 2 * n_pre * sizeof(unsigned int), stream);
  }
  TF_RETURN_IF_ERROR(GpuLaunchKernel(TileSegmentsKernel<true, kTargets>, blocks, kThreads, 0, stream,
                                     n_pre, row_splits.flat<uint32>().data(),
                                     post_ids.flat<uint32>().data(), splits, starts, tiles,
                                     reinterpret_cast<unsigned long long*>(masks)));
  const cudaError_t filled = cudaStreamSynchronize(stream);
  if (filled != cudaSuccess) {
    return errors::Internal("filling the forward tile segments failed: ",
                            cudaGetErrorString(filled));
  }
  segments->device = context->device();
  segments->post_ids = post_ids;
  segments->row_splits = row_splits;
  segments->n_post = n_post;
  return OkStatus();
}

inline bool MatchesTileSegments(const ForwardTileSegments& segments,
                                OpKernelContext* context, const Tensor& post_ids,
                                const Tensor& row_splits, int n_post) {
  return segments.device == context->device() && segments.n_post == n_post &&
         segments.post_ids.tensor_data().data() == post_ids.tensor_data().data() &&
         segments.row_splits.tensor_data().data() == row_splits.tensor_data().data() &&
         segments.post_ids.NumElements() == post_ids.NumElements() &&
         segments.row_splits.NumElements() == row_splits.NumElements();
}

// The registry owns only weak references: destroying the last operator releases
// its table and source tensors. Each operator keeps a strong reference, so the
// warmed timestep never takes the registry lock, allocates, or synchronizes.
// A registry per tile size also separates libraries built with different tuning.
// `held` is the operator's reference for this tile size.
template <int kTargets>
inline Status EnsureTileSegments(OpKernelContext* context, const Tensor& post_ids,
                                 const Tensor& row_splits, int n_post,
                                 std::shared_ptr<const ForwardTileSegments>* held) {
  if (*held && MatchesTileSegments(**held, context, post_ids, row_splits, n_post)) {
    return OkStatus();
  }
  struct Registry {
    mutex mu;
    std::vector<std::weak_ptr<const ForwardTileSegments>> tables;
  };
  static Registry registry;
  mutex_lock lock(registry.mu);
  for (auto it = registry.tables.begin(); it != registry.tables.end();) {
    auto table = it->lock();
    if (!table) {
      it = registry.tables.erase(it);
    } else {
      if (MatchesTileSegments(*table, context, post_ids, row_splits, n_post)) {
        *held = std::move(table);
        return OkStatus();
      }
      ++it;
    }
  }
  auto table = std::make_shared<ForwardTileSegments>();
  TF_RETURN_IF_ERROR(BuildTileSegments<kTargets>(
      context, post_ids, row_splits, n_post, table.get()));
  registry.tables.emplace_back(table);
  *held = std::move(table);
  return OkStatus();
}

// Row `row`'s segment in `tile`, if it has one: [*start, *stop) in CSR order.
// The mask names the row's tiles (the segment's rank is a popcount); without
// masks (more than 64 tiles) the row's sorted tiles are binary-searched.
__device__ __forceinline__ bool FindTileSegment(
    unsigned int row, unsigned int tile, const unsigned int* segment_splits,
    const unsigned int* segment_starts, const unsigned int* segment_tiles,
    const unsigned long long* segment_masks, const uint32* row_splits, uint32* start,
    uint32* stop) {
  unsigned int low = segment_splits[row];
  unsigned int last = 0;
  bool found;
  if (segment_masks != nullptr) {
    const unsigned long long mask = segment_masks[row];
    found = (mask >> tile) & 1;
    low += __popcll(mask & ((1ull << tile) - 1));
    last = segment_splits[row + 1];
  } else {
    unsigned int high = segment_splits[row + 1];
    last = high;
    while (low < high) {
      const unsigned int middle = (low + high) >> 1;
      if (segment_tiles[middle] < tile) low = middle + 1; else high = middle;
    }
    found = low < last && segment_tiles[low] == tile;
  }
  if (found) {
    *start = segment_starts[low];
    *stop = low + 1 < last ? segment_starts[low + 1] : row_splits[row + 1];
  }
  return found;
}

// Writes a tile's FP32 sums once, as T(initial + sum) in accumulate mode,
// else T(sum); `kStride` is the block size.
template <typename T, int kStride>
__device__ __forceinline__ void WriteTileSums(const ::float4* sums, int targets,
                                              bool accumulate, bool aligned, T* output) {
  for (int target = threadIdx.x; target < targets; target += kStride) {
    const ::float4 sum = sums[target];
    if constexpr (std::is_same<T, Eigen::half>::value) {
      float values[4] = {sum.x, sum.y, sum.z, sum.w};
      __half2* pairs = reinterpret_cast<__half2*>(output + target * 4);
      if (aligned) {   // 8-byte aligned row: one vector load and store
        ::uint2 packed = {0, 0};
        if (accumulate) packed = *reinterpret_cast<const ::uint2*>(pairs);
        __half2* halves = reinterpret_cast<__half2*>(&packed);
        const float2 low = __half22float2(halves[0]);
        const float2 high = __half22float2(halves[1]);
        halves[0] = __floats2half2_rn(values[0] + low.x, values[1] + low.y);
        halves[1] = __floats2half2_rn(values[2] + high.x, values[3] + high.y);
        *reinterpret_cast<::uint2*>(pairs) = packed;
      } else {
        for (int index = 0; index < 4; ++index) {
          const float start = accumulate ? AsFloat(output[target * 4 + index]) : 0.0f;
          output[target * 4 + index] = FromFloat<T>(start + values[index]);
        }
      }
    } else {
      float* row = output + target * 4;
      const float values[4] = {sum.x, sum.y, sum.z, sum.w};
      for (int index = 0; index < 4; ++index) {
        row[index] = (accumulate ? row[index] : 0.0f) + values[index];
      }
    }
  }
}

template <typename T, typename W>
__global__ void __launch_bounds__(kTileThreads) TiledForwardKernel(
    int64_t n_pre, int n_post, int tiles, bool accumulate, bool aligned,
    SpikeSlots<const T*> spikes,
    const unsigned int* queue, const unsigned int* sample_starts,
    const unsigned int* segment_splits, const unsigned int* segment_starts,
    const unsigned int* segment_tiles, const unsigned long long* segment_masks,
    const uint32* row_splits, const W* weights, const uint32* post_ids,
    const uint8* synapse_types, const uint32* edge_ids, const float* basis, T* currents) {
  extern __shared__ ::float4 sums[];   // [kTileTargets]
  __shared__ uint32 item_start[kTileThreads];
  __shared__ uint32 item_offset[kTileThreads + 1];   // exclusive prefix of lengths
  __shared__ float item_spike[kTileThreads];
  __shared__ int items;
  using Scan = cub::BlockScan<uint32, kTileThreads>;
  __shared__ typename Scan::TempStorage scan_storage;
  const unsigned int tile = blockIdx.x % tiles;
  const int64_t sample = blockIdx.x / tiles;
  const int first_target = tile * kTileTargets;
  const int targets = min(kTileTargets, n_post - first_target);
  for (int target = threadIdx.x; target < targets; target += kTileThreads) {
    sums[target] = ::float4{0, 0, 0, 0};
  }
  if (threadIdx.x == 0) items = 0;
  const unsigned int begin = sample_starts[sample];
  const unsigned int end = sample_starts[sample + 1];
  const unsigned int sample_first = static_cast<unsigned int>(sample * n_pre);
  bool touched = false;
  __syncthreads();
  for (unsigned int base = begin; base < end; base += kTileThreads) {
    // One thread per active row: find the row's segment in this tile.
    const unsigned int index = base + threadIdx.x;
    int item = -1;
    uint32 length = 0;
    if (index < end) {
      const unsigned int slot = queue[index];
      const unsigned int row = slot - sample_first;
      uint32 start, stop;
      if (FindTileSegment(row, tile, segment_splits, segment_starts, segment_tiles,
                          segment_masks, row_splits, &start, &stop)) {
        item = atomicAdd(&items, 1);
        item_start[item] = start;
        item_spike[item] = AsFloat(spikes.At(sample, row));
        length = stop - start;
      }
    }
    __syncthreads();
    const int count = items;
    // Scatter each item's length to its slot, then prefix-sum in item order.
    if (item >= 0) item_offset[item] = length;
    __syncthreads();
    uint32 mine = threadIdx.x < count ? item_offset[threadIdx.x] : 0;
    uint32 total;
    Scan(scan_storage).ExclusiveSum(mine, mine, total);
    __syncthreads();
    if (threadIdx.x < count) item_offset[threadIdx.x] = mine;
    if (threadIdx.x == 0) item_offset[count] = total;
    __syncthreads();
    touched = touched || count > 0;
    // The block walks the concatenated edges of the round's segments.
    for (uint32 edge = threadIdx.x; edge < total; edge += kTileThreads) {
      int low = 0;
      int high = count;   // owner: last item whose offset <= edge
      while (high - low > 1) {
        const int middle = (low + high) >> 1;
        if (item_offset[middle] <= edge) low = middle; else high = middle;
      }
      const uint32 csr = item_start[low] + (edge - item_offset[low]);
      const float weighted = item_spike[low] * AsFloat(__ldcs(&weights[V1_EDGE_INDEX(csr)]));
      const ::float4 bv = LoadBasis4(basis, __ldcs(&synapse_types[csr]));
      float* sum = reinterpret_cast<float*>(sums + (__ldcs(&post_ids[csr]) - first_target));
      atomicAdd(sum, weighted * bv.x);
      atomicAdd(sum + 1, weighted * bv.y);
      atomicAdd(sum + 2, weighted * bv.z);
      atomicAdd(sum + 3, weighted * bv.w);
    }
    __syncthreads();
    if (threadIdx.x == 0) items = 0;
    __syncthreads();
  }
  // With `initial` in place and nothing added, the tile is already right.
  if (accumulate && !touched) return;
  WriteTileSums<T, kTileThreads>(sums, targets, accumulate, aligned,
                                 currents + (sample * n_post + first_target) * 4);
}

// Check a tile kernel's actual static allocation as well as the device's
// opt-in dynamic allowance before selecting it, once per operator (`checked`,
// `available` live in the operator's ForwardTileCache, under its lock).
// Unsupported resources fall back; API failures unrelated to resource limits
// remain visible.
inline Status ConfigureTileKernel(const void* kernel, int threads, size_t shared,
                                  bool* checked, bool* available) {
  if (!*checked) {
    int ordinal;
    cudaFuncAttributes attributes;
    int shared_limit;
    cudaError_t status = cudaGetDevice(&ordinal);
    if (status == cudaSuccess) status = cudaFuncGetAttributes(&attributes, kernel);
    if (status == cudaSuccess)
      status = cudaDeviceGetAttribute(&shared_limit,
                                      cudaDevAttrMaxSharedMemoryPerBlockOptin, ordinal);
    if (status != cudaSuccess)
      return errors::Internal("checking tiled forward resources failed: ",
                              cudaGetErrorString(status));
    *available = threads <= attributes.maxThreadsPerBlock &&
                 attributes.sharedSizeBytes + shared <= static_cast<size_t>(shared_limit);
    if (*available) {
      status = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    static_cast<int>(shared));
      if (status == cudaErrorInvalidValue || status == cudaErrorNotSupported) {
        // GpuLaunchKernel checks the thread's last CUDA error. Clear only the
        // configuration error being handled; report any different pending error.
        const cudaError_t pending = cudaGetLastError();
        if (pending != cudaSuccess && pending != status)
          return errors::Internal("configuring tiled forward left another CUDA error: ",
                                  cudaGetErrorString(pending));
        *available = false;
      } else if (status != cudaSuccess) {
        return errors::Internal("configuring the tiled forward failed: ",
                                cudaGetErrorString(status));
      }
    }
    *checked = true;
  }
  return OkStatus();
}

template <typename T, typename W>
Status TiledForwardAvailable(ForwardTileCache* cache, bool* available) {
  mutex_lock lock(cache->mu);
  TF_RETURN_IF_ERROR(ConfigureTileKernel(
      reinterpret_cast<const void*>(TiledForwardKernel<T, W>), kTileThreads,
      kTileTargets * sizeof(::float4), &cache->kernel_checked, &cache->kernel_available));
  *available = cache->kernel_available;
  return OkStatus();
}

// ---------------------------------------------------------------------------
// Tile sweep (four basis columns): the LGN forward (rows that repeat targets,
// one slot), alone or fused with the recurrent forward over three delay slots
// with carried queue records (V1CsrForwardFused). SweepTargets is its gate.
//
// One block per (tile, sample), numbered tile-major so that the blocks resident
// at once share a few tiles across samples: a row segment active in several
// samples is then read from L2 after its first sample. A batch too small to
// fill the device with wide tiles takes narrow ones (SweepTargets). As in
// TiledForwardKernel
// the block accumulates its tile in FP32 shared memory and writes it once,
// rounded once. Differences:
// - The active rows come straight from the slots' queue records (the newest
//   built here, the others carried), in the merged queue's order (slot, then
//   column), so no merged queue, merge kernel or sample-start pass is needed.
// - Target-repeating rows (the LGN) are walked in block rounds, the recurrent
//   rows in independent warp batches (SweepBlockRounds, SweepWarpBatches).
// - kAggregate (rows that repeat a target): lanes of one step hold consecutive
//   edges, so a run of edges to one target sits in consecutive lanes; the runs
//   are found from neighbour comparisons, summed in registers (a segmented
//   suffix scan stopped at the longest run) and added by their lowest lane.
// - Each target's four sums are updated by one 16-byte shared compare-and-swap
//   (SharedAddQuad): shared FP32 atomicAdd is itself a CAS loop per float here.
// The order of the FP32 additions follows the shared atomics, as before.
// ---------------------------------------------------------------------------

#ifndef V1_SWEEP_TARGETS
#define V1_SWEEP_TARGETS 5120
#endif
#ifndef V1_SWEEP_THREADS
#define V1_SWEEP_THREADS 1024
#endif
// Edges per lane per walk step: the LGN's long rows lose to contention with
// longer steps; the recurrent warp batches gain from loads in flight.
#ifndef V1_SWEEP_LGN_EDGES
#define V1_SWEEP_LGN_EDGES 1
#endif
#ifndef V1_SWEEP_BATCHES_PER_WARP
#define V1_SWEEP_BATCHES_PER_WARP 8
#endif
#ifndef V1_SWEEP_REC_EDGES
#define V1_SWEEP_REC_EDGES 4
#endif
// The narrow tile, for grids of wide tiles smaller than one block per SM:
// at batch 1 on the 203,816-neuron network (40 wide tiles) it is 31 % faster
// at 4/20 Hz, but already at 320 wide blocks (batch 8) 54 % slower, its blocks
// each looking up every active row of their sample for a quarter of the edges.
#ifndef V1_SWEEP_SMALL_TARGETS
#define V1_SWEEP_SMALL_TARGETS 1280
#endif
constexpr int kSweepTargets = V1_SWEEP_TARGETS;
constexpr int kSweepSmallTargets = V1_SWEEP_SMALL_TARGETS;
constexpr int kSweepThreads = V1_SWEEP_THREADS;
constexpr int kSweepWarps = kSweepThreads / 32;
// The aggregation key packs (item, target in tile) into 32 bits.
static_assert(kSweepTargets <= 8192 && kSweepSmallTargets <= kSweepTargets &&
                  kSweepThreads <= 1024 && kSweepThreads % 32 == 0,
              "tile sweep shape");
// A tile shape's slot in ForwardTileCache::sweep.
template <int kTargets>
constexpr int kSweepShape = kTargets == kSweepTargets ? 0 : 1;
// The recurrent history's three delay slots (the most a sweep reads).
constexpr int kSweepSlots = 3;

// Adds `value` to the four FP32 sums of one shared target. Shared FP32
// atomicAdd is a compare-and-swap loop per float on this hardware; sm_90+ has a
// native 16-byte shared CAS, so one loop updates the target. Each component is
// still `old + value`, rounded (and flushed) as the scalar atomics do.
__device__ __forceinline__ void SharedAddQuad(::float4* address, const float (&value)[4]) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  using Word = unsigned __int128;
  Word* word = reinterpret_cast<Word*>(address);
  ::float4 old = *address;
  for (;;) {
    const ::float4 sum = {old.x + value[0], old.y + value[1], old.z + value[2],
                          old.w + value[3]};
    const Word seen = atomicCAS(word, *reinterpret_cast<const Word*>(&old),
                                *reinterpret_cast<const Word*>(&sum));
    if (seen == *reinterpret_cast<const Word*>(&old)) return;
    old = *reinterpret_cast<const ::float4*>(&seen);
  }
#else
  float* sum = reinterpret_cast<float*>(address);
  for (int index = 0; index < 4; ++index) atomicAdd(sum + index, value[index]);
#endif
}

// SharedAddQuad of weighted * basis, each component one fused multiply-add on
// the shared sum: one rounding per contribution instead of the product's and the
// sum's.
__device__ __forceinline__ void SharedFmaQuad(::float4* address, float weighted,
                                              const ::float4& basis) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  using Word = unsigned __int128;
  Word* word = reinterpret_cast<Word*>(address);
  ::float4 old = *address;
  for (;;) {
    const ::float4 sum = {fmaf(weighted, basis.x, old.x), fmaf(weighted, basis.y, old.y),
                          fmaf(weighted, basis.z, old.z), fmaf(weighted, basis.w, old.w)};
    const Word seen = atomicCAS(word, *reinterpret_cast<const Word*>(&old),
                                *reinterpret_cast<const Word*>(&sum));
    if (seen == *reinterpret_cast<const Word*>(&old)) return;
    old = *reinterpret_cast<const ::float4*>(&seen);
  }
#else
  const float value[4] = {weighted * basis.x, weighted * basis.y, weighted * basis.z,
                          weighted * basis.w};
  SharedAddQuad(address, value);
#endif
}

// One connectivity as the sweep reads it: the CSR, its weights and its tile
// segment table.
template <typename W>
struct SweepCsr {
  const unsigned int* segment_splits;
  const unsigned int* segment_starts;
  const unsigned int* segment_tiles;
  const unsigned long long* segment_masks;   // null above 64 tiles
  const uint32* row_splits;
  const W* weights;
  const uint32* post_ids;
  const uint8* synapse_types;
};

// The active rows of a [batch, slots * width] presynaptic matrix: slot d's
// queue record (QueueRecordWords) lists its active flat indices
// sample * width + column, then their count and the sample starts.
template <typename T>
struct SweepRows {
  SpikeSlots<const T*> spikes;
  QueueRecords records;
  int slots;
  unsigned int width;
  int64_t capacity;   // batch * width
};

// A block's view of its sample's active rows, in the merged queue's order
// (slot, then column): index i < count of them is entry first[d] + i - prefix[d]
// of record d.
template <typename T>
struct SampleRows {
  unsigned int first[kSweepSlots];
  unsigned int prefix[kSweepSlots + 1];

  __device__ __forceinline__ SampleRows(const SweepRows<T>& rows, int64_t sample) {
    prefix[0] = 0;
#pragma unroll
    for (int d = 0; d < kSweepSlots; ++d) {
      const unsigned int* starts =
          d < rows.slots ? rows.records.record[d] + rows.capacity + 1 : nullptr;
      first[d] = d < rows.slots ? starts[sample] : 0;
      prefix[d + 1] = prefix[d] + (d < rows.slots ? starts[sample + 1] - first[d] : 0);
    }
  }

  __device__ __forceinline__ unsigned int count() const { return prefix[kSweepSlots]; }

  // Row `index`'s segment in `tile`, if it has one, its spike and (optionally)
  // the presynaptic row.
  template <typename W>
  __device__ __forceinline__ bool Find(const SweepRows<T>& rows, const SweepCsr<W>& csr,
                                       int64_t sample, unsigned int index, unsigned int tile,
                                       uint32* start, uint32* stop, float* spike,
                                       unsigned int* row_of = nullptr) const {
    const unsigned int* entries = rows.records.record[0];
    const T* values = rows.spikes.slot[0];
    unsigned int slot = 0, at = first[0] + index;
#pragma unroll
    for (int d = 1; d < kSweepSlots; ++d) {
      if (d < rows.slots && index >= prefix[d]) {
        slot = d;
        entries = rows.records.record[d];
        values = rows.spikes.slot[d];
        at = first[d] + index - prefix[d];
      }
    }
    const unsigned int entry = entries[at];
    const unsigned int row =
        slot * rows.width + (entry - static_cast<unsigned int>(sample * rows.width));
    const bool found = FindTileSegment(row, tile, csr.segment_splits, csr.segment_starts,
                                       csr.segment_tiles, csr.segment_masks, csr.row_splits,
                                       start, stop);
    if (found) *spike = AsFloat(values[entry]);
    if (row_of != nullptr) *row_of = row;
    return found;
  }
};

// One step group of an edge walk: each lane's kEdges edges (csr, spike, valid)
// are loaded, weighted and added to the shared tile. With kAggregate a step's
// run of edges to one target (equal key | post in consecutive lanes) is summed
// first and its lowest lane adds it. With kFma (rows that never repeat a
// target) each edge's weighted basis is fused into the sums (SharedFmaQuad).
template <int kEdges, bool kAggregate, typename W, bool kFma = false>
__device__ __forceinline__ void SweepSteps(const bool (&valid)[kEdges],
                                           const uint32 (&csr)[kEdges],
                                           const float (&spike)[kEdges],
                                           const unsigned int (&keys)[kEdges], int lane,
                                           int first_target, const SweepCsr<W>& graph,
                                           const float* basis, ::float4* sums) {
  float weight[kEdges];
  uint32 post[kEdges];
  int type[kEdges];
#pragma unroll
  for (int step = 0; step < kEdges; ++step) {
    if (valid[step]) {
      weight[step] = AsFloat(__ldg(&graph.weights[V1_EDGE_INDEX(csr[step])]));
      post[step] = __ldg(&graph.post_ids[csr[step]]) - first_target;
      type[step] = __ldg(&graph.synapse_types[csr[step]]);
    } else {
      weight[step] = 0.0f;
      post[step] = 0;
      type[step] = 0;
    }
  }
#pragma unroll
  for (int step = 0; step < kEdges; ++step) {
    if constexpr (!kAggregate) {
      if (!valid[step]) continue;
    }
    const float weighted = valid[step] ? spike[step] * weight[step] : 0.0f;
    const ::float4 bv = LoadBasis4(basis, type[step]);
    if constexpr (kFma && !kAggregate) {
      SharedFmaQuad(sums + post[step], weighted, bv);
      continue;
    }
    float value[4] = {weighted * bv.x, weighted * bv.y, weighted * bv.z, weighted * bv.w};
    bool leader = valid[step];
    if constexpr (kAggregate) {
      // Invalid lanes share a key no edge has; the `valid` test drops their run.
      const unsigned int key = valid[step] ? keys[step] | post[step] : 0xffffffffu;
      // Every lane must reach the shuffle, so it is not short-circuited.
      const unsigned int below = __shfl_up_sync(0xffffffffu, key, 1);
      const bool head = lane == 0 || key != below;
      const unsigned int heads = __ballot_sync(0xffffffffu, head);
      if (heads != 0xffffffffu) {
        const unsigned int upto = (2u << lane) - 1;   // lanes <= this one
        const int run_start = 31 - __clz(heads & upto);
        const unsigned int later = heads & ~upto;
        const int next = later ? __ffs(later) - 1 : 32;
        const unsigned int group =
            (next == 32 ? 0xffffffffu : (1u << next) - 1) & ~((1u << run_start) - 1);
        // Rounds at offsets beyond the longest run add nothing, so stop there:
        // the sums are those of RunSuffixSums.
        const int longest = __reduce_max_sync(0xffffffffu, head ? next - lane : 0);
#pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1) {
          if (offset >= longest) break;
          const int source = lane + offset;
          const bool take = source < 32 && ((group >> source) & 1u);
#pragma unroll
          for (int index = 0; index < 4; ++index) {
            const float other = __shfl_down_sync(0xffffffffu, value[index], offset);
            if (take) value[index] += other;
          }
        }
      }
      leader = valid[step] && head;
    }
    if (leader) SharedAddQuad(sums + post[step], value);
  }
}

// Shared item lists of the walks: the block's round (block rounds) or one per
// warp (warp batches), each [start, offset (exclusive prefix of lengths), spike].
struct SweepItems {
  uint32 start[kSweepThreads];
  uint32 offset[kSweepThreads + kSweepWarps];
  float spike[kSweepThreads];
};

// Block rounds (rows that repeat targets: the LGN, ~5,500-edge rows, a few
// hundred per sample). One thread per active row per round finds its segment;
// one block scan of packed (count, length) pairs compacts them in row order;
// each warp then walks chunks of 32 * kEdges consecutive edges of the round's
// concatenated segments (first owner binary-searched, then advanced linearly).
// The whole block shares the walk, which balances the long, uneven segments.
template <int kEdges, bool kAggregate, typename T, typename W, typename Scan>
__device__ __forceinline__ bool SweepBlockRounds(
    const SweepRows<T>& rows, const SweepCsr<W>& graph, int64_t sample, unsigned int tile,
    int first_target, const float* basis, ::float4* sums, SweepItems& items,
    typename Scan::TempStorage& scan_storage) {
  const SampleRows<T> sample_rows(rows, sample);
  const int lane = threadIdx.x & 31;
  bool touched = false;
  for (unsigned int base = 0; base < sample_rows.count(); base += kSweepThreads) {
    const unsigned int index = base + threadIdx.x;
    uint32 start = 0, stop = 0;
    float spike = 0.0f;
    const bool found = index < sample_rows.count() &&
                       sample_rows.Find(rows, graph, sample, index, tile, &start, &stop, &spike);
    unsigned long long packed =
        found ? (1ull << 32) | static_cast<unsigned long long>(stop - start) : 0ull;
    unsigned long long round;
    Scan(scan_storage).ExclusiveSum(packed, packed, round);
    const int count = static_cast<int>(round >> 32);
    const uint32 total = static_cast<uint32>(round);
    if (found) {
      const int item = static_cast<int>(packed >> 32);
      items.start[item] = start;
      items.offset[item] = static_cast<uint32>(packed);
      items.spike[item] = spike;
    }
    if (threadIdx.x == 0) items.offset[count] = total;
    __syncthreads();
    touched = touched || count > 0;
    constexpr uint32 kChunk = 32 * kEdges;
    for (uint32 chunk = (threadIdx.x >> 5) * kChunk; chunk < total;
         chunk += kSweepWarps * kChunk) {
      int owner = 0;
      int high = count;
      while (high - owner > 1) {
        const int middle = (owner + high) >> 1;
        if (items.offset[middle] <= chunk) owner = middle; else high = middle;
      }
      bool valid[kEdges];
      uint32 csr[kEdges];
      float spikes[kEdges];
      unsigned int keys[kEdges];
#pragma unroll
      for (int step = 0; step < kEdges; ++step) {
        const uint32 edge = chunk + step * 32 + lane;
        valid[step] = edge < total;
        if (valid[step]) {
          while (items.offset[owner + 1] <= edge) ++owner;
          csr[step] = items.start[owner] + (edge - items.offset[owner]);
          spikes[step] = items.spike[owner];
        }
        keys[step] = static_cast<unsigned int>(owner) << 13;
      }
      SweepSteps<kEdges, kAggregate>(valid, csr, spikes, keys, lane, first_target, graph,
                                     basis, sums);
    }
    __syncthreads();
  }
  return touched;
}

// Warp batches (rows that never repeat a target: the recurrent history,
// ~140-edge rows, thousands per sample, ~11 edges per tile segment). Each warp
// claims a batch of active rows at a time, finds their segments, compacts them
// into its own item list and walks their concatenated edges alone, so no block
// barrier separates the lookups of one batch from the walk of another: the
// warps' dependent loads overlap. A batch is 4 to 32 rows, sized for about
// eight batches per warp: the walk of one batch is serial in its warp, so fewer
// rows per sample need smaller batches to keep every warp busy (32-row batches
// measured 8 % slower at 0.7 Hz, 4-row ones 23 % slower at 4 Hz). Returns
// whether this warp added anything. kFma: see SweepSteps.
template <int kEdges, typename T, typename W, bool kFma = false>
__device__ __forceinline__ bool SweepWarpBatches(
    const SweepRows<T>& rows, const SweepCsr<W>& graph, int64_t sample, unsigned int tile,
    int first_target, const float* basis, ::float4* sums, SweepItems& items,
    unsigned int* next_batch) {
  const SampleRows<T> sample_rows(rows, sample);
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  uint32* starts = items.start + warp * 32;
  uint32* offsets = items.offset + warp * 33;
  float* spikes_of = items.spike + warp * 32;
  const unsigned int share = sample_rows.count() / (kSweepWarps * V1_SWEEP_BATCHES_PER_WARP);
  const unsigned int batch_rows = share >= 32 ? 32 : (share >= 16 ? 16 : (share >= 8 ? 8 : 4));
  bool touched = false;
  for (;;) {
    unsigned int base = 0;
    if (lane == 0) base = atomicAdd(next_batch, batch_rows);
    base = __shfl_sync(0xffffffffu, base, 0);
    if (base >= sample_rows.count()) break;
    const unsigned int index = base + lane;
    uint32 start = 0, stop = 0;
    float spike = 0.0f;
    const bool found = lane < batch_rows && index < sample_rows.count() &&
                       sample_rows.Find(rows, graph, sample, index, tile, &start, &stop, &spike);
    // Exclusive prefix of the lengths in lane order (zero for rows without a
    // segment here), and each found row's rank among the found ones.
    const uint32 length = found ? stop - start : 0;
    uint32 prefix = length;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
      const uint32 other = __shfl_up_sync(0xffffffffu, prefix, offset);
      if (lane >= offset) prefix += other;
    }
    const uint32 total = __shfl_sync(0xffffffffu, prefix, 31);
    const unsigned int found_lanes = __ballot_sync(0xffffffffu, found);
    if (found) {
      const int item = __popc(found_lanes & ((1u << lane) - 1));
      starts[item] = start;
      offsets[item] = prefix - length;
      spikes_of[item] = spike;
    }
    if (lane == 0) offsets[__popc(found_lanes)] = total;
    __syncwarp();
    touched = touched || total > 0;
    int owner = 0;
    for (uint32 chunk = 0; chunk < total; chunk += 32 * kEdges) {
      bool valid[kEdges];
      uint32 csr[kEdges];
      float spikes[kEdges];
      unsigned int keys[kEdges];
#pragma unroll
      for (int step = 0; step < kEdges; ++step) {
        const uint32 edge = chunk + step * 32 + lane;
        valid[step] = edge < total;
        if (valid[step]) {
          while (offsets[owner + 1] <= edge) ++owner;
          csr[step] = starts[owner] + (edge - offsets[owner]);
          spikes[step] = spikes_of[owner];
        }
        keys[step] = 0;
      }
      SweepSteps<kEdges, false, W, kFma>(valid, csr, spikes, keys, lane, first_target, graph,
                                         basis, sums);
    }
    __syncwarp();   // the next batch overwrites this warp's items
  }
  return touched;
}

// ---------------------------------------------------------------------------
// Scalar external input (the LGN under DIRECT_CSR): every postsynaptic neuron
// receives the input through ONE synapse type (203,783 posts, 203,783 (post,
// type) pairs on the 203,816-neuron network), so its four currents are
// basis[type(post)] * s(post) with the scalar s = sum spike * weight over its
// incoming edges. The sweep accumulates s in FP32 shared memory (one value per
// target instead of four) and expands it once per target.
//
// A row's edges to one target (on average 1.42, up to 1,260) are consecutive in
// CSR order. ScalarExternalTable cuts each row into runs of at most kRunEdges
// edges to one target and stores a run as one word:
//   bits 0-12  post mod 8192 (a tile is at most 8,192 wide, so the block
//              recovers the post from its first target),
//   bits 13-27 offset of the run's first edge within the row (rows <= 2^15),
//   bits 28-31 edges - 1.
// A lane takes a run, sums its weights in FP32 registers in edge order and adds
// spike * sum to the target with one shared atomic. There is no per-edge synapse
// type, basis product, run detection or 16-byte compare-and-swap; the weights are
// read at every call, so trained weights need no cache. The tile segments are
// built over the runs, so a block finds its run range as before.
// ---------------------------------------------------------------------------

#ifndef V1_SWEEP_RUN_EDGES
#define V1_SWEEP_RUN_EDGES 4
#endif
#ifndef V1_SWEEP_SCALAR_RUNS
#define V1_SWEEP_SCALAR_RUNS 4
#endif
// Pairs per lane per walk step in the coalesced (pair) mode: 8 measured 836 /
// 338 us against 846 / 338 for 4 at 4/20 and 0.65/5.8 Hz.
#ifndef V1_SWEEP_PAIR_RUNS
#define V1_SWEEP_PAIR_RUNS 8
#endif
constexpr int kRunEdges = V1_SWEEP_RUN_EDGES;
constexpr int kRunPostBits = 13;
constexpr int kRunOffsetBits = 15;
static_assert(kRunEdges >= 1 && kRunEdges <= (1 << (32 - kRunPostBits - kRunOffsetBits)) &&
                  kSweepTargets <= (1 << kRunPostBits),
              "scalar run word layout");

// The scalar external connectivity as the sweep reads it: `runs` is the run CSR
// (row_splits = run splits, post_ids = run words) with its tile segments.
template <typename W>
struct ScalarCsr {
  SweepCsr<W> runs;
  const unsigned int* edge_splits;   // the original row splits (edge offsets)
  const unsigned char* post_types;   // [n_post] the one synapse type per target
  // Coalesced (pair) mode: one entry per distinct (row, target) pair instead of
  // per run, its post mod 8192 and its weight sum (V1ExternalPairWeights).
  const unsigned short* pair_posts;
  const float* pair_weights;
};

// Block rounds over the runs of the sample's active rows (as SweepBlockRounds),
// accumulating the scalars of tile `tile` into `scalars`. Ends with a barrier.
template <int kRuns, typename T, typename W, typename Scan, bool kPairs = false>
__device__ __forceinline__ bool SweepScalarRounds(
    const SweepRows<T>& rows, const ScalarCsr<W>& graph, int64_t sample, unsigned int tile,
    int first_target, float* scalars, SweepItems& items,
    typename Scan::TempStorage& scan_storage) {
  __shared__ unsigned int bases[kSweepThreads];   // each item's first edge
  const SampleRows<T> sample_rows(rows, sample);
  const int lane = threadIdx.x & 31;
  const unsigned int* words = reinterpret_cast<const unsigned int*>(graph.runs.post_ids);
  const unsigned int low_target = static_cast<unsigned int>(first_target);
  bool touched = false;
  for (unsigned int base = 0; base < sample_rows.count(); base += kSweepThreads) {
    const unsigned int index = base + threadIdx.x;
    uint32 start = 0, stop = 0;
    float spike = 0.0f;
    unsigned int row = 0;
    const bool found =
        index < sample_rows.count() &&
        sample_rows.Find(rows, graph.runs, sample, index, tile, &start, &stop, &spike, &row);
    unsigned long long packed =
        found ? (1ull << 32) | static_cast<unsigned long long>(stop - start) : 0ull;
    unsigned long long round;
    Scan(scan_storage).ExclusiveSum(packed, packed, round);
    const int count = static_cast<int>(round >> 32);
    const unsigned int total = static_cast<unsigned int>(round);
    if (found) {
      const int item = static_cast<int>(packed >> 32);
      items.start[item] = start;
      items.offset[item] = static_cast<uint32>(packed);
      items.spike[item] = spike;
      if constexpr (!kPairs) bases[item] = graph.edge_splits[row];
    }
    if (threadIdx.x == 0) items.offset[count] = total;
    __syncthreads();
    touched = touched || count > 0;
    constexpr unsigned int kChunk = 32 * kRuns;
    for (unsigned int chunk = (threadIdx.x >> 5) * kChunk; chunk < total;
         chunk += kSweepWarps * kChunk) {
      int owner = 0;
      int high = count;
      while (high - owner > 1) {
        const int middle = (owner + high) >> 1;
        if (items.offset[middle] <= chunk) owner = middle; else high = middle;
      }
      bool valid[kRuns];
      unsigned int word[kRuns], first_edge[kRuns];
      float spikes[kRuns], pair_weight[kRuns];
#pragma unroll
      for (int step = 0; step < kRuns; ++step) {
        const unsigned int run = chunk + step * 32 + lane;
        valid[step] = run < total;
        word[step] = 0;
        if (valid[step]) {
          while (items.offset[owner + 1] <= run) ++owner;
          const unsigned int at = items.start[owner] + (run - items.offset[owner]);
          if constexpr (kPairs) {
            word[step] = __ldg(&graph.pair_posts[at]);
            pair_weight[step] = __ldg(&graph.pair_weights[at]);
          } else {
            word[step] = __ldg(&words[at]);
            first_edge[step] = bases[owner];
          }
          spikes[step] = items.spike[owner];
        }
      }
#pragma unroll
      for (int step = 0; step < kRuns; ++step) {
        if (!valid[step]) continue;
        if constexpr (kPairs) {
          const unsigned int target = (word[step] - low_target) & ((1u << kRunPostBits) - 1);
          atomicAdd(scalars + target, spikes[step] * pair_weight[step]);
          continue;
        }
        const unsigned int target =
            ((word[step] & ((1u << kRunPostBits) - 1)) - low_target) & ((1u << kRunPostBits) - 1);
        const W* weight = graph.runs.weights + first_edge[step] +
                          ((word[step] >> kRunPostBits) & ((1u << kRunOffsetBits) - 1));
        const unsigned int edges = (word[step] >> (kRunPostBits + kRunOffsetBits)) + 1;
        float sum = AsFloat(__ldg(weight));
#pragma unroll
        for (int edge = 1; edge < kRunEdges; ++edge) {
          if (edge < edges) sum += AsFloat(__ldg(weight + edge));
        }
        atomicAdd(scalars + target, spikes[step] * sum);
      }
    }
    __syncthreads();
  }
  return touched;
}

// What one sweep accumulates: the target-repeating external input (block
// rounds), or that and the recurrent history (warp batches) into one tile. The
// recurrent history alone keeps TiledForwardKernel: its sweep was 20 % faster
// at 4 Hz, but its reordered FP32 sums moved the mean errors by ~1e-9 relative
// outside the baseline's run-to-run envelope.
enum SweepMode { kSweepExternal, kSweepFused, kSweepFusedScalar, kSweepFusedPairs };

// One block per (tile of kTargets, sample), numbered tile-major so that the
// blocks resident at once share a few tiles across samples. (Repeating the
// batch-64 order slice by slice above 64 samples measured 1-2 % slower.) The
// block zeroes its FP32 tile, accumulates its rows and writes the tile once,
// rounded once. kSweepFused adds both sources before that single rounding,
// where the two-operator chain rounds the recurrent currents to T and the
// external ones onto them. kSweepFusedScalar does the same with the scalar
// external input (ScalarCsr): it first accumulates the external scalars in the
// front of the tile's shared memory, expands each target's scalar into its four
// sums (scalar * basis[type]; exactly zero without input), then adds the
// recurrent history onto them.
template <typename T, typename W, SweepMode kMode, int kTargets>
__global__ void __launch_bounds__(kSweepThreads) CsrTileSweepKernel(
    int n_post, int batch, bool accumulate, bool aligned,
    SweepRows<T> recurrent_rows, SweepCsr<W> recurrent_graph, SweepRows<T> external_rows,
    SweepCsr<W> external_graph, const float* basis, T* currents,
    ScalarCsr<W> scalar_graph) {
  extern __shared__ ::float4 sums[];   // [kTargets]
  __shared__ SweepItems items;
  // Warp scans: their storage is one word per warp, where raking would need
  // one per thread and leave too little room next to a 5,120-target tile.
  using Scan = cub::BlockScan<unsigned long long, kSweepThreads, cub::BLOCK_SCAN_WARP_SCANS>;
  __shared__ typename Scan::TempStorage scan_storage;
  __shared__ unsigned int next_batch;
  const unsigned int tile = blockIdx.x / batch;
  const int64_t sample = blockIdx.x % batch;
  const int first_target = tile * kTargets;
  const int targets = min(kTargets, n_post - first_target);
  bool touched = false;
  if constexpr (kMode == kSweepFusedScalar || kMode == kSweepFusedPairs) {
    float* scalars = reinterpret_cast<float*>(sums);
    for (int target = threadIdx.x; target < targets; target += kSweepThreads) {
      scalars[target] = 0.0f;
    }
    if (threadIdx.x == 0) next_batch = 0;
    __syncthreads();
    touched = SweepScalarRounds<kMode == kSweepFusedPairs ? V1_SWEEP_PAIR_RUNS
                                                          : V1_SWEEP_SCALAR_RUNS,
                                T, W, Scan, kMode == kSweepFusedPairs>(
        external_rows, scalar_graph, sample, tile, first_target, scalars, items, scan_storage);
    constexpr int kHeld = (kTargets + kSweepThreads - 1) / kSweepThreads;
    float held[kHeld];
#pragma unroll
    for (int index = 0; index < kHeld; ++index) {
      const int target = threadIdx.x + index * kSweepThreads;
      held[index] = target < targets ? scalars[target] : 0.0f;
    }
    __syncthreads();   // every scalar is read before the sums overwrite them
#pragma unroll
    for (int index = 0; index < kHeld; ++index) {
      const int target = threadIdx.x + index * kSweepThreads;
      if (target >= targets) continue;
      const float scalar = held[index];
      ::float4 sum = {0, 0, 0, 0};
      if (scalar != 0.0f) {
        const ::float4 bv = LoadBasis4(basis, scalar_graph.post_types[first_target + target]);
        sum = {scalar * bv.x, scalar * bv.y, scalar * bv.z, scalar * bv.w};
      }
      sums[target] = sum;
    }
    __syncthreads();
    touched |= SweepWarpBatches<V1_SWEEP_REC_EDGES, T, W, true>(
        recurrent_rows, recurrent_graph, sample, tile, first_target, basis, sums, items,
        &next_batch);
  } else {
    for (int target = threadIdx.x; target < targets; target += kSweepThreads) {
      sums[target] = ::float4{0, 0, 0, 0};
    }
    if (threadIdx.x == 0) next_batch = 0;
    __syncthreads();
    if constexpr (kMode == kSweepFused) {
      touched = SweepWarpBatches<V1_SWEEP_REC_EDGES>(recurrent_rows, recurrent_graph, sample,
                                                     tile, first_target, basis, sums, items,
                                                     &next_batch);
      __syncthreads();   // the walks share `items`
    }
    touched |= SweepBlockRounds<V1_SWEEP_LGN_EDGES, true, T, W, Scan>(
        external_rows, external_graph, sample, tile, first_target, basis, sums, items,
        scan_storage);
  }
  // Also completes every warp's atomics. With `initial` in place and nothing
  // added, the tile is already right.
  if (!__syncthreads_or(touched) && accumulate) return;
  WriteTileSums<T, kSweepThreads>(sums, targets, accumulate, aligned,
                                  currents + (sample * n_post + first_target) * 4);
}

// The one gate of the sweep, which returns its tile width (0: no sweep) for
// `spikes`: the LGN's single slot (kSweepExternal; V1CsrForward) or the
// recurrent history's three slots with slots 1 and 2's records carried and
// slot 0's requested (kSweepFused, with the LGN in FusedForward). Four basis
// columns are checked by the caller at compile time. Every batch and both
// precisions are served: in FP16 the single rounding is far more accurate than
// the frozen paths' atomics; in FP32 the sums came out 3-7 % farther from fp64
// on average than theirs, a cost the production decision accepted
// (tracks/prod/NOTES.md). The tile width is a pure performance choice: the
// narrow tile serves grids of at most one wide-tile block per two SMs
// (`multiprocessors`), its blocks each looking up every active row of their
// sample for a quarter of the edges, which repays only there. Everything else
// takes the frozen paths, including a matrix too large for 32-bit queue
// indices, which LaunchForward then refuses.
template <typename T>
int SweepTargets(SweepMode mode, const SpikeMatrix<const T*>& spikes, int n_post,
                 const QueueRecords& carried, const unsigned int* newest,
                 int multiprocessors) {
  if (n_post <= 0 || spikes.NumElements() == 0 || spikes.NumElements() > kQueueMaxSlots) {
    return 0;
  }
  const int count = spikes.slots.count;
  const bool served = mode == kSweepExternal
                          ? count == 1
                          : count == kSweepSlots && newest != nullptr &&
                                carried.record[1] != nullptr && carried.record[2] != nullptr;
  if (!served) return 0;
  const int64_t wide_blocks = spikes.batch * ((n_post + kSweepTargets - 1) / kSweepTargets);
  if (2 * wide_blocks <= multiprocessors) return kSweepSmallTargets;
  return wide_blocks <= std::numeric_limits<int>::max() ? kSweepTargets : 0;
}

// Calls launch(std::integral_constant<int, kTargets>) for the tile width
// `targets` that SweepTargets chose.
template <typename Launch>
Status WithSweepTargets(int targets, Launch&& launch) {
  if (targets == kSweepSmallTargets) {
    return launch(std::integral_constant<int, kSweepSmallTargets>{});
  }
  return launch(std::integral_constant<int, kSweepTargets>{});
}

// Whether the device can run sweep `kernel` with kTargets of shared sums,
// checked once per operator and recorded in `cache` (its sweep flags).
template <int kTargets>
Status SweepKernelAvailable(const void* kernel, ForwardTileCache* cache, bool* available) {
  mutex_lock lock(cache->mu);
  ForwardTileCache::Sweep& sweep = cache->sweep[kSweepShape<kTargets>];
  TF_RETURN_IF_ERROR(ConfigureTileKernel(kernel, kSweepThreads, kTargets * sizeof(::float4),
                                         &sweep.checked, &sweep.available));
  *available = sweep.available;
  return OkStatus();
}

// The connectivity's sweep view, building or reusing its table under the
// operator's lock.
template <typename W, int kTargets>
Status PrepareSweepCsr(OpKernelContext* context, const Tensor& weights,
                       const Tensor& post_ids, const Tensor& synapse_types,
                       const Tensor& row_splits, int n_post, ForwardTileCache* cache,
                       SweepCsr<W>* graph) {
  mutex_lock lock(cache->mu);
  std::shared_ptr<const ForwardTileSegments>& held =
      cache->sweep[kSweepShape<kTargets>].segments;
  TF_RETURN_IF_ERROR(EnsureTileSegments<kTargets>(context, post_ids, row_splits, n_post, &held));
  const ForwardTileSegments& table = *held;
  const auto words = [](const Tensor& tensor) {
    return reinterpret_cast<const unsigned int*>(tensor.flat<int32>().data());
  };
  *graph = SweepCsr<W>{
      words(table.segment_splits), words(table.segment_starts), words(table.segment_tiles),
      table.segment_masks.NumElements() > 0
          ? reinterpret_cast<const unsigned long long*>(table.segment_masks.flat<int32>().data())
          : nullptr,
      row_splits.flat<uint32>().data(), weights.flat<W>().data(),
      post_ids.flat<uint32>().data(), synapse_types.flat<uint8>().data()};
  return OkStatus();
}

// The rows of `spikes` with every slot's queue record (missing ones built into
// `scratch`, which must outlive the launch).
template <typename T>
Status PrepareSweepRows(OpKernelContext* context, const SpikeMatrix<const T*>& spikes,
                        const QueueRecords& carried, unsigned int* newest, Tensor* scratch,
                        SweepRows<T>* rows) {
  rows->spikes = spikes.slots;
  rows->slots = spikes.slots.count;
  rows->width = spikes.slots.width;
  rows->capacity = spikes.batch * spikes.slots.width;
  return BuildMissingRecords<T>(context, spikes, carried, newest, &rows->records, scratch);
}

// The run table of a scalar external connectivity (see ScalarCsr) for one tile
// width, built once on the host from the CSR and shared like the tile segments.
// `usable` is false when the connectivity does not qualify: a target reached
// through two synapse types, a row longer than 2^15 edges or a post id not
// ascending within its row. Without DIRECT_CSR the weights would need the
// edge-id gather, so the table is never built there. A `pairs` table has one
// entry per distinct (row, target) pair (no edge limit, no row-length limit):
// `runs` holds each pair's post mod 8192 as uint16 and `pair_edges` its first
// edge (n_pairs + 1 entries), for V1ExternalPairWeights.
struct ScalarExternalTable {
  const DeviceBase* device = nullptr;
  Tensor post_ids;
  Tensor row_splits;
  Tensor synapse_types;
  int n_post = -1;
  bool pairs = false;
  bool usable = false;
  Tensor pair_edges;
  Tensor runs;
  Tensor run_splits;
  Tensor post_types;
  Tensor segment_splits;
  Tensor segment_starts;
  Tensor segment_tiles;
  Tensor segment_masks;
};

template <typename Value>
Status CopyToHost(const Tensor& tensor, cudaStream_t stream, std::vector<Value>* values) {
  values->resize(tensor.TotalBytes() / sizeof(Value));
  if (values->empty()) return OkStatus();
  cudaMemcpyAsync(values->data(), tensor.tensor_data().data(), tensor.TotalBytes(),
                  cudaMemcpyDeviceToHost, stream);
  const cudaError_t status = cudaStreamSynchronize(stream);
  return status == cudaSuccess ? OkStatus()
                               : errors::Internal("reading the CSR for the scalar table failed: ",
                                                  cudaGetErrorString(status));
}

// A device copy of `values` in a tensor of `dtype` (4-byte words or bytes).
template <typename Value>
Status CopyToDevice(OpKernelContext* context, const std::vector<Value>& values, DataType dtype,
                    Tensor* tensor) {
  const int64_t bytes = std::max<int64_t>(values.size() * sizeof(Value), 4);
  const int64_t width = dtype == DT_UINT8 ? 1 : (dtype == DT_UINT16 ? 2 : 4);
  TF_RETURN_IF_ERROR(context->allocate_temp(dtype, TensorShape({bytes / width}), tensor));
  if (!values.empty()) {
    cudaMemcpyAsync(const_cast<char*>(tensor->tensor_data().data()), values.data(),
                    values.size() * sizeof(Value), cudaMemcpyHostToDevice,
                    context->eigen_device<GPUDevice>().stream());
  }
  return OkStatus();
}

template <int kTargets>
Status BuildScalarExternal(OpKernelContext* context, const Tensor& post_ids,
                           const Tensor& row_splits, const Tensor& synapse_types, int n_post,
                           bool pairs, ScalarExternalTable* table) {
  table->device = context->device();
  table->pairs = pairs;
  table->post_ids = post_ids;
  table->row_splits = row_splits;
  table->synapse_types = synapse_types;
  table->n_post = n_post;
  table->usable = false;
  if (!V1_DIRECT_CSR || n_post <= 0) return OkStatus();
  auto stream = context->eigen_device<GPUDevice>().stream();
  std::vector<unsigned int> posts, splits;
  std::vector<unsigned char> types;
  TF_RETURN_IF_ERROR(CopyToHost(post_ids, stream, &posts));
  TF_RETURN_IF_ERROR(CopyToHost(row_splits, stream, &splits));
  TF_RETURN_IF_ERROR(CopyToHost(synapse_types, stream, &types));
  const int64_t n_pre = static_cast<int64_t>(splits.size()) - 1;
  if (n_pre < 0 || types.size() != posts.size() || splits.back() != posts.size()) {
    return OkStatus();
  }
  std::vector<unsigned char> post_types(n_post, 0), seen(n_post, 0);
  for (size_t edge = 0; edge < posts.size(); ++edge) {
    const unsigned int post = posts[edge];
    if (post >= static_cast<unsigned int>(n_post)) return OkStatus();
    if (seen[post] && post_types[post] != types[edge]) return OkStatus();
    seen[post] = 1;
    post_types[post] = types[edge];
  }
  const int tiles = (n_post + kTargets - 1) / kTargets;
  std::vector<unsigned int> runs, run_splits(n_pre + 1), segment_splits(n_pre + 1),
      segment_starts, segment_tiles, pair_edges;
  std::vector<unsigned short> pair_posts;
  const unsigned int run_limit = pairs ? 0xffffffffu : kRunEdges;
  std::vector<unsigned long long> masks(tiles <= 64 ? n_pre : 0, 0ull);
  runs.reserve(posts.size());
  for (int64_t row = 0; row < n_pre; ++row) {
    const unsigned int begin = splits[row], end = splits[row + 1];
    if (end < begin || (!pairs && end - begin > (1u << kRunOffsetBits))) return OkStatus();
    run_splits[row] = static_cast<unsigned int>(pairs ? pair_posts.size() : runs.size());
    segment_splits[row] = static_cast<unsigned int>(segment_starts.size());
    unsigned int tile = 0xffffffffu;
    for (unsigned int edge = begin; edge < end;) {
      const unsigned int post = posts[edge];
      if (edge > begin && post < posts[edge - 1]) return OkStatus();
      unsigned int stop = edge + 1;
      while (stop < end && posts[stop] == post && stop - edge < run_limit) ++stop;
      const unsigned int index = static_cast<unsigned int>(pairs ? pair_posts.size() : runs.size());
      if (post / kTargets != tile) {
        tile = post / kTargets;
        segment_starts.push_back(index);
        segment_tiles.push_back(tile);
        if (!masks.empty()) masks[row] |= 1ull << tile;
      }
      if (pairs) {
        pair_posts.push_back(static_cast<unsigned short>(post & ((1u << kRunPostBits) - 1)));
        pair_edges.push_back(edge);
      } else {
        runs.push_back((post & ((1u << kRunPostBits) - 1)) | ((edge - begin) << kRunPostBits) |
                       ((stop - edge - 1) << (kRunPostBits + kRunOffsetBits)));
      }
      edge = stop;
    }
  }
  run_splits[n_pre] = static_cast<unsigned int>(pairs ? pair_posts.size() : runs.size());
  if (pairs) {
    pair_edges.push_back(static_cast<unsigned int>(posts.size()));
    TF_RETURN_IF_ERROR(CopyToDevice(context, pair_posts, DT_UINT16, &table->runs));
    TF_RETURN_IF_ERROR(CopyToDevice(context, pair_edges, DT_INT32, &table->pair_edges));
  }
  segment_splits[n_pre] = static_cast<unsigned int>(segment_starts.size());
  if (!pairs) TF_RETURN_IF_ERROR(CopyToDevice(context, runs, DT_INT32, &table->runs));
  TF_RETURN_IF_ERROR(CopyToDevice(context, run_splits, DT_INT32, &table->run_splits));
  TF_RETURN_IF_ERROR(CopyToDevice(context, post_types, DT_UINT8, &table->post_types));
  TF_RETURN_IF_ERROR(CopyToDevice(context, segment_splits, DT_INT32, &table->segment_splits));
  TF_RETURN_IF_ERROR(CopyToDevice(context, segment_starts, DT_INT32, &table->segment_starts));
  TF_RETURN_IF_ERROR(CopyToDevice(context, segment_tiles, DT_INT32, &table->segment_tiles));
  table->segment_masks = Tensor();
  if (!masks.empty()) {
    TF_RETURN_IF_ERROR(CopyToDevice(context, masks, DT_INT32, &table->segment_masks));
  }
  const cudaError_t status = cudaStreamSynchronize(stream);
  if (status != cudaSuccess) {
    return errors::Internal("uploading the scalar table failed: ", cudaGetErrorString(status));
  }
  table->usable = true;
  return OkStatus();
}

inline bool MatchesScalarExternal(const ScalarExternalTable& table, OpKernelContext* context,
                                  const Tensor& post_ids, const Tensor& row_splits,
                                  const Tensor& synapse_types, int n_post, bool pairs) {
  const auto same = [](const Tensor& held, const Tensor& given) {
    return held.tensor_data().data() == given.tensor_data().data() &&
           held.NumElements() == given.NumElements();
  };
  return table.device == context->device() && table.n_post == n_post && table.pairs == pairs &&
         same(table.post_ids, post_ids) && same(table.row_splits, row_splits) &&
         same(table.synapse_types, synapse_types);
}

// As EnsureTileSegments: a weak registry per tile width, a strong reference held
// by each operator in `held`.
template <int kTargets>
Status EnsureScalarExternal(OpKernelContext* context, const Tensor& post_ids,
                            const Tensor& row_splits, const Tensor& synapse_types, int n_post,
                            bool pairs, std::shared_ptr<const ScalarExternalTable>* held) {
  if (*held &&
      MatchesScalarExternal(**held, context, post_ids, row_splits, synapse_types, n_post,
                            pairs)) {
    return OkStatus();
  }
  struct Registry {
    mutex mu;
    std::vector<std::weak_ptr<const ScalarExternalTable>> tables;
  };
  static Registry registry;
  mutex_lock lock(registry.mu);
  for (auto it = registry.tables.begin(); it != registry.tables.end();) {
    auto table = it->lock();
    if (!table) {
      it = registry.tables.erase(it);
    } else {
      if (MatchesScalarExternal(*table, context, post_ids, row_splits, synapse_types, n_post,
                                pairs)) {
        *held = std::move(table);
        return OkStatus();
      }
      ++it;
    }
  }
  auto table = std::make_shared<ScalarExternalTable>();
  TF_RETURN_IF_ERROR(BuildScalarExternal<kTargets>(context, post_ids, row_splits,
                                                   synapse_types, n_post, pairs, table.get()));
  registry.tables.emplace_back(table);
  *held = std::move(table);
  return OkStatus();
}

// The scalar view of a usable table, with the current weights.
template <typename W>
ScalarCsr<W> ScalarView(const ScalarExternalTable& table, const Tensor& weights,
                        const Tensor* pair_weights = nullptr) {
  const auto words = [](const Tensor& tensor) {
    return reinterpret_cast<const unsigned int*>(tensor.tensor_data().data());
  };
  ScalarCsr<W> view;
  view.runs = SweepCsr<W>{
      words(table.segment_splits), words(table.segment_starts), words(table.segment_tiles),
      table.segment_masks.NumElements() > 0
          ? reinterpret_cast<const unsigned long long*>(table.segment_masks.tensor_data().data())
          : nullptr,
      reinterpret_cast<const uint32*>(words(table.run_splits)), weights.flat<W>().data(),
      reinterpret_cast<const uint32*>(words(table.runs)), nullptr};
  view.edge_splits = words(table.row_splits);
  view.post_types = reinterpret_cast<const unsigned char*>(table.post_types.tensor_data().data());
  view.pair_posts = table.pairs
      ? reinterpret_cast<const unsigned short*>(table.runs.tensor_data().data()) : nullptr;
  view.pair_weights = pair_weights != nullptr ? pair_weights->flat<float>().data() : nullptr;
  return view;
}

// Launches CsrTileSweepKernel; *launched is false when the device cannot run
// it, and the caller falls back.
template <typename T, typename W, SweepMode kMode, int kTargets>
Status LaunchSweepKernel(OpKernelContext* context, int64_t batch, int n_post,
                         const Tensor& basis, Tensor* output, bool accumulate,
                         const SweepRows<T>& recurrent_rows, const SweepCsr<W>& recurrent_graph,
                         const SweepRows<T>& external_rows, const SweepCsr<W>& external_graph,
                         const ScalarCsr<W>& scalar_graph = ScalarCsr<W>{}) {
  const int tiles = (n_post + kTargets - 1) / kTargets;
  const bool aligned = reinterpret_cast<uintptr_t>(output->flat<T>().data()) % 8 == 0;
  return GpuLaunchKernel(CsrTileSweepKernel<T, W, kMode, kTargets>,
                         static_cast<int>(batch * tiles), kSweepThreads,
                         kTargets * sizeof(::float4),
                         context->eigen_device<GPUDevice>().stream(), n_post,
                         static_cast<int>(batch), accumulate, aligned, recurrent_rows,
                         recurrent_graph, external_rows, external_graph,
                         basis.flat<float>().data(), output->flat<T>().data(), scalar_graph);
}

// Launches CsrTileSweepKernel for the external source alone; *launched is
// false when the device cannot run it, and the caller falls back.
template <typename T, typename W, int kTargets>
Status LaunchTileSweep(OpKernelContext* context, const SpikeMatrix<const T*>& spikes,
                       const Tensor& weights, const Tensor& post_ids,
                       const Tensor& synapse_types, const Tensor& row_splits,
                       const Tensor& basis, int n_post, Tensor* output, bool zero_output,
                       ForwardTileCache* cache, const QueueRecords& carried,
                       unsigned int* newest, bool* launched) {
  TF_RETURN_IF_ERROR(SweepKernelAvailable<kTargets>(
      reinterpret_cast<const void*>(CsrTileSweepKernel<T, W, kSweepExternal, kTargets>), cache,
      launched));
  if (!*launched) return OkStatus();
  SweepCsr<W> graph;
  TF_RETURN_IF_ERROR(PrepareSweepCsr<W, kTargets>(context, weights, post_ids, synapse_types,
                                                  row_splits, n_post, cache, &graph));
  SweepRows<T> rows;
  Tensor scratch;
  TF_RETURN_IF_ERROR(PrepareSweepRows<T>(context, spikes, carried, newest, &scratch, &rows));
  return LaunchSweepKernel<T, W, kSweepExternal, kTargets>(
      context, spikes.batch, n_post, basis, output, !zero_output, SweepRows<T>{}, SweepCsr<W>{},
      rows, graph);
}

template <typename T, typename W, int kBasis>
Status LaunchForward(OpKernelContext* context, const SpikeMatrix<const T*>& spikes,
                     const Tensor& weights, const Tensor& post_ids,
                     const Tensor& synapse_types, const Tensor& row_splits,
                     const Tensor& edge_ids, const Tensor& basis, int n_post,
                     bool aggregate_runs, Tensor* output, bool zero_output,
                     ForwardTileCache* tile_segments,
                     const QueueRecords& carried = QueueRecords{},
                     unsigned int* newest_record = nullptr) {
  const int64_t batch = spikes.batch;
  const int64_t n_pre = spikes.n_pre();
  if (spikes.NumElements() > kQueueMaxSlots) {
    return errors::InvalidArgument(
        "the active-row queue indexes slots with 32-bit counts; got ", batch, " x ",
        n_pre, " slots");
  }
  const int64_t slots = spikes.NumElements();
  const int n_basis = basis.dim_size(1);
  auto device = context->eigen_device<GPUDevice>();
  if constexpr (kBasis == 4 && !V1_FORWARD_FLOAT_ACCUM) {
    const int sweep_targets =
        aggregate_runs && tile_segments != nullptr
            ? SweepTargets<T>(kSweepExternal, spikes, n_post, carried, newest_record,
                              device.getNumGpuMultiProcessors())
            : 0;
    if (sweep_targets > 0) {
      bool launched;
      TF_RETURN_IF_ERROR(WithSweepTargets(sweep_targets, [&](auto shape) {
        return LaunchTileSweep<T, W, decltype(shape)::value>(
            context, spikes, weights, post_ids, synapse_types, row_splits, basis, n_post,
            output, zero_output, tile_segments, carried, newest_record, &launched);
      }));
      if (launched) return OkStatus();
    }
  }
  // The tiled kernel writes every output element itself; everything else
  // accumulates onto zeroed (or `initial`-seeded) currents.
  const int tiles = (n_post + kTileTargets - 1) / kTileTargets;
  bool tiled = !aggregate_runs && kBasis == 4 && !V1_FORWARD_FLOAT_ACCUM &&
                     tile_segments != nullptr && slots > 0 &&
                     batch * tiles <= std::numeric_limits<int>::max();
  if (tiled) TF_RETURN_IF_ERROR((TiledForwardAvailable<T, W>(tile_segments, &tiled)));
  if (zero_output && !tiled) {
    cudaMemsetAsync(output->flat<T>().data(), 0, output->NumElements() * sizeof(T),
                    device.stream());
  }
  if (slots == 0) return OkStatus();
  // One slot per (batch, row): 78 MiB at batch 32 on the 203,816-neuron
  // network, and a temp, so it lives only for the op.
  Tensor queue_tensor;
  unsigned int* queue;
  TF_RETURN_IF_ERROR(AllocateQueueWords(context, slots + 2, &queue_tensor, &queue));
  unsigned int* queue_count = queue + slots;   // then the consumers' ticket
  cudaMemsetAsync(queue_count, 0, 2 * sizeof(unsigned int), device.stream());
  TF_RETURN_IF_ERROR(
      BuildActiveQueue<T>(context, spikes, queue, queue_count, carried, newest_record));
  if (tiled) {
    mutex_lock lock(tile_segments->mu);
    TF_RETURN_IF_ERROR(EnsureTileSegments<kTileTargets>(context, post_ids, row_splits, n_post,
                                          &tile_segments->segments));
    const auto& segments = *tile_segments->segments;
    Tensor starts_tensor;
    unsigned int* sample_starts;
    TF_RETURN_IF_ERROR(AllocateQueueWords(context, batch + 1, &starts_tensor, &sample_starts));
    TF_RETURN_IF_ERROR(GpuLaunchKernel(QueueSampleStartsKernel, 512, 256, 0, device.stream(),
                                       n_pre, static_cast<int>(batch), queue, queue_count,
                                       sample_starts));
    const bool aligned = reinterpret_cast<uintptr_t>(output->flat<T>().data()) % 8 == 0;
    constexpr int kShared = kTileTargets * sizeof(::float4);
    return GpuLaunchKernel(
        TiledForwardKernel<T, W>, static_cast<int>(batch * tiles), kTileThreads, kShared,
        device.stream(), n_pre, n_post, tiles, !zero_output, aligned,
        spikes.slots, queue, sample_starts,
        reinterpret_cast<const unsigned int*>(
            segments.segment_splits.flat<int32>().data()),
        reinterpret_cast<const unsigned int*>(
            segments.segment_starts.flat<int32>().data()),
        reinterpret_cast<const unsigned int*>(
            segments.segment_tiles.flat<int32>().data()),
        segments.segment_masks.NumElements() > 0
            ? reinterpret_cast<const unsigned long long*>(
                  segments.segment_masks.flat<int32>().data())
            : nullptr,
        row_splits.flat<uint32>().data(), weights.flat<W>().data(),
        post_ids.flat<uint32>().data(), synapse_types.flat<uint8>().data(),
        edge_ids.flat<uint32>().data(), basis.flat<float>().data(),
        output->flat<T>().data());
  }
  // About 1,024 edges of work per ticket; see ForwardKernel.
  const int64_t mean_edges = std::max<int64_t>(1, post_ids.NumElements() / n_pre);
  const unsigned int slots_per_ticket =
      static_cast<unsigned int>(std::clamp<int64_t>(1024 / mean_edges, 1, 8));
#define LAUNCH_FORWARD_WITH(OUTPUT_TYPE, OUTPUT_PTR, AGGREGATE)                \
  TF_RETURN_IF_ERROR(GpuLaunchKernel(                                          \
      ForwardKernel<T, W, OUTPUT_TYPE, kBasis, AGGREGATE>, kEventBlocks,       \
      V1_FORWARD_THREADS, 0, device.stream(), n_pre, n_post, n_basis,          \
      spikes.slots, queue, queue_count, queue_count + 1,                       \
      slots_per_ticket,                                                        \
      weights.flat<W>().data(),                                                \
      post_ids.flat<uint32>().data(), synapse_types.flat<uint8>().data(),      \
      row_splits.flat<uint32>().data(), edge_ids.flat<uint32>().data(),        \
      basis.flat<float>().data(), OUTPUT_PTR))
#define LAUNCH_FORWARD(OUTPUT_TYPE, OUTPUT_PTR)                                 \
  if (aggregate_runs) {                                                        \
    LAUNCH_FORWARD_WITH(OUTPUT_TYPE, OUTPUT_PTR, true);                        \
  } else {                                                                     \
    LAUNCH_FORWARD_WITH(OUTPUT_TYPE, OUTPUT_PTR, false);                       \
  }
#if V1_FORWARD_FLOAT_ACCUM
  Tensor accumulation;
  TF_RETURN_IF_ERROR(context->allocate_temp(DT_FLOAT, output->shape(), &accumulation));
  constexpr int kElementThreads = 256;
  const int64_t elements = output->NumElements();
  const int element_blocks =
      static_cast<int>((elements + kElementThreads - 1) / kElementThreads);
  TF_RETURN_IF_ERROR(GpuLaunchKernel(
      SeedForwardAccumKernel<T>, element_blocks, kElementThreads, 0, device.stream(),
      elements, output->flat<T>().data(), accumulation.flat<float>().data()));
  LAUNCH_FORWARD(float, accumulation.flat<float>().data());
  TF_RETURN_IF_ERROR(GpuLaunchKernel(
      CastForwardOutputKernel<T>, element_blocks, kElementThreads, 0, device.stream(),
      elements, accumulation.flat<float>().data(), output->flat<T>().data()));
#else
  LAUNCH_FORWARD(T, output->flat<T>().data());
#endif
#undef LAUNCH_FORWARD
#undef LAUNCH_FORWARD_WITH
  return OkStatus();
}

// ---------------------------------------------------------------------------
// Backward
//
// The spike gradient is an SpMM over the compact (postsynaptic neuron, synapse
// type) pairs: each distinct pair is projected onto the basis once instead of
// once per edge (1,675,972 pairs for 84,145,692 edges on the 203,816-neuron
// network), and a row kernel sums projection * weight over each row's edges.
// The weight gradient is the event-driven SDDMM in event_weight_grad.cuh.
//
// The batch is processed in slices of kSlice = min(64, next power of two)
// samples (at most 32 with FP32), the projection is laid out
// [pair, batch_stride] with batch_stride a whole number of slices, and one grid
// row of the SpMM runs per slice. Padding
// samples project to zero and are never written back.
// ---------------------------------------------------------------------------

// A zero spike gradient plus its addend, as the separate add would form it.
template <typename T>
__global__ void AddToZeroKernel(int64_t elements, const T* addend, T* output) {
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < elements; index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    output[index] = FromFloat<T>(__fadd_rn(0.0f, AsFloat(addend[index])));
  }
}

// Largest finite |current_grad|, as raw float bits in a uint32. Non-negative
// floats order identically to their bit patterns, so an integer atomicMax is a
// float max. Non-finite entries are skipped on purpose: the backward
// legitimately propagates NaNs from `current_grad`, and letting one NaN poison
// the scale would turn every spike gradient into a NaN instead of only the
// affected ones.
//
// One warp's share: every lane strides from `first` by `stride`, and the
// warp's maximum is returned to every lane.
template <typename T>
__device__ __forceinline__ float WarpAbsMaxFinite(int64_t elements, const T* values,
                                                  int64_t first, int64_t stride) {
  float local = 0.0f;
  for (int64_t index = first; index < elements; index += stride) {
    const float value = fabsf(AsFloat(values[index]));
    if (isfinite(value) && value > local) local = value;
  }
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    local = fmaxf(local, __shfl_xor_sync(0xffffffff, local, mask));
  }
  return local;
}

template <typename T>
__global__ void AbsMaxFiniteKernel(int64_t elements, const T* values,
                                   unsigned int* result) {
  const float local = WarpAbsMaxFinite(
      elements, values, static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x,
      static_cast<int64_t>(blockDim.x) * gridDim.x);
  if ((threadIdx.x & 31) == 0 && local > 0.0f) {
    atomicMax(result, __float_as_uint(local));
  }
}

// Turn that bound into a power-of-two scale that puts the largest projection
// near 8192, three binades below fp16's 65504 ceiling.
//
// With FP16 spikes the pair projection is stored as FP16, which halves the row
// kernel's hottest gather and makes the 16-byte packed load possible. The
// values are an fp16 upstream gradient dotted with the basis, so unscaled they
// sit near fp16's 6e-5 flush-to-zero floor; the scale removes that underflow. A
// power of two is exact in both directions, so the only error introduced is
// mantissa rounding, and the spike gradient is linear in the projection, so
// the row kernel undoes the scale once after its FP32 accumulation.
//
// |projected| <= max|current_grad| * max_type sum_r |basis[type][r]|, so the
// bound is exact and one pass over `current_grad` suffices -- the projection
// itself never has to be measured.
//
// `max_bits` is a measured bound or a caller's (`abs_max_bound`). A caller's
// bound may be non-finite: a rescaled bound that overflowed (see
// V1CsrBackwardPairProjected). The warp then measures `values` itself -- slow,
// but it only happens when the upstream gradient overflowed.
template <typename T>
__global__ void ProjectionScaleKernel(const unsigned int* max_bits, const float* basis,
                                      int n_types, int n_basis, int64_t elements,
                                      const T* values, float* scale_out) {
  // One warp: each lane takes every 32nd synapse type, then a max-reduce.
  // Row sums keep their receptor order and max is order-independent, so the
  // scale is bit-identical to a serial loop -- which, as one thread doing 360
  // dependent global loads, cost ~18 us per call.
  float basis_l1 = 0.0f;
  for (int type = threadIdx.x; type < n_types; type += 32) {
    float row = 0.0f;
    for (int receptor = 0; receptor < n_basis; ++receptor) {
      row += fabsf(basis[type * n_basis + receptor]);
    }
    basis_l1 = fmaxf(basis_l1, row);
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    basis_l1 = fmaxf(basis_l1, __shfl_xor_sync(0xffffffffu, basis_l1, offset));
  }
  float max_value = __uint_as_float(*max_bits);
  if (!isfinite(max_value)) max_value = WarpAbsMaxFinite(elements, values, threadIdx.x, 32);
  if (threadIdx.x != 0) return;
  const float bound = max_value * basis_l1;
  float scale = 1.0f;
  if (isfinite(bound) && bound > 0.0f) {
    scale = exp2f(floorf(log2f(8192.0f / bound)));
    if (!isfinite(scale) || scale <= 0.0f) scale = 1.0f;
  }
  scale_out[0] = scale;
  scale_out[1] = 1.0f / scale;
}

template <typename T, int kBasis>
__device__ __forceinline__ float BasisProjection(const T* upstream, const float* basis,
                                                 int type, int n_basis) {
  const int width = kBasis == 0 ? n_basis : kBasis;
  float result = 0.0f;
#pragma unroll
  for (int receptor = 0; receptor < (kBasis == 0 ? n_basis : kBasis); ++receptor) {
    result += AsFloat(upstream[receptor]) * basis[type * width + receptor];
  }
  return result;
}

__device__ __forceinline__ float ProjToFloat(float value) { return value; }
__device__ __forceinline__ float ProjToFloat(__half value) { return __half2float(value); }

__device__ __forceinline__ void ProjStore(float* address, float value) { *address = value; }
__device__ __forceinline__ void ProjStore(__half* address, float value) {
  *address = __float2half(value);
}

// One sample's `current_grad` row of kBasis == 4 receptors in one vector load:
// 8 bytes of FP16 or 16 of FP32 (rows are n_basis elements, so aligned).
template <typename T>
__device__ __forceinline__ void LoadGradRow4(const T* row, float* out) {
  if constexpr (std::is_same<T, Eigen::half>::value) {
    const ::uint2 raw = *reinterpret_cast<const ::uint2*>(row);
    const __half* packed = reinterpret_cast<const __half*>(&raw);
#pragma unroll
    for (int receptor = 0; receptor < 4; ++receptor) out[receptor] = __half2float(packed[receptor]);
  } else {
    const ::float4 raw = *reinterpret_cast<const ::float4*>(row);
    out[0] = raw.x; out[1] = raw.y; out[2] = raw.z; out[3] = raw.w;
  }
}

// kCount consecutive projection elements (2 to 32 bytes, aligned to their
// size) in as few vector stores as possible.
template <typename P, int kCount>
__device__ __forceinline__ void StoreProjectionGroup(P* destination, const float* values) {
  constexpr int kBytes = kCount * static_cast<int>(sizeof(P));
  union {
    P element[kCount];
    ::uint4 quad[kBytes >= 16 ? kBytes / 16 : 1];
  } packed;
#pragma unroll
  for (int index = 0; index < kCount; ++index) ProjStore(&packed.element[index], values[index]);
  if constexpr (kBytes >= 16) {
#pragma unroll
    for (int quad = 0; quad < kBytes / 16; ++quad) {
      reinterpret_cast<::uint4*>(destination)[quad] = packed.quad[quad];
    }
  } else {
#pragma unroll
    for (int index = 0; index < kCount; ++index) destination[index] = packed.element[index];
  }
}

// Compact projection, emitted as [pair, batch_stride]. One thread per pair
// walks the whole batch kGroup samples at a time (32 bytes of its line, one
// whole sector per store). A warp-wide load is then one sample of 32
// consecutive pairs, which cover about four postsynaptic neurons -- one or two
// sectors of `current_grad` -- and a four-receptor row is a single vector
// load. The kGroup loads of a group are independent, so enough of them are in
// flight to stream: the shared-memory tile transpose this replaces was
// latency-bound on four dependent 2-byte loads per element and a barrier per
// 32 x 32 tile. The arithmetic is BasisProjection's term for term, then the
// same scale and rounding, so the projection is bitwise unchanged.
template <typename T, typename P, int kBasis, int kGroup>
__global__ __launch_bounds__(256) void PreprojectPairsKernel(
    int n_post, int n_basis, int64_t n_pairs, int64_t batch, int64_t batch_stride,
    const T* current_grad, const float* basis, const uint32* pair_posts,
    const uint8* pair_types, P* projected, const float* scale) {
  const int64_t pair = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pair >= n_pairs) return;
  const float factor = scale == nullptr ? 1.0f : scale[0];
  const int type = pair_types[pair];
  const T* grad = current_grad + static_cast<int64_t>(pair_posts[pair]) * n_basis;
  const int64_t sample_stride = static_cast<int64_t>(n_post) * n_basis;
  ::float4 coefficient = ::make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  if constexpr (kBasis == 4) coefficient = LoadBasis4(basis, type);
  P* line = projected + pair * batch_stride;
  for (int64_t first = 0; first < batch_stride; first += kGroup) {
    float value[kGroup];
#pragma unroll
    for (int index = 0; index < kGroup; ++index) {
      const int64_t b = first + index;
      float result = 0.0f;
      if (b < batch) {
        if constexpr (kBasis == 4) {
          float upstream[4];
          LoadGradRow4<T>(grad + b * sample_stride, upstream);
          result += upstream[0] * coefficient.x;
          result += upstream[1] * coefficient.y;
          result += upstream[2] * coefficient.z;
          result += upstream[3] * coefficient.w;
        } else {
          result = BasisProjection<T, kBasis>(grad + b * sample_stride, basis, type, n_basis);
        }
      }
      value[index] = result * factor;
    }
    StoreProjectionGroup<P, kGroup>(line + first, value);
  }
}

// Cache policy for the row kernel's operands. The projection is re-read once
// per edge sharing a pair (50.2 times on the 203,816-neuron network) and is the
// only array with reuse. `pair_ids` and `weights` are each touched once per
// edge, so evict-first hints keep that single-use traffic from displacing the
// projection in L2.
__device__ __forceinline__ uint32 LoadEdgeIndex(const uint32* address) {
  return __ldcs(address);
}

template <typename W>
__device__ __forceinline__ float LoadEdgeWeight(const W* address) {
  if constexpr (std::is_same<W, float>::value) return __ldcs(address);
  return AsFloat(*address);
}

// One lane's kPack consecutive samples of a pair's projected slice line, in
// one vector load: 16 B carries four FP32 or eight FP16 samples.
template <typename P, int kPack>
__device__ __forceinline__ void LoadProjection(const P* source, float* out) {
  if constexpr (std::is_same<P, float>::value && kPack == 4) {
    const ::float4 raw = *reinterpret_cast<const ::float4*>(source);
    out[0] = raw.x; out[1] = raw.y; out[2] = raw.z; out[3] = raw.w;
  } else if constexpr (std::is_same<P, float>::value && kPack == 2) {
    const ::float2 raw = *reinterpret_cast<const ::float2*>(source);
    out[0] = raw.x; out[1] = raw.y;
  } else if constexpr (std::is_same<P, __half>::value && (kPack == 8 || kPack == 4)) {
    using Vector = typename std::conditional<kPack == 8, ::uint4, ::uint2>::type;
    const Vector raw = *reinterpret_cast<const Vector*>(source);
    const __half* packed = reinterpret_cast<const __half*>(&raw);
#pragma unroll
    for (int index = 0; index < kPack; ++index) out[index] = __half2float(packed[index]);
  } else {
#pragma unroll
    for (int index = 0; index < kPack; ++index) out[index] = ProjToFloat(source[index]);
  }
}

// Spike gradient for one batch slice: an SpMM over the compact pairs, one CSR
// row per warp. A block owns kTileRows consecutive presynaptic rows, which its
// warps take from a shared counter (row lengths run from 1 to ~3,000 edges, so
// a static split would leave warps idle behind the longest row).
//
// A lane holds kPack consecutive samples, so an edge needs kSlice / kPack
// lanes, a warp splits into that many independent edge slots, and one load
// serves all of them. The descriptors (`pair_ids`, `weights`) are read once per
// 32-edge tile cooperatively and broadcast with `__shfl_sync`. Out-of-range
// lanes point at `sentinel_pair`, an all-zero projection row, so the hot loop
// loads unconditionally instead of branching and zero-filling.
//
// With FP16 and a batch above 32 the slice is 64 samples: each edge then reads
// its pair's whole 128-byte line in one pass, and the descriptors are streamed
// once instead of once per 32-sample slice.
//
// Each row's slice line lands in shared memory, and the block writes the tile
// out as [batch, pre] -- kTileRows consecutive rows of one sample per warp
// store -- so no [pre, batch] scratch buffer and no transpose pass are needed.
// Edgeless rows are written as zeros, which is what lets the op skip clearing
// its output. The odd row stride kSlice + 1 keeps the transposed read
// conflict-free.
constexpr int kTileRows = 32;
constexpr int kTileWarps = 8;

template <typename T, typename W, typename P, int kSlice, int kPack, bool kMultiSlice>
__global__ __launch_bounds__(32 * kTileWarps) void BackwardRowTileKernel(
    const P* projected, int64_t runtime_stride, const W* weights, const uint32* pair_ids,
    const uint32* edge_ids, const uint32* row_splits, int64_t n_pre, int64_t batch,
    const T* dampening, SpikeSlots<T*> spike_grad, const float* inverse_scale,
    uint32 sentinel_pair, SpikeSlots<const T*> addends) {
  constexpr int kTile = 32;
  constexpr int kLanesPerEdge = kSlice / kPack;
  constexpr int kSlots = 32 / kLanesPerEdge;
  constexpr int kPerSlot = kTile / kSlots;
  // A 64-sample slice has half the edge slots of a 32-sample one. Each lane
  // keeps kPartials = kSlice / 32 accumulators, taking edge `column` into
  // partial `step % kPartials`, so every partial sums exactly the edges one
  // 32-sample slot summed, in the same order, and the fold below combines
  // them in the 32-sample butterfly's order: the result is bitwise identical
  // to the two-slice kernel's (IEEE addition is commutative).
  constexpr int kPartials = kSlice > 32 ? kSlice / 32 : 1;
  static_assert(kPerSlot % kPartials == 0, "partials must tile the steps");
  // Stored already rounded to T: the value the output gets either way. A row
  // stride of an odd number of 32-bit words keeps the transposed read
  // conflict-free.
  constexpr int kRowStride = kSlice + (sizeof(T) == 2 ? 2 : 1);
  __shared__ T result[kTileRows][kRowStride];
  __shared__ int next_row;
  // One slice keeps the stride a compile-time constant, so the hot loop's
  // address arithmetic stays a shift.
  const int64_t batch_stride = kMultiSlice ? runtime_stride : kSlice;
  const int lane = threadIdx.x & 31;
  const int slot = lane / kLanesPerEdge;
  const int sub = lane % kLanesPerEdge;
  const int64_t pre_base = static_cast<int64_t>(blockIdx.x) * kTileRows;
  const int64_t batch_base = kMultiSlice ? static_cast<int64_t>(blockIdx.y) * kSlice : 0;
  const int64_t sample_base = batch_base + sub * kPack;
  const float factor =
      (inverse_scale == nullptr ? 1.0f : inverse_scale[1]) * AsFloat(*dampening);
  if (threadIdx.x == 0) next_row = kTileWarps;
  __syncthreads();
  for (int local = threadIdx.x >> 5; local < kTileRows;) {
    const int64_t pre = pre_base + local;
    if (pre >= n_pre) break;
    const uint32 begin = row_splits[pre];
    const uint32 end = row_splits[pre + 1];
    float grad[kPartials][kPack] = {};
    for (uint32 base = begin; base < end; base += kTile) {
      const uint32 edge_lane = base + lane;
      const bool own = edge_lane < end;
      const uint32 my_pair = own ? LoadEdgeIndex(pair_ids + edge_lane) : sentinel_pair;
      const float my_weight =
          own ? LoadEdgeWeight<W>(weights + V1_EDGE_INDEX(edge_lane)) : 0.0f;
#pragma unroll
      for (int step = 0; step < kPerSlot; ++step) {
        const int column = kSlots * step + slot;
        const uint32 pair = __shfl_sync(0xffffffff, my_pair, column);
        const float weight = __shfl_sync(0xffffffff, my_weight, column);
        float value[kPack];
        LoadProjection<P, kPack>(projected + static_cast<int64_t>(pair) * batch_stride +
                                     sample_base, value);
#pragma unroll
        for (int sample = 0; sample < kPack; ++sample) {
          grad[step % kPartials][sample] += value[sample] * weight;
        }
      }
    }
    // Fold the slots, which each accumulated the same samples from a different
    // edge subset, then the partials. Afterwards every lane sharing `sub` holds
    // the total.
#pragma unroll
    for (int mask = kLanesPerEdge; mask < 32; mask <<= 1) {
#pragma unroll
      for (int partial = 0; partial < kPartials; ++partial) {
#pragma unroll
        for (int sample = 0; sample < kPack; ++sample) {
          grad[partial][sample] += __shfl_xor_sync(0xffffffff, grad[partial][sample], mask);
        }
      }
    }
    if (lane < kLanesPerEdge) {
#pragma unroll
      for (int sample = 0; sample < kPack; ++sample) {
        float total = grad[0][sample];
#pragma unroll
        for (int partial = 1; partial < kPartials; ++partial) total += grad[partial][sample];
        result[local][sub * kPack + sample] =
            FromFloat<T>(begin == end ? 0.0f : total * factor);
      }
    }
    if (lane == 0) local = atomicAdd(&next_row, 1);
    local = __shfl_sync(0xffffffff, local, 0);
  }
  __syncthreads();
  static_assert(kTileRows % 32 == 0, "the tile is written 32 rows per warp store");
  // Each lane stores one row throughout, so it locates the row's slot once.
  // Rows of the last of several slots store +0 + g (see V1CsrBackwardPairProjected).
  const int64_t oldest_first =
      spike_grad.count > 1 ? static_cast<int64_t>(spike_grad.count - 1) * spike_grad.width
                           : n_pre;
  for (int group = 0; group < kTileRows / 32; ++group) {
    const int row = group * 32 + lane;
    const int64_t pre = pre_base + row;
    if (pre >= n_pre) continue;
    T* column = spike_grad.Row(static_cast<unsigned int>(pre));
    const bool oldest = pre >= oldest_first;
    // The row's addend (see `spike_grad_addends`), if its slot has one.
    const T* added = nullptr;
    if (addends.count > 0) {
      const unsigned int slot =
          addends.count == 1 ? 0u : spike_grad.by_width.Div(static_cast<unsigned int>(pre));
#pragma unroll
      for (int k = 0; k < kMaxSpikeSlots; ++k) {
        if (k == static_cast<int>(slot) && addends.slot[k] != nullptr) {
          added = addends.slot[k] + (pre - static_cast<int64_t>(slot) * spike_grad.width);
        }
      }
    }
    for (int sample = threadIdx.x >> 5; sample < kSlice; sample += kTileWarps) {
      const int64_t b = batch_base + sample;
      if (b >= batch) continue;
      T value = result[row][sample];
      // A select rather than an add of +0, which fast math may fold away.
      if (oldest && AsFloat(value) == 0.0f) value = FromFloat<T>(0.0f);
      // The add a caller would otherwise launch: float sum, rounded once to T.
      if (added != nullptr) {
        value = FromFloat<T>(__fadd_rn(AsFloat(value), AsFloat(added[b * spike_grad.width])));
      }
      column[b * spike_grad.width] = value;
    }
  }
}

// One slice width's backward. P is the projection element: scaled FP16 with
// FP16 spikes, FP32 with FP32 spikes, which keeps the precision the caller
// chose.
template <typename T, typename W, int kBasis, int kSlice, int kPack>
Status LaunchSlicedPairBackward(
    OpKernelContext* context, const SpikeMatrix<const T*>& spikes, const Tensor& current_grad,
    const Tensor& weights, const Tensor& post_ids, const Tensor& synapse_types,
    const Tensor& row_splits, const Tensor& edge_ids,
    const Tensor& nonempty_rows, const Tensor& basis, const Tensor& dampening,
    const Tensor& pair_ids, const Tensor& pair_posts, const Tensor& pair_types,
    int n_post, const SpikeSlots<T*>& spike_grad, Tensor* weight_grad, bool accumulate,
    const ActivityMasks* masks, const unsigned int* abs_max_bits,
    const SpikeSlots<const T*>& addends) {
  constexpr bool kScaled = std::is_same<T, Eigen::half>::value;
  using P = typename std::conditional<kScaled, __half, float>::type;
  const int64_t n_pairs = pair_posts.NumElements();
  const int64_t batch = spikes.batch;
  const int64_t n_pre = spikes.n_pre();
  const int64_t n_slices = (batch + kSlice - 1) / kSlice;
  const int64_t batch_stride = n_slices * kSlice;
  const int n_basis = basis.dim_size(1);
  auto device = context->eigen_device<GPUDevice>();
  // The projection, plus one all-zero sentinel pair at index n_pairs. It has
  // the element type of T, so it is allocated as one.
  const int64_t projected_elements = batch_stride * n_pairs;
  Tensor projected_tensor;
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DataTypeToEnum<T>::value, TensorShape({projected_elements + batch_stride}),
      &projected_tensor));
  P* projected = reinterpret_cast<P*>(projected_tensor.flat<T>().data());
  cudaMemsetAsync(projected + projected_elements, 0, batch_stride * sizeof(P),
                  device.stream());
  float* scale = nullptr;
  Tensor scale_tensor;
  if constexpr (kScaled) {
    // [scale, 1 / scale] in floats, then the abs-max bits.
    TF_RETURN_IF_ERROR(context->allocate_temp(DT_FLOAT, TensorShape({3}), &scale_tensor));
    scale = scale_tensor.flat<float>().data();
    // A caller-supplied bound (the GLIF backward's fused abs-max of the same
    // current_grad) replaces the pass over current_grad.
    const unsigned int* max_bits = abs_max_bits;
    if (max_bits == nullptr) {
      unsigned int* measured = reinterpret_cast<unsigned int*>(scale + 2);
      cudaMemsetAsync(measured, 0, sizeof(unsigned int), device.stream());
      TF_RETURN_IF_ERROR(GpuLaunchKernel(
          AbsMaxFiniteKernel<T>, 1024, 256, 0, device.stream(),
          current_grad.NumElements(), current_grad.flat<T>().data(), measured));
      max_bits = measured;
    }
    TF_RETURN_IF_ERROR(GpuLaunchKernel(
        ProjectionScaleKernel<T>, 1, 32, 0, device.stream(), max_bits,
        basis.flat<float>().data(), static_cast<int>(basis.dim_size(0)), n_basis,
        current_grad.NumElements(), current_grad.flat<T>().data(), scale));
  }
  constexpr int kGroup = std::min(kSlice, 32 / static_cast<int>(sizeof(P)));
  TF_RETURN_IF_ERROR(GpuLaunchKernel(
      PreprojectPairsKernel<T, P, kBasis, kGroup>,
      static_cast<unsigned>((n_pairs + 255) / 256), 256, 0, device.stream(), n_post,
      n_basis, n_pairs, batch, batch_stride, current_grad.flat<T>().data(),
      basis.flat<float>().data(), pair_posts.flat<uint32>().data(),
      pair_types.flat<uint8>().data(), projected, scale));
  const dim3 row_grid(static_cast<unsigned>((n_pre + kTileRows - 1) / kTileRows),
                      static_cast<unsigned>(n_slices));
#define LAUNCH_ROWS(MULTI_SLICE)                                                    \
  TF_RETURN_IF_ERROR(GpuLaunchKernel(                                               \
      BackwardRowTileKernel<T, W, P, kSlice, kPack, MULTI_SLICE>, row_grid,         \
      32 * kTileWarps, 0, device.stream(), projected, batch_stride,                 \
      weights.flat<W>().data(), pair_ids.flat<uint32>().data(),                     \
      edge_ids.flat<uint32>().data(), row_splits.flat<uint32>().data(), n_pre,      \
      batch, dampening.flat<T>().data(), spike_grad, scale,                         \
      static_cast<uint32>(n_pairs), addends))
  if (n_slices > 1) {
    LAUNCH_ROWS(true);
  } else {
    LAUNCH_ROWS(false);
  }
#undef LAUNCH_ROWS
  return LaunchEventWeightGrad<T, kBasis>(context, spikes, current_grad, basis,
                                          post_ids, synapse_types, row_splits,
                                          edge_ids, n_post, weight_grad, accumulate, masks);
}

// Any batch size, basis dimension and spike dtype. The slice is the next power
// of two up to 64 with FP16 (up to 32 with FP32); a lane loads 16 bytes, eight
// FP16 or four FP32 samples. One 64-sample pass streams the edge descriptors
// once where two 32-sample slices streamed them twice. With `accumulate` the
// weight gradient is added to what `weight_grad` already holds instead of
// replacing it.
// `spike_grad` has the slot layout of `spikes`.
template <typename T, typename W, int kBasis>
Status LaunchPairProjectedBackward(
    OpKernelContext* context, const SpikeMatrix<const T*>& spikes, const Tensor& current_grad,
    const Tensor& weights, const Tensor& post_ids, const Tensor& synapse_types,
    const Tensor& row_splits, const Tensor& edge_ids,
    const Tensor& nonempty_rows, const Tensor& basis, const Tensor& dampening,
    const Tensor& pair_ids, const Tensor& pair_posts, const Tensor& pair_types,
    int n_post, const SpikeSlots<T*>& spike_grad, Tensor* weight_grad,
    bool accumulate = false, const ActivityMasks* masks = nullptr,
    const unsigned int* abs_max_bits = nullptr,
    const SpikeSlots<const T*>& addends = SpikeSlots<const T*>{}) {
  const int64_t batch = spikes.batch;
  if (batch == 0 || nonempty_rows.NumElements() == 0) {
    auto stream = context->eigen_device<GPUDevice>().stream();
    for (int k = 0; k < spike_grad.count; ++k) {
      cudaMemsetAsync(spike_grad.slot[k], 0, batch * spike_grad.width * sizeof(T), stream);
      const int64_t elements = batch * static_cast<int64_t>(spike_grad.width);
      if (k < addends.count && addends.slot[k] != nullptr && elements > 0) {
        TF_RETURN_IF_ERROR(GpuLaunchKernel(
            AddToZeroKernel<T>, static_cast<int>((elements + 255) / 256), 256, 0, stream,
            elements, addends.slot[k], spike_grad.slot[k]));
      }
    }
    if (!accumulate)
      cudaMemsetAsync(weight_grad->flat<float>().data(), 0, weight_grad->TotalBytes(), stream);
    return OkStatus();
  }
  constexpr int kWidePack = std::is_same<T, Eigen::half>::value ? 8 : 4;
#define LAUNCH_SLICE(SLICE, PACK)                                                \
  return LaunchSlicedPairBackward<T, W, kBasis, SLICE, PACK>(                  \
      context, spikes, current_grad, weights, post_ids, synapse_types,         \
      row_splits, edge_ids, nonempty_rows, basis, dampening, pair_ids,         \
      pair_posts, pair_types, n_post, spike_grad, weight_grad, accumulate, masks,  \
      abs_max_bits, addends)
  // FP32 keeps 32-sample slices: at four samples per lane a 64-sample slice
  // would leave two edge slots per warp, lengthening each lane's serial sum
  // (measured 1.4x the FP32 spike-gradient error), and FP32 is not the hot path.
  if (batch > 32 && kWidePack == 8) LAUNCH_SLICE(64, kWidePack);
  if (batch > 16) LAUNCH_SLICE(32, kWidePack);
  if (batch > 8) LAUNCH_SLICE(16, 4);
  if (batch > 4) LAUNCH_SLICE(8, 4);
  if (batch > 2) LAUNCH_SLICE(4, 4);
  if (batch > 1) LAUNCH_SLICE(2, 2);
  LAUNCH_SLICE(1, 1);
#undef LAUNCH_SLICE
}

// Hands `launch` the buffer the recurrent weight gradient goes to. By default
// that is output `weight_grad`, which the kernels overwrite. With `kAccumulate` it is the
// FP32 [n_edges] variable behind input `accumulator_input`, which the kernels add
// into while its lock is held, so the training graph never carries a dense
// per-step weight gradient. The handle lookup rejects a variable that lives on
// another device, so a replica can only ever add into its own accumulator.
template <bool kAccumulate, typename Launch>
Status WithWeightGradBuffer(OpKernelContext* context, int accumulator_input,
                            int64_t n_edges, Launch launch) {
  if constexpr (!kAccumulate) {
    Tensor* weight_grad;
    TF_RETURN_IF_ERROR(
        context->allocate_output("weight_grad", TensorShape({n_edges}), &weight_grad));
    return launch(weight_grad);
  } else {
    core::RefCountPtr<Var> variable;
    TF_RETURN_IF_ERROR(LookupResource(
        context, HandleFromInput(context, accumulator_input), &variable));
    mutex_lock lock(*variable->mu());
    Tensor* accumulator = variable->tensor();
    if (!variable->is_initialized || accumulator->dtype() != DT_FLOAT ||
        accumulator->NumElements() != n_edges) {
      return errors::InvalidArgument(
          "the accumulator must be an initialized float32 [n_edges] variable");
    }
    if (!accumulator->RefCountIsOne()) {
      // A read of the current value is still alive, so adding in place would
      // change what it sees. Copy on write, as TensorFlow's own resource
      // update ops do.
      Tensor copy;
      TF_RETURN_IF_ERROR(
          context->allocate_temp(DT_FLOAT, accumulator->shape(), &copy));
      cudaMemcpyAsync(copy.flat<float>().data(), accumulator->flat<float>().data(),
                      accumulator->TotalBytes(), cudaMemcpyDeviceToDevice,
                      context->eigen_device<GPUDevice>().stream());
      *accumulator = copy;
    }
    return launch(accumulator);
  }
}

// ---------------------------------------------------------------------------
// Operators
// ---------------------------------------------------------------------------

// The forward ops' carried queue records (`queues`, input list starting at
// `first`) and their `queue` output (output 1): the newest slot's record with
// several slots, else empty. `queues` is empty, or holds the records of slots 1
// to N - 1 (the previous step's slots 0 to N - 2).
template <typename T>
Status ReadQueueRecords(OpKernelContext* context, const SpikeMatrix<const T*>& spikes,
                        int first, QueueRecords* carried, unsigned int** newest) {
  const int count = spikes.slots.count;
  const int given = context->num_inputs() - first;
  *carried = QueueRecords{};
  *newest = nullptr;
  if (given != 0 && given != count - 1) {
    return errors::InvalidArgument("queues must be empty or hold the records of slots 1 to N - 1");
  }
  const int64_t words = QueueRecordWords(spikes.batch, spikes.slots.width);
  for (int k = 0; k < given; ++k) {
    const Tensor& record = context->input(first + k);
    if (record.NumElements() != words) {
      return errors::InvalidArgument("a carried queue record has ", record.NumElements(),
                                     " words, expected ", words);
    }
    carried->record[k + 1] =
        reinterpret_cast<const unsigned int*>(record.flat<uint32>().data());
  }
  Tensor* output;
  TF_RETURN_IF_ERROR(context->allocate_output(
      1, TensorShape({count > 1 ? words : 0}), &output));
  if (count > 1) {
    *newest = reinterpret_cast<unsigned int*>(output->flat<uint32>().data());
    // An empty batch builds no queue: its record is a zero count and starts.
    if (spikes.NumElements() == 0) {
      cudaMemsetAsync(*newest, 0, words * sizeof(unsigned int),
                      context->eigen_device<GPUDevice>().stream());
      *newest = nullptr;
    }
  }
  return OkStatus();
}

// The `spike_grad` outputs, one per spike slot and shaped like it.
template <typename T>
Status AllocateSpikeGradients(OpKernelContext* context, const OpInputList& spikes,
                              const SpikeMatrix<const T*>& matrix, SpikeSlots<T*>* grads) {
  OpOutputList outputs;
  TF_RETURN_IF_ERROR(context->output_list("spike_grad", &outputs));
  grads->count = matrix.slots.count;
  grads->width = matrix.slots.width;
  grads->n_pre = matrix.slots.n_pre;
  grads->by_width = matrix.slots.by_width;
  for (int k = 0; k < spikes.size(); ++k) {
    Tensor* grad;
    TF_RETURN_IF_ERROR(outputs.allocate(k, spikes[k].shape(), &grad));
    grads->slot[k] = grad->flat<T>().data();
  }
  return OkStatus();
}

// The kernels index the edge-aligned arrays by CSR position without bounds
// checks, so a length mismatch would read (and, for the weight gradient,
// write) out of bounds instead of failing. Direct-CSR kernels never read
// `edge_ids`, which the wrapper then passes empty.
static Status ValidateEdgeArrays(const Tensor& weights, const Tensor& post_ids,
                                 const Tensor& synapse_types, const Tensor& edge_ids) {
  const int64_t n_edges = post_ids.NumElements();
  const bool edge_ids_valid =
      edge_ids.NumElements() == n_edges || (V1_DIRECT_CSR && edge_ids.NumElements() == 0);
  if (weights.NumElements() != n_edges || synapse_types.NumElements() != n_edges ||
      !edge_ids_valid) {
    return errors::InvalidArgument(
        "weights, post_ids, synapse_types and edge_ids must have one entry per "
        "edge; got ", weights.NumElements(), ", ", n_edges, ", ",
        synapse_types.NumElements(), " and ", edge_ids.NumElements());
  }
  return OkStatus();
}

template <typename T, typename W>
class V1CsrForwardOp : public OpKernel {
 public:
  explicit V1CsrForwardOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("n_post", &n_post_));
    OP_REQUIRES_OK(context, context->GetAttr("aggregate_runs", &aggregate_runs_));
  }

  void Compute(OpKernelContext* context) override {
    OpInputList spike_slots;
    OP_REQUIRES_OK(context, context->input_list("spikes", &spike_slots));
    SpikeMatrix<const T*> spikes;
    OP_REQUIRES_OK(context, ReadSpikeSlots<T>(spike_slots, &spikes));
    // The named inputs follow the spike list.
    const int first = spike_slots.size() - 1;
    const Tensor& weights = context->input(first + 1);
    const Tensor& post_ids = context->input(first + 2);
    const Tensor& synapse_types = context->input(first + 3);
    const Tensor& row_splits = context->input(first + 4);
    const Tensor& edge_ids = context->input(first + 5);
    const Tensor& basis = context->input(first + 6);
    const int initial_index = first + 7;
    const Tensor& initial = context->input(initial_index);
    OP_REQUIRES(context, basis.dims() == 2 && basis.dim_size(1) > 0,
                errors::InvalidArgument("basis must be [n_types,n_basis], n_basis > 0"));
    OP_REQUIRES(context, row_splits.NumElements() == spikes.n_pre() + 1,
                errors::InvalidArgument("row_splits does not match spike width"));
    OP_REQUIRES_OK(context, ValidateEdgeArrays(weights, post_ids, synapse_types, edge_ids));
    QueueRecords carried;
    unsigned int* newest;
    OP_REQUIRES_OK(context, ReadQueueRecords<T>(context, spikes, initial_index + 1, &carried,
                                                &newest));
    Tensor* output;
    const int64_t batch = spikes.batch;
    const int n_basis = basis.dim_size(1);
    const TensorShape shape({batch * n_post_, n_basis});
    const bool accumulate = initial.NumElements() > 0;
    OP_REQUIRES(
        context, !accumulate || initial.shape() == shape,
        errors::InvalidArgument("initial must be empty or match the output shape"));
    auto device = context->eigen_device<GPUDevice>();
    if (accumulate) {
      // Reuse the incoming buffer when TensorFlow can hand it over, so the
      // scatter lands directly on the previous source's currents and costs no
      // initialization traffic at all. Otherwise seed a fresh buffer with it.
      OP_REQUIRES_OK(context,
                     context->forward_input_or_allocate_output({initial_index}, 0, shape,
                                                               &output));
      if (output->flat<T>().data() != initial.flat<T>().data()) {
        cudaMemcpyAsync(output->flat<T>().data(), initial.flat<T>().data(),
                        output->NumElements() * sizeof(T),
                        cudaMemcpyDeviceToDevice, device.stream());
      }
    } else {
      // Left uninitialized: LaunchForward zeroes it or writes all of it.
      OP_REQUIRES_OK(context, context->allocate_output(0, shape, &output));
    }
    if (n_basis == 4) {
      OP_REQUIRES_OK(context, LaunchForward<T, W, 4>(
                                  context, spikes, weights, post_ids,
                                  synapse_types, row_splits, edge_ids, basis,
                                  n_post_, aggregate_runs_, output, !accumulate,
                                  &tile_segments_, carried, newest));
    } else {
      OP_REQUIRES_OK(context, LaunchForward<T, W, 0>(
                                  context, spikes, weights, post_ids,
                                  synapse_types, row_splits, edge_ids, basis,
                                  n_post_, aggregate_runs_, output, !accumulate,
                                  &tile_segments_, carried, newest));
    }
  }

 private:
  int n_post_;
  bool aggregate_runs_ = true;
  ForwardTileCache tile_segments_;
};

// One connectivity as a forward reads it, from the op's inputs or from a
// GPU-local resource (cuda_csr_resources).
struct ForwardCsr {
  const Tensor& weights;
  const Tensor& post_ids;
  const Tensor& synapse_types;
  const Tensor& row_splits;
  const Tensor& edge_ids;
  bool aggregate_runs;
};

// The recurrent history's currents plus an external input's (the LGN's), as
// V1CsrForward(external, initial = V1CsrForward(history)) computes them, for
// V1CsrForwardFused and V1CsrForwardFusedResource, which validate their own
// connectivities first. Where SweepTargets accepts the configuration one tile
// sweep accumulates both sources in FP32 and rounds once, saving the external
// pass's read and rewrite of the currents; otherwise the two forwards run in
// sequence, as the two operators would. Each op owns one: the tables and kernel
// flags live here, never in a static (both libraries compile this source).
template <typename T, typename W>
class FusedForward {
 public:
  // Allocates the currents (output 0) and the newest queue record (output 1,
  // from the carried records at input `queues`) and launches.
  Status Compute(OpKernelContext* context, const SpikeMatrix<const T*>& spikes,
                 const ForwardCsr& recurrent, const Tensor& activity,
                 const ForwardCsr& external, const Tensor& basis, int queues, int n_post,
                 const Tensor* pair_weights = nullptr) {
    if (!(activity.dims() == 2 && activity.dim_size(0) == spikes.batch &&
          activity.NumElements() <= kQueueMaxSlots)) {
      return errors::InvalidArgument(
          "activity must be [batch, n_external] with the spikes' batch");
    }
    QueueRecords carried;
    unsigned int* newest;
    TF_RETURN_IF_ERROR(ReadQueueRecords<T>(context, spikes, queues, &carried, &newest));
    Tensor* output;
    TF_RETURN_IF_ERROR(context->allocate_output(
        0, TensorShape({spikes.batch * n_post, basis.dim_size(1)}), &output));
    const SpikeMatrix<const T*> external_spikes =
        SingleSlot<T>(activity.flat<T>().data(), activity.dim_size(0), activity.dim_size(1));
    return basis.dim_size(1) == 4
               ? Launch<4>(context, spikes, recurrent, external_spikes, external, basis,
                           n_post, carried, newest, output, pair_weights)
               : Launch<0>(context, spikes, recurrent, external_spikes, external, basis,
                           n_post, carried, newest, output, pair_weights);
  }

 private:
  template <int kBasis>
  Status Launch(OpKernelContext* context, const SpikeMatrix<const T*>& spikes,
                const ForwardCsr& recurrent, const SpikeMatrix<const T*>& activity,
                const ForwardCsr& external, const Tensor& basis, int n_post,
                const QueueRecords& carried, unsigned int* newest, Tensor* output,
                const Tensor* pair_weights) {
    if constexpr (kBasis == 4 && !V1_FORWARD_FLOAT_ACCUM) {
      // The external input is one slot of the history's batch (validated).
      const int targets =
          !recurrent.aggregate_runs && external.aggregate_runs && activity.NumElements() > 0
              ? SweepTargets<T>(kSweepFused, spikes, n_post, carried, newest,
                                context->eigen_device<GPUDevice>().getNumGpuMultiProcessors())
              : 0;
      if (targets > 0) {
        bool available;
        bool launched = false;
        TF_RETURN_IF_ERROR(WithSweepTargets(targets, [&](auto shape) {
          constexpr int kTargets = decltype(shape)::value;
          // The scalar external input where the connectivity qualifies, in
          // FP16 only: in FP32, where no output rounding hides the FP32 sums,
          // its max_abs came out 59 % above the plain sweep's.
          // With the caller's per-pair weight sums (V1ExternalPairWeights, one
          // entry per distinct (row, target) pair), the pair table replaces the
          // run table: one entry and one weight per pair.
          const bool pairs = pair_weights != nullptr && pair_weights->NumElements() > 0;
          std::shared_ptr<const ScalarExternalTable> scalar;
          if (std::is_same<T, Eigen::half>::value) {
            mutex_lock lock(fused_scalar_.mu);
            auto& held = (pairs ? pairs_ : scalar_)[kSweepShape<kTargets>];
            TF_RETURN_IF_ERROR(EnsureScalarExternal<kTargets>(
                context, external.post_ids, external.row_splits, external.synapse_types,
                n_post, pairs, &held));
            scalar = held;
          }
          const bool scalar_mode_wanted = scalar != nullptr && scalar->usable;
          if (scalar_mode_wanted && pairs &&
              pair_weights->NumElements() != scalar->pair_edges.NumElements() - 1) {
            return errors::InvalidArgument("external_pair_weights has ",
                                           pair_weights->NumElements(), " entries, the CSR ",
                                           scalar->pair_edges.NumElements() - 1, " pairs");
          }
          if (scalar_mode_wanted) {
            TF_RETURN_IF_ERROR(SweepKernelAvailable<kTargets>(
                pairs ? reinterpret_cast<const void*>(
                            CsrTileSweepKernel<T, W, kSweepFusedPairs, kTargets>)
                      : reinterpret_cast<const void*>(
                            CsrTileSweepKernel<T, W, kSweepFusedScalar, kTargets>),
                pairs ? &fused_pairs_ : &fused_scalar_, &available));
          }
          const bool scalar_mode = scalar_mode_wanted && available;
          if (!scalar_mode) {
            TF_RETURN_IF_ERROR(SweepKernelAvailable<kTargets>(
                reinterpret_cast<const void*>(CsrTileSweepKernel<T, W, kSweepFused, kTargets>),
                &fused_, &available));
            if (!available) return OkStatus();
          }
          SweepCsr<W> graph, external_graph;
          TF_RETURN_IF_ERROR(PrepareSweepCsr<W, kTargets>(
              context, recurrent.weights, recurrent.post_ids, recurrent.synapse_types,
              recurrent.row_splits, n_post, &recurrent_, &graph));
          if (!scalar_mode) {
            TF_RETURN_IF_ERROR(PrepareSweepCsr<W, kTargets>(
                context, external.weights, external.post_ids, external.synapse_types,
                external.row_splits, n_post, &external_, &external_graph));
          }
          SweepRows<T> rows, external_rows;
          Tensor scratch, external_scratch;
          TF_RETURN_IF_ERROR(
              PrepareSweepRows<T>(context, spikes, carried, newest, &scratch, &rows));
          TF_RETURN_IF_ERROR(PrepareSweepRows<T>(context, activity, QueueRecords{}, nullptr,
                                                 &external_scratch, &external_rows));
          launched = true;
          if (scalar_mode && pairs) {
            return LaunchSweepKernel<T, W, kSweepFusedPairs, kTargets>(
                context, spikes.batch, n_post, basis, output, false, rows, graph,
                external_rows, external_graph,
                ScalarView<W>(*scalar, external.weights, pair_weights));
          }
          if (scalar_mode) {
            return LaunchSweepKernel<T, W, kSweepFusedScalar, kTargets>(
                context, spikes.batch, n_post, basis, output, false, rows, graph,
                external_rows, external_graph, ScalarView<W>(*scalar, external.weights));
          }
          return LaunchSweepKernel<T, W, kSweepFused, kTargets>(
              context, spikes.batch, n_post, basis, output, false, rows, graph,
              external_rows, external_graph);
        }));
        if (launched) return OkStatus();
      }
    }
    TF_RETURN_IF_ERROR(LaunchForward<T, W, kBasis>(
        context, spikes, recurrent.weights, recurrent.post_ids, recurrent.synapse_types,
        recurrent.row_splits, recurrent.edge_ids, basis, n_post, recurrent.aggregate_runs,
        output, true, &recurrent_, carried, newest));
    return LaunchForward<T, W, kBasis>(
        context, activity, external.weights, external.post_ids, external.synapse_types,
        external.row_splits, external.edge_ids, basis, n_post, external.aggregate_runs,
        output, false, &external_);
  }

  // Each connectivity's tables and kernel flags, and the fused kernels' flags
  // (fused_scalar_ also guards the scalar tables, one per tile width).
  ForwardTileCache recurrent_;
  ForwardTileCache external_;
  ForwardTileCache fused_;
  ForwardTileCache fused_scalar_;
  ForwardTileCache fused_pairs_;
  std::shared_ptr<const ScalarExternalTable> scalar_[2];
  std::shared_ptr<const ScalarExternalTable> pairs_[2];
};

// FusedForward over connectivities passed as tensors.
template <typename T, typename W>
class V1CsrForwardFusedOp : public OpKernel {
 public:
  explicit V1CsrForwardFusedOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("n_post", &n_post_));
    OP_REQUIRES_OK(context, context->GetAttr("aggregate_runs", &aggregate_runs_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("external_aggregate_runs", &external_aggregate_runs_));
  }

  void Compute(OpKernelContext* context) override {
    OpInputList spike_slots;
    OP_REQUIRES_OK(context, context->input_list("spikes", &spike_slots));
    SpikeMatrix<const T*> spikes;
    OP_REQUIRES_OK(context, ReadSpikeSlots<T>(spike_slots, &spikes));
    // The named inputs follow the spike list; the queue records come last.
    const int first = spike_slots.size() - 1;
    const ForwardCsr recurrent{context->input(first + 1), context->input(first + 2),
                               context->input(first + 3), context->input(first + 4),
                               context->input(first + 5), aggregate_runs_};
    const Tensor& basis = context->input(first + 6);
    const Tensor& activity = context->input(first + 7);
    const ForwardCsr external{context->input(first + 8), context->input(first + 9),
                              context->input(first + 10), context->input(first + 11),
                              context->input(first + 12), external_aggregate_runs_};
    OP_REQUIRES(context, basis.dims() == 2 && basis.dim_size(1) > 0,
                errors::InvalidArgument("basis must be [n_types,n_basis], n_basis > 0"));
    OP_REQUIRES(context, recurrent.row_splits.NumElements() == spikes.n_pre() + 1,
                errors::InvalidArgument("row_splits does not match spike width"));
    OP_REQUIRES_OK(context, ValidateEdgeArrays(recurrent.weights, recurrent.post_ids,
                                               recurrent.synapse_types, recurrent.edge_ids));
    OP_REQUIRES(context,
                activity.dims() == 2 &&
                    external.row_splits.NumElements() == activity.dim_size(1) + 1,
                errors::InvalidArgument("external_row_splits does not match activity width"));
    OP_REQUIRES_OK(context, ValidateEdgeArrays(external.weights, external.post_ids,
                                               external.synapse_types, external.edge_ids));
    const Tensor& pair_weights = context->input(first + 13);
    OP_REQUIRES_OK(context, fused_.Compute(context, spikes, recurrent, activity, external,
                                           basis, first + 14, n_post_, &pair_weights));
  }

 private:
  int n_post_;
  bool aggregate_runs_ = false;
  bool external_aggregate_runs_ = true;
  FusedForward<T, W> fused_;
};

// The backward's optional `activity_masks` input list: empty, or one
// [width, ceil(batch / 32)] uint32 tensor per spike slot.
template <typename T>
Status ReadActivityMasks(OpKernelContext* context, const SpikeMatrix<const T*>& spikes,
                         ActivityMasks* masks) {
  OpInputList list;
  TF_RETURN_IF_ERROR(context->input_list("activity_masks", &list));
  masks->words = 0;
  if (list.size() == 0) return OkStatus();
  const int64_t words = (spikes.batch + 31) / 32;
  if (list.size() != spikes.slots.count) {
    return errors::InvalidArgument("activity_masks needs one tensor per spike slot, got ",
                                   list.size(), " for ", spikes.slots.count);
  }
  for (int k = 0; k < list.size(); ++k) {
    if (list[k].dims() != 2 || list[k].dim_size(0) != spikes.slots.width ||
        list[k].dim_size(1) != words) {
      return errors::InvalidArgument("activity_masks[", k, "] must be [", spikes.slots.width,
                                     ", ", words, "]");
    }
    masks->slot[k] = reinterpret_cast<const unsigned int*>(list[k].tensor_data().data());
  }
  masks->words = words > 0 ? static_cast<int>(words) : 0;
  return OkStatus();
}

// The backward's optional `spike_grad_addends` list: empty, or one tensor per
// spike slot, each that slot's shape or empty (no addend).
template <typename T>
Status ReadSpikeGradAddends(OpKernelContext* context, const OpInputList& spikes,
                            SpikeSlots<const T*>* addends) {
  OpInputList list;
  TF_RETURN_IF_ERROR(context->input_list("spike_grad_addends", &list));
  addends->count = 0;
  if (list.size() == 0) return OkStatus();
  if (list.size() != spikes.size()) {
    return errors::InvalidArgument("spike_grad_addends needs one tensor per spike slot, got ",
                                   list.size(), " for ", spikes.size());
  }
  for (int k = 0; k < list.size(); ++k) {
    if (list[k].NumElements() == 0) {
      addends->slot[k] = nullptr;
    } else if (list[k].shape() == spikes[k].shape()) {
      addends->slot[k] = list[k].flat<T>().data();
    } else {
      return errors::InvalidArgument("spike_grad_addends[", k, "] must be empty or ",
                                     spikes[k].shape().DebugString());
    }
  }
  addends->count = list.size();
  return OkStatus();
}

// The recurrent backward's optional inputs (see V1CsrBackwardPairProjected),
// read alike by the tensor and the resource ops.
template <typename T>
struct BackwardOptions {
  ActivityMasks masks{};
  const unsigned int* abs_max_bits = nullptr;
  SpikeSlots<const T*> addends{};

  const ActivityMasks* carried_masks() const { return masks.words > 0 ? &masks : nullptr; }
};

template <typename T>
Status ReadBackwardOptions(OpKernelContext* context, const OpInputList& spike_slots,
                           const SpikeMatrix<const T*>& spikes, BackwardOptions<T>* options) {
  TF_RETURN_IF_ERROR(ReadActivityMasks(context, spikes, &options->masks));
  OpInputList bound;
  TF_RETURN_IF_ERROR(context->input_list("abs_max_bound", &bound));
  if (bound.size() > 1 || (bound.size() == 1 && bound[0].NumElements() != 1)) {
    return errors::InvalidArgument("abs_max_bound must be empty or one uint32");
  }
  options->abs_max_bits =
      bound.size() == 0 ? nullptr
                        : reinterpret_cast<const unsigned int*>(bound[0].tensor_data().data());
  return ReadSpikeGradAddends<T>(context, spike_slots, &options->addends);
}

template <typename T>
__global__ void OverwriteEntriesKernel(int64_t count, int64_t size,
                                       const int64_t* __restrict__ indices,
                                       const T* __restrict__ values, T* __restrict__ output) {
  const int64_t k = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (k >= count) return;
  const int64_t index = indices[k];
  if (index >= 0 && index < size) output[index] = values[k];
}

// See V1OverwriteEntries.
template <typename T>
class V1OverwriteEntriesOp : public OpKernel {
 public:
  explicit V1OverwriteEntriesOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& base = context->input(0);
    const Tensor& indices = context->input(1);
    const Tensor& values = context->input(2);
    OP_REQUIRES(context, indices.dims() == 1 && values.shape() == indices.shape(),
                errors::InvalidArgument("indices and values must be equal-length vectors"));
    Tensor* output;
    OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                {0}, 0, base.shape(), &output));
    const cudaStream_t stream = context->eigen_device<GPUDevice>().stream();
    if (output->flat<T>().data() != base.flat<T>().data() && base.NumElements() > 0) {
      OP_REQUIRES(context,
                  cudaMemcpyAsync(output->flat<T>().data(), base.flat<T>().data(),
                                  base.TotalBytes(), cudaMemcpyDeviceToDevice,
                                  stream) == cudaSuccess,
                  errors::Internal("V1OverwriteEntries: copying base failed"));
    }
    const int64_t count = indices.NumElements();
    if (count == 0) return;
    OP_REQUIRES_OK(context,
                   GpuLaunchKernel(OverwriteEntriesKernel<T>,
                                   static_cast<int>((count + kEventThreads - 1) / kEventThreads),
                                   kEventThreads, 0, stream, count, base.NumElements(),
                                   indices.flat<int64_t>().data(), values.flat<T>().data(),
                                   output->flat<T>().data()));
  }
};

// One spike slot's activity masks (see ActivityMasks).
template <typename T>
class V1SpikeActivityMasksOp : public OpKernel {
 public:
  explicit V1SpikeActivityMasksOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& spikes = context->input(0);
    OP_REQUIRES(context, spikes.dims() == 2,
                errors::InvalidArgument("spikes must be a rank-two slot"));
    const int64_t batch = spikes.dim_size(0), width = spikes.dim_size(1);
    const int64_t words = (batch + 31) / 32;
    Tensor* masks;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({width, words}), &masks));
    if (masks->NumElements() == 0) return;
    // Two columns per thread when the width allows the vector load.
    const bool paired = width % 2 == 0;
    const int64_t threads = paired ? width / 2 : width;
    OP_REQUIRES_OK(context,
                   GpuLaunchKernel(paired ? SpikeActivityMasksKernel<T, 2>
                                          : SpikeActivityMasksKernel<T, 1>,
                                   static_cast<int>((threads + kEventThreads - 1) / kEventThreads),
                                   kEventThreads, 0, context->eigen_device<GPUDevice>().stream(),
                                   batch, width, static_cast<int>(words), spikes.flat<T>().data(),
                                   reinterpret_cast<unsigned int*>(masks->flat<uint32>().data())));
  }
};

template <typename T, typename W, bool kAccumulate = false>
class V1CsrBackwardPairProjectedOp : public OpKernel {
 public:
  explicit V1CsrBackwardPairProjectedOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("n_post", &n_post_));
    OP_REQUIRES_OK(context, context->GetAttr("n_edges", &n_edges_));
  }

  void Compute(OpKernelContext* context) override {
    OpInputList spike_slots;
    OP_REQUIRES_OK(context, context->input_list("spikes", &spike_slots));
    SpikeMatrix<const T*> spikes;
    OP_REQUIRES_OK(context, ReadSpikeSlots<T>(spike_slots, &spikes));
    // The named inputs follow the spike list.
    const int first = spike_slots.size() - 1;
    const Tensor& current_grad = context->input(first + 1);
    const Tensor& weights = context->input(first + 2);
    const Tensor& post_ids = context->input(first + 3);
    const Tensor& synapse_types = context->input(first + 4);
    const Tensor& row_splits = context->input(first + 5);
    const Tensor& edge_ids = context->input(first + 6);
    const Tensor& nonempty_rows = context->input(first + 7);
    const Tensor& basis = context->input(first + 8);
    const Tensor& dampening = context->input(first + 9);
    const Tensor& pair_ids = context->input(first + 10);
    const Tensor& pair_posts = context->input(first + 11);
    const Tensor& pair_types = context->input(first + 12);
    OP_REQUIRES(context, basis.dims() == 2 && basis.dim_size(1) > 0,
                errors::InvalidArgument("basis dimension must be positive"));
    OP_REQUIRES(context, row_splits.NumElements() == spikes.n_pre() + 1,
                errors::InvalidArgument("row_splits does not match spike width"));
    OP_REQUIRES(context,
                current_grad.dims() == 2 &&
                    current_grad.dim_size(0) == spikes.batch * n_post_ &&
                    current_grad.dim_size(1) == basis.dim_size(1),
                errors::InvalidArgument("current_grad has an incompatible shape"));
    OP_REQUIRES(context, dampening.NumElements() == 1,
                errors::InvalidArgument("dampening must be scalar"));
    OP_REQUIRES_OK(context, ValidateEdgeArrays(weights, post_ids, synapse_types, edge_ids));
    OP_REQUIRES(context, n_edges_ == post_ids.NumElements(),
                errors::InvalidArgument("n_edges does not match post_ids"));
    OP_REQUIRES(context, pair_ids.NumElements() == post_ids.NumElements(),
                errors::InvalidArgument("pair_ids must align with CSR edges"));
    OP_REQUIRES(context, pair_posts.NumElements() == pair_types.NumElements(),
                errors::InvalidArgument("pair metadata lengths differ"));
    BackwardOptions<T> options;
    OP_REQUIRES_OK(context, ReadBackwardOptions<T>(context, spike_slots, spikes, &options));
    SpikeSlots<T*> spike_grad;
    OP_REQUIRES_OK(context, AllocateSpikeGradients<T>(context, spike_slots, spikes, &spike_grad));
#define LAUNCH_BACKWARD(BASIS)                                                     \
  LaunchPairProjectedBackward<T, W, BASIS>(                                        \
      context, spikes, current_grad, weights, post_ids, synapse_types,            \
      row_splits, edge_ids, nonempty_rows, basis, dampening, pair_ids,            \
      pair_posts, pair_types, n_post_, spike_grad, weight_grad, kAccumulate,      \
      options.carried_masks(), options.abs_max_bits, options.addends)
    OP_REQUIRES_OK(context, WithWeightGradBuffer<kAccumulate>(
                                context, first + 13, n_edges_, [&](Tensor* weight_grad) {
                                  return basis.dim_size(1) == 4 ? LAUNCH_BACKWARD(4)
                                                                : LAUNCH_BACKWARD(0);
                                }));
#undef LAUNCH_BACKWARD
  }

 private:
  int n_post_;
  int n_edges_;
};

// sums[p] = the sum of the weights of pair p's edges [pair_edges[p],
// pair_edges[p + 1]) (CSR order: DIRECT_CSR weights), accumulated in FP64 and
// rounded once to FP32 (once per rollout, so the FP64 adds cost nothing): an
// FP32 accumulation over the heaviest pairs (up to 1,260 edges) left the
// currents slightly farther from fp64 than the per-run sweep.
__global__ void PairWeightSumsKernel(int64_t pairs, const unsigned int* pair_edges,
                                     const float* weights, float* sums) {
  for (int64_t pair = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; pair < pairs;
       pair += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const unsigned int begin = pair_edges[pair], end = pair_edges[pair + 1];
    // One or two edges (94 % of the pairs): the FP32 sum is already the
    // correctly rounded one; FP64 is slow on this card.
    if (end - begin <= 2) {
      sums[pair] = end - begin == 1 ? weights[begin] : weights[begin] + weights[begin + 1];
      continue;
    }
    double sum = 0.0;
    for (unsigned int edge = begin; edge < end; ++edge) sum += weights[edge];
    sums[pair] = static_cast<float>(sum);
  }
}

// An external connectivity's per-pair weight sums for the coalesced sweep of
// V1CsrForwardFused (V1ExternalPairWeights) and V1CsrForwardFusedResource
// (V1ExternalPairWeightsResource): one FP32 sum per distinct (row, target)
// pair, in the order of the pair table, or an empty vector (output 0) where
// that sweep does not apply (no DIRECT_CSR, or a target reached through two
// synapse types). Weights stay constant over a rollout, so a caller computes
// these once per rollout. Each op owns one: the table lives here, never in a
// static (both libraries compile this source).
class PairWeightSums {
 public:
  Status Compute(OpKernelContext* context, const Tensor& weights, const Tensor& post_ids,
                 const Tensor& synapse_types, const Tensor& row_splits, int n_post) {
    if (weights.NumElements() != post_ids.NumElements()) {
      return errors::InvalidArgument("weights and post_ids must have one entry per edge");
    }
    std::shared_ptr<const ScalarExternalTable> table;
    {
      mutex_lock lock(mu_);
      TF_RETURN_IF_ERROR(EnsureScalarExternal<kSweepTargets>(
          context, post_ids, row_splits, synapse_types, n_post, true, &table_));
      table = table_;
    }
    const int64_t pairs = table->usable ? table->pair_edges.NumElements() - 1 : 0;
    Tensor* output;
    TF_RETURN_IF_ERROR(context->allocate_output(0, TensorShape({pairs}), &output));
    if (pairs == 0) return OkStatus();
    return GpuLaunchKernel(
        PairWeightSumsKernel, 1024, 256, 0, context->eigen_device<GPUDevice>().stream(), pairs,
        reinterpret_cast<const unsigned int*>(table->pair_edges.tensor_data().data()),
        weights.flat<float>().data(), output->flat<float>().data());
  }

 private:
  mutex mu_;
  std::shared_ptr<const ScalarExternalTable> table_;
};

class V1ExternalPairWeightsOp : public OpKernel {
 public:
  explicit V1ExternalPairWeightsOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("n_post", &n_post_));
  }

  void Compute(OpKernelContext* context) override {
    OP_REQUIRES_OK(context, sums_.Compute(context, context->input(0), context->input(1),
                                          context->input(2), context->input(3), n_post_));
  }

 private:
  int n_post_;
  PairWeightSums sums_;
};

// One slot's queue record (see QueueRecordWords), for the first step of a
// carried spike history.
template <typename T>
class V1SpikeQueueOp : public OpKernel {
 public:
  explicit V1SpikeQueueOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& spikes = context->input(0);
    OP_REQUIRES(context, spikes.dims() == 2 && spikes.NumElements() < kQueueMaxSlots,
                errors::InvalidArgument("spikes must be a rank-two slot of fewer than 2^31 entries"));
    const int64_t words = QueueRecordWords(spikes.dim_size(0), spikes.dim_size(1));
    Tensor* record;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({words}), &record));
    unsigned int* data = reinterpret_cast<unsigned int*>(record->flat<uint32>().data());
    if (spikes.NumElements() == 0) {
      cudaMemsetAsync(data, 0, words * sizeof(unsigned int),
                      context->eigen_device<GPUDevice>().stream());
      return;
    }
    OP_REQUIRES_OK(context, BuildQueueRecord<T>(context, spikes.flat<T>().data(),
                                                spikes.dim_size(0), spikes.dim_size(1), data));
  }
};

#ifndef V1_KERNEL_IMPLEMENTATION_ONLY
#define REGISTER_TYPE(T)                                                   \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("V1CsrForward").Device(DEVICE_GPU).TypeConstraint<T>("T"),       \
      V1CsrForwardOp<T, float>);                                           \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("V1CsrForwardFused").Device(DEVICE_GPU).TypeConstraint<T>("T"),  \
      V1CsrForwardFusedOp<T, float>);                                      \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("V1SpikeQueue").Device(DEVICE_GPU).TypeConstraint<T>("T"),       \
      V1SpikeQueueOp<T>);                                                  \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("V1SpikeActivityMasks").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      V1SpikeActivityMasksOp<T>);                                          \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("V1OverwriteEntries").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      V1OverwriteEntriesOp<T>);                                            \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("V1CsrBackwardPairProjected")                                    \
          .Device(DEVICE_GPU).TypeConstraint<T>("T"),                      \
      V1CsrBackwardPairProjectedOp<T, float>);                              \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("V1CsrBackwardPairProjectedAccumulate")                          \
          .Device(DEVICE_GPU).TypeConstraint<T>("T")                       \
          .HostMemory("accumulator"),                                      \
      V1CsrBackwardPairProjectedOp<T, float, true>);

TF_CALL_half(REGISTER_TYPE);
TF_CALL_float(REGISTER_TYPE);
#undef REGISTER_TYPE
REGISTER_KERNEL_BUILDER(Name("V1ExternalPairWeights").Device(DEVICE_GPU),
                        V1ExternalPairWeightsOp);
#endif

#endif

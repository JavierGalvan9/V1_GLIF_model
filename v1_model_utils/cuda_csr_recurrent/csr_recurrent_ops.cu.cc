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

// The queue of active (batch, row) slots of the concatenated [batch, n_pre]
// matrix, in its row-major order. With several slot tensors, a single
// compaction over the concatenated index would locate every element's tensor
// with two divisions, which made the sweep compute-bound (132 us against 50 at
// B = 64); each tensor is compacted with the plain streaming select instead,
// and the few active entries are merged into the concatenated order.
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

// With several slots, `carried.record[d]` is slot d's record when the caller
// has it (null otherwise) and `newest` receives slot 0's (null: a temporary).
template <typename T>
Status BuildActiveQueue(OpKernelContext* context, const SpikeMatrix<const T*>& spikes,
                        unsigned int* queue, unsigned int* queue_count,
                        const QueueRecords& carried, unsigned int* newest) {
  const SpikeSlots<const T*>& slots = spikes.slots;
  if (slots.count == 1) {
    return SelectNonzero<T>(context, slots.slot[0], spikes.NumElements(), queue, queue_count);
  }
  const int64_t words = QueueRecordWords(spikes.batch, slots.width);
  int missing = 0;
  for (int slot = 0; slot < slots.count; ++slot) {
    missing += carried.record[slot] == nullptr && !(slot == 0 && newest != nullptr);
  }
  Tensor scratch;
  unsigned int* next = nullptr;
  if (missing) TF_RETURN_IF_ERROR(AllocateQueueWords(context, missing * words, &scratch, &next));
  QueueRecords records = carried;
  for (int slot = 0; slot < slots.count; ++slot) {
    if (records.record[slot] != nullptr) continue;
    unsigned int* record = slot == 0 && newest != nullptr ? newest : next;
    if (record == next) next += words;
    TF_RETURN_IF_ERROR(BuildQueueRecord<T>(context, slots.slot[slot], spikes.batch,
                                           slots.width, record));
    records.record[slot] = record;
  }
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
// csr_order.repeats_targets), so they take a plain one-thread-per-edge scatter
// with no warp collectives.
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
      // Invalid lanes take a post no real edge can have, so they form their own
      // singleton run and never merge with a live one.
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
template <int kTargets>
inline Status EnsureTileSegments(OpKernelContext* context, const Tensor& post_ids,
                                 const Tensor& row_splits, int n_post,
                                 ForwardTileCache* cache) {
  if (cache->segments &&
      MatchesTileSegments(*cache->segments, context, post_ids, row_splits, n_post)) {
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
        cache->segments = std::move(table);
        return OkStatus();
      }
      ++it;
    }
  }
  auto table = std::make_shared<ForwardTileSegments>();
  TF_RETURN_IF_ERROR(BuildTileSegments<kTargets>(
      context, post_ids, row_splits, n_post, table.get()));
  registry.tables.emplace_back(table);
  cache->segments = std::move(table);
  return OkStatus();
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
    // One thread per active row: binary-search the row's segment in this tile.
    const unsigned int index = base + threadIdx.x;
    int item = -1;
    uint32 length = 0;
    if (index < end) {
      const unsigned int slot = queue[index];
      const unsigned int row = slot - sample_first;
      unsigned int low = segment_splits[row];
      unsigned int last = 0;
      bool found;
      if (segment_masks != nullptr) {
        // The mask names the row's tiles; the segment's rank is a popcount.
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
        const uint32 start = segment_starts[low];
        const uint32 stop = low + 1 < last ? segment_starts[low + 1] : row_splits[row + 1];
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
  T* output = currents + (sample * n_post + first_target) * 4;
  for (int target = threadIdx.x; target < targets; target += kTileThreads) {
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

// Check the kernel's actual static allocation as well as the device's opt-in
// dynamic allowance before selecting the tiled path. Unsupported resources
// use scatter; API failures unrelated to resource limits remain visible.
template <typename T, typename W>
Status TiledForwardAvailable(ForwardTileCache* cache, bool* available) {
  mutex_lock lock(cache->mu);
  if (!cache->kernel_checked) {
    int ordinal;
    cudaFuncAttributes attributes;
    int shared_limit;
    cudaError_t status = cudaGetDevice(&ordinal);
    if (status == cudaSuccess)
      status = cudaFuncGetAttributes(&attributes, TiledForwardKernel<T, W>);
    if (status == cudaSuccess)
      status = cudaDeviceGetAttribute(&shared_limit,
                                      cudaDevAttrMaxSharedMemoryPerBlockOptin, ordinal);
    if (status != cudaSuccess)
      return errors::Internal("checking tiled forward resources failed: ",
                              cudaGetErrorString(status));
    constexpr size_t kShared = kTileTargets * sizeof(::float4);
    cache->kernel_available =
        kTileThreads <= attributes.maxThreadsPerBlock &&
        attributes.sharedSizeBytes + kShared <= static_cast<size_t>(shared_limit);
    if (cache->kernel_available) {
      status = cudaFuncSetAttribute(TiledForwardKernel<T, W>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, kShared);
      if (status == cudaErrorInvalidValue || status == cudaErrorNotSupported) {
        // GpuLaunchKernel checks the thread's last CUDA error. Clear only the
        // configuration error being handled; report any different pending error.
        const cudaError_t pending = cudaGetLastError();
        if (pending != cudaSuccess && pending != status)
          return errors::Internal("configuring tiled forward left another CUDA error: ",
                                  cudaGetErrorString(pending));
        cache->kernel_available = false;
      } else if (status != cudaSuccess) {
        return errors::Internal("configuring the tiled forward failed: ",
                                cudaGetErrorString(status));
      }
    }
    cache->kernel_checked = true;
  }
  *available = cache->kernel_available;
  return OkStatus();
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
                                          tile_segments));
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

// Largest finite |current_grad|, as raw float bits in a uint32. Non-negative
// floats order identically to their bit patterns, so an integer atomicMax is a
// float max. Non-finite entries are skipped on purpose: the backward
// legitimately propagates NaNs from `current_grad`, and letting one NaN poison
// the scale would turn every spike gradient into a NaN instead of only the
// affected ones.
template <typename T>
__global__ void AbsMaxFiniteKernel(int64_t elements, const T* values,
                                   unsigned int* result) {
  float local = 0.0f;
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < elements; index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const float value = fabsf(AsFloat(values[index]));
    if (isfinite(value) && value > local) local = value;
  }
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    local = fmaxf(local, __shfl_xor_sync(0xffffffff, local, mask));
  }
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
__global__ void ProjectionScaleKernel(const unsigned int* max_bits, const float* basis,
                                      int n_types, int n_basis, float* scale_out) {
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
  if (threadIdx.x != 0) return;
  const float bound = __uint_as_float(*max_bits) * basis_l1;
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
    uint32 sentinel_pair) {
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
    for (int sample = threadIdx.x >> 5; sample < kSlice; sample += kTileWarps) {
      const int64_t b = batch_base + sample;
      if (b >= batch) continue;
      const T value = result[row][sample];
      // A select rather than an add of +0, which fast math may fold away.
      column[b * spike_grad.width] =
          oldest && AsFloat(value) == 0.0f ? FromFloat<T>(0.0f) : value;
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
    int n_post, const SpikeSlots<T*>& spike_grad, Tensor* weight_grad, bool accumulate) {
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
    unsigned int* max_bits = reinterpret_cast<unsigned int*>(scale + 2);
    cudaMemsetAsync(max_bits, 0, sizeof(unsigned int), device.stream());
    TF_RETURN_IF_ERROR(GpuLaunchKernel(
        AbsMaxFiniteKernel<T>, 1024, 256, 0, device.stream(),
        current_grad.NumElements(), current_grad.flat<T>().data(), max_bits));
    TF_RETURN_IF_ERROR(GpuLaunchKernel(
        ProjectionScaleKernel, 1, 32, 0, device.stream(), max_bits,
        basis.flat<float>().data(), static_cast<int>(basis.dim_size(0)), n_basis, scale));
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
      static_cast<uint32>(n_pairs)))
  if (n_slices > 1) {
    LAUNCH_ROWS(true);
  } else {
    LAUNCH_ROWS(false);
  }
#undef LAUNCH_ROWS
  return LaunchEventWeightGrad<T, kBasis>(context, spikes, current_grad, basis,
                                          post_ids, synapse_types, row_splits,
                                          edge_ids, n_post, weight_grad, accumulate);
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
    bool accumulate = false) {
  const int64_t batch = spikes.batch;
  if (batch == 0 || nonempty_rows.NumElements() == 0) {
    auto stream = context->eigen_device<GPUDevice>().stream();
    for (int k = 0; k < spike_grad.count; ++k) {
      cudaMemsetAsync(spike_grad.slot[k], 0, batch * spike_grad.width * sizeof(T), stream);
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
      pair_posts, pair_types, n_post, spike_grad, weight_grad, accumulate)
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
    OP_REQUIRES(context, pair_ids.NumElements() == post_ids.NumElements(),
                errors::InvalidArgument("pair_ids must align with CSR edges"));
    OP_REQUIRES(context, pair_posts.NumElements() == pair_types.NumElements(),
                errors::InvalidArgument("pair metadata lengths differ"));
    SpikeSlots<T*> spike_grad;
    OP_REQUIRES_OK(context, AllocateSpikeGradients<T>(context, spike_slots, spikes, &spike_grad));
#define LAUNCH_BACKWARD(BASIS)                                                     \
  LaunchPairProjectedBackward<T, W, BASIS>(                                        \
      context, spikes, current_grad, weights, post_ids, synapse_types,            \
      row_splits, edge_ids, nonempty_rows, basis, dampening, pair_ids,            \
      pair_posts, pair_types, n_post_, spike_grad, weight_grad, kAccumulate)
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
      Name("V1SpikeQueue").Device(DEVICE_GPU).TypeConstraint<T>("T"),       \
      V1SpikeQueueOp<T>);                                                  \
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
#endif

#endif

#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include <cub/device/device_select.cuh>
#include <cub/iterator/counting_input_iterator.cuh>
#include <cub/iterator/transform_input_iterator.cuh>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <limits>
#include <type_traits>

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

// The forward's device-built active-row queue packs one (batch, presynaptic
// row) entry as `batch << row_bits | row`. Choose the row width from the
// network's presynaptic row count, leaving the remaining bits for the batch.
// The entries are plain `unsigned int`, not `uint32`: the
// resource library compiles this file with `uint32` redefined as a signed type,
// and a signed shift would corrupt batch indices of 1024 and above.
constexpr int kQueueWordBits = std::numeric_limits<unsigned int>::digits;
// Backward row kernel: one CSR row per warp, four warps per block.
constexpr int kBackwardRowsPerBlock = 4;

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
struct PackActiveSlot {
  int64_t n_pre;
  int row_bits;
  __host__ __device__ unsigned int operator()(int64_t index) const {
    return static_cast<unsigned int>(((index / n_pre) << row_bits) | (index % n_pre));
  }
};

template <typename T>
struct SlotIsActive {
  const T* spikes;
  __host__ __device__ bool operator()(int64_t index) const {
    return static_cast<float>(spikes[index]) != 0.0f;
  }
};

template <typename T>
Status BuildActiveQueue(OpKernelContext* context, const T* spikes, int64_t slots,
                        int64_t n_pre, int row_bits, unsigned int* queue,
                        unsigned int* queue_count) {
  const cub::CountingInputIterator<int64_t> indices(0);
  const cub::TransformInputIterator<unsigned int, PackActiveSlot,
                                    cub::CountingInputIterator<int64_t>>
      packed(indices, PackActiveSlot{n_pre, row_bits});
  const cub::TransformInputIterator<bool, SlotIsActive<T>,
                                    cub::CountingInputIterator<int64_t>>
      active(indices, SlotIsActive<T>{spikes});
  auto stream = context->eigen_device<GPUDevice>().stream();
  size_t scratch_bytes = 0;
  cub::DeviceSelect::Flagged(nullptr, scratch_bytes, packed, active, queue, queue_count,
                             slots, stream);
  Tensor scratch;
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DT_INT8, TensorShape({static_cast<int64_t>(scratch_bytes)}), &scratch));
  if (cub::DeviceSelect::Flagged(scratch.flat<int8>().data(), scratch_bytes, packed, active,
                                 queue, queue_count, slots, stream) != cudaSuccess) {
    return errors::Internal("building the active-row queue failed");
  }
  return OkStatus();
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
    int64_t n_pre, int n_post, int n_basis, int row_bits, const T* spikes,
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
    const unsigned int packed = queue[active_id];
    const int64_t batch = packed >> row_bits;
    const int64_t pre = packed & ((1u << row_bits) - 1);
    const float spike = AsFloat(spikes[batch * n_pre + pre]);
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
          AtomicAddPair(output, weighted * bv.x, weighted * bv.y);
          AtomicAddPair(output + 2, weighted * bv.z, weighted * bv.w);
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
          AtomicAddPair(output, value[0], value[1]);
          AtomicAddPair(output + 2, value[2], value[3]);
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

template <typename T, typename W, int kBasis>
Status LaunchForward(OpKernelContext* context, const Tensor& spikes,
                     const Tensor& weights, const Tensor& post_ids,
                     const Tensor& synapse_types, const Tensor& row_splits,
                     const Tensor& edge_ids, const Tensor& basis, int n_post,
                     bool aggregate_runs, Tensor* output) {
  const int64_t batch = spikes.dim_size(0);
  const int64_t n_pre = spikes.dim_size(1);
  const int64_t slots = spikes.NumElements();
  if (slots == 0) return OkStatus();
  int row_bits = 1;
  while ((uint64_t{1} << row_bits) < static_cast<uint64_t>(n_pre)) ++row_bits;
  if (row_bits >= kQueueWordBits ||
      batch > (uint64_t{1} << (kQueueWordBits - row_bits)) ||
      slots > std::numeric_limits<unsigned int>::max()) {
    return errors::InvalidArgument(
        "the active-row queue cannot encode ", n_pre, " rows and a batch of ",
        batch, " in ", kQueueWordBits, " bits");
  }
  const int n_basis = basis.dim_size(1);
  auto device = context->eigen_device<GPUDevice>();
  // One slot per (batch, row): 78 MiB at batch 32 on the 203,816-neuron
  // network, and a temp, so it lives only for the op.
  Tensor queue_tensor;
  unsigned int* queue;
  TF_RETURN_IF_ERROR(AllocateQueueWords(context, slots + 2, &queue_tensor, &queue));
  unsigned int* queue_count = queue + slots;   // then the consumers' ticket
  cudaMemsetAsync(queue_count, 0, 2 * sizeof(unsigned int), device.stream());
  TF_RETURN_IF_ERROR(BuildActiveQueue<T>(context, spikes.flat<T>().data(), slots, n_pre,
                                         row_bits, queue, queue_count));
  // About 1,024 edges of work per ticket; see ForwardKernel.
  const int64_t mean_edges = std::max<int64_t>(1, post_ids.NumElements() / n_pre);
  const unsigned int slots_per_ticket =
      static_cast<unsigned int>(std::clamp<int64_t>(1024 / mean_edges, 1, 8));
#define LAUNCH_FORWARD_WITH(OUTPUT_TYPE, OUTPUT_PTR, AGGREGATE)                \
  TF_RETURN_IF_ERROR(GpuLaunchKernel(                                          \
      ForwardKernel<T, W, OUTPUT_TYPE, kBasis, AGGREGATE>, kEventBlocks,       \
      V1_FORWARD_THREADS, 0, device.stream(), n_pre, n_post, n_basis,          \
      row_bits,                                                                 \
      spikes.flat<T>().data(), queue, queue_count, queue_count + 1,            \
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
// The batch is processed in slices of kSlice = min(32, next power of two)
// samples, the projection is laid out [pair, batch_stride] with batch_stride a
// whole number of slices, and one grid row of the SpMM runs per slice. Padding
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

// Compact projection for one batch slice, emitted as [pair, batch_stride] so
// that a warp reading one pair's slice line touches a single cache line.
//
// Writing that layout directly makes consecutive threads take consecutive batch
// samples of one pair, whose current gradients are `n_post * n_basis` apart --
// 1.6 MiB on the 203,816-neuron network -- so a warp fetches 1 KiB to use
// 256 B. Pairs are sorted by (postsynaptic neuron, synapse type), so thirty-two
// consecutive pairs touch about four neurons: reading pair-contiguous and
// transposing through shared memory coalesces both halves. The 34-float row
// keeps the transposed read to two conflict-free halves.
template <typename T, typename P, int kBasis, int kSlice, int kPairsPerTile>
__global__ void PreprojectPairsTiledKernel(
    int n_post, int n_basis, int64_t n_pairs, int64_t batch, int64_t batch_stride,
    const T* current_grad, const float* basis, const uint32* pair_posts,
    const uint8* pair_types, P* projected, const float* scale) {
  const float factor = scale == nullptr ? 1.0f : scale[0];
  constexpr int kTileElements = kPairsPerTile * kSlice;
  __shared__ float tile[kSlice][kPairsPerTile + 2];
  const int64_t pair_base = static_cast<int64_t>(blockIdx.x) * kPairsPerTile;
  const int64_t batch_base = static_cast<int64_t>(blockIdx.y) * kSlice;
  for (int index = threadIdx.x; index < kTileElements; index += blockDim.x) {
    const int column = index % kPairsPerTile;
    const int sample = index / kPairsPerTile;
    const int64_t pair = pair_base + column;
    const int64_t b = batch_base + sample;
    float value = 0.0f;
    if (pair < n_pairs && b < batch) {
      value = BasisProjection<T, kBasis>(
          current_grad + (b * n_post + pair_posts[pair]) * n_basis, basis,
          pair_types[pair], n_basis);
    }
    tile[sample][column] = value;
  }
  __syncthreads();
  for (int index = threadIdx.x; index < kTileElements; index += blockDim.x) {
    const int sample = index % kSlice;
    const int column = index / kSlice;
    const int64_t pair = pair_base + column;
    if (pair < n_pairs) {
      ProjStore(projected + pair * batch_stride + batch_base + sample,
                tile[sample][column] * factor);
    }
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

// Spike gradient for one batch slice: one CSR row per warp, kRows warps per
// block, no shared memory and no barrier, so a 32 * kRows block reaches full
// occupancy.
//
// A lane holds kPack consecutive samples, so an edge needs kSlice / kPack
// lanes, a warp splits into that many independent edge slots, and one load
// serves all of them. The descriptors (`pair_ids`, `weights`) are read once per
// 32-edge tile cooperatively and broadcast with `__shfl_sync`. Out-of-range
// lanes point at `sentinel_pair`, an all-zero projection row, so the hot loop
// loads unconditionally instead of branching and zero-filling.
//
// The result goes to a [pre, batch_stride] scratch buffer, one contiguous line
// per row, and TransposeSpikeGradKernel restores [batch, pre].
template <typename T, typename W, typename P, int kSlice, int kPack, int kRows,
          bool kMultiSlice>
__global__ __launch_bounds__(32 * kRows) void BackwardRowPerWarpKernel(
    const P* projected, int64_t runtime_stride, const W* weights, const uint32* pair_ids,
    const uint32* edge_ids, const uint32* row_splits, const uint32* nonempty_rows,
    int64_t n_rows, const T* dampening, T* spike_grad_t, const float* inverse_scale,
    uint32 sentinel_pair) {
  constexpr int kTile = 32;
  constexpr int kLanesPerEdge = kSlice / kPack;
  constexpr int kSlots = 32 / kLanesPerEdge;
  constexpr int kPerSlot = kTile / kSlots;
  // One slice (batch <= 32) keeps the stride a compile-time constant, so the hot
  // loop's address arithmetic stays a shift.
  const int64_t batch_stride = kMultiSlice ? runtime_stride : kSlice;
  const int lane = threadIdx.x & 31;
  const int slot = lane / kLanesPerEdge;
  const int sub = lane % kLanesPerEdge;
  const int64_t row_id = static_cast<int64_t>(blockIdx.x) * kRows + (threadIdx.x >> 5);
  if (row_id >= n_rows) return;
  const int64_t sample_base =
      (kMultiSlice ? static_cast<int64_t>(blockIdx.y) * kSlice : 0) + sub * kPack;
  const uint32 pre = nonempty_rows[row_id];
  const uint32 end = row_splits[pre + 1];
  float grad[kPack] = {};
  for (uint32 base = row_splits[pre]; base < end; base += kTile) {
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
      for (int sample = 0; sample < kPack; ++sample) grad[sample] += value[sample] * weight;
    }
  }

  // Fold the slots, which each accumulated the same samples from a different
  // edge subset. Afterwards every lane sharing `sub` holds the total, so the
  // kLanesPerEdge lowest lanes cover the slice between them and store it
  // without a shared-memory round trip.
#pragma unroll
  for (int mask = kLanesPerEdge; mask < 32; mask <<= 1) {
#pragma unroll
    for (int sample = 0; sample < kPack; ++sample) {
      grad[sample] += __shfl_xor_sync(0xffffffff, grad[sample], mask);
    }
  }
  if (lane < kLanesPerEdge) {
    const float factor =
        (inverse_scale == nullptr ? 1.0f : inverse_scale[1]) * AsFloat(*dampening);
#pragma unroll
    for (int sample = 0; sample < kPack; ++sample) {
      spike_grad_t[static_cast<int64_t>(pre) * batch_stride + sample_base + sample] =
          FromFloat<T>(grad[sample] * factor);
    }
  }
}

// [pre, batch_stride] -> [batch, pre], one 32-row by kSlice-sample tile per
// block. Each read pass takes one row's slice line, each write pass 32
// consecutive rows of one sample, so both halves are coalesced; the odd row
// stride kSlice + 1 keeps the transposed read conflict-free. Rows without edges
// were never written by the row kernel, so they are emitted as zeros here,
// which is what lets the op skip clearing its output.
template <typename T, int kSlice>
__global__ void TransposeSpikeGradKernel(int64_t n_pre, int64_t batch, int64_t batch_stride,
                                         const T* source, T* destination,
                                         const uint32* row_splits) {
  __shared__ float tile[32][kSlice + 1];
  const int64_t pre_base = static_cast<int64_t>(blockIdx.x) * 32;
  const int64_t batch_base = static_cast<int64_t>(blockIdx.y) * kSlice;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int warps = blockDim.x >> 5;
  for (int index = warp; index < 32; index += warps) {
    const int64_t pre = pre_base + index;
    if (lane < kSlice) {
      const bool written = pre < n_pre && row_splits[pre + 1] != row_splits[pre];
      tile[index][lane] =
          written ? AsFloat(source[pre * batch_stride + batch_base + lane]) : 0.0f;
    }
  }
  __syncthreads();
  const int64_t pre = pre_base + lane;
  for (int sample = warp; sample < kSlice; sample += warps) {
    const int64_t b = batch_base + sample;
    if (pre < n_pre && b < batch) {
      destination[b * n_pre + pre] = FromFloat<T>(tile[lane][sample]);
    }
  }
}

// One slice width's backward. P is the projection element: scaled FP16 with
// FP16 spikes, FP32 with FP32 spikes, which keeps the precision the caller
// chose.
template <typename T, typename W, int kBasis, int kSlice, int kPack>
Status LaunchSlicedPairBackward(
    OpKernelContext* context, const Tensor& spikes, const Tensor& current_grad,
    const Tensor& weights, const Tensor& post_ids, const Tensor& synapse_types,
    const Tensor& row_splits, const Tensor& edge_ids,
    const Tensor& nonempty_rows, const Tensor& basis, const Tensor& dampening,
    const Tensor& pair_ids, const Tensor& pair_posts, const Tensor& pair_types,
    int n_post, Tensor* spike_grad, Tensor* weight_grad, bool accumulate) {
  constexpr bool kScaled = std::is_same<T, Eigen::half>::value;
  using P = typename std::conditional<kScaled, __half, float>::type;
  const int64_t n_rows = nonempty_rows.NumElements();
  const int64_t n_pairs = pair_posts.NumElements();
  const int64_t batch = spikes.dim_size(0);
  const int64_t n_pre = spikes.dim_size(1);
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
  constexpr int kPairsPerTile = 32;
  TF_RETURN_IF_ERROR(GpuLaunchKernel(
      PreprojectPairsTiledKernel<T, P, kBasis, kSlice, kPairsPerTile>,
      dim3(static_cast<unsigned>((n_pairs + kPairsPerTile - 1) / kPairsPerTile),
           static_cast<unsigned>(n_slices)),
      128, 0, device.stream(), n_post, n_basis, n_pairs, batch, batch_stride,
      current_grad.flat<T>().data(), basis.flat<float>().data(),
      pair_posts.flat<uint32>().data(), pair_types.flat<uint8>().data(), projected, scale));
  Tensor spike_grad_t;
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DataTypeToEnum<T>::value, TensorShape({n_pre * batch_stride}), &spike_grad_t));
  const dim3 row_grid(
      static_cast<unsigned>((n_rows + kBackwardRowsPerBlock - 1) / kBackwardRowsPerBlock),
      static_cast<unsigned>(n_slices));
#define LAUNCH_ROWS(MULTI_SLICE)                                                    \
  TF_RETURN_IF_ERROR(GpuLaunchKernel(                                               \
      BackwardRowPerWarpKernel<T, W, P, kSlice, kPack, kBackwardRowsPerBlock,       \
                               MULTI_SLICE>,                                        \
      row_grid, 32 * kBackwardRowsPerBlock, 0, device.stream(), projected,          \
      batch_stride, weights.flat<W>().data(), pair_ids.flat<uint32>().data(),       \
      edge_ids.flat<uint32>().data(), row_splits.flat<uint32>().data(),             \
      nonempty_rows.flat<uint32>().data(), n_rows, dampening.flat<T>().data(),      \
      spike_grad_t.flat<T>().data(), scale, static_cast<uint32>(n_pairs)))
  if (n_slices > 1) {
    LAUNCH_ROWS(true);
  } else {
    LAUNCH_ROWS(false);
  }
#undef LAUNCH_ROWS
  TF_RETURN_IF_ERROR(GpuLaunchKernel(
      TransposeSpikeGradKernel<T, kSlice>,
      dim3(static_cast<unsigned>((n_pre + 31) / 32), static_cast<unsigned>(n_slices)), 256,
      0, device.stream(), n_pre, batch, batch_stride, spike_grad_t.flat<T>().data(),
      spike_grad->flat<T>().data(), row_splits.flat<uint32>().data()));
  return LaunchEventWeightGrad<T, kBasis>(context, spikes, current_grad, basis,
                                          post_ids, synapse_types, row_splits,
                                          edge_ids, n_post, weight_grad, accumulate);
}

// Any batch size, basis dimension and spike dtype. The slice is the next power
// of two up to 32; at eight FP16 samples per lane a 32-sample slice line is
// exactly one 16-byte load, and FP32 packs four. With `accumulate` the weight
// gradient is added to what `weight_grad` already holds instead of replacing it.
template <typename T, typename W, int kBasis>
Status LaunchPairProjectedBackward(
    OpKernelContext* context, const Tensor& spikes, const Tensor& current_grad,
    const Tensor& weights, const Tensor& post_ids, const Tensor& synapse_types,
    const Tensor& row_splits, const Tensor& edge_ids,
    const Tensor& nonempty_rows, const Tensor& basis, const Tensor& dampening,
    const Tensor& pair_ids, const Tensor& pair_posts, const Tensor& pair_types,
    int n_post, Tensor* spike_grad, Tensor* weight_grad, bool accumulate = false) {
  const int64_t batch = spikes.dim_size(0);
  if (batch == 0 || nonempty_rows.NumElements() == 0) {
    auto stream = context->eigen_device<GPUDevice>().stream();
    cudaMemsetAsync(spike_grad->flat<T>().data(), 0, spike_grad->TotalBytes(), stream);
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
  if (batch > 16) LAUNCH_SLICE(32, kWidePack);
  if (batch > 8) LAUNCH_SLICE(16, 4);
  if (batch > 4) LAUNCH_SLICE(8, 4);
  if (batch > 2) LAUNCH_SLICE(4, 4);
  if (batch > 1) LAUNCH_SLICE(2, 2);
  LAUNCH_SLICE(1, 1);
#undef LAUNCH_SLICE
}

// Hands `launch` the buffer the recurrent weight gradient goes to. By default
// that is output 1, which the kernels overwrite. With `kAccumulate` it is the
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
        context->allocate_output(1, TensorShape({n_edges}), &weight_grad));
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

template <typename T, typename W>
class V1CsrForwardOp : public OpKernel {
 public:
  explicit V1CsrForwardOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("n_post", &n_post_));
    OP_REQUIRES_OK(context, context->GetAttr("aggregate_runs", &aggregate_runs_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& spikes = context->input(0);
    const Tensor& weights = context->input(1);
    const Tensor& post_ids = context->input(2);
    const Tensor& synapse_types = context->input(3);
    const Tensor& row_splits = context->input(4);
    const Tensor& edge_ids = context->input(5);
    const Tensor& basis = context->input(6);
    const Tensor& initial = context->input(7);
    OP_REQUIRES(context, spikes.dims() == 2,
                errors::InvalidArgument("spikes must be rank two"));
    OP_REQUIRES(context, basis.dims() == 2 && basis.dim_size(1) > 0,
                errors::InvalidArgument("basis must be [n_types,n_basis], n_basis > 0"));
    OP_REQUIRES(context, row_splits.NumElements() == spikes.dim_size(1) + 1,
                errors::InvalidArgument("row_splits does not match spike width"));
    Tensor* output;
    const int64_t batch = spikes.dim_size(0);
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
                     context->forward_input_or_allocate_output({7}, 0, shape,
                                                               &output));
      if (output->flat<T>().data() != initial.flat<T>().data()) {
        cudaMemcpyAsync(output->flat<T>().data(), initial.flat<T>().data(),
                        output->NumElements() * sizeof(T),
                        cudaMemcpyDeviceToDevice, device.stream());
      }
    } else {
      OP_REQUIRES_OK(context, context->allocate_output(0, shape, &output));
      cudaMemsetAsync(output->flat<T>().data(), 0,
                      output->NumElements() * sizeof(T), device.stream());
    }
    if (n_basis == 4) {
      OP_REQUIRES_OK(context, LaunchForward<T, W, 4>(
                                  context, spikes, weights, post_ids,
                                  synapse_types, row_splits, edge_ids, basis,
                                  n_post_, aggregate_runs_, output));
    } else {
      OP_REQUIRES_OK(context, LaunchForward<T, W, 0>(
                                  context, spikes, weights, post_ids,
                                  synapse_types, row_splits, edge_ids, basis,
                                  n_post_, aggregate_runs_, output));
    }
  }

 private:
  int n_post_;
  bool aggregate_runs_ = true;
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
    const Tensor& spikes = context->input(0);
    const Tensor& current_grad = context->input(1);
    const Tensor& weights = context->input(2);
    const Tensor& post_ids = context->input(3);
    const Tensor& synapse_types = context->input(4);
    const Tensor& row_splits = context->input(5);
    const Tensor& edge_ids = context->input(6);
    const Tensor& nonempty_rows = context->input(7);
    const Tensor& basis = context->input(8);
    const Tensor& dampening = context->input(9);
    const Tensor& pair_ids = context->input(10);
    const Tensor& pair_posts = context->input(11);
    const Tensor& pair_types = context->input(12);
    OP_REQUIRES(context, spikes.dims() == 2,
                errors::InvalidArgument("spikes must be rank two"));
    OP_REQUIRES(context, basis.dims() == 2 && basis.dim_size(1) > 0,
                errors::InvalidArgument("basis dimension must be positive"));
    OP_REQUIRES(context,
                current_grad.dims() == 2 &&
                    current_grad.dim_size(0) == spikes.dim_size(0) * n_post_ &&
                    current_grad.dim_size(1) == basis.dim_size(1),
                errors::InvalidArgument("current_grad has an incompatible shape"));
    OP_REQUIRES(context, dampening.NumElements() == 1,
                errors::InvalidArgument("dampening must be scalar"));
    OP_REQUIRES(context, pair_ids.NumElements() == post_ids.NumElements(),
                errors::InvalidArgument("pair_ids must align with CSR edges"));
    OP_REQUIRES(context, pair_posts.NumElements() == pair_types.NumElements(),
                errors::InvalidArgument("pair metadata lengths differ"));
    Tensor* spike_grad;
    OP_REQUIRES_OK(context, context->allocate_output(0, spikes.shape(), &spike_grad));
#define LAUNCH_BACKWARD(BASIS)                                                     \
  LaunchPairProjectedBackward<T, W, BASIS>(                                        \
      context, spikes, current_grad, weights, post_ids, synapse_types,            \
      row_splits, edge_ids, nonempty_rows, basis, dampening, pair_ids,            \
      pair_posts, pair_types, n_post_, spike_grad, weight_grad, kAccumulate)
    OP_REQUIRES_OK(context, WithWeightGradBuffer<kAccumulate>(
                                context, 13, n_edges_, [&](Tensor* weight_grad) {
                                  return basis.dim_size(1) == 4 ? LAUNCH_BACKWARD(4)
                                                                : LAUNCH_BACKWARD(0);
                                }));
#undef LAUNCH_BACKWARD
  }

 private:
  int n_post_;
  int n_edges_;
};

#ifndef V1_KERNEL_IMPLEMENTATION_ONLY
#define REGISTER_TYPE(T)                                                   \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("V1CsrForward").Device(DEVICE_GPU).TypeConstraint<T>("T"),       \
      V1CsrForwardOp<T, float>);                                           \
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

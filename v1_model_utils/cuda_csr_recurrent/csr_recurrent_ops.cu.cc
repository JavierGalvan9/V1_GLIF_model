#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <type_traits>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

using namespace tensorflow;
using GPUDevice = Eigen::GpuDevice;

#ifndef V1_THREADS
#define V1_THREADS 128
#endif
#ifndef V1_BATCH32_TILE
#define V1_BATCH32_TILE 4
#endif
#ifndef V1_WARP_REDUCTION
#define V1_WARP_REDUCTION 0
#endif
#ifndef V1_HALF2_PROJECTION
#define V1_HALF2_PROJECTION 0
#endif
#ifndef V1_WARP_PER_ROW
#define V1_WARP_PER_ROW 0
#endif
#ifndef V1_LARGE_BACKWARD_TILE
#define V1_LARGE_BACKWARD_TILE 32
#endif
#ifndef V1_PREPROJECT
#define V1_PREPROJECT 0
#endif
#ifndef V1_PAIR_PREPROJECT
#define V1_PAIR_PREPROJECT 0
#endif
#ifndef V1_FORWARD_THREADS
#define V1_FORWARD_THREADS 128
#endif
#ifndef V1_FORWARD_HALF2_ATOMICS
#define V1_FORWARD_HALF2_ATOMICS 0
#endif
#ifndef V1_FORWARD_FLOAT_ACCUM
#define V1_FORWARD_FLOAT_ACCUM 0
#endif
#ifndef V1_FORWARD_GROUPED
#define V1_FORWARD_GROUPED 0
#endif
#ifndef V1_DIRECT_CSR
#define V1_DIRECT_CSR 0
#endif
// Enables the packed batch-lane backward specialization. It uses no tensor
// cores -- only vector loads and warp shuffles -- but it is qualified on SM120
// only, so older architectures keep the batch-lane path until measured.
#ifndef V1_PACKED_BACKWARD
#define V1_PACKED_BACKWARD 0
#endif

// With direct-CSR weights the caller keeps `weights` and `weight_grad` in CSR
// edge order, so the CSR position is the weight index and the effectively
// random `edge_ids[csr]` gather disappears from every inner loop. The choice is
// global: forward and backward, and every batch specialization, must agree.
#if V1_DIRECT_CSR
#define V1_EDGE_INDEX(csr) (csr)
#else
#define V1_EDGE_INDEX(csr) (edge_ids[csr])
#endif

constexpr int kThreads = V1_THREADS;

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

template <typename T, int kBasis>
struct BasisProjection {
  __device__ __forceinline__ static float Apply(const T* upstream,
                                                 const T* basis, int type,
                                                 int n_basis) {
    float result = 0.0f;
#pragma unroll
    for (int receptor = 0; receptor < kBasis; ++receptor) {
      result += AsFloat(upstream[receptor]) *
                AsFloat(basis[type * kBasis + receptor]);
    }
    return result;
  }
};

template <typename T>
struct BasisProjection<T, 0> {
  __device__ __forceinline__ static float Apply(const T* upstream,
                                                 const T* basis, int type,
                                                 int n_basis) {
    float result = 0.0f;
    for (int receptor = 0; receptor < n_basis; ++receptor) {
      result += AsFloat(upstream[receptor]) *
                AsFloat(basis[type * n_basis + receptor]);
    }
    return result;
  }
};

#if V1_HALF2_PROJECTION
template <>
struct BasisProjection<Eigen::half, 4> {
  __device__ __forceinline__ static float Apply(const Eigen::half* upstream,
                                                 const Eigen::half* basis,
                                                 int type, int n_basis) {
    const __half2 upstream01 = *reinterpret_cast<const __half2*>(upstream);
    const __half2 upstream23 = *reinterpret_cast<const __half2*>(upstream + 2);
    const Eigen::half* type_basis = basis + type * 4;
    const __half2 basis01 = *reinterpret_cast<const __half2*>(type_basis);
    const __half2 basis23 = *reinterpret_cast<const __half2*>(type_basis + 2);
    const float2 u01 = __half22float2(upstream01);
    const float2 u23 = __half22float2(upstream23);
    const float2 b01 = __half22float2(basis01);
    const float2 b23 = __half22float2(basis23);
    return fmaf(u01.x, b01.x,
                fmaf(u01.y, b01.y,
                     fmaf(u23.x, b23.x, u23.y * b23.y)));
  }
};
#endif

template <typename T, typename W, typename O, int kBasis, int kBatch>
__global__ void ForwardKernel(
    int64_t n_active, int64_t n_pre, int n_post, int n_basis,
    const T* spikes, const int64_t* active, const W* weights,
    const uint32* post_ids, const uint8* synapse_types,
    const uint32* row_splits, const uint32* edge_ids, const T* basis,
    O* currents) {
  const int64_t active_id = blockIdx.x;
  if (active_id >= n_active) return;
  const int64_t batch = active[2 * active_id];
  if (kBatch != 0 && batch >= kBatch) return;
  const int64_t pre = active[2 * active_id + 1];
  const float spike = AsFloat(spikes[batch * n_pre + pre]);
  const uint32 start = row_splits[pre];
  const uint32 end = row_splits[pre + 1];
  for (uint32 csr = start + threadIdx.x; csr < end; csr += blockDim.x) {
    const uint32 edge = V1_EDGE_INDEX(csr);
    const uint32 post = post_ids[csr];
    const uint32 type = synapse_types[csr];
    const float weighted_spike = spike * AsFloat(weights[edge]);
    O* output = currents +
                (batch * static_cast<int64_t>(n_post) + post) * n_basis;
    if (kBasis == 4) {
#if V1_FORWARD_HALF2_ATOMICS
      if constexpr (std::is_same<T, Eigen::half>::value) {
        const Eigen::half* type_basis = basis + type * 4;
        const float2 basis01 = __half22float2(
            *reinterpret_cast<const __half2*>(type_basis));
        const float2 basis23 = __half22float2(
            *reinterpret_cast<const __half2*>(type_basis + 2));
        if constexpr (std::is_same<O, Eigen::half>::value) {
          atomicAdd(reinterpret_cast<__half2*>(output),
                    __floats2half2_rn(weighted_spike * basis01.x,
                                      weighted_spike * basis01.y));
          atomicAdd(reinterpret_cast<__half2*>(output + 2),
                    __floats2half2_rn(weighted_spike * basis23.x,
                                      weighted_spike * basis23.y));
        } else {
          atomicAdd(output, weighted_spike * basis01.x);
          atomicAdd(output + 1, weighted_spike * basis01.y);
          atomicAdd(output + 2, weighted_spike * basis23.x);
          atomicAdd(output + 3, weighted_spike * basis23.y);
        }
      } else {
#pragma unroll
        for (int receptor = 0; receptor < 4; ++receptor) {
          AtomicAddValue(output + receptor,
                         weighted_spike * AsFloat(basis[type * 4 + receptor]));
        }
      }
#else
#pragma unroll
      for (int receptor = 0; receptor < 4; ++receptor) {
        AtomicAddValue(output + receptor,
                       weighted_spike * AsFloat(basis[type * 4 + receptor]));
      }
#endif
    } else {
      for (int receptor = 0; receptor < n_basis; ++receptor) {
        AtomicAddValue(
            output + receptor,
            weighted_spike * AsFloat(basis[type * n_basis + receptor]));
      }
    }
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

template <typename T, typename W, int kBasis, int kBatch>
__global__ void ForwardGroupedStaticBatchKernel(
    int64_t n_active_rows, int64_t n_pre, int n_post, int n_basis,
    const T* spikes, const int64_t* active_rows, const W* weights,
    const uint32* post_ids, const uint8* synapse_types,
    const uint32* row_splits, const uint32* edge_ids, const T* basis,
    T* currents) {
  const int64_t row_id = blockIdx.x;
  if (row_id >= n_active_rows) return;
  const uint32 pre = static_cast<uint32>(active_rows[2 * row_id + 1]);
  constexpr int kMasks = (kBatch + 31) / 32;
  __shared__ uint32 batch_masks[kMasks];
  if (threadIdx.x < 32) {
    for (int mask_id = 0; mask_id < kMasks; ++mask_id) {
      const int batch = mask_id * 32 + threadIdx.x;
      const bool active = batch < kBatch &&
                          AsFloat(spikes[batch * n_pre + pre]) != 0.0f;
      const uint32 mask = __ballot_sync(0xffffffff, active);
      if (threadIdx.x == 0) batch_masks[mask_id] = mask;
    }
  }
  __syncthreads();
  for (uint32 csr = row_splits[pre] + threadIdx.x;
       csr < row_splits[pre + 1]; csr += blockDim.x) {
    const uint32 edge = V1_EDGE_INDEX(csr);
    const uint32 post = post_ids[csr];
    const uint32 type = synapse_types[csr];
    const float weight = AsFloat(weights[edge]);
    for (int mask_id = 0; mask_id < kMasks; ++mask_id) {
      uint32 remaining = batch_masks[mask_id];
      while (remaining != 0) {
      const int batch = mask_id * 32 + __ffs(remaining) - 1;
      remaining &= remaining - 1;
      const float weighted_spike =
          AsFloat(spikes[batch * n_pre + pre]) * weight;
      T* output = currents +
                  (batch * static_cast<int64_t>(n_post) + post) * n_basis;
      if constexpr (std::is_same<T, Eigen::half>::value) {
        const Eigen::half* type_basis = basis + type * 4;
        const float2 basis01 = __half22float2(
            *reinterpret_cast<const __half2*>(type_basis));
        const float2 basis23 = __half22float2(
            *reinterpret_cast<const __half2*>(type_basis + 2));
        atomicAdd(reinterpret_cast<__half2*>(output),
                  __floats2half2_rn(weighted_spike * basis01.x,
                                    weighted_spike * basis01.y));
        atomicAdd(reinterpret_cast<__half2*>(output + 2),
                  __floats2half2_rn(weighted_spike * basis23.x,
                                    weighted_spike * basis23.y));
      } else {
#pragma unroll
        for (int receptor = 0; receptor < 4; ++receptor) {
          AtomicAddValue(output + receptor,
                         weighted_spike * AsFloat(basis[type * 4 + receptor]));
        }
      }
      }
    }
  }
}

template <typename T, typename W, int kBasis, int kTile>
__global__ void BackwardRuntimeBatchKernel(
    int64_t batch_size, int64_t n_pre, int n_post, int n_basis, const T* spikes,
    const T* current_grad, const W* weights, const uint32* post_ids,
    const uint8* synapse_types, const uint32* row_splits,
    const uint32* edge_ids, const uint32* nonempty_rows, int64_t n_rows,
    const T* basis, const T* dampening, T* spike_grad,
    float* weight_grad) {
  __shared__ float reduction[kTile][kThreads];
  const int64_t tile_id = blockIdx.x / n_rows;
  const int64_t row_id = blockIdx.x - tile_id * n_rows;
  if (row_id >= n_rows) return;
  const int64_t first_batch = tile_id * kTile;
  const uint32 pre = nonempty_rows[row_id];
  float local_spike_grad[kTile] = {};
    const uint32 start = row_splits[pre];
    const uint32 end = row_splits[pre + 1];
    for (uint32 csr = start + threadIdx.x; csr < end; csr += blockDim.x) {
      const uint32 edge = V1_EDGE_INDEX(csr);
      const uint32 post = post_ids[csr];
      const uint32 type = synapse_types[csr];
    float tile_weight_grad = 0.0f;
#pragma unroll
    for (int offset = 0; offset < kTile; ++offset) {
      const int64_t batch = first_batch + offset;
      if (batch < batch_size) {
        const T* upstream = current_grad +
                            (batch * static_cast<int64_t>(n_post) + post) *
                                n_basis;
        const float projected = BasisProjection<T, kBasis>::Apply(
            upstream, basis, type, n_basis);
        local_spike_grad[offset] += projected * AsFloat(weights[edge]);
        tile_weight_grad += projected * AsFloat(spikes[batch * n_pre + pre]);
      }
    }
    if (tile_weight_grad != 0.0f) {
      if (batch_size <= kTile) {
        weight_grad[edge] = tile_weight_grad;
      } else {
        atomicAdd(weight_grad + edge, tile_weight_grad);
      }
    }
  }
#pragma unroll
  for (int offset = 0; offset < kTile; ++offset) {
    reduction[offset][threadIdx.x] = local_spike_grad[offset];
  }
  __syncthreads();
  for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
#pragma unroll
      for (int offset = 0; offset < kTile; ++offset) {
        reduction[offset][threadIdx.x] +=
            reduction[offset][threadIdx.x + stride];
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
#pragma unroll
    for (int offset = 0; offset < kTile; ++offset) {
      const int64_t batch = first_batch + offset;
      if (batch < batch_size) {
        spike_grad[batch * n_pre + pre] =
            FromFloat<T>(reduction[offset][0] * AsFloat(*dampening));
      }
    }
  }
}

template <typename T, typename W, int kBasis, int kBatch, int kTile>
__global__ void BackwardStaticBatchKernel(
    int64_t n_pre, int n_post, int n_basis, const T* spikes,
    const T* current_grad, const W* weights, const uint32* post_ids,
    const uint8* synapse_types, const uint32* row_splits,
    const uint32* edge_ids, const uint32* nonempty_rows, int64_t n_rows,
    const T* basis, const T* dampening, T* spike_grad,
    float* weight_grad) {
  static_assert(kBatch % kTile == 0, "batch tiles must divide the batch");
#if V1_WARP_REDUCTION
  constexpr int kWarps = (kThreads + 31) / 32;
  __shared__ float reduction[kTile][kWarps];
#else
  __shared__ float reduction[kTile][kThreads];
#endif
  const int64_t tile_id = blockIdx.x / n_rows;
  const int64_t row_id = blockIdx.x - tile_id * n_rows;
  if (row_id >= n_rows) return;
  const int first_batch = static_cast<int>(tile_id) * kTile;
  const uint32 pre = nonempty_rows[row_id];
  float local_spike_grad[kTile] = {};
  const uint32 start = row_splits[pre];
  const uint32 end = row_splits[pre + 1];
  for (uint32 csr = start + threadIdx.x; csr < end; csr += blockDim.x) {
    const uint32 edge = V1_EDGE_INDEX(csr);
    const uint32 post = post_ids[csr];
    const uint32 type = synapse_types[csr];
    float tile_weight_grad = 0.0f;
#pragma unroll
    for (int offset = 0; offset < kTile; ++offset) {
      const int batch = first_batch + offset;
      const T* upstream = current_grad +
                          (batch * static_cast<int64_t>(n_post) + post) *
                              n_basis;
      const float projected = BasisProjection<T, kBasis>::Apply(
          upstream, basis, type, n_basis);
      local_spike_grad[offset] += projected * AsFloat(weights[edge]);
      tile_weight_grad += projected * AsFloat(spikes[batch * n_pre + pre]);
    }
    if (tile_weight_grad != 0.0f) {
      if (kBatch <= kTile) {
        weight_grad[edge] = tile_weight_grad;
      } else {
        atomicAdd(weight_grad + edge, tile_weight_grad);
      }
    }
  }
#if V1_WARP_REDUCTION
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int offset = 0; offset < kTile; ++offset) {
    float value = local_spike_grad[offset];
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
      value += __shfl_down_sync(0xffffffff, value, delta);
    }
    if (lane == 0) reduction[offset][warp] = value;
  }
  __syncthreads();
  if (warp == 0) {
#pragma unroll
    for (int offset = 0; offset < kTile; ++offset) {
      float value = lane < kWarps ? reduction[offset][lane] : 0.0f;
#pragma unroll
      for (int delta = 16; delta > 0; delta >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, delta);
      }
      if (lane == 0) {
        const int batch = first_batch + offset;
        spike_grad[batch * n_pre + pre] =
            FromFloat<T>(value * AsFloat(*dampening));
      }
    }
  }
#else
#pragma unroll
  for (int offset = 0; offset < kTile; ++offset) {
    reduction[offset][threadIdx.x] = local_spike_grad[offset];
  }
  __syncthreads();
  for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
#pragma unroll
      for (int offset = 0; offset < kTile; ++offset) {
        reduction[offset][threadIdx.x] += reduction[offset][threadIdx.x + stride];
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
#pragma unroll
    for (int offset = 0; offset < kTile; ++offset) {
      const int batch = first_batch + offset;
      spike_grad[batch * n_pre + pre] =
          FromFloat<T>(reduction[offset][0] * AsFloat(*dampening));
    }
  }
#endif
}

template <typename T, typename W, int kBasis, int kBatch, int kTile,
          int kBlockThreads>
__global__ void BackwardWarpPerRowStaticBatchKernel(
    int64_t n_pre, int n_post, int n_basis, const T* spikes,
    const T* current_grad, const W* weights, const uint32* post_ids,
    const uint8* synapse_types, const uint32* row_splits,
    const uint32* edge_ids, const uint32* nonempty_rows, int64_t n_rows,
    const T* basis, const float* preprojected, int n_types,
    const uint32* pair_ids, int64_t n_pairs,
    const T* dampening, T* spike_grad,
    float* weight_grad) {
  static_assert(kBatch % kTile == 0, "batch tiles must divide the batch");
  constexpr int kWarps = kBlockThreads / 32;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int64_t work_id = static_cast<int64_t>(blockIdx.x) * kWarps + warp;
  const int64_t tile_id = work_id / n_rows;
  const int64_t row_id = work_id - tile_id * n_rows;
  if (tile_id >= kBatch / kTile) return;
  if (row_id >= n_rows) return;
  const int first_batch = static_cast<int>(tile_id) * kTile;
  const uint32 pre = nonempty_rows[row_id];
  float local_spike_grad[kTile] = {};
  const uint32 start = row_splits[pre];
  const uint32 end = row_splits[pre + 1];
  for (uint32 csr = start + lane; csr < end; csr += 32) {
    const uint32 edge = V1_EDGE_INDEX(csr);
    const uint32 post = post_ids[csr];
    const uint32 type = synapse_types[csr];
    float edge_weight_grad = 0.0f;
#pragma unroll
    for (int offset = 0; offset < kTile; ++offset) {
      const int batch = first_batch + offset;
#if V1_PAIR_PREPROJECT
      const float projected = preprojected[
          batch * n_pairs + pair_ids[csr]];
#elif V1_PREPROJECT
      const float projected = preprojected[
          (batch * static_cast<int64_t>(n_post) + post) * n_types + type];
#else
      const T* upstream = current_grad +
                          (batch * static_cast<int64_t>(n_post) + post) * n_basis;
      const float projected = BasisProjection<T, kBasis>::Apply(
          upstream, basis, type, n_basis);
#endif
      local_spike_grad[offset] += projected * AsFloat(weights[edge]);
      edge_weight_grad += projected * AsFloat(spikes[batch * n_pre + pre]);
    }
    if constexpr (kBatch == kTile) {
      weight_grad[edge] = edge_weight_grad;
    } else {
      atomicAdd(weight_grad + edge, edge_weight_grad);
    }
  }
#pragma unroll
  for (int offset = 0; offset < kTile; ++offset) {
    float value = local_spike_grad[offset];
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
      value += __shfl_down_sync(0xffffffff, value, delta);
    }
    if (lane == 0) {
      const int batch = first_batch + offset;
      spike_grad[batch * n_pre + pre] =
          FromFloat<T>(value * AsFloat(*dampening));
    }
  }
}

template <typename T, int kBasis>
__global__ void PreprojectBatch32Kernel(int64_t elements, int n_post,
                                        int n_basis, int n_types,
                                        const T* current_grad, const T* basis,
                                        float* projected) {
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < elements; index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int type = index % n_types;
    const int64_t row = index / n_types;
    projected[index] = BasisProjection<T, kBasis>::Apply(
        current_grad + row * n_basis, basis, type, n_basis);
  }
}

template <typename T, int kBasis>
__global__ void PreprojectPairsBatch32Kernel(
    int64_t elements, int n_post, int n_basis, int64_t n_pairs,
    const T* current_grad, const T* basis, const uint32* pair_posts,
    const uint8* pair_types, float* projected) {
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < elements; index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t pair = index % n_pairs;
    const int64_t batch = index / n_pairs;
    const uint32 post = pair_posts[pair];
    const uint32 type = pair_types[pair];
    projected[index] = BasisProjection<T, kBasis>::Apply(
        current_grad + (batch * static_cast<int64_t>(n_post) + post) * n_basis,
        basis, type, n_basis);
  }
}

// Compact projection, emitted as [pair, batch] so that one warp reading a
// pair's whole batch line touches a single 128-byte line.
//
// That output order is what the backward row kernels want, but writing it
// directly makes consecutive threads take consecutive batch samples of one
// pair, whose current gradients are `n_post * n_basis` apart -- 1.6 MiB on the
// 203,816-neuron network. Each lane then pulls its own 32-byte sector for the
// 8 bytes it needs, and a warp fetches 1 KiB to use 256 B.
//
// Pairs are sorted by (postsynaptic neuron, synapse type) and average 8.2 types
// per neuron, so thirty-two consecutive pairs touch about four distinct
// neurons. Reading pair-contiguous, transposing through shared memory and
// writing batch-contiguous coalesces both halves. The padding keeps the
// transpose read conflict-free: a 34-float row is 34 banks, and 34 is even but
// coprime to 32 in the stride that matters here -- consecutive lanes land on
// banks (34 * lane) % 32 = (2 * lane) % 32, so the row is read in two
// conflict-free halves rather than one 32-way conflict.
template <typename T, int kBasis, int kBatch, int kPairsPerTile>
__global__ void PreprojectPairsTiledKernel(
    int n_post, int n_basis, int64_t n_pairs, const T* current_grad,
    const T* basis, const uint32* pair_posts, const uint8* pair_types,
    float* projected) {
  constexpr int kTileElements = kPairsPerTile * kBatch;
  __shared__ float tile[kBatch][kPairsPerTile + 2];
  const int64_t pair_base = static_cast<int64_t>(blockIdx.x) * kPairsPerTile;
  for (int index = threadIdx.x; index < kTileElements; index += blockDim.x) {
    const int column = index % kPairsPerTile;
    const int batch = index / kPairsPerTile;
    const int64_t pair = pair_base + column;
    float value = 0.0f;
    if (pair < n_pairs) {
      value = BasisProjection<T, kBasis>::Apply(
          current_grad + (static_cast<int64_t>(batch) * n_post +
                          pair_posts[pair]) * n_basis,
          basis, pair_types[pair], n_basis);
    }
    tile[batch][column] = value;
  }
  __syncthreads();
  for (int index = threadIdx.x; index < kTileElements; index += blockDim.x) {
    const int batch = index % kBatch;
    const int column = index / kBatch;
    const int64_t pair = pair_base + column;
    if (pair < n_pairs) projected[pair * kBatch + batch] = tile[batch][column];
  }
}

// Backward with lane == batch sample instead of lane == edge.
//
// Every lane walks the whole CSR row, so each lane keeps a single spike-gradient
// accumulator rather than a per-tile array; that removes the register array and
// its stack spill from the hot kernel. The weight gradient of one edge is the
// sum across the batch, which is exactly a warp reduction.
template <typename T, typename W, int kBatch, int kWarps>
__global__ void BackwardBatchLaneKernel(
    int64_t n_pre, const T* spikes, const float* projected, const W* weights,
    const uint32* pair_ids, const uint32* edge_ids, const uint32* row_splits,
    const uint32* nonempty_rows, int64_t n_rows, const T* dampening,
    T* spike_grad, float* weight_grad) {
  static_assert(kBatch == 32, "the batch-lane mapping needs one lane per sample");
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int64_t row_id = static_cast<int64_t>(blockIdx.x) * kWarps + warp;
  if (row_id >= n_rows) return;
  const uint32 pre = nonempty_rows[row_id];
  const uint32 start = row_splits[pre];
  const uint32 end = row_splits[pre + 1];
  const float spike = AsFloat(spikes[static_cast<int64_t>(lane) * n_pre + pre]);
  float local_spike_grad = 0.0f;
  for (uint32 csr = start; csr < end; ++csr) {
    const uint32 edge = V1_EDGE_INDEX(csr);
    const float value =
        projected[static_cast<int64_t>(pair_ids[csr]) * kBatch + lane];
    local_spike_grad += value * AsFloat(weights[edge]);
    float edge_weight_grad = value * spike;
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
      edge_weight_grad += __shfl_down_sync(0xffffffff, edge_weight_grad, delta);
    }
    // The full batch is covered in one pass, so this is the only writer.
    if (lane == 0) weight_grad[edge] = edge_weight_grad;
  }
  spike_grad[static_cast<int64_t>(lane) * n_pre + pre] =
      FromFloat<T>(local_spike_grad * AsFloat(*dampening));
}

// Cache policy for the backward row kernel's operands.
//
// The projection is re-read once per edge sharing a pair (50.2 times on the
// 203,816-neuron network) and is the only array with reuse. `pair_ids`,
// `weights` and the `weight_grad` output are each touched exactly once per
// edge, so evict-first hints keep their roughly 1 GiB of single-use traffic
// from displacing the projection in L2.
__device__ __forceinline__ uint32 LoadEdgeIndex(const uint32* address) {
  return __ldcs(address);
}

template <typename W>
__device__ __forceinline__ float LoadEdgeWeight(const W* address) {
  if constexpr (std::is_same<W, float>::value) return __ldcs(address);
  return AsFloat(*address);
}

__device__ __forceinline__ void StoreEdgeGrad(float* address, float value) {
  __stcs(address, value);
}

// Reduce `partial[0 .. 2 * kHalf)` across the lane pair (lane, lane ^ kMask)
// into `partial[0 .. kHalf)`, then recurse. One shuffle and one add per
// surviving value, so a whole tile costs about one shuffle per edge -- against
// a shared-memory round trip per tile for a tensor-core reduction.
//
// Each round drops the lane bit it reduced over into the surviving value's
// index, so after the recursion lane l holds the value for index
// bit-reverse(l). `ReverseBits` undoes that when the results are stored.
template <int kHalf, int kMask>
__device__ __forceinline__ void ButterflyReduce(float* partial, int lane) {
  const bool upper = (lane & kMask) != 0;
#pragma unroll
  for (int index = 0; index < kHalf; ++index) {
    const float keep = upper ? partial[kHalf + index] : partial[index];
    const float send = upper ? partial[index] : partial[kHalf + index];
    partial[index] = keep + __shfl_xor_sync(0xffffffff, send, kMask);
  }
  if constexpr (kHalf > 1) ButterflyReduce<kHalf / 2, kMask * 2>(partial, lane);
}

template <int kBits>
__device__ __forceinline__ int ReverseBits(int value) {
  int result = 0;
#pragma unroll
  for (int bit = 0; bit < kBits; ++bit) {
    result |= ((value >> bit) & 1) << (kBits - 1 - bit);
  }
  return result;
}

// One lane's slice of a pair's projected batch line. A 16-byte load is the
// widest a lane can issue, so it carries four FP32 batch samples.
template <int kPack>
struct PackedProjection {
  __device__ __forceinline__ static void Load(const float* source, float* out);
};

template <>
struct PackedProjection<4> {
  __device__ __forceinline__ static void Load(const float* source, float* out) {
    const ::float4 raw = *reinterpret_cast<const ::float4*>(source);
    out[0] = raw.x;
    out[1] = raw.y;
    out[2] = raw.z;
    out[3] = raw.w;
  }
};

// FP16 batch-32 specialization: one CSR row per two-warp block, packed batch
// lanes, butterfly weight-gradient reduction.
//
// Giving a lane `kPack` consecutive batch samples shrinks an edge from 32 lanes
// to `32 / kPack`, so a warp splits into that many independent edge slots and
// one load instruction serves all of them. At kPack = 4 a 32-edge tile issues
// eight projection loads instead of thirty-two, and the descriptors
// (`pair_ids`, `weights`) are read once cooperatively and broadcast with
// `__shfl_sync` rather than re-read at one address by all 32 lanes. The bytes
// moved are unchanged; the instruction count is not.
//
// The block holds 256 B of shared memory, so an SM runs its full 24 blocks.
// Both outputs keep FP32 arithmetic: the spike gradient accumulates the FP32
// projected value, and the butterfly sums the weight gradient in FP32.
template <typename W, int kPack, int kWarps, int kTile>
__global__ __launch_bounds__(32 * kWarps) void BackwardPackedRowKernel(
    int64_t n_pre, const Eigen::half* spikes, const float* projected,
    const W* weights, const uint32* pair_ids, const uint32* edge_ids,
    const uint32* row_splits, const uint32* nonempty_rows, int64_t n_rows,
    const Eigen::half* dampening, Eigen::half* spike_grad,
    float* weight_grad) {
  constexpr int kBatch = 32;
  constexpr int kLanesPerEdge = kBatch / kPack;
  constexpr int kSlots = 32 / kLanesPerEdge;
  constexpr int kPerSlot = kTile / kSlots;
  constexpr int kIndexBits = kPerSlot == 8 ? 3 : (kPerSlot == 4 ? 2 : 1);
  static_assert(kTile == kSlots * kPerSlot, "tile must divide into edge slots");
  static_assert(kPerSlot == (1 << kIndexBits), "slot depth must be a power of two");
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int slot = lane / kLanesPerEdge;
  const int sub = lane % kLanesPerEdge;   // batch samples sub * kPack ... + kPack
  const int64_t row_id = blockIdx.x;
  if (row_id >= n_rows) return;

  __shared__ float spike_partials[kWarps][32];

  const uint32 pre = nonempty_rows[row_id];
  const uint32 start = row_splits[pre];
  const uint32 end = row_splits[pre + 1];
  float spike[kPack];
  float grad[kPack];
#pragma unroll
  for (int sample = 0; sample < kPack; ++sample) {
    spike[sample] = AsFloat(
        spikes[static_cast<int64_t>(sub * kPack + sample) * n_pre + pre]);
    grad[sample] = 0.0f;
  }
  const int target = ReverseBits<kIndexBits>(sub & (kPerSlot - 1));

  for (uint32 base = start + warp * kTile; base < end; base += kWarps * kTile) {
    // One coalesced load of the tile's edge descriptors, broadcast to the batch
    // lanes below instead of re-read by each of them.
    const uint32 edge_lane = base + lane;
    const bool own = lane < kTile && edge_lane < end;
    const uint32 my_pair = own ? LoadEdgeIndex(pair_ids + edge_lane) : 0u;
    const float my_weight =
        own ? LoadEdgeWeight<W>(weights + V1_EDGE_INDEX(edge_lane)) : 0.0f;
    float partial[kPerSlot];
#pragma unroll
    for (int step = 0; step < kPerSlot; ++step) {
      const int column = kSlots * step + slot;
      const uint32 pair = __shfl_sync(0xffffffff, my_pair, column);
      const float weight = __shfl_sync(0xffffffff, my_weight, column);
      float value[kPack];
      if (base + column < end) {
        PackedProjection<kPack>::Load(
            projected + static_cast<int64_t>(pair) * kBatch + sub * kPack, value);
      } else {
#pragma unroll
        for (int sample = 0; sample < kPack; ++sample) value[sample] = 0.0f;
      }
      float sum = 0.0f;
#pragma unroll
      for (int sample = 0; sample < kPack; ++sample) {
        grad[sample] += value[sample] * weight;
        sum += value[sample] * spike[sample];
      }
      partial[step] = sum;
    }
    // Compact the slot's `kPerSlot` values down its lanes, then fold the lanes
    // left holding duplicates of the same edge.
    if constexpr (kPerSlot > 1) ButterflyReduce<kPerSlot / 2, 1>(partial, lane);
#pragma unroll
    for (int mask = kPerSlot; mask < kLanesPerEdge; mask <<= 1) {
      partial[0] += __shfl_xor_sync(0xffffffff, partial[0], mask);
    }
    const uint32 edge = base + kSlots * target + slot;
    if (sub < kPerSlot && edge < end) {
      StoreEdgeGrad(weight_grad + V1_EDGE_INDEX(edge), partial[0]);
    }
  }

  // Every slot accumulated the same batch samples from a different edge subset.
#pragma unroll
  for (int mask = kLanesPerEdge; mask < 32; mask <<= 1) {
#pragma unroll
    for (int sample = 0; sample < kPack; ++sample) {
      grad[sample] += __shfl_xor_sync(0xffffffff, grad[sample], mask);
    }
  }
  if (lane < kLanesPerEdge) {
#pragma unroll
    for (int sample = 0; sample < kPack; ++sample) {
      spike_partials[warp][sub * kPack + sample] = grad[sample];
    }
  }
  __syncthreads();
  if (warp == 0) {
    float total = 0.0f;
#pragma unroll
    for (int source = 0; source < kWarps; ++source) {
      total += spike_partials[source][lane];
    }
    spike_grad[static_cast<int64_t>(lane) * n_pre + pre] =
        FromFloat<Eigen::half>(total * AsFloat(*dampening));
  }
}

#define V1_BATCH_CASE(BATCH, TILE, LAUNCH) \
  case BATCH:                              \
    LAUNCH(BATCH, TILE);                   \
    break

template <typename T, typename W, int kBasis>
Status LaunchForward(OpKernelContext* context, const Tensor& spikes,
                     const Tensor& active, const Tensor& weights,
                     const Tensor& post_ids, const Tensor& synapse_types,
                     const Tensor& row_splits, const Tensor& edge_ids,
                     const Tensor& basis, int n_post, Tensor* output) {
  const int64_t n_active = active.dim_size(0);
  if (n_active == 0) return OkStatus();
  const int n_basis = basis.dim_size(1);
  auto device = context->eigen_device<GPUDevice>();
#define LAUNCH_FORWARD_NORMAL(BATCH, OUTPUT_TYPE, OUTPUT_PTR)            \
  TF_RETURN_IF_ERROR(GpuLaunchKernel(                                   \
      ForwardKernel<T, W, OUTPUT_TYPE, kBasis, BATCH>,                   \
      static_cast<int>(n_active),                                       \
      V1_FORWARD_THREADS, 0, device.stream(), n_active, spikes.dim_size(1), \
      n_post, n_basis, spikes.flat<T>().data(),                         \
      active.flat<int64_t>().data(), weights.flat<W>().data(),          \
      post_ids.flat<uint32>().data(), synapse_types.flat<uint8>().data(), \
      row_splits.flat<uint32>().data(), edge_ids.flat<uint32>().data(), \
      basis.flat<T>().data(), OUTPUT_PTR))
#define LAUNCH_FORWARD_GROUPED(BATCH)                                   \
  TF_RETURN_IF_ERROR(GpuLaunchKernel(                                   \
      ForwardGroupedStaticBatchKernel<T, W, kBasis, BATCH>,              \
      static_cast<int>(n_active), V1_FORWARD_THREADS, 0, device.stream(),\
      n_active, spikes.dim_size(1), n_post, n_basis,                     \
      spikes.flat<T>().data(), active.flat<int64_t>().data(),            \
      weights.flat<W>().data(), post_ids.flat<uint32>().data(),          \
      synapse_types.flat<uint8>().data(), row_splits.flat<uint32>().data(),\
      edge_ids.flat<uint32>().data(), basis.flat<T>().data(),             \
      output->flat<T>().data()))
#if V1_FORWARD_FLOAT_ACCUM
  Tensor accumulation;
  TF_RETURN_IF_ERROR(context->allocate_temp(DT_FLOAT, output->shape(), &accumulation));
  cudaMemsetAsync(accumulation.flat<float>().data(), 0,
                  accumulation.NumElements() * sizeof(float), device.stream());
#define LAUNCH_FORWARD(BATCH, TILE) \
  LAUNCH_FORWARD_NORMAL(BATCH, float, accumulation.flat<float>().data())
#else
#define LAUNCH_FORWARD(BATCH, TILE) \
  LAUNCH_FORWARD_NORMAL(BATCH, T, output->flat<T>().data())
#endif
  switch (spikes.dim_size(0)) {
#if V1_FORWARD_GROUPED
#define V1_GROUPED_CASE(BATCH) case BATCH: if constexpr (kBasis == 4) { LAUNCH_FORWARD_GROUPED(BATCH); } else { LAUNCH_FORWARD(BATCH, 4); } break
    V1_GROUPED_CASE(1);
    V1_GROUPED_CASE(2);
    V1_GROUPED_CASE(4);
    V1_GROUPED_CASE(8);
    V1_GROUPED_CASE(16);
    V1_GROUPED_CASE(32);
    V1_GROUPED_CASE(64);
    V1_GROUPED_CASE(128);
#undef V1_GROUPED_CASE
    V1_BATCH_CASE(256, 4, LAUNCH_FORWARD);
#else
    V1_BATCH_CASE(1, 1, LAUNCH_FORWARD);
    V1_BATCH_CASE(2, 2, LAUNCH_FORWARD);
    V1_BATCH_CASE(4, 4, LAUNCH_FORWARD);
    V1_BATCH_CASE(8, 4, LAUNCH_FORWARD);
    V1_BATCH_CASE(16, 4, LAUNCH_FORWARD);
    V1_BATCH_CASE(32, 4, LAUNCH_FORWARD);
    V1_BATCH_CASE(64, 4, LAUNCH_FORWARD);
    V1_BATCH_CASE(128, 4, LAUNCH_FORWARD);
    V1_BATCH_CASE(256, 4, LAUNCH_FORWARD);
#endif
    default:
      LAUNCH_FORWARD(0, 1);
  }
#undef LAUNCH_FORWARD
#undef LAUNCH_FORWARD_NORMAL
#undef LAUNCH_FORWARD_GROUPED
#if V1_FORWARD_FLOAT_ACCUM
  constexpr int kCastThreads = 256;
  const int64_t elements = output->NumElements();
  const int cast_blocks = static_cast<int>((elements + kCastThreads - 1) / kCastThreads);
  TF_RETURN_IF_ERROR(GpuLaunchKernel(
      CastForwardOutputKernel<T>, cast_blocks, kCastThreads, 0, device.stream(),
      elements, accumulation.flat<float>().data(), output->flat<T>().data()));
#endif
  return OkStatus();
}

template <typename T, typename W, int kBasis>
Status LaunchBackward(
    OpKernelContext* context, const Tensor& spikes, const Tensor& current_grad,
    const Tensor& weights, const Tensor& post_ids, const Tensor& synapse_types,
    const Tensor& row_splits, const Tensor& edge_ids,
    const Tensor& nonempty_rows, const Tensor& basis, const Tensor& dampening,
    int n_post, Tensor* spike_grad, Tensor* weight_grad) {
  const int64_t n_rows = nonempty_rows.NumElements();
  const int64_t batch = spikes.dim_size(0);
  if (batch == 0 || n_rows == 0) return OkStatus();
  const int n_basis = basis.dim_size(1);
  const int n_types = basis.dim_size(0);
  auto device = context->eigen_device<GPUDevice>();
#if V1_PREPROJECT
  Tensor projected_tensor;
  const int64_t projected_elements = batch * static_cast<int64_t>(n_post) * n_types;
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DT_FLOAT, TensorShape({projected_elements}), &projected_tensor));
  float* preprojected = projected_tensor.flat<float>().data();
  constexpr int kProjectionThreads = 256;
  const int projection_blocks = static_cast<int>(
      (projected_elements + kProjectionThreads - 1) / kProjectionThreads);
  TF_RETURN_IF_ERROR(GpuLaunchKernel(
      PreprojectBatch32Kernel<T, kBasis>, projection_blocks,
      kProjectionThreads, 0, device.stream(), projected_elements, n_post,
      n_basis, n_types, current_grad.flat<T>().data(), basis.flat<T>().data(),
      preprojected));
#else
  const float* preprojected = nullptr;
#endif
#define LAUNCH_BACKWARD(BATCH, TILE)                                      \
  TF_RETURN_IF_ERROR(GpuLaunchKernel(                                     \
      BackwardStaticBatchKernel<T, W, kBasis, BATCH, TILE>,               \
      static_cast<int>(n_rows * (BATCH / TILE)), kThreads, 0,             \
      device.stream(), spikes.dim_size(1), n_post, n_basis,               \
      spikes.flat<T>().data(), current_grad.flat<T>().data(),             \
      weights.flat<W>().data(), post_ids.flat<uint32>().data(),           \
      synapse_types.flat<uint8>().data(), row_splits.flat<uint32>().data(), \
      edge_ids.flat<uint32>().data(), nonempty_rows.flat<uint32>().data(), \
      n_rows, basis.flat<T>().data(), dampening.flat<T>().data(),         \
      spike_grad->flat<T>().data(), weight_grad->flat<float>().data()))
#define LAUNCH_WARP_PER_ROW(BATCH, TILE, THREADS)                         \
  do {                                                                    \
    constexpr int kWarps = THREADS / 32;                                  \
    constexpr int kTiles = BATCH / TILE;                                  \
    const int64_t work_items = n_rows * kTiles;                            \
    TF_RETURN_IF_ERROR(GpuLaunchKernel(                                    \
        BackwardWarpPerRowStaticBatchKernel<T, W, kBasis, BATCH, TILE, THREADS>, \
        static_cast<int>((work_items + kWarps - 1) / kWarps), THREADS, 0,  \
        device.stream(), spikes.dim_size(1), n_post, n_basis,              \
        spikes.flat<T>().data(), current_grad.flat<T>().data(),            \
        weights.flat<W>().data(), post_ids.flat<uint32>().data(),          \
        synapse_types.flat<uint8>().data(), row_splits.flat<uint32>().data(), \
        edge_ids.flat<uint32>().data(), nonempty_rows.flat<uint32>().data(), \
        n_rows, basis.flat<T>().data(), preprojected, n_types, nullptr, 0,  \
        dampening.flat<T>().data(), spike_grad->flat<T>().data(),          \
        weight_grad->flat<float>().data()));                               \
  } while (false)
  switch (batch) {
#if V1_WARP_PER_ROW
    V1_BATCH_CASE(1, 1, LAUNCH_BACKWARD);
    V1_BATCH_CASE(2, 2, LAUNCH_BACKWARD);
    V1_BATCH_CASE(4, 4, LAUNCH_BACKWARD);
    case 8: LAUNCH_WARP_PER_ROW(8, 8, 256); break;
    case 16: LAUNCH_WARP_PER_ROW(16, 16, 256); break;
    case 32: LAUNCH_WARP_PER_ROW(32, 32, 256); break;
    case 64: LAUNCH_WARP_PER_ROW(64, V1_LARGE_BACKWARD_TILE, 256); break;
    case 128: LAUNCH_WARP_PER_ROW(128, V1_LARGE_BACKWARD_TILE, 128); break;
    case 256: LAUNCH_WARP_PER_ROW(256, V1_LARGE_BACKWARD_TILE, 128); break;
#else
    V1_BATCH_CASE(1, 1, LAUNCH_BACKWARD);
    V1_BATCH_CASE(2, 2, LAUNCH_BACKWARD);
    V1_BATCH_CASE(4, 4, LAUNCH_BACKWARD);
    V1_BATCH_CASE(8, 4, LAUNCH_BACKWARD);
    V1_BATCH_CASE(16, 4, LAUNCH_BACKWARD);
    V1_BATCH_CASE(32, V1_BATCH32_TILE, LAUNCH_BACKWARD);
    V1_BATCH_CASE(64, 4, LAUNCH_BACKWARD);
    V1_BATCH_CASE(128, 4, LAUNCH_BACKWARD);
    V1_BATCH_CASE(256, 4, LAUNCH_BACKWARD);
#endif
    default: {
      constexpr int kRuntimeTile = 4;
      const int64_t tiles = (batch + kRuntimeTile - 1) / kRuntimeTile;
      TF_RETURN_IF_ERROR(GpuLaunchKernel(
          BackwardRuntimeBatchKernel<T, W, kBasis, kRuntimeTile>,
          static_cast<int>(tiles * n_rows), kThreads, 0, device.stream(), batch,
          spikes.dim_size(1), n_post, n_basis, spikes.flat<T>().data(),
          current_grad.flat<T>().data(),
          weights.flat<W>().data(), post_ids.flat<uint32>().data(),
          synapse_types.flat<uint8>().data(), row_splits.flat<uint32>().data(),
          edge_ids.flat<uint32>().data(), nonempty_rows.flat<uint32>().data(),
          n_rows, basis.flat<T>().data(), dampening.flat<T>().data(),
          spike_grad->flat<T>().data(), weight_grad->flat<float>().data()));
    }
  }
#undef LAUNCH_BACKWARD
#undef LAUNCH_WARP_PER_ROW
  return OkStatus();
}

template <typename T, typename W, int kBasis>
Status LaunchPairProjectedBackward(
    OpKernelContext* context, const Tensor& spikes, const Tensor& current_grad,
    const Tensor& weights, const Tensor& post_ids, const Tensor& synapse_types,
    const Tensor& row_splits, const Tensor& edge_ids,
    const Tensor& nonempty_rows, const Tensor& basis, const Tensor& dampening,
    const Tensor& pair_ids, const Tensor& pair_posts, const Tensor& pair_types,
    int n_post, Tensor* spike_grad, Tensor* weight_grad) {
  const int64_t n_rows = nonempty_rows.NumElements();
  const int64_t n_pairs = pair_posts.NumElements();
  const int64_t batch = spikes.dim_size(0);
  if (batch == 0 || n_rows == 0) return OkStatus();
  if (batch != 32) {
    return errors::InvalidArgument("pair-projected backward requires batch 32");
  }
  const int n_basis = basis.dim_size(1);
  auto device = context->eigen_device<GPUDevice>();
  Tensor projected_tensor;
  const int64_t projected_elements = batch * n_pairs;
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DT_FLOAT, TensorShape({projected_elements}), &projected_tensor));
  float* projected = projected_tensor.flat<float>().data();
  constexpr int kProjectionThreads = 256;
  constexpr int kPairsPerTile = 32;
  // Project each distinct (post, synapse type) pair once, pair-major so the
  // backward kernel reads one line per pair.
  TF_RETURN_IF_ERROR(GpuLaunchKernel(
      PreprojectPairsTiledKernel<T, kBasis, 32, kPairsPerTile>,
      static_cast<int>((n_pairs + kPairsPerTile - 1) / kPairsPerTile),
      kProjectionThreads, 0, device.stream(), n_post, n_basis, n_pairs,
      current_grad.flat<T>().data(), basis.flat<T>().data(),
      pair_posts.flat<uint32>().data(), pair_types.flat<uint8>().data(),
      projected));
#if V1_PACKED_BACKWARD
  if constexpr (std::is_same<T, Eigen::half>::value) {
    // Four FP32 batch samples per lane: eight lanes per edge, four edge slots
    // per warp, two warps per CSR row, 32 edges per warp iteration.
    constexpr int kPack = 4;
    constexpr int kPackedWarps = 2;
    constexpr int kPackedTile = 32;
    TF_RETURN_IF_ERROR(GpuLaunchKernel(
        BackwardPackedRowKernel<W, kPack, kPackedWarps, kPackedTile>,
        static_cast<int>(n_rows), 32 * kPackedWarps, 0, device.stream(),
        spikes.dim_size(1), spikes.flat<T>().data(), projected,
        weights.flat<W>().data(), pair_ids.flat<uint32>().data(),
        edge_ids.flat<uint32>().data(), row_splits.flat<uint32>().data(),
        nonempty_rows.flat<uint32>().data(), n_rows, dampening.flat<T>().data(),
        spike_grad->flat<T>().data(), weight_grad->flat<float>().data()));
  } else
#endif
  {
    constexpr int kLaneThreads = 128;
    constexpr int kLaneWarps = kLaneThreads / 32;
    TF_RETURN_IF_ERROR(GpuLaunchKernel(
        BackwardBatchLaneKernel<T, W, 32, kLaneWarps>,
        static_cast<int>((n_rows + kLaneWarps - 1) / kLaneWarps), kLaneThreads,
        0, device.stream(), spikes.dim_size(1), spikes.flat<T>().data(),
        projected, weights.flat<W>().data(), pair_ids.flat<uint32>().data(),
        edge_ids.flat<uint32>().data(), row_splits.flat<uint32>().data(),
        nonempty_rows.flat<uint32>().data(), n_rows, dampening.flat<T>().data(),
        spike_grad->flat<T>().data(), weight_grad->flat<float>().data()));
  }
  return OkStatus();
}

#undef V1_BATCH_CASE

template <typename T, typename W>
class V1CsrForwardOp : public OpKernel {
 public:
  explicit V1CsrForwardOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("n_post", &n_post_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& spikes = context->input(0);
    const Tensor& active = context->input(1);
    const Tensor& weights = context->input(2);
    const Tensor& post_ids = context->input(3);
    const Tensor& synapse_types = context->input(4);
    const Tensor& row_splits = context->input(5);
    const Tensor& edge_ids = context->input(6);
    const Tensor& basis = context->input(7);
    const Tensor& initial = context->input(8);
    OP_REQUIRES(context, spikes.dims() == 2,
                errors::InvalidArgument("spikes must be rank two"));
    OP_REQUIRES(context, active.dims() == 2 && active.dim_size(1) == 2,
                errors::InvalidArgument("active_indices must have shape [N,2]"));
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
                     context->forward_input_or_allocate_output({8}, 0, shape,
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
                                  context, spikes, active, weights, post_ids,
                                  synapse_types, row_splits, edge_ids, basis,
                                  n_post_, output));
    } else {
      OP_REQUIRES_OK(context, LaunchForward<T, W, 0>(
                                  context, spikes, active, weights, post_ids,
                                  synapse_types, row_splits, edge_ids, basis,
                                  n_post_, output));
    }
  }

 private:
  int n_post_;
};

template <typename T, typename W>
class V1CsrBackwardOp : public OpKernel {
 public:
  explicit V1CsrBackwardOp(OpKernelConstruction* context) : OpKernel(context) {
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
    OP_REQUIRES(context, spikes.dims() == 2 && basis.dims() == 2,
                errors::InvalidArgument("spikes and basis must be rank two"));
    OP_REQUIRES(context, basis.dim_size(1) > 0,
                errors::InvalidArgument("basis dimension must be positive"));
    OP_REQUIRES(context,
                current_grad.dims() == 2 &&
                    current_grad.dim_size(0) == spikes.dim_size(0) * n_post_ &&
                    current_grad.dim_size(1) == basis.dim_size(1),
                errors::InvalidArgument("current_grad has an incompatible shape"));
    OP_REQUIRES(context, dampening.NumElements() == 1,
                errors::InvalidArgument("dampening must be scalar"));
    Tensor* spike_grad;
    Tensor* weight_grad;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, spikes.shape(), &spike_grad));
    OP_REQUIRES_OK(context, context->allocate_output(
                                1, TensorShape({n_edges_}), &weight_grad));
    auto device = context->eigen_device<GPUDevice>();
    cudaMemsetAsync(spike_grad->flat<T>().data(), 0,
                    spike_grad->NumElements() * sizeof(T), device.stream());
    cudaMemsetAsync(weight_grad->flat<float>().data(), 0,
                    weight_grad->NumElements() * sizeof(float), device.stream());
    if (basis.dim_size(1) == 4) {
      OP_REQUIRES_OK(context, LaunchBackward<T, W, 4>(
                                  context, spikes, current_grad, weights,
                                  post_ids, synapse_types, row_splits, edge_ids,
                                  nonempty_rows, basis, dampening, n_post_,
                                  spike_grad, weight_grad));
    } else {
      OP_REQUIRES_OK(context, LaunchBackward<T, W, 0>(
                                  context, spikes, current_grad, weights,
                                  post_ids, synapse_types, row_splits, edge_ids,
                                  nonempty_rows, basis, dampening, n_post_,
                                  spike_grad, weight_grad));
    }
  }

 private:
  int n_post_;
  int n_edges_;
};

template <typename T, typename W>
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
    OP_REQUIRES(context, spikes.dims() == 2 && spikes.dim_size(0) == 32,
                errors::InvalidArgument("spikes must be [32,n_pre]"));
    OP_REQUIRES(context, basis.dims() == 2 && basis.dim_size(1) == 4,
                errors::InvalidArgument("basis must have four columns"));
    OP_REQUIRES(context, pair_ids.NumElements() == post_ids.NumElements(),
                errors::InvalidArgument("pair_ids must align with CSR edges"));
    OP_REQUIRES(context, pair_posts.NumElements() == pair_types.NumElements(),
                errors::InvalidArgument("pair metadata lengths differ"));
    Tensor* spike_grad;
    Tensor* weight_grad;
    OP_REQUIRES_OK(context, context->allocate_output(0, spikes.shape(), &spike_grad));
    OP_REQUIRES_OK(context, context->allocate_output(
                                1, TensorShape({n_edges_}), &weight_grad));
    auto device = context->eigen_device<GPUDevice>();
    // Rows with no edges are never visited, so the spike gradient is cleared.
    // The weight gradient is not: both backward kernels write every CSR
    // position exactly once, and `nonempty_rows` covers every position, so
    // clearing 321 MiB first would only cost write bandwidth.
    cudaMemsetAsync(spike_grad->flat<T>().data(), 0,
                    spike_grad->NumElements() * sizeof(T), device.stream());
    OP_REQUIRES_OK(context, LaunchPairProjectedBackward<T, W, 4>(
                                context, spikes, current_grad, weights,
                                post_ids, synapse_types, row_splits, edge_ids,
                                nonempty_rows, basis, dampening, pair_ids,
                                pair_posts, pair_types, n_post_, spike_grad,
                                weight_grad));
  }

 private:
  int n_post_;
  int n_edges_;
};

#ifndef V1_KERNEL_IMPLEMENTATION_ONLY
#define REGISTER_TYPE(T)                                                   \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("V1CsrForward").Device(DEVICE_GPU).TypeConstraint<T>("T"),     \
      V1CsrForwardOp<T, float>);                                           \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("V1CsrBackward").Device(DEVICE_GPU).TypeConstraint<T>("T"),    \
      V1CsrBackwardOp<T, float>);                                          \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("V1CsrBackwardPairProjected")                                  \
          .Device(DEVICE_GPU).TypeConstraint<T>("T"),                      \
      V1CsrBackwardPairProjectedOp<T, float>);

TF_CALL_half(REGISTER_TYPE);
TF_CALL_float(REGISTER_TYPE);
#undef REGISTER_TYPE
#endif

#endif

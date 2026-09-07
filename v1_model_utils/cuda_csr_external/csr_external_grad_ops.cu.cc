#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include <cuda_runtime.h>
#include <algorithm>
#include <cuda_fp16.h>
#include <type_traits>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

using namespace tensorflow;
using GPUDevice = Eigen::GpuDevice;

#ifndef V1_EXTERNAL_THREADS
#define V1_EXTERNAL_THREADS 128
#endif
#ifndef V1_EXTERNAL_BATCH32_TILE
#define V1_EXTERNAL_BATCH32_TILE 4
#endif
#ifndef V1_EXTERNAL_HALF2
#define V1_EXTERNAL_HALF2 0
#endif
#ifndef V1_DIRECT_CSR
#define V1_DIRECT_CSR 0
#endif

// See the recurrent operator: with direct-CSR weights the caller supplies
// `weights` and receives `weight_grad` in CSR edge order, removing the random
// `edge_ids[csr]` indirection from the inner loop.
#if V1_DIRECT_CSR
#define V1_EDGE_INDEX(csr) (csr)
#else
#define V1_EDGE_INDEX(csr) (edge_ids[csr])
#endif

// Nothing dereferences `edge_ids` under direct CSR, so the caller sends an
// empty tensor rather than one dead uint32 per edge. Requiring exactly that
// keeps the two sides from drifting: a full-length permutation arriving here
// would mean the caller still believes the kernels gather through it, and a
// short one under the gathering build would read out of bounds.
constexpr int64_t kEdgeIdsPerEdge = V1_DIRECT_CSR ? 0 : 1;

constexpr int kThreads = V1_EXTERNAL_THREADS;

template <typename T>
__device__ __forceinline__ float4 LoadBkgFour(const T* values) {
  if constexpr (std::is_same<T, Eigen::half>::value) {
    const uint2 packed = *reinterpret_cast<const uint2*>(values);
    const float2 low =
        __half22float2(*reinterpret_cast<const __half2*>(&packed.x));
    const float2 high =
        __half22float2(*reinterpret_cast<const __half2*>(&packed.y));
    return make_float4(low.x, low.y, high.x, high.y);
  }
  return make_float4(values[0], values[1], values[2], values[3]);
}

template <typename T>
__device__ __forceinline__ void StoreBkgFour(T* values, float4 result) {
  if constexpr (std::is_same<T, Eigen::half>::value) {
    const __half2 low = __floats2half2_rn(result.x, result.y);
    const __half2 high = __floats2half2_rn(result.z, result.w);
    const uint2 packed =
        make_uint2(*reinterpret_cast<const unsigned*>(&low),
                   *reinterpret_cast<const unsigned*>(&high));
    *reinterpret_cast<uint2*>(values) = packed;
  } else {
    values[0] = result.x;
    values[1] = result.y;
    values[2] = result.z;
    values[3] = result.w;
  }
}

template <typename T>
__global__ void BkgGatherKernel(
    int n_pre, int n_post, const T* activity, const float* weights,
    const uint32* rows, const uint32* pre_ids, const uint32* edge_ids,
    const uint8* types, const T* basis, const T* initial, T* output) {
  const int post = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (post >= n_post) return;
  const int batch = blockIdx.y;
  const int64_t index = static_cast<int64_t>(batch) * n_post + post;
  float4 sums = initial ? LoadBkgFour(initial + index * 4)
                        : make_float4(0, 0, 0, 0);
#pragma unroll
  for (uint32 incoming = rows[post]; incoming < rows[post] + 4; ++incoming) {
    const float spike =
        static_cast<float>(activity[batch * n_pre + pre_ids[incoming]]);
    if (spike == 0.0f) continue;
    const float weighted = weights[edge_ids[incoming]] * spike;
    const float4 projection = LoadBkgFour(basis + types[incoming] * 4);
    sums.x += weighted * projection.x;
    sums.y += weighted * projection.y;
    sums.z += weighted * projection.z;
    sums.w += weighted * projection.w;
  }
  StoreBkgFour(output + index * 4, sums);
}

template <typename T>
class BkgCsrForwardOp : public OpKernel {
 public:
  explicit BkgCsrForwardOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("n_post", &n_post_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& activity = context->input(0);
    const Tensor& weights = context->input(1);
    const Tensor& rows = context->input(2);
    const Tensor& pre_ids = context->input(3);
    const Tensor& edge_ids = context->input(4);
    const Tensor& types = context->input(5);
    const Tensor& basis = context->input(6);
    const Tensor& initial = context->input(7);
    OP_REQUIRES(context, activity.dims() == 2,
                errors::InvalidArgument("activity must be rank two"));
    OP_REQUIRES(context, basis.dims() == 2 && basis.dim_size(1) == 4,
                errors::InvalidArgument("basis must have four columns"));
    OP_REQUIRES(context, rows.NumElements() == n_post_ + 1,
                errors::InvalidArgument(
                    "incoming_row_splits must have n_post + 1 entries"));
    OP_REQUIRES(context,
                pre_ids.NumElements() == edge_ids.NumElements() &&
                    types.NumElements() == edge_ids.NumElements() &&
                    edge_ids.NumElements() == 4 * n_post_,
                errors::InvalidArgument(
                    "BKG fast path requires exactly four edges per post"));
    const int64_t count = activity.dim_size(0) * n_post_;
    const TensorShape shape({count, 4});
    OP_REQUIRES(context,
                initial.NumElements() == 0 || initial.shape() == shape,
                errors::InvalidArgument("initial must be empty or match output"));
    Tensor* output;
    OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                {7}, 0, shape, &output));
    if (count == 0) return;
    auto device = context->eigen_device<GPUDevice>();
    OP_REQUIRES_OK(context, GpuLaunchKernel(
                                BkgGatherKernel<T>,
                                dim3((n_post_ + 127) / 128,
                                     activity.dim_size(0)),
                                128, 0, device.stream(),
                                static_cast<int>(activity.dim_size(1)), n_post_,
                                activity.flat<T>().data(),
                                weights.flat<float>().data(),
                                rows.flat<uint32>().data(),
                                pre_ids.flat<uint32>().data(),
                                edge_ids.flat<uint32>().data(),
                                types.flat<uint8>().data(), basis.flat<T>().data(),
                                initial.NumElements() ? initial.flat<T>().data()
                                                      : nullptr,
                                output->flat<T>().data()));
  }

 private:
  int n_post_;
};

template <typename T>
constexpr int LaunchThreads() {
  return std::is_same<T, Eigen::half>::value ? kThreads : 256;
}

template <typename T>
__device__ __forceinline__ float AsFloat(T value) {
  return static_cast<float>(value);
}

template <typename T, int kBasis>
struct BasisProjection {
  __device__ __forceinline__ static float Apply(
      const T* upstream, const T* basis, int type, int n_basis) {
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
  __device__ __forceinline__ static float Apply(
      const T* upstream, const T* basis, int type, int n_basis) {
    float result = 0.0f;
    for (int receptor = 0; receptor < n_basis; ++receptor) {
      result += AsFloat(upstream[receptor]) *
                AsFloat(basis[type * n_basis + receptor]);
    }
    return result;
  }
};

#if V1_EXTERNAL_HALF2
template <>
struct BasisProjection<Eigen::half, 4> {
  __device__ __forceinline__ static float Apply(
      const Eigen::half* upstream, const Eigen::half* basis, int type,
      int n_basis) {
    const float2 u01 = __half22float2(
        *reinterpret_cast<const __half2*>(upstream));
    const float2 u23 = __half22float2(
        *reinterpret_cast<const __half2*>(upstream + 2));
    const Eigen::half* type_basis = basis + type * 4;
    const float2 b01 = __half22float2(
        *reinterpret_cast<const __half2*>(type_basis));
    const float2 b23 = __half22float2(
        *reinterpret_cast<const __half2*>(type_basis + 2));
    return fmaf(u01.x, b01.x,
                fmaf(u01.y, b01.y,
                     fmaf(u23.x, b23.x, u23.y * b23.y)));
  }
};
#endif

template <typename T, int kBasis, int kBatch, int kTile>
__global__ void WeightBackwardStaticBatchKernel(
    int64_t n_pre, int n_post, int n_basis, const T* activity,
    const T* current_grad, const uint32* post_ids,
    const uint8* synapse_types, const uint32* row_splits,
    const uint32* edge_ids, const uint32* nonempty_rows, int64_t n_rows,
    const T* basis, float* weight_grad) {
  static_assert(kBatch % kTile == 0, "batch tiles must divide the batch");
  const int64_t tile_id = blockIdx.x / n_rows;
  const int64_t row_id = blockIdx.x - tile_id * n_rows;
  if (row_id >= n_rows) return;
  const int first_batch = static_cast<int>(tile_id) * kTile;
  const uint32 pre = nonempty_rows[row_id];
  for (uint32 csr = row_splits[pre] + threadIdx.x;
       csr < row_splits[pre + 1]; csr += blockDim.x) {
    const uint32 edge = V1_EDGE_INDEX(csr);
    const uint32 post = post_ids[csr];
    const uint32 type = synapse_types[csr];
    float tile_weight_grad = 0.0f;
#pragma unroll
    for (int offset = 0; offset < kTile; ++offset) {
      const int batch = first_batch + offset;
      const T* upstream =
          current_grad +
          (batch * static_cast<int64_t>(n_post) + post) * n_basis;
      tile_weight_grad +=
          BasisProjection<T, kBasis>::Apply(upstream, basis, type, n_basis) *
          AsFloat(activity[batch * n_pre + pre]);
    }
    if (kBatch == kTile) {
      weight_grad[edge] = tile_weight_grad;
    } else if (tile_weight_grad != 0.0f) {
      atomicAdd(weight_grad + edge, tile_weight_grad);
    }
  }
}

template <typename T, int kBasis, int kTile>
__global__ void WeightBackwardRuntimeBatchKernel(
    int64_t batch_size, int64_t n_pre, int n_post, int n_basis,
    const T* activity, const T* current_grad, const uint32* post_ids,
    const uint8* synapse_types, const uint32* row_splits,
    const uint32* edge_ids, const uint32* nonempty_rows, int64_t n_rows,
    const T* basis, float* weight_grad) {
  const int64_t tile_id = blockIdx.x / n_rows;
  const int64_t row_id = blockIdx.x - tile_id * n_rows;
  if (row_id >= n_rows) return;
  const int64_t first_batch = tile_id * kTile;
  const uint32 pre = nonempty_rows[row_id];
  for (uint32 csr = row_splits[pre] + threadIdx.x;
       csr < row_splits[pre + 1]; csr += blockDim.x) {
    const uint32 edge = V1_EDGE_INDEX(csr);
    const uint32 post = post_ids[csr];
    const uint32 type = synapse_types[csr];
    float tile_weight_grad = 0.0f;
#pragma unroll
    for (int offset = 0; offset < kTile; ++offset) {
      const int64_t batch = first_batch + offset;
      if (batch < batch_size) {
        const T* upstream =
            current_grad +
            (batch * static_cast<int64_t>(n_post) + post) * n_basis;
        tile_weight_grad +=
            BasisProjection<T, kBasis>::Apply(upstream, basis, type, n_basis) *
            AsFloat(activity[batch * n_pre + pre]);
      }
    }
    if (batch_size <= kTile) {
      weight_grad[edge] = tile_weight_grad;
    } else if (tile_weight_grad != 0.0f) {
      atomicAdd(weight_grad + edge, tile_weight_grad);
    }
  }
}

// ---------------------------------------------------------------------------
// Packed batch-lane rewrite of the external (LGN/BKG) gradient kernels.
//
// These mirror the recurrent operator's promoted design. Giving a lane `kPack`
// consecutive FP32 batch samples shrinks an edge from 32 lanes to 32 / kPack,
// so a warp splits into that many independent edge slots and one 16-byte load
// serves all of them. Nothing about the arithmetic changes: the projection
// stays FP32 and both reductions stay FP32.
// ---------------------------------------------------------------------------

__device__ __forceinline__ uint32 ExtLoadIndex(const uint32* address) {
  return __ldcs(address);
}

__device__ __forceinline__ float ExtLoadWeight(const float* address) {
  return __ldcs(address);
}

__device__ __forceinline__ void ExtStoreGrad(float* address, float value) {
  __stcs(address, value);
}

template <int kHalf, int kMask>
__device__ __forceinline__ void ExtButterflyReduce(float* partial, int lane) {
  const bool upper = (lane & kMask) != 0;
#pragma unroll
  for (int index = 0; index < kHalf; ++index) {
    const float keep = upper ? partial[kHalf + index] : partial[index];
    const float send = upper ? partial[index] : partial[kHalf + index];
    partial[index] = keep + __shfl_xor_sync(0xffffffff, send, kMask);
  }
  if constexpr (kHalf > 1) ExtButterflyReduce<kHalf / 2, kMask * 2>(partial, lane);
}

template <int kBits>
__device__ __forceinline__ int ExtReverseBits(int value) {
  int result = 0;
#pragma unroll
  for (int bit = 0; bit < kBits; ++bit) {
    result |= ((value >> bit) & 1) << (kBits - 1 - bit);
  }
  return result;
}

template <int kPack>
struct ExtPackedProjection {
  __device__ __forceinline__ static void Load(const float* source, float* out);
};

template <>
struct ExtPackedProjection<4> {
  __device__ __forceinline__ static void Load(const float* source, float* out) {
    const ::float4 raw = *reinterpret_cast<const ::float4*>(source);
    out[0] = raw.x;
    out[1] = raw.y;
    out[2] = raw.z;
    out[3] = raw.w;
  }
};

// Compact projection with a shared-memory transpose. The pair-major output the
// row kernels want makes the direct mapping give consecutive threads
// consecutive batch samples of one pair, whose current gradients are
// `n_post * n_basis` apart; each lane then pulls its own 32-byte sector for the
// 8 bytes it needs. Reading pair-contiguous and writing batch-contiguous
// coalesces both halves.
// One CSR row per block leaves most of the GPU idle when a population has few
// presynaptic sources: BKG has 100 rows against a 4,512-block resident
// capacity. Splitting each row across `blockIdx.y` restores it. LGN's 17,400
// rows already exceed the capacity, so it gets one split and is unaffected.
// The compact pair-projected path writes every CSR position exactly once, so
// it needs no cleared output. Every other weight-gradient path splits the batch
// across tiles and accumulates with `atomicAdd`, which does. `LaunchWeightBackward`
// selects on exactly this predicate, so the two must agree.
template <typename T>
__host__ __forceinline__ bool UsesCompactWeightPath(int64_t batch, int n_basis,
                                                    int64_t n_pairs) {
  return std::is_same<T, Eigen::half>::value && batch == 32 && n_basis == 4 &&
         n_pairs > 0;
}

__host__ __forceinline__ uint32 RowSplitCount(int64_t n_rows) {
  constexpr int64_t kTargetBlocks = 4512;
  if (n_rows <= 0) return 1;
  const int64_t splits = (kTargetBlocks + n_rows - 1) / n_rows;
  return static_cast<uint32>(std::min<int64_t>(64, std::max<int64_t>(1, splits)));
}

template <typename T, int kBasis, int kBatch, int kPairsPerTile>
__global__ void ProjectCompactTiledKernel(
    int n_post, int n_basis, int64_t n_pairs, const uint32* pair_posts,
    const uint8* pair_types, const T* current_grad, const T* basis,
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

// Weight gradient. `blockIdx.y` splits one CSR row across several blocks, which
// the BKG population needs: it has 100 rows of about 8,150 edges each, so the
// row-per-block decomposition fills 2% of a GPU that holds 4,512 blocks. Splits
// are aligned to the `kWarps * kTile` granularity so a split boundary never
// adds tail waste beyond the row's own.
template <typename T, int kPack, int kWarps, int kTile>
__global__ __launch_bounds__(32 * kWarps) void WeightBackwardPackedKernel(
    int64_t n_pre, const T* activity, const float* projected,
    const uint32* pair_ids, const uint32* edge_ids, const uint32* row_splits,
    const uint32* nonempty_rows, int64_t n_rows, uint32 splits,
    float* weight_grad) {
  constexpr int kBatch = 32;
  constexpr int kLanesPerEdge = kBatch / kPack;
  constexpr int kSlots = 32 / kLanesPerEdge;
  constexpr int kPerSlot = kTile / kSlots;
  constexpr int kIndexBits = kPerSlot == 8 ? 3 : (kPerSlot == 4 ? 2 : 1);
  constexpr uint32 kGrain = kWarps * kTile;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int slot = lane / kLanesPerEdge;
  const int sub = lane % kLanesPerEdge;
  const int64_t row_id = blockIdx.x;
  if (row_id >= n_rows) return;

  const uint32 pre = nonempty_rows[row_id];
  const uint32 row_start = row_splits[pre];
  const uint32 row_end = row_splits[pre + 1];
  const uint32 chunks = (row_end - row_start + kGrain - 1) / kGrain;
  const uint32 chunks_per_split = (chunks + splits - 1) / splits;
  const uint32 start = row_start + blockIdx.y * chunks_per_split * kGrain;
  if (start >= row_end) return;
  const uint32 end = min(row_end, start + chunks_per_split * kGrain);

  float sample[kPack];
#pragma unroll
  for (int index = 0; index < kPack; ++index) {
    sample[index] = AsFloat(
        activity[static_cast<int64_t>(sub * kPack + index) * n_pre + pre]);
  }
  const int target = ExtReverseBits<kIndexBits>(sub & (kPerSlot - 1));

  for (uint32 base = start + warp * kTile; base < end; base += kGrain) {
    const uint32 edge_lane = base + lane;
    const bool own = lane < kTile && edge_lane < end;
    const uint32 my_pair = own ? ExtLoadIndex(pair_ids + edge_lane) : 0u;
    float partial[kPerSlot];
#pragma unroll
    for (int step = 0; step < kPerSlot; ++step) {
      const int column = kSlots * step + slot;
      const uint32 pair = __shfl_sync(0xffffffff, my_pair, column);
      float value[kPack];
      if (base + column < end) {
        ExtPackedProjection<kPack>::Load(
            projected + static_cast<int64_t>(pair) * kBatch + sub * kPack, value);
      } else {
#pragma unroll
        for (int index = 0; index < kPack; ++index) value[index] = 0.0f;
      }
      float sum = 0.0f;
#pragma unroll
      for (int index = 0; index < kPack; ++index) sum += value[index] * sample[index];
      partial[step] = sum;
    }
    if constexpr (kPerSlot > 1) ExtButterflyReduce<kPerSlot / 2, 1>(partial, lane);
#pragma unroll
    for (int mask = kPerSlot; mask < kLanesPerEdge; mask <<= 1) {
      partial[0] += __shfl_xor_sync(0xffffffff, partial[0], mask);
    }
    const uint32 edge = base + kSlots * target + slot;
    if (sub < kPerSlot && edge < end) {
      ExtStoreGrad(weight_grad + V1_EDGE_INDEX(edge), partial[0]);
    }
  }
}

// Activity gradient. The batch-lane mapping removes the reduction rather than
// replacing it: each lane owns `kPack` batch samples and accumulates the whole
// row itself, so there is no block-wide tree and no 8 KiB scratch. It also
// covers all 32 samples in one pass, where the tiled form re-read every edge's
// `weights` and `pair_ids` once per eight-sample tile.
// `splits` divides a row across `blockIdx.y`, for populations whose row count
// alone cannot fill the GPU: BKG has 100 rows against a 4,512-block capacity.
// Split blocks cannot each own the output, so they accumulate into a float
// scratch that `CastActivityGradKernel` then narrows; with `splits == 1` the
// scratch is skipped and the block writes the output directly.
template <typename T, int kPack, int kWarps, int kTile>
__global__ __launch_bounds__(32 * kWarps) void ActivityBackwardPackedKernel(
    int64_t n_pre, const float* projected, const float* weights,
    const uint32* pair_ids, const uint32* edge_ids, const uint32* row_splits,
    const uint32* nonempty_rows, int64_t n_rows, uint32 splits, float* scratch,
    T* activity_grad) {
  constexpr int kBatch = 32;
  constexpr int kLanesPerEdge = kBatch / kPack;
  constexpr int kSlots = 32 / kLanesPerEdge;
  constexpr int kPerSlot = kTile / kSlots;
  constexpr uint32 kGrain = kWarps * kTile;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int slot = lane / kLanesPerEdge;
  const int sub = lane % kLanesPerEdge;
  const int64_t row_id = blockIdx.x;
  if (row_id >= n_rows) return;

  __shared__ float partials[kWarps][32];

  const uint32 pre = nonempty_rows[row_id];
  const uint32 row_start = row_splits[pre];
  const uint32 row_end = row_splits[pre + 1];
  const uint32 chunks = (row_end - row_start + kGrain - 1) / kGrain;
  const uint32 chunks_per_split = (chunks + splits - 1) / splits;
  const uint32 start = row_start + blockIdx.y * chunks_per_split * kGrain;
  if (start >= row_end) return;
  const uint32 end = min(row_end, start + chunks_per_split * kGrain);
  float grad[kPack];
#pragma unroll
  for (int index = 0; index < kPack; ++index) grad[index] = 0.0f;

  for (uint32 base = start + warp * kTile; base < end; base += kWarps * kTile) {
    const uint32 edge_lane = base + lane;
    const bool own = lane < kTile && edge_lane < end;
    const uint32 my_pair = own ? ExtLoadIndex(pair_ids + edge_lane) : 0u;
    const float my_weight =
        own ? ExtLoadWeight(weights + V1_EDGE_INDEX(edge_lane)) : 0.0f;
#pragma unroll
    for (int step = 0; step < kPerSlot; ++step) {
      const int column = kSlots * step + slot;
      const uint32 pair = __shfl_sync(0xffffffff, my_pair, column);
      const float weight = __shfl_sync(0xffffffff, my_weight, column);
      if (base + column < end) {
        float value[kPack];
        ExtPackedProjection<kPack>::Load(
            projected + static_cast<int64_t>(pair) * kBatch + sub * kPack, value);
#pragma unroll
        for (int index = 0; index < kPack; ++index) {
          grad[index] += value[index] * weight;
        }
      }
    }
  }

  // Every slot accumulated the same batch samples from a different edge subset.
#pragma unroll
  for (int mask = kLanesPerEdge; mask < 32; mask <<= 1) {
#pragma unroll
    for (int index = 0; index < kPack; ++index) {
      grad[index] += __shfl_xor_sync(0xffffffff, grad[index], mask);
    }
  }
  if (lane < kLanesPerEdge) {
#pragma unroll
    for (int index = 0; index < kPack; ++index) {
      partials[warp][sub * kPack + index] = grad[index];
    }
  }
  __syncthreads();
  if (warp == 0) {
    float total = 0.0f;
#pragma unroll
    for (int source = 0; source < kWarps; ++source) total += partials[source][lane];
    const int64_t offset = static_cast<int64_t>(lane) * n_pre + pre;
    if (scratch != nullptr) {
      atomicAdd(scratch + offset, total);
    } else {
      activity_grad[offset] = static_cast<T>(total);
    }
  }
}

template <typename T>
__global__ void CastActivityGradKernel(int64_t elements, const float* scratch,
                                       T* activity_grad) {
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < elements; index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    activity_grad[index] = static_cast<T>(scratch[index]);
  }
}

template <typename T, int kBasis, int kTile>
__global__ void ActivityBackwardKernel(
    int64_t batch_size, int64_t n_pre, int n_post, int n_basis,
    const T* current_grad, const float* weights, const uint32* post_ids,
    const uint8* synapse_types, const uint32* row_splits,
    const uint32* edge_ids, const uint32* nonempty_rows, int64_t n_rows,
    const T* basis, T* activity_grad) {
  __shared__ float partial[kTile][256];
  const int64_t tile_id = blockIdx.x / n_rows;
  const int64_t row_id = blockIdx.x - tile_id * n_rows;
  if (row_id >= n_rows) return;
  const int64_t first_batch = tile_id * kTile;
  const uint32 pre = nonempty_rows[row_id];
  float local[kTile] = {};
  for (uint32 csr = row_splits[pre] + threadIdx.x;
       csr < row_splits[pre + 1]; csr += blockDim.x) {
    const uint32 edge = V1_EDGE_INDEX(csr);
#pragma unroll
    for (int offset = 0; offset < kTile; ++offset) {
      const int64_t batch = first_batch + offset;
      if (batch < batch_size) {
        local[offset] += weights[edge] * BasisProjection<T, kBasis>::Apply(
            current_grad +
                (batch * static_cast<int64_t>(n_post) + post_ids[csr]) *
                    n_basis,
            basis, synapse_types[csr], n_basis);
      }
    }
  }
#pragma unroll
  for (int offset = 0; offset < kTile; ++offset) {
    partial[offset][threadIdx.x] = local[offset];
  }
  __syncthreads();
  for (int stride = 128; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
#pragma unroll
      for (int offset = 0; offset < kTile; ++offset) {
        partial[offset][threadIdx.x] += partial[offset][threadIdx.x + stride];
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
#pragma unroll
    for (int offset = 0; offset < kTile; ++offset) {
      const int64_t batch = first_batch + offset;
      if (batch < batch_size) {
        activity_grad[batch * n_pre + pre] =
            static_cast<T>(partial[offset][0]);
      }
    }
  }
}

#define EXTERNAL_BATCH_CASE(BATCH, TILE, LAUNCH) \
  case BATCH:                                    \
    LAUNCH(BATCH, TILE);                         \
    break

template <typename T, int kBasis>
Status LaunchWeightBackward(
    OpKernelContext* context, const Tensor& activity,
    const Tensor& current_grad, const Tensor& post_ids,
    const Tensor& synapse_types, const Tensor& row_splits,
    const Tensor& edge_ids, const Tensor& nonempty_rows, const Tensor& basis,
    const Tensor& pair_ids, const Tensor& pair_posts, const Tensor& pair_types,
    int n_post, Tensor* weight_grad) {
  const int64_t n_rows = nonempty_rows.NumElements();
  const int64_t batch = activity.dim_size(0);
  if (batch == 0 || n_rows == 0) return OkStatus();
  const int n_basis = basis.dim_size(1);
  auto device = context->eigen_device<GPUDevice>();
  if (std::is_same<T, Eigen::half>::value && batch == 32 && kBasis == 4 &&
      pair_posts.NumElements() > 0) {
    Tensor projected_tensor;
    const int64_t projected_elements = pair_posts.NumElements() * batch;
    TF_RETURN_IF_ERROR(context->allocate_temp(
        DT_FLOAT, TensorShape({projected_elements}), &projected_tensor));
    constexpr int kPairsPerTile = 32;
    TF_RETURN_IF_ERROR(GpuLaunchKernel(
        ProjectCompactTiledKernel<T, kBasis, 32, kPairsPerTile>,
        static_cast<int>((pair_posts.NumElements() + kPairsPerTile - 1) /
                         kPairsPerTile),
        256, 0, device.stream(), n_post, n_basis, pair_posts.NumElements(),
        pair_posts.flat<uint32>().data(), pair_types.flat<uint8>().data(),
        current_grad.flat<T>().data(), basis.flat<T>().data(),
        projected_tensor.flat<float>().data()));
    constexpr int kPack = 4;
    constexpr int kWarps = 2;
    constexpr int kTile = 32;
    const uint32 splits = RowSplitCount(n_rows);
    TF_RETURN_IF_ERROR(GpuLaunchKernel(
        WeightBackwardPackedKernel<T, kPack, kWarps, kTile>,
        dim3(static_cast<unsigned>(n_rows), splits), 32 * kWarps, 0,
        device.stream(), activity.dim_size(1), activity.flat<T>().data(),
        projected_tensor.flat<float>().data(), pair_ids.flat<uint32>().data(),
        edge_ids.flat<uint32>().data(), row_splits.flat<uint32>().data(),
        nonempty_rows.flat<uint32>().data(), n_rows, splits,
        weight_grad->flat<float>().data()));
    return OkStatus();
  }
#define LAUNCH_STATIC(BATCH, TILE)                                        \
  TF_RETURN_IF_ERROR(GpuLaunchKernel(                                    \
      WeightBackwardStaticBatchKernel<T, kBasis, BATCH, TILE>,           \
      static_cast<int>(n_rows * (BATCH / TILE)), LaunchThreads<T>(), 0, \
      device.stream(), activity.dim_size(1), n_post, n_basis,            \
      activity.flat<T>().data(), current_grad.flat<T>().data(),          \
      post_ids.flat<uint32>().data(), synapse_types.flat<uint8>().data(), \
      row_splits.flat<uint32>().data(), edge_ids.flat<uint32>().data(),  \
      nonempty_rows.flat<uint32>().data(), n_rows, basis.flat<T>().data(), \
      weight_grad->flat<float>().data()))
  switch (batch) {
    EXTERNAL_BATCH_CASE(1, 1, LAUNCH_STATIC);
    EXTERNAL_BATCH_CASE(2, 2, LAUNCH_STATIC);
    EXTERNAL_BATCH_CASE(4, 4, LAUNCH_STATIC);
    EXTERNAL_BATCH_CASE(8, 8, LAUNCH_STATIC);
    EXTERNAL_BATCH_CASE(16, 16, LAUNCH_STATIC);
    EXTERNAL_BATCH_CASE(32, V1_EXTERNAL_BATCH32_TILE, LAUNCH_STATIC);
    EXTERNAL_BATCH_CASE(64, 32, LAUNCH_STATIC);
    EXTERNAL_BATCH_CASE(128, 32, LAUNCH_STATIC);
    EXTERNAL_BATCH_CASE(256, 32, LAUNCH_STATIC);
    default: {
      constexpr int kRuntimeTile = 4;
      const int64_t tiles = (batch + kRuntimeTile - 1) / kRuntimeTile;
      TF_RETURN_IF_ERROR(GpuLaunchKernel(
          WeightBackwardRuntimeBatchKernel<T, kBasis, kRuntimeTile>,
          static_cast<int>(tiles * n_rows), kThreads, 0, device.stream(),
          batch, activity.dim_size(1), n_post, n_basis,
          activity.flat<T>().data(), current_grad.flat<T>().data(),
          post_ids.flat<uint32>().data(), synapse_types.flat<uint8>().data(),
          row_splits.flat<uint32>().data(), edge_ids.flat<uint32>().data(),
          nonempty_rows.flat<uint32>().data(), n_rows, basis.flat<T>().data(),
          weight_grad->flat<float>().data()));
    }
  }
#undef LAUNCH_STATIC
  return OkStatus();
}

template <typename T, int kBasis>
Status LaunchActivityBackward(
    OpKernelContext* context, const Tensor& current_grad, const Tensor& weights,
    const Tensor& post_ids, const Tensor& synapse_types,
    const Tensor& row_splits, const Tensor& edge_ids,
    const Tensor& nonempty_rows, const Tensor& basis, const Tensor& pair_ids,
    const Tensor& pair_posts, const Tensor& pair_types, int n_post,
    Tensor* activity_grad) {
  const int64_t n_rows = nonempty_rows.NumElements();
  const int64_t batch = current_grad.dim_size(0) / n_post;
  if (batch == 0 || n_rows == 0) return OkStatus();
  const int n_basis = basis.dim_size(1);
  auto device = context->eigen_device<GPUDevice>();
  if (std::is_same<T, Eigen::half>::value && batch == 32 && kBasis == 4 &&
      pair_posts.NumElements() > 0) {
    Tensor projected_tensor;
    const int64_t projected_elements = pair_posts.NumElements() * batch;
    TF_RETURN_IF_ERROR(context->allocate_temp(
        DT_FLOAT, TensorShape({projected_elements}), &projected_tensor));
    constexpr int kActivityPairsPerTile = 32;
    TF_RETURN_IF_ERROR(GpuLaunchKernel(
        ProjectCompactTiledKernel<T, kBasis, 32, kActivityPairsPerTile>,
        static_cast<int>((pair_posts.NumElements() + kActivityPairsPerTile - 1) /
                         kActivityPairsPerTile),
        256, 0, device.stream(), n_post, n_basis, pair_posts.NumElements(),
        pair_posts.flat<uint32>().data(), pair_types.flat<uint8>().data(),
        current_grad.flat<T>().data(), basis.flat<T>().data(),
        projected_tensor.flat<float>().data()));
    const uint32 activity_splits = RowSplitCount(n_rows);
    Tensor scratch_tensor;
    float* scratch = nullptr;
    const int64_t activity_elements = activity_grad->NumElements();
    if (activity_splits > 1) {
      TF_RETURN_IF_ERROR(context->allocate_temp(
          DT_FLOAT, TensorShape({activity_elements}), &scratch_tensor));
      scratch = scratch_tensor.flat<float>().data();
      cudaMemsetAsync(scratch, 0, activity_elements * sizeof(float),
                      device.stream());
    }
    TF_RETURN_IF_ERROR(GpuLaunchKernel(
        ActivityBackwardPackedKernel<T, 4, 2, 32>,
        dim3(static_cast<unsigned>(n_rows), activity_splits), 64, 0,
        device.stream(), row_splits.NumElements() - 1,
        projected_tensor.flat<float>().data(), weights.flat<float>().data(),
        pair_ids.flat<uint32>().data(), edge_ids.flat<uint32>().data(),
        row_splits.flat<uint32>().data(), nonempty_rows.flat<uint32>().data(),
        n_rows, activity_splits, scratch, activity_grad->flat<T>().data()));
    if (scratch != nullptr) {
      TF_RETURN_IF_ERROR(GpuLaunchKernel(
          CastActivityGradKernel<T>,
          static_cast<int>((activity_elements + 255) / 256), 256, 0,
          device.stream(), activity_elements, scratch,
          activity_grad->flat<T>().data()));
    }
    return OkStatus();
  }
  constexpr int kRuntimeTile = 4;
  const int64_t tiles = (batch + kRuntimeTile - 1) / kRuntimeTile;
  return GpuLaunchKernel(
      ActivityBackwardKernel<T, kBasis, kRuntimeTile>,
      static_cast<int>(tiles * n_rows), 256, 0, device.stream(), batch,
      row_splits.NumElements() - 1, n_post, n_basis,
      current_grad.flat<T>().data(), weights.flat<float>().data(),
      post_ids.flat<uint32>().data(), synapse_types.flat<uint8>().data(),
      row_splits.flat<uint32>().data(), edge_ids.flat<uint32>().data(),
      nonempty_rows.flat<uint32>().data(), n_rows, basis.flat<T>().data(),
      activity_grad->flat<T>().data());
}

#undef EXTERNAL_BATCH_CASE

template <typename T>
class ExternalCsrWeightBackwardOp : public OpKernel {
 public:
  explicit ExternalCsrWeightBackwardOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("n_post", &n_post_));
    OP_REQUIRES_OK(context, context->GetAttr("n_edges", &n_edges_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& activity = context->input(0);
    const Tensor& current_grad = context->input(1);
    const Tensor& post_ids = context->input(2);
    const Tensor& synapse_types = context->input(3);
    const Tensor& row_splits = context->input(4);
    const Tensor& edge_ids = context->input(5);
    const Tensor& nonempty_rows = context->input(6);
    const Tensor& basis = context->input(7);
    const Tensor& pair_ids = context->input(8);
    const Tensor& pair_posts = context->input(9);
    const Tensor& pair_types = context->input(10);
    OP_REQUIRES(context, activity.dims() == 2 && basis.dims() == 2,
                errors::InvalidArgument("activity and basis must be rank two"));
    OP_REQUIRES(context, basis.dim_size(1) > 0,
                errors::InvalidArgument("basis dimension must be positive"));
    OP_REQUIRES(context,
                current_grad.dims() == 2 &&
                    current_grad.dim_size(0) == activity.dim_size(0) * n_post_ &&
                    current_grad.dim_size(1) == basis.dim_size(1),
                errors::InvalidArgument("current_grad has an incompatible shape"));
    OP_REQUIRES(context,
                row_splits.NumElements() == activity.dim_size(1) + 1,
                errors::InvalidArgument("row_splits does not match activity width"));
    OP_REQUIRES(context, post_ids.NumElements() == n_edges_ &&
                                 synapse_types.NumElements() == n_edges_ &&
                                 edge_ids.NumElements() ==
                                     n_edges_ * kEdgeIdsPerEdge &&
                                 pair_ids.NumElements() == n_edges_,
                errors::InvalidArgument("edge metadata size mismatch"));
    OP_REQUIRES(context, pair_posts.NumElements() == pair_types.NumElements(),
                errors::InvalidArgument("pair metadata size mismatch"));
    Tensor* weight_grad;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, TensorShape({n_edges_}), &weight_grad));
    auto device = context->eigen_device<GPUDevice>();
    // Skipping this clear is worth 365 MiB of write bandwidth on LGN, but only
    // the compact path earns it: its packed kernel's row splits partition every
    // row, so each CSR position is written exactly once.
    if (!UsesCompactWeightPath<T>(activity.dim_size(0), basis.dim_size(1),
                                  pair_posts.NumElements())) {
      cudaMemsetAsync(weight_grad->flat<float>().data(), 0,
                      weight_grad->NumElements() * sizeof(float),
                      context->eigen_device<GPUDevice>().stream());
    }
    if (basis.dim_size(1) == 4) {
      OP_REQUIRES_OK(context, LaunchWeightBackward<T, 4>(
                                  context, activity, current_grad, post_ids,
                                  synapse_types, row_splits, edge_ids,
                                  nonempty_rows, basis, pair_ids, pair_posts,
                                  pair_types, n_post_, weight_grad));
    } else {
      OP_REQUIRES_OK(context, LaunchWeightBackward<T, 0>(
                                  context, activity, current_grad, post_ids,
                                  synapse_types, row_splits, edge_ids,
                                  nonempty_rows, basis, pair_ids, pair_posts,
                                  pair_types, n_post_, weight_grad));
    }
  }

 private:
  int n_post_;
  int n_edges_;
};

template <typename T>
class ExternalCsrActivityBackwardOp : public OpKernel {
 public:
  explicit ExternalCsrActivityBackwardOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("n_post", &n_post_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& current_grad = context->input(0);
    const Tensor& weights = context->input(1);
    const Tensor& post_ids = context->input(2);
    const Tensor& synapse_types = context->input(3);
    const Tensor& row_splits = context->input(4);
    const Tensor& edge_ids = context->input(5);
    const Tensor& nonempty_rows = context->input(6);
    const Tensor& basis = context->input(7);
    const Tensor& pair_ids = context->input(8);
    const Tensor& pair_posts = context->input(9);
    const Tensor& pair_types = context->input(10);
    OP_REQUIRES(context, current_grad.dims() == 2 && basis.dims() == 2,
                errors::InvalidArgument("current_grad and basis must be rank two"));
    OP_REQUIRES(context, current_grad.dim_size(0) % n_post_ == 0 &&
                                 current_grad.dim_size(1) == basis.dim_size(1),
                errors::InvalidArgument("current_grad has an incompatible shape"));
    OP_REQUIRES(context, row_splits.NumElements() >= 2,
                errors::InvalidArgument("row_splits must describe at least one row"));
    OP_REQUIRES(context, post_ids.NumElements() == weights.NumElements() &&
                                 synapse_types.NumElements() == weights.NumElements() &&
                                 pair_ids.NumElements() == weights.NumElements() &&
                                 edge_ids.NumElements() ==
                                     weights.NumElements() * kEdgeIdsPerEdge,
                errors::InvalidArgument("edge metadata size mismatch"));
    OP_REQUIRES(context, pair_posts.NumElements() == pair_types.NumElements(),
                errors::InvalidArgument("pair metadata size mismatch"));
    Tensor* activity_grad;
    const int64_t batch = current_grad.dim_size(0) / n_post_;
    const int64_t n_pre = row_splits.NumElements() - 1;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, TensorShape({batch, n_pre}), &activity_grad));
    auto device = context->eigen_device<GPUDevice>();
    cudaMemsetAsync(activity_grad->flat<T>().data(), 0,
                    activity_grad->NumElements() * sizeof(T), device.stream());
    if (basis.dim_size(1) == 4) {
      OP_REQUIRES_OK(context, LaunchActivityBackward<T, 4>(
                                  context, current_grad, weights, post_ids,
                                  synapse_types, row_splits, edge_ids,
                                  nonempty_rows, basis, pair_ids,
                                  pair_posts, pair_types, n_post_, activity_grad));
    } else {
      OP_REQUIRES_OK(context, LaunchActivityBackward<T, 0>(
                                  context, current_grad, weights, post_ids,
                                  synapse_types, row_splits, edge_ids,
                                  nonempty_rows, basis, pair_ids,
                                  pair_posts, pair_types, n_post_, activity_grad));
    }
  }

 private:
  int n_post_;
};

#ifndef V1_KERNEL_IMPLEMENTATION_ONLY
#define REGISTER_BKG_TYPE(T)                                           \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("BkgCsrForward").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      BkgCsrForwardOp<T>);

TF_CALL_half(REGISTER_BKG_TYPE);
TF_CALL_float(REGISTER_BKG_TYPE);
#undef REGISTER_BKG_TYPE

#define REGISTER_TYPE(T)                                                \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("ExternalCsrWeightBackward")                               \
          .Device(DEVICE_GPU)                                          \
          .TypeConstraint<T>("T"),                                    \
      ExternalCsrWeightBackwardOp<T>);

#define REGISTER_ACTIVITY_TYPE(T)                                       \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("ExternalCsrActivityBackward")                             \
          .Device(DEVICE_GPU)                                           \
          .TypeConstraint<T>("T"),                                    \
      ExternalCsrActivityBackwardOp<T>);

TF_CALL_half(REGISTER_TYPE);
TF_CALL_float(REGISTER_TYPE);
#undef REGISTER_TYPE
TF_CALL_half(REGISTER_ACTIVITY_TYPE);
TF_CALL_float(REGISTER_ACTIVITY_TYPE);
#undef REGISTER_ACTIVITY_TYPE
#endif

#endif

#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
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

constexpr int kThreads = V1_EXTERNAL_THREADS;

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

template <typename T, int kBasis>
__global__ void ProjectCompactExternalKernel(
    int64_t elements, int batch_size, int n_post, int n_basis,
    const uint32* pair_posts, const uint8* pair_types,
    const T* current_grad, const T* basis, float* projected) {
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
       index < elements;
       index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int batch = index % batch_size;
    const int64_t pair = index / batch_size;
    const uint32 post = pair_posts[pair];
    const uint32 type = pair_types[pair];
    projected[index] = BasisProjection<T, kBasis>::Apply(
        current_grad +
            (static_cast<int64_t>(batch) * n_post + post) * n_basis,
        basis, type, n_basis);
  }
}

template <typename T>
__global__ void WeightBackwardCompactTensorRowKernel(
    int64_t n_pre, const T* activity, const float* projected,
    const uint32* pair_ids, const uint32* edge_ids,
    const uint32* row_splits, const uint32* nonempty_rows, int64_t n_rows,
    float* weight_grad) {
  namespace wmma = nvcuda::wmma;
  constexpr int kWarps = 4;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int64_t row_id = blockIdx.x;
  if (row_id >= n_rows) return;
  __shared__ __half matrix_a[kWarps][16 * 32];
  __shared__ __half matrix_b[kWarps][32 * 16];
  __shared__ float matrix_c[kWarps][16 * 16];
  const uint32 pre = nonempty_rows[row_id];
  const float value_activity =
      AsFloat(activity[static_cast<int64_t>(lane) * n_pre + pre]);
#pragma unroll
  for (int row = 0; row < 16; ++row) {
    matrix_a[warp][row * 32 + lane] = __float2half(value_activity);
  }
  const uint32 start = row_splits[pre];
  const uint32 end = row_splits[pre + 1];
  for (uint32 base = start + warp * 16; base < end; base += kWarps * 16) {
#pragma unroll
    for (int column = 0; column < 16; ++column) {
      const uint32 csr = base + column;
      matrix_b[warp][column * 32 + lane] = __float2half(
          csr < end
              ? projected[static_cast<int64_t>(pair_ids[csr]) * 32 + lane]
              : 0.0f);
    }
    __syncwarp();
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
    wmma::fill_fragment(c, 0.0f);
    wmma::load_matrix_sync(a, matrix_a[warp], 32);
    wmma::load_matrix_sync(b, matrix_b[warp], 32);
    wmma::mma_sync(c, a, b, c);
    wmma::load_matrix_sync(a, matrix_a[warp] + 16, 32);
    wmma::load_matrix_sync(b, matrix_b[warp] + 16, 32);
    wmma::mma_sync(c, a, b, c);
    wmma::store_matrix_sync(matrix_c[warp], c, 16, wmma::mem_row_major);
    __syncwarp();
    if (lane < 16 && base + lane < end) {
      weight_grad[V1_EDGE_INDEX(base + lane)] = matrix_c[warp][lane];
    }
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

template <typename T, int kTile>
__global__ void ActivityBackwardCompactKernel(
    int64_t n_pre, const float* projected, const float* weights,
    const uint32* pair_ids, const uint32* edge_ids,
    const uint32* row_splits, const uint32* nonempty_rows, int64_t n_rows,
    T* activity_grad) {
  __shared__ float partial[kTile][256];
  const int64_t tile_id = blockIdx.x / n_rows;
  const int64_t row_id = blockIdx.x - tile_id * n_rows;
  if (row_id >= n_rows) return;
  const int first_batch = static_cast<int>(tile_id) * kTile;
  const uint32 pre = nonempty_rows[row_id];
  float local[kTile] = {};
  for (uint32 csr = row_splits[pre] + threadIdx.x;
       csr < row_splits[pre + 1]; csr += blockDim.x) {
    const float weight = weights[V1_EDGE_INDEX(csr)];
    const int64_t projection = static_cast<int64_t>(pair_ids[csr]) * 32;
#pragma unroll
    for (int offset = 0; offset < kTile; ++offset) {
      local[offset] += projected[projection + first_batch + offset] * weight;
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
      activity_grad[static_cast<int64_t>(first_batch + offset) * n_pre + pre] =
          static_cast<T>(partial[offset][0]);
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
    TF_RETURN_IF_ERROR(GpuLaunchKernel(
        ProjectCompactExternalKernel<T, kBasis>,
        static_cast<int>((projected_elements + 255) / 256), 256, 0,
        device.stream(), projected_elements, static_cast<int>(batch), n_post,
        n_basis, pair_posts.flat<uint32>().data(),
        pair_types.flat<uint8>().data(), current_grad.flat<T>().data(),
        basis.flat<T>().data(), projected_tensor.flat<float>().data()));
    TF_RETURN_IF_ERROR(GpuLaunchKernel(
        WeightBackwardCompactTensorRowKernel<T>, static_cast<int>(n_rows), 128,
        0, device.stream(), activity.dim_size(1), activity.flat<T>().data(),
        projected_tensor.flat<float>().data(), pair_ids.flat<uint32>().data(),
        edge_ids.flat<uint32>().data(), row_splits.flat<uint32>().data(),
        nonempty_rows.flat<uint32>().data(), n_rows,
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
    TF_RETURN_IF_ERROR(GpuLaunchKernel(
        ProjectCompactExternalKernel<T, kBasis>,
        static_cast<int>((projected_elements + 255) / 256), 256, 0,
        device.stream(), projected_elements, static_cast<int>(batch), n_post,
        n_basis, pair_posts.flat<uint32>().data(),
        pair_types.flat<uint8>().data(), current_grad.flat<T>().data(),
        basis.flat<T>().data(), projected_tensor.flat<float>().data()));
    TF_RETURN_IF_ERROR(GpuLaunchKernel(
        ActivityBackwardCompactKernel<T, 8>,
        static_cast<int>(n_rows * (batch / 8)), 256, 0, device.stream(),
        row_splits.NumElements() - 1, projected_tensor.flat<float>().data(),
        weights.flat<float>().data(), pair_ids.flat<uint32>().data(),
        edge_ids.flat<uint32>().data(), row_splits.flat<uint32>().data(),
        nonempty_rows.flat<uint32>().data(), n_rows,
        activity_grad->flat<T>().data()));
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
                                 edge_ids.NumElements() == n_edges_ &&
                                 pair_ids.NumElements() == n_edges_,
                errors::InvalidArgument("edge metadata size mismatch"));
    OP_REQUIRES(context, pair_posts.NumElements() == pair_types.NumElements(),
                errors::InvalidArgument("pair metadata size mismatch"));
    Tensor* weight_grad;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, TensorShape({n_edges_}), &weight_grad));
    auto device = context->eigen_device<GPUDevice>();
    cudaMemsetAsync(weight_grad->flat<float>().data(), 0,
                    weight_grad->NumElements() * sizeof(float),
                    device.stream());
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
                                 edge_ids.NumElements() == weights.NumElements(),
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

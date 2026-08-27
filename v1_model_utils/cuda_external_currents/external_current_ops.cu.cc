#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include <cuda_runtime.h>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

using namespace tensorflow;
using GPUDevice = Eigen::GpuDevice;

constexpr int kThreads = 128;

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
    const uint32 edge = edge_ids[csr];
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
    const uint32 edge = edge_ids[csr];
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
    int n_post, Tensor* weight_grad) {
  const int64_t n_rows = nonempty_rows.NumElements();
  const int64_t batch = activity.dim_size(0);
  if (batch == 0 || n_rows == 0) return OkStatus();
  const int n_basis = basis.dim_size(1);
  auto device = context->eigen_device<GPUDevice>();
#define LAUNCH_STATIC(BATCH, TILE)                                        \
  TF_RETURN_IF_ERROR(GpuLaunchKernel(                                    \
      WeightBackwardStaticBatchKernel<T, kBasis, BATCH, TILE>,           \
      static_cast<int>(n_rows * (BATCH / TILE)), kThreads, 0,            \
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
    EXTERNAL_BATCH_CASE(8, 4, LAUNCH_STATIC);
    EXTERNAL_BATCH_CASE(16, 4, LAUNCH_STATIC);
    EXTERNAL_BATCH_CASE(32, 4, LAUNCH_STATIC);
    EXTERNAL_BATCH_CASE(64, 4, LAUNCH_STATIC);
    EXTERNAL_BATCH_CASE(128, 4, LAUNCH_STATIC);
    EXTERNAL_BATCH_CASE(256, 4, LAUNCH_STATIC);
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
                                 edge_ids.NumElements() == n_edges_,
                errors::InvalidArgument("edge metadata size mismatch"));
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
                                  nonempty_rows, basis, n_post_, weight_grad));
    } else {
      OP_REQUIRES_OK(context, LaunchWeightBackward<T, 0>(
                                  context, activity, current_grad, post_ids,
                                  synapse_types, row_splits, edge_ids,
                                  nonempty_rows, basis, n_post_, weight_grad));
    }
  }

 private:
  int n_post_;
  int n_edges_;
};

#define REGISTER_TYPE(T)                                                \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("ExternalCsrWeightBackward")                               \
          .Device(DEVICE_GPU)                                          \
          .TypeConstraint<T>("T"),                                    \
      ExternalCsrWeightBackwardOp<T>);

TF_CALL_half(REGISTER_TYPE);
TF_CALL_float(REGISTER_TYPE);
#undef REGISTER_TYPE

#endif

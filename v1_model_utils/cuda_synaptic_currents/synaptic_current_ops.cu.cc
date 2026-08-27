#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include <cuda_fp16.h>
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

template <typename T, typename W, int kBasis, int kBatch>
__global__ void ForwardKernel(
    int64_t n_active, int64_t n_pre, int n_post, int n_basis,
    const T* spikes, const int64_t* active, const W* weights,
    const uint32* post_ids, const uint8* synapse_types,
    const uint32* row_splits, const uint32* edge_ids, const T* basis,
    T* currents) {
  const int64_t active_id = blockIdx.x;
  if (active_id >= n_active) return;
  const int64_t batch = active[2 * active_id];
  if (kBatch != 0 && batch >= kBatch) return;
  const int64_t pre = active[2 * active_id + 1];
  const float spike = AsFloat(spikes[batch * n_pre + pre]);
  const uint32 start = row_splits[pre];
  const uint32 end = row_splits[pre + 1];
  for (uint32 csr = start + threadIdx.x; csr < end; csr += blockDim.x) {
    const uint32 edge = edge_ids[csr];
    const uint32 post = post_ids[csr];
    const uint32 type = synapse_types[csr];
    const float weighted_spike = spike * AsFloat(weights[edge]);
    T* output = currents +
                (batch * static_cast<int64_t>(n_post) + post) * n_basis;
    if (kBasis == 4) {
#pragma unroll
      for (int receptor = 0; receptor < 4; ++receptor) {
        AtomicAddValue(output + receptor,
                       weighted_spike * AsFloat(basis[type * 4 + receptor]));
      }
    } else {
      for (int receptor = 0; receptor < n_basis; ++receptor) {
        AtomicAddValue(
            output + receptor,
            weighted_spike * AsFloat(basis[type * n_basis + receptor]));
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
      const uint32 edge = edge_ids[csr];
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
  __shared__ float reduction[kTile][kThreads];
  const int64_t tile_id = blockIdx.x / n_rows;
  const int64_t row_id = blockIdx.x - tile_id * n_rows;
  if (row_id >= n_rows) return;
  const int first_batch = static_cast<int>(tile_id) * kTile;
  const uint32 pre = nonempty_rows[row_id];
  float local_spike_grad[kTile] = {};
  const uint32 start = row_splits[pre];
  const uint32 end = row_splits[pre + 1];
  for (uint32 csr = start + threadIdx.x; csr < end; csr += blockDim.x) {
    const uint32 edge = edge_ids[csr];
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
      const int batch = first_batch + offset;
      spike_grad[batch * n_pre + pre] =
          FromFloat<T>(reduction[offset][0] * AsFloat(*dampening));
    }
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
#define LAUNCH_FORWARD(BATCH, TILE)                                      \
  TF_RETURN_IF_ERROR(GpuLaunchKernel(                                   \
      ForwardKernel<T, W, kBasis, BATCH>, static_cast<int>(n_active),   \
      kThreads, 0, device.stream(), n_active, spikes.dim_size(1),       \
      n_post, n_basis, spikes.flat<T>().data(),                         \
      active.flat<int64_t>().data(), weights.flat<W>().data(),          \
      post_ids.flat<uint32>().data(), synapse_types.flat<uint8>().data(), \
      row_splits.flat<uint32>().data(), edge_ids.flat<uint32>().data(), \
      basis.flat<T>().data(), output->flat<T>().data()))
  switch (spikes.dim_size(0)) {
    V1_BATCH_CASE(1, 1, LAUNCH_FORWARD);
    V1_BATCH_CASE(2, 2, LAUNCH_FORWARD);
    V1_BATCH_CASE(4, 4, LAUNCH_FORWARD);
    V1_BATCH_CASE(8, 4, LAUNCH_FORWARD);
    V1_BATCH_CASE(16, 4, LAUNCH_FORWARD);
    V1_BATCH_CASE(32, 4, LAUNCH_FORWARD);
    V1_BATCH_CASE(64, 4, LAUNCH_FORWARD);
    V1_BATCH_CASE(128, 4, LAUNCH_FORWARD);
    V1_BATCH_CASE(256, 4, LAUNCH_FORWARD);
    default:
      LAUNCH_FORWARD(0, 1);
  }
#undef LAUNCH_FORWARD
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
  auto device = context->eigen_device<GPUDevice>();
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
  switch (batch) {
    V1_BATCH_CASE(1, 1, LAUNCH_BACKWARD);
    V1_BATCH_CASE(2, 2, LAUNCH_BACKWARD);
    V1_BATCH_CASE(4, 4, LAUNCH_BACKWARD);
    V1_BATCH_CASE(8, 4, LAUNCH_BACKWARD);
    V1_BATCH_CASE(16, 4, LAUNCH_BACKWARD);
    V1_BATCH_CASE(32, 4, LAUNCH_BACKWARD);
    V1_BATCH_CASE(64, 4, LAUNCH_BACKWARD);
    V1_BATCH_CASE(128, 4, LAUNCH_BACKWARD);
    V1_BATCH_CASE(256, 4, LAUNCH_BACKWARD);
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
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, TensorShape({batch * n_post_, n_basis}),
                                &output));
    auto device = context->eigen_device<GPUDevice>();
    cudaMemsetAsync(output->flat<T>().data(), 0,
                    output->NumElements() * sizeof(T), device.stream());
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

#define REGISTER_PAIR(T, W)                                                \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("V1CsrForward").Device(DEVICE_GPU).TypeConstraint<T>("T")      \
          .TypeConstraint<W>("W"),                                        \
      V1CsrForwardOp<T, W>);                                               \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("V1CsrBackward").Device(DEVICE_GPU).TypeConstraint<T>("T")     \
          .TypeConstraint<W>("W"),                                        \
      V1CsrBackwardOp<T, W>);

#define REGISTER_WEIGHTS(T)                                                \
  REGISTER_PAIR(T, Eigen::half);                                          \
  REGISTER_PAIR(T, float)

TF_CALL_half(REGISTER_WEIGHTS);
TF_CALL_float(REGISTER_WEIGHTS);
#undef REGISTER_WEIGHTS
#undef REGISTER_PAIR

#endif

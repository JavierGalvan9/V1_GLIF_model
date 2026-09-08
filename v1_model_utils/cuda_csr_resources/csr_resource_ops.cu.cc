#if GOOGLE_CUDA

#define EIGEN_USE_GPU
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <type_traits>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

// MultiWorkerMirroredStrategy cannot broadcast uint8/uint32 variables in
// TensorFlow 2.15.  The network values fit int32, so compile this isolated
// kernel with a consistent signed 32-bit metadata representation.
#define uint8 int32
#define uint32 int32
#define V1_KERNEL_IMPLEMENTATION_ONLY
#include "../cuda_csr_recurrent/csr_recurrent_ops.cu.cc"
#undef V1_KERNEL_IMPLEMENTATION_ONLY
#undef uint32
#undef uint8

namespace external_resource_kernel {
#define uint8 int32
#define uint32 int32
#define AsFloat ExternalAsFloat
#define BasisProjection ExternalBasisProjection
#define V1_KERNEL_IMPLEMENTATION_ONLY
#include "../cuda_csr_external/csr_external_grad_ops.cu.cc"
#undef V1_KERNEL_IMPLEMENTATION_ONLY
#undef BasisProjection
#undef AsFloat
#undef uint32
#undef uint8
}  // namespace external_resource_kernel

#include "tensorflow/core/framework/resource_mgr.h"

class V1CsrResource : public ResourceBase {
 public:
  V1CsrResource(const Tensor& post_ids, const Tensor& synapse_types,
                const Tensor& row_splits, const Tensor& edge_ids,
                const Tensor& nonempty_rows, const Tensor& pair_ids,
                const Tensor& pair_posts, const Tensor& pair_types)
      : post_ids(post_ids),
        synapse_types(synapse_types),
        row_splits(row_splits),
        edge_ids(edge_ids),
        nonempty_rows(nonempty_rows),
        pair_ids(pair_ids),
        pair_posts(pair_posts),
        pair_types(pair_types) {}

  string DebugString() const override { return "V1CsrResource"; }

  Tensor post_ids;
  Tensor synapse_types;
  Tensor row_splits;
  Tensor edge_ids;
  Tensor nonempty_rows;
  Tensor pair_ids;
  Tensor pair_posts;
  Tensor pair_types;
};

Status ValidateMetadata(const Tensor& post_ids, const Tensor& synapse_types,
                        const Tensor& row_splits, const Tensor& edge_ids,
                        const Tensor& nonempty_rows, const Tensor& pair_ids,
                        const Tensor& pair_posts, const Tensor& pair_types) {
  if (!TensorShapeUtils::IsVector(post_ids.shape()) ||
      !TensorShapeUtils::IsVector(synapse_types.shape()) ||
      !TensorShapeUtils::IsVector(row_splits.shape()) ||
      !TensorShapeUtils::IsVector(edge_ids.shape()) ||
      !TensorShapeUtils::IsVector(nonempty_rows.shape()) ||
      !TensorShapeUtils::IsVector(pair_ids.shape()) ||
      !TensorShapeUtils::IsVector(pair_posts.shape()) ||
      !TensorShapeUtils::IsVector(pair_types.shape())) {
    return errors::InvalidArgument("CSR metadata tensors must be rank one");
  }
  if (post_ids.NumElements() != synapse_types.NumElements() ||
      post_ids.NumElements() != edge_ids.NumElements()) {
    return errors::InvalidArgument(
        "post_ids, synapse_types, and edge_ids must have equal lengths");
  }
  // The per-edge pair projection only feeds the backward kernels. A
  // connectivity declared without a backward (the LGN and BKG inputs) uploads
  // it empty, so accept that and let the backward ops reject a missing
  // projection if one is ever requested.
  if (pair_ids.NumElements() != 0 &&
      pair_ids.NumElements() != post_ids.NumElements()) {
    return errors::InvalidArgument(
        "pair_ids must be empty or match the edge count");
  }
  if (row_splits.NumElements() < 2) {
    return errors::InvalidArgument("row_splits must contain at least two values");
  }
  if (nonempty_rows.NumElements() >= row_splits.NumElements()) {
    return errors::InvalidArgument(
        "nonempty_rows cannot exceed the number of CSR rows");
  }
  if (pair_posts.NumElements() != pair_types.NumElements()) {
    return errors::InvalidArgument(
        "pair_posts and pair_types must have equal lengths");
  }
  return OkStatus();
}

Status ValidateRuntimeInputs(const V1CsrResource& resource,
                             const Tensor& values, const Tensor& weights,
                             const Tensor& basis, int n_post, int n_edges) {
  if (!TensorShapeUtils::IsMatrix(values.shape()) ||
      !TensorShapeUtils::IsMatrix(basis.shape())) {
    return errors::InvalidArgument("activity and basis must be rank two");
  }
  if (values.dim_size(1) + 1 != resource.row_splits.NumElements()) {
    return errors::InvalidArgument(
        "activity width does not match resource row_splits");
  }
  if (basis.dim_size(1) <= 0) {
    return errors::InvalidArgument("basis dimension must be positive");
  }
  if (!TensorShapeUtils::IsVector(weights.shape()) ||
      weights.NumElements() != n_edges ||
      resource.post_ids.NumElements() != n_edges) {
    return errors::InvalidArgument(
        "weights and resource metadata do not match n_edges");
  }
  if (n_post <= 0) {
    return errors::InvalidArgument("n_post must be positive");
  }
  return OkStatus();
}

string DeviceResourceName(OpKernelContext* context, const string& base) {
  const string& device_name = context->device()->name();
  const size_t separator = device_name.find_last_of(':');
  return base + "_gpu" + device_name.substr(separator + 1);
}

class InitializeV1CsrResourceOp : public OpKernel {
 public:
  explicit InitializeV1CsrResourceOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("resource_name", &resource_name_));
  }

  void Compute(OpKernelContext* context) override {
    OP_REQUIRES_OK(context,
                   ValidateMetadata(context->input(0), context->input(1),
                                    context->input(2), context->input(3),
                                    context->input(4), context->input(5),
                                    context->input(6), context->input(7)));
    V1CsrResource* resource = new V1CsrResource(
        context->input(0), context->input(1), context->input(2),
        context->input(3), context->input(4), context->input(5),
        context->input(6), context->input(7));
    Status status = context->resource_manager()->Create(
        "distributed_connectivity", resource_name_, resource);
    if (!status.ok()) {
      resource->Unref();
      OP_REQUIRES(context, errors::IsAlreadyExists(status), status);
    }
    Tensor* initialized;
    OP_REQUIRES_OK(context, context->allocate_output(0, {}, &initialized));
    initialized->scalar<bool>()() = true;
  }

 private:
  string resource_name_;
};

template <typename T, typename W>
class V1CsrForwardResourceOp : public OpKernel {
 public:
  explicit V1CsrForwardResourceOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("n_post", &n_post_));
    OP_REQUIRES_OK(context, context->GetAttr("resource_name", &resource_name_));
  }

  void Compute(OpKernelContext* context) override {
    V1CsrResource* resource = nullptr;
    OP_REQUIRES_OK(context, context->resource_manager()->Lookup(
                                "distributed_connectivity",
                                DeviceResourceName(context, resource_name_),
                                &resource));
    core::ScopedUnref resource_unref(resource);
    const Tensor& spikes = context->input(0);
    const Tensor& active = context->input(1);
    const Tensor& weights = context->input(2);
    const Tensor& basis = context->input(3);
    const Tensor& initial = context->input(4);
    const Tensor& posts = resource->post_ids;
    const Tensor& types = resource->synapse_types;
    const Tensor& rows = resource->row_splits;
    const Tensor& edges = resource->edge_ids;
    OP_REQUIRES_OK(context, ValidateRuntimeInputs(
                                *resource, spikes, weights, basis, n_post_,
                                resource->post_ids.NumElements()));
    OP_REQUIRES(context, spikes.dims() == 2,
                errors::InvalidArgument("spikes must be rank two"));
    OP_REQUIRES(context, active.dims() == 2 && active.dim_size(1) == 2,
                errors::InvalidArgument("active_indices must be [N,2]"));
    Tensor* output;
    const TensorShape shape(
        {spikes.dim_size(0) * n_post_, basis.dim_size(1)});
    const bool accumulate = initial.NumElements() > 0;
    OP_REQUIRES(
        context, !accumulate || initial.shape() == shape,
        errors::InvalidArgument("initial must be empty or match the output shape"));
    auto device = context->eigen_device<GPUDevice>();
    if (accumulate) {
      // The scatter is atomic, so it can land directly on the previous
      // source's currents. Taking that buffer over removes a full pass over a
      // [batch * n_post, n_basis] tensor per current source, which is the
      // largest elementwise traffic in the step.
      OP_REQUIRES_OK(context,
                     context->forward_input_or_allocate_output({4}, 0, shape,
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
    if (basis.dim_size(1) == 4) {
      OP_REQUIRES_OK(context, LaunchForward<T, W, 4>(
                                  context, spikes, active, weights, posts, types,
                                  rows, edges, basis, n_post_, output));
    } else {
      OP_REQUIRES_OK(context, LaunchForward<T, W, 0>(
                                  context, spikes, active, weights, posts, types,
                                  rows, edges, basis, n_post_, output));
    }
  }

 private:
  int n_post_;
  string resource_name_;
};

template <typename T, typename W>
class V1CsrBackwardResourceOp : public OpKernel {
 public:
  explicit V1CsrBackwardResourceOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("n_post", &n_post_));
    OP_REQUIRES_OK(context, context->GetAttr("n_edges", &n_edges_));
    OP_REQUIRES_OK(context, context->GetAttr("resource_name", &resource_name_));
    OP_REQUIRES_OK(
        context, context->GetAttr("pair_projected", &pair_projected_));
  }

  void Compute(OpKernelContext* context) override {
    V1CsrResource* resource = nullptr;
    OP_REQUIRES_OK(context, context->resource_manager()->Lookup(
                                "distributed_connectivity",
                                DeviceResourceName(context, resource_name_),
                                &resource));
    core::ScopedUnref resource_unref(resource);
    const Tensor& spikes = context->input(0);
    const Tensor& current_grad = context->input(1);
    const Tensor& weights = context->input(2);
    const Tensor& basis = context->input(3);
    const Tensor& dampening = context->input(4);
    const Tensor& posts = resource->post_ids;
    const Tensor& types = resource->synapse_types;
    const Tensor& rows = resource->row_splits;
    const Tensor& edges = resource->edge_ids;
    const Tensor& nonempty = resource->nonempty_rows;
    OP_REQUIRES_OK(context, ValidateRuntimeInputs(
                                *resource, spikes, weights, basis, n_post_,
                                n_edges_));
    OP_REQUIRES(context,
                TensorShapeUtils::IsMatrix(current_grad.shape()) &&
                    current_grad.dim_size(0) == spikes.dim_size(0) * n_post_ &&
                    current_grad.dim_size(1) == basis.dim_size(1),
                errors::InvalidArgument("current_grad has an incompatible shape"));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(dampening.shape()),
                errors::InvalidArgument("dampening must be scalar"));
    Tensor* spike_grad;
    Tensor* weight_grad;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, spikes.shape(), &spike_grad));
    OP_REQUIRES_OK(context, context->allocate_output(
                                1, TensorShape({n_edges_}), &weight_grad));
    auto device = context->eigen_device<GPUDevice>();
    // Rows with no edges are never visited, so the spike gradient is cleared.
    // The weight gradient is not when the pair-projected kernel runs: it writes
    // every CSR position exactly once, and clearing the whole edge vector first
    // would only cost write bandwidth - the same reasoning as the tensor
    // backend's dedicated pair-projected operator.
    cudaMemsetAsync(spike_grad->flat<T>().data(), 0,
                    spike_grad->NumElements() * sizeof(T), device.stream());
    if (!pair_projected_) {
      cudaMemsetAsync(weight_grad->flat<float>().data(), 0,
                      weight_grad->NumElements() * sizeof(float),
                      device.stream());
    }
    // Mirror the tensor backend's specialization. `pair_projection_applies` in
    // cuda_csr_recurrent/wrapper.py gates the compact pair-projected backward
    // on the same measured hot shape, and a resource-backed replica has to
    // reach that kernel too or replicated training pays a large penalty at
    // exactly the production batch size.
    OP_REQUIRES(
        context,
        !pair_projected_ ||
            (resource->pair_ids.NumElements() ==
                 resource->post_ids.NumElements() &&
             resource->pair_posts.NumElements() > 0),
        errors::InvalidArgument(
            "the pair-projected backward was requested, but this resource "
            "holds no per-edge pair projection"));
    if (pair_projected_ && basis.dim_size(1) == 4) {
      OP_REQUIRES_OK(context, LaunchPairProjectedBackward<T, W, 4>(
                                  context, spikes, current_grad, weights, posts,
                                  types, rows, edges, nonempty, basis, dampening,
                                  resource->pair_ids, resource->pair_posts,
                                  resource->pair_types, n_post_, spike_grad,
                                  weight_grad));
    } else if (basis.dim_size(1) == 4) {
      OP_REQUIRES_OK(context, LaunchBackward<T, W, 4>(
                                  context, spikes, current_grad, weights, posts,
                                  types, rows, edges, nonempty, basis, dampening,
                                  n_post_, spike_grad, weight_grad));
    } else {
      OP_REQUIRES_OK(context, LaunchBackward<T, W, 0>(
                                  context, spikes, current_grad, weights, posts,
                                  types, rows, edges, nonempty, basis, dampening,
                                  n_post_, spike_grad, weight_grad));
    }
  }

 private:
  int n_post_;
  int n_edges_;
  string resource_name_;
  bool pair_projected_ = false;
};

template <typename T>
class ExternalCsrWeightBackwardResourceOp : public OpKernel {
 public:
  explicit ExternalCsrWeightBackwardResourceOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("n_post", &n_post_));
    OP_REQUIRES_OK(context, context->GetAttr("n_edges", &n_edges_));
    OP_REQUIRES_OK(context, context->GetAttr("resource_name", &resource_name_));
  }

  void Compute(OpKernelContext* context) override {
    V1CsrResource* resource = nullptr;
    OP_REQUIRES_OK(context, context->resource_manager()->Lookup(
                                "distributed_connectivity",
                                DeviceResourceName(context, resource_name_),
                                &resource));
    core::ScopedUnref resource_unref(resource);
    const Tensor& activity = context->input(0);
    const Tensor& current_grad = context->input(1);
    const Tensor& basis = context->input(2);
    OP_REQUIRES(context,
                TensorShapeUtils::IsMatrix(activity.shape()) &&
                    TensorShapeUtils::IsMatrix(basis.shape()),
                errors::InvalidArgument("activity and basis must be rank two"));
    OP_REQUIRES(context,
                activity.dim_size(1) + 1 == resource->row_splits.NumElements(),
                errors::InvalidArgument(
                    "activity width does not match resource row_splits"));
    OP_REQUIRES(context, basis.dim_size(1) > 0,
                errors::InvalidArgument("basis dimension must be positive"));
    OP_REQUIRES(context, resource->post_ids.NumElements() == n_edges_,
                errors::InvalidArgument(
                    "resource metadata does not match n_edges"));
    OP_REQUIRES(context,
                TensorShapeUtils::IsMatrix(current_grad.shape()) &&
                    current_grad.dim_size(0) == activity.dim_size(0) * n_post_ &&
                    current_grad.dim_size(1) == basis.dim_size(1),
                errors::InvalidArgument("current_grad has an incompatible shape"));
    OP_REQUIRES(context, resource->pair_ids.NumElements() ==
                             resource->post_ids.NumElements(),
                errors::InvalidArgument(
                    "this resource was initialized without the per-edge pair "
                    "projection, so its backward cannot run"));
    Tensor* weight_grad;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, TensorShape({n_edges_}), &weight_grad));
    auto device = context->eigen_device<GPUDevice>();
    cudaMemsetAsync(weight_grad->flat<float>().data(), 0,
                    weight_grad->NumElements() * sizeof(float),
                    device.stream());
    if (basis.dim_size(1) == 4) {
      OP_REQUIRES_OK(context,
                     external_resource_kernel::LaunchWeightBackward<T, 4>(
                         context, activity, current_grad, resource->post_ids,
                         resource->synapse_types, resource->row_splits,
                         resource->edge_ids, resource->nonempty_rows, basis,
                         resource->pair_ids, resource->pair_posts,
                         resource->pair_types, n_post_, weight_grad));
    } else {
      OP_REQUIRES_OK(context,
                     external_resource_kernel::LaunchWeightBackward<T, 0>(
                         context, activity, current_grad, resource->post_ids,
                         resource->synapse_types, resource->row_splits,
                         resource->edge_ids, resource->nonempty_rows, basis,
                         resource->pair_ids, resource->pair_posts,
                         resource->pair_types, n_post_, weight_grad));
    }
  }

 private:
  int n_post_;
  int n_edges_;
  string resource_name_;
};

template <typename T>
class ExternalCsrActivityBackwardResourceOp : public OpKernel {
 public:
  explicit ExternalCsrActivityBackwardResourceOp(
      OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("n_post", &n_post_));
    OP_REQUIRES_OK(context, context->GetAttr("resource_name", &resource_name_));
  }

  void Compute(OpKernelContext* context) override {
    V1CsrResource* resource = nullptr;
    OP_REQUIRES_OK(context, context->resource_manager()->Lookup(
                                "distributed_connectivity",
                                DeviceResourceName(context, resource_name_),
                                &resource));
    core::ScopedUnref resource_unref(resource);
    const Tensor& current_grad = context->input(0);
    const Tensor& weights = context->input(1);
    const Tensor& basis = context->input(2);
    OP_REQUIRES(context, current_grad.dims() == 2 && basis.dims() == 2,
                errors::InvalidArgument(
                    "current_grad and basis must be rank two"));
    OP_REQUIRES(context, current_grad.dim_size(0) % n_post_ == 0 &&
                                 current_grad.dim_size(1) == basis.dim_size(1),
                errors::InvalidArgument(
                    "current_grad has an incompatible shape"));
    OP_REQUIRES(context,
                weights.NumElements() == resource->post_ids.NumElements(),
                errors::InvalidArgument(
                    "weights and resource metadata size mismatch"));
    OP_REQUIRES(context, resource->pair_ids.NumElements() ==
                             resource->post_ids.NumElements(),
                errors::InvalidArgument(
                    "this resource was initialized without the per-edge pair "
                    "projection, so its backward cannot run"));
    const int64_t batch = current_grad.dim_size(0) / n_post_;
    const int64_t n_pre = resource->row_splits.NumElements() - 1;
    Tensor* activity_grad;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, TensorShape({batch, n_pre}), &activity_grad));
    auto device = context->eigen_device<GPUDevice>();
    cudaMemsetAsync(activity_grad->flat<T>().data(), 0,
                    activity_grad->NumElements() * sizeof(T), device.stream());
    if (basis.dim_size(1) == 4) {
      OP_REQUIRES_OK(
          context, external_resource_kernel::LaunchActivityBackward<T, 4>(
                       context, current_grad, weights, resource->post_ids,
                       resource->synapse_types, resource->row_splits,
                       resource->edge_ids, resource->nonempty_rows, basis,
                       resource->pair_ids, resource->pair_posts,
                       resource->pair_types, n_post_, activity_grad));
    } else {
      OP_REQUIRES_OK(
          context, external_resource_kernel::LaunchActivityBackward<T, 0>(
                       context, current_grad, weights, resource->post_ids,
                       resource->synapse_types, resource->row_splits,
                       resource->edge_ids, resource->nonempty_rows, basis,
                       resource->pair_ids, resource->pair_posts,
                       resource->pair_types, n_post_, activity_grad));
    }
  }

 private:
  int n_post_;
  string resource_name_;
};

#define REGISTER_RESOURCE_TYPE(T)                                         \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("V1CsrForwardResource")                                       \
          .Device(DEVICE_GPU)                                              \
          .TypeConstraint<T>("T"),                                       \
      V1CsrForwardResourceOp<T, float>);                                   \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("V1CsrBackwardResource")                                      \
          .Device(DEVICE_GPU)                                              \
          .TypeConstraint<T>("T"),                                       \
      V1CsrBackwardResourceOp<T, float>);
      
#define REGISTER_EXTERNAL_RESOURCE_TYPE(T)                               \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("ExternalCsrWeightBackwardResource")                         \
          .Device(DEVICE_GPU)                                            \
          .TypeConstraint<T>("T"),                                     \
      ExternalCsrWeightBackwardResourceOp<T>);

#define REGISTER_EXTERNAL_ACTIVITY_RESOURCE_TYPE(T)                      \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("ExternalCsrActivityBackwardResource")                      \
          .Device(DEVICE_GPU)                                            \
          .TypeConstraint<T>("T"),                                     \
      ExternalCsrActivityBackwardResourceOp<T>);

TF_CALL_half(REGISTER_RESOURCE_TYPE);
TF_CALL_float(REGISTER_RESOURCE_TYPE);
#undef REGISTER_RESOURCE_TYPE
TF_CALL_half(REGISTER_EXTERNAL_RESOURCE_TYPE);
TF_CALL_float(REGISTER_EXTERNAL_RESOURCE_TYPE);
#undef REGISTER_EXTERNAL_RESOURCE_TYPE
TF_CALL_half(REGISTER_EXTERNAL_ACTIVITY_RESOURCE_TYPE);
TF_CALL_float(REGISTER_EXTERNAL_ACTIVITY_RESOURCE_TYPE);
#undef REGISTER_EXTERNAL_ACTIVITY_RESOURCE_TYPE

REGISTER_KERNEL_BUILDER(Name("InitializeV1CsrResource")
                            .Device(DEVICE_GPU)
                            .HostMemory("initialized"),
                        InitializeV1CsrResourceOp);

#endif

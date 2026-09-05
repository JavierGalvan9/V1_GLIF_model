#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include <cuda_runtime.h>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

using namespace tensorflow;
using GPUDevice = Eigen::GpuDevice;

template <typename T>
__device__ __forceinline__ float ToFloat(T value) {
  return static_cast<float>(value);
}

template <typename T>
__global__ void ForwardKernel(int64 count, int64 neurons, int64 width,
                              const T* voltage, const bool* refractory,
                              const T* history, T* spikes, T* new_history) {
  GPU_1D_KERNEL_LOOP(index, count) {
    const int64 batch = index / width;
    const int64 column = index - batch * width;
    if (column < neurons) {
      const int64 neuron_index = batch * neurons + column;
      const T spike = static_cast<T>(
          !refractory[neuron_index] && ToFloat(voltage[neuron_index]) > 0.0f);
      spikes[neuron_index] = spike;
      new_history[index] = spike;
    } else {
      new_history[index] = history[batch * width + column - neurons];
    }
  }
}

template <typename T>
__global__ void BackwardKernel(int64 history_count, int64 neurons, int64 width,
                               const T* voltage,
                               const bool* refractory, const T* spike_grad,
                               const T* history_grad, const T* sigma,
                               const T* amplitude, int surrogate,
                               T* voltage_grad, T* old_history_grad) {
  GPU_1D_KERNEL_LOOP(index, history_count) {
    const int64 batch = index / width;
    const int64 column = index - batch * width;
    old_history_grad[index] = column + neurons < width
                                  ? history_grad[index + neurons]
                                  : static_cast<T>(0.0f);
    if (column < neurons) {
      const int64 neuron_index = batch * neurons + column;
      const float v = ToFloat(voltage[neuron_index]);
      const float scale = ToFloat(sigma[0]);
      float shape;
      if (surrogate == 1) {
        shape = expf(-(v * v) / (scale * scale));
      } else if (surrogate == 2) {
        shape = expf(-scale * fabsf(v));
      } else {
        shape = fmaxf(1.0f - fabsf(v), 0.0f);
      }
      // Preserve TensorFlow's compute-dtype rounding order: cast the shape
      // before multiplying it by the surrogate amplitude.
      const T shape_value = static_cast<T>(shape);
      const T derivative = static_cast<T>(
          ToFloat(amplitude[0]) * ToFloat(shape_value));
      const T upstream = static_cast<T>(
          ToFloat(spike_grad[neuron_index]) + ToFloat(history_grad[index]));
      voltage_grad[neuron_index] = static_cast<T>(
          refractory[neuron_index]
              ? 0.0f
              : ToFloat(upstream) * ToFloat(derivative));
    }
  }
}

template <typename T>
class ForwardOp : public OpKernel {
 public:
  explicit ForwardOp(OpKernelConstruction* context) : OpKernel(context) {}
  void Compute(OpKernelContext* context) override {
    const Tensor& voltage = context->input(0);
    const Tensor& refractory = context->input(1);
    const Tensor& history = context->input(2);
    OP_REQUIRES(context, voltage.shape() == refractory.shape(),
                errors::InvalidArgument("voltage and refractory shapes differ"));
    OP_REQUIRES(context, voltage.dim_size(0) == history.dim_size(0),
                errors::InvalidArgument("batch dimensions differ"));
    const int64 neurons = voltage.dim_size(1);
    const int64 width = history.dim_size(1);
    OP_REQUIRES(context, neurons > 0 && width % neurons == 0,
                errors::InvalidArgument("history width must be a positive multiple of neurons"));
    Tensor* spikes;
    Tensor* new_history;
    OP_REQUIRES_OK(context, context->allocate_output(0, voltage.shape(), &spikes));
    OP_REQUIRES_OK(context, context->allocate_output(1, history.shape(), &new_history));
    const int64 count = history.NumElements();
    auto config = GetGpuLaunchConfig(count, context->eigen_device<GPUDevice>());
    OP_REQUIRES_OK(context, GpuLaunchKernel(
        ForwardKernel<T>, config.block_count, config.thread_per_block, 0,
        context->eigen_device<GPUDevice>().stream(), count, neurons, width,
        voltage.flat<T>().data(), refractory.flat<bool>().data(), history.flat<T>().data(),
        spikes->flat<T>().data(), new_history->flat<T>().data()));
  }
};

template <typename T>
class BackwardOp : public OpKernel {
 public:
  explicit BackwardOp(OpKernelConstruction* context) : OpKernel(context) {
    string surrogate;
    OP_REQUIRES_OK(context, context->GetAttr("surrogate", &surrogate));
    surrogate_ = surrogate == "gaussian" ? 1 : surrogate == "slayer" ? 2 : 0;
  }
  void Compute(OpKernelContext* context) override {
    const Tensor& voltage = context->input(0);
    const Tensor& refractory = context->input(1);
    const Tensor& spike_grad = context->input(2);
    const Tensor& history_grad = context->input(3);
    const Tensor& sigma = context->input(4);
    const Tensor& amplitude = context->input(5);
    OP_REQUIRES(context,
                voltage.shape() == refractory.shape() && voltage.shape() == spike_grad.shape(),
                errors::InvalidArgument("neuron tensor shapes differ"));
    OP_REQUIRES(context, voltage.dim_size(0) == history_grad.dim_size(0),
                errors::InvalidArgument("batch dimensions differ"));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(sigma.shape()),
                errors::InvalidArgument("sigma must be scalar"));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(amplitude.shape()),
                errors::InvalidArgument("amplitude must be scalar"));
    const int64 neurons = voltage.dim_size(1);
    const int64 width = history_grad.dim_size(1);
    OP_REQUIRES(context, neurons > 0 && width % neurons == 0,
                errors::InvalidArgument("history width must be a positive multiple of neurons"));
    Tensor* voltage_grad;
    Tensor* old_history_grad;
    OP_REQUIRES_OK(context, context->allocate_output(0, voltage.shape(), &voltage_grad));
    OP_REQUIRES_OK(context, context->allocate_output(1, history_grad.shape(), &old_history_grad));
    const int64 history_count = history_grad.NumElements();
    auto config = GetGpuLaunchConfig(history_count, context->eigen_device<GPUDevice>());
    OP_REQUIRES_OK(context, GpuLaunchKernel(
        BackwardKernel<T>, config.block_count, config.thread_per_block, 0,
        context->eigen_device<GPUDevice>().stream(), history_count,
        neurons, width, voltage.flat<T>().data(),
        refractory.flat<bool>().data(), spike_grad.flat<T>().data(),
        history_grad.flat<T>().data(), sigma.flat<T>().data(),
        amplitude.flat<T>().data(), surrogate_,
        voltage_grad->flat<T>().data(), old_history_grad->flat<T>().data()));
  }

 private:
  int surrogate_;
};

#define REGISTER(T)                                                        \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("V1FusedSpikeShift").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      ForwardOp<T>);                                                       \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("V1FusedSpikeShiftBackward")                                   \
          .Device(DEVICE_GPU)                                              \
          .TypeConstraint<T>("T"),                                        \
      BackwardOp<T>);

REGISTER(Eigen::half)
REGISTER(float)
#undef REGISTER

void RegisterFusedSpikeShiftGpuKernels() {}
#endif

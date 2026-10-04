#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/random_distributions.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

using namespace tensorflow;
using GPUDevice = Eigen::GpuDevice;
using random::PhiloxRandom;

namespace {

constexpr int kThreads = 128;
constexpr uint32_t kReplicaStride = 1000003u;

// StatelessRandomGetKeyCounter (GenerateKey in stateless_random_ops.cc): each
// int32 seed word is widened to uint64 with sign extension, both are split
// into the counter, and one Philox round of that counter under a fixed key
// gives the key and the high counter words.
__device__ PhiloxRandom SeededGenerator(int32_t seed0, int32_t seed1) {
  const uint64_t wide0 = static_cast<uint64_t>(static_cast<int64_t>(seed0));
  const uint64_t wide1 = static_cast<uint64_t>(static_cast<int64_t>(seed1));
  PhiloxRandom::ResultType counter;
  counter[0] = static_cast<uint32_t>(wide0);
  counter[1] = static_cast<uint32_t>(wide0 >> 32);
  counter[2] = static_cast<uint32_t>(wide1);
  counter[3] = static_cast<uint32_t>(wide1 >> 32);
  PhiloxRandom::Key key;
  key[0] = 0x3ec8f720;
  key[1] = 0x02461e29;
  const PhiloxRandom::ResultType mix = PhiloxRandom(counter, key)();
  key[0] = mix[0];
  key[1] = mix[1];
  counter[0] = counter[1] = 0;
  counter[2] = mix[2];
  counter[3] = mix[3];
  return PhiloxRandom(counter, key);
}

// Counts are small integers: exact in half and float.
template <typename T> __device__ T FromCount(int count) {
  return static_cast<T>(static_cast<float>(count));
}
template <> __device__ int32_t FromCount<int32_t>(int count) { return count; }

// One thread per Philox block of two float64 uniforms (UniformDistribution<
// PhiloxRandom, double>): block g is Philox(counter + g), as in
// FillPhiloxRandomKernel, whatever its launch geometry.
template <typename T>
__global__ void __launch_bounds__(kThreads)
CountsKernel(const int64_t* __restrict__ noise_seed, int32_t replica_id, int32_t step,
             const double* __restrict__ cdf, int levels, int64_t size,
             T* __restrict__ counts) {
  const int64_t block = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t first = 2 * block;
  if (first >= size) return;
  // int32(noise_seed) + replica_id * 1000003, in wrapping int32 arithmetic.
  const uint32_t base = static_cast<uint32_t>(static_cast<uint64_t>(*noise_seed));
  const int32_t seed0 = static_cast<int32_t>(
      base + static_cast<uint32_t>(replica_id) * kReplicaStride);
  PhiloxRandom generator = SeededGenerator(seed0, step);
  generator.Skip(static_cast<uint64_t>(block));
  const PhiloxRandom::ResultType words = generator();
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    if (first + j >= size) break;
    const double uniform = random::Uint64ToDouble(words[2 * j], words[2 * j + 1]);
    int count = 0;  // searchsorted(cdf, uniform, side='right') on a sorted cdf
    for (int level = 0; level < levels; ++level) count += cdf[level] <= uniform;
    counts[first + j] = FromCount<T>(count);
  }
}

template <typename T>
class CountsOp : public OpKernel {
 public:
  explicit CountsOp(OpKernelConstruction* c) : OpKernel(c) {}
  void Compute(OpKernelContext* c) override {
    const Tensor& seed = c->input(0);
    const Tensor& step = c->input(2);
    const Tensor& shape = c->input(3);
    const Tensor& cdf = c->input(4);
    OP_REQUIRES(c, seed.NumElements() == 1 && c->input(1).NumElements() == 1,
                errors::InvalidArgument("noise_seed and replica_id must be scalars"));
    OP_REQUIRES(c, step.NumElements() >= 1,
                errors::InvalidArgument("step must hold at least one element"));
    OP_REQUIRES(c, shape.dims() == 1 && shape.NumElements() == 2,
                errors::InvalidArgument("shape must be [batch, n_bkg]"));
    OP_REQUIRES(c, cdf.dims() == 1 && cdf.NumElements() <= std::numeric_limits<int>::max(),
                errors::InvalidArgument("cdf must be a vector"));
    const auto dims = shape.flat<int32_t>();
    OP_REQUIRES(c, dims(0) >= 0 && dims(1) >= 0,
                errors::InvalidArgument("shape must be non-negative"));
    Tensor* counts;
    OP_REQUIRES_OK(c, c->allocate_output(0, TensorShape({dims(0), dims(1)}), &counts));
    const int64_t size = counts->NumElements();
    if (size == 0) return;
    const int64_t blocks = (size + 1) / 2;
    OP_REQUIRES_OK(c, GpuLaunchKernel(
        CountsKernel<T>, static_cast<int>((blocks + kThreads - 1) / kThreads), kThreads, 0,
        c->eigen_device<GPUDevice>().stream(), seed.flat<int64_t>().data(),
        c->input(1).flat<int32_t>()(0), step.flat<int32_t>()(0),
        cdf.flat<double>().data(), static_cast<int>(cdf.NumElements()), size,
        counts->flat<T>().data()));
  }
};

}  // namespace

#define REG(T)                                                          \
  REGISTER_KERNEL_BUILDER(Name("BkgPoissonCounts")                      \
                              .Device(DEVICE_GPU)                       \
                              .HostMemory("replica_id")                 \
                              .HostMemory("step")                       \
                              .HostMemory("shape")                      \
                              .TypeConstraint<T>("T"),                  \
                          CountsOp<T>);
REG(Eigen::half); REG(float); REG(int32_t);
#undef REG
#endif

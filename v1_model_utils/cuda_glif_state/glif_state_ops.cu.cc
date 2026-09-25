#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

using namespace tensorflow;
using GPUDevice = Eigen::GpuDevice;
template <typename T> __device__ float F(T x) { return static_cast<float>(x); }
template <typename T> __device__ T V(float x) { return static_cast<T>(x); }

constexpr int kBasis = 4;

// The synaptic state is read and written in T; everything else is float32
// (see glif_state_ops.cc). Each syn_coeffs entry is one (neuron, basis) pair,
// [x, y, z, w] = [syn_decay, psc_initial, psc_factor, psc_rise_factor].
template <typename T, typename R, int Basis>
__global__ void ForwardKernel(int64_t count, int neurons,
    const T* z, const float* v, const R* r, const float* asc, const T* rise,
    const T* psc, const T* inputs, const float4* syn_coeffs,
    const float* asc_decay, const float* asc_amps, const float* decay,
    const float* asc_factor, const float* reset_coeff,
    const float* asc_spike_factor, const R* t_ref, const float* dt,
    const float* v_reset, bool hard_reset,
    float* new_v, R* new_r, float* new_asc, T* new_rise, T* new_psc) {
  const float step = *dt; const float rest = *v_reset;
  GPU_1D_KERNEL_LOOP(i, count) {
    const int neuron = i % neurons;
    const int parameter_base = neuron * Basis;
    const int64_t state_base = i * Basis;
    const float reset = F(z[i]);
    // Each current source is convolved with the membrane kernel through its own
    // coefficient; the coefficients encode the integration scheme.
    float drive = asc_factor[2*neuron]*asc[2*i] +
                  asc_factor[2*neuron+1]*asc[2*i+1];
    #pragma unroll
    for (int k = 0; k < Basis; ++k) {
      const int64_t j = state_base + k;
      const float4 c = syn_coeffs[parameter_base + k];
      const float old_rise = F(rise[j]); const float old_psc = F(psc[j]);
      drive += c.z*old_psc + c.w*old_rise;
      new_rise[j] = V<T>(old_rise*c.x + F(inputs[j])*c.y);
      new_psc[j] = V<T>(old_psc*c.x + step*c.x*old_rise);
    }
    const int refractory = max(static_cast<int>(r[i]) +
        static_cast<int>(reset)*static_cast<int>(t_ref[neuron])-1, 0);
    // The reset and the ASC this spike injects both act within this step.
    float voltage = decay[neuron]*v[i] + drive +
        (reset_coeff[neuron] + asc_spike_factor[neuron])*reset;
    // A hard reset puts the membrane at v_reset at the spike time instead.
    if (hard_reset) voltage += reset*(decay[neuron]*(rest - v[i]) -
                                      reset_coeff[neuron]);
    if (hard_reset && refractory > 0) voltage = rest;
    new_v[i] = voltage; new_r[i] = static_cast<R>(refractory);
    new_asc[2*i] = asc_decay[2*neuron]*asc[2*i] + reset*asc_amps[2*neuron];
    new_asc[2*i+1] = asc_decay[2*neuron+1]*asc[2*i+1] +
                     reset*asc_amps[2*neuron+1];
  }
}

template <typename T, typename R, int Basis>
__global__ void BackwardKernel(int64_t count, int neurons,
    const T* z, const R* r, const float4* syn_coeffs, const float* asc_decay,
    const float* asc_amps, const float* decay, const float* asc_factor,
    const float* reset_coeff, const float* asc_spike_factor, const R* t_ref,
    const float* dt, const float* gv, const float* ga, const T* grise,
    const T* gpsc, bool hard_reset, bool detach_reset, bool detach_asc_reset,
    T* z_grad, float* v_grad, float* asc_grad, T* rise_grad, T* psc_grad,
    T* input_grad) {
  const float step = *dt;
  GPU_1D_KERNEL_LOOP(i, count) {
    const int neuron = i % neurons;
    const int refractory = max(static_cast<int>(r[i]) +
        static_cast<int>(F(z[i]))*static_cast<int>(t_ref[neuron])-1, 0);
    const float active_gv = hard_reset && refractory > 0 ? 0.0f : gv[i];
    // detach_reset gates the membrane reset; detach_asc_reset gates every path
    // through the ASC this spike injects - both its own state and the drive it
    // contributes to the membrane within this step.
    const float asc_reset_g = detach_asc_reset ? 0.0f :
        active_gv*asc_spike_factor[neuron] +
        ga[2*i]*asc_amps[2*neuron] + ga[2*i+1]*asc_amps[2*neuron+1];
    z_grad[i] = V<T>((detach_reset ? 0.0f : active_gv*reset_coeff[neuron])
                     + asc_reset_g);
    // The hard reset replaces the pre-spike membrane, so it carries no gradient.
    v_grad[i] = active_gv*decay[neuron]*(hard_reset ? 1.0f - F(z[i]) : 1.0f);
    asc_grad[2*i] = active_gv*asc_factor[2*neuron] + ga[2*i]*asc_decay[2*neuron];
    asc_grad[2*i+1] = active_gv*asc_factor[2*neuron+1] +
                      ga[2*i+1]*asc_decay[2*neuron+1];
    const int parameter_base = neuron*Basis; const int64_t state_base = i*Basis;
    #pragma unroll
    for (int k = 0; k < Basis; ++k) {
      const int64_t j = state_base+k;
      const float4 c = syn_coeffs[parameter_base + k];
      rise_grad[j] = V<T>(F(grise[j])*c.x + F(gpsc[j])*step*c.x +
                          active_gv*c.w);
      psc_grad[j] = V<T>(F(gpsc[j])*c.x + active_gv*c.z);
      input_grad[j] = V<T>(F(grise[j])*c.y);
    }
  }
}

// The float4 loads need a whole, 16-byte aligned coefficient block.
static absl::Status ValidateSynapticCoefficients(const Tensor& coeffs, int64_t neurons,
                                           int64_t basis) {
  if (basis != kBasis)
    return errors::InvalidArgument("the fused GLIF kernel requires four synaptic bases");
  if (coeffs.NumElements() != neurons * basis * 4)
    return errors::InvalidArgument("syn_coeffs must hold four constants per neuron and basis");
  if (reinterpret_cast<uintptr_t>(coeffs.flat<float>().data()) % alignof(float4) != 0)
    return errors::InvalidArgument("syn_coeffs must be 16-byte aligned");
  return absl::OkStatus();
}

static const float4* SynapticCoefficients(const Tensor& coeffs) {
  return reinterpret_cast<const float4*>(coeffs.flat<float>().data());
}

template <typename T, typename R> class ForwardOp : public OpKernel {
 public:
  explicit ForwardOp(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("hard_reset", &hard_));
  }
  void Compute(OpKernelContext* c) override {
    const Tensor& v = c->input(1); const Tensor& psc = c->input(5);
    const Tensor& coeffs = c->input(7);
    const int neurons = v.dim_size(1);
    OP_REQUIRES_OK(c, ValidateSynapticCoefficients(coeffs, neurons,
                                                   psc.dim_size(1) / neurons));
    Tensor *ov, *orr, *oa, *orise, *opsc;
    OP_REQUIRES_OK(c, c->allocate_output(0, v.shape(), &ov));
    OP_REQUIRES_OK(c, c->allocate_output(1, c->input(2).shape(), &orr));
    OP_REQUIRES_OK(c, c->allocate_output(2, c->input(3).shape(), &oa));
    OP_REQUIRES_OK(c, c->allocate_output(3, c->input(4).shape(), &orise));
    OP_REQUIRES_OK(c, c->allocate_output(4, psc.shape(), &opsc));
    auto& d = c->eigen_device<GPUDevice>();
    auto cfg = GetGpuLaunchConfig(v.NumElements(), d);
    OP_REQUIRES_OK(c, GpuLaunchKernel(ForwardKernel<T, R, kBasis>,
        cfg.block_count, cfg.thread_per_block, 0, d.stream(),
        v.NumElements(), neurons,
        c->input(0).flat<T>().data(), v.flat<float>().data(),
        c->input(2).flat<R>().data(), c->input(3).flat<float>().data(),
        c->input(4).flat<T>().data(), psc.flat<T>().data(),
        c->input(6).flat<T>().data(), SynapticCoefficients(coeffs),
        c->input(8).flat<float>().data(), c->input(9).flat<float>().data(),
        c->input(10).flat<float>().data(), c->input(11).flat<float>().data(),
        c->input(12).flat<float>().data(), c->input(13).flat<float>().data(),
        c->input(14).flat<R>().data(), c->input(15).flat<float>().data(),
        c->input(16).flat<float>().data(), hard_,
        ov->flat<float>().data(), orr->flat<R>().data(),
        oa->flat<float>().data(), orise->flat<T>().data(),
        opsc->flat<T>().data()));
  }
 private:
  bool hard_;
};

template <typename T, typename R> class BackwardOp : public OpKernel {
 public:
  explicit BackwardOp(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("hard_reset", &hard_));
    OP_REQUIRES_OK(c, c->GetAttr("detach_reset", &detach_));
    OP_REQUIRES_OK(c, c->GetAttr("detach_asc_reset", &detach_asc_));
  }
  void Compute(OpKernelContext* c) override {
    const Tensor& gv = c->input(11); const Tensor& grise = c->input(13);
    const Tensor& coeffs = c->input(2);
    const int neurons = gv.dim_size(1);
    OP_REQUIRES_OK(c, ValidateSynapticCoefficients(coeffs, neurons,
                                                   grise.dim_size(1) / neurons));
    Tensor *zg, *vg, *ag, *rg, *pg, *ig;
    OP_REQUIRES_OK(c, c->allocate_output(0, c->input(0).shape(), &zg));
    OP_REQUIRES_OK(c, c->allocate_output(1, gv.shape(), &vg));
    OP_REQUIRES_OK(c, c->allocate_output(2, c->input(12).shape(), &ag));
    OP_REQUIRES_OK(c, c->allocate_output(3, grise.shape(), &rg));
    OP_REQUIRES_OK(c, c->allocate_output(4, grise.shape(), &pg));
    OP_REQUIRES_OK(c, c->allocate_output(5, grise.shape(), &ig));
    auto& d = c->eigen_device<GPUDevice>();
    auto cfg = GetGpuLaunchConfig(gv.NumElements(), d);
    OP_REQUIRES_OK(c, GpuLaunchKernel(BackwardKernel<T, R, kBasis>,
        cfg.block_count, cfg.thread_per_block, 0, d.stream(),
        gv.NumElements(), neurons,
        c->input(0).flat<T>().data(), c->input(1).flat<R>().data(),
        SynapticCoefficients(coeffs), c->input(3).flat<float>().data(),
        c->input(4).flat<float>().data(), c->input(5).flat<float>().data(),
        c->input(6).flat<float>().data(), c->input(7).flat<float>().data(),
        c->input(8).flat<float>().data(), c->input(9).flat<R>().data(),
        c->input(10).flat<float>().data(), gv.flat<float>().data(),
        c->input(12).flat<float>().data(), grise.flat<T>().data(),
        c->input(14).flat<T>().data(), hard_, detach_, detach_asc_,
        zg->flat<T>().data(), vg->flat<float>().data(),
        ag->flat<float>().data(), rg->flat<T>().data(),
        pg->flat<T>().data(), ig->flat<T>().data()));
  }
 private:
  bool hard_; bool detach_; bool detach_asc_;
};

#define REG(T,R) REGISTER_KERNEL_BUILDER(Name("FusedGlifSingleForward").Device(DEVICE_GPU).TypeConstraint<T>("T").TypeConstraint<R>("R"),ForwardOp<T,R>); REGISTER_KERNEL_BUILDER(Name("FusedGlifSingleBackward").Device(DEVICE_GPU).TypeConstraint<T>("T").TypeConstraint<R>("R"),BackwardOp<T,R>);
REG(float,int8); REG(float,int16); REG(Eigen::half,int8); REG(Eigen::half,int16);
#undef REG
#endif

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

// The per-neuron float32 constants both kernels read. Each syn entry is one
// (neuron, basis) pair, [x, y, z, w] = [syn_decay, psc_initial, psc_factor,
// psc_rise_factor].
struct Constants {
  const float4* syn; const float* asc_decay; const float* asc_amps;
  const float* decay; const float* asc_factor; const float* reset_coeff;
  const float* asc_spike_factor;
};

// One thread per (sample, neuron). The synaptic state is read and written in
// T; everything else is float32 (see glif_state_ops.cc). z_buf holds `slots`
// delay slots of `neurons` spikes per sample, newest first.
template <typename T, typename R, int Basis>
__global__ void ForwardKernel(int64_t count, int neurons, int slots,
    const T* z_buf, const float* v, const R* r, const float* asc,
    const T* rise, const T* psc, const T* inputs, Constants k, const R* t_ref,
    const float* dt, const float* v_reset, const float* v_th, bool hard_reset,
    bool emit_voltage, T* spikes, T* new_z_buf, float* new_v, R* new_r,
    float* new_asc, T* new_rise, T* new_psc, bool* refractory_out,
    T* voltage_out) {
  const float step = *dt; const float rest = *v_reset;
  const float threshold = *v_th;
  GPU_1D_KERNEL_LOOP(i, count) {
    const int64_t sample = i / neurons;
    const int neuron = i - sample * neurons;
    const int64_t history = sample * slots * neurons + neuron;
    const int parameter_base = neuron * Basis;
    const int64_t state_base = i * Basis;
    const float reset = F(z_buf[history]);
    // Each current source is convolved with the membrane kernel through its own
    // coefficient; the coefficients encode the integration scheme.
    float drive = k.asc_factor[2*neuron]*asc[2*i] +
                  k.asc_factor[2*neuron+1]*asc[2*i+1];
    #pragma unroll
    for (int b = 0; b < Basis; ++b) {
      const int64_t j = state_base + b;
      const float4 c = k.syn[parameter_base + b];
      const float old_rise = F(rise[j]); const float old_psc = F(psc[j]);
      drive += c.z*old_psc + c.w*old_rise;
      new_rise[j] = V<T>(old_rise*c.x + F(inputs[j])*c.y);
      new_psc[j] = V<T>(old_psc*c.x + step*c.x*old_rise);
    }
    const int refractory = max(static_cast<int>(r[i]) +
        static_cast<int>(reset)*static_cast<int>(t_ref[neuron])-1, 0);
    // The reset and the ASC this spike injects both act within this step.
    float voltage = k.decay[neuron]*v[i] + drive +
        (k.reset_coeff[neuron] + k.asc_spike_factor[neuron])*reset;
    // A hard reset puts the membrane at v_reset at the spike time instead.
    if (hard_reset) voltage += reset*(k.decay[neuron]*(rest - v[i]) -
                                      k.reset_coeff[neuron]);
    if (hard_reset && refractory > 0) voltage = rest;
    const R counter = static_cast<R>(refractory);
    new_v[i] = voltage; new_r[i] = counter;
    new_asc[2*i] = k.asc_decay[2*neuron]*asc[2*i] + reset*k.asc_amps[2*neuron];
    new_asc[2*i+1] = k.asc_decay[2*neuron+1]*asc[2*i+1] +
                     reset*k.asc_amps[2*neuron+1];
    // The threshold reads the membrane still in registers.
    const bool blocked = counter > 0;
    const T spike = static_cast<T>(!blocked && voltage - threshold > 0.0f);
    spikes[i] = spike; refractory_out[i] = blocked;
    new_z_buf[history] = spike;
    for (int slot = 1; slot < slots; ++slot)
      new_z_buf[history + slot*neurons] = z_buf[history + (slot-1)*neurons];
    if (emit_voltage) voltage_out[i] = V<T>(voltage);
  }
}

template <typename T, int Basis>
__global__ void BackwardKernel(int64_t count, int neurons, int slots,
    const T* z_buf, const float* new_v, const bool* refractory, Constants k,
    const float* dt, const float* v_th, const float* sigma,
    const float* amplitude, int surrogate, const T* gspikes, const T* gz_buf,
    const float* gv, const float* ga, const T* grise, const T* gpsc,
    const T* gvoltage, bool hard_reset, bool detach_reset,
    bool detach_asc_reset, bool emit_voltage, T* z_buf_grad, float* v_grad,
    float* asc_grad, T* rise_grad, T* psc_grad, T* input_grad) {
  const float step = *dt; const float threshold = *v_th;
  const float scale = *sigma; const float gain = *amplitude;
  GPU_1D_KERNEL_LOOP(i, count) {
    const int64_t sample = i / neurons;
    const int neuron = i - sample * neurons;
    const int64_t history = sample * slots * neurons + neuron;
    const bool blocked = refractory[i];
    // The surrogate of the threshold. Both consumers of a spike read it in T,
    // so the upstream gradient is summed in T before it meets the float32
    // surrogate of the float32 membrane.
    const float vs = new_v[i] - threshold;
    float shape;
    if (surrogate == 1) {
      shape = expf(-(vs * vs) / (scale * scale));
    } else if (surrogate == 2) {
      shape = expf(-scale * fabsf(vs));
    } else {
      shape = fmaxf(1.0f - fabsf(vs), 0.0f);
    }
    const T upstream = static_cast<T>(F(gspikes[i]) + F(gz_buf[history]));
    float membrane_g = gv[i] + (blocked ? 0.0f : F(upstream) * (shape * gain));
    if (emit_voltage) membrane_g += F(gvoltage[i]);
    const float active_gv = hard_reset && blocked ? 0.0f : membrane_g;
    // detach_reset gates the membrane reset; detach_asc_reset gates every path
    // through the ASC this spike injects - both its own state and the drive it
    // contributes to the membrane within this step.
    const float asc_reset_g = detach_asc_reset ? 0.0f :
        active_gv*k.asc_spike_factor[neuron] +
        ga[2*i]*k.asc_amps[2*neuron] + ga[2*i+1]*k.asc_amps[2*neuron+1];
    const T z_grad = V<T>((detach_reset ? 0.0f : active_gv*k.reset_coeff[neuron])
                          + asc_reset_g);
    // The shifted history hands each slot's gradient to the slot before it;
    // the newest slot also carries prev_z's gradient.
    z_buf_grad[history] = slots > 1
        ? V<T>(F(gz_buf[history + neurons]) + F(z_grad)) : z_grad;
    for (int slot = 1; slot < slots; ++slot)
      z_buf_grad[history + slot*neurons] = slot + 1 < slots
          ? gz_buf[history + (slot+1)*neurons] : static_cast<T>(0.0f);
    // The hard reset replaces the pre-spike membrane, so it carries no gradient.
    v_grad[i] = active_gv*k.decay[neuron]*
        (hard_reset ? 1.0f - F(z_buf[history]) : 1.0f);
    asc_grad[2*i] = active_gv*k.asc_factor[2*neuron] +
                    ga[2*i]*k.asc_decay[2*neuron];
    asc_grad[2*i+1] = active_gv*k.asc_factor[2*neuron+1] +
                      ga[2*i+1]*k.asc_decay[2*neuron+1];
    const int parameter_base = neuron*Basis; const int64_t state_base = i*Basis;
    #pragma unroll
    for (int b = 0; b < Basis; ++b) {
      const int64_t j = state_base+b;
      const float4 c = k.syn[parameter_base + b];
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

// The spike history must hold whole delay slots of the membrane's neurons.
static absl::Status ValidateHistory(const Tensor& z_buf, const Tensor& v) {
  if (z_buf.dims() != 2 || v.dims() != 2 || z_buf.dim_size(0) != v.dim_size(0))
    return errors::InvalidArgument("z_buf and v must be [batch, ...] with one batch");
  const int64_t neurons = v.dim_size(1);
  if (neurons == 0 || z_buf.dim_size(1) == 0 || z_buf.dim_size(1) % neurons != 0)
    return errors::InvalidArgument("z_buf width must be a positive multiple of neurons");
  return absl::OkStatus();
}

// Inputs `first` to `first + 6` of either op, in registration order.
static Constants ReadConstants(OpKernelContext* c, int first) {
  return Constants{
      reinterpret_cast<const float4*>(c->input(first).flat<float>().data()),
      c->input(first + 1).flat<float>().data(),
      c->input(first + 2).flat<float>().data(),
      c->input(first + 3).flat<float>().data(),
      c->input(first + 4).flat<float>().data(),
      c->input(first + 5).flat<float>().data(),
      c->input(first + 6).flat<float>().data()};
}

template <typename T, typename R> class ForwardOp : public OpKernel {
 public:
  explicit ForwardOp(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("hard_reset", &hard_));
    OP_REQUIRES_OK(c, c->GetAttr("emit_voltage", &emit_voltage_));
  }
  void Compute(OpKernelContext* c) override {
    const Tensor& z_buf = c->input(0); const Tensor& v = c->input(1);
    const Tensor& psc = c->input(5);
    OP_REQUIRES_OK(c, ValidateHistory(z_buf, v));
    const int neurons = v.dim_size(1);
    const int slots = z_buf.dim_size(1) / neurons;
    OP_REQUIRES_OK(c, ValidateSynapticCoefficients(c->input(7), neurons,
                                                   psc.dim_size(1) / neurons));
    Tensor *spikes, *nz, *ov, *orr, *oa, *orise, *opsc, *refractory, *voltage;
    OP_REQUIRES_OK(c, c->allocate_output(0, v.shape(), &spikes));
    OP_REQUIRES_OK(c, c->allocate_output(1, z_buf.shape(), &nz));
    OP_REQUIRES_OK(c, c->allocate_output(2, v.shape(), &ov));
    OP_REQUIRES_OK(c, c->allocate_output(3, c->input(2).shape(), &orr));
    OP_REQUIRES_OK(c, c->allocate_output(4, c->input(3).shape(), &oa));
    OP_REQUIRES_OK(c, c->allocate_output(5, c->input(4).shape(), &orise));
    OP_REQUIRES_OK(c, c->allocate_output(6, psc.shape(), &opsc));
    OP_REQUIRES_OK(c, c->allocate_output(7, v.shape(), &refractory));
    OP_REQUIRES_OK(c, c->allocate_output(
        8, emit_voltage_ ? v.shape() : TensorShape({0}), &voltage));
    auto& d = c->eigen_device<GPUDevice>();
    auto cfg = GetGpuLaunchConfig(v.NumElements(), d);
    OP_REQUIRES_OK(c, GpuLaunchKernel(ForwardKernel<T, R, kBasis>,
        cfg.block_count, cfg.thread_per_block, 0, d.stream(),
        v.NumElements(), neurons, slots,
        z_buf.flat<T>().data(), v.flat<float>().data(),
        c->input(2).flat<R>().data(), c->input(3).flat<float>().data(),
        c->input(4).flat<T>().data(), psc.flat<T>().data(),
        c->input(6).flat<T>().data(), ReadConstants(c, 7),
        c->input(14).flat<R>().data(), c->input(15).flat<float>().data(),
        c->input(16).flat<float>().data(), c->input(17).flat<float>().data(),
        hard_, emit_voltage_,
        spikes->flat<T>().data(), nz->flat<T>().data(),
        ov->flat<float>().data(), orr->flat<R>().data(),
        oa->flat<float>().data(), orise->flat<T>().data(),
        opsc->flat<T>().data(), refractory->flat<bool>().data(),
        voltage->flat<T>().data()));
  }
 private:
  bool hard_; bool emit_voltage_;
};

template <typename T> class BackwardOp : public OpKernel {
 public:
  explicit BackwardOp(OpKernelConstruction* c) : OpKernel(c) {
    string surrogate;
    OP_REQUIRES_OK(c, c->GetAttr("surrogate", &surrogate));
    surrogate_ = surrogate == "gaussian" ? 1 : surrogate == "slayer" ? 2 : 0;
    OP_REQUIRES_OK(c, c->GetAttr("hard_reset", &hard_));
    OP_REQUIRES_OK(c, c->GetAttr("detach_reset", &detach_));
    OP_REQUIRES_OK(c, c->GetAttr("detach_asc_reset", &detach_asc_));
    OP_REQUIRES_OK(c, c->GetAttr("emit_voltage", &emit_voltage_));
  }
  void Compute(OpKernelContext* c) override {
    const Tensor& z_buf = c->input(0); const Tensor& new_v = c->input(1);
    const Tensor& grise = c->input(18);
    OP_REQUIRES_OK(c, ValidateHistory(z_buf, new_v));
    const int neurons = new_v.dim_size(1);
    const int slots = z_buf.dim_size(1) / neurons;
    OP_REQUIRES_OK(c, ValidateSynapticCoefficients(c->input(3), neurons,
                                                   grise.dim_size(1) / neurons));
    OP_REQUIRES(c, !emit_voltage_ ||
                c->input(20).NumElements() == new_v.NumElements(),
                errors::InvalidArgument("grad_voltage must match new_v"));
    Tensor *zg, *vg, *ag, *rg, *pg, *ig;
    OP_REQUIRES_OK(c, c->allocate_output(0, z_buf.shape(), &zg));
    OP_REQUIRES_OK(c, c->allocate_output(1, new_v.shape(), &vg));
    OP_REQUIRES_OK(c, c->allocate_output(2, c->input(17).shape(), &ag));
    OP_REQUIRES_OK(c, c->allocate_output(3, grise.shape(), &rg));
    OP_REQUIRES_OK(c, c->allocate_output(4, grise.shape(), &pg));
    OP_REQUIRES_OK(c, c->allocate_output(5, grise.shape(), &ig));
    auto& d = c->eigen_device<GPUDevice>();
    auto cfg = GetGpuLaunchConfig(new_v.NumElements(), d);
    OP_REQUIRES_OK(c, GpuLaunchKernel(BackwardKernel<T, kBasis>,
        cfg.block_count, cfg.thread_per_block, 0, d.stream(),
        new_v.NumElements(), neurons, slots,
        z_buf.flat<T>().data(), new_v.flat<float>().data(),
        c->input(2).flat<bool>().data(), ReadConstants(c, 3),
        c->input(10).flat<float>().data(), c->input(11).flat<float>().data(),
        c->input(12).flat<float>().data(), c->input(13).flat<float>().data(),
        surrogate_, c->input(14).flat<T>().data(),
        c->input(15).flat<T>().data(), c->input(16).flat<float>().data(),
        c->input(17).flat<float>().data(), grise.flat<T>().data(),
        c->input(19).flat<T>().data(), c->input(20).flat<T>().data(),
        hard_, detach_, detach_asc_, emit_voltage_,
        zg->flat<T>().data(), vg->flat<float>().data(),
        ag->flat<float>().data(), rg->flat<T>().data(),
        pg->flat<T>().data(), ig->flat<T>().data()));
  }
 private:
  int surrogate_; bool hard_; bool detach_; bool detach_asc_;
  bool emit_voltage_;
};

#define REG(T,R) REGISTER_KERNEL_BUILDER(Name("FusedGlifStep").Device(DEVICE_GPU).TypeConstraint<T>("T").TypeConstraint<R>("R"),ForwardOp<T,R>);
REG(float,int8); REG(float,int16); REG(Eigen::half,int8); REG(Eigen::half,int16);
#undef REG
#define REG(T) REGISTER_KERNEL_BUILDER(Name("FusedGlifStepBackward").Device(DEVICE_GPU).TypeConstraint<T>("T"),BackwardOp<T>);
REG(float); REG(Eigen::half);
#undef REG
#endif

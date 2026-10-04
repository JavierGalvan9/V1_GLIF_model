#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include <type_traits>
#include <limits>

using namespace tensorflow;
using GPUDevice = Eigen::GpuDevice;
template <typename T> __device__ float F(T x) { return static_cast<float>(x); }
template <typename T> __device__ T V(float x) { return static_cast<T>(x); }

constexpr int kBasis = 4;
constexpr int kTileBatch = 2;
constexpr int kTileThreads = 512;

// The per-neuron float32 constants both kernels read. Each syn entry is one
// (neuron, basis) pair, [x, y, z, w] = [syn_decay, psc_initial, psc_factor,
// psc_rise_factor].
struct Constants {
  const float4* syn; const float* asc_decay; const float* asc_amps;
  const float* decay; const float* asc_factor; const float* reset_coeff;
  const float* asc_spike_factor;
};

// The online voltage penalty (models.py, _range/_threshold_voltage_penalty_mean)
// of one float32 membrane value, and its membrane gradient, rounded step by
// step in the order the unfused TensorFlow graph evaluates them (the _rn
// intrinsics keep nvcc from contracting them into FMAs). Modes: 1 = range,
// 2 = threshold; 0 disables the penalty.
__device__ __forceinline__ float PenaltyValue(int mode, float v) {
  if (mode == 1) {
    const float outside = fmaxf(__fsub_rn(fabsf(__fsub_rn(v, 0.5f)), 0.5f), 0.0f);
    return __fmul_rn(outside, outside);
  }
  const float offset = __fsub_rn(v, 1.0f);
  return __fmul_rn(offset, offset);
}

__device__ __forceinline__ float PenaltyGradient(int mode, float v, float dy,
                                                 float inverse_neurons) {
  float factor;
  if (mode == 1) {
    const float centered = __fsub_rn(v, 0.5f);
    const float outside = fmaxf(__fsub_rn(fabsf(centered), 0.5f), 0.0f);
    const float sign = centered > 0.0f ? 1.0f : (centered < 0.0f ? -1.0f : 0.0f);
    factor = __fmul_rn(__fmul_rn(2.0f, outside), sign);
  } else {
    factor = __fmul_rn(2.0f, __fsub_rn(v, 1.0f));
  }
  return __fmul_rn(__fmul_rn(dy, inverse_neurons), factor);
}

// The four-edge background (BKG) current of one (sample, neuron), added to the
// neuron's input current exactly as BkgGatherKernel adds it: the fp16 input is
// widened, the edges are accumulated in fp32 in incoming order, and the sums
// are rounded back to T once. Its result replaces inputs[j] for the step.
struct Bkg {
  const void* activity; int n_pre; const float* weights; const unsigned* pre_ids;
  const unsigned* edge_ids; const unsigned char* types; const float* basis;
};

// A neuron's four incoming BKG edges: everything but the sample's activity, so
// it is loaded once per neuron and reused for every sample of the tile.
struct BkgEdges { unsigned pre[4]; float weight[4]; float4 projection[4]; };

__device__ __forceinline__ void LoadBkgEdges(const Bkg& bkg, int neuron, BkgEdges& edges) {
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const unsigned incoming = 4u * neuron + k;
    edges.pre[k] = bkg.pre_ids[incoming];
    edges.weight[k] = bkg.weights[bkg.edge_ids[incoming]];
    edges.projection[k] = *reinterpret_cast<const float4*>(bkg.basis + bkg.types[incoming] * 4);
  }
}

template <typename T>
__device__ __forceinline__ void AddBkg(const Bkg& bkg, const BkgEdges& edges, int64_t sample,
                                       const T* inputs, int64_t state_base, float (&out)[4]) {
  float4 sums = make_float4(F(inputs[state_base]), F(inputs[state_base + 1]),
                            F(inputs[state_base + 2]), F(inputs[state_base + 3]));
  const T* activity = static_cast<const T*>(bkg.activity) + sample * bkg.n_pre;
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const float spike = F(activity[edges.pre[k]]);
    if (spike == 0.0f) continue;
    const float weighted = edges.weight[k] * spike;
    const float4 projection = edges.projection[k];
    sums.x += weighted * projection.x;
    sums.y += weighted * projection.y;
    sums.z += weighted * projection.z;
    sums.w += weighted * projection.w;
  }
  out[0] = F(V<T>(sums.x)); out[1] = F(V<T>(sums.y));
  out[2] = F(V<T>(sums.z)); out[3] = F(V<T>(sums.w));
}

// Sums each sample's per-thread penalty over the block and writes one
// partial per (sample, block); FinalizePenaltyKernel adds the partials in a
// fixed order, so the reduction is deterministic. Needs blockDim.x % 32 == 0.
template <int Tile>
__device__ void StoreBlockPenalty(const float (&local)[Tile], float* partials,
                                  int64_t first_sample, int64_t batch) {
  __shared__ float warp_sums[kTileThreads / 32][Tile];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
  for (int s = 0; s < Tile; ++s) {
    float x = local[s];
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
      x += __shfl_down_sync(0xffffffffu, x, offset);
    if (lane == 0) warp_sums[warp][s] = x;
  }
  __syncthreads();
  if (threadIdx.x < Tile && first_sample + threadIdx.x < batch) {
    float total = 0.f;
    for (int w = 0; w < static_cast<int>(blockDim.x / 32); ++w) total += warp_sums[w][threadIdx.x];
    partials[(first_sample + threadIdx.x) * gridDim.x + blockIdx.x] = total;
  }
}

// new_acc[b] = acc[b] + mean_n p(v[b, n]).
// One warp per sample: each lane sums a fixed stride of the partials, then a
// fixed shuffle tree combines the lanes, so the order never changes.
__global__ void FinalizePenaltyKernel(int batch, int parts, const float* partials,
                                      const float* acc, const float* inverse_neurons,
                                      float* new_acc) {
  const int b = blockIdx.x;
  if (b >= batch) return;
  float total = 0.f;
  for (int part = threadIdx.x; part < parts; part += 32) total += partials[b * parts + part];
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    total += __shfl_down_sync(0xffffffffu, total, offset);
  if (threadIdx.x == 0)
    new_acc[b] = __fadd_rn(acc[b], __fmul_rn(total, *inverse_neurons));
}

// One thread per (sample, neuron). The synaptic state is read and written in
// T; everything else is float32 (see glif_state_ops.cc). prev_z holds the
// previous step's spikes.
template <typename T, typename R, int Basis>
__global__ void ForwardKernel(int64_t count, int neurons, int n_basis,
    const T* prev_z, const float* v, const R* r, const float* asc,
    const T* rise, const T* psc, const T* inputs, Constants k, const R* t_ref,
    const float* dt, const float* v_reset, const float* v_th, bool hard_reset,
    bool emit_voltage, T* spikes, float* new_v, R* new_r,
    float* new_asc, T* new_rise, T* new_psc, bool* refractory_out,
    T* voltage_out, int penalty, float* penalty_partials) {
  const float step = *dt; const float rest = *v_reset;
  const float threshold = *v_th;
  const int width = Basis == 0 ? n_basis : Basis;
  GPU_1D_KERNEL_LOOP(i, count) {
    const int64_t sample = i / neurons;
    const int neuron = i - sample * neurons;
    const int parameter_base = neuron * width;
    const int64_t state_base = i * width;
    const float reset = F(prev_z[i]);
    // Each current source is convolved with the membrane kernel through its own
    // coefficient; the coefficients encode the integration scheme.
    float drive = k.asc_factor[2*neuron]*asc[2*i] +
                  k.asc_factor[2*neuron+1]*asc[2*i+1];
    #pragma unroll
    for (int b = 0; b < width; ++b) {
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
    if (emit_voltage) voltage_out[i] = V<T>(voltage);
    // The generic width is not the production path: one atomic per element
    // into the sample's single partial, so its rounding depends on the order.
    if (penalty) atomicAdd(penalty_partials + sample, PenaltyValue(penalty, voltage));
  }
}

template <typename T, int Basis>
__global__ void BackwardKernel(int64_t count, int neurons, int n_basis,
    const T* prev_z, const float* new_v, const bool* refractory, Constants k,
    const float* dt, const float* v_th, const float* sigma,
    const float* amplitude, int surrogate, const T* gspikes, const T* gnew_z,
    const T* gprev_z, const float* gv, const float* ga, const T* grise, const T* gpsc,
    const T* gvoltage, bool hard_reset, bool detach_reset,
    bool detach_asc_reset, bool emit_voltage, T* prev_z_grad, float* v_grad,
    float* asc_grad, T* rise_grad, T* psc_grad, T* input_grad, int penalty,
    const float* grad_penalty, const float* inverse_neurons) {
  const float step = *dt; const float threshold = *v_th;
  const float scale = *sigma; const float gain = *amplitude;
  const int width = Basis == 0 ? n_basis : Basis;
  GPU_1D_KERNEL_LOOP(i, count) {
    const int64_t sample = i / neurons;
    const int neuron = i - sample * neurons;
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
    const T upstream = static_cast<T>(F(gspikes[i]) + F(gnew_z[i]));
    // The penalty's membrane gradient joins grad_v first, as AddN does in the
    // unfused graph.
    const float gv_total = penalty
        ? __fadd_rn(gv[i], PenaltyGradient(penalty, new_v[i], grad_penalty[sample],
                                           *inverse_neurons))
        : gv[i];
    float membrane_g = gv_total + (blocked ? 0.0f : F(upstream) * (shape * gain));
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
    // prev_z also moves on as the next step's second history slot, whose
    // gradient joins the reset's here.
    prev_z_grad[i] = gprev_z ? V<T>(F(gprev_z[i]) + F(z_grad)) : z_grad;
    // The hard reset replaces the pre-spike membrane, so it carries no gradient.
    v_grad[i] = active_gv*k.decay[neuron]*
        (hard_reset ? 1.0f - F(prev_z[i]) : 1.0f);
    asc_grad[2*i] = active_gv*k.asc_factor[2*neuron] +
                    ga[2*i]*k.asc_decay[2*neuron];
    asc_grad[2*i+1] = active_gv*k.asc_factor[2*neuron+1] +
                      ga[2*i+1]*k.asc_decay[2*neuron+1];
    const int parameter_base = neuron*width; const int64_t state_base = i*width;
    #pragma unroll
    for (int b = 0; b < width; ++b) {
      const int64_t j = state_base+b;
      const float4 c = k.syn[parameter_base + b];
      rise_grad[j] = V<T>(F(grise[j])*c.x + F(gpsc[j])*step*c.x +
                          active_gv*c.w);
      psc_grad[j] = V<T>(F(gpsc[j])*c.x + active_gv*c.z);
      input_grad[j] = V<T>(F(grise[j])*c.y);
    }
  }
}

// For four bases, neighbouring samples reuse the same four neuron coefficients.
// The generic kernel below handles every other positive basis width.
// Launched with kTileThreads threads; the bound keeps the register allocation
// (with the fused penalty and BKG gather) within one 512-thread block per SM.
template <typename T, typename R, int Basis>
__global__ __launch_bounds__(kTileThreads, 1) void ForwardKernelTiled(int64_t count, int neurons, int n_basis,
    const T* prev_z, const float* v, const R* r, const float* asc,
    const T* rise, const T* psc, const T* inputs, Constants k, const R* t_ref,
    const float* dt, const float* v_reset, const float* v_th, bool hard_reset,
    bool emit_voltage, T* spikes, float* new_v, R* new_r,
    float* new_asc, T* new_rise, T* new_psc, bool* refractory_out,
    T* voltage_out, int penalty, float* penalty_partials, Bkg bkg) {
  const float step = *dt; const float rest = *v_reset;
  const float threshold = *v_th;
  const int64_t batch = count / neurons;
  const int64_t tile_first = static_cast<int64_t>(blockIdx.y) * kTileBatch;
  float penalty_sum[kTileBatch] = {};
  for (int neuron = blockIdx.x * blockDim.x + threadIdx.x;
       neuron < neurons; neuron += blockDim.x * gridDim.x) {
    float4 coeff[Basis];
#pragma unroll
    for (int b = 0; b < Basis; ++b) coeff[b] = k.syn[neuron * Basis + b];
    BkgEdges bkg_edges;
    if constexpr (Basis == 4) { if (bkg.activity) LoadBkgEdges(bkg, neuron, bkg_edges); }
    const int64_t first_sample = static_cast<int64_t>(blockIdx.y) * kTileBatch;
    for (int64_t sample = first_sample;
         sample < batch && sample < first_sample + kTileBatch; ++sample) {
    const int64_t i = sample * neurons + neuron;
    const int parameter_base = neuron * Basis;
    const int64_t state_base = i * Basis;
    const float reset = F(prev_z[i]);
    // Each current source is convolved with the membrane kernel through its own
    // coefficient; the coefficients encode the integration scheme.
    float drive = k.asc_factor[2*neuron]*asc[2*i] +
                  k.asc_factor[2*neuron+1]*asc[2*i+1];
    float input_current[Basis];
    if (Basis == 4 && bkg.activity) {
      AddBkg<T>(bkg, bkg_edges, sample, inputs, state_base,
                reinterpret_cast<float (&)[4]>(input_current));
    } else {
#pragma unroll
      for (int b = 0; b < Basis; ++b) input_current[b] = F(inputs[state_base + b]);
    }
    #pragma unroll
    for (int b = 0; b < Basis; ++b) {
      const int64_t j = state_base + b;
      const float4 c = coeff[b];
      const float old_rise = F(rise[j]); const float old_psc = F(psc[j]);
      drive += c.z*old_psc + c.w*old_rise;
      new_rise[j] = V<T>(old_rise*c.x + input_current[b]*c.y);
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
    if (emit_voltage) voltage_out[i] = V<T>(voltage);
    if (penalty) penalty_sum[sample - first_sample] += PenaltyValue(penalty, voltage);
    }
  }
  if (penalty) StoreBlockPenalty<kTileBatch>(penalty_sum, penalty_partials, tile_first, batch);
}

template <typename T, int Basis>
__global__ void BackwardKernelTiled(int64_t count, int neurons, int n_basis,
    const T* prev_z, const float* new_v, const bool* refractory, Constants k,
    const float* dt, const float* v_th, const float* sigma,
    const float* amplitude, int surrogate, const T* gspikes, const T* gnew_z,
    const T* gprev_z, const float* gv, const float* ga, const T* grise, const T* gpsc,
    const T* gvoltage, bool hard_reset, bool detach_reset,
    bool detach_asc_reset, bool emit_voltage, T* prev_z_grad, float* v_grad,
    float* asc_grad, T* rise_grad, T* psc_grad, T* input_grad, int penalty,
    const float* grad_penalty, const float* inverse_neurons) {
  const float step = *dt; const float threshold = *v_th;
  const float scale = *sigma; const float gain = *amplitude;
  const int64_t batch = count / neurons;
  for (int neuron = blockIdx.x * blockDim.x + threadIdx.x;
       neuron < neurons; neuron += blockDim.x * gridDim.x) {
    float4 coeff[Basis];
#pragma unroll
    for (int b = 0; b < Basis; ++b) coeff[b] = k.syn[neuron * Basis + b];
    const int64_t first_sample = static_cast<int64_t>(blockIdx.y) * kTileBatch;
    for (int64_t sample = first_sample;
         sample < batch && sample < first_sample + kTileBatch; ++sample) {
    const int64_t i = sample * neurons + neuron;
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
    const T upstream = static_cast<T>(F(gspikes[i]) + F(gnew_z[i]));
    // The penalty's membrane gradient joins grad_v first, as AddN does in the
    // unfused graph.
    const float gv_total = penalty
        ? __fadd_rn(gv[i], PenaltyGradient(penalty, new_v[i], grad_penalty[sample],
                                           *inverse_neurons))
        : gv[i];
    float membrane_g = gv_total + (blocked ? 0.0f : F(upstream) * (shape * gain));
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
    // prev_z also moves on as the next step's second history slot, whose
    // gradient joins the reset's here.
    prev_z_grad[i] = gprev_z ? V<T>(F(gprev_z[i]) + F(z_grad)) : z_grad;
    // The hard reset replaces the pre-spike membrane, so it carries no gradient.
    v_grad[i] = active_gv*k.decay[neuron]*
        (hard_reset ? 1.0f - F(prev_z[i]) : 1.0f);
    asc_grad[2*i] = active_gv*k.asc_factor[2*neuron] +
                    ga[2*i]*k.asc_decay[2*neuron];
    asc_grad[2*i+1] = active_gv*k.asc_factor[2*neuron+1] +
                      ga[2*i+1]*k.asc_decay[2*neuron+1];
    const int parameter_base = neuron*Basis; const int64_t state_base = i*Basis;
    #pragma unroll
    for (int b = 0; b < Basis; ++b) {
      const int64_t j = state_base+b;
      const float4 c = coeff[b];
      rise_grad[j] = V<T>(F(grise[j])*c.x + F(gpsc[j])*step*c.x +
                          active_gv*c.w);
      psc_grad[j] = V<T>(F(gpsc[j])*c.x + active_gv*c.z);
      input_grad[j] = V<T>(F(grise[j])*c.y);
    }
    }
  }
}

// The float4 loads need a whole, 16-byte aligned coefficient block.
static absl::Status ValidateSynapticCoefficients(const Tensor& coeffs, int64_t neurons,
                                           int64_t basis) {
  if (basis <= 0 || basis > std::numeric_limits<int>::max())
    return errors::InvalidArgument("synaptic basis dimension is out of range");
  if (coeffs.NumElements() != neurons * basis * 4)
    return errors::InvalidArgument("syn_coeffs must hold four constants per neuron and basis");
  if (reinterpret_cast<uintptr_t>(coeffs.flat<float>().data()) % alignof(float4) != 0)
    return errors::InvalidArgument("syn_coeffs must be 16-byte aligned");
  return absl::OkStatus();
}

// The previous spikes are one [batch, neurons] slot, shaped like the membrane.
static absl::Status ValidateHistory(const Tensor& prev_z, const Tensor& v) {
  if (v.dims() != 2 || prev_z.shape() != v.shape())
    return errors::InvalidArgument("prev_z and v must both be [batch, neurons]");
  if (v.dim_size(1) == 0 || v.dim_size(1) > std::numeric_limits<int>::max())
    return errors::InvalidArgument("the neuron count is out of range");
  return absl::OkStatus();
}

static absl::Status ValidateState(const Tensor& value, int64_t batch,
                                   int64_t neurons, int64_t width,
                                   const char* name) {
  if (value.dims() != 2 || value.dim_size(0) != batch ||
      width <= 0 || value.dim_size(1) != neurons * width)
    return errors::InvalidArgument(name, " has an incompatible state shape");
  return absl::OkStatus();
}

static absl::Status ValidateConstants(OpKernelContext* c, int first,
                                       int64_t neurons) {
  for (int index = 1; index < 7; ++index) {
    const int64_t width = index == 1 || index == 2 || index == 4 ? 2 : 1;
    if (c->input(first + index).NumElements() != neurons * width)
      return errors::InvalidArgument("GLIF constant shape mismatch at ", first + index);
  }
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
    string penalty;
    OP_REQUIRES_OK(c, c->GetAttr("penalty", &penalty));
    penalty_ = penalty == "range" ? 1 : penalty == "threshold" ? 2 : 0;
  }
  void Compute(OpKernelContext* c) override {
    const Tensor& prev_z = c->input(0); const Tensor& v = c->input(1);
    const Tensor& psc = c->input(5);
    OP_REQUIRES_OK(c, ValidateHistory(prev_z, v));
    const int neurons = v.dim_size(1);
    OP_REQUIRES(c, psc.dims() == 2 && psc.dim_size(1) % neurons == 0,
                errors::InvalidArgument("psc must contain whole synaptic bases"));
    const int64_t basis = psc.dim_size(1) / neurons;
    OP_REQUIRES_OK(c, ValidateState(psc, v.dim_size(0), neurons, basis, "psc"));
    OP_REQUIRES_OK(c, ValidateState(c->input(4), v.dim_size(0), neurons, basis, "rise"));
    OP_REQUIRES_OK(c, ValidateState(c->input(6), v.dim_size(0), neurons, basis, "inputs"));
    OP_REQUIRES_OK(c, ValidateState(c->input(3), v.dim_size(0), neurons, 2, "asc"));
    OP_REQUIRES_OK(c, ValidateState(c->input(2), v.dim_size(0), neurons, 1, "r"));
    OP_REQUIRES_OK(c, ValidateConstants(c, 7, neurons));
    OP_REQUIRES(c, c->input(14).NumElements() == neurons &&
                   c->input(15).NumElements() == 1 && c->input(16).NumElements() == 1 &&
                   c->input(17).NumElements() == 1,
                errors::InvalidArgument("refractory or scalar constant shape mismatch"));
    OP_REQUIRES_OK(c, ValidateSynapticCoefficients(c->input(7), neurons,
                                                   psc.dim_size(1) / neurons));
    Tensor *spikes, *ov, *orr, *oa, *orise, *opsc, *refractory, *voltage;
    OP_REQUIRES_OK(c, c->allocate_output(0, v.shape(), &spikes));
    OP_REQUIRES_OK(c, c->allocate_output(1, v.shape(), &ov));
    OP_REQUIRES_OK(c, c->allocate_output(2, c->input(2).shape(), &orr));
    OP_REQUIRES_OK(c, c->allocate_output(3, c->input(3).shape(), &oa));
    OP_REQUIRES_OK(c, c->allocate_output(4, c->input(4).shape(), &orise));
    OP_REQUIRES_OK(c, c->allocate_output(5, psc.shape(), &opsc));
    OP_REQUIRES_OK(c, c->allocate_output(6, v.shape(), &refractory));
    OP_REQUIRES_OK(c, c->allocate_output(
        7, emit_voltage_ ? v.shape() : TensorShape({0}), &voltage));
    const Tensor& penalty_acc = c->input(18);
    Tensor* new_penalty_acc = nullptr;
    Tensor partials;
    if (!penalty_) {
      c->set_output(8, penalty_acc);
    } else {
      OP_REQUIRES(c, penalty_acc.NumElements() == v.dim_size(0) &&
                     c->input(19).NumElements() == 1,
                  errors::InvalidArgument("penalty_acc must be [batch, 1] and inverse_neurons a scalar"));
      OP_REQUIRES_OK(c, c->allocate_output(8, penalty_acc.shape(), &new_penalty_acc));
    }
    const Tensor& bkg_activity = c->input(20);
    Bkg bkg{nullptr, 0, nullptr, nullptr, nullptr, nullptr, nullptr};
    if (bkg_activity.NumElements()) {
      OP_REQUIRES(c, basis == kBasis && bkg_activity.dims() == 2 &&
                     bkg_activity.dim_size(0) == v.dim_size(0) &&
                     c->input(22).NumElements() == 4 * neurons &&
                     c->input(23).NumElements() == 4 * neurons &&
                     c->input(24).NumElements() == 4 * neurons &&
                     c->input(25).dims() == 2 && c->input(25).dim_size(1) == 4,
                  errors::InvalidArgument("the fused BKG gather needs four bases, [batch, n_bkg] "
                                          "activity and four incoming edges per neuron"));
      bkg = Bkg{bkg_activity.flat<T>().data(), static_cast<int>(bkg_activity.dim_size(1)),
                c->input(21).flat<float>().data(), c->input(22).flat<uint32>().data(),
                c->input(23).flat<uint32>().data(), c->input(24).flat<uint8>().data(),
                c->input(25).flat<float>().data()};
    }
    auto& d = c->eigen_device<GPUDevice>();
    auto launch = [&](auto tag) {
      constexpr int B = decltype(tag)::value;
      auto kernel = [] {
        if constexpr (B > 0) return ForwardKernelTiled<T, R, B>;
        else return ForwardKernel<T, R, 0>;
      }();
      dim3 blocks;
      int threads;
      if constexpr (B > 0) {
        auto cfg = GetGpuLaunchConfigFixedBlockSize(
            neurons, d, kernel, 0, kTileThreads);
        blocks = dim3(cfg.block_count,
                      (v.dim_size(0) + kTileBatch - 1) / kTileBatch);
        threads = cfg.thread_per_block;
      } else {
        auto cfg = GetGpuLaunchConfig(v.NumElements(), d);
        blocks = dim3(cfg.block_count);
        threads = cfg.thread_per_block;
      }
      // One partial per (sample, block) for the tiled kernel, one per
      // sample (accumulated atomically, so zeroed first) for the generic one.
      const int parts = B > 0 ? static_cast<int>(blocks.x) : 1;
      float* partial_data = nullptr;
      if (penalty_) {
        TF_RETURN_IF_ERROR(c->allocate_temp(
            DT_FLOAT, TensorShape({v.dim_size(0) * parts}), &partials));
        partial_data = partials.flat<float>().data();
        if (B == 0 && cudaMemsetAsync(partial_data, 0, partials.TotalBytes(),
                                           d.stream()) != cudaSuccess)
          return errors::Internal("penalty partial memset failed");
        if (B > 0 && threads % 32 != 0)
          return errors::Internal("the penalty reduction needs whole warps");
      }
      auto launch_kernel = [&](auto... tail) {
        if constexpr (B > 0) return GpuLaunchKernel(kernel, blocks, threads, 0, d.stream(), tail..., bkg);
        else return GpuLaunchKernel(kernel, blocks, threads, 0, d.stream(), tail...);
      };
      TF_RETURN_IF_ERROR(launch_kernel(
        v.NumElements(), neurons, psc.dim_size(1) / neurons,
        prev_z.flat<T>().data(), v.flat<float>().data(),
        c->input(2).flat<R>().data(), c->input(3).flat<float>().data(),
        c->input(4).flat<T>().data(), psc.flat<T>().data(),
        c->input(6).flat<T>().data(), ReadConstants(c, 7),
        c->input(14).flat<R>().data(), c->input(15).flat<float>().data(),
        c->input(16).flat<float>().data(), c->input(17).flat<float>().data(),
        hard_, emit_voltage_,
        spikes->flat<T>().data(), ov->flat<float>().data(), orr->flat<R>().data(),
        oa->flat<float>().data(), orise->flat<T>().data(),
        opsc->flat<T>().data(), refractory->flat<bool>().data(),
        voltage->flat<T>().data(), penalty_, partial_data));
      if (!penalty_) return absl::OkStatus();
      const int batch = static_cast<int>(v.dim_size(0));
      return GpuLaunchKernel(FinalizePenaltyKernel, dim3(batch), 32, 0,
                             d.stream(), batch, parts, partial_data,
                             penalty_acc.flat<float>().data(), c->input(19).flat<float>().data(),
                             new_penalty_acc->flat<float>().data());
    };
    if (basis == 3 && penalty_) {
      OP_REQUIRES_OK(c, launch(std::integral_constant<int, 3>{}));
    } else if (basis == 5 && penalty_) {
      OP_REQUIRES_OK(c, launch(std::integral_constant<int, 5>{}));
    } else if (basis == kBasis) {
      OP_REQUIRES_OK(c, launch(std::integral_constant<int, 4>{}));
    } else {
      OP_REQUIRES_OK(c, launch(std::integral_constant<int, 0>{}));
    }
  }
 private:
  bool hard_; bool emit_voltage_; int penalty_;
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
    string penalty;
    OP_REQUIRES_OK(c, c->GetAttr("penalty", &penalty));
    penalty_ = penalty == "range" ? 1 : penalty == "threshold" ? 2 : 0;
  }
  void Compute(OpKernelContext* c) override {
    const Tensor& prev_z = c->input(0); const Tensor& new_v = c->input(1);
    const Tensor& grise = c->input(19);
    OP_REQUIRES_OK(c, ValidateHistory(prev_z, new_v));
    const int neurons = new_v.dim_size(1);
    OP_REQUIRES(c, grise.dims() == 2 && grise.dim_size(1) % neurons == 0,
                errors::InvalidArgument("grad_rise must contain whole synaptic bases"));
    const int64_t basis = grise.dim_size(1) / neurons;
    OP_REQUIRES_OK(c, ValidateState(grise, new_v.dim_size(0), neurons, basis, "grad_rise"));
    OP_REQUIRES_OK(c, ValidateState(c->input(20), new_v.dim_size(0), neurons, basis, "grad_psc"));
    OP_REQUIRES_OK(c, ValidateState(c->input(18), new_v.dim_size(0), neurons, 2, "grad_asc"));
    for (int index : {2, 14, 15, 17})
      OP_REQUIRES_OK(c, ValidateState(c->input(index), new_v.dim_size(0), neurons, 1, "gradient"));
    // An empty grad_prev_z means prev_z has no later history slot.
    const bool has_prev_grad = c->input(16).NumElements() > 0;
    OP_REQUIRES(c, !has_prev_grad || c->input(16).shape() == prev_z.shape(),
                errors::InvalidArgument("grad_prev_z must be empty or match prev_z"));
    OP_REQUIRES_OK(c, ValidateConstants(c, 3, neurons));
    for (int index = 10; index <= 13; ++index)
      OP_REQUIRES(c, c->input(index).NumElements() == 1,
                  errors::InvalidArgument("backward scalar constant shape mismatch"));
    OP_REQUIRES_OK(c, ValidateSynapticCoefficients(c->input(3), neurons,
                                                   grise.dim_size(1) / neurons));
    OP_REQUIRES(c, !emit_voltage_ ||
                c->input(21).NumElements() == new_v.NumElements(),
                errors::InvalidArgument("grad_voltage must match new_v"));
    OP_REQUIRES(c, !penalty_ || (c->input(22).NumElements() == new_v.dim_size(0) &&
                                 c->input(23).NumElements() == 1),
                errors::InvalidArgument("grad_penalty_acc must be [batch, 1] and inverse_neurons a scalar"));
    Tensor *zg, *vg, *ag, *rg, *pg, *ig;
    OP_REQUIRES_OK(c, c->allocate_output(0, prev_z.shape(), &zg));
    OP_REQUIRES_OK(c, c->allocate_output(1, new_v.shape(), &vg));
    OP_REQUIRES_OK(c, c->allocate_output(2, c->input(18).shape(), &ag));
    OP_REQUIRES_OK(c, c->allocate_output(3, grise.shape(), &rg));
    OP_REQUIRES_OK(c, c->allocate_output(4, grise.shape(), &pg));
    OP_REQUIRES_OK(c, c->allocate_output(5, grise.shape(), &ig));
    auto& d = c->eigen_device<GPUDevice>();
    auto launch = [&](auto tag) {
      constexpr int B = decltype(tag)::value;
      auto kernel = [] {
        if constexpr (B == kBasis) return BackwardKernelTiled<T, kBasis>;
        else return BackwardKernel<T, 0>;
      }();
      dim3 blocks;
      int threads;
      if constexpr (B == kBasis) {
        auto cfg = GetGpuLaunchConfigFixedBlockSize(
            neurons, d, kernel, 0, kTileThreads);
        blocks = dim3(cfg.block_count,
                      (new_v.dim_size(0) + kTileBatch - 1) / kTileBatch);
        threads = cfg.thread_per_block;
      } else {
        auto cfg = GetGpuLaunchConfig(new_v.NumElements(), d);
        blocks = dim3(cfg.block_count);
        threads = cfg.thread_per_block;
      }
      return GpuLaunchKernel(kernel,
        blocks, threads, 0, d.stream(),
        new_v.NumElements(), neurons, grise.dim_size(1) / neurons,
        prev_z.flat<T>().data(), new_v.flat<float>().data(),
        c->input(2).flat<bool>().data(), ReadConstants(c, 3),
        c->input(10).flat<float>().data(), c->input(11).flat<float>().data(),
        c->input(12).flat<float>().data(), c->input(13).flat<float>().data(),
        surrogate_, c->input(14).flat<T>().data(), c->input(15).flat<T>().data(),
        has_prev_grad ? c->input(16).flat<T>().data() : nullptr,
        c->input(17).flat<float>().data(),
        c->input(18).flat<float>().data(), grise.flat<T>().data(),
        c->input(20).flat<T>().data(), c->input(21).flat<T>().data(),
        hard_, detach_, detach_asc_, emit_voltage_,
        zg->flat<T>().data(), vg->flat<float>().data(),
        ag->flat<float>().data(), rg->flat<T>().data(),
        pg->flat<T>().data(), ig->flat<T>().data(), penalty_,
        c->input(22).flat<float>().data(), c->input(23).flat<float>().data());
    };
    if (basis == kBasis) {
      OP_REQUIRES_OK(c, launch(std::integral_constant<int, 4>{}));
    } else {
      OP_REQUIRES_OK(c, launch(std::integral_constant<int, 0>{}));
    }
  }
 private:
  int surrogate_; bool hard_; bool detach_; bool detach_asc_;
  bool emit_voltage_; int penalty_;
};

#define REG(T,R) REGISTER_KERNEL_BUILDER(Name("FusedGlifStep").Device(DEVICE_GPU).TypeConstraint<T>("T").TypeConstraint<R>("R"),ForwardOp<T,R>);
REG(float,int8); REG(float,int16); REG(Eigen::half,int8); REG(Eigen::half,int16);
#undef REG
#define REG(T) REGISTER_KERNEL_BUILDER(Name("FusedGlifStepBackward").Device(DEVICE_GPU).TypeConstraint<T>("T"),BackwardOp<T>);
REG(float); REG(Eigen::half);
#undef REG
#endif

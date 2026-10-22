#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include <type_traits>
#include <algorithm>
#include <limits>

using namespace tensorflow;
using GPUDevice = Eigen::GpuDevice;
template <typename T> __device__ float F(T x) { return static_cast<float>(x); }
template <typename T> __device__ T V(float x) { return static_cast<T>(x); }

constexpr int kBasis = 4;
// Samples per forward thread (sharing the neuron constants in registers).
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

// `inputs` holds the four widened T input currents of the (sample, neuron).
template <typename T>
__device__ __forceinline__ void AddBkg(const Bkg& bkg, const BkgEdges& edges, int64_t sample,
                                       const float (&inputs)[4], float (&out)[4]) {
  float4 sums = make_float4(inputs[0], inputs[1], inputs[2], inputs[3]);
  const T* activity = static_cast<const T*>(bkg.activity) + sample * bkg.n_pre;
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const float spike = F(activity[edges.pre[k]]);
    if (spike == 0.0f) continue;
    const float weighted = __fmul_rn(edges.weight[k], spike);
    const float4 projection = edges.projection[k];
    sums.x = __fmaf_rn(weighted, projection.x, sums.x);
    sums.y = __fmaf_rn(weighted, projection.y, sums.y);
    sums.z = __fmaf_rn(weighted, projection.z, sums.z);
    sums.w = __fmaf_rn(weighted, projection.w, sums.w);
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
    spikes[i] = spike;
    if (refractory_out) refractory_out[i] = blocked;
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

// A row of N adjacent values moved as one aligned access (N * sizeof(T) bytes:
// 8 for four fp16 bases, 16 for four fp32 ones). Only the memory movement is
// vectorized; every value is widened and rounded exactly as a scalar access.
template <typename T, int N> struct alignas(sizeof(T) * N) Pack { T v[N]; };

template <typename T, int N, bool Vec>
__device__ __forceinline__ void LoadRow(const T* __restrict__ p, float (&out)[N]) {
  if constexpr (Vec) {
    const Pack<T, N> row = *reinterpret_cast<const Pack<T, N>*>(p);
#pragma unroll
    for (int b = 0; b < N; ++b) out[b] = F(row.v[b]);
  } else {
#pragma unroll
    for (int b = 0; b < N; ++b) out[b] = F(p[b]);
  }
}

template <typename T, int N, bool Vec>
__device__ __forceinline__ void StoreRow(T* __restrict__ p, const float (&in)[N]) {
  if constexpr (Vec) {
    Pack<T, N> row;
#pragma unroll
    for (int b = 0; b < N; ++b) row.v[b] = V<T>(in[b]);
    *reinterpret_cast<Pack<T, N>*>(p) = row;
  } else {
#pragma unroll
    for (int b = 0; b < N; ++b) p[b] = V<T>(in[b]);
  }
}

// Whether every pointer allows the row accesses of LoadRow/StoreRow: the
// synaptic rows of `basis` T values and the two-value float32 ASC rows.
template <typename T>
static bool RowsAligned(int64_t basis, std::initializer_list<const void*> state,
                        std::initializer_list<const void*> asc) {
  if (basis != kBasis) return false;
  for (const void* p : state)
    if (reinterpret_cast<uintptr_t>(p) % (sizeof(T) * kBasis)) return false;
  for (const void* p : asc)
    if (reinterpret_cast<uintptr_t>(p) % (2 * sizeof(float))) return false;
  return true;
}

// The per-neuron float32 constants the tiled backward loads once per
// (sample, neuron) into registers (the values the generic kernels index).
template <int Basis> struct NeuronConstants {
  float4 syn[Basis];
  float asc_decay[2], asc_amps[2], asc_factor[2];
  float decay, reset_coeff, asc_spike_factor;
  __device__ __forceinline__ NeuronConstants(const Constants& k, int neuron) {
#pragma unroll
    for (int b = 0; b < Basis; ++b) syn[b] = k.syn[neuron * Basis + b];
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      asc_decay[j] = k.asc_decay[2 * neuron + j];
      asc_amps[j] = k.asc_amps[2 * neuron + j];
      asc_factor[j] = k.asc_factor[2 * neuron + j];
    }
    decay = k.decay[neuron]; reset_coeff = k.reset_coeff[neuron];
    asc_spike_factor = k.asc_spike_factor[neuron];
  }
};

// The spikes and the refractory mask as bit-packed int32 words: rows of
// `words` words per sample, value n in bit n % word_bits of word n /
// word_bits, the layout (and the values) of models._pack_spike_words. The
// tiled forward writes 32-bit words (one warp ballot each) to `spikes` and
// `blocked`; with 31-bit words those are a [batch, ceil(neurons / 32)] staging
// buffer that RepackWordsKernel turns into the outputs.
struct PackedMasks { int* spikes; int* blocked; int words; int word_bits; };

// Every thread of the block calls it (`fired` and `blocked` false past the
// last neuron); the warp's 32 neurons start at `warp_first`, a multiple of 32.
__device__ __forceinline__ void StoreWarpMasks(const PackedMasks& stage, int64_t sample,
                                               int warp_first, int neurons, bool fired,
                                               bool blocked) {
  const unsigned fired_bits = __ballot_sync(0xffffffffu, fired);
  const unsigned blocked_bits = __ballot_sync(0xffffffffu, blocked);
  if ((threadIdx.x & 31) == 0 && warp_first < neurons) {
    const int64_t index = sample * stage.words + warp_first / 32;
    stage.spikes[index] = static_cast<int>(fired_bits);
    stage.blocked[index] = static_cast<int>(blocked_bits);
  }
}

// 32-bit staged words -> word_bits-bit output words, one thread per output word.
__global__ void RepackWordsKernel(int64_t rows, PackedMasks stage, PackedMasks out) {
  GPU_1D_KERNEL_LOOP(index, rows * out.words) {
    const int64_t sample = index / out.words;
    const int word = index - sample * out.words;
    const int64_t first = static_cast<int64_t>(word) * out.word_bits;
    const int source = static_cast<int>(first >> 5), shift = static_cast<int>(first & 31);
    const unsigned long long mask = (1ull << out.word_bits) - 1;
    const int* rows_in[2] = {stage.spikes, stage.blocked};
    int* rows_out[2] = {out.spikes, out.blocked};
    for (int which = 0; which < 2; ++which) {
      const int* row = rows_in[which] + sample * stage.words;
      unsigned long long pair = static_cast<unsigned>(row[source]);
      if (source + 1 < stage.words)
        pair |= static_cast<unsigned long long>(static_cast<unsigned>(row[source + 1])) << 32;
      rows_out[which][index] = static_cast<int>((pair >> shift) & mask);
    }
  }
}

// The same words from the dense outputs, one thread per word: for the generic
// kernel, which interleaves samples within a warp.
template <typename T, typename R>
__global__ void PackMasksKernel(int64_t rows, int neurons, const T* spikes, const R* counters,
                                PackedMasks pack) {
  GPU_1D_KERNEL_LOOP(index, rows * pack.words) {
    const int64_t sample = index / pack.words;
    const int word = index - sample * pack.words;
    unsigned fired = 0, blocked = 0;
    for (int bit = 0; bit < pack.word_bits; ++bit) {
      const int neuron = word * pack.word_bits + bit;
      if (neuron >= neurons) break;
      const int64_t i = sample * neurons + neuron;
      fired |= static_cast<unsigned>(F(spikes[i]) != 0.0f) << bit;
      blocked |= static_cast<unsigned>(counters[i] > 0) << bit;
    }
    pack.spikes[index] = static_cast<int>(fired);
    pack.blocked[index] = static_cast<int>(blocked);
  }
}

// For four bases, the kTileBatch samples of a thread reuse the neuron's
// synaptic coefficients and BKG edges, held in registers. The generic kernel
// below handles every other positive basis width. Vec moves the synaptic and
// ASC rows as single aligned accesses (four bases only); a null
// refractory_out skips that output; Pack also writes the packed masks.
// Launched with kTileThreads threads; the bound keeps the register allocation
// (with the fused penalty and BKG gather) within one 512-thread block per SM.
template <typename T, typename R, int Basis, bool Vec, bool Pack>
__global__ __launch_bounds__(kTileThreads, 1) void ForwardKernelTiled(int64_t count, int neurons, int n_basis,
    const T* prev_z, const float* v, const R* r,
    const float* asc, const T* rise, const T* psc,
    const T* inputs, Constants k, const R* t_ref,
    const float* dt, const float* v_reset, const float* v_th, bool hard_reset,
    bool emit_voltage, T* spikes, float* new_v, R* new_r,
    float* new_asc, T* new_rise, T* new_psc,
    bool* refractory_out, T* voltage_out, int penalty,
    float* penalty_partials, Bkg bkg, PackedMasks pack) {
  const float step = *dt; const float rest = *v_reset;
  const float threshold = *v_th;
  const int64_t batch = count / neurons;
  const int64_t tile_first = static_cast<int64_t>(blockIdx.y) * kTileBatch;
  float penalty_sum[kTileBatch] = {};
  // With Pack every thread runs every block iteration (each warp ballots its
  // masks); those past the last neuron only join the ballot.
  for (int start = blockIdx.x * blockDim.x;
       Pack ? start < neurons : start + static_cast<int>(threadIdx.x) < neurons;
       start += blockDim.x * gridDim.x) {
    const int neuron = start + threadIdx.x;
    const bool valid = !Pack || neuron < neurons;
    float4 syn[Basis];
    BkgEdges bkg_edges;
    if (valid) {
#pragma unroll
    for (int b = 0; b < Basis; ++b) syn[b] = k.syn[neuron * Basis + b];
    if constexpr (Basis == 4) { if (bkg.activity) LoadBkgEdges(bkg, neuron, bkg_edges); }
    }
    for (int64_t sample = tile_first;
         sample < batch && sample < tile_first + kTileBatch; ++sample) {
    [[maybe_unused]] bool fired = false, blocked_bit = false;
    if (valid) {
    const int64_t i = sample * neurons + neuron;
    const int64_t state_base = i * Basis;
    // The accesses keep the former per-element order: hoisting the constants
    // or issuing every row load first measured slower under the power cap.
    const float reset = F(prev_z[i]);
    float old_asc[2], old_rise[Basis], old_psc[Basis], input_current[Basis];
    LoadRow<float, 2, Vec>(asc + 2 * i, old_asc);
    // Each current source is convolved with the membrane kernel through its own
    // coefficient; the coefficients encode the integration scheme.
    // The arithmetic spells out the contraction nvcc chose for the former
    // per-element code (read from its SASS), so the results stay bit for bit.
    float drive = __fmaf_rn(k.asc_factor[2 * neuron], old_asc[0],
                            __fmul_rn(k.asc_factor[2 * neuron + 1], old_asc[1]));
    if (Basis == 4 && bkg.activity) {
      float raw[4];
      LoadRow<T, 4, Vec>(inputs + state_base, raw);
      AddBkg<T>(bkg, bkg_edges, sample, raw, reinterpret_cast<float (&)[4]>(input_current));
    } else {
      LoadRow<T, Basis, Vec>(inputs + state_base, input_current);
    }
    LoadRow<T, Basis, Vec>(rise + state_base, old_rise);
    LoadRow<T, Basis, Vec>(psc + state_base, old_psc);
    float next_rise[Basis], next_psc[Basis];
    #pragma unroll
    for (int b = 0; b < Basis; ++b) {
      const float4 c = syn[b];
      drive = __fadd_rn(drive, __fmaf_rn(c.w, old_rise[b], __fmul_rn(c.z, old_psc[b])));
      next_rise[b] = __fmaf_rn(c.x, old_rise[b], __fmul_rn(c.y, input_current[b]));
      next_psc[b] = __fmaf_rn(old_rise[b], __fmul_rn(step, c.x), __fmul_rn(c.x, old_psc[b]));
    }
    StoreRow<T, Basis, Vec>(new_rise + state_base, next_rise);
    StoreRow<T, Basis, Vec>(new_psc + state_base, next_psc);
    const int refractory = max(static_cast<int>(r[i]) +
        static_cast<int>(reset)*static_cast<int>(t_ref[neuron])-1, 0);
    const float old_v = v[i]; const float decay = k.decay[neuron];
    const float reset_coeff = k.reset_coeff[neuron];
    // The reset and the ASC this spike injects both act within this step.
    float voltage = __fmaf_rn(reset, __fadd_rn(reset_coeff, k.asc_spike_factor[neuron]),
                              __fmaf_rn(decay, old_v, drive));
    // A hard reset puts the membrane at v_reset at the spike time instead.
    if (hard_reset)
      voltage = __fmaf_rn(reset, __fmaf_rn(decay, __fsub_rn(rest, old_v), -reset_coeff), voltage);
    if (hard_reset && refractory > 0) voltage = rest;
    const R counter = static_cast<R>(refractory);
    new_v[i] = voltage; new_r[i] = counter;
    const float next_asc[2] = {
        __fmaf_rn(k.asc_decay[2 * neuron], old_asc[0], __fmul_rn(reset, k.asc_amps[2 * neuron])),
        __fmaf_rn(k.asc_decay[2 * neuron + 1], old_asc[1],
                  __fmul_rn(reset, k.asc_amps[2 * neuron + 1]))};
    StoreRow<float, 2, Vec>(new_asc + 2 * i, next_asc);
    // The threshold reads the membrane still in registers.
    const bool blocked = counter > 0;
    const T spike = static_cast<T>(!blocked && voltage - threshold > 0.0f);
    spikes[i] = spike;
    if (refractory_out) refractory_out[i] = blocked;
    if (emit_voltage) voltage_out[i] = V<T>(voltage);
    if (penalty) penalty_sum[sample - tile_first] += PenaltyValue(penalty, voltage);
    if constexpr (Pack) { fired = F(spike) != 0.0f; blocked_bit = blocked; }
    }
    if constexpr (Pack)
      StoreWarpMasks(pack, sample, neuron - static_cast<int>(threadIdx.x & 31), neurons, fired,
                     blocked_bit);
    }
  }
  if (penalty) StoreBlockPenalty<kTileBatch>(penalty_sum, penalty_partials, tile_first, batch);
}

// max(largest, |value|) over finite values only: the recurrent backward
// skips non-finite gradients when it bounds them (csr_recurrent_ops.cu.cc).
__device__ __forceinline__ float LargerFinite(float largest, float value) {
  value = fabsf(value);
  return isfinite(value) && value > largest ? value : largest;
}

// Warp max, then one integer atomicMax per warp: non-negative floats order
// like their bit patterns. Every lane of the warp must call it.
__device__ __forceinline__ void StoreAbsMax(float largest, unsigned int* result) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1)
    largest = fmaxf(largest, __shfl_xor_sync(0xffffffffu, largest, mask));
  if ((threadIdx.x & 31) == 0 && largest > 0.0f) atomicMax(result, __float_as_uint(largest));
}

// Where the backward reads the previous spikes and the refractory mask. Dense
// (WordBits = 0): the forward's [batch, neurons] T spikes and bool mask.
// Packed: WordBits (31 or 32) binary values per 32-bit word, rows of `words`
// words per sample, value n in bit n % WordBits of word n / WordBits (the
// layout of models._pack_spike_words). Spikes are binary, so the unpacked
// value is exactly F(prev_z).
template <typename T, int WordBits> struct SpikeMasks {
  const void* prev_z; const void* refractory; int words;
  __device__ __forceinline__ bool Bit(const void* bits, int64_t sample, int neuron) const {
    const int word = __ldg(static_cast<const int*>(bits) + sample * words + neuron / WordBits);
    return (word >> (neuron % WordBits)) & 1;
  }
  __device__ __forceinline__ bool Blocked(int64_t i, int64_t sample, int neuron) const {
    if constexpr (WordBits == 0) return static_cast<const bool*>(refractory)[i];
    else return Bit(refractory, sample, neuron);
  }
  __device__ __forceinline__ float PrevZ(int64_t i, int64_t sample, int neuron) const {
    if constexpr (WordBits == 0) return F(static_cast<const T*>(prev_z)[i]);
    else return Bit(prev_z, sample, neuron) ? 1.0f : 0.0f;
  }
};

// One sample per blockIdx.y (grid-strided beyond 65535 samples): unlike the
// forward, the backward measured faster without sharing the neuron constants
// across samples.
template <typename T, int Basis, bool Vec, int WordBits>
__global__ void BackwardKernelTiled(int64_t count, int neurons, int n_basis,
    SpikeMasks<T, WordBits> masks, const float* __restrict__ new_v, Constants k,
    const float* dt, const float* v_th, const float* sigma,
    const float* amplitude, int surrogate, const T* __restrict__ gspikes,
    const T* __restrict__ gnew_z, const T* __restrict__ gprev_z, const float* __restrict__ gv,
    const float* __restrict__ ga, const T* __restrict__ grise, const T* __restrict__ gpsc,
    const T* __restrict__ gvoltage, bool hard_reset, bool detach_reset,
    bool detach_asc_reset, bool emit_voltage, T* __restrict__ prev_z_grad,
    float* __restrict__ v_grad, float* __restrict__ asc_grad, T* __restrict__ rise_grad,
    T* __restrict__ psc_grad, T* __restrict__ input_grad, int penalty,
    const float* grad_penalty, const float* inverse_neurons, unsigned int* abs_max) {
  const float step = *dt; const float threshold = *v_th;
  const float scale = *sigma; const float gain = *amplitude;
  const int64_t batch = count / neurons;
  float largest = 0.0f;
  for (int64_t sample = blockIdx.y; sample < batch; sample += gridDim.y)
  for (int neuron = blockIdx.x * blockDim.x + threadIdx.x;
       neuron < neurons; neuron += blockDim.x * gridDim.x) {
    const NeuronConstants<Basis> kn(k, neuron);
    const int64_t i = sample * neurons + neuron;
    const bool blocked = masks.Blocked(i, sample, neuron);
    // The surrogate of the threshold. Both consumers of a spike read it in T,
    // so the upstream gradient is summed in T before it meets the float32
    // surrogate of the float32 membrane.
    const float membrane = new_v[i];
    const float vs = membrane - threshold;
    float shape;
    if (surrogate == 1) {
      shape = expf(-(vs * vs) / (scale * scale));
    } else if (surrogate == 2) {
      shape = expf(-scale * fabsf(vs));
    } else {
      shape = fmaxf(1.0f - fabsf(vs), 0.0f);
    }
    const T upstream = static_cast<T>(F(gspikes[i]) + F(gnew_z[i]));
    float g_asc[2], g_rise[Basis], g_psc[Basis];
    LoadRow<float, 2, Vec>(ga + 2 * i, g_asc);
    LoadRow<T, Basis, Vec>(grise + i * Basis, g_rise);
    LoadRow<T, Basis, Vec>(gpsc + i * Basis, g_psc);
    // The penalty's membrane gradient joins grad_v first, as AddN does in the
    // unfused graph.
    const float gv_total = penalty
        ? __fadd_rn(gv[i], PenaltyGradient(penalty, membrane, grad_penalty[sample],
                                           *inverse_neurons))
        : gv[i];
    // As in the forward, the contraction nvcc chose for the per-element code
    // is spelled out, so the results stay bit for bit.
    float membrane_g = __fadd_rn(
        blocked ? 0.0f : __fmul_rn(F(upstream), __fmul_rn(shape, gain)), gv_total);
    if (emit_voltage) membrane_g = __fadd_rn(membrane_g, F(gvoltage[i]));
    const float active_gv = hard_reset && blocked ? 0.0f : membrane_g;
    // detach_reset gates the membrane reset; detach_asc_reset gates every path
    // through the ASC this spike injects - both its own state and the drive it
    // contributes to the membrane within this step.
    const float asc_reset_g = detach_asc_reset ? 0.0f :
        __fmaf_rn(g_asc[1], kn.asc_amps[1],
                  __fmaf_rn(active_gv, kn.asc_spike_factor, __fmul_rn(g_asc[0], kn.asc_amps[0])));
    const T z_grad = V<T>(__fadd_rn(
        detach_reset ? 0.0f : __fmul_rn(active_gv, kn.reset_coeff), asc_reset_g));
    // prev_z also moves on as the next step's second history slot, whose
    // gradient joins the reset's here.
    prev_z_grad[i] = gprev_z ? V<T>(__fadd_rn(F(gprev_z[i]), F(z_grad))) : z_grad;
    // The hard reset replaces the pre-spike membrane, so it carries no gradient.
    v_grad[i] = __fmul_rn(__fmul_rn(active_gv, kn.decay),
                          hard_reset ? __fsub_rn(1.0f, masks.PrevZ(i, sample, neuron)) : 1.0f);
    const float next_asc[2] = {
        __fmaf_rn(active_gv, kn.asc_factor[0], __fmul_rn(g_asc[0], kn.asc_decay[0])),
        __fmaf_rn(active_gv, kn.asc_factor[1], __fmul_rn(g_asc[1], kn.asc_decay[1]))};
    StoreRow<float, 2, Vec>(asc_grad + 2 * i, next_asc);
    float to_rise[Basis], to_psc[Basis], to_input[Basis];
    #pragma unroll
    for (int b = 0; b < Basis; ++b) {
      const float4 c = kn.syn[b];
      to_rise[b] = __fmaf_rn(c.w, active_gv, __fmaf_rn(c.x, g_rise[b],
                             __fmul_rn(__fmul_rn(g_psc[b], step), c.x)));
      to_psc[b] = __fmaf_rn(c.z, active_gv, __fmul_rn(c.x, g_psc[b]));
      to_input[b] = __fmul_rn(c.y, g_rise[b]);
    }
    StoreRow<T, Basis, Vec>(rise_grad + i * Basis, to_rise);
    StoreRow<T, Basis, Vec>(psc_grad + i * Basis, to_psc);
    StoreRow<T, Basis, Vec>(input_grad + i * Basis, to_input);
    if (abs_max) {
#pragma unroll
      for (int b = 0; b < Basis; ++b) largest = LargerFinite(largest, F(V<T>(to_input[b])));
    }
  }
  if (abs_max) StoreAbsMax(largest, abs_max);
}

// The largest finite |input_grad| as float bits, as the recurrent backward's
// AbsMaxFiniteKernel computes it (a max is order-independent, so the bound is
// the same bit for bit). For the generic backward, which does not fuse it.
template <typename T>
__global__ void GlifAbsMaxFiniteKernel(int64_t elements, const T* values, unsigned int* result) {
  float largest = 0.0f;
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < elements; index += static_cast<int64_t>(blockDim.x) * gridDim.x)
    largest = LargerFinite(largest, F(values[index]));
  StoreAbsMax(largest, result);
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
    OP_REQUIRES_OK(c, c->GetAttr("emit_refractory", &emit_refractory_));
    OP_REQUIRES_OK(c, c->GetAttr("pack_masks", &pack_masks_));
    OP_REQUIRES_OK(c, c->GetAttr("mask_word_bits", &mask_word_bits_));
    OP_REQUIRES(c, mask_word_bits_ == 31 || mask_word_bits_ == 32,
                errors::InvalidArgument("mask_word_bits must be 31 or 32"));
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
    OP_REQUIRES_OK(c, c->allocate_output(
        6, emit_refractory_ ? v.shape() : TensorShape({0}), &refractory));
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
    PackedMasks pack{nullptr, nullptr, 0, mask_word_bits_};
    const int64_t words = pack_masks_ ? (neurons + mask_word_bits_ - 1) / mask_word_bits_ : 0;
    Tensor *spike_words, *blocked_words;
    for (auto [index, out] : {std::pair{9, &spike_words}, std::pair{10, &blocked_words}})
      OP_REQUIRES_OK(c, c->allocate_output(
          index, pack_masks_ ? TensorShape({v.dim_size(0), words}) : TensorShape({0}), out));
    if (pack_masks_)
      pack = PackedMasks{spike_words->flat<int32>().data(), blocked_words->flat<int32>().data(),
                         static_cast<int>(words), mask_word_bits_};
    const Tensor& bkg_activity = c->input(20);
    Bkg bkg{nullptr, 0, nullptr, nullptr, nullptr, nullptr, nullptr};
    if (bkg_activity.NumElements()) {
      OP_REQUIRES(c, basis == kBasis && bkg_activity.dims() == 2 &&
                     bkg_activity.dim_size(0) == v.dim_size(0) &&
                     c->input(21).NumElements() == 4 * neurons &&
                     c->input(22).NumElements() == 4 * neurons &&
                     c->input(23).NumElements() == 4 * neurons &&
                     c->input(24).NumElements() == 4 * neurons &&
                     c->input(25).dims() == 2 && c->input(25).dim_size(1) == 4,
                  errors::InvalidArgument("the fused BKG gather needs four bases, [batch, n_bkg] "
                                          "activity and four incoming edges (and weights) "
                                          "per neuron"));
      bkg = Bkg{bkg_activity.flat<T>().data(), static_cast<int>(bkg_activity.dim_size(1)),
                c->input(21).flat<float>().data(), c->input(22).flat<uint32>().data(),
                c->input(23).flat<uint32>().data(), c->input(24).flat<uint8>().data(),
                c->input(25).flat<float>().data()};
    }
    auto& d = c->eigen_device<GPUDevice>();
    auto launch_masks = [&](auto tag, auto vec, auto pack_tag) {
      constexpr int B = decltype(tag)::value;
      auto kernel = [] {
        if constexpr (B > 0)
          return ForwardKernelTiled<T, R, B, decltype(vec)::value, decltype(pack_tag)::value>;
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
      // The tiled kernel writes 32-bit words: straight into the outputs, or
      // into a staging buffer that RepackWordsKernel turns into 31-bit words.
      PackedMasks stage = pack;
      Tensor staged;
      if (B > 0 && pack_masks_ && mask_word_bits_ != 32) {
        const int64_t stage_words = (neurons + 31) / 32;
        TF_RETURN_IF_ERROR(c->allocate_temp(
            DT_INT32, TensorShape({2, v.dim_size(0), stage_words}), &staged));
        int* data = staged.flat<int32>().data();
        stage = PackedMasks{data, data + v.dim_size(0) * stage_words,
                            static_cast<int>(stage_words), 32};
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
        if constexpr (B > 0)
          return GpuLaunchKernel(kernel, blocks, threads, 0, d.stream(), tail..., bkg, stage);
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
        opsc->flat<T>().data(),
        emit_refractory_ ? refractory->flat<bool>().data() : nullptr,
        voltage->flat<T>().data(), penalty_, partial_data));
      if (B == 0 && pack_masks_) {
        auto cfg = GetGpuLaunchConfig(v.dim_size(0) * words, d);
        TF_RETURN_IF_ERROR(GpuLaunchKernel(
            PackMasksKernel<T, R>, cfg.block_count, cfg.thread_per_block, 0, d.stream(),
            v.dim_size(0), neurons, spikes->flat<T>().data(), orr->flat<R>().data(), pack));
      }
      if (B > 0 && pack_masks_ && mask_word_bits_ != 32) {
        auto cfg = GetGpuLaunchConfig(v.dim_size(0) * words, d);
        TF_RETURN_IF_ERROR(GpuLaunchKernel(
            RepackWordsKernel, cfg.block_count, cfg.thread_per_block, 0, d.stream(),
            v.dim_size(0), stage, pack));
      }
      if (!penalty_) return absl::OkStatus();
      const int batch = static_cast<int>(v.dim_size(0));
      return GpuLaunchKernel(FinalizePenaltyKernel, dim3(batch), 32, 0,
                             d.stream(), batch, parts, partial_data,
                             penalty_acc.flat<float>().data(), c->input(19).flat<float>().data(),
                             new_penalty_acc->flat<float>().data());
    };
    auto launch = [&](auto tag, auto vec) {
      return pack_masks_ ? launch_masks(tag, vec, std::true_type{})
                         : launch_masks(tag, vec, std::false_type{});
    };
    using Scalar = std::false_type;
    if (basis == 3 && penalty_) {
      OP_REQUIRES_OK(c, launch(std::integral_constant<int, 3>{}, Scalar{}));
    } else if (basis == 5 && penalty_) {
      OP_REQUIRES_OK(c, launch(std::integral_constant<int, 5>{}, Scalar{}));
    } else if (basis == kBasis &&
               RowsAligned<T>(basis, {c->input(4).data(), psc.data(), c->input(6).data(),
                                      orise->data(), opsc->data()},
                              {c->input(3).data(), oa->data()})) {
      OP_REQUIRES_OK(c, launch(std::integral_constant<int, 4>{}, std::true_type{}));
    } else if (basis == kBasis) {
      OP_REQUIRES_OK(c, launch(std::integral_constant<int, 4>{}, Scalar{}));
    } else {
      OP_REQUIRES_OK(c, launch(std::integral_constant<int, 0>{}, Scalar{}));
    }
  }
 private:
  bool hard_; bool emit_voltage_; bool emit_refractory_; bool pack_masks_;
  int mask_word_bits_; int penalty_;
};

// FusedGlifStepBackward (Packed = false) and FusedGlifStepBackwardPacked,
// which reads the previous spikes and the refractory mask as packed 32-bit
// words (`word_bits` per word) instead of the dense T and bool tensors.
template <typename T, bool Packed> class BackwardOp : public OpKernel {
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
    OP_REQUIRES_OK(c, c->GetAttr("emit_abs_max", &emit_abs_max_));
    if constexpr (Packed) {
      OP_REQUIRES_OK(c, c->GetAttr("word_bits", &word_bits_));
      OP_REQUIRES(c, word_bits_ == 31 || word_bits_ == 32,
                  errors::InvalidArgument("word_bits must be 31 or 32"));
    }
  }
  void Compute(OpKernelContext* c) override {
    const Tensor& prev_z = c->input(0); const Tensor& new_v = c->input(1);
    const Tensor& grise = c->input(19);
    int64_t words = 0;
    if constexpr (Packed) {
      OP_REQUIRES(c, new_v.dims() == 2 && new_v.dim_size(1) > 0 &&
                     new_v.dim_size(1) <= std::numeric_limits<int>::max(),
                  errors::InvalidArgument("new_v must be [batch, neurons]"));
      words = (new_v.dim_size(1) + word_bits_ - 1) / word_bits_;
      for (int index : {0, 2})
        OP_REQUIRES(c, c->input(index).dims() == 2 &&
                       c->input(index).dim_size(0) == new_v.dim_size(0) &&
                       c->input(index).dim_size(1) == words,
                    errors::InvalidArgument("packed prev_z and refractory must be [batch, ",
                                            words, "] words"));
    } else {
      OP_REQUIRES_OK(c, ValidateHistory(prev_z, new_v));
    }
    const int neurons = new_v.dim_size(1);
    OP_REQUIRES(c, grise.dims() == 2 && grise.dim_size(1) % neurons == 0,
                errors::InvalidArgument("grad_rise must contain whole synaptic bases"));
    const int64_t basis = grise.dim_size(1) / neurons;
    OP_REQUIRES_OK(c, ValidateState(grise, new_v.dim_size(0), neurons, basis, "grad_rise"));
    OP_REQUIRES_OK(c, ValidateState(c->input(20), new_v.dim_size(0), neurons, basis, "grad_psc"));
    OP_REQUIRES_OK(c, ValidateState(c->input(18), new_v.dim_size(0), neurons, 2, "grad_asc"));
    for (int index : {14, 15, 17})
      OP_REQUIRES_OK(c, ValidateState(c->input(index), new_v.dim_size(0), neurons, 1, "gradient"));
    if constexpr (!Packed)
      OP_REQUIRES_OK(c, ValidateState(c->input(2), new_v.dim_size(0), neurons, 1, "refractory"));
    // An empty grad_prev_z means prev_z has no later history slot.
    const bool has_prev_grad = c->input(16).NumElements() > 0;
    OP_REQUIRES(c, !has_prev_grad || c->input(16).shape() == new_v.shape(),
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
    // Other basis widths take the generic kernel, which reads dense masks.
    const bool tiled = basis == kBasis;
    OP_REQUIRES(c, !Packed || tiled,
                errors::Unimplemented("packed masks need four synaptic bases"));
    Tensor *zg, *vg, *ag, *rg, *pg, *ig;
    OP_REQUIRES_OK(c, c->allocate_output(0, new_v.shape(), &zg));
    OP_REQUIRES_OK(c, c->allocate_output(1, new_v.shape(), &vg));
    OP_REQUIRES_OK(c, c->allocate_output(2, c->input(18).shape(), &ag));
    OP_REQUIRES_OK(c, c->allocate_output(3, grise.shape(), &rg));
    OP_REQUIRES_OK(c, c->allocate_output(4, grise.shape(), &pg));
    OP_REQUIRES_OK(c, c->allocate_output(5, grise.shape(), &ig));
    Tensor* bound;
    OP_REQUIRES_OK(c, c->allocate_output(6, TensorShape({emit_abs_max_ ? 1 : 0}), &bound));
    auto& d = c->eigen_device<GPUDevice>();
    unsigned int* abs_max = emit_abs_max_ ? bound->flat<uint32>().data() : nullptr;
    if (abs_max)
      OP_REQUIRES(c, cudaMemsetAsync(abs_max, 0, sizeof(unsigned int), d.stream()) == cudaSuccess,
                  errors::Internal("abs-max memset failed"));
    // The arguments after the masks, shared by every kernel.
    auto launch_with = [&](auto kernel, dim3 blocks, int threads, auto masks) {
      return GpuLaunchKernel(kernel,
        blocks, threads, 0, d.stream(),
        new_v.NumElements(), neurons, grise.dim_size(1) / neurons, masks,
        new_v.flat<float>().data(), ReadConstants(c, 3),
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
        c->input(22).flat<float>().data(), c->input(23).flat<float>().data(), abs_max);
    };
    auto launch_tiled = [&](auto vec, auto word_bits) {
      constexpr int Bits = decltype(word_bits)::value;
      auto kernel = BackwardKernelTiled<T, kBasis, decltype(vec)::value, Bits>;
      auto cfg = GetGpuLaunchConfigFixedBlockSize(neurons, d, kernel, 0, kTileThreads);
      const int64_t rows = std::min<int64_t>(new_v.dim_size(0), 65535);
      return launch_with(kernel, dim3(cfg.block_count, rows),
                         cfg.thread_per_block,
                         SpikeMasks<T, Bits>{prev_z.data(), c->input(2).data(),
                                             static_cast<int>(words)});
    };
    auto launch_masks = [&](auto vec) {
      if constexpr (Packed) {
        return word_bits_ == 31 ? launch_tiled(vec, std::integral_constant<int, 31>{})
                                : launch_tiled(vec, std::integral_constant<int, 32>{});
      } else {
        return launch_tiled(vec, std::integral_constant<int, 0>{});
      }
    };
    if (!tiled) {
      auto cfg = GetGpuLaunchConfig(new_v.NumElements(), d);
      OP_REQUIRES_OK(c, GpuLaunchKernel(BackwardKernel<T, 0>,
        dim3(cfg.block_count), cfg.thread_per_block, 0, d.stream(),
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
        c->input(22).flat<float>().data(), c->input(23).flat<float>().data()));
      if (abs_max)
        OP_REQUIRES_OK(c, GpuLaunchKernel(GlifAbsMaxFiniteKernel<T>, 1024, 256, 0, d.stream(),
                                          ig->NumElements(), ig->flat<T>().data(), abs_max));
    } else if (RowsAligned<T>(basis, {grise.data(), c->input(20).data(), rg->data(),
                                      pg->data(), ig->data()},
                              {c->input(18).data(), ag->data()})) {
      OP_REQUIRES_OK(c, launch_masks(std::true_type{}));
    } else {
      OP_REQUIRES_OK(c, launch_masks(std::false_type{}));
    }
  }
 private:
  int surrogate_; bool hard_; bool detach_; bool detach_asc_;
  bool emit_voltage_; int penalty_; bool emit_abs_max_; int word_bits_ = 0;
};

#define REG(T,R) REGISTER_KERNEL_BUILDER(Name("FusedGlifStep").Device(DEVICE_GPU).TypeConstraint<T>("T").TypeConstraint<R>("R"),ForwardOp<T,R>);
REG(float,int8); REG(float,int16); REG(Eigen::half,int8); REG(Eigen::half,int16);
#undef REG
#define REG(T) \
  REGISTER_KERNEL_BUILDER(Name("FusedGlifStepBackward").Device(DEVICE_GPU).TypeConstraint<T>("T"),BackwardOp<T, false>); \
  REGISTER_KERNEL_BUILDER(Name("FusedGlifStepBackwardPacked").Device(DEVICE_GPU).TypeConstraint<T>("T").TypeConstraint<int32>("W"),BackwardOp<T, true>); \
  REGISTER_KERNEL_BUILDER(Name("FusedGlifStepBackwardPacked").Device(DEVICE_GPU).TypeConstraint<T>("T").TypeConstraint<uint32>("W"),BackwardOp<T, true>);
REG(float); REG(Eigen::half);
#undef REG
#endif

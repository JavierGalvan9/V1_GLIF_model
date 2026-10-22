#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

// One GLIF timestep: the state update and the threshold, fused so the float32
// membrane is written once and never read back.
//
// The membrane step is coefficient-driven:
//   new_v = decay * v + sum_b (psc_factor_b * psc_b + psc_rise_factor_b * rise_b)
//           + sum_j asc_factor_j * asc_j
//           + reset_coeff * z + asc_spike_factor * z
// The historical Euler scheme is the special case
// psc_factor = asc_factor = current_factor, psc_rise_factor = 0,
// reset_coeff = -1, asc_spike_factor = 0, so the kernel carries no branch for
// the integrator choice. reset_coeff (the reset) and asc_spike_factor (the ASC
// that spike injects, driving V across the same step) stay separate because
// detach_reset and detach_asc_reset gate them separately in the backward pass.
//
// prev_z is the newest slot of the delayed spike history: the spikes of the
// previous step, [batch, neurons]. A neuron spikes when new_v exceeds v_th
// outside refractoriness. The op reads no other history slot and writes none:
// the caller keeps the history as one tensor per delay slot, so shifting it is
// a relabelling of tensors (models.V1Column), and `spikes` becomes its new
// newest slot.
//
// Precision: T is the dtype of the synaptic state (psc_rise, psc), their
// inputs and the spikes; it follows the layer's compute dtype. The membrane
// and ASC state and every constant are float32 whatever T is, because float16
// rounding of the decay factors and of the slowly decaying ASCs is the
// dominant error of a mixed-precision step, while the synaptic state is not.
//
// syn_coeffs packs the four per-(neuron, basis) constants as
// [syn_decay, psc_initial, psc_factor, psc_rise_factor], so the kernels read
// them with a single aligned float4 load.
//
// `refractory` (new_r > 0) exists for the backward pass: it masks the
// surrogate, and the hard reset's voltage gradient, without the refractory
// counter, whose dtype has no GPU TensorList kernel. A caller that never takes
// the gradient (a rollout outside any tape, or one that records the mask
// itself) can set emit_refractory = false: `refractory` is then empty and the
// op writes one byte less per (sample, neuron). With emit_voltage the op
// also writes new_v in T, the exposed voltage sequence; otherwise `voltage` is
// empty.
//
// With pack_masks, spike_words and blocked_words ([batch, ceil(neurons /
// mask_word_bits)] int32) hold the spikes and the refractory mask (new_r > 0)
// bit-packed mask_word_bits (31 or 32) per word, value n in bit n %
// mask_word_bits of word n / mask_word_bits: the words models._pack_spike_words
// builds from `spikes` and `new_r > 0`, bit for bit, and those
// FusedGlifStepBackwardPacked reads. Without it both are empty.
//
// With `penalty` other than 'none' the op also advances the online voltage
// penalty: new_penalty_acc[b] = penalty_acc[b] + inverse_neurons *
// sum_n p(new_v[b, n]), with p(v) = relu(|v - 0.5| - 0.5)^2 ('range') or
// (v - 1)^2 ('threshold'), evaluated on the float32 membrane still in registers.
// The neuron sum is reduced deterministically (per-block fp32 partials, summed
// in a fixed order), so it replaces the separate elementwise and reduction ops
// over new_v. penalty_acc is [batch, 1]; with 'none' it may be empty and is
// passed through.
//
// A non-empty bkg_activity ([batch, n_bkg] counts) adds the background current
// to rec_inputs before the step reads it: each neuron has exactly four incoming
// BKG edges (edges 4n..4n+3 of the incoming order), with pre ids, edge ids into
// bkg_weights and synapse types into the [n_types, 4] bkg_basis. The sum is
// computed exactly as BkgCsrForward computes it and rounded to T once, so the
// step sees the same input bit for bit, without a separate pass over the
// [batch, neurons * 4] current. Empty bkg_* inputs disable it.
REGISTER_OP("FusedGlifStep")
    .Attr("T: {half, float}").Attr("R: {int8, int16}")
    .Attr("hard_reset: bool = false").Attr("emit_voltage: bool = false")
    .Attr("emit_refractory: bool = true")
    .Attr("pack_masks: bool = false").Attr("mask_word_bits: int = 31")
    .Attr("penalty: {'none', 'range', 'threshold'} = 'none'")
    .Input("prev_z: T").Input("v: float").Input("r: R").Input("asc: float")
    .Input("psc_rise: T").Input("psc: T").Input("rec_inputs: T")
    .Input("syn_coeffs: float").Input("asc_decay: float")
    .Input("asc_amps: float").Input("decay: float").Input("asc_factor: float")
    .Input("reset_coeff: float").Input("asc_spike_factor: float")
    .Input("t_ref_steps: R").Input("dt: float").Input("v_reset: float")
    .Input("v_th: float")
    .Input("penalty_acc: float").Input("inverse_neurons: float")
    .Input("bkg_activity: T").Input("bkg_weights: float")
    .Input("bkg_pre_ids: uint32").Input("bkg_edge_ids: uint32")
    .Input("bkg_types: uint8").Input("bkg_basis: float")
    .Output("spikes: T")
    .Output("new_v: float").Output("new_r: R").Output("new_asc: float")
    .Output("new_psc_rise: T").Output("new_psc: T")
    .Output("refractory: bool").Output("voltage: T")
    .Output("new_penalty_acc: float")
    .Output("spike_words: int32").Output("blocked_words: int32")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      bool emit_voltage;
      TF_RETURN_IF_ERROR(c->GetAttr("emit_voltage", &emit_voltage));
      c->set_output(0, c->input(1)); c->set_output(1, c->input(1));
      c->set_output(2, c->input(2)); c->set_output(3, c->input(3));
      c->set_output(4, c->input(4)); c->set_output(5, c->input(5));
      bool emit_refractory;
      TF_RETURN_IF_ERROR(c->GetAttr("emit_refractory", &emit_refractory));
      c->set_output(6, emit_refractory ? c->input(1) : c->Vector(0));
      c->set_output(7, emit_voltage ? c->input(1) : c->Vector(0));
      c->set_output(8, c->input(18));
      bool pack_masks;
      int word_bits;
      TF_RETURN_IF_ERROR(c->GetAttr("pack_masks", &pack_masks));
      TF_RETURN_IF_ERROR(c->GetAttr("mask_word_bits", &word_bits));
      shape_inference::ShapeHandle words = c->Vector(0);
      if (pack_masks) {
        const auto neurons = c->Dim(c->input(1), 1);
        words = c->Matrix(c->Dim(c->input(1), 0),
                          c->ValueKnown(neurons)
                              ? c->MakeDim((c->Value(neurons) + word_bits - 1) / word_bits)
                              : c->UnknownDim());
      }
      c->set_output(9, words); c->set_output(10, words);
      return absl::OkStatus();
    });

// The step is linear in the state, so its Jacobian needs only the previous
// spikes, the membrane the surrogate is evaluated on, and the refractory
// mask; every other output takes its shape from an upstream gradient, which
// keeps the forward state history out of the backward graph. With `penalty`,
// grad_penalty_acc ([batch, 1]) is the gradient of new_penalty_acc, and the
// penalty's membrane gradient is added to grad_v in the same order the
// unfused graph adds them (AddN), so the result is unchanged bit for bit.
//
// The spikes reach two consumers: the exposed spike sequence (grad_spikes) and
// the newest history slot of the next step (grad_new_z). prev_z also passes on
// unchanged as the next step's second slot, so its gradient there
// (grad_prev_z, empty with a single delay slot) joins prev_z_grad here, in the
// order the former shifted history added it.
//
// With emit_abs_max, rec_inputs_grad_abs_max ([1] uint32) holds the float bits
// of the largest finite |rec_inputs_grad| (0 if none), exactly the bound the
// recurrent backward's AbsMaxFiniteKernel measures; otherwise it is empty.
REGISTER_OP("FusedGlifStepBackward")
    .Attr("T: {half, float}")
    .Attr("surrogate: {'triangular', 'gaussian', 'slayer'} = 'triangular'")
    .Attr("hard_reset: bool = false")
    .Attr("detach_reset: bool = true")
    .Attr("detach_asc_reset: bool = false")
    .Attr("emit_voltage: bool = false")
    .Attr("penalty: {'none', 'range', 'threshold'} = 'none'")
    .Attr("emit_abs_max: bool = false")
    .Input("prev_z: T").Input("new_v: float").Input("refractory: bool")
    .Input("syn_coeffs: float").Input("asc_decay: float")
    .Input("asc_amps: float").Input("decay: float").Input("asc_factor: float")
    .Input("reset_coeff: float").Input("asc_spike_factor: float")
    .Input("dt: float").Input("v_th: float")
    .Input("sigma: float").Input("amplitude: float")
    .Input("grad_spikes: T").Input("grad_new_z: T").Input("grad_prev_z: T")
    .Input("grad_v: float").Input("grad_asc: float")
    .Input("grad_psc_rise: T").Input("grad_psc: T").Input("grad_voltage: T")
    .Input("grad_penalty_acc: float").Input("inverse_neurons: float")
    .Output("prev_z_grad: T").Output("v_grad: float").Output("asc_grad: float")
    .Output("psc_rise_grad: T").Output("psc_grad: T").Output("rec_inputs_grad: T")
    .Output("rec_inputs_grad_abs_max: uint32")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0)); c->set_output(1, c->input(1));
      c->set_output(2, c->input(18)); c->set_output(3, c->input(19));
      c->set_output(4, c->input(19)); c->set_output(5, c->input(19));
      bool emit_abs_max;
      TF_RETURN_IF_ERROR(c->GetAttr("emit_abs_max", &emit_abs_max));
      c->set_output(6, c->Vector(emit_abs_max ? 1 : 0));
      return absl::OkStatus();
    });

// FusedGlifStepBackward with the previous spikes and the refractory mask read
// as binary values packed `word_bits` (31 or 32) to a 32-bit word (int32 or
// uint32 W; prefer uint32, which TensorFlow never places in host memory) instead of
// the dense prev_z (T) and refractory (bool) tensors: [batch, words] each, with
// words = ceil(neurons / word_bits) and value n in bit n % word_bits of word
// n / word_bits (models._pack_spike_words). The results are those of
// FusedGlifStepBackward on the unpacked tensors, bit for bit. Four synaptic
// bases only; every other input and output is as in FusedGlifStepBackward,
// and prev_z_grad takes new_v's shape.
REGISTER_OP("FusedGlifStepBackwardPacked")
    .Attr("T: {half, float}")
    .Attr("surrogate: {'triangular', 'gaussian', 'slayer'} = 'triangular'")
    .Attr("hard_reset: bool = false")
    .Attr("detach_reset: bool = true")
    .Attr("detach_asc_reset: bool = false")
    .Attr("emit_voltage: bool = false")
    .Attr("penalty: {'none', 'range', 'threshold'} = 'none'")
    .Attr("emit_abs_max: bool = false")
    .Attr("word_bits: int = 31").Attr("W: {int32, uint32} = DT_INT32")
    .Input("prev_z_bits: W").Input("new_v: float").Input("refractory_bits: W")
    .Input("syn_coeffs: float").Input("asc_decay: float")
    .Input("asc_amps: float").Input("decay: float").Input("asc_factor: float")
    .Input("reset_coeff: float").Input("asc_spike_factor: float")
    .Input("dt: float").Input("v_th: float")
    .Input("sigma: float").Input("amplitude: float")
    .Input("grad_spikes: T").Input("grad_new_z: T").Input("grad_prev_z: T")
    .Input("grad_v: float").Input("grad_asc: float")
    .Input("grad_psc_rise: T").Input("grad_psc: T").Input("grad_voltage: T")
    .Input("grad_penalty_acc: float").Input("inverse_neurons: float")
    .Output("prev_z_grad: T").Output("v_grad: float").Output("asc_grad: float")
    .Output("psc_rise_grad: T").Output("psc_grad: T").Output("rec_inputs_grad: T")
    .Output("rec_inputs_grad_abs_max: uint32")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(1)); c->set_output(1, c->input(1));
      c->set_output(2, c->input(18)); c->set_output(3, c->input(19));
      c->set_output(4, c->input(19)); c->set_output(5, c->input(19));
      bool emit_abs_max;
      TF_RETURN_IF_ERROR(c->GetAttr("emit_abs_max", &emit_abs_max));
      c->set_output(6, c->Vector(emit_abs_max ? 1 : 0));
      return absl::OkStatus();
    });

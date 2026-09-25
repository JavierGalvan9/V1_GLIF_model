#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

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
// Precision: T is the dtype of the synaptic state (psc_rise, psc), their
// inputs and the spikes; it follows the layer's compute dtype. The membrane
// and ASC state and every constant are float32 whatever T is, because float16
// rounding of the decay factors and of the slowly decaying ASCs is the
// dominant error of a mixed-precision step, while the synaptic state is not.
//
// syn_coeffs packs the four per-(neuron, basis) constants as
// [syn_decay, psc_initial, psc_factor, psc_rise_factor], so the kernels read
// them with a single aligned float4 load.
REGISTER_OP("FusedGlifSingleForward")
    .Attr("T: {half, float}").Attr("R: {int8, int16}")
    .Attr("hard_reset: bool = false")
    .Input("prev_z: T").Input("v: float").Input("r: R").Input("asc: float")
    .Input("psc_rise: T").Input("psc: T").Input("rec_inputs: T")
    .Input("syn_coeffs: float").Input("asc_decay: float")
    .Input("asc_amps: float").Input("decay: float").Input("asc_factor: float")
    .Input("reset_coeff: float").Input("asc_spike_factor: float")
    .Input("t_ref_steps: R").Input("dt: float").Input("v_reset: float")
    .Output("new_v: float").Output("new_r: R").Output("new_asc: float")
    .Output("new_psc_rise: T").Output("new_psc: T")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(1)); c->set_output(1, c->input(2));
      c->set_output(2, c->input(3)); c->set_output(3, c->input(4));
      c->set_output(4, c->input(5)); return absl::OkStatus();
    });

// The step is linear in the state, so its Jacobian needs only the spikes and
// the refractory counter; every output takes its shape from an upstream
// gradient, which keeps the forward state history out of the backward graph.
REGISTER_OP("FusedGlifSingleBackward")
    .Attr("T: {half, float}").Attr("R: {int8, int16}")
    .Attr("hard_reset: bool = false")
    .Attr("detach_reset: bool = true")
    .Attr("detach_asc_reset: bool = false")
    .Input("prev_z: T").Input("r: R")
    .Input("syn_coeffs: float").Input("asc_decay: float")
    .Input("asc_amps: float").Input("decay: float").Input("asc_factor: float")
    .Input("reset_coeff: float").Input("asc_spike_factor: float")
    .Input("t_ref_steps: R").Input("dt: float")
    .Input("grad_v: float").Input("grad_asc: float")
    .Input("grad_psc_rise: T").Input("grad_psc: T")
    .Output("prev_z_grad: T").Output("v_grad: float").Output("asc_grad: float")
    .Output("psc_rise_grad: T").Output("psc_grad: T").Output("rec_inputs_grad: T")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0)); c->set_output(1, c->input(11));
      c->set_output(2, c->input(12)); c->set_output(3, c->input(13));
      c->set_output(4, c->input(13)); c->set_output(5, c->input(13));
      return absl::OkStatus();
    });

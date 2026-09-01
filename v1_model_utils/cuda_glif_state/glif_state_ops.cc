#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("FusedGlifSingleForward")
    .Attr("T: {half, float}").Attr("R: {int8, int16}")
    .Attr("hard_reset: bool = false")
    .Input("prev_z: T").Input("v: T").Input("r: R").Input("asc: T")
    .Input("psc_rise: T").Input("psc: T").Input("rec_inputs: T")
    .Input("syn_decay: T").Input("psc_initial: T").Input("asc_decay: T")
    .Input("asc_amps: T").Input("decay: T").Input("current_factor: T")
    .Input("t_ref_steps: R").Input("dt: T").Input("v_reset: T")
    .Output("new_v: T").Output("new_r: R").Output("new_asc: T")
    .Output("new_psc_rise: T").Output("new_psc: T")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(1)); c->set_output(1, c->input(2));
      c->set_output(2, c->input(3)); c->set_output(3, c->input(4));
      c->set_output(4, c->input(5)); return OkStatus();
    });

REGISTER_OP("FusedGlifSingleBackward")
    .Attr("T: {half, float}").Attr("R: {int8, int16}")
    .Attr("hard_reset: bool = false")
    .Attr("detach_reset: bool = true")
    .Attr("detach_asc_reset: bool = false")
    .Input("prev_z: T").Input("r: R").Input("asc: T").Input("psc_rise: T")
    .Input("syn_decay: T").Input("psc_initial: T").Input("asc_decay: T")
    .Input("decay: T").Input("current_factor: T").Input("t_ref_steps: R")
    .Input("asc_amps: T").Input("dt: T").Input("grad_v: T")
    .Input("grad_asc: T").Input("grad_psc_rise: T").Input("grad_psc: T")
    .Output("prev_z_grad: T").Output("v_grad: T").Output("asc_grad: T")
    .Output("psc_rise_grad: T").Output("psc_grad: T").Output("rec_inputs_grad: T")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0)); c->set_output(1, c->input(12));
      c->set_output(2, c->input(2)); c->set_output(3, c->input(3));
      c->set_output(4, c->input(3)); c->set_output(5, c->input(3));
      return OkStatus();
    });

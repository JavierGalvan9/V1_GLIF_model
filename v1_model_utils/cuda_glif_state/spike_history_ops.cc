#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("V1FusedSpikeShift")
    .Attr("T: {half, float}")
    .Input("voltage: T")
    .Input("refractory: bool")
    .Input("history: T")
    .Output("spikes: T")
    .Output("new_history: T")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle voltage, refractory, history;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &voltage));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &refractory));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &history));
      TF_RETURN_IF_ERROR(c->Merge(voltage, refractory, &voltage));
      c->set_output(0, voltage);
      c->set_output(1, history);
      return OkStatus();
    });

REGISTER_OP("V1FusedSpikeShiftBackward")
    .Attr("T: {half, float}")
    .Attr("surrogate: {'triangular', 'gaussian', 'slayer'} = 'triangular'")
    .Input("voltage: T")
    .Input("refractory: bool")
    .Input("spike_grad: T")
    .Input("history_grad: T")
    .Input("sigma: T")
    .Input("amplitude: T")
    .Output("voltage_grad: T")
    .Output("old_history_grad: T")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle voltage, refractory, spike_grad, history_grad,
          sigma, amplitude;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &voltage));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &refractory));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &spike_grad));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 2, &history_grad));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &sigma));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 0, &amplitude));
      TF_RETURN_IF_ERROR(c->Merge(voltage, refractory, &voltage));
      TF_RETURN_IF_ERROR(c->Merge(voltage, spike_grad, &voltage));
      c->set_output(0, voltage);
      c->set_output(1, history_grad);
      return OkStatus();
    });

#if GOOGLE_CUDA
void RegisterFusedSpikeShiftGpuKernels();
#endif

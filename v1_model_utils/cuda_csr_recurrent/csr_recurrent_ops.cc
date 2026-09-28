#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

// `initial` lets one current source accumulate on top of another's output
// instead of producing a separate tensor that a later add has to combine. Pass
// an empty tensor to start from zero. The active (batch, row) slots are found on
// the device, so the forward never waits on a host round trip. The synaptic
// basis is FP32 in every op: it is 360 constants, and rounding them to the
// compute dtype would bias every contribution.
REGISTER_OP("V1CsrForward")
    .Attr("T: {half, float}")
    .Attr("n_post: int >= 1")
    // Sum edges of a row that share a target inside a warp before the atomic.
    // Only connectivities whose rows repeat a target (LGN) benefit.
    .Attr("aggregate_runs: bool = true")
    .Input("spikes: T")
    .Input("weights: float")
    .Input("post_ids: uint32")
    .Input("synapse_types: uint8")
    .Input("row_splits: uint32")
    .Input("edge_ids: uint32")
    .Input("basis: float")
    .Input("initial: T")
    .Output("currents: T")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle spikes;
      shape_inference::ShapeHandle basis;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &spikes));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(6), 2, &basis));
      int n_post;
      TF_RETURN_IF_ERROR(c->GetAttr("n_post", &n_post));
      shape_inference::DimensionHandle rows;
      TF_RETURN_IF_ERROR(c->Multiply(c->Dim(spikes, 0), n_post, &rows));
      c->set_output(0, c->Matrix(rows, c->Dim(basis, 1)));
      return OkStatus();
    });

// Spike and weight gradients for any batch size and basis dimension. Each
// distinct (postsynaptic neuron, synapse type) pair is projected onto the basis
// once instead of once per edge, and the weight gradient is computed only for
// rows with an active sample.
REGISTER_OP("V1CsrBackwardPairProjected")
    .Attr("T: {half, float}")
    .Attr("n_post: int >= 1")
    .Attr("n_edges: int >= 0")
    .Input("spikes: T")
    .Input("current_grad: T")
    .Input("weights: float")
    .Input("post_ids: uint32")
    .Input("synapse_types: uint8")
    .Input("row_splits: uint32")
    .Input("edge_ids: uint32")
    .Input("nonempty_rows: uint32")
    .Input("basis: float")
    .Input("dampening: T")
    .Input("pair_ids: uint32")
    .Input("pair_posts: uint32")
    .Input("pair_types: uint8")
    .Output("spike_grad: T")
    .Output("weight_grad: float")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle spikes;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &spikes));
      int n_edges;
      TF_RETURN_IF_ERROR(c->GetAttr("n_edges", &n_edges));
      c->set_output(0, spikes);
      c->set_output(1, c->Vector(n_edges));
      return OkStatus();
    });

// V1CsrBackwardPairProjected, but the weight gradient is added in place into
// the FP32 [n_edges] resource variable `accumulator` instead of being returned.
// A training graph built on it never carries a dense per-step weight gradient
// for the loop gradient to sum. Stateful, since it mutates the accumulator: the
// caller zeroes the variable before the backward and reads it afterwards.
REGISTER_OP("V1CsrBackwardPairProjectedAccumulate")
    .Attr("T: {half, float}")
    .Attr("n_post: int >= 1")
    .Attr("n_edges: int >= 0")
    .Input("spikes: T")
    .Input("current_grad: T")
    .Input("weights: float")
    .Input("post_ids: uint32")
    .Input("synapse_types: uint8")
    .Input("row_splits: uint32")
    .Input("edge_ids: uint32")
    .Input("nonempty_rows: uint32")
    .Input("basis: float")
    .Input("dampening: T")
    .Input("pair_ids: uint32")
    .Input("pair_posts: uint32")
    .Input("pair_types: uint8")
    .Input("accumulator: resource")
    .Output("spike_grad: T")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle spikes;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &spikes));
      c->set_output(0, spikes);
      return OkStatus();
    });

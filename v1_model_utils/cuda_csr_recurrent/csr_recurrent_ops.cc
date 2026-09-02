#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

// `initial` lets one current source accumulate on top of another's output
// instead of producing a separate tensor that a later add has to combine. Pass
// an empty tensor to start from zero.
REGISTER_OP("V1CsrForward")
    .Attr("T: {half, float}")
    .Attr("n_post: int >= 1")
    .Input("spikes: T")
    .Input("active_indices: int64")
    .Input("weights: float")
    .Input("post_ids: uint32")
    .Input("synapse_types: uint8")
    .Input("row_splits: uint32")
    .Input("edge_ids: uint32")
    .Input("basis: T")
    .Input("initial: T")
    .Output("currents: T")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle spikes;
      shape_inference::ShapeHandle basis;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &spikes));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(7), 2, &basis));
      int n_post;
      TF_RETURN_IF_ERROR(c->GetAttr("n_post", &n_post));
      shape_inference::DimensionHandle rows;
      TF_RETURN_IF_ERROR(c->Multiply(c->Dim(spikes, 0), n_post, &rows));
      c->set_output(0, c->Matrix(rows, c->Dim(basis, 1)));
      return OkStatus();
    });

REGISTER_OP("V1CsrBackward")
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
    .Input("basis: T")
    .Input("dampening: T")
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

// Batch-32 four-basis specialization. Each distinct (postsynaptic neuron,
// synapse type) pair is projected onto the basis once instead of once per edge,
// and the edge loop is mapped one batch sample per lane.
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
    .Input("basis: T")
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

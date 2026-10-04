#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("InitializeV1CsrResource")
    .Attr("resource_name: string")
    .Input("post_ids: uint32")
    .Input("synapse_types: uint8")
    .Input("row_splits: uint32")
    .Input("edge_ids: uint32")
    .Input("nonempty_rows: uint32")
    .Input("pair_ids: uint32")
    .Input("pair_posts: uint32")
    .Input("pair_types: uint8")
    // The fixed-four incoming CSR of the BKG forward gather, or four empty
    // tensors for a connectivity without it.
    .Input("incoming_row_splits: uint32")
    .Input("incoming_pre_ids: uint32")
    .Input("incoming_edge_ids: uint32")
    .Input("incoming_types: uint8")
    .Output("initialized: bool")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* context) {
      context->set_output(0, context->Scalar());
      return OkStatus();
    });

// `spikes` as in V1CsrForward: N equal tensors side by side (one per delay
// slot of the recurrent spike history), with one spike gradient per tensor.
// `queues` and `queue`: see V1CsrForward.
REGISTER_OP("V1CsrForwardResource")
    .Attr("T: {half, float}")
    .Attr("N: int >= 1")
    .Attr("carried: int >= 0 = 0")
    .Attr("n_post: int >= 1")
    .Attr("resource_name: string")
    .Attr("aggregate_runs: bool = true")
    .Input("spikes: N * T")
    .Input("weights: float")
    .Input("basis: float")
    // Currents to accumulate on top of, or an empty tensor to start from zero.
    .Input("initial: T")
    .Input("queues: carried * uint32")
    .Output("currents: T")
    .Output("queue: uint32")
    .SetShapeFn([](shape_inference::InferenceContext* context) {
      context->set_output(1, context->Vector(context->UnknownDim()));
      int slots;
      TF_RETURN_IF_ERROR(context->GetAttr("N", &slots));
      shape_inference::ShapeHandle spikes;
      shape_inference::ShapeHandle basis;
      TF_RETURN_IF_ERROR(context->WithRank(context->input(0), 2, &spikes));
      TF_RETURN_IF_ERROR(context->WithRank(context->input(slots + 1), 2, &basis));
      int n_post;
      TF_RETURN_IF_ERROR(context->GetAttr("n_post", &n_post));
      shape_inference::DimensionHandle rows;
      TF_RETURN_IF_ERROR(
          context->Multiply(context->Dim(spikes, 0), n_post, &rows));
      context->set_output(0, context->Matrix(rows, context->Dim(basis, 1)));
      return OkStatus();
    });

// BkgCsrForward reading its incoming CSR from the resource: one thread per
// (post, batch) sums the four incoming edges, with no atomics.
REGISTER_OP("BkgCsrForwardResource")
    .Attr("T: {half, float}")
    .Attr("n_post: int >= 1")
    .Attr("resource_name: string")
    .Input("activity: T")
    .Input("weights: float")
    .Input("basis: float")
    .Input("initial: T")
    .Output("currents: T")
    .SetShapeFn([](shape_inference::InferenceContext* context) {
      shape_inference::ShapeHandle activity;
      TF_RETURN_IF_ERROR(context->WithRank(context->input(0), 2, &activity));
      int n_post;
      TF_RETURN_IF_ERROR(context->GetAttr("n_post", &n_post));
      shape_inference::DimensionHandle rows;
      TF_RETURN_IF_ERROR(
          context->Multiply(context->Dim(activity, 0), n_post, &rows));
      context->set_output(0, context->Matrix(rows, 4));
      return OkStatus();
    });

REGISTER_OP("V1CsrBackwardResource")
    .Attr("T: {half, float}")
    .Attr("N: int >= 1")
    .Attr("n_post: int >= 1")
    .Attr("n_edges: int >= 0")
    .Attr("resource_name: string")
    .Input("spikes: N * T")
    .Input("current_grad: T")
    .Input("weights: float")
    .Input("basis: float")
    .Input("dampening: T")
    .Output("spike_grad: N * T")
    .Output("weight_grad: float")
    .SetShapeFn([](shape_inference::InferenceContext* context) {
      int slots;
      TF_RETURN_IF_ERROR(context->GetAttr("N", &slots));
      for (int slot = 0; slot < slots; ++slot) {
        shape_inference::ShapeHandle spikes;
        TF_RETURN_IF_ERROR(context->WithRank(context->input(slot), 2, &spikes));
        context->set_output(slot, spikes);
      }
      int n_edges;
      TF_RETURN_IF_ERROR(context->GetAttr("n_edges", &n_edges));
      context->set_output(slots, context->Vector(n_edges));
      return OkStatus();
    });

// V1CsrBackwardResource, adding the weight gradient in place into the FP32
// [n_edges] resource variable `accumulator` instead of returning it. See
// V1CsrBackwardPairProjectedAccumulate.
REGISTER_OP("V1CsrBackwardAccumulateResource")
    .Attr("T: {half, float}")
    .Attr("N: int >= 1")
    .Attr("n_post: int >= 1")
    .Attr("n_edges: int >= 0")
    .Attr("resource_name: string")
    .Input("spikes: N * T")
    .Input("current_grad: T")
    .Input("weights: float")
    .Input("basis: float")
    .Input("dampening: T")
    .Input("accumulator: resource")
    .Output("spike_grad: N * T")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* context) {
      int slots;
      TF_RETURN_IF_ERROR(context->GetAttr("N", &slots));
      for (int slot = 0; slot < slots; ++slot) {
        shape_inference::ShapeHandle spikes;
        TF_RETURN_IF_ERROR(context->WithRank(context->input(slot), 2, &spikes));
        context->set_output(slot, spikes);
      }
      return OkStatus();
    });

REGISTER_OP("ExternalCsrWeightBackwardResource")
    .Attr("T: {half, float}")
    .Attr("n_post: int >= 1")
    .Attr("n_edges: int >= 0")
    .Attr("resource_name: string")
    // Event-driven for sparse activity (LGN), dense pair-projected otherwise.
    .Attr("sparse_activity: bool = true")
    .Input("activity: T")
    .Input("current_grad: T")
    .Input("basis: float")
    .Output("weight_grad: float")
    .SetShapeFn([](shape_inference::InferenceContext* context) {
      shape_inference::ShapeHandle activity;
      shape_inference::ShapeHandle basis;
      TF_RETURN_IF_ERROR(context->WithRank(context->input(0), 2, &activity));
      TF_RETURN_IF_ERROR(context->WithRank(context->input(2), 2, &basis));
      int n_edges;
      TF_RETURN_IF_ERROR(context->GetAttr("n_edges", &n_edges));
      context->set_output(0, context->Vector(n_edges));
      return OkStatus();
    });

REGISTER_OP("ExternalCsrActivityBackwardResource")
    .Attr("T: {half, float}")
    .Attr("n_post: int >= 1")
    .Attr("resource_name: string")
    .Input("current_grad: T")
    .Input("weights: float")
    .Input("basis: float")
    .Output("activity_grad: T")
    .SetShapeFn([](shape_inference::InferenceContext* context) {
      shape_inference::ShapeHandle current_grad;
      TF_RETURN_IF_ERROR(
          context->WithRank(context->input(0), 2, &current_grad));
      int n_post;
      TF_RETURN_IF_ERROR(context->GetAttr("n_post", &n_post));
      shape_inference::DimensionHandle batch;
      TF_RETURN_IF_ERROR(context->Divide(
          context->Dim(current_grad, 0), n_post, true, &batch));
      context->set_output(0, context->Matrix(batch, context->UnknownDim()));
      return OkStatus();
    });

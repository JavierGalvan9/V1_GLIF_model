#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("ExternalCsrWeightBackward")
    .Attr("T: {half, float}")
    .Attr("n_post: int >= 1")
    .Attr("n_edges: int >= 0")
    .Input("activity: T")
    .Input("current_grad: T")
    .Input("post_ids: uint32")
    .Input("synapse_types: uint8")
    .Input("row_splits: uint32")
    .Input("edge_ids: uint32")
    .Input("nonempty_rows: uint32")
    .Input("basis: T")
    .Input("pair_ids: uint32")
    .Input("pair_posts: uint32")
    .Input("pair_types: uint8")
    .Output("weight_grad: float")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle activity;
      shape_inference::ShapeHandle basis;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &activity));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(7), 2, &basis));
      int n_edges;
      TF_RETURN_IF_ERROR(c->GetAttr("n_edges", &n_edges));
      c->set_output(0, c->Vector(n_edges));
      return OkStatus();
    });

REGISTER_OP("ExternalCsrActivityBackward")
    .Attr("T: {half, float}")
    .Attr("n_post: int >= 1")
    .Input("current_grad: T")
    .Input("weights: float")
    .Input("post_ids: uint32")
    .Input("synapse_types: uint8")
    .Input("row_splits: uint32")
    .Input("edge_ids: uint32")
    .Input("nonempty_rows: uint32")
    .Input("basis: T")
    .Input("pair_ids: uint32")
    .Input("pair_posts: uint32")
    .Input("pair_types: uint8")
    .Output("activity_grad: T")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle current_grad;
      shape_inference::ShapeHandle row_splits;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &current_grad));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 1, &row_splits));
      int n_post;
      TF_RETURN_IF_ERROR(c->GetAttr("n_post", &n_post));
      shape_inference::DimensionHandle batch;
      TF_RETURN_IF_ERROR(c->Divide(c->Dim(current_grad, 0), n_post,
                                  true, &batch));
      shape_inference::DimensionHandle n_pre;
      TF_RETURN_IF_ERROR(c->Subtract(c->Dim(row_splits, 0), 1, &n_pre));
      c->set_output(0, c->Matrix(batch, n_pre));
      return OkStatus();
    });

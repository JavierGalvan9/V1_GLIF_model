#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

// `initial` lets one current source accumulate on top of another's output
// instead of producing a separate tensor that a later add has to combine. Pass
// an empty tensor to start from zero. The active (batch, row) slots are found on
// the device, so the forward never waits on a host round trip. The synaptic
// basis is FP32 in every op: it is 360 constants, and rounding them to the
// compute dtype would bias every contribution.
//
// `spikes` is the [batch, n_pre] presynaptic matrix as N equal [batch, n_pre / N]
// tensors side by side: the recurrent spike history passes one tensor per delay
// slot, so the history never has to be concatenated; every other caller passes
// one tensor. The backward ops return one spike gradient per tensor.
//
// `queues` carries the spike history's per-slot queue records across steps: it
// is empty, or holds the records of slots 1 to N - 1, which are the previous
// step's records of slots 0 to N - 2. Output `queue` is slot 0's record with
// several slots (empty with one), for the next step to carry. The records only
// save the sweeps over the slots they cover; the currents are the same.
REGISTER_OP("V1CsrForward")
    .Attr("T: {half, float}")
    .Attr("N: int >= 1")
    .Attr("carried: int >= 0 = 0")
    .Attr("n_post: int >= 1")
    // Sum edges of a row that share a target inside a warp before the atomic.
    // Only connectivities whose rows repeat a target (LGN) benefit.
    .Attr("aggregate_runs: bool = true")
    .Input("spikes: N * T")
    .Input("weights: float")
    .Input("post_ids: uint32")
    .Input("synapse_types: uint8")
    .Input("row_splits: uint32")
    .Input("edge_ids: uint32")
    .Input("basis: float")
    .Input("initial: T")
    .Input("queues: carried * uint32")
    .Output("currents: T")
    .Output("queue: uint32")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(1, c->Vector(c->UnknownDim()));
      int slots;
      TF_RETURN_IF_ERROR(c->GetAttr("N", &slots));
      shape_inference::ShapeHandle spikes;
      shape_inference::ShapeHandle basis;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &spikes));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(slots + 5), 2, &basis));
      int n_post;
      TF_RETURN_IF_ERROR(c->GetAttr("n_post", &n_post));
      shape_inference::DimensionHandle rows;
      TF_RETURN_IF_ERROR(c->Multiply(c->Dim(spikes, 0), n_post, &rows));
      c->set_output(0, c->Matrix(rows, c->Dim(basis, 1)));
      return OkStatus();
    });

// One [batch, width] spike slot's queue record, as V1CsrForward's `queue`
// output holds it: the start of a carried history.
REGISTER_OP("V1SpikeQueue")
    .Attr("T: {half, float}")
    .Input("spikes: T")
    .Output("queue: uint32")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Vector(c->UnknownDim()));
      return OkStatus();
    });

// Spike and weight gradients for any batch size and basis dimension. Each
// distinct (postsynaptic neuron, synapse type) pair is projected onto the basis
// once instead of once per edge, and the weight gradient is computed only for
// rows with an active sample.
//
// With several spike slots, the gradient of the last one (the oldest delay slot
// of the spike history) is returned as +0 + g, so a -0 (a tiny negative rounded
// to zero) comes back as +0. That is what the history gradient was when the
// history was one tensor shifted by the GLIF op: the shift gave its oldest slot
// a zero gradient, which TensorFlow added to this one. The result is therefore
// bitwise that of the shifted history, sign of zero included.
REGISTER_OP("V1CsrBackwardPairProjected")
    .Attr("T: {half, float}")
    .Attr("N: int >= 1")
    .Attr("n_post: int >= 1")
    .Attr("n_edges: int >= 0")
    .Input("spikes: N * T")
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
    .Output("spike_grad: N * T")
    .Output("weight_grad: float")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      int slots;
      TF_RETURN_IF_ERROR(c->GetAttr("N", &slots));
      for (int slot = 0; slot < slots; ++slot) {
        shape_inference::ShapeHandle spikes;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(slot), 2, &spikes));
        c->set_output(slot, spikes);
      }
      int n_edges;
      TF_RETURN_IF_ERROR(c->GetAttr("n_edges", &n_edges));
      c->set_output(slots, c->Vector(n_edges));
      return OkStatus();
    });

// V1CsrBackwardPairProjected, but the weight gradient is added in place into
// the FP32 [n_edges] resource variable `accumulator` instead of being returned.
// A training graph built on it never carries a dense per-step weight gradient
// for the loop gradient to sum. Stateful, since it mutates the accumulator: the
// caller zeroes the variable before the backward and reads it afterwards.
REGISTER_OP("V1CsrBackwardPairProjectedAccumulate")
    .Attr("T: {half, float}")
    .Attr("N: int >= 1")
    .Attr("n_post: int >= 1")
    .Attr("n_edges: int >= 0")
    .Input("spikes: N * T")
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
    .Output("spike_grad: N * T")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      int slots;
      TF_RETURN_IF_ERROR(c->GetAttr("N", &slots));
      for (int slot = 0; slot < slots; ++slot) {
        shape_inference::ShapeHandle spikes;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(slot), 2, &spikes));
        c->set_output(slot, spikes);
      }
      return OkStatus();
    });

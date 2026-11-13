#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

// A frame store parks a sequence of equally shaped GPU frames (index 0 ..
// capacity - 1) in pinned host memory and hands them back on the GPU, with the
// copies on two copy streams of its own and every ordering against the
// compute stream made with CUDA events (no host synchronization).
// One store per (GPU, key) lives for the process; the host buffer is
// allocated by the first offload and reused by every later rollout.
// Both ops take an unread `order` resource, one per replica: TensorFlow's
// automatic control dependencies then order a replica's store ops among
// themselves only. Stateful ops without a resource input join one global
// chain, which would hold each replica's rollout behind the previous one's.
// Copies run in groups of `group` frames: one compute-stream event starts a
// group and one copy event ends it, so the compute stream sees one event
// record (and, coming back, one wait) per group rather than per frame.
//
// V1FrameStoreOffload hands frame `index` to the store. Frames at or above
// `resident_from` stay on the GPU (the reverse pass needs them first, so
// copying them out and straight back would only cost bandwidth). The others
// are queued for copying out once `group` of them are waiting (or at frame
// capacity - 1, the forward's last, or when the first frame is taken), and
// each is released on the GPU once its copy has finished. Queued groups enter
// the copy stream, earliest first, only while fewer than two copies are
// unfinished; those still queued when the GPU passes frame capacity - 1 stay
// on the GPU (so the number of frames offloaded adapts to the bandwidth the
// copy engine gets, e.g. when several GPUs share one PCIe root port).
// With `codec` "tq", float32 frames go out raw (exact) while the link keeps
// up and encoded with TQ-pack (tq_codec.cu.h; lossy, on the grid 2^-16:
// absolute error <= 2^-17 per finite value) as far as needed for the
// forward's frames to fit through the link by the forward's end (the store
// projects this from its measured link and frame rates). TQ-pack predicts
// frame s from frames s + 1 and s + 2, so it encodes a frame once those are
// recorded and decodes in reverse order. An encoded frame is copied out
// straight from its record in a ring arena on the GPU and the raw frame is
// released once encoded; one that stays on the GPU is packed to its encoded
// length and decoded when taken. Other dtypes, and "raw", store the exact
// bits. The forward never waits for the copies or the encoder: a frame the
// ring cannot take stays raw.
// With `drop_outside`, an index outside [0, capacity) records nothing, so a
// loop can hand over frames before the record's start or past its end without
// a conditional (whose host predicate TensorFlow copies to the GPU, and waits
// for, every step).
// Returns an empty token (a data edge for ordering).
REGISTER_OP("V1FrameStoreOffload")
    .Attr("T: {float, half, int32}")
    .Attr("key: string")
    .Attr("capacity: int >= 1")
    .Attr("resident_from: int >= 0")
    .Attr("group: int >= 1 = 1")
    .Attr("codec: {'raw', 'tq'} = 'raw'")
    .Attr("drop_outside: bool = false")
    .Input("frame: T")
    .Input("index: int32")
    .Input("order: resource")
    .Output("token: float")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Vector(0));
      return absl::OkStatus();
    });

// V1FrameStoreTake returns frame `index` on the GPU (with `prefetch` > 0, it
// then starts bringing back the frames index - 1 .. index - prefetch that are
// only on the host, once `group` of them, or the lowest one, need it). A frame that is still on the
// GPU (kept resident, never admitted to the copy stream, or its copy out not
// finished when it was first fetched) is returned as the recorded tensor
// itself, and such a taken frame leaves the store. Otherwise the compute stream
// waits for its copy's event (once per group), not the host. A taken frame
// leaves the store's GPU bookkeeping; its host copy stays valid until the
// slot is offloaded again, so a second reverse pass can take it again. A TQ
// frame decodes from the reconstructions of the frames after it: those a
// take handed out are kept while a stored frame below still needs them (until
// the next rollout for frames not on the host); others are rebuilt from the
// stored chain. With `zero_before_start`, a negative index returns a zero
// frame (without a conditional, as above) and leaves the store untouched.
REGISTER_OP("V1FrameStoreTake")
    .Attr("T: {float, half, int32}")
    .Attr("key: string")
    .Attr("shape: shape")
    .Attr("prefetch: int >= 0 = 0")
    .Attr("group: int >= 1 = 1")
    .Attr("zero_before_start: bool = false")
    .Input("index: int32")
    .Input("order: resource")
    .Output("frame: T")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      PartialTensorShape shape;
      TF_RETURN_IF_ERROR(c->GetAttr("shape", &shape));
      shape_inference::ShapeHandle output;
      TF_RETURN_IF_ERROR(c->MakeShapeFromPartialTensorShape(shape, &output));
      c->set_output(0, output);
      return absl::OkStatus();
    });

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

// One timestep's background Poisson counts, [batch, n_bkg], computed on the
// device with no host copy. It is bitwise identical to models.sample_poisson_counts
// with the seed [int32(noise_seed) + replica_id * 1000003, int32(step[0])]:
//   - the Philox key and counter come from that int32 seed exactly as
//     StatelessRandomGetKeyCounter scrambles it;
//   - the uniforms are StatelessRandomUniformV2's float64 ones: element e is
//     Uint64ToDouble of words (2j, 2j+1) of Philox(counter + e / 2), j = e % 2;
//   - the count is searchsorted(cdf, u, side='right'), cast to T.
// noise_seed stays in device memory and is read by the kernel; replica_id, step
// (the int32 noise_step loop state; element 0 is used) and shape are host
// memory, so nothing crosses PCIe per call.
REGISTER_OP("BkgPoissonCounts")
    .Attr("T: {half, float, int32}")
    .Input("noise_seed: int64").Input("replica_id: int32").Input("step: int32")
    .Input("shape: int32").Input("cdf: double")
    .Output("counts: T")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle shape;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(3, &shape));
      TF_RETURN_IF_ERROR(c->WithRank(shape, 2, &shape));
      c->set_output(0, shape);
      return absl::OkStatus();
    });

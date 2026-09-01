"""Early process bootstrap imported before TensorFlow initializes CUDA."""

import sys

from v1_model_utils import training_orchestration


_launch_result, TRAINING_LAUNCH_PLAN = (
    training_orchestration.maybe_launch_training_workers()
)
if _launch_result is not None:
    raise SystemExit(_launch_result)
sys.argv[:] = TRAINING_LAUNCH_PLAN.argv

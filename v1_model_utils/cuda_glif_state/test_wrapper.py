import unittest

from v1_model_utils import models


# The CUDA surrogate parity is checked through the fused step in
# tests/test_cuda_glif_state.py::test_fused_step_matches_tensorflow_for_every_surrogate_and_output.
class SurrogateResolutionTest(unittest.TestCase):
    def test_legacy_pseudo_gauss_alias(self):
        self.assertEqual(models.resolve_surrogate_gradient(None, True), "gaussian")

    def test_all_surrogates_are_accepted(self):
        for surrogate in models.SURROGATE_GRADIENTS:
            self.assertEqual(models.resolve_surrogate_gradient(surrogate), surrogate)

    def test_conflicting_legacy_flag_is_rejected(self):
        with self.assertRaises(ValueError):
            models.resolve_surrogate_gradient("slayer", True)


if __name__ == "__main__":
    unittest.main()

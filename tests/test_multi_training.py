import unittest

import tensorflow as tf

from multi_training import require_python_bool


class RequirePythonBoolTest(unittest.TestCase):
    def test_accepts_python_booleans(self):
        self.assertIs(require_python_bool(False, "selector"), False)
        self.assertIs(require_python_bool(True, "selector"), True)

    def test_rejects_tensor_predicate(self):
        with self.assertRaisesRegex(
            TypeError, "selector must be a Python bool"
        ):
            require_python_bool(tf.constant(True), "selector")

    def test_rejects_integer_predicate(self):
        with self.assertRaisesRegex(
            TypeError, "selector must be a Python bool"
        ):
            require_python_bool(1, "selector")


if __name__ == "__main__":
    unittest.main()

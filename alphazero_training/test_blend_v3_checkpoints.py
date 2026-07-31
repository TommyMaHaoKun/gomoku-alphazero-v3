import unittest

import torch

from alphazero_training.blend_v3_checkpoints import blend_states


class BlendStatesTests(unittest.TestCase):
    def test_blends_float_and_selects_integer_buffers(self) -> None:
        anchor = {
            "weight": torch.tensor([0.0, 2.0]),
            "counter": torch.tensor(3, dtype=torch.int64),
        }
        update = {
            "weight": torch.tensor([4.0, 6.0]),
            "counter": torch.tensor(9, dtype=torch.int64),
        }
        result = blend_states(anchor, update, 0.25)
        torch.testing.assert_close(result["weight"], torch.tensor([1.0, 3.0]))
        self.assertEqual(result["counter"].item(), 3)

    def test_rejects_mismatched_states(self) -> None:
        with self.assertRaisesRegex(ValueError, "different parameter names"):
            blend_states({"a": torch.tensor(1.0)}, {"b": torch.tensor(1.0)}, 0.5)


if __name__ == "__main__":
    unittest.main()

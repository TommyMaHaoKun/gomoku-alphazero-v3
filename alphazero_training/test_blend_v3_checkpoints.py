import unittest

import torch

from alphazero_training.blend_v3_checkpoints import (
    approved_parent_aliases,
    blend_states,
    verify_model_compatibility,
    verify_parent_link,
)


class BlendStatesTests(unittest.TestCase):
    def test_allows_runtime_search_config_difference(self) -> None:
        anchor = {
            "model_spec": {"board_size": 19, "channels": 96},
            "config": {"board_size": 19, "channels": 96, "simulations": 192},
        }
        update = {
            "model_spec": {"board_size": 19, "channels": 96},
            "config": {"board_size": 19, "channels": 96, "simulations": 128},
        }
        verify_model_compatibility(anchor, update)

    def test_rejects_architecture_config_difference(self) -> None:
        anchor = {
            "model_spec": {"board_size": 19, "channels": 96},
            "config": {"board_size": 19, "channels": 96},
        }
        update = {
            "model_spec": {"board_size": 19, "channels": 96},
            "config": {"board_size": 15, "channels": 96},
        }
        with self.assertRaisesRegex(ValueError, "board_size"):
            verify_model_compatibility(anchor, update)

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

    def test_approved_parent_is_valid_anchor_by_identity(self) -> None:
        verify_parent_link(
            {}, "parent-sha", "parent-sha", label="anchor", allow_parent_identity=True
        )

    def test_update_still_requires_parent_provenance(self) -> None:
        with self.assertRaisesRegex(ValueError, "not linked to approved parent"):
            verify_parent_link({}, "update-sha", "parent-sha", label="update")

    def test_approved_wrapper_source_is_a_trusted_parent_alias(self) -> None:
        wrapper = {"source_checkpoint_sha256": "a" * 64}
        aliases = approved_parent_aliases(wrapper)
        verify_parent_link(
            {"parent_checkpoint_sha256": "a" * 64},
            "update-sha",
            "b" * 64,
            label="update",
            parent_aliases=aliases,
        )


if __name__ == "__main__":
    unittest.main()

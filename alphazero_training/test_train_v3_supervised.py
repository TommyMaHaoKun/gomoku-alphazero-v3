from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from alphazero_training.train_alphazero import Config, PolicyValueNet
from alphazero_training.train_v3_supervised import (
    DatasetPool,
    concatenate_batch,
    configure_selective_freeze,
    evaluate,
    policy_distillation_kl,
    resolve_dataset_mix,
    safe_hard_negative_margin_loss,
    save_checkpoint,
    validate_white_defense_pool_union,
    weighted_loss,
    unfreeze_all_parameters,
)


def _write_pool(path: Path, *, white_defense: bool, split: str = "train") -> None:
    count = 2
    states = np.zeros((count, 4, 19, 19), dtype=np.uint8)
    policies = np.zeros((count, 361), dtype=np.float32)
    arrays: dict[str, np.ndarray] = {}
    if white_defense:
        states[:, 1, 0, 0] = 1
        states[:, 2, 0, 0] = 1
        policies[:, 1:3] = 0.5
        arrays.update(
            {
                "safe_mask": (policies > 0).astype(np.uint8),
                "candidate_mask": np.repeat(
                    np.eye(1, 361, 1, dtype=np.uint8)
                    | np.eye(1, 361, 2, dtype=np.uint8)
                    | np.eye(1, 361, 3, dtype=np.uint8),
                    count,
                    axis=0,
                ),
                "source": np.asarray(
                    ["white_defense|report=aaaaaaaaaaaaaaaa|pair=0"] * count
                ),
            }
        )
        value_weights = np.zeros(count, dtype=np.float32)
    else:
        states[:, 3].fill(1)
        policies[:, 180] = 1.0
        value_weights = np.ones(count, dtype=np.float32)
    np.savez_compressed(
        path,
        states=states,
        policies=policies,
        values=np.zeros(count, dtype=np.float32),
        policy_weights=np.ones(count, dtype=np.float32),
        value_weights=value_weights,
        priority=np.ones(count, dtype=np.float32),
        group_id=np.asarray(["g0", "g1"]),
        split=np.asarray([split] * count),
        **arrays,
    )


class PolicyDistillationTests(unittest.TestCase):
    def test_zero_scale_preserves_loss_and_student_gradient(self) -> None:
        student_reference = torch.tensor(
            [[1.2, -0.4, 0.3], [-0.7, 1.1, 0.2]], requires_grad=True
        )
        student_distilled = student_reference.detach().clone().requires_grad_(True)
        teacher = torch.tensor(
            [[-0.5, 1.3, 0.1], [0.8, -0.2, 1.6]], requires_grad=True
        )
        targets = torch.tensor([0, 2])

        reference_loss = F.cross_entropy(student_reference, targets)
        reference_loss.backward()
        combined_loss = F.cross_entropy(student_distilled, targets)
        combined_loss = combined_loss + 0.0 * policy_distillation_kl(
            student_distilled, teacher
        )
        combined_loss.backward()

        torch.testing.assert_close(combined_loss.detach(), reference_loss.detach())
        torch.testing.assert_close(student_distilled.grad, student_reference.grad)
        self.assertIsNone(teacher.grad)

    def test_positive_scale_has_finite_gradient_and_pulls_toward_teacher(self) -> None:
        student = torch.tensor(
            [[2.5, -1.0, 0.2], [-1.5, 2.0, 0.4]], requires_grad=True
        )
        teacher = torch.tensor(
            [[-0.8, 2.2, 0.1], [1.4, -1.1, 2.0]], requires_grad=True
        )
        scale = 0.7

        initial_kl = policy_distillation_kl(student, teacher)
        (scale * initial_kl).backward()

        self.assertTrue(torch.isfinite(initial_kl))
        self.assertIsNotNone(student.grad)
        self.assertTrue(bool(torch.isfinite(student.grad).all()))
        self.assertGreater(float(student.grad.abs().sum()), 0.0)
        self.assertIsNone(teacher.grad)

        with torch.no_grad():
            updated_student = student - 0.05 * student.grad
            updated_kl = policy_distillation_kl(updated_student, teacher)
        self.assertLess(float(updated_kl), float(initial_kl.detach()))

        bf16_student = student.detach().to(torch.bfloat16).requires_grad_(True)
        bf16_teacher = teacher.detach().to(torch.bfloat16)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            bf16_kl = policy_distillation_kl(bf16_student, bf16_teacher)
        self.assertEqual(torch.float32, bf16_kl.dtype)
        self.assertTrue(torch.isfinite(bf16_kl))
        bf16_kl.backward()
        self.assertTrue(bool(torch.isfinite(bf16_student.grad).all()))


class WhiteDefenseSupervisedTests(unittest.TestCase):
    def test_safe_hard_negative_margin_targets_candidate_top1(self) -> None:
        logits = torch.tensor(
            [[2.0, 0.5, 1.5, 100.0], [0.0, 1.0, 2.0, 3.0]],
            requires_grad=True,
        )
        targets = torch.tensor(
            [[0.5, 0.5, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
        )
        candidates = torch.tensor(
            [[True, True, True, False], [True, True, True, True]]
        )
        loss = safe_hard_negative_margin_loss(
            logits,
            targets,
            torch.tensor([2.0, 100.0]),
            torch.tensor([True, False]),
            candidates,
            margin=1.0,
        )
        # The out-of-candidate logit 100 and ordinary second row are ignored.
        self.assertAlmostEqual(0.5, float(loss.detach()), places=6)
        loss.backward()
        self.assertAlmostEqual(-1.0, float(logits.grad[0, 0]), places=6)
        self.assertAlmostEqual(1.0, float(logits.grad[0, 2]), places=6)
        self.assertEqual(0.0, float(logits.grad[0, 3]))
        self.assertTrue(bool((logits.grad[1] == 0).all()))

    def test_safe_hard_negative_margin_is_zero_after_required_lead(self) -> None:
        logits = torch.tensor([[3.0, 0.0, 1.5, 50.0]], requires_grad=True)
        loss = safe_hard_negative_margin_loss(
            logits,
            torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            torch.ones(1),
            torch.tensor([True]),
            torch.tensor([[True, False, True, False]]),
            margin=1.0,
        )
        self.assertEqual(0.0, float(loss.detach()))
        loss.backward()
        self.assertTrue(bool((logits.grad == 0).all()))

    def test_safe_hard_negative_margin_handles_no_unsafe_candidate(self) -> None:
        logits = torch.tensor([[0.2, -0.1]], requires_grad=True)
        loss = safe_hard_negative_margin_loss(
            logits,
            torch.tensor([[0.5, 0.5]]),
            torch.ones(1),
            torch.tensor([True]),
            torch.tensor([[True, True]]),
        )
        self.assertEqual(0.0, float(loss.detach()))
        loss.backward()
        self.assertTrue(bool((logits.grad == 0).all()))

    def test_multiple_white_sources_are_paired_and_weighted_independently(self) -> None:
        specs, weights = resolve_dataset_mix(
            [Path("ordinary.npz")],
            [2.0],
            [Path("white_a.npz"), Path("white_b.npz")],
            [Path("manifest_a.json"), Path("manifest_b.json")],
            None,
        )
        self.assertEqual(
            [
                (Path("ordinary.npz"), None),
                (Path("white_a.npz"), Path("manifest_a.json")),
                (Path("white_b.npz"), Path("manifest_b.json")),
            ],
            specs,
        )
        np.testing.assert_array_equal(weights, np.asarray([2.0, 1.0, 1.0]))

        _, explicit = resolve_dataset_mix(
            [Path("ordinary.npz")],
            None,
            [Path("white_a.npz"), Path("white_b.npz")],
            [Path("manifest_a.json"), Path("manifest_b.json")],
            [0.3, 0.7],
        )
        np.testing.assert_array_equal(explicit, np.asarray([1.0, 0.3, 0.7]))
        with self.assertRaisesRegex(ValueError, "one --white-defense-manifest"):
            resolve_dataset_mix(
                [Path("ordinary.npz")],
                None,
                [Path("white_a.npz"), Path("white_b.npz")],
                [Path("manifest_a.json")],
                None,
            )
        with self.assertRaisesRegex(ValueError, "one positive --white-defense-weight"):
            resolve_dataset_mix(
                [Path("ordinary.npz")],
                None,
                [Path("white_a.npz"), Path("white_b.npz")],
                [Path("manifest_a.json"), Path("manifest_b.json")],
                [1.0],
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ordinary_path = root / "ordinary.npz"
            white_a = root / "white_a.npz"
            white_b = root / "white_b.npz"
            _write_pool(ordinary_path, white_defense=False)
            _write_pool(white_a, white_defense=True)
            _write_pool(white_b, white_defense=True)
            pools = [
                DatasetPool(ordinary_path, 1, 0.0),
                DatasetPool(white_a, 2, 0.0),
                DatasetPool(white_b, 3, 0.0),
            ]
            audit = validate_white_defense_pool_union(pools)
            self.assertEqual(audit["sources"], 2)
            self.assertEqual(audit["records"], 4)
            self.assertEqual(audit["unique_states"], 1)
            pools[2].safe_masks[:, 2] = False
            with self.assertRaisesRegex(ValueError, "cross-source white-defense conflict"):
                validate_white_defense_pool_union(pools)
            batch = concatenate_batch(
                [pools[0].sample(2), pools[1].sample(1), pools[2].sample(1)],
                np.random.default_rng(9),
            )
            self.assertEqual(2, int(batch["safe_set_rows"].sum()))
            self.assertFalse(
                np.any((batch["policies"] > 0) & ~batch["candidate_masks"])
            )

    def test_selective_freeze_opens_only_last_blocks_and_heads(self) -> None:
        model = PolicyValueNet(5, 4, 4)
        configure_selective_freeze(model, 2)
        self.assertFalse(any(parameter.requires_grad for parameter in model.stem.parameters()))
        self.assertFalse(any(parameter.requires_grad for parameter in model.tower[0].parameters()))
        self.assertFalse(any(parameter.requires_grad for parameter in model.tower[1].parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in model.tower[2].parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in model.tower[3].parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in model.policy_fc.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in model.value_fc2.parameters()))
        unfreeze_all_parameters(model)
        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))
        with self.assertRaisesRegex(ValueError, "tower depth"):
            configure_selective_freeze(model, 5)

        legacy_default = PolicyValueNet(5, 4, 2)
        configure_selective_freeze(legacy_default, 0)
        self.assertFalse(any(parameter.requires_grad for parameter in legacy_default.tower.parameters()))

    def test_checkpoint_records_selective_freeze_configuration(self) -> None:
        config = Config(board_size=5, channels=4, residual_blocks=2)
        model = PolicyValueNet(5, 4, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "warm.pt"
            save_checkpoint(
                path,
                parent={"iteration": 7},
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                config=config,
                step=9,
                parent_sha256="a" * 64,
                manifests=[],
                metrics={},
                value_loss_scale=1.0,
                value_distill_scale=0.2,
                policy_distill_scale=0.3,
                freeze_trunk_steps=300,
                train_last_residual_blocks_during_freeze=2,
            )
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        self.assertEqual(300, checkpoint["warmstart_config"]["freeze_trunk_steps"])
        self.assertEqual(
            0.0,
            checkpoint["warmstart_config"]["safe_hard_negative_scale"],
        )
        self.assertEqual(
            1.0,
            checkpoint["warmstart_config"]["safe_hard_negative_margin"],
        )
        self.assertEqual(
            2,
            checkpoint["warmstart_config"][
                "train_last_residual_blocks_during_freeze"
            ],
        )

    def test_safe_set_loss_ignores_probability_redistribution_inside_set(self) -> None:
        targets = torch.tensor([[0.5, 0.5, 0.0, 0.0]])
        candidates = torch.tensor([[True, True, True, False]])

        def loss(probabilities: list[float], safe: bool) -> float:
            _, policy, _ = weighted_loss(
                torch.log(torch.tensor([probabilities], dtype=torch.float32)),
                torch.zeros(1),
                targets,
                torch.zeros(1),
                torch.ones(1),
                torch.zeros(1),
                safe_set_rows=torch.tensor([safe]),
                candidate_masks=candidates,
            )
            return float(policy)

        self.assertAlmostEqual(
            loss([0.45, 0.45, 0.099, 0.001], True),
            loss([0.89, 0.01, 0.099, 0.001], True),
            places=6,
        )
        self.assertAlmostEqual(
            loss([0.45, 0.45, 0.099, 0.001], True),
            loss([0.225, 0.225, 0.0495, 0.5005], True),
            places=6,
        )
        self.assertGreater(
            loss([0.40, 0.40, 0.199, 0.001], True),
            loss([0.45, 0.45, 0.099, 0.001], True),
        )
        self.assertNotAlmostEqual(
            loss([0.45, 0.45, 0.099, 0.001], False),
            loss([0.89, 0.01, 0.099, 0.001], False),
            places=4,
        )

    def test_zero_validation_fraction_keeps_every_group_for_training(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ordinary.npz"
            _write_pool(path, white_defense=False)
            pool = DatasetPool(path, seed=7, validation_fraction=0.0)
            self.assertEqual(2, len(pool.training_indices))
            self.assertEqual(0, len(pool.validation_indices))

    def test_eval_npz_is_never_accepted_by_supervised_pool(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "white_eval.npz"
            _write_pool(path, white_defense=True, split="eval")
            with self.assertRaisesRegex(ValueError, "pure train split"):
                DatasetPool(path, seed=7, validation_fraction=0.0)

    def test_white_validation_reports_safe_set_metrics_not_single_label_top1(self) -> None:
        class FixedModel(torch.nn.Module):
            def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                logits = torch.zeros((len(states), 361), device=states.device)
                logits[:, 1] = 4.0  # One of the two safe actions.
                return logits, torch.zeros(len(states), device=states.device)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "white_train.npz"
            _write_pool(path, white_defense=True)
            pool = DatasetPool(path, seed=7, validation_fraction=0.0)
            metrics = evaluate(FixedModel(), [pool], torch.device("cpu"))["datasets"][0]
            self.assertEqual("white_defense", metrics["dataset_kind"])
            self.assertNotIn("policy_top1", metrics)
            self.assertEqual(1.0, metrics["top1_in_safe_set"])
            self.assertEqual(
                "renormalized_within_candidate_mask", metrics["probability_scope"]
            )
            self.assertGreater(metrics["safe_probability_mass"], 0.0)
            self.assertGreater(metrics["unsafe_mass"], 0.0)
            self.assertAlmostEqual(
                1.0,
                metrics["safe_probability_mass"] + metrics["unsafe_mass"],
                places=6,
            )


if __name__ == "__main__":
    unittest.main()

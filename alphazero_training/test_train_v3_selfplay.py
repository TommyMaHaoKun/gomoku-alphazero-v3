from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from alphazero_training.train_alphazero import Config, PolicyValueNet
import alphazero_training.train_v3_selfplay as trainer
from alphazero_training.train_v3_selfplay import (
    CircularSelfplayReplay,
    StaticReplaySource,
    V3SelfplayConfig,
    _loop_config,
    allocate_source_counts,
    generate_v3_selfplay,
    load_selfplay_chunks,
    make_scheduler,
    prune_uncommitted_replay_chunks,
    retained_replay_manifest,
    restore_v3_checkpoint,
    safe_hard_negative_margin_loss,
    save_v3_checkpoint,
    save_selfplay_chunk,
    train_mixed_steps,
    weighted_loss,
)


def _legal_arrays(count: int, board_size: int = 5) -> dict[str, np.ndarray]:
    states = np.zeros((count, 4, board_size, board_size), dtype=np.uint8)
    states[:, 3].fill(1)  # Empty board, black to move.
    policies = np.zeros((count, board_size * board_size), dtype=np.float32)
    policies[:, (board_size * board_size) // 2] = 1.0
    return {
        "states": states,
        "policies": policies,
        "values": np.zeros(count, dtype=np.float32),
        "policy_weights": np.ones(count, dtype=np.float32),
        "value_weights": np.ones(count, dtype=np.float32),
    }


def _loop_args(**overrides: object) -> SimpleNamespace:
    names = (
        "iterations",
        "selfplay_games",
        "parallel_games",
        "simulations",
        "temperature_moves",
        "max_game_plies",
        "train_steps",
        "batch_size",
        "learning_rate",
        "min_learning_rate",
        "warmup_steps",
        "weight_decay",
        "replay_capacity",
        "max_replay_chunks",
        "selfplay_quota",
        "ddqk_quota",
        "tactical_quota",
        "white_defense_quota",
        "safe_hard_negative_scale",
        "safe_hard_negative_margin",
        "selfplay_policy_weight",
        "selfplay_value_weight",
        "seed",
        "log_every_steps",
    )
    values = {name: None for name in names}
    values.update(overrides)
    values["smoke"] = False
    return SimpleNamespace(**values)


class V3SelfplayTrainerTests(unittest.TestCase):
    def test_safe_hard_negative_margin_only_uses_white_rows(self) -> None:
        logits = torch.tensor(
            [[2.0, 0.5, 1.5, 100.0], [0.0, 1.0, 2.0, 3.0]],
            requires_grad=True,
        )
        loss = safe_hard_negative_margin_loss(
            logits,
            torch.tensor(
                [[0.5, 0.5, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
            ),
            torch.tensor([2.0, 100.0]),
            torch.tensor([True, False]),
            torch.tensor(
                [[True, True, True, False], [True, True, True, True]]
            ),
            margin=1.0,
        )
        self.assertAlmostEqual(0.5, float(loss.detach()), places=6)
        loss.backward()
        self.assertAlmostEqual(-1.0, float(logits.grad[0, 0]), places=6)
        self.assertAlmostEqual(1.0, float(logits.grad[0, 2]), places=6)
        self.assertEqual(0.0, float(logits.grad[0, 3]))
        self.assertTrue(bool((logits.grad[1] == 0).all()))

    def test_margin_config_requires_authenticated_white_mix(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive white-defense quota"):
            V3SelfplayConfig(safe_hard_negative_scale=0.1).validate()
        config = V3SelfplayConfig(
            white_defense_quota=0.1,
            safe_hard_negative_scale=0.1,
            safe_hard_negative_margin=0.5,
        )
        config.validate()

    def test_mixed_training_adds_scaled_margin_only_for_white_rows(self) -> None:
        class FixedMixer:
            def sample(self, count: int) -> dict[str, np.ndarray]:
                if count != 2:
                    raise AssertionError(f"unexpected mixed batch size: {count}")
                return {
                    "states": np.zeros((2, 4, 1, 1), dtype=np.uint8),
                    "policies": np.asarray(
                        [[0.5, 0.5, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
                        dtype=np.float32,
                    ),
                    "values": np.zeros(2, dtype=np.float32),
                    "policy_weights": np.asarray([2.0, 100.0], dtype=np.float32),
                    "value_weights": np.zeros(2, dtype=np.float32),
                    "candidate_masks": np.asarray(
                        [
                            [True, True, True, False],
                            [True, True, True, True],
                        ]
                    ),
                    "source_names": np.asarray(["white_defense", "selfplay"]),
                }

        class FixedModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.logits = torch.nn.Parameter(
                    torch.tensor(
                        [[2.0, 0.5, 1.5, 100.0], [0.0, 1.0, 2.0, 3.0]]
                    )
                )
                self.values = torch.nn.Parameter(torch.zeros(2))

            def forward(
                self, _states: torch.Tensor
            ) -> tuple[torch.Tensor, torch.Tensor]:
                return self.logits, self.values

        def run(scale: float) -> dict[str, object]:
            model = FixedModel()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lambda _: 1.0
            )
            metrics, completed = train_mixed_steps(
                model,
                optimizer,
                scheduler,
                FixedMixer(),  # type: ignore[arg-type]
                V3SelfplayConfig(
                    train_steps=1,
                    batch_size=2,
                    log_every_steps=1,
                    white_defense_quota=0.5,
                    safe_hard_negative_scale=scale,
                    safe_hard_negative_margin=1.0,
                ),
                torch.device("cpu"),
            )
            self.assertEqual(1, completed)
            return metrics

        disabled = run(0.0)
        enabled = run(0.15)
        self.assertEqual(0.0, disabled["safe_hard_negative_loss"])
        self.assertAlmostEqual(
            0.5,
            float(enabled["safe_hard_negative_loss"]),
            places=6,
        )
        self.assertAlmostEqual(
            0.15 * 0.5,
            float(enabled["loss"]) - float(disabled["loss"]),
            places=5,
        )

    def test_weighted_loss_honours_independent_masks(self) -> None:
        logits = torch.tensor([[2.0, -1.0], [-2.0, 3.0]], requires_grad=True)
        predicted_values = torch.tensor([0.25, -0.50], requires_grad=True)
        policies = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        values = torch.tensor([1.0, 1.0])
        # Policy row two and value row one are independently masked out.
        policy_weights = torch.tensor([2.0, 0.0])
        value_weights = torch.tensor([0.0, 4.0])

        loss, policy_loss, value_loss = weighted_loss(
            logits,
            predicted_values,
            policies,
            values,
            policy_weights,
            value_weights,
        )
        expected_policy = -torch.log_softmax(logits[0], dim=0)[0]
        expected_value = (predicted_values[1] - values[1]).square()
        torch.testing.assert_close(policy_loss, expected_policy)
        torch.testing.assert_close(value_loss, expected_value)
        torch.testing.assert_close(loss, expected_policy + expected_value)
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertIsNotNone(predicted_values.grad)

    def test_weighted_loss_allows_a_fully_masked_head(self) -> None:
        logits = torch.zeros((1, 3), requires_grad=True)
        predicted = torch.tensor([0.2], requires_grad=True)
        target_policy = torch.tensor([[1.0, 0.0, 0.0]])
        target_value = torch.tensor([0.0])
        loss, policy_loss, value_loss = weighted_loss(
            logits,
            predicted,
            target_policy,
            target_value,
            torch.tensor([1.0]),
            torch.tensor([0.0]),
        )
        self.assertEqual(0.0, float(value_loss.detach()))
        self.assertGreater(float(policy_loss.detach()), 0.0)
        loss.backward()

    def test_source_quota_rounding_is_exact_and_stable(self) -> None:
        quotas = {"selfplay": 0.50, "ddqk": 0.30, "tactical": 0.20}
        self.assertEqual(
            {"selfplay": 5, "ddqk": 3, "tactical": 2},
            allocate_source_counts(10, quotas),
        )
        # All remainders tie at one third; stable order receives the spare row.
        self.assertEqual(
            {"selfplay": 3, "ddqk": 2, "tactical": 2},
            allocate_source_counts(
                7, {"selfplay": 1.0, "ddqk": 1.0, "tactical": 1.0}
            ),
        )
        self.assertEqual(7, sum(allocate_source_counts(7, quotas).values()))

    def test_atomic_format3_checkpoint_restores_training_and_rng(self) -> None:
        torch.manual_seed(7)
        random.seed(7)
        search_config = Config(board_size=5, win_length=5, channels=4, residual_blocks=1)
        loop_config = V3SelfplayConfig(
            iterations=2,
            selfplay_games=1,
            simulations=1,
            temperature_moves=1,
            max_game_plies=2,
            train_steps=1,
            batch_size=3,
            replay_capacity=8,
            max_replay_chunks=1,
            warmup_steps=0,
            log_every_steps=1,
            white_defense_quota=0.1,
            safe_hard_negative_scale=0.15,
            safe_hard_negative_margin=0.5,
        )
        model = PolicyValueNet(5, 4, 1)
        torch.manual_seed(17)
        approved_model = PolicyValueNet(5, 4, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=loop_config.learning_rate)
        scheduler = make_scheduler(optimizer, loop_config)
        # Create non-empty optimizer state and a non-zero scheduler position.
        output = sum(parameter.sum() for parameter in model.parameters())
        output.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        rng = np.random.default_rng(123)
        rng.random(3)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latest.pt"
            replay_manifest = [
                {
                    "iteration": 4,
                    "filename": "selfplay_000004.npz",
                    "positions": 2,
                    "bytes": 1,
                    "sha256": "0" * 64,
                }
            ]
            save_v3_checkpoint(
                path,
                iteration=4,
                global_step=9,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                search_config=search_config,
                loop_config=loop_config,
                replay_size=2,
                replay_manifest=replay_manifest,
                rng=rng,
                dataset_manifest=[],
                parent_checkpoint_sha256="abc",
                approved_model_state=approved_model.state_dict(),
                approved_checkpoint_sha256="d" * 64,
                metrics={"ok": True},
            )
            self.assertTrue(path.exists())
            self.assertFalse(path.with_name(path.name + ".tmp").exists())
            expected_rng_value = rng.random()
            expected_state = {
                name: tensor.detach().clone() for name, tensor in model.state_dict().items()
            }

            restored_model = PolicyValueNet(5, 4, 1)
            restored_optimizer = torch.optim.AdamW(
                restored_model.parameters(), lr=loop_config.learning_rate
            )
            restored_scheduler = make_scheduler(restored_optimizer, loop_config)
            restored_rng = np.random.default_rng(999)
            checkpoint = restore_v3_checkpoint(
                path,
                model=restored_model,
                optimizer=restored_optimizer,
                scheduler=restored_scheduler,
                rng=restored_rng,
                device=torch.device("cpu"),
            )
            self.assertEqual(3, checkpoint["format_version"])
            self.assertEqual("selfplay", checkpoint["v3_stage"])
            self.assertEqual(4, checkpoint["iteration"])
            self.assertEqual(9, checkpoint["global_step"])
            self.assertEqual(2, checkpoint["replay_size"])
            self.assertEqual(replay_manifest, checkpoint["replay_manifest"])
            self.assertEqual("d" * 64, checkpoint["approved_checkpoint_sha256"])
            self.assertEqual(
                0.15,
                checkpoint["v3_selfplay_config"]["safe_hard_negative_scale"],
            )
            self.assertEqual(
                0.5,
                checkpoint["v3_selfplay_config"]["safe_hard_negative_margin"],
            )
            self.assertAlmostEqual(expected_rng_value, restored_rng.random())
            self.assertEqual(scheduler.last_epoch, restored_scheduler.last_epoch)
            for name, expected in expected_state.items():
                torch.testing.assert_close(restored_model.state_dict()[name], expected)
                torch.testing.assert_close(checkpoint["train_model"][name], expected)
                torch.testing.assert_close(
                    checkpoint["best_model"][name], approved_model.state_dict()[name]
                )
            self.assertTrue(
                any(
                    not torch.equal(
                        checkpoint["train_model"][name], checkpoint["best_model"][name]
                    )
                    for name in checkpoint["train_model"]
                )
            )

    def test_replay_manifest_roundtrip_ignores_orphan_and_never_prunes_needed_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            replay_dir = Path(directory)
            replay = CircularSelfplayReplay(10, 5, 11)
            manifest: list[dict[str, object]] = []
            for iteration in range(1, 5):
                arrays = _legal_arrays(3)
                replay.add(arrays)
                entry = save_selfplay_chunk(replay_dir, iteration, arrays)
                manifest = retained_replay_manifest(
                    manifest,
                    entry,
                    replay_size=len(replay),
                    replay_capacity=replay.capacity,
                    max_chunks=3,
                )

            # Four chunks are required to reproduce the last ten positions;
            # max_chunks is therefore a soft target, never a data-loss order.
            self.assertEqual(10, len(replay))
            self.assertEqual(4, len(manifest))

            orphan = save_selfplay_chunk(replay_dir, 5, _legal_arrays(2))
            self.assertTrue((replay_dir / str(orphan["filename"])).exists())
            restored = CircularSelfplayReplay(10, 5, 99)
            loaded = load_selfplay_chunks(replay_dir, restored, manifest, 10)
            self.assertEqual(12, loaded)
            self.assertEqual(10, len(restored))
            np.testing.assert_array_equal(restored.states, replay.states)
            np.testing.assert_array_equal(restored.policies, replay.policies)

            prune_uncommitted_replay_chunks(replay_dir, manifest)
            self.assertFalse((replay_dir / str(orphan["filename"])).exists())

    def test_replay_manifest_hash_or_missing_file_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            replay_dir = Path(directory)
            entry = save_selfplay_chunk(replay_dir, 1, _legal_arrays(2))
            bad_hash = [dict(entry, sha256="0" * 64)]
            with self.assertRaisesRegex(ValueError, "hash changed"):
                load_selfplay_chunks(
                    replay_dir,
                    CircularSelfplayReplay(8, 5, 1),
                    bad_hash,
                    2,
                )

            (replay_dir / str(entry["filename"])).unlink()
            with self.assertRaisesRegex(FileNotFoundError, "missing"):
                load_selfplay_chunks(
                    replay_dir,
                    CircularSelfplayReplay(8, 5, 1),
                    [entry],
                    2,
                )

    def test_truncated_games_mask_value_and_stop_at_game_boundary(self) -> None:
        class ZeroModel(torch.nn.Module):
            def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                return (
                    torch.zeros((len(states), 25), device=states.device),
                    torch.zeros(len(states), device=states.device),
                )

        search_config = Config(
            board_size=5,
            win_length=5,
            channels=4,
            residual_blocks=1,
            simulations=1,
            inference_batch_per_game=1,
            heuristic_prior_weight=0.0,
        )
        loop_config = V3SelfplayConfig(
            iterations=1,
            selfplay_games=3,
            simulations=1,
            temperature_moves=1,
            max_game_plies=1,
            train_steps=1,
            batch_size=3,
            replay_capacity=8,
            max_replay_chunks=1,
            warmup_steps=0,
            log_every_steps=1,
        )
        trainer.STOP_REQUESTED = True
        try:
            arrays, metrics = generate_v3_selfplay(
                ZeroModel(),
                search_config,
                loop_config,
                torch.device("cpu"),
                np.random.default_rng(3),
            )
        finally:
            trainer.STOP_REQUESTED = False
        self.assertEqual(1, metrics["games"])
        self.assertTrue(metrics["stopped_early"])
        np.testing.assert_array_equal(arrays["values"], np.zeros(1, dtype=np.float32))
        np.testing.assert_array_equal(
            arrays["value_weights"], np.zeros(1, dtype=np.float32)
        )

    def test_parallel_small_games_preserve_targets_and_batch_network_work(self) -> None:
        class ZeroModel(torch.nn.Module):
            def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                return (
                    torch.zeros((len(states), 25), device=states.device),
                    torch.zeros(len(states), device=states.device),
                )

        search_config = Config(
            board_size=5,
            win_length=5,
            channels=4,
            residual_blocks=1,
            simulations=2,
            inference_batch_per_game=2,
            heuristic_prior_weight=0.0,
        )

        def run(parallel_games: int) -> tuple[dict[str, np.ndarray], dict[str, object]]:
            loop_config = V3SelfplayConfig(
                iterations=1,
                selfplay_games=4,
                parallel_games=parallel_games,
                simulations=2,
                temperature_moves=2,
                max_game_plies=2,
                train_steps=1,
                batch_size=3,
                replay_capacity=32,
                max_replay_chunks=4,
                warmup_steps=0,
                log_every_steps=1,
            )
            return generate_v3_selfplay(
                ZeroModel(),
                search_config,
                loop_config,
                torch.device("cpu"),
                np.random.default_rng(123),
            )

        sequential_arrays, sequential = run(1)
        parallel_arrays, parallel = run(4)
        for arrays, metrics in (
            (sequential_arrays, sequential),
            (parallel_arrays, parallel),
        ):
            self.assertEqual((8, 4, 5, 5), arrays["states"].shape)
            self.assertEqual((8, 25), arrays["policies"].shape)
            np.testing.assert_allclose(
                arrays["policies"].astype(np.float32).sum(axis=1), 1.0
            )
            np.testing.assert_array_equal(arrays["values"], np.zeros(8))
            np.testing.assert_array_equal(arrays["value_weights"], np.zeros(8))
            self.assertEqual(4, metrics["games"])
            self.assertEqual({"truncated": 4}, metrics["results"])
            self.assertEqual({"mcts": 8}, metrics["decision_reasons"])
            self.assertEqual(8, metrics["mcts_search_positions"])
            self.assertEqual(0, metrics["direct_search_positions"])

        self.assertEqual(1, sequential["peak_active_games"])
        self.assertEqual(4, parallel["peak_active_games"])
        self.assertGreaterEqual(parallel["max_inference_batch_size"], 4)
        self.assertLess(
            parallel["network_inference_calls"],
            sequential["network_inference_calls"],
        )

    def test_resume_rejects_optimizer_or_margin_changes(self) -> None:
        saved = V3SelfplayConfig()
        resume = {"v3_selfplay_config": asdict(saved)}
        same = _loop_config(
            _loop_args(
                learning_rate=saved.learning_rate,
                weight_decay=saved.weight_decay,
            ),
            resume,
        )
        self.assertEqual(saved.learning_rate, same.learning_rate)
        self.assertEqual(saved.weight_decay, same.weight_decay)
        for name, changed in (
            ("learning_rate", saved.learning_rate * 2),
            ("min_learning_rate", saved.min_learning_rate * 2),
            ("warmup_steps", saved.warmup_steps + 1),
            ("weight_decay", saved.weight_decay * 2),
            ("safe_hard_negative_scale", 0.1),
            ("safe_hard_negative_margin", saved.safe_hard_negative_margin + 0.5),
        ):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "cannot change"):
                    _loop_config(_loop_args(**{name: changed}), resume)

        enabled = V3SelfplayConfig(
            white_defense_quota=0.1,
            safe_hard_negative_scale=0.15,
            safe_hard_negative_margin=0.5,
        )
        restored = _loop_config(
            _loop_args(),
            {"v3_selfplay_config": asdict(enabled)},
        )
        self.assertEqual(0.15, restored.safe_hard_negative_scale)
        self.assertEqual(0.5, restored.safe_hard_negative_margin)

    def test_legacy_three_source_config_defaults_white_quota_to_zero(self) -> None:
        saved = asdict(V3SelfplayConfig())
        saved.pop("white_defense_quota")
        saved.pop("safe_hard_negative_scale")
        saved.pop("safe_hard_negative_margin")
        legacy_args = _loop_args()
        delattr(legacy_args, "safe_hard_negative_scale")
        delattr(legacy_args, "safe_hard_negative_margin")
        restored = _loop_config(
            legacy_args,
            {"v3_selfplay_config": saved},
        )
        self.assertEqual(0.0, restored.white_defense_quota)
        self.assertEqual(0.0, restored.safe_hard_negative_scale)
        self.assertEqual(1.0, restored.safe_hard_negative_margin)
        self.assertEqual(0.0, restored.quotas()["white_defense"])

    def test_static_source_rejects_negative_or_illegal_policy_and_state(self) -> None:
        def write(path: Path, arrays: dict[str, np.ndarray]) -> None:
            np.savez(
                path,
                **arrays,
                priority=np.ones(len(arrays["states"]), dtype=np.float32),
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid_path = root / "valid.npz"
            valid = _legal_arrays(1)
            write(valid_path, valid)
            self.assertEqual(1, len(StaticReplaySource("tactical", valid_path, 5, 1)))

            for filename, split in (
                ("eval_split.npz", np.asarray(["eval"])),
                ("mixed_split.npz", np.asarray(["train", "eval"])),
            ):
                split_arrays = _legal_arrays(len(split))
                split_path = root / filename
                np.savez(
                    split_path,
                    **split_arrays,
                    priority=np.ones(len(split), dtype=np.float32),
                    split=split,
                )
                with self.subTest(split=filename):
                    with self.assertRaisesRegex(ValueError, "only split=train"):
                        StaticReplaySource("tactical", split_path, 5, 1)

            train_split_path = root / "train_split.npz"
            np.savez(
                train_split_path,
                **valid,
                priority=np.ones(1, dtype=np.float32),
                split=np.asarray(["train"]),
            )
            train_source = StaticReplaySource("tactical", train_split_path, 5, 1)
            self.assertEqual("train", train_source.split_contract)

            negative = _legal_arrays(1)
            negative["policies"][0, 0] = 1.2
            negative["policies"][0, 12] = -0.2
            negative_path = root / "negative.npz"
            write(negative_path, negative)
            with self.assertRaisesRegex(ValueError, "non-negative"):
                StaticReplaySource("tactical", negative_path, 5, 1)

            occupied = _legal_arrays(1)
            occupied["states"][0, 0, 0, 0] = 1
            occupied["states"][0, 1, 0, 1] = 1
            occupied["states"][0, 2, 0, 1] = 1
            occupied["policies"][0].fill(0)
            occupied["policies"][0, 0] = 1
            occupied_path = root / "occupied_policy.npz"
            write(occupied_path, occupied)
            with self.assertRaisesRegex(ValueError, "occupied"):
                StaticReplaySource("tactical", occupied_path, 5, 1)

            missing_last = _legal_arrays(1)
            missing_last["states"][0, 0, 0, 0] = 1
            missing_last["states"][0, 1, 0, 1] = 1
            missing_last_path = root / "missing_last.npz"
            write(missing_last_path, missing_last)
            with self.assertRaisesRegex(ValueError, "last-move"):
                StaticReplaySource("tactical", missing_last_path, 5, 1)


if __name__ == "__main__":
    unittest.main()

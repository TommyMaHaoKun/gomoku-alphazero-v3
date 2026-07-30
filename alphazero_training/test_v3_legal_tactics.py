from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from alphazero_training.tactical_solver import TacticalSolver
from alphazero_training.train_alphazero import BLACK, Config, PolicyValueNet
from alphazero_training.train_v3_supervised import DatasetPool
from alphazero_training.v3_legal_tactics import (
    ACTION_COUNT,
    BOARD_SIZE,
    EVAL_SPLIT,
    TRAIN_SPLIT,
    LegalTacticsConfig,
    _board_from_stones,
    _d4_point_board,
    _oracle_actions,
    _sha256_file,
    _stones_from_state,
    evaluate_checkpoint,
    evaluate_model,
    family_catalog,
    family_fingerprint,
    generate_legal_tactics,
    load_archive,
    sample_fingerprint,
    validate_dataset,
    write_split_archives,
)


class _MappedPolicyModel(torch.nn.Module):
    """Prefer one oracle move, while preferring an occupied point even more."""

    def __init__(self, mapping: dict[bytes, tuple[int, int]]) -> None:
        super().__init__()
        self.mapping = mapping

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = torch.zeros(
            (len(states), ACTION_COUNT), dtype=torch.float32, device=states.device
        )
        for index, state in enumerate(states):
            key = state.detach().cpu().numpy().astype(np.uint8).tobytes()
            if key in self.mapping:
                occupied, oracle = self.mapping[key]
                logits[index, occupied] = 100.0
                logits[index, oracle] = 10.0
        return logits, torch.zeros(len(states), device=states.device)


class LegalTacticsGenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = LegalTacticsConfig(
            train_seed=112233,
            eval_seed=445566,
            train_samples_per_family=2,
            eval_samples_per_family=2,
            distractor_pairs=1,
            max_sample_attempts=64,
        )
        cls.dataset = generate_legal_tactics(cls.config)

    def test_catalog_has_disjoint_structural_families_and_all_oracles(self) -> None:
        solver = TacticalSolver(board_size=BOARD_SIZE, win_length=5)
        train_ids: set[str] = set()
        eval_ids: set[str] = set()
        kinds = {TRAIN_SPLIT: set(), EVAL_SPLIT: set()}
        for spec in family_catalog():
            actions = _oracle_actions(
                _board_from_stones(spec.stones),
                spec.side_to_move,
                spec.oracle_kind,
                solver,
            )
            self.assertTrue(actions, spec.family_name)
            family_id = family_fingerprint(spec, solver)
            (train_ids if spec.split == TRAIN_SPLIT else eval_ids).add(family_id)
            kinds[spec.split].add(spec.oracle_kind)
        self.assertFalse(train_ids & eval_ids)
        self.assertEqual(len(kinds[TRAIN_SPLIT]), 4)
        self.assertEqual(len(kinds[EVAL_SPLIT]), 4)

    def test_every_sample_is_reachable_and_has_real_opponent_last_move(self) -> None:
        result = validate_dataset(self.dataset)
        self.assertEqual(result["records"], len(self.dataset.states))
        self.assertEqual(result["alternating_histories"], len(self.dataset.states))
        self.assertEqual(result["real_last_move_planes"], len(self.dataset.states))
        self.assertEqual(result["overlap"], {
            "family": 0,
            "state_hash": 0,
            "d4_translation": 0,
            "generation_seed": 0,
        })
        for index, state in enumerate(self.dataset.states):
            side = int(self.dataset.side_to_move[index])
            black_count = int(state[0].sum() if side == BLACK else state[1].sum())
            white_count = int(state[1].sum() if side == BLACK else state[0].sum())
            if side == BLACK:
                self.assertEqual(black_count, white_count)
            else:
                self.assertEqual(black_count, white_count + 1)
            last = int(self.dataset.last_action[index])
            y, x = divmod(last, BOARD_SIZE)
            self.assertEqual(int(state[2].sum()), 1)
            self.assertEqual(int(state[2, y, x]), 1)
            self.assertEqual(int(state[1, y, x]), 1)
            count = int(self.dataset.move_count[index])
            self.assertEqual(int(self.dataset.move_history[index, count - 1]), last)
            self.assertTrue(np.all(self.dataset.move_history[index, count:] == -1))

    def test_both_sides_to_move_appear_in_both_splits(self) -> None:
        for split in (TRAIN_SPLIT, EVAL_SPLIT):
            sides = set(map(int, self.dataset.side_to_move[self.dataset.split == split]))
            self.assertEqual(sides, {-1, 1})

    def test_independent_oracle_reproduces_every_stored_policy(self) -> None:
        solver = TacticalSolver(board_size=BOARD_SIZE, win_length=5)
        for index, state in enumerate(self.dataset.states):
            side = int(self.dataset.side_to_move[index])
            expected = _oracle_actions(
                _board_from_stones(_stones_from_state(state, side)),
                side,
                str(self.dataset.oracle_kind[index]),
                solver,
            )
            self.assertEqual(
                expected,
                tuple(map(int, np.flatnonzero(self.dataset.policies[index] > 0))),
            )

    def test_tampered_last_move_plane_is_rejected(self) -> None:
        states = self.dataset.states.copy()
        states[0, 2].fill(0)
        tampered = replace(self.dataset, states=states)
        with self.assertRaisesRegex(ValueError, "state is not the replay encoding"):
            validate_dataset(tampered)

    def test_sample_fingerprint_is_d4_and_translation_invariant(self) -> None:
        spec = next(
            item for item in family_catalog() if item.family_name == "eval_broken_four_win"
        )
        solver = TacticalSolver(board_size=BOARD_SIZE, win_length=5)
        actions = _oracle_actions(
            _board_from_stones(spec.stones), spec.side_to_move, spec.oracle_kind, solver
        )
        last_action = next(
            y * BOARD_SIZE + x
            for x, y, stone in spec.stones
            if stone == -spec.side_to_move
        )
        baseline = sample_fingerprint(
            spec.stones, spec.side_to_move, last_action, actions
        )
        for symmetry in range(8):
            transformed_stones = tuple(
                (*_d4_point_board(x, y, symmetry), stone)
                for x, y, stone in spec.stones
            )
            transformed_actions = tuple(
                _d4_point_board(action % BOARD_SIZE, action // BOARD_SIZE, symmetry)[1]
                * BOARD_SIZE
                + _d4_point_board(action % BOARD_SIZE, action // BOARD_SIZE, symmetry)[0]
                for action in actions
            )
            lx, ly = _d4_point_board(
                last_action % BOARD_SIZE, last_action // BOARD_SIZE, symmetry
            )
            transformed_last = ly * BOARD_SIZE + lx
            self.assertEqual(
                baseline,
                sample_fingerprint(
                    transformed_stones,
                    spec.side_to_move,
                    transformed_last,
                    transformed_actions,
                ),
            )

    def test_same_config_is_deterministic(self) -> None:
        small = LegalTacticsConfig(
            train_seed=91,
            eval_seed=92,
            train_samples_per_family=1,
            eval_samples_per_family=1,
            max_sample_attempts=64,
        )
        first = generate_legal_tactics(small)
        second = generate_legal_tactics(small)
        for name in first.arrays():
            self.assertTrue(np.array_equal(first.arrays()[name], second.arrays()[name]), name)


class LegalTacticsArtifactAndEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = generate_legal_tactics(
            LegalTacticsConfig(
                train_seed=7001,
                eval_seed=8001,
                train_samples_per_family=1,
                eval_samples_per_family=1,
                max_sample_attempts=64,
            )
        )

    def test_split_npz_round_trip_and_trainer_grouping(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifacts = write_split_archives(
                self.dataset,
                root / "legal_train.npz",
                root / "sealed_eval.npz",
                root / "manifest.json",
            )
            train = load_archive(artifacts["train"])
            held_out = load_archive(artifacts["eval"])
            self.assertEqual(set(map(str, train.split)), {TRAIN_SPLIT})
            self.assertEqual(set(map(str, held_out.split)), {EVAL_SPLIT})
            with np.load(artifacts["train"], allow_pickle=False) as archive:
                groups = archive["group_id"].copy()
                for name in archive.files:
                    self.assertNotEqual(archive[name].dtype.kind, "O")
            pool = DatasetPool(artifacts["train"], seed=19, validation_fraction=0.2)
            self.assertFalse(
                set(map(str, groups[pool.training_indices]))
                & set(map(str, groups[pool.validation_indices]))
            )
            with self.assertRaisesRegex(ValueError, "pure train split"):
                DatasetPool(artifacts["eval"], seed=19, validation_fraction=0.2)

            mixed_path = root / "mixed_train_eval.npz"
            np.savez_compressed(mixed_path, **self.dataset.arrays())
            with self.assertRaisesRegex(ValueError, "pure train split"):
                DatasetPool(mixed_path, seed=19, validation_fraction=0.2)
            self.assertTrue(artifacts["manifest"].is_file())

    def test_requested_split_must_exist_without_single_split_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifacts = write_split_archives(
                self.dataset, root / "legal_train.npz", root / "sealed_eval.npz"
            )
            train = load_archive(artifacts["train"])
            with self.assertRaisesRegex(ValueError, "no samples for split eval"):
                evaluate_model(
                    _MappedPolicyModel({}),
                    Config(
                        simulations=0,
                        inference_batch_per_game=1,
                        heuristic_prior_weight=0.0,
                    ),
                    train,
                    simulations=0,
                    split=EVAL_SPLIT,
                    device="cpu",
                    limit=1,
                )

    def test_checkpoint_model_key_is_explicit_and_missing_key_is_rejected(self) -> None:
        config = Config(
            channels=4,
            residual_blocks=1,
            simulations=0,
            inference_batch_per_game=1,
            heuristic_prior_weight=0.0,
        )
        base_model = PolicyValueNet(
            config.board_size, config.channels, config.residual_blocks
        )
        states: dict[str, dict[str, torch.Tensor]] = {}
        for key, marker in (
            ("best_model", 1.0),
            ("candidate_model", 2.0),
            ("train_model", 3.0),
        ):
            state = {
                name: tensor.detach().clone()
                for name, tensor in base_model.state_dict().items()
            }
            state["policy_fc.bias"].fill_(marker)
            states[key] = state

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifacts = write_split_archives(
                self.dataset, root / "legal_train.npz", root / "sealed_eval.npz"
            )
            checkpoint = root / "checkpoint.pt"
            torch.save(
                {
                    "config": vars(config),
                    "iteration": 17,
                    **states,
                },
                checkpoint,
            )

            def fake_evaluate(model: PolicyValueNet, *_args: object, **_kwargs: object):
                marker = float(model.state_dict()["policy_fc.bias"][0])
                return {"loaded_marker": marker}

            with patch(
                "alphazero_training.v3_legal_tactics.evaluate_model",
                side_effect=fake_evaluate,
            ):
                default_result = evaluate_checkpoint(
                    checkpoint, artifacts["eval"], device="cpu"
                )
                self.assertEqual(default_result["checkpoint_model_key"], "best_model")
                self.assertEqual(default_result["loaded_marker"], 1.0)
                for key, marker in (
                    ("candidate_model", 2.0),
                    ("train_model", 3.0),
                ):
                    result = evaluate_checkpoint(
                        checkpoint,
                        artifacts["eval"],
                        model_key=key,
                        device="cpu",
                    )
                    self.assertEqual(result["checkpoint_model_key"], key)
                    self.assertEqual(result["loaded_marker"], marker)

            missing_checkpoint = root / "missing_candidate.pt"
            torch.save(
                {
                    "config": vars(config),
                    "iteration": 17,
                    "best_model": states["best_model"],
                },
                missing_checkpoint,
            )
            with self.assertRaisesRegex(ValueError, "does not contain requested model key"):
                evaluate_checkpoint(
                    missing_checkpoint,
                    artifacts["eval"],
                    model_key="candidate_model",
                    device="cpu",
                )

    def test_npz_bytes_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = write_split_archives(
                self.dataset, root / "a_train.npz", root / "a_eval.npz"
            )
            second = write_split_archives(
                self.dataset, root / "b_train.npz", root / "b_eval.npz"
            )
            self.assertEqual(_sha256_file(first["train"]), _sha256_file(second["train"]))
            self.assertEqual(_sha256_file(first["eval"]), _sha256_file(second["eval"]))

    def test_raw_network_masks_occupied_logits_and_search_metric_is_separate(self) -> None:
        mapping: dict[bytes, tuple[int, int]] = {}
        eval_indices = np.flatnonzero(self.dataset.split == EVAL_SPLIT)
        for index in eval_indices:
            state = self.dataset.states[index]
            occupied = int(np.flatnonzero((state[0] | state[1]).reshape(-1))[0])
            expected = int(np.flatnonzero(self.dataset.oracle_mask[index])[0])
            mapping[state.tobytes()] = (occupied, expected)
        model = _MappedPolicyModel(mapping)
        result = evaluate_model(
            model,
            Config(
                simulations=0,
                inference_batch_per_game=1,
                heuristic_prior_weight=0.0,
            ),
            self.dataset,
            simulations=0,
            split=EVAL_SPLIT,
            device="cpu",
        )
        self.assertEqual(result["raw_network"]["top1"], 1.0)
        self.assertEqual(result["raw_network"]["mean_best_rank"], 1.0)
        self.assertEqual(result["v3_search_with_exact_oracle"]["accuracy"], 1.0)
        self.assertIn("raw_network", result["interpretation"])


if __name__ == "__main__":
    unittest.main()

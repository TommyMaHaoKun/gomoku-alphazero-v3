from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from alphazero_training.tactical_solver import (
    SolveResult,
    SolveStatus,
    TacticalSolver,
)
from alphazero_training.train_alphazero import BLACK, WHITE, GomokuGame
from alphazero_training.train_v3_supervised import DatasetPool
from alphazero_training.white_defense_dataset import (
    ACTION_COUNT,
    ActionLabels,
    EVAL_SPLIT,
    TRAIN_SPLIT,
    WhiteDefenseConfig,
    build_dataset_from_validated_report,
    generate_white_defense_dataset,
    label_safe_actions,
    sha256_file,
    validate_dataset,
    write_split_archives,
)


class _NoThreatSolver:
    def immediate_wins(self, board: object, player: int) -> tuple[int, ...]:
        return ()

    def forced_wins_in_three(self, board: object, player: int) -> tuple[int, ...]:
        return ()

    def solve_vcf(
        self, board: object, player: int, limits: object
    ) -> SolveResult:
        return SolveResult(SolveStatus.PROVEN_NO_VCF, (), (), (), 1)


class _UnknownSolver(_NoThreatSolver):
    def solve_vcf(
        self, board: object, player: int, limits: object
    ) -> SolveResult:
        return SolveResult(SolveStatus.UNKNOWN_BUDGET, (), (), (), 3)


def _black_line_loss(pair_index: int, y: int) -> dict[str, object]:
    """A legal game where white blocks one end of an open four and loses."""

    moves = [
        [5, y, 1],
        [0, y - 2, 2],
        [6, y, 1],
        [1, y - 2, 2],
        [7, y, 1],
        [2, y - 2, 2],
        [8, y, 1],
        [4, y, 2],
        [9, y, 1],
    ]
    return {
        "pair_index": pair_index,
        "model_color": 2,
        # The first two moves are a fixed opening, not model decisions.
        "opening": [[5, y], [0, y - 2]],
        "moves": moves,
        "winner": 1,
        "model_result": 0.0,
        "plies": len(moves),
        "model_seconds": 0.1,
        "ddqk_seconds": 0.1,
        "model_moves": 3,
        "ddqk_moves": 4,
        "termination": "win",
        "error": None,
        "model_decision_reasons": [],
    }


def _report(groups: int = 4) -> dict[str, object]:
    return {
        "format_version": 3,
        "games": [_black_line_loss(index, 6 + index * 2) for index in range(groups)],
    }


def _config(**overrides: object) -> WhiteDefenseConfig:
    values: dict[str, object] = {
        "decision_distances": (1, 2),
        "eval_fraction": 0.25,
        "split_seed": 91,
        "candidate_radius": 2,
        "vcf_plies": (5, 7),
        "vcf_max_nodes": 100,
        "vcf_time_ms": 0.0,
    }
    values.update(overrides)
    return WhiteDefenseConfig(**values)  # type: ignore[arg-type]


class WhiteDefenseDatasetTests(unittest.TestCase):
    report_sha = "a" * 64

    def test_duplicate_encoded_state_is_labelled_once_and_reused(self) -> None:
        report = _report(1)
        duplicate = json.loads(json.dumps(report["games"][0]))  # type: ignore[index]
        report["games"].append(duplicate)  # type: ignore[union-attr]

        def classify(game: GomokuGame, config: WhiteDefenseConfig, **_: object) -> ActionLabels:
            actions = tuple(map(int, game.search_actions(config.candidate_radius)))
            return ActionLabels(actions, actions, (), (), (), (), 7, 2)

        with patch(
            "alphazero_training.white_defense_dataset.label_safe_actions",
            side_effect=classify,
        ) as labeler:
            dataset = build_dataset_from_validated_report(
                report,
                report_sha256=self.report_sha,
                config=_config(eval_fraction=0.0, decision_distances=(1,)),
                solver=_NoThreatSolver(),
            )
        self.assertEqual(labeler.call_count, 1)
        self.assertEqual(len(dataset.states), 2)
        self.assertEqual(dataset.summary["counts"]["unique_labelled_states"], 1)
        self.assertEqual(dataset.summary["counts"]["state_label_cache_hits"], 1)
        validate_dataset(dataset)

    def test_identical_state_with_conflicting_labels_is_rejected(self) -> None:
        report = _report(1)
        report["games"].append(  # type: ignore[union-attr]
            json.loads(json.dumps(report["games"][0]))  # type: ignore[index]
        )
        dataset = build_dataset_from_validated_report(
            report,
            report_sha256=self.report_sha,
            config=_config(eval_fraction=0.0, decision_distances=(1,)),
            solver=_NoThreatSolver(),
        )
        action = next(
            int(value)
            for value in np.flatnonzero(dataset.safe_mask[1])
            if int(value) != int(dataset.original_action[1])
        )
        dataset.safe_mask[1, action] = 0
        dataset.unsafe_vcf_mask[1, action] = 1
        dataset.safe_count[1] -= 1
        dataset.unsafe_count[1] += 1
        dataset.unsafe_vcf_count[1] += 1
        dataset.policies[1] = (
            dataset.safe_mask[1].astype(np.float32) / float(dataset.safe_count[1])
        ).astype(np.float16)
        with self.assertRaisesRegex(ValueError, "conflicting tactical labels"):
            validate_dataset(dataset)

    def test_white_state_encoding_and_legal_replay(self) -> None:
        dataset = build_dataset_from_validated_report(
            _report(2),
            report_sha256=self.report_sha,
            config=_config(eval_fraction=0.0, decision_distances=(1,)),
            solver=_NoThreatSolver(),
        )
        validate_dataset(dataset)
        state = dataset.states[0]
        self.assertEqual(int(state[3].sum()), 0, "white-to-move plane must be zero")
        self.assertEqual(int(state[0].sum()), 3, "plane 0 is current white")
        self.assertEqual(int(state[1].sum()), 4, "plane 1 is opponent black")
        last = int(dataset.last_action[0])
        y, x = divmod(last, 19)
        self.assertEqual(int(state[2].sum()), 1)
        self.assertEqual(int(state[2, y, x]), 1)
        self.assertEqual(int(state[1, y, x]), 1)
        count = int(dataset.move_count[0])
        game = GomokuGame()
        for action in map(int, dataset.move_history[0, :count]):
            game.play(action)
        self.assertEqual(game.player, WHITE)
        self.assertTrue(np.array_equal(game.encode(), state))

    def test_malformed_replay_is_rejected(self) -> None:
        report = _report(2)
        first = report["games"][0]  # type: ignore[index]
        first["moves"][2] = list(first["moves"][0])  # type: ignore[index]
        first["moves"][2][2] = 1  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "repeated point"):
            build_dataset_from_validated_report(
                report,
                report_sha256=self.report_sha,
                config=_config(eval_fraction=0.0),
                solver=_NoThreatSolver(),
            )

    def test_already_lost_open_four_position_has_no_safe_action(self) -> None:
        game = GomokuGame()
        history = [
            9 * 19 + 5,
            0,
            9 * 19 + 6,
            1,
            9 * 19 + 7,
            2,
            9 * 19 + 8,
        ]
        for action in history:
            game.play(action)
        self.assertEqual(game.player, WHITE)
        labels = label_safe_actions(
            game,
            _config(
                decision_distances=(1,),
                eval_fraction=0.0,
                vcf_plies=(1,),
                vcf_max_nodes=1_000,
            ),
            solver=TacticalSolver(board_size=19, win_length=5),
        )
        self.assertEqual(set(labels.candidate_actions), {9 * 19 + 4, 9 * 19 + 9})
        self.assertEqual(labels.safe_actions, ())
        self.assertEqual(
            set(labels.unsafe_immediate_actions),
            set(labels.candidate_actions),
        )

    def test_unknown_budget_is_recorded_and_retained_in_policy(self) -> None:
        dataset = build_dataset_from_validated_report(
            _report(2),
            report_sha256=self.report_sha,
            config=_config(eval_fraction=0.0, decision_distances=(2,)),
            solver=_UnknownSolver(),
        )
        self.assertTrue(np.all(dataset.vcf_unknown_count == dataset.candidate_count))
        self.assertTrue(np.array_equal(dataset.vcf_unknown_mask, dataset.safe_mask))
        self.assertTrue(
            np.array_equal(dataset.policies > 0, dataset.vcf_unknown_mask.astype(bool))
        )
        self.assertEqual(dataset.summary["claim_boundary"]["unknown_budget_policy"],
                         "retained_as_safe_and_recorded_not_unsafe")

    def test_group_split_is_decided_before_positions_and_has_no_leakage(self) -> None:
        dataset = build_dataset_from_validated_report(
            _report(4),
            report_sha256=self.report_sha,
            config=_config(),
            solver=_NoThreatSolver(),
        )
        result = validate_dataset(dataset)
        self.assertGreater(result["train_records"], 0)
        self.assertGreater(result["eval_records"], 0)
        for group in np.unique(dataset.group_id):
            self.assertEqual(len(set(dataset.split[dataset.group_id == group])), 1)
        for opening in np.unique(dataset.opening_sha256):
            self.assertEqual(
                len(set(dataset.split[dataset.opening_sha256 == opening])), 1
            )
        self.assertTrue(
            dataset.summary["split"]["assigned_before_replay_or_tactical_labelling"]
        )

    def test_generation_is_deterministic(self) -> None:
        first = build_dataset_from_validated_report(
            _report(4),
            report_sha256=self.report_sha,
            config=_config(),
            solver=_NoThreatSolver(),
        )
        second = build_dataset_from_validated_report(
            _report(4),
            report_sha256=self.report_sha,
            config=_config(),
            solver=_NoThreatSolver(),
        )
        self.assertEqual(first.summary, second.summary)
        for name, first_array in first.arrays().items():
            self.assertTrue(
                np.array_equal(first_array, second.arrays()[name]),
                name,
            )

    def test_strict_loader_requires_format3_validation(self) -> None:
        report = _report(2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            with patch(
                "alphazero_training.white_defense_dataset.validate_benchmark_report",
                return_value={"format_version": 3, "complete_pairs": 2},
            ) as validator:
                dataset = generate_white_defense_dataset(
                    path,
                    config=_config(eval_fraction=0.0, decision_distances=(2,)),
                    solver=_NoThreatSolver(),
                )
            validator.assert_called_once()
            self.assertEqual(str(dataset.summary["report"]), str(path.resolve()))
            self.assertEqual(
                str(dataset.report_sha256[0]),
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )

    def test_split_archives_are_trainer_compatible_and_hash_bound(self) -> None:
        dataset = build_dataset_from_validated_report(
            _report(4),
            report_sha256=self.report_sha,
            config=_config(),
            solver=_NoThreatSolver(),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "white_defense_train.npz"
            evaluation = root / "white_defense_eval.npz"
            manifest = root / "white_defense_manifest.json"
            artifacts = write_split_archives(dataset, train, evaluation, manifest)
            DatasetPool(train, seed=7, validation_fraction=0.2)
            with np.load(train, allow_pickle=False) as archive:
                self.assertEqual(set(map(str, archive["split"])), {TRAIN_SPLIT})
                self.assertTrue(np.all(archive["value_weights"] == 0))
            with np.load(evaluation, allow_pickle=False) as archive:
                self.assertEqual(set(map(str, archive["split"])), {EVAL_SPLIT})
            recorded = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(
                recorded["artifacts"]["train"]["path"], train.name
            )
            self.assertEqual(
                recorded["artifacts"]["eval"]["path"], evaluation.name
            )
            self.assertEqual(recorded["artifacts"]["train"]["sha256"], sha256_file(train))
            self.assertEqual(recorded["artifacts"]["eval"]["sha256"], sha256_file(evaluation))
            sidecar = artifacts["manifest_sha256"].read_text(encoding="utf-8")
            self.assertTrue(sidecar.startswith(sha256_file(manifest)))
            with self.assertRaises(FileExistsError):
                write_split_archives(dataset, train, evaluation, manifest)

    def test_requested_eval_split_cannot_be_written_empty(self) -> None:
        dataset = build_dataset_from_validated_report(
            _report(2),
            report_sha256=self.report_sha,
            config=WhiteDefenseConfig(
                decision_distances=(1,), eval_fraction=0.25, split_seed=91
            ),
            solver=_UnknownSolver(),
        )
        dataset.split[:] = TRAIN_SPLIT
        dataset.summary["validation"] = validate_dataset(dataset)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "no salvageable eval samples"):
                write_split_archives(
                    dataset,
                    root / "train.npz",
                    root / "eval.npz",
                    root / "manifest.json",
                )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from alphazero_training.evaluate_white_defense import (
    WhiteDefenseEvaluationError,
    evaluate_white_defense,
    sha256_file,
)
from alphazero_training.train_alphazero import GomokuGame, PolicyValueNet
from alphazero_training.white_defense_dataset import (
    ACTION_COUNT,
    BOARD_SIZE,
    EVAL_SPLIT,
    TRAIN_SPLIT,
    WhiteDefenseDataset,
    stable_json_sha256,
    write_split_archives,
)


def _unicode(values: list[str]) -> np.ndarray:
    width = max(1, *(len(value) for value in values))
    return np.asarray(values, dtype=f"<U{width}")


def _clone_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in state.items()}


def _model_state_with_logits(logits: np.ndarray) -> dict[str, torch.Tensor]:
    model = PolicyValueNet(BOARD_SIZE, 4, 1)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.policy_fc.bias.copy_(torch.from_numpy(logits.astype(np.float32)))
    return _clone_state(model.state_dict())


def _fixture_dataset() -> tuple[WhiteDefenseDataset, dict[str, np.ndarray]]:
    black_actions = [9 * BOARD_SIZE + 9, 8 * BOARD_SIZE + 8]
    splits = [TRAIN_SPLIT, EVAL_SPLIT]
    states: list[np.ndarray] = []
    candidates: list[np.ndarray] = []
    safes: list[np.ndarray] = []
    histories: list[np.ndarray] = []
    state_hashes: list[str] = []
    original_actions: list[int] = []
    for index, black_action in enumerate(black_actions):
        game = GomokuGame(BOARD_SIZE, 5)
        game.play(black_action)
        state = game.encode()
        candidate_actions = game.search_actions(2)
        candidate = np.zeros(ACTION_COUNT, dtype=np.uint8)
        candidate[candidate_actions] = 1
        # Two actions are labelled bounded-safe; all other candidates get one
        # mutually exclusive unsafe reason.  Deep validation checks that these
        # masks completely and exclusively classify search_actions(radius=2).
        safe = np.zeros(ACTION_COUNT, dtype=np.uint8)
        safe[candidate_actions[:2]] = 1
        history = np.full(ACTION_COUNT, -1, dtype=np.int16)
        history[0] = black_action
        states.append(state)
        candidates.append(candidate)
        safes.append(safe)
        histories.append(history)
        state_hashes.append(hashlib.sha256(state.tobytes()).hexdigest())
        original_actions.append(int(candidate_actions[index % 2]))

    candidate_mask = np.stack(candidates)
    safe_mask = np.stack(safes)
    unsafe_three = candidate_mask - safe_mask
    policies = safe_mask.astype(np.float32) / safe_mask.sum(axis=1, keepdims=True)
    zeros_mask = np.zeros_like(candidate_mask)
    report_hash = "a" * 64
    opening_hashes = ["b" * 64, "c" * 64]
    group_ids = [
        f"white_defense|report={report_hash}|pair={index}|opening={opening_hashes[index]}"
        for index in range(2)
    ]
    candidate_count = candidate_mask.sum(axis=1).astype(np.int16)
    safe_count = safe_mask.sum(axis=1).astype(np.int16)
    unsafe_count = unsafe_three.sum(axis=1).astype(np.int16)
    dataset = WhiteDefenseDataset(
        states=np.stack(states),
        policies=policies,
        values=np.zeros(2, dtype=np.float32),
        policy_weights=np.ones(2, dtype=np.float32),
        value_weights=np.zeros(2, dtype=np.float32),
        source=_unicode(["white_defense", "white_defense"]),
        priority=np.ones(2, dtype=np.float32),
        group_id=_unicode(group_ids),
        split=_unicode(splits),
        report_sha256=_unicode([report_hash, report_hash]),
        opening_sha256=_unicode(opening_hashes),
        game_index=np.asarray([0, 1], dtype=np.int32),
        pair_index=np.asarray([0, 1], dtype=np.int32),
        ply_index=np.ones(2, dtype=np.int16),
        white_decision_distance=np.asarray([2, 2], dtype=np.int16),
        original_action=np.asarray(original_actions, dtype=np.int16),
        original_action_in_candidates=np.ones(2, dtype=np.uint8),
        original_action_safe=np.ones(2, dtype=np.uint8),
        last_action=np.asarray(black_actions, dtype=np.int16),
        move_count=np.ones(2, dtype=np.int16),
        move_history=np.stack(histories),
        state_hash=_unicode(state_hashes),
        candidate_mask=candidate_mask,
        safe_mask=safe_mask,
        vcf_unknown_mask=zeros_mask.copy(),
        unsafe_immediate_mask=zeros_mask.copy(),
        unsafe_three_ply_mask=unsafe_three,
        unsafe_vcf_mask=zeros_mask.copy(),
        candidate_count=candidate_count,
        safe_count=safe_count,
        unsafe_count=unsafe_count,
        vcf_unknown_count=np.zeros(2, dtype=np.int16),
        unsafe_immediate_count=np.zeros(2, dtype=np.int16),
        unsafe_three_ply_count=unsafe_count.copy(),
        unsafe_vcf_count=np.zeros(2, dtype=np.int16),
        vcf_nodes=np.zeros(2, dtype=np.int32),
        vcf_queries=np.zeros(2, dtype=np.int16),
        candidate_radius=np.full(2, 2, dtype=np.int8),
        vcf_max_plies=np.full(2, 5, dtype=np.int8),
        summary={
            "schema_version": 1,
            "source": "unit_test_white_defense",
            "config": {"eval_fraction": 0.2},
            "split": {
                "exported_records": {"train": 1, "eval": 1},
                "assigned_before_replay_or_tactical_labelling": True,
            },
        },
    )
    return dataset, {
        "eval_candidate": candidate_mask[1].astype(bool),
        "eval_safe": safe_mask[1].astype(bool),
    }


class WhiteDefenseEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        dataset, self.fixture = _fixture_dataset()
        self.train = self.root / "train.npz"
        self.eval = self.root / "eval.npz"
        self.manifest = self.root / "manifest.json"
        self.output = self.root / "evaluation.json"
        write_split_archives(dataset, self.train, self.eval, self.manifest)

        candidate = self.fixture["eval_candidate"]
        safe = self.fixture["eval_safe"]
        safe_actions = np.flatnonzero(safe)
        unsafe_actions = np.flatnonzero(candidate & ~safe)
        outside_actions = np.flatnonzero(~candidate)
        self.safe_action = int(safe_actions[0])
        self.unsafe_action = int(unsafe_actions[0])
        self.outside_action = int(outside_actions[0])

        candidate_logits = np.zeros(ACTION_COUNT, dtype=np.float32)
        candidate_logits[self.safe_action] = 3.0
        candidate_logits[self.unsafe_action] = 2.0
        candidate_logits[self.outside_action] = 10.0
        train_logits = np.zeros(ACTION_COUNT, dtype=np.float32)
        train_logits[self.unsafe_action] = 5.0
        best_logits = np.zeros(ACTION_COUNT, dtype=np.float32)
        best_logits[int(safe_actions[1])] = 4.0
        self.states = {
            "candidate_model": _model_state_with_logits(candidate_logits),
            "train_model": _model_state_with_logits(train_logits),
            "best_model": _model_state_with_logits(best_logits),
        }
        self.checkpoint = self.root / "checkpoint.pt"
        self._write_checkpoint(self.states)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_checkpoint(
        self,
        states: dict[str, dict[str, torch.Tensor]],
        *,
        v3_stage: str | None = "selfplay",
    ) -> None:
        payload = {
            "iteration": 17,
            "format_version": 3,
            "config": {
                "board_size": BOARD_SIZE,
                "win_length": 5,
                "channels": 4,
                "residual_blocks": 1,
            },
            **states,
        }
        if v3_stage is not None:
            payload["v3_stage"] = v3_stage
        torch.save(payload, self.checkpoint)

    def _resign_manifest(self, mutate) -> None:
        payload = json.loads(self.manifest.read_text(encoding="utf-8"))
        mutate(payload)
        payload.pop("manifest_payload_sha256", None)
        payload["manifest_payload_sha256"] = stable_json_sha256(payload)
        self.manifest.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        sidecar = self.manifest.with_suffix(self.manifest.suffix + ".sha256")
        sidecar.write_text(
            f"{sha256_file(self.manifest)}  {self.manifest.name}\n", encoding="utf-8"
        )

    def _replace_eval_arrays(self, mutate) -> None:
        with np.load(self.eval, allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
        mutate(arrays)
        temporary = self.root / "changed.npz"
        np.savez_compressed(temporary, **arrays)
        temporary.replace(self.eval)

        def update(payload: dict[str, object]) -> None:
            artifact = payload["artifacts"]["eval"]  # type: ignore[index]
            artifact["bytes"] = self.eval.stat().st_size
            artifact["sha256"] = sha256_file(self.eval)
            artifact["records"] = len(arrays["states"])
            payload["split"]["exported_records"]["eval"] = len(arrays["states"])  # type: ignore[index]
            payload["validation"]["eval_records"] = len(arrays["states"])  # type: ignore[index]

        self._resign_manifest(update)

    def test_numeric_metrics_use_candidate_normalization_and_exclude_outside(self) -> None:
        report = evaluate_white_defense(
            self.checkpoint, self.eval, self.manifest, self.output, device="cpu"
        )
        self.assertEqual(report["checkpoint"]["checkpoint_model_key"], "candidate_model")
        candidate = self.fixture["eval_candidate"]
        safe = self.fixture["eval_safe"]
        logits = np.zeros(ACTION_COUNT, dtype=np.float64)
        logits[self.safe_action] = 3.0
        logits[self.unsafe_action] = 2.0
        logits[self.outside_action] = 10.0
        weights = np.exp(logits[candidate] - logits[candidate].max())
        expected_safe = float(weights[safe[candidate]].sum() / weights.sum())
        metrics = report["metrics"]
        self.assertEqual(metrics["top1_in_safe_set"], 1.0)
        self.assertEqual(metrics["global_top1_in_candidate"], 0.0)
        self.assertAlmostEqual(metrics["safe_probability_mass"], expected_safe, places=7)
        self.assertAlmostEqual(metrics["unsafe_mass"], 1.0 - expected_safe, places=7)
        self.assertIn("candidate-external", metrics["unsafe_definition"])
        published = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertEqual(published, report)

    def test_relative_manifest_and_relocated_legacy_windows_paths_are_portable(self) -> None:
        payload = json.loads(self.manifest.read_text(encoding="utf-8"))
        self.assertEqual(payload["artifacts"]["train"]["path"], self.train.name)
        self.assertEqual(payload["artifacts"]["eval"]["path"], self.eval.name)

        def make_legacy(manifest: dict[str, object]) -> None:
            artifacts = manifest["artifacts"]  # type: ignore[index]
            artifacts["train"]["path"] = rf"Z:\\retired-host\\{self.train.name}"  # type: ignore[index]
            artifacts["eval"]["path"] = rf"Z:\\retired-host\\{self.eval.name}"  # type: ignore[index]

        self._resign_manifest(make_legacy)
        report = evaluate_white_defense(
            self.checkpoint, self.eval, self.manifest, self.output, device="cpu"
        )
        self.assertEqual(report["metrics"]["records"], 1)

    def test_model_key_auto_priority_and_explicit_selection(self) -> None:
        auto = evaluate_white_defense(
            self.checkpoint, self.eval, self.manifest, self.output, model_key="auto"
        )
        self.assertEqual(auto["checkpoint"]["checkpoint_model_key"], "candidate_model")
        self.assertEqual(auto["metrics"]["top1_in_safe_set"], 1.0)

        explicit_train = evaluate_white_defense(
            self.checkpoint,
            self.eval,
            self.manifest,
            self.output,
            model_key="train_model",
        )
        self.assertEqual(explicit_train["checkpoint"]["checkpoint_model_key"], "train_model")
        self.assertEqual(explicit_train["metrics"]["top1_in_safe_set"], 0.0)

        self._write_checkpoint({"train_model": self.states["train_model"], "best_model": self.states["best_model"]})
        train_fallback = evaluate_white_defense(
            self.checkpoint, self.eval, self.manifest, self.output
        )
        self.assertEqual(train_fallback["checkpoint"]["checkpoint_model_key"], "train_model")

        self._write_checkpoint({"best_model": self.states["best_model"]})
        best_fallback = evaluate_white_defense(
            self.checkpoint, self.eval, self.manifest, self.output
        )
        self.assertEqual(best_fallback["checkpoint"]["checkpoint_model_key"], "best_model")
        self.assertIn("old champion", best_fallback["checkpoint"]["selection_warning"])
        with self.assertRaisesRegex(WhiteDefenseEvaluationError, "requested model key"):
            evaluate_white_defense(
                self.checkpoint,
                self.eval,
                self.manifest,
                self.output,
                model_key="candidate_model",
            )

    def test_auto_selection_is_stage_aware_and_legacy_matches_player(self) -> None:
        # A supervised warm-start is the next train state.  Its best_model is
        # currently identical by contract, but selection must still say train.
        self._write_checkpoint(self.states, v3_stage="tactical_expert_warmstart")
        warm = evaluate_white_defense(
            self.checkpoint, self.eval, self.manifest, self.output
        )
        self.assertEqual(warm["checkpoint_model_key"], "train_model")
        self.assertEqual(
            warm["checkpoint"]["auto_priority"], ["train_model", "best_model"]
        )

        # Legacy/V2 approved checkpoints have no v3_stage.  The desktop game
        # loads best_model, so auto evaluation must do exactly the same even if
        # an unapproved train_model (or stray candidate_model) is also present.
        self._write_checkpoint(self.states, v3_stage=None)
        legacy = evaluate_white_defense(
            self.checkpoint, self.eval, self.manifest, self.output
        )
        self.assertEqual(legacy["checkpoint_model_key"], "best_model")
        self.assertEqual(
            legacy["checkpoint"]["auto_priority"], ["best_model", "train_model"]
        )
        self.assertEqual(legacy["metrics"]["top1_in_safe_set"], 1.0)
        explicit_candidate = evaluate_white_defense(
            self.checkpoint,
            self.eval,
            self.manifest,
            self.output,
            model_key="candidate_model",
        )
        self.assertEqual(explicit_candidate["checkpoint_model_key"], "candidate_model")

        self._write_checkpoint(self.states, v3_stage="external_champion")
        champion = evaluate_white_defense(
            self.checkpoint, self.eval, self.manifest, self.output
        )
        self.assertEqual(champion["checkpoint_model_key"], "best_model")

        self._write_checkpoint(self.states, v3_stage="future_unknown_stage")
        with self.assertRaisesRegex(WhiteDefenseEvaluationError, "unsupported v3_stage"):
            evaluate_white_defense(
                self.checkpoint, self.eval, self.manifest, self.output
            )
        explicit = evaluate_white_defense(
            self.checkpoint,
            self.eval,
            self.manifest,
            self.output,
            model_key="candidate_model",
        )
        self.assertEqual(explicit["checkpoint_model_key"], "candidate_model")

    def test_rejects_wrong_artifact_and_integrity_tampering(self) -> None:
        with self.assertRaisesRegex(WhiteDefenseEvaluationError, "not manifest artifacts.eval"):
            evaluate_white_defense(
                self.checkpoint, self.train, self.manifest, self.output
            )

        original = self.eval.read_bytes()
        self.eval.write_bytes(original + b"tamper")
        with self.assertRaisesRegex(WhiteDefenseEvaluationError, "size mismatch"):
            evaluate_white_defense(
                self.checkpoint, self.eval, self.manifest, self.output
            )
        self.eval.write_bytes(original)

        sidecar = self.manifest.with_suffix(self.manifest.suffix + ".sha256")
        sidecar.write_text(f"{'0' * 64}  {self.manifest.name}\n", encoding="utf-8")
        with self.assertRaisesRegex(WhiteDefenseEvaluationError, "manifest SHA256 mismatch"):
            evaluate_white_defense(
                self.checkpoint, self.eval, self.manifest, self.output
            )

    def test_rejects_missing_training_prohibition_and_record_mismatch(self) -> None:
        self._resign_manifest(
            lambda payload: payload["artifacts"]["eval"].pop("training_prohibition")
        )
        with self.assertRaisesRegex(WhiteDefenseEvaluationError, "training_prohibition"):
            evaluate_white_defense(
                self.checkpoint, self.eval, self.manifest, self.output
            )

        # Re-create clean artifacts, then forge a self-consistent manifest with
        # a declared count that disagrees with the archive.
        dataset, _fixture = _fixture_dataset()
        self.manifest.unlink()
        self.manifest.with_suffix(self.manifest.suffix + ".sha256").unlink()
        self.train.unlink()
        self.eval.unlink()
        write_split_archives(dataset, self.train, self.eval, self.manifest)
        self._resign_manifest(
            lambda payload: payload["artifacts"]["eval"].__setitem__("records", 2)
        )
        with self.assertRaisesRegex(WhiteDefenseEvaluationError, "record count mismatch"):
            evaluate_white_defense(
                self.checkpoint, self.eval, self.manifest, self.output
            )

    def test_rejects_misleading_nonempty_training_prohibition(self) -> None:
        self._resign_manifest(
            lambda payload: payload["artifacts"]["eval"].__setitem__(
                "training_prohibition", "training is allowed"
            )
        )
        with self.assertRaisesRegex(WhiteDefenseEvaluationError, "training_prohibition"):
            evaluate_white_defense(
                self.checkpoint, self.eval, self.manifest, self.output
            )

    def test_rejects_non_eval_split_and_zero_samples_even_when_resigned(self) -> None:
        self._replace_eval_arrays(
            lambda arrays: arrays.__setitem__(
                "split", np.asarray([TRAIN_SPLIT], dtype="<U5")
            )
        )
        with self.assertRaisesRegex(WhiteDefenseEvaluationError, "not a pure eval split"):
            evaluate_white_defense(
                self.checkpoint, self.eval, self.manifest, self.output
            )

        dataset, _fixture = _fixture_dataset()
        for path in (
            self.manifest,
            self.manifest.with_suffix(self.manifest.suffix + ".sha256"),
            self.train,
            self.eval,
        ):
            path.unlink(missing_ok=True)
        write_split_archives(dataset, self.train, self.eval, self.manifest)
        self._replace_eval_arrays(
            lambda arrays: arrays.update(
                {name: value[:0] for name, value in arrays.items()}
            )
        )
        with self.assertRaisesRegex(WhiteDefenseEvaluationError, "zero samples"):
            evaluate_white_defense(
                self.checkpoint, self.eval, self.manifest, self.output
            )


if __name__ == "__main__":
    unittest.main()

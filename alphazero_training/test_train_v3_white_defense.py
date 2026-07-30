from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from alphazero_training.train_v3_selfplay import (
    SourceMixer,
    V3SelfplayConfig,
    WHITE_DEFENSE_SOURCE,
    WHITE_DEFENSE_TRAINING_PROHIBITION,
    WhiteDefenseReplaySource,
    validate_white_defense_manifest,
    weighted_loss,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_hash(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _white_arrays(board_size: int = 5) -> dict[str, np.ndarray]:
    action_count = board_size * board_size
    report_sha = "a" * 64
    states = np.zeros((1, 4, board_size, board_size), dtype=np.uint8)
    # White to move after black's first move: planes are current/opponent/last/side.
    states[0, 1, 0, 0] = 1
    states[0, 2, 0, 0] = 1
    safe = np.zeros((1, action_count), dtype=np.uint8)
    safe[0, 1] = 1
    policies = safe.astype(np.float32)
    zeros = np.zeros_like(safe)
    return {
        "states": states,
        "policies": policies,
        "values": np.zeros(1, dtype=np.float32),
        "policy_weights": np.ones(1, dtype=np.float32),
        "value_weights": np.zeros(1, dtype=np.float32),
        "source": np.asarray(
            [f"white_defense|report={report_sha[:16]}|pair=0|game=1"]
        ),
        "priority": np.ones(1, dtype=np.float32),
        "group_id": np.asarray([f"report={report_sha}|opening={'b' * 64}"]),
        "split": np.asarray(["train"]),
        "report_sha256": np.asarray([report_sha]),
        "opening_sha256": np.asarray(["b" * 64]),
        "pair_index": np.asarray([0], dtype=np.int32),
        "state_hash": np.asarray([hashlib.sha256(states[0].tobytes()).hexdigest()]),
        "candidate_mask": safe.copy(),
        "safe_mask": safe.copy(),
        "vcf_unknown_mask": zeros.copy(),
        "unsafe_immediate_mask": zeros.copy(),
        "unsafe_three_ply_mask": zeros.copy(),
        "unsafe_vcf_mask": zeros.copy(),
    }


def _write_fixture(root: Path) -> tuple[Path, Path, Path]:
    train = root / "white_train.npz"
    evaluation = root / "white_eval.npz"
    manifest_path = root / "white_manifest.json"
    np.savez_compressed(train, **_white_arrays())
    evaluation.write_bytes(b"held-out evaluation archive")
    manifest: dict[str, object] = {
        "schema_version": 1,
        "source": WHITE_DEFENSE_SOURCE,
        "report_sha256": "a" * 64,
        "benchmark_audit": {"provenance_generation": "current6"},
        "rules": {
            "board_size": 5,
            "win_length": 5,
            "freestyle": True,
            "side_to_move": "white",
        },
        "claim_boundary": {
            "label": "bounded_non_loss_within_search_candidates",
        },
        "split": {
            "assigned_before_replay_or_tactical_labelling": True,
            "augmentation": "none",
        },
        "validation": {"train_records": 1, "eval_records": 1},
        "artifacts": {
            "train": {
                "path": str(train),
                "records": 1,
                "bytes": train.stat().st_size,
                "sha256": _sha256(train),
            },
            "eval": {
                "path": str(evaluation),
                "records": 1,
                "bytes": evaluation.stat().st_size,
                "sha256": _sha256(evaluation),
                "training_prohibition": WHITE_DEFENSE_TRAINING_PROHIBITION,
            },
        },
    }
    manifest["manifest_payload_sha256"] = _stable_hash(manifest)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_path.with_suffix(".json.sha256").write_text(
        f"{_sha256(manifest_path)}  {manifest_path.name}\n",
        encoding="utf-8",
    )
    return train, evaluation, manifest_path


def _rewrite_manifest(manifest_path: Path, mutate: object) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("manifest_payload_sha256")
    mutate(manifest)  # type: ignore[operator]
    manifest["manifest_payload_sha256"] = _stable_hash(manifest)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_path.with_suffix(".json.sha256").write_text(
        f"{_sha256(manifest_path)}  {manifest_path.name}\n",
        encoding="utf-8",
    )


class _Source:
    def __init__(self, arrays: dict[str, np.ndarray]):
        self.arrays = arrays

    def __len__(self) -> int:
        return len(self.arrays["states"])

    def sample(self, count: int) -> dict[str, np.ndarray]:
        return {
            name: np.repeat(value, count, axis=0)
            for name, value in self.arrays.items()
            if name
            in {
                "states",
                "policies",
                "values",
                "policy_weights",
                "value_weights",
            }
        }


class V3WhiteDefenseSourceTests(unittest.TestCase):
    def test_manifest_authentication_rejects_conflicting_duplicate_state_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train, _, manifest = _write_fixture(root)
            arrays = {
                name: np.repeat(value, 2, axis=0)
                for name, value in _white_arrays().items()
            }
            arrays["candidate_mask"][:, 2] = 1
            arrays["safe_mask"][0, 2] = 1
            arrays["unsafe_vcf_mask"][1, 2] = 1
            arrays["policies"][0] = 0
            arrays["policies"][0, [1, 2]] = 0.5
            arrays["policies"][1] = 0
            arrays["policies"][1, 1] = 1.0
            np.savez_compressed(train, **arrays)

            def update(payload: dict[str, object]) -> None:
                artifact = payload["artifacts"]["train"]  # type: ignore[index]
                artifact["records"] = 2
                artifact["bytes"] = train.stat().st_size
                artifact["sha256"] = _sha256(train)
                payload["validation"]["train_records"] = 2  # type: ignore[index]

            _rewrite_manifest(manifest, update)
            with self.assertRaisesRegex(ValueError, "conflicting tactical labels"):
                validate_white_defense_manifest(train, manifest, board_size=5)

    def test_safe_set_mass_loss_ignores_internal_redistribution(self) -> None:
        target = torch.tensor([[0.5, 0.5, 0.0, 0.0]])
        values = torch.zeros(1)
        candidates = torch.tensor([[True, True, True, False]])

        def policy_loss(probabilities: list[float], *, safe_set: bool) -> float:
            logits = torch.log(torch.tensor([probabilities], dtype=torch.float32))
            _, loss, _ = weighted_loss(
                logits,
                torch.zeros(1),
                target,
                values,
                torch.ones(1),
                torch.zeros(1),
                safe_set_rows=torch.tensor([safe_set]),
                candidate_masks=candidates,
            )
            return float(loss)

        balanced = policy_loss([0.45, 0.45, 0.099, 0.001], safe_set=True)
        redistributed = policy_loss([0.89, 0.01, 0.099, 0.001], safe_set=True)
        outside_candidate = policy_loss([0.225, 0.225, 0.0495, 0.5005], safe_set=True)
        outside_mass = policy_loss([0.40, 0.40, 0.199, 0.001], safe_set=True)
        self.assertAlmostEqual(balanced, redistributed, places=6)
        self.assertAlmostEqual(balanced, outside_candidate, places=6)
        self.assertGreater(outside_mass, balanced)
        # Ordinary curriculum rows still use full soft-target cross entropy.
        self.assertNotAlmostEqual(
            policy_loss([0.45, 0.45, 0.099, 0.001], safe_set=False),
            policy_loss([0.89, 0.01, 0.099, 0.001], safe_set=False),
            places=4,
        )

    def test_valid_manifest_source_and_four_way_mix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            train, _, manifest = _write_fixture(Path(directory))
            source = WhiteDefenseReplaySource(train, manifest, 5, 17)
            provenance = source.manifest()
            self.assertEqual("white_defense", provenance["name"])
            self.assertEqual("current6", provenance["provenance_generation"])
            self.assertEqual("train", provenance["split"])
            self.assertEqual("group_id", provenance["group_key"])
            self.assertEqual(
                WHITE_DEFENSE_TRAINING_PROHIBITION,
                provenance["eval_training_prohibition"],
            )

            ordinary = _Source(_white_arrays())
            mixer = SourceMixer(
                {
                    "selfplay": ordinary,
                    "ddqk": ordinary,
                    "tactical": ordinary,
                    "white_defense": source,
                },
                {
                    "selfplay": 0.25,
                    "ddqk": 0.25,
                    "tactical": 0.25,
                    "white_defense": 0.25,
                },
                19,
            )
            batch = mixer.sample(8)
            self.assertEqual(8, len(batch["states"]))
            self.assertEqual(
                {"selfplay", "ddqk", "tactical", "white_defense"},
                set(batch["source_names"].tolist()),
            )
            self.assertEqual(2, int(np.count_nonzero(batch["source_names"] == "white_defense")))
            white_rows = batch["source_names"] == "white_defense"
            self.assertFalse(
                np.any(
                    (batch["policies"][white_rows] > 0)
                    & ~batch["candidate_masks"][white_rows]
                )
            )

    def test_eval_artifact_is_refused_even_with_matching_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, evaluation, manifest = _write_fixture(Path(directory))
            with self.assertRaisesRegex(ValueError, "refusing to train.*eval"):
                validate_white_defense_manifest(evaluation, manifest, board_size=5)

    def test_eval_training_prohibition_is_mandatory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            train, _, manifest = _write_fixture(Path(directory))
            _rewrite_manifest(
                manifest,
                lambda payload: payload["artifacts"]["eval"].pop(  # type: ignore[index]
                    "training_prohibition"
                ),
            )
            with self.assertRaisesRegex(ValueError, "training-prohibited"):
                validate_white_defense_manifest(train, manifest, board_size=5)

    def test_manifest_sidecar_and_row_provenance_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            train, _, manifest = _write_fixture(Path(directory))
            manifest.with_suffix(".json.sha256").write_text(
                f"{'0' * 64}  {manifest.name}\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "does not match its sidecar"):
                validate_white_defense_manifest(train, manifest, board_size=5)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train, _, manifest = _write_fixture(root)
            arrays = _white_arrays()
            arrays["report_sha256"] = np.asarray(["c" * 64])
            np.savez_compressed(train, **arrays)
            _rewrite_manifest(
                manifest,
                lambda payload: payload["artifacts"]["train"].update(  # type: ignore[index]
                    {"bytes": train.stat().st_size, "sha256": _sha256(train)}
                ),
            )
            with self.assertRaisesRegex(ValueError, "row provenance"):
                validate_white_defense_manifest(train, manifest, board_size=5)

    def test_default_quota_preserves_three_source_runs(self) -> None:
        config = V3SelfplayConfig()
        self.assertEqual(0.0, config.white_defense_quota)
        ordinary = _Source(_white_arrays())
        mixer = SourceMixer(
            {"selfplay": ordinary, "ddqk": ordinary, "tactical": ordinary},
            config.quotas(),
            31,
        )
        counts = mixer.counts(20)
        self.assertEqual(0, counts["white_defense"])
        self.assertEqual(20, sum(counts.values()))

        with self.assertRaisesRegex(ValueError, "missing sources"):
            SourceMixer(
                {"selfplay": ordinary, "ddqk": ordinary, "tactical": ordinary},
                {"selfplay": 0.7, "ddqk": 0.1, "tactical": 0.1, "white_defense": 0.1},
                37,
            )


if __name__ == "__main__":
    unittest.main()

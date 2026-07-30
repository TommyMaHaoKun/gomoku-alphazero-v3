from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from alphazero_training.benchmark_ddqk import (
    BLACK,
    DEVELOPMENT_MODE,
    WHITE,
    GameRecord,
    build_report,
    evaluation_code_signature,
    stable_json_sha256,
)
from alphazero_training.ddqk_replay_export import export_report


LEGACY4_FILES = (
    "play_agent.py",
    "v3_search.py",
    "tactical_solver.py",
    "benchmark_ddqk.py",
)


def _set_evaluation_code_files(
    report: dict[str, object],
    names: tuple[str, ...],
) -> None:
    evaluation_code = report["signature"]["evaluation_code"]
    original_hashes = evaluation_code["files"]
    file_hashes = {name: original_hashes[name] for name in names}
    evaluation_code["files"] = file_hashes
    evaluation_code["bundle_sha256"] = stable_json_sha256(file_hashes)


def legacy4_v3_report() -> dict[str, object]:
    report = valid_v3_report()
    _set_evaluation_code_files(report, LEGACY4_FILES)
    report["signature"].pop("ddqk_assets")
    for field in (
        "observed_score",
        "hoeffding_bounded_pair_score_lower95",
        "hoeffding_bounded_pair_score_lower95_method",
        "pair_sweep_successes",
        "pair_sweep_trials",
        "observed_pair_sweep_rate",
        "exact_pair_sweep_lower95",
        "exact_pair_sweep_lower95_method",
    ):
        report["summary"].pop(field)
    report["certification"]["requirements"] = {
        "minimum_independent_paired_openings": 600,
        "minimum_observed_score": 0.995,
        "minimum_observed_black_score": 0.99,
        "minimum_observed_white_score": 0.99,
        "requires_conservative_one_sided_95_lower_bound": True,
    }
    return report


def valid_v3_report() -> dict[str, object]:
    moves = [
        [0, 0, BLACK],
        [0, 1, WHITE],
        [1, 0, BLACK],
        [1, 1, WHITE],
        [2, 0, BLACK],
        [2, 1, WHITE],
        [3, 0, BLACK],
        [3, 1, WHITE],
        [4, 0, BLACK],
    ]
    record = GameRecord(
        pair_index=0,
        model_color=WHITE,
        opening=[],
        moves=moves,
        winner=BLACK,
        model_result=0.0,
        plies=len(moves),
        model_seconds=0.4,
        ddqk_seconds=0.5,
        model_moves=4,
        ddqk_moves=5,
        termination="win",
        error=None,
        model_decision_reasons=["mcts"] * 4,
    )
    args = SimpleNamespace(
        checkpoint=Path("candidate.pt"),
        pairs=1,
        certification_mode=DEVELOPMENT_MODE,
        opening_plies=0,
        simulations=64,
        max_moves=361,
        seed=12345,
        workers=1,
    )
    openings: list[list[tuple[int, int]]] = [[]]
    serialized_openings = [[]]
    asset_hashes = {
        "dll.so": "2" * 64,
        "guess_data.txt": "3" * 64,
        "black_calculated_value_19.txt": "4" * 64,
        "white_calculated_value_19.txt": "5" * 64,
    }
    signature = {
        "checkpoint_sha256": "0" * 64,
        "ddqk_source": "DDQK-CONQUER-6-16-15-fixed.py",
        "ddqk_source_sha256": "1" * 64,
        "ddqk_dll_sha256": "2" * 64,
        "ddqk_assets": {
            "files": asset_hashes,
            "bundle_sha256": stable_json_sha256(asset_hashes),
        },
        "ddqk_depth": 7,
        "evaluation_code": evaluation_code_signature(),
        "opening_manifest_sha256": stable_json_sha256(serialized_openings),
        "certification_mode": DEVELOPMENT_MODE,
        "pairs": 1,
        "opening_plies": 0,
        "simulations": 64,
        "max_moves": 361,
        "seed": 12345,
    }
    return build_report(
        args=args,
        signature=signature,
        openings=openings,
        records=[record],
    )


class BenchmarkV3ReplayExportTests(unittest.TestCase):
    def _write(self, root: Path, report: dict[str, object]) -> Path:
        path = root / "benchmark-v3.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        return path

    def test_valid_v3_report_exports_and_keeps_audit_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = self._write(Path(temporary), valid_v3_report())
            arrays, metadata = export_report(path, smoothing=0.0, policy_only=True)
        self.assertEqual(arrays["states"].shape, (5, 4, 19, 19))
        self.assertEqual(float(arrays["value_weights"].sum()), 0.0)
        self.assertEqual(metadata["benchmark_collection"]["format_version"], 3)
        self.assertEqual(
            metadata["benchmark_collection"]["certification_mode"],
            DEVELOPMENT_MODE,
        )
        self.assertEqual(
            metadata["benchmark_collection"]["provenance_generation"],
            "current6",
        )
        self.assertEqual(
            metadata["benchmark_collection"]["ddqk_assets_bundle_sha256"],
            valid_v3_report()["signature"]["ddqk_assets"]["bundle_sha256"],
        )
        self.assertEqual(metadata["provenance_generation"], "current6")

    def test_legacy4_development_report_exports_with_explicit_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = self._write(Path(temporary), legacy4_v3_report())
            arrays, metadata = export_report(path, smoothing=0.0, policy_only=True)
        self.assertEqual(arrays["states"].shape, (5, 4, 19, 19))
        self.assertEqual(
            metadata["benchmark_collection"]["provenance_generation"],
            "legacy4",
        )
        self.assertIsNone(
            metadata["benchmark_collection"]["ddqk_assets_bundle_sha256"]
        )
        self.assertEqual(metadata["provenance_generation"], "legacy4")

    def test_mixed_or_missing_evaluation_code_file_sets_are_rejected(self) -> None:
        original = valid_v3_report()
        current_names = tuple(
            original["signature"]["evaluation_code"]["files"].keys()
        )
        invalid_sets = {
            "mixed": (
                "play_agent.py",
                "v3_search.py",
                "tactical_solver.py",
                "train_alphazero.py",
            ),
            "missing": current_names[:-1],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for label, names in invalid_sets.items():
                with self.subTest(label=label):
                    report = copy.deepcopy(original)
                    _set_evaluation_code_files(report, names)
                    path = self._write(root, report)
                    with self.assertRaisesRegex(ValueError, "file set is invalid"):
                        export_report(path)

    def test_each_provenance_generation_requires_valid_file_hashes(self) -> None:
        for label, report in (
            ("legacy4", legacy4_v3_report()),
            ("current6", valid_v3_report()),
        ):
            with self.subTest(label=label):
                first_name = next(
                    iter(report["signature"]["evaluation_code"]["files"])
                )
                report["signature"]["evaluation_code"]["files"][first_name] = "bad"
                with tempfile.TemporaryDirectory() as temporary:
                    path = self._write(Path(temporary), report)
                    with self.assertRaisesRegex(ValueError, "SHA256 hex digest"):
                        export_report(path)

    def test_each_provenance_generation_rejects_bundle_tampering(self) -> None:
        for label, report in (
            ("legacy4", legacy4_v3_report()),
            ("current6", valid_v3_report()),
        ):
            with self.subTest(label=label):
                report["signature"]["evaluation_code"]["bundle_sha256"] = "f" * 64
                with tempfile.TemporaryDirectory() as temporary:
                    path = self._write(Path(temporary), report)
                    with self.assertRaisesRegex(ValueError, "bundle SHA256"):
                        export_report(path)

    def test_current6_requires_complete_consistent_ddqk_asset_bundle(self) -> None:
        mutations = (
            (
                "missing",
                lambda report: report["signature"].pop("ddqk_assets"),
                "missing the DDQK asset bundle",
            ),
            (
                "wrong file set",
                lambda report: report["signature"]["ddqk_assets"]["files"].pop(
                    "guess_data.txt"
                ),
                "asset file set is invalid",
            ),
            (
                "wrong bundle",
                lambda report: report["signature"]["ddqk_assets"].__setitem__(
                    "bundle_sha256", "f" * 64
                ),
                "asset bundle SHA256",
            ),
            (
                "DLL disagreement",
                lambda report: report["signature"].__setitem__(
                    "ddqk_dll_sha256", "e" * 64
                ),
                "DLL SHA256 disagrees",
            ),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for label, mutate, message in mutations:
                with self.subTest(label=label):
                    report = valid_v3_report()
                    mutate(report)
                    path = self._write(root, report)
                    with self.assertRaisesRegex(ValueError, message):
                        export_report(path)

    def test_game_history_is_replayed_before_results_are_trusted(self) -> None:
        def wrong_result(report: dict[str, object]) -> None:
            report["games"][0]["model_result"] = 1.0

        def wrong_winner(report: dict[str, object]) -> None:
            report["games"][0]["winner"] = WHITE

        def wrong_plies(report: dict[str, object]) -> None:
            report["games"][0]["plies"] += 1

        def repeated_move(report: dict[str, object]) -> None:
            report["games"][0]["moves"][-1][:2] = [0, 0]

        mutations = (
            ("result", wrong_result, "model_result does not match"),
            ("winner", wrong_winner, "winner does not match"),
            ("plies", wrong_plies, "plies do not match"),
            ("legality", repeated_move, "repeated move"),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for label, mutate, message in mutations:
                with self.subTest(label=label):
                    report = valid_v3_report()
                    mutate(report)
                    path = self._write(root, report)
                    with self.assertRaisesRegex(ValueError, message):
                        export_report(path)

    def test_legacy4_provenance_is_never_accepted_for_final_certification(self) -> None:
        report = legacy4_v3_report()
        report["signature"]["certification_mode"] = "final-certification"
        report["certification"]["mode"] = "final-certification"
        with tempfile.TemporaryDirectory() as temporary:
            path = self._write(Path(temporary), report)
            with self.assertRaisesRegex(ValueError, "only for development export"):
                export_report(path)

    def test_v3_provenance_and_results_tampering_is_rejected(self) -> None:
        original = valid_v3_report()
        mutations = (
            (
                "code bundle",
                lambda report: report["signature"]["evaluation_code"].__setitem__(
                    "bundle_sha256", "f" * 64
                ),
            ),
            (
                "opening manifest",
                lambda report: report["openings"].__setitem__(0, [[9, 9]]),
            ),
            (
                "summary",
                lambda report: report["summary"].__setitem__("score", 1.0),
            ),
            (
                "certification",
                lambda report: report["certification"].__setitem__(
                    "status", "benchmark_final_requirements_passed"
                ),
            ),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for label, mutate in mutations:
                with self.subTest(label=label):
                    report = copy.deepcopy(original)
                    mutate(report)
                    path = self._write(root, report)
                    with self.assertRaises(ValueError):
                        export_report(path)


if __name__ == "__main__":
    unittest.main()

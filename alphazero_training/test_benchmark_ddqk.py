from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from alphazero_training.benchmark_ddqk import (
    BLACK,
    DDQK_DECISION_ASSET_FILES,
    EVALUATION_CODE_FILES,
    FINAL_CERTIFICATION_MODE,
    FINAL_MIN_EXACT_PAIR_SWEEP_LOWER95,
    GameRecord,
    WHITE,
    bounded_mean_one_sided_lower95,
    build_report,
    ddqk_asset_signature,
    evaluation_code_signature,
    exact_binomial_one_sided_lower95,
)


class BenchmarkCertificationTests(unittest.TestCase):
    def test_all_win_bootstrap_cannot_be_used_as_confidence_lower_bound(self) -> None:
        lower = bounded_mean_one_sided_lower95([1.0] * 600)
        self.assertGreaterEqual(lower, 0.95)
        self.assertLess(lower, 1.0)

    def test_exact_pair_sweep_lower_bound_distinguishes_600_from_599(self) -> None:
        all_sweeps = exact_binomial_one_sided_lower95(600, 600)
        one_failed_sweep = exact_binomial_one_sided_lower95(599, 600)
        self.assertAlmostEqual(all_sweeps, 0.995019556619639, places=12)
        self.assertGreaterEqual(all_sweeps, FINAL_MIN_EXACT_PAIR_SWEEP_LOWER95)
        self.assertLess(one_failed_sweep, FINAL_MIN_EXACT_PAIR_SWEEP_LOWER95)

    @staticmethod
    def _winning_records(pair_sweep_successes: int) -> list[GameRecord]:
        records: list[GameRecord] = []
        for pair_index in range(600):
            for model_color in (BLACK, WHITE):
                won = pair_index < pair_sweep_successes
                # A failed sweep loses only the white game.  Thus 599/600 still
                # clears every observed-score gate and isolates the exact bound.
                model_result = 1.0 if won or model_color == BLACK else 0.0
                winner = model_color if model_result == 1.0 else BLACK
                records.append(
                    GameRecord(
                        pair_index=pair_index,
                        model_color=model_color,
                        opening=[],
                        moves=[],
                        winner=winner,
                        model_result=model_result,
                        plies=1,
                        model_seconds=0.0,
                        ddqk_seconds=0.0,
                        model_moves=1,
                        ddqk_moves=1,
                        termination="win",
                    )
                )
        return records

    @staticmethod
    def _report_args() -> argparse.Namespace:
        return argparse.Namespace(
            checkpoint=Path("candidate.pt"),
            pairs=600,
            opening_plies=0,
            simulations=256,
            max_moves=361,
            seed=12345,
            workers=1,
            certification_mode=FINAL_CERTIFICATION_MODE,
        )

    def test_final_certification_enforces_exact_pair_sweep_lower_bound(self) -> None:
        args = self._report_args()
        signature = {"ddqk_source": "DDQK.py"}
        openings = [[] for _ in range(600)]
        with patch(
            "alphazero_training.benchmark_ddqk.paired_bootstrap_ci95",
            return_value=[0.0, 1.0],
        ):
            all_wins = build_report(
                args=args,
                signature=signature,
                openings=openings,
                records=self._winning_records(600),
            )
            one_failed_sweep = build_report(
                args=args,
                signature=signature,
                openings=openings,
                records=self._winning_records(599),
            )

        self.assertTrue(all_wins["certification"]["final_certified"])
        self.assertFalse(one_failed_sweep["certification"]["final_certified"])
        self.assertGreaterEqual(one_failed_sweep["summary"]["observed_score"], 0.995)
        self.assertGreaterEqual(
            one_failed_sweep["summary"]["by_color"]["black"]["score"], 0.99
        )
        self.assertGreaterEqual(
            one_failed_sweep["summary"]["by_color"]["white"]["score"], 0.99
        )
        self.assertLess(
            one_failed_sweep["summary"]["exact_pair_sweep_lower95"],
            FINAL_MIN_EXACT_PAIR_SWEEP_LOWER95,
        )

    def test_code_bundle_hash_is_stable_and_content_bound(self) -> None:
        self.assertIn("train_alphazero.py", EVALUATION_CODE_FILES)
        self.assertIn("ddqk_adapter.py", EVALUATION_CODE_FILES)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index, name in enumerate(EVALUATION_CODE_FILES):
                (root / name).write_text(f"source {index}\n", encoding="utf-8")
            first = evaluation_code_signature(root)
            second = evaluation_code_signature(root)
            self.assertEqual(first, second)
            (root / "v3_search.py").write_text("changed\n", encoding="utf-8")
            changed = evaluation_code_signature(root)
            self.assertNotEqual(first["bundle_sha256"], changed["bundle_sha256"])
            self.assertNotEqual(
                first["files"]["v3_search.py"], changed["files"]["v3_search.py"]
            )

    def test_ddqk_asset_bundle_covers_native_and_table_dependencies(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index, name in enumerate(DDQK_DECISION_ASSET_FILES):
                (root / name).write_text(f"asset {index}\n", encoding="utf-8")
            first = ddqk_asset_signature(root)
            self.assertEqual(set(first["files"]), set(DDQK_DECISION_ASSET_FILES))
            (root / "white_calculated_value_19.txt").write_text(
                "changed\n", encoding="utf-8"
            )
            changed = ddqk_asset_signature(root)
            self.assertNotEqual(first["bundle_sha256"], changed["bundle_sha256"])


if __name__ == "__main__":
    unittest.main()

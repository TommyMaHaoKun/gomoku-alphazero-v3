"""Fast mock tests for the auditable DDQK teacher-data pipeline."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

try:
    from .ddqk_replay_export import (
        export_report,
        sha256_file,
        write_dataset_bundle,
    )
    from .generate_ddqk_teacher import (
        BLACK,
        REPORT_TYPE,
        WHITE,
        OpeningMember,
        atomic_write_json,
        build_opening_manifest,
        build_report,
        canonical_sha256,
        is_win,
        load_resume_records,
        manifest_payload,
        opening_board,
        play_teacher_game,
        record_key,
        validate_record,
        worker_failure_record,
    )
except ImportError:  # Allow direct invocation from this directory.
    from ddqk_replay_export import export_report, sha256_file, write_dataset_bundle
    from generate_ddqk_teacher import (
        BLACK,
        REPORT_TYPE,
        WHITE,
        OpeningMember,
        atomic_write_json,
        build_opening_manifest,
        build_report,
        canonical_sha256,
        is_win,
        load_resume_records,
        manifest_payload,
        opening_board,
        play_teacher_game,
        record_key,
        validate_record,
        worker_failure_record,
    )


class ScriptedDDQK:
    """Small deterministic adapter double; it never imports the native DLL."""

    def __init__(self, replies: list[tuple[int, int]], fail_at: int | None = None):
        self.replies = replies
        self.fail_at = fail_at
        self.index = 0
        self.last_engine_error = None
        self.synced: list[tuple[int, int]] = []

    def reset(self) -> None:
        self.index = 0
        self.last_engine_error = None

    def sync_opening(self, opening, starting_player=BLACK):
        if starting_player != BLACK:
            raise AssertionError("teacher openings must start with black")
        self.synced = list(opening)

    def choose_move(self, board, player, last_move=None):
        del board, player, last_move
        if self.fail_at is not None and self.index == self.fail_at:
            raise RuntimeError("mock engine failed")
        move = self.replies[self.index]
        self.index += 1
        return move


def empty_member() -> OpeningMember:
    return build_opening_manifest(
        seed=7, groups=1, games_per_group=1, opening_plies=0
    )[0]


def winning_script() -> list[tuple[int, int]]:
    # Black wins horizontally on ply nine.
    return [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (2, 0),
        (2, 1),
        (3, 0),
        (3, 1),
        (4, 0),
    ]


def fake_signature(groups: int = 1, games_per_group: int = 1) -> dict[str, object]:
    members = build_opening_manifest(
        seed=7,
        groups=groups,
        games_per_group=games_per_group,
        opening_plies=0,
    )
    return {
        "generator_sha256": "3" * 64,
        "ddqk_source": "mock.py",
        "ddqk_source_sha256": "1" * 64,
        "ddqk_dll": "mock-dll.so",
        "ddqk_dll_sha256": "2" * 64,
        "ddqk_depth": 7,
        "board_size": 19,
        "win_length": 5,
        "groups": groups,
        "games_per_group": games_per_group,
        "opening_plies": 0,
        "max_moves": 361,
        "seed": 7,
        "opening_manifest_sha256": canonical_sha256(manifest_payload(members)),
    }


class OpeningManifestTests(unittest.TestCase):
    def test_manifest_is_deterministic_legal_and_grouped(self) -> None:
        kwargs = dict(seed=991, groups=3, games_per_group=2, opening_plies=8)
        first = build_opening_manifest(**kwargs)
        second = build_opening_manifest(**kwargs)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 6)
        self.assertEqual(len({record_key(member) for member in first}), 6)

        for member in first:
            board, next_player = opening_board(member.opening)
            self.assertEqual(int(np.count_nonzero(board)), 8)
            self.assertEqual(next_player, BLACK)
            self.assertEqual(member.pair_index, member.group_index)
            for ply, (x, y) in enumerate(member.opening):
                player = BLACK if ply % 2 == 0 else WHITE
                self.assertFalse(is_win(board, x, y, player))

        for group_index in range(3):
            group = [m for m in first if m.group_index == group_index]
            self.assertEqual(len({m.group_seed for m in group}), 1)
            self.assertEqual(len({m.game_seed for m in group}), 2)
            self.assertEqual(len({m.symmetry for m in group}), 2)

    def test_manifest_rejects_more_than_eight_correlated_symmetries(self) -> None:
        with self.assertRaisesRegex(ValueError, "games_per_group"):
            build_opening_manifest(
                seed=1, groups=1, games_per_group=9, opening_plies=4
            )


class TeacherGameTests(unittest.TestCase):
    def test_complete_win_has_reconstructable_terminal_and_checksum(self) -> None:
        member = empty_member()
        record = play_teacher_game(
            ScriptedDDQK(winning_script()), member, max_moves=361
        )
        self.assertTrue(record.complete)
        self.assertIsNone(record.error)
        self.assertEqual(record.termination, "win")
        self.assertEqual(record.winner, BLACK)
        self.assertEqual(record.plies, 9)
        self.assertEqual(record.ddqk_moves, 9)
        validate_record(record, member)

    def test_engine_error_is_retained_but_never_marked_complete(self) -> None:
        member = empty_member()
        record = play_teacher_game(
            ScriptedDDQK(winning_script(), fail_at=3), member, max_moves=361
        )
        self.assertFalse(record.complete)
        self.assertEqual(record.termination, "engine_error")
        self.assertIn("mock engine failed", record.error or "")
        self.assertEqual(record.winner, 0)
        validate_record(record, member)

    def test_move_limit_is_explicit_truncation(self) -> None:
        member = empty_member()
        record = play_teacher_game(
            ScriptedDDQK(winning_script()), member, max_moves=4
        )
        self.assertFalse(record.complete)
        self.assertEqual(record.termination, "truncated")
        self.assertIsNone(record.error)
        validate_record(record, member)

    def test_out_of_game_worker_failure_is_an_auditable_unusable_record(self) -> None:
        member = empty_member()
        record = worker_failure_record(member, RuntimeError("spawn broke"))
        self.assertFalse(record.complete)
        self.assertEqual(record.termination, "worker_error")
        self.assertIn("spawn broke", record.error or "")
        validate_record(record, member)


class ResumeTests(unittest.TestCase):
    def test_atomic_partial_report_round_trips_and_rejects_tampering(self) -> None:
        member = empty_member()
        record = play_teacher_game(
            ScriptedDDQK(winning_script()), member, max_moves=361
        )
        signature = fake_signature()
        report = build_report(
            signature=signature, members=[member], records=[record], workers=3
        )
        self.assertEqual(report["format_version"], 2)
        self.assertEqual(report["report_type"], REPORT_TYPE)
        self.assertEqual(report["summary"]["collection_status"], "complete")

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "teacher.json"
            atomic_write_json(path, report)
            self.assertFalse(path.with_name(path.name + ".tmp").exists())
            restored = load_resume_records(
                path, signature=signature, members=[member]
            )
            self.assertEqual(list(restored), [(0, 0)])

            tampered = json.loads(path.read_text(encoding="utf-8"))
            tampered["games"][0]["moves"][0][0] = 7
            # Re-hashing only the recorded board cannot hide move corruption;
            # strict replay validation still rejects the report.
            path.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "final board|winner"):
                load_resume_records(path, signature=signature, members=[member])

    def test_resume_rejects_signature_change(self) -> None:
        member = empty_member()
        record = play_teacher_game(
            ScriptedDDQK(winning_script()), member, max_moves=361
        )
        signature = fake_signature()
        report = build_report(
            signature=signature, members=[member], records=[record], workers=1
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "teacher.json"
            atomic_write_json(path, report)
            changed = dict(signature)
            changed["ddqk_dll_sha256"] = "9" * 64
            with self.assertRaisesRegex(ValueError, "signature"):
                load_resume_records(path, signature=changed, members=[member])


class TeacherExportTests(unittest.TestCase):
    def test_export_uses_both_teachers_and_true_terminal_values(self) -> None:
        member = empty_member()
        record = play_teacher_game(
            ScriptedDDQK(winning_script()), member, max_moves=361
        )
        signature = fake_signature()
        report = build_report(
            signature=signature, members=[member], records=[record], workers=1
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "teacher.json"
            atomic_write_json(path, report)
            arrays, metadata = export_report(path, smoothing=0.0)

        self.assertEqual(arrays["states"].shape, (9, 4, 19, 19))
        self.assertEqual(arrays["policies"].shape, (9, 361))
        np.testing.assert_array_equal(
            arrays["values"],
            np.asarray([1, -1, 1, -1, 1, -1, 1, -1, 1], dtype=np.float32),
        )
        np.testing.assert_array_equal(arrays["value_weights"], np.ones(9))
        np.testing.assert_array_equal(arrays["pair_index"], np.zeros(9))
        self.assertEqual(metadata["source"], "ddqk_teacher_selfplay_complete_games")
        self.assertEqual(metadata["ddqk_source_sha256"], "1" * 64)
        self.assertEqual(metadata["ddqk_dll_sha256"], "2" * 64)

    def test_export_rejects_false_complete_teacher_game(self) -> None:
        member = empty_member()
        record = play_teacher_game(
            ScriptedDDQK(winning_script()), member, max_moves=361
        )
        report = build_report(
            signature=fake_signature(), members=[member], records=[record], workers=1
        )
        report["games"][0]["complete"] = False
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "teacher.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "completion flag"):
                export_report(path)

    def test_export_rejects_partial_by_default_and_audits_opt_in(self) -> None:
        members = build_opening_manifest(
            seed=7, groups=2, games_per_group=1, opening_plies=0
        )
        record = play_teacher_game(
            ScriptedDDQK(winning_script()), members[0], max_moves=361
        )
        report = build_report(
            signature=fake_signature(groups=2, games_per_group=1),
            members=members,
            records=[record],
            workers=1,
        )
        self.assertEqual(report["summary"]["collection_status"], "in_progress")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "partial.json"
            atomic_write_json(path, report)
            with self.assertRaisesRegex(ValueError, "collection_status"):
                export_report(path)
            arrays, metadata = export_report(path, allow_partial=True)

        self.assertEqual(len(arrays["states"]), 9)
        self.assertTrue(metadata["allow_partial"])
        self.assertEqual(
            metadata["teacher_collection"],
            {
                "collection_status": "in_progress",
                "expected_games": 2,
                "recorded_games": 1,
                "usable_complete_games": 1,
                "failed_or_truncated_games": 0,
                "complete_groups": 1,
            },
        )

    def test_export_rejects_manifest_pair_history_and_provenance_tampering(self) -> None:
        member = empty_member()
        record = play_teacher_game(
            ScriptedDDQK(winning_script()), member, max_moves=361
        )
        original = build_report(
            signature=fake_signature(), members=[member], records=[record], workers=1
        )
        mutations = (
            ("pair", lambda report: report["games"][0].__setitem__("pair_index", 99)),
            ("move accounting", lambda report: report["games"][0].__setitem__("ddqk_moves", 0)),
            (
                "final-board checksum",
                lambda report: report["games"][0].__setitem__(
                    "final_board_sha256", "0" * 64
                ),
            ),
            (
                "manifest recipe",
                lambda report: report["opening_manifest"][0]["members"][0].__setitem__(
                    "game_seed", 123
                ),
            ),
            (
                "engine provenance",
                lambda report: report["signature"].__setitem__(
                    "ddqk_dll_sha256", "not-a-sha"
                ),
            ),
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "tampered.json"
            for label, mutate in mutations:
                with self.subTest(label=label):
                    report = copy.deepcopy(original)
                    mutate(report)
                    path.write_text(json.dumps(report), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        export_report(path)

    def test_old_benchmark_v2_remains_compatible(self) -> None:
        report = {
            "format_version": 2,
            "signature": {
                "ddqk_source_sha256": "1" * 64,
                "ddqk_dll_sha256": "2" * 64,
            },
            "games": [
                {
                    "pair_index": 4,
                    "model_color": WHITE,
                    "opening": [],
                    "moves": [
                        [x, y, BLACK if ply % 2 == 0 else WHITE]
                        for ply, (x, y) in enumerate(winning_script())
                    ],
                    "winner": BLACK,
                    "termination": "win",
                    "error": None,
                }
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "benchmark-v2.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            arrays, metadata = export_report(path, smoothing=0.0)

        self.assertEqual(arrays["states"].shape, (5, 4, 19, 19))
        np.testing.assert_array_equal(arrays["pair_index"], np.full(5, 4))
        self.assertEqual(metadata["source"], "ddqk_benchmark_complete_games")


class DatasetBundleTests(unittest.TestCase):
    def test_sidecar_is_sha_bound_and_default_is_no_clobber(self) -> None:
        arrays = {
            "states": np.zeros((1, 4, 19, 19), dtype=np.uint8),
            "policies": np.full((1, 361), 1.0 / 361, dtype=np.float16),
            "values": np.zeros(1, dtype=np.float32),
        }
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "dataset.npz"
            committed = write_dataset_bundle(output, arrays, {"samples": 1})
            sidecar = json.loads(
                output.with_suffix(".npz.json").read_text(encoding="utf-8")
            )
            self.assertEqual(committed, sidecar)
            self.assertEqual(sidecar["dataset_sha256"], sha256_file(output))
            self.assertEqual(sidecar["dataset_bytes"], output.stat().st_size)
            with self.assertRaises(FileExistsError):
                write_dataset_bundle(output, arrays, {"samples": 1})
            overwritten = write_dataset_bundle(
                output, arrays, {"samples": 1, "revision": 2}, overwrite=True
            )
            self.assertEqual(overwritten["dataset_sha256"], sha256_file(output))
            self.assertEqual(overwritten["revision"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)

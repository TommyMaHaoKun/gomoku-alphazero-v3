from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pytest

from alphazero_training.build_rapfi_loss_curriculum import build_curriculum
from alphazero_training.build_rapfi_joint_dataset import build_joint_dataset
from alphazero_training.compare_rapfi_reports import exact_two_sided_sign_p
from alphazero_training.rapfi_distill import (
    BLACK,
    WHITE,
    GameRecord,
    MoveRecord,
    _write_ai_loss_library,
    canonical_sha256,
    export_dataset,
    validate_record,
)


WINNING_MOVES = [
    (0, 0, BLACK),
    (0, 1, WHITE),
    (1, 0, BLACK),
    (1, 1, WHITE),
    (2, 0, BLACK),
    (2, 1, WHITE),
    (3, 0, BLACK),
    (3, 1, WHITE),
    (4, 0, BLACK),
]


def _complete_record(*, pair_index: int, student_color: int) -> GameRecord:
    records: list[MoveRecord] = []
    for ply, (x, y, player) in enumerate(WINNING_MOVES):
        source = "student" if player == student_color else "rapfi"
        # Give one student decision a different, still-legal teacher target.
        teacher_x, teacher_y = x, y
        if ply == 2 and source == "student":
            teacher_x, teacher_y = 2, 0
        elif ply == 3 and source == "student":
            teacher_x, teacher_y = 2, 2
        disagreed = (x, y) != (teacher_x, teacher_y)
        records.append(
            MoveRecord(
                ply=ply,
                x=x,
                y=y,
                player=player,
                source=source,
                seconds=0.01,
                decision_reason="test" if source == "student" else None,
                teacher_x=teacher_x,
                teacher_y=teacher_y,
                teacher_seconds=0.02,
                student_disagreed=disagreed,
            )
        )
    result = 1.0 if student_color == BLACK else 0.0
    record = GameRecord(
        pair_index=pair_index,
        student_color=student_color,
        opening=[],
        moves=records,
        winner=BLACK,
        student_result=result,
        termination="win",
    )
    payload = asdict(record)
    payload.pop("record_sha256")
    record.record_sha256 = canonical_sha256(payload)
    return record


def test_export_uses_teacher_actions_on_student_turns(tmp_path: Path) -> None:
    win = _complete_record(pair_index=4, student_color=BLACK)
    loss = _complete_record(pair_index=4, student_color=WHITE)
    output = tmp_path / "rapfi.npz"

    metadata = export_dataset([win, loss], output)
    with np.load(output) as data:
        assert len(data["states"]) == 18
        assert set(data["group_id"].tolist()) == {4}
        assert int(data["student_turn"].sum()) == 9
        assert int(data["student_disagreed"].sum()) == 2
        # Ply 2 was played at (1,0), but its teacher target is (2,0).
        target_action = int(np.argmax(data["policies"][2]))
        assert target_action == 2
        assert float(data["priority"][2]) == 2.0
        assert int(data["ai_loss"].sum()) == 9
        assert np.all(data["value_weights"] == 0)
    assert metadata["samples"] == 18
    assert metadata["student_disagreement_samples"] == 2


def test_record_hash_and_teacher_legality_are_replayed() -> None:
    record = _complete_record(pair_index=1, student_color=BLACK)
    validate_record(record)
    record.moves[0].teacher_x = 99
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_record(record)


def test_only_losses_enter_pending_library(tmp_path: Path) -> None:
    win = _complete_record(pair_index=1, student_color=BLACK)
    loss = _complete_record(pair_index=1, student_color=WHITE)
    assert _write_ai_loss_library([win, loss], tmp_path) == 1
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    assert "student2" in files[0].name


def test_exact_sign_test_is_conservative() -> None:
    assert exact_two_sided_sign_p(10, 5) == pytest.approx(0.3017578125)
    assert exact_two_sided_sign_p(0, 0) == 1.0
    assert exact_two_sided_sign_p(6, 0) == pytest.approx(0.03125)


def test_loss_curriculum_keeps_only_loss_disagreements_and_groups_symmetries(
    tmp_path: Path,
) -> None:
    win = _complete_record(pair_index=8, student_color=BLACK)
    loss = _complete_record(pair_index=8, student_color=WHITE)
    report = {
        "format_version": 2,
        "report_type": "rapfi_student_distillation",
        "complete": True,
        "signature": {"checkpoint_sha256": "a" * 64},
        "games": [asdict(win), asdict(loss)],
    }
    report["report_sha256"] = canonical_sha256(report)
    report_path = tmp_path / "games.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    output = tmp_path / "losses.npz"
    metadata = build_curriculum(report_path, output, symmetries=8)
    assert metadata["base_positions"] == 1
    assert metadata["white_base_positions"] == 1
    assert metadata["black_base_positions"] == 0
    with np.load(output) as data:
        assert len(data["states"]) == 8
        assert set(data["group_id"].tolist()) == {8}
        assert np.all(data["student_color"] == WHITE)
        assert np.all(data["mistake_action"] != data["teacher_action"])


def test_joint_dataset_adds_terminal_values_and_mistake_pairs(tmp_path: Path) -> None:
    win = _complete_record(pair_index=8, student_color=BLACK)
    loss = _complete_record(pair_index=8, student_color=WHITE)
    report = {
        "format_version": 2,
        "report_type": "rapfi_student_distillation",
        "complete": True,
        "signature": {"checkpoint_sha256": "a" * 64, "pairs": 1},
        "games": [asdict(win), asdict(loss)],
    }
    report["report_sha256"] = canonical_sha256(report)
    report_path = tmp_path / "games.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    output = tmp_path / "joint.npz"
    metadata = build_joint_dataset([report_path], output)
    assert metadata["completed_games"] == 2
    assert metadata["groups"] == 1
    with np.load(output) as data:
        assert len(data["states"]) == 18
        assert np.all(data["value_weights"] > 0)
        assert set(data["values"].tolist()) == {-1.0, 1.0}
        assert int(np.sum(data["mistake_action"] >= 0)) == 2
        assert int(np.sum(data["student_win"])) == 9
        assert int(np.sum(data["student_loss"])) == 9
        assert np.all(
            data["mistake_action"][data["mistake_action"] >= 0]
            != data["teacher_action"][data["mistake_action"] >= 0]
        )

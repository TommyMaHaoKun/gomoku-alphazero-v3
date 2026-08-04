"""Build a grouped hard-negative curriculum from authenticated Rapfi losses."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

try:
    from .rapfi_distill import (
        BLACK,
        WHITE,
        BOARD_SIZE,
        _record_from_dict,
        canonical_sha256,
        sha256_file,
        validate_record,
    )
except ImportError:
    from rapfi_distill import (  # type: ignore[no-redef]
        BLACK,
        WHITE,
        BOARD_SIZE,
        _record_from_dict,
        canonical_sha256,
        sha256_file,
        validate_record,
    )


FORMAT_VERSION = 1


def _transform_xy(x: int, y: int, symmetry: int) -> tuple[int, int]:
    limit = BOARD_SIZE - 1
    if symmetry >= 4:
        x = limit - x
        symmetry -= 4
    for _ in range(symmetry):
        x, y = limit - y, x
    return x, y


def _transform_state(state: np.ndarray, symmetry: int) -> np.ndarray:
    transformed = np.zeros_like(state)
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            tx, ty = _transform_xy(x, y, symmetry)
            transformed[:, ty, tx] = state[:, y, x]
    return transformed


def _encode_state(
    board: np.ndarray,
    player: int,
    last_action: int,
) -> np.ndarray:
    state = np.zeros((4, BOARD_SIZE, BOARD_SIZE), dtype=np.uint8)
    opponent = WHITE if player == BLACK else BLACK
    state[0] = board == player
    state[1] = board == opponent
    if last_action >= 0:
        y, x = divmod(last_action, BOARD_SIZE)
        state[2, y, x] = 1
    if player == BLACK:
        state[3].fill(1)
    return state


def load_authenticated_report(path: Path):
    report = json.loads(path.read_text(encoding="utf-8"))
    recorded_hash = report.pop("report_sha256", None)
    if recorded_hash != canonical_sha256(report):
        raise ValueError("Rapfi report hash mismatch")
    if report.get("report_type") != "rapfi_student_distillation":
        raise ValueError("wrong Rapfi report type")
    if not report.get("complete"):
        raise ValueError("Rapfi report is incomplete")
    records = [_record_from_dict(raw) for raw in report.get("games", [])]
    for record in records:
        validate_record(record)
    return report, records, recorded_hash


def build_curriculum(
    report_path: Path,
    output: Path,
    *,
    symmetries: int = 8,
) -> dict[str, object]:
    if symmetries not in (1, 8):
        raise ValueError("symmetries must be 1 or 8")
    report, records, report_hash = load_authenticated_report(report_path)

    states: list[np.ndarray] = []
    policies: list[np.ndarray] = []
    priorities: list[float] = []
    mistake_actions: list[int] = []
    teacher_actions: list[int] = []
    group_ids: list[int] = []
    game_indices: list[int] = []
    ply_indices: list[int] = []
    student_colors: list[int] = []
    moves_to_end: list[int] = []
    sources: list[str] = []
    base_positions = 0
    base_white = 0
    base_black = 0
    critical_positions = 0

    for game_index, record in enumerate(records):
        if record.error is not None or record.student_result != 0.0:
            continue
        board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        last_action = -1
        for move in record.moves:
            if move.source == "student" and move.student_disagreed:
                if move.teacher_x is None or move.teacher_y is None:
                    raise ValueError("student disagreement is missing its teacher action")
                state = _encode_state(board, move.player, last_action)
                teacher_action = move.teacher_y * BOARD_SIZE + move.teacher_x
                mistake_action = move.y * BOARD_SIZE + move.x
                distance = len(record.moves) - 1 - move.ply
                base_priority = 2.0 if record.student_color == WHITE else 1.0
                if distance <= 8:
                    base_priority *= 2.0
                    critical_positions += 1
                elif distance <= 16:
                    base_priority *= 1.5
                base_positions += 1
                base_white += int(record.student_color == WHITE)
                base_black += int(record.student_color == BLACK)

                for symmetry in range(symmetries):
                    transformed_state = _transform_state(state, symmetry)
                    teacher_x, teacher_y = _transform_xy(
                        move.teacher_x, move.teacher_y, symmetry
                    )
                    mistake_x, mistake_y = _transform_xy(move.x, move.y, symmetry)
                    transformed_teacher = teacher_y * BOARD_SIZE + teacher_x
                    transformed_mistake = mistake_y * BOARD_SIZE + mistake_x
                    policy = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32)
                    policy[transformed_teacher] = 1.0
                    states.append(transformed_state)
                    policies.append(policy)
                    priorities.append(base_priority)
                    mistake_actions.append(transformed_mistake)
                    teacher_actions.append(transformed_teacher)
                    group_ids.append(record.pair_index)
                    game_indices.append(game_index)
                    ply_indices.append(move.ply)
                    student_colors.append(record.student_color)
                    moves_to_end.append(distance)
                    sources.append(
                        "rapfi_loss_correction|"
                        f"pair={record.pair_index}|game={game_index}|"
                        f"ply={move.ply}|sym={symmetry}"
                    )
            board[move.y, move.x] = move.player
            last_action = move.y * BOARD_SIZE + move.x

    if not states:
        raise ValueError("report contains no student disagreements in completed losses")
    count = len(states)
    arrays = {
        "states": np.stack(states),
        "policies": np.stack(policies),
        "values": np.zeros(count, dtype=np.float32),
        "priority": np.asarray(priorities, dtype=np.float32),
        "policy_weights": np.ones(count, dtype=np.float32),
        "value_weights": np.zeros(count, dtype=np.float32),
        "mistake_action": np.asarray(mistake_actions, dtype=np.int16),
        "teacher_action": np.asarray(teacher_actions, dtype=np.int16),
        "group_id": np.asarray(group_ids, dtype=np.int32),
        "game_index": np.asarray(game_indices, dtype=np.int32),
        "ply_index": np.asarray(ply_indices, dtype=np.int16),
        "student_color": np.asarray(student_colors, dtype=np.int8),
        "moves_to_end": np.asarray(moves_to_end, dtype=np.int16),
        "source": np.asarray(sources),
        "split": np.full(count, "train", dtype="<U5"),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".tmp-{os.getpid()}.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, output)
    metadata: dict[str, object] = {
        "format_version": FORMAT_VERSION,
        "dataset_kind": "rapfi_loss_hard_negative",
        "source_report": str(report_path.resolve()),
        "source_report_sha256": report_hash,
        "source_checkpoint_sha256": report["signature"]["checkpoint_sha256"],
        "samples": count,
        "base_positions": base_positions,
        "white_base_positions": base_white,
        "black_base_positions": base_black,
        "critical_within_8_plies": critical_positions,
        "symmetries": symmetries,
        "groups": len(set(group_ids)),
        "npz_sha256": sha256_file(output),
        "priority": {
            "white_multiplier": 2.0,
            "loss_within_8_plies_multiplier": 2.0,
            "loss_within_16_plies_multiplier": 1.5,
        },
        "arrays": {name: list(value.shape) for name, value in arrays.items()},
    }
    sidecar = output.with_suffix(output.suffix + ".json")
    temporary_json = sidecar.with_name(sidecar.name + f".tmp-{os.getpid()}")
    with temporary_json.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary_json, sidecar)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--symmetries", type=int, choices=(1, 8), default=8)
    args = parser.parse_args()
    metadata = build_curriculum(
        args.report.resolve(), args.output.resolve(), symmetries=args.symmetries
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

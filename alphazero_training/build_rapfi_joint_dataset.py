"""Build grouped policy/value supervision from authenticated Rapfi matches."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

from .build_rapfi_loss_curriculum import (
    _encode_state,
    _transform_state,
    _transform_xy,
    load_authenticated_report,
)
from .rapfi_distill import BLACK, BOARD_SIZE, EMPTY, WHITE, sha256_file


FORMAT_VERSION = 1


def build_joint_dataset(
    report_paths: list[Path],
    output: Path,
    *,
    symmetries: int = 1,
) -> dict[str, object]:
    if not report_paths:
        raise ValueError("at least one Rapfi report is required")
    if symmetries not in (1, 8):
        raise ValueError("symmetries must be 1 or 8")

    states: list[np.ndarray] = []
    policies: list[np.ndarray] = []
    values: list[float] = []
    priorities: list[float] = []
    value_weights: list[float] = []
    mistake_actions: list[int] = []
    teacher_actions: list[int] = []
    group_ids: list[int] = []
    report_indices: list[int] = []
    pair_indices: list[int] = []
    game_indices: list[int] = []
    ply_indices: list[int] = []
    student_colors: list[int] = []
    players: list[int] = []
    moves_to_end: list[int] = []
    student_turns: list[int] = []
    student_losses: list[int] = []
    student_wins: list[int] = []

    report_metadata: list[dict[str, object]] = []
    group_by_source: dict[tuple[int, int], int] = {}
    next_group = 0
    completed_games = 0
    base_positions = 0
    disagreement_positions = 0

    for report_index, report_path in enumerate(report_paths):
        report, records, report_hash = load_authenticated_report(report_path)
        report_metadata.append(
            {
                "path": str(report_path.resolve()),
                "report_sha256": report_hash,
                "checkpoint_sha256": report["signature"]["checkpoint_sha256"],
                "pairs": report["signature"]["pairs"],
            }
        )
        for game_index, record in enumerate(records):
            if record.error is not None or record.termination not in (
                "win",
                "full_board_draw",
            ):
                continue
            completed_games += 1
            group_key = (report_index, record.pair_index)
            if group_key not in group_by_source:
                group_by_source[group_key] = next_group
                next_group += 1
            group_id = group_by_source[group_key]
            board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
            last_action = -1
            is_student_loss = record.student_result == 0.0
            is_student_win = record.student_result == 1.0
            for move in record.moves:
                if move.teacher_x is not None and move.teacher_y is not None:
                    state = _encode_state(board, move.player, last_action)
                    teacher_action = move.teacher_y * BOARD_SIZE + move.teacher_x
                    actual_action = move.y * BOARD_SIZE + move.x
                    is_student_turn = move.source == "student"
                    has_mistake = is_student_turn and move.student_disagreed
                    distance = len(record.moves) - 1 - move.ply
                    if record.winner == EMPTY:
                        value = 0.0
                    else:
                        value = 1.0 if record.winner == move.player else -1.0
                    # Terminal outcomes are exact for the played trajectory but
                    # less informative far from the terminal position.
                    value_weight = 0.35 + 0.65 * math.exp(-distance / 24.0)
                    priority = 1.0
                    if is_student_turn:
                        priority *= 1.25
                    if has_mistake:
                        priority *= 3.0
                        disagreement_positions += 1
                    if is_student_loss:
                        priority *= 1.25
                    if is_student_loss and record.student_color == WHITE:
                        priority *= 1.5
                    if is_student_win:
                        priority *= 2.0
                    if is_student_win and record.student_color == WHITE:
                        # White wins against Rapfi are exceptionally rare and
                        # must not disappear inside the much larger loss pool.
                        priority *= 4.0
                    if distance <= 12:
                        priority *= 1.25
                    base_positions += 1

                    for symmetry in range(symmetries):
                        transformed_state = _transform_state(state, symmetry)
                        teacher_x, teacher_y = _transform_xy(
                            move.teacher_x, move.teacher_y, symmetry
                        )
                        transformed_teacher = teacher_y * BOARD_SIZE + teacher_x
                        if has_mistake:
                            mistake_x, mistake_y = _transform_xy(
                                move.x, move.y, symmetry
                            )
                            transformed_mistake = (
                                mistake_y * BOARD_SIZE + mistake_x
                            )
                        else:
                            transformed_mistake = -1
                        policy = np.zeros(
                            BOARD_SIZE * BOARD_SIZE, dtype=np.uint8
                        )
                        policy[transformed_teacher] = 1
                        states.append(transformed_state)
                        policies.append(policy)
                        values.append(value)
                        priorities.append(priority)
                        value_weights.append(value_weight)
                        mistake_actions.append(transformed_mistake)
                        teacher_actions.append(transformed_teacher)
                        group_ids.append(group_id)
                        report_indices.append(report_index)
                        pair_indices.append(record.pair_index)
                        game_indices.append(game_index)
                        ply_indices.append(move.ply)
                        student_colors.append(record.student_color)
                        players.append(move.player)
                        moves_to_end.append(distance)
                        student_turns.append(int(is_student_turn))
                        student_losses.append(int(is_student_loss))
                        student_wins.append(int(is_student_win))
                board[move.y, move.x] = move.player
                last_action = move.y * BOARD_SIZE + move.x

    if not states:
        raise ValueError("reports contain no completed Rapfi decisions")
    count = len(states)
    arrays = {
        "states": np.stack(states),
        "policies": np.stack(policies),
        "values": np.asarray(values, dtype=np.float32),
        "priority": np.asarray(priorities, dtype=np.float32),
        "policy_weights": np.ones(count, dtype=np.float32),
        "value_weights": np.asarray(value_weights, dtype=np.float32),
        "mistake_action": np.asarray(mistake_actions, dtype=np.int16),
        "teacher_action": np.asarray(teacher_actions, dtype=np.int16),
        "group_id": np.asarray(group_ids, dtype=np.int32),
        "report_index": np.asarray(report_indices, dtype=np.int16),
        "pair_index": np.asarray(pair_indices, dtype=np.int32),
        "game_index": np.asarray(game_indices, dtype=np.int32),
        "ply_index": np.asarray(ply_indices, dtype=np.int16),
        "student_color": np.asarray(student_colors, dtype=np.int8),
        "player": np.asarray(players, dtype=np.int8),
        "moves_to_end": np.asarray(moves_to_end, dtype=np.int16),
        "student_turn": np.asarray(student_turns, dtype=np.uint8),
        "student_loss": np.asarray(student_losses, dtype=np.uint8),
        "student_win": np.asarray(student_wins, dtype=np.uint8),
        "source": np.full(count, "rapfi_joint", dtype="<U11"),
        "split": np.full(count, "train", dtype="<U5"),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".tmp-{os.getpid()}.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, output)
    metadata: dict[str, object] = {
        "format_version": FORMAT_VERSION,
        "dataset_kind": "rapfi_joint_policy_value_distillation",
        "reports": report_metadata,
        "samples": count,
        "base_positions": base_positions,
        "completed_games": completed_games,
        "groups": next_group,
        "student_disagreement_base_positions": disagreement_positions,
        "positive_value_samples": int(np.sum(arrays["values"] > 0)),
        "negative_value_samples": int(np.sum(arrays["values"] < 0)),
        "draw_value_samples": int(np.sum(arrays["values"] == 0)),
        "student_turn_samples": int(np.sum(arrays["student_turn"])),
        "student_loss_samples": int(np.sum(arrays["student_loss"])),
        "student_win_samples": int(np.sum(arrays["student_win"])),
        "white_student_win_samples": int(
            np.sum(
                (arrays["student_color"] == WHITE)
                & (arrays["student_win"] == 1)
            )
        ),
        "white_student_samples": int(np.sum(arrays["student_color"] == WHITE)),
        "mean_value_weight": float(np.mean(arrays["value_weights"])),
        "symmetries": symmetries,
        "value_target": "terminal_result_from_side_to_move",
        "value_weight": "0.35 + 0.65 * exp(-moves_to_end / 24)",
        "npz_sha256": sha256_file(output),
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
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--symmetries", type=int, choices=(1, 8), default=1)
    args = parser.parse_args()
    result = build_joint_dataset(
        [path.resolve() for path in args.report],
        args.output.resolve(),
        symmetries=args.symmetries,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

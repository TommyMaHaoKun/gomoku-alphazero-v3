"""Roll out Rapfi continuations after replacing recorded student mistakes."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
import json
import multiprocessing
import os
from pathlib import Path
import time

import numpy as np

from .build_rapfi_loss_curriculum import _encode_state, load_authenticated_report
from .rapfi_adapter import RapfiAdapter
from .rapfi_distill import (
    BLACK,
    BOARD_SIZE,
    EMPTY,
    WHITE,
    canonical_sha256,
    is_win,
    other,
    sha256_file,
)


FORMAT_VERSION = 1
REPORT_TYPE = "rapfi_counterfactual_correction"
_WORKER_ENGINE: RapfiAdapter | None = None


@dataclass
class BranchTask:
    task_index: int
    pair_index: int
    game_index: int
    source_ply: int
    student_color: int
    player: int
    history_before: list[list[int]]
    teacher_x: int
    teacher_y: int
    mistake_x: int
    mistake_y: int
    original_moves_to_end: int
    priority: float


@dataclass
class BranchMove:
    branch_ply: int
    x: int
    y: int
    player: int
    source: str
    seconds: float


@dataclass
class BranchRecord:
    task: BranchTask
    moves: list[BranchMove]
    winner: int
    termination: str
    error: str | None = None
    record_sha256: str = field(default="")


def extract_tasks(
    report_path: Path,
    *,
    max_tasks: int,
    seed: int,
) -> tuple[list[BranchTask], str]:
    _, records, report_hash = load_authenticated_report(report_path)
    tasks: list[BranchTask] = []
    for game_index, record in enumerate(records):
        if record.error is not None or record.student_result != 0.0:
            continue
        history: list[list[int]] = []
        for move in record.moves:
            if move.source == "student" and move.student_disagreed:
                if move.teacher_x is None or move.teacher_y is None:
                    raise ValueError("student disagreement is missing teacher action")
                distance = len(record.moves) - 1 - move.ply
                priority = 1.0
                if record.student_color == WHITE:
                    priority *= 2.0
                if distance <= 8:
                    priority *= 2.0
                elif distance <= 16:
                    priority *= 1.5
                tasks.append(
                    BranchTask(
                        task_index=len(tasks),
                        pair_index=record.pair_index,
                        game_index=game_index,
                        source_ply=move.ply,
                        student_color=record.student_color,
                        player=move.player,
                        history_before=[row.copy() for row in history],
                        teacher_x=move.teacher_x,
                        teacher_y=move.teacher_y,
                        mistake_x=move.x,
                        mistake_y=move.y,
                        original_moves_to_end=distance,
                        priority=priority,
                    )
                )
            history.append([move.x, move.y, move.player])
    if not tasks:
        raise ValueError("report contains no completed-loss disagreements")
    if max_tasks > 0 and len(tasks) > max_tasks:
        rng = np.random.default_rng(seed)
        weights = np.asarray([task.priority for task in tasks], dtype=np.float64)
        selected = rng.choice(
            len(tasks), size=max_tasks, replace=False, p=weights / weights.sum()
        )
        tasks = [tasks[int(index)] for index in sorted(selected.tolist())]
        for task_index, task in enumerate(tasks):
            task.task_index = task_index
    return tasks, report_hash


def _worker_initialize(
    engine: str,
    timeout_turn_ms: int,
    max_nodes: int,
    engine_threads: int,
    engine_memory_mb: int,
) -> None:
    global _WORKER_ENGINE
    if _WORKER_ENGINE is not None:
        raise RuntimeError("counterfactual worker initialized twice")
    _WORKER_ENGINE = RapfiAdapter(
        Path(engine),
        timeout_turn_ms=timeout_turn_ms,
        max_nodes=max_nodes,
        threads=engine_threads,
        max_memory_mb=engine_memory_mb,
    )


def _rollout(task: BranchTask, max_branch_plies: int) -> BranchRecord:
    if _WORKER_ENGINE is None:
        raise RuntimeError("counterfactual worker has no Rapfi engine")
    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
    history = [row.copy() for row in task.history_before]
    for x, y, player in history:
        if board[y, x] != EMPTY:
            raise ValueError("task prehistory contains duplicate move")
        board[y, x] = player
    player = task.player
    moves: list[BranchMove] = []
    winner = EMPTY
    termination = "truncated"
    error: str | None = None
    for branch_ply in range(max_branch_plies):
        try:
            started = time.perf_counter()
            if branch_ply == 0:
                chosen = (task.teacher_x, task.teacher_y)
                source = "teacher_correction"
            else:
                chosen = _WORKER_ENGINE.choose_move(history, player)
                source = "rapfi_rollout"
            seconds = time.perf_counter() - started
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            termination = "engine_error"
            break
        x, y = map(int, chosen)
        if not (0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE):
            error = f"Rapfi returned out-of-board move {(x, y)}"
            termination = "engine_error"
            break
        if board[y, x] != EMPTY:
            error = f"Rapfi returned occupied move {(x, y)}"
            termination = "engine_error"
            break
        board[y, x] = player
        history.append([x, y, player])
        moves.append(BranchMove(branch_ply, x, y, player, source, seconds))
        if is_win(board, x, y, player):
            winner = player
            termination = "win"
            break
        if not np.any(board == EMPTY):
            termination = "full_board_draw"
            break
        player = other(player)
    record = BranchRecord(task, moves, winner, termination, error)
    payload = asdict(record)
    payload.pop("record_sha256")
    record.record_sha256 = canonical_sha256(payload)
    return record


def export_branch_dataset(
    branches: list[BranchRecord], output: Path
) -> dict[str, object]:
    states: list[np.ndarray] = []
    policies: list[np.ndarray] = []
    values: list[float] = []
    priorities: list[float] = []
    value_weights: list[float] = []
    mistake_actions: list[int] = []
    group_ids: list[int] = []
    task_indices: list[int] = []
    branch_plies: list[int] = []
    players: list[int] = []
    terminal_branches = 0
    truncated_branches = 0
    error_branches = 0
    for branch in branches:
        task = branch.task
        board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        last_action = -1
        for x, y, player in task.history_before:
            board[y, x] = player
            last_action = y * BOARD_SIZE + x
        terminal = branch.error is None and branch.termination in (
            "win",
            "full_board_draw",
        )
        terminal_branches += int(terminal)
        truncated_branches += int(branch.termination == "truncated")
        error_branches += int(branch.error is not None)
        for index, move in enumerate(branch.moves):
            state = _encode_state(board, move.player, last_action)
            action = move.y * BOARD_SIZE + move.x
            policy = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.uint8)
            policy[action] = 1
            if terminal and branch.winner != EMPTY:
                value = 1.0 if branch.winner == move.player else -1.0
            else:
                value = 0.0
            distance = len(branch.moves) - 1 - index
            weight = (
                0.35 + 0.65 * np.exp(-distance / 24.0) if terminal else 0.0
            )
            priority = task.priority * (2.0 if index == 0 else 1.0)
            states.append(state)
            policies.append(policy)
            values.append(value)
            priorities.append(priority)
            value_weights.append(float(weight))
            mistake_actions.append(
                task.mistake_y * BOARD_SIZE + task.mistake_x if index == 0 else -1
            )
            group_ids.append(task.pair_index)
            task_indices.append(task.task_index)
            branch_plies.append(index)
            players.append(move.player)
            board[move.y, move.x] = move.player
            last_action = action
    if not states:
        raise ValueError("counterfactual rollouts contain no usable moves")
    count = len(states)
    arrays = {
        "states": np.stack(states),
        "policies": np.stack(policies),
        "values": np.asarray(values, dtype=np.float32),
        "priority": np.asarray(priorities, dtype=np.float32),
        "policy_weights": np.ones(count, dtype=np.float32),
        "value_weights": np.asarray(value_weights, dtype=np.float32),
        "mistake_action": np.asarray(mistake_actions, dtype=np.int16),
        "group_id": np.asarray(group_ids, dtype=np.int32),
        "task_index": np.asarray(task_indices, dtype=np.int32),
        "branch_ply": np.asarray(branch_plies, dtype=np.int16),
        "player": np.asarray(players, dtype=np.int8),
        "source": np.full(count, "rapfi_counterfactual", dtype="<U20"),
        "split": np.full(count, "train", dtype="<U5"),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".tmp-{os.getpid()}.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, output)
    metadata = {
        "format_version": FORMAT_VERSION,
        "dataset_kind": "rapfi_counterfactual_policy_value",
        "samples": count,
        "branches": len(branches),
        "terminal_branches": terminal_branches,
        "truncated_branches": truncated_branches,
        "error_branches": error_branches,
        "groups": len(set(group_ids)),
        "mistake_rows": int(np.sum(arrays["mistake_action"] >= 0)),
        "value_weighted_rows": int(np.sum(arrays["value_weights"] > 0)),
        "npz_sha256": sha256_file(output),
        "arrays": {name: list(value.shape) for name, value in arrays.items()},
    }
    sidecar = output.with_suffix(output.suffix + ".json")
    sidecar.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--output-dataset", type=Path, required=True)
    parser.add_argument("--max-tasks", type=int, default=4096)
    parser.add_argument("--max-branch-plies", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout-turn-ms", type=int, default=300)
    parser.add_argument("--max-nodes", type=int, default=100000)
    parser.add_argument("--engine-threads", type=int, default=4)
    parser.add_argument("--engine-memory-mb", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20270501)
    args = parser.parse_args()
    if args.max_branch_plies <= 0 or args.workers <= 0:
        raise ValueError("branch plies and workers must be positive")
    tasks, source_report_hash = extract_tasks(
        args.report.resolve(), max_tasks=args.max_tasks, seed=args.seed
    )
    context = multiprocessing.get_context("spawn")
    branches: list[BranchRecord] = []
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=context,
        initializer=_worker_initialize,
        initargs=(
            str(args.engine.resolve()),
            args.timeout_turn_ms,
            args.max_nodes,
            args.engine_threads,
            args.engine_memory_mb,
        ),
    ) as executor:
        futures = {
            executor.submit(_rollout, task, args.max_branch_plies): task.task_index
            for task in tasks
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            branches.append(future.result())
            if completed == 1 or completed % 25 == 0 or completed == len(tasks):
                print(f"counterfactual branches: {completed}/{len(tasks)}", flush=True)
    branches.sort(key=lambda branch: branch.task.task_index)
    report_payload = {
        "format_version": FORMAT_VERSION,
        "report_type": REPORT_TYPE,
        "complete": True,
        "signature": {
            "source_report": str(args.report.resolve()),
            "source_report_sha256": source_report_hash,
            "engine": str(args.engine.resolve()),
            "engine_sha256": sha256_file(args.engine),
            "max_tasks": args.max_tasks,
            "selected_tasks": len(tasks),
            "max_branch_plies": args.max_branch_plies,
            "timeout_turn_ms": args.timeout_turn_ms,
            "max_nodes": args.max_nodes,
            "engine_threads": args.engine_threads,
            "seed": args.seed,
        },
        "summary": {
            "branches": len(branches),
            "terminal": sum(
                branch.termination in ("win", "full_board_draw")
                for branch in branches
            ),
            "truncated": sum(branch.termination == "truncated" for branch in branches),
            "errors": sum(branch.error is not None for branch in branches),
        },
        "branches": [asdict(branch) for branch in branches],
    }
    report_payload["report_sha256"] = canonical_sha256(report_payload)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(
        json.dumps(report_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    metadata = export_branch_dataset(branches, args.output_dataset.resolve())
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

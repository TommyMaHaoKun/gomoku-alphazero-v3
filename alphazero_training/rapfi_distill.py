"""Play paired games against Rapfi and export authenticated policy targets.

Every game retains its full move list.  Rapfi is queried at every non-opening
position, including student turns, so each label is a genuine teacher decision
for that exact pre-move state.  Random opening moves are never mislabeled.
Both colours share a ``pair_index`` so the supervised trainer's group split
cannot leak a colour-swapped opening into validation.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
from typing import Iterable, Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from alphazero_training.benchmark_ddqk import generate_opening
    from alphazero_training.play_agent import AlphaZeroGomokuAgent
    from alphazero_training.rapfi_adapter import RapfiAdapter
else:
    from .benchmark_ddqk import generate_opening
    from .play_agent import AlphaZeroGomokuAgent
    from .rapfi_adapter import RapfiAdapter


BOARD_SIZE = 19
BLACK = 1
WHITE = 2
EMPTY = 0
DIRECTIONS = ((1, 0), (0, 1), (1, 1), (1, -1))
FORMAT_VERSION = 2
REPORT_TYPE = "rapfi_student_distillation"

_WORKER_AGENT: AlphaZeroGomokuAgent | None = None
_WORKER_RAPFI: RapfiAdapter | None = None
_WORKER_RAPFI_CONFIG: dict[str, object] | None = None
PAIR_ATTEMPTS = 3


@dataclass
class MoveRecord:
    ply: int
    x: int
    y: int
    player: int
    source: str
    seconds: float = 0.0
    decision_reason: str | None = None
    teacher_x: int | None = None
    teacher_y: int | None = None
    teacher_seconds: float = 0.0
    student_disagreed: bool = False


@dataclass
class GameRecord:
    pair_index: int
    student_color: int
    opening: list[list[int]]
    moves: list[MoveRecord]
    winner: int
    student_result: float | None
    termination: str
    error: str | None = None
    record_sha256: str = field(default="")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def other(player: int) -> int:
    return WHITE if player == BLACK else BLACK


def is_win(board: np.ndarray, x: int, y: int, player: int) -> bool:
    for dx, dy in DIRECTIONS:
        length = 1
        for sign in (-1, 1):
            nx, ny = x + sign * dx, y + sign * dy
            while (
                0 <= nx < BOARD_SIZE
                and 0 <= ny < BOARD_SIZE
                and int(board[ny, nx]) == player
            ):
                length += 1
                nx += sign * dx
                ny += sign * dy
        if length >= 5:
            return True
    return False


def replay_opening(opening: Iterable[tuple[int, int]]) -> tuple[np.ndarray, int]:
    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
    player = BLACK
    for x, y in opening:
        if not (0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE):
            raise ValueError(f"opening move outside board: {(x, y)}")
        if board[y, x] != EMPTY:
            raise ValueError(f"duplicate opening move: {(x, y)}")
        board[y, x] = player
        if is_win(board, x, y, player):
            raise ValueError("opening is already terminal")
        player = other(player)
    return board, player


def _worker_initialize(
    checkpoint: str,
    simulations: int,
    engine: str,
    timeout_turn_ms: int,
    max_nodes: int,
    engine_threads: int,
    engine_memory_mb: int,
) -> None:
    global _WORKER_AGENT, _WORKER_RAPFI, _WORKER_RAPFI_CONFIG
    if _WORKER_AGENT is not None or _WORKER_RAPFI is not None:
        raise RuntimeError("Rapfi distillation worker initialized twice")
    _WORKER_AGENT = AlphaZeroGomokuAgent(Path(checkpoint), simulations=simulations)
    if (
        _WORKER_AGENT.config.board_size != BOARD_SIZE
        or _WORKER_AGENT.config.win_length != 5
    ):
        raise RuntimeError("distillation requires a 19x19 freestyle-five checkpoint")
    _WORKER_RAPFI_CONFIG = {
        "engine_path": Path(engine),
        "timeout_turn_ms": timeout_turn_ms,
        "max_nodes": max_nodes,
        "threads": engine_threads,
        "max_memory_mb": engine_memory_mb,
    }
    _restart_worker_rapfi()


def _restart_worker_rapfi() -> None:
    """Replace a possibly desynchronised protocol subprocess."""

    global _WORKER_RAPFI
    if _WORKER_RAPFI_CONFIG is None:
        raise RuntimeError("Rapfi worker configuration is unavailable")
    if _WORKER_RAPFI is not None:
        _WORKER_RAPFI.close(force=True)
    _WORKER_RAPFI = RapfiAdapter(**_WORKER_RAPFI_CONFIG)


def _play_game(
    pair_index: int,
    opening: list[tuple[int, int]],
    student_color: int,
    max_moves: int,
) -> GameRecord:
    if _WORKER_AGENT is None or _WORKER_RAPFI is None:
        raise RuntimeError("worker has not been initialized")
    board, player = replay_opening(opening)
    history: list[list[int]] = []
    moves: list[MoveRecord] = []
    for ply, (x, y) in enumerate(opening):
        stone = BLACK if ply % 2 == 0 else WHITE
        history.append([x, y, stone])
        moves.append(MoveRecord(ply, x, y, stone, "opening"))

    last_move = tuple(opening[-1]) if opening else None
    winner = EMPTY
    termination = "truncated"
    error: str | None = None
    while len(history) < max_moves and np.any(board == EMPTY):
        source = "student" if player == student_color else "rapfi"
        decision_reason: str | None = None
        try:
            teacher_started = time.perf_counter()
            teacher_move = _WORKER_RAPFI.choose_move(history, player)
            teacher_seconds = time.perf_counter() - teacher_started
            if source == "student":
                started = time.perf_counter()
                chosen = _WORKER_AGENT.choose_move(
                    board.tolist(), last_move, ai_color=player
                )
                elapsed = time.perf_counter() - started
                decision_reason = _WORKER_AGENT.last_decision_reason or "unknown"
            else:
                chosen = teacher_move
                elapsed = teacher_seconds
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            termination = "engine_error"
            break
        if chosen is None:
            error = f"{source} returned no move"
            termination = "engine_error"
            break
        x, y = map(int, chosen)
        teacher_x, teacher_y = map(int, teacher_move)
        if not (0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE):
            error = f"{source} returned out-of-board move {(x, y)}"
            termination = "engine_error"
            break
        if board[y, x] != EMPTY:
            error = f"{source} returned occupied move {(x, y)}"
            termination = "engine_error"
            break

        ply = len(history)
        board[y, x] = player
        history.append([x, y, player])
        moves.append(
            MoveRecord(
                ply,
                x,
                y,
                player,
                source,
                elapsed,
                decision_reason,
                teacher_x,
                teacher_y,
                teacher_seconds,
                (x, y) != (teacher_x, teacher_y),
            )
        )
        last_move = (x, y)
        if is_win(board, x, y, player):
            winner = player
            termination = "win"
            break
        player = other(player)

    if error is None and winner == EMPTY and not np.any(board == EMPTY):
        termination = "full_board_draw"
    if error is not None or termination == "truncated":
        result = None
    elif winner == EMPTY:
        result = 0.5
    else:
        result = 1.0 if winner == student_color else 0.0

    record = GameRecord(
        pair_index=pair_index,
        student_color=student_color,
        opening=[[x, y] for x, y in opening],
        moves=moves,
        winner=winner,
        student_result=result,
        termination=termination,
        error=error,
    )
    payload = asdict(record)
    payload.pop("record_sha256")
    record.record_sha256 = canonical_sha256(payload)
    return record


def _worker_pair(
    pair_index: int,
    opening: list[tuple[int, int]],
    max_moves: int,
) -> list[GameRecord]:
    last_records: list[GameRecord] = []
    for attempt in range(1, PAIR_ATTEMPTS + 1):
        records: list[GameRecord] = []
        for student_color in (BLACK, WHITE):
            record = _play_game(pair_index, opening, student_color, max_moves)
            records.append(record)
            if record.error is not None:
                # Protocol errors can leave BOARD/DONE responses queued.  A
                # fresh engine prevents one transient fault from poisoning the
                # colour-swapped game or the next pair assigned to this worker.
                _restart_worker_rapfi()
        if len(records) == 2 and all(record.error is None for record in records):
            return records
        last_records = records
        if attempt < PAIR_ATTEMPTS:
            _restart_worker_rapfi()
    return last_records


def successful_pair_indices(records: Sequence[GameRecord]) -> set[int]:
    """Return only fully successful colour-swapped pairs eligible for resume."""

    grouped: dict[int, list[GameRecord]] = {}
    for record in records:
        grouped.setdefault(record.pair_index, []).append(record)
    return {
        pair_index
        for pair_index, group in grouped.items()
        if len(group) == 2
        and {record.student_color for record in group} == {BLACK, WHITE}
        and all(
            record.error is None
            and record.student_result is not None
            and record.termination in ("win", "full_board_draw")
            for record in group
        )
    }


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _record_from_dict(raw: dict[str, object]) -> GameRecord:
    data = dict(raw)
    data["moves"] = [MoveRecord(**move) for move in data["moves"]]  # type: ignore[arg-type]
    return GameRecord(**data)  # type: ignore[arg-type]


def validate_record(record: GameRecord) -> None:
    payload = asdict(record)
    recorded_hash = payload.pop("record_sha256")
    if recorded_hash != canonical_sha256(payload):
        raise ValueError(f"game pair {record.pair_index} record hash mismatch")
    if record.student_color not in (BLACK, WHITE):
        raise ValueError("invalid student color")
    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
    expected = BLACK
    terminal = False
    for index, move in enumerate(record.moves):
        if move.ply != index or move.player != expected:
            raise ValueError(f"game pair {record.pair_index} has invalid move sequence")
        if move.source not in ("opening", "student", "rapfi"):
            raise ValueError("invalid move source")
        if index < len(record.opening):
            if move.source != "opening" or [move.x, move.y] != record.opening[index]:
                raise ValueError("recorded opening does not match moves")
        elif move.source == "opening":
            raise ValueError("opening source appears after opening")
        elif (move.source == "student") != (move.player == record.student_color):
            raise ValueError("move source disagrees with student color")
        if terminal or not (0 <= move.x < BOARD_SIZE and 0 <= move.y < BOARD_SIZE):
            raise ValueError("move occurs after terminal state or outside board")
        if board[move.y, move.x] != EMPTY:
            raise ValueError("move repeats an occupied point")
        if index >= len(record.opening):
            if move.teacher_x is None or move.teacher_y is None:
                raise ValueError("non-opening move is missing its Rapfi target")
            if not (
                0 <= move.teacher_x < BOARD_SIZE
                and 0 <= move.teacher_y < BOARD_SIZE
            ):
                raise ValueError("Rapfi target lies outside the board")
            if board[move.teacher_y, move.teacher_x] != EMPTY:
                raise ValueError("Rapfi target is not legal in the pre-move state")
            if move.student_disagreed != (
                (move.x, move.y) != (move.teacher_x, move.teacher_y)
            ):
                raise ValueError("student disagreement flag is inconsistent")
            if move.source == "rapfi" and move.student_disagreed:
                raise ValueError("Rapfi's played move must equal its teacher target")
        elif move.teacher_x is not None or move.teacher_y is not None:
            raise ValueError("random opening move cannot carry a teacher target")
        board[move.y, move.x] = move.player
        terminal = is_win(board, move.x, move.y, move.player)
        expected = other(expected)
    replay_winner = record.moves[-1].player if terminal and record.moves else EMPTY
    if replay_winner != record.winner:
        raise ValueError("recorded winner does not match replay")
    if record.error is None:
        expected_termination = "win" if terminal else "full_board_draw" if not np.any(board == EMPTY) else "truncated"
        if record.termination != expected_termination:
            raise ValueError("recorded termination does not match replay")


def summarize(records: list[GameRecord], requested_pairs: int) -> dict[str, object]:
    completed = [
        record
        for record in records
        if record.error is None
        and record.termination in ("win", "full_board_draw")
        and record.student_result is not None
    ]
    grouped: dict[int, list[GameRecord]] = {}
    for record in completed:
        grouped.setdefault(record.pair_index, []).append(record)
    complete_pairs = sum(
        len(group) == 2 and {r.student_color for r in group} == {BLACK, WHITE}
        for group in grouped.values()
    )
    losses = sum(record.student_result == 0.0 for record in completed)
    wins = sum(record.student_result == 1.0 for record in completed)
    draws = sum(record.student_result == 0.5 for record in completed)
    teacher_moves = sum(
        move.teacher_x is not None for record in completed for move in record.moves
    )
    student_decisions = [
        move
        for record in completed
        for move in record.moves
        if move.source == "student"
    ]
    return {
        "requested_pairs": requested_pairs,
        "complete_pairs": complete_pairs,
        "games": len(records),
        "completed_games": len(completed),
        "errors": sum(record.error is not None for record in records),
        "truncated": sum(record.termination == "truncated" for record in records),
        "student_wins": wins,
        "student_losses": losses,
        "draws": draws,
        "student_score": (
            (wins + 0.5 * draws) / len(completed) if completed else 0.0
        ),
        "rapfi_policy_samples": teacher_moves,
        "student_teacher_disagreements": sum(
            move.student_disagreed for move in student_decisions
        ),
        "student_teacher_agreement_rate": (
            sum(not move.student_disagreed for move in student_decisions)
            / len(student_decisions)
            if student_decisions
            else 0.0
        ),
    }


def export_dataset(records: list[GameRecord], output: Path) -> dict[str, object]:
    states: list[np.ndarray] = []
    policies: list[np.ndarray] = []
    priorities: list[float] = []
    group_ids: list[int] = []
    game_indices: list[int] = []
    ply_indices: list[int] = []
    student_colors: list[int] = []
    ai_losses: list[int] = []
    student_turns: list[int] = []
    student_disagreements: list[int] = []

    for game_index, record in enumerate(records):
        validate_record(record)
        if record.error is not None or record.termination not in ("win", "full_board_draw"):
            continue
        board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        last_action = -1
        is_loss = record.student_result == 0.0
        for move in record.moves:
            if move.teacher_x is not None and move.teacher_y is not None:
                state = np.zeros((4, BOARD_SIZE, BOARD_SIZE), dtype=np.uint8)
                state[0] = board == move.player
                state[1] = board == other(move.player)
                if last_action >= 0:
                    last_y, last_x = divmod(last_action, BOARD_SIZE)
                    state[2, last_y, last_x] = 1
                if move.player == BLACK:
                    state[3].fill(1)
                policy = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32)
                policy[move.teacher_y * BOARD_SIZE + move.teacher_x] = 1.0
                states.append(state)
                policies.append(policy)
                priorities.append(
                    3.0
                    if is_loss and move.student_disagreed
                    else 2.0
                    if is_loss or move.student_disagreed
                    else 1.0
                )
                group_ids.append(record.pair_index)
                game_indices.append(game_index)
                ply_indices.append(move.ply)
                student_colors.append(record.student_color)
                ai_losses.append(int(is_loss))
                student_turns.append(int(move.source == "student"))
                student_disagreements.append(int(move.student_disagreed))
            board[move.y, move.x] = move.player
            last_action = move.y * BOARD_SIZE + move.x

    if not states:
        raise ValueError("no completed Rapfi decisions are available for export")
    count = len(states)
    arrays = {
        "states": np.stack(states),
        "policies": np.stack(policies),
        "values": np.zeros(count, dtype=np.float32),
        "priority": np.asarray(priorities, dtype=np.float32),
        "policy_weights": np.ones(count, dtype=np.float32),
        "value_weights": np.zeros(count, dtype=np.float32),
        "group_id": np.asarray(group_ids, dtype=np.int32),
        "game_index": np.asarray(game_indices, dtype=np.int32),
        "ply_index": np.asarray(ply_indices, dtype=np.int16),
        "student_color": np.asarray(student_colors, dtype=np.int8),
        "ai_loss": np.asarray(ai_losses, dtype=np.uint8),
        "student_turn": np.asarray(student_turns, dtype=np.uint8),
        "student_disagreed": np.asarray(student_disagreements, dtype=np.uint8),
        "source": np.full(count, "rapfi", dtype="<U8"),
    }
    _atomic_npz(output, arrays)
    metadata = {
        "format_version": FORMAT_VERSION,
        "dataset_kind": "rapfi_policy_distillation",
        "samples": count,
        "groups": len(set(group_ids)),
        "ai_loss_samples": int(sum(ai_losses)),
        "student_turn_samples": int(sum(student_turns)),
        "student_disagreement_samples": int(sum(student_disagreements)),
        "policy_only": True,
        "npz_sha256": sha256_file(output),
        "arrays": {name: list(array.shape) for name, array in arrays.items()},
    }
    _atomic_json(output.with_suffix(output.suffix + ".json"), metadata)
    return metadata


def _write_ai_loss_library(records: list[GameRecord], directory: Path) -> int:
    directory.mkdir(parents=True, exist_ok=True)
    count = 0
    for record in records:
        if record.error is None and record.student_result == 0.0:
            filename = f"rapfi_pair{record.pair_index:06d}_student{record.student_color}.json"
            payload = {
                "format_version": FORMAT_VERSION,
                "source": "rapfi_distillation",
                "training_status": "pending",
                "game": asdict(record),
            }
            payload["sha256"] = canonical_sha256(payload)
            _atomic_json(directory / filename, payload)
            count += 1
    return count


def _runtime_signature(args: argparse.Namespace, openings: list[list[tuple[int, int]]]) -> dict[str, object]:
    runtime_dir = args.engine.resolve().parent
    runtime_files = [
        path
        for path in runtime_dir.iterdir()
        if path.is_file() and path.name != "SHA256SUMS"
    ]
    return {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint.resolve()),
        "rapfi_engine": str(args.engine.resolve()),
        "rapfi_engine_sha256": sha256_file(args.engine.resolve()),
        "rapfi_runtime_files": {
            path.name: sha256_file(path) for path in sorted(runtime_files)
        },
        "generator_sha256": sha256_file(Path(__file__).resolve()),
        "adapter_sha256": sha256_file(Path(__file__).with_name("rapfi_adapter.py")),
        "opening_manifest_sha256": canonical_sha256(openings),
        "pairs": args.pairs,
        "opening_plies": args.opening_plies,
        "max_moves": args.max_moves,
        "seed": args.seed,
        "simulations": args.simulations,
        "timeout_turn_ms": args.timeout_turn_ms,
        "max_nodes": args.max_nodes,
        "engine_threads": args.engine_threads,
    }


def _build_report(
    signature: dict[str, object],
    records: list[GameRecord],
    requested_pairs: int,
    complete: bool,
) -> dict[str, object]:
    records = sorted(records, key=lambda r: (r.pair_index, r.student_color))
    payload = {
        "format_version": FORMAT_VERSION,
        "report_type": REPORT_TYPE,
        "rules": {"board_size": 19, "win_length": 5, "freestyle": True},
        "complete": complete,
        "signature": signature,
        "summary": summarize(records, requested_pairs),
        "games": [asdict(record) for record in records],
    }
    payload["report_sha256"] = canonical_sha256(payload)
    return payload


def _load_resume(path: Path, signature: dict[str, object]) -> list[GameRecord]:
    if not path.is_file():
        return []
    report = json.loads(path.read_text(encoding="utf-8"))
    reported_hash = report.pop("report_sha256", None)
    if reported_hash != canonical_sha256(report):
        raise ValueError("existing Rapfi report hash mismatch")
    if report.get("signature") != signature:
        raise ValueError("existing Rapfi report signature does not match this run")
    records = [_record_from_dict(raw) for raw in report.get("games", [])]
    for record in records:
        validate_record(record)
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--ai-loss-dir", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=16)
    parser.add_argument("--opening-plies", type=int, default=4)
    parser.add_argument("--max-moves", type=int, default=361)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--simulations", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout-turn-ms", type=int, default=500)
    parser.add_argument("--max-nodes", type=int, default=0)
    parser.add_argument("--engine-threads", type=int, default=4)
    parser.add_argument("--engine-memory-mb", type=int, default=2048)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.pairs <= 0 or args.workers <= 0 or args.simulations <= 0:
        raise ValueError("pairs, workers, and simulations must be positive")
    if not 0 <= args.opening_plies <= args.max_moves <= BOARD_SIZE**2:
        raise ValueError("invalid opening/max-move limits")
    if not args.checkpoint.is_file() or not args.engine.is_file():
        raise FileNotFoundError("checkpoint or Rapfi engine is missing")

    rng = np.random.default_rng(args.seed)
    openings = [generate_opening(rng, args.opening_plies) for _ in range(args.pairs)]
    signature = _runtime_signature(args, openings)
    records = _load_resume(args.report, signature)
    complete_pairs = successful_pair_indices(records)
    pending = [index for index in range(args.pairs) if index not in complete_pairs]
    print(
        f"Rapfi distillation: {len(complete_pairs)}/{args.pairs} pairs resumed, "
        f"{len(pending)} pending",
        flush=True,
    )

    if pending:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=min(args.workers, len(pending)),
            mp_context=context,
            initializer=_worker_initialize,
            initargs=(
                str(args.checkpoint.resolve()),
                args.simulations,
                str(args.engine.resolve()),
                args.timeout_turn_ms,
                args.max_nodes,
                args.engine_threads,
                args.engine_memory_mb,
            ),
        ) as executor:
            futures = {
                executor.submit(
                    _worker_pair,
                    pair_index,
                    openings[pair_index],
                    args.max_moves,
                ): pair_index
                for pair_index in pending
            }
            for future in as_completed(futures):
                pair_index = futures[future]
                try:
                    pair_records = future.result()
                except Exception as exc:
                    print(f"pair {pair_index} worker failure: {type(exc).__name__}: {exc}", flush=True)
                    continue
                records = [r for r in records if r.pair_index != pair_index] + pair_records
                report = _build_report(signature, records, args.pairs, complete=False)
                _atomic_json(args.report, report)
                pair_summary = summarize(pair_records, 1)
                print(
                    f"pair {pair_index + 1}/{args.pairs}: "
                    f"student W/L/D={pair_summary['student_wins']}/"
                    f"{pair_summary['student_losses']}/{pair_summary['draws']}",
                    flush=True,
                )

    final_summary = summarize(records, args.pairs)
    complete = final_summary["complete_pairs"] == args.pairs and final_summary["errors"] == 0
    _atomic_json(args.report, _build_report(signature, records, args.pairs, complete))
    metadata = export_dataset(records, args.dataset)
    losses_written = _write_ai_loss_library(records, args.ai_loss_dir)
    print(json.dumps(final_summary, ensure_ascii=False, sort_keys=True), flush=True)
    print(
        f"dataset={args.dataset} samples={metadata['samples']} "
        f"ai_loss_games={losses_written} complete={complete}",
        flush=True,
    )
    if not complete:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

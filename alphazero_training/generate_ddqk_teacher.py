"""Generate auditable DDQK-vs-DDQK expert games for Gomoku V3.

The generator runs the original DDQK engine on *both* colours.  Random
openings are legal, non-terminal, deterministic from a master seed, and are
grouped as spatially symmetric variants of one base opening.  Keeping those
closely related games under one ``pair_index`` lets downstream train/validation
splits keep an entire opening family together.

Every finished worker result is written by the coordinator with ``os.replace``.
An interrupted run can therefore resume without replaying durable games.  A
game is usable only when ``error is None`` and ``termination`` is either
``win`` or ``full_board_draw``; truncation and engine failures are deliberately
retained in the report but never labelled complete.
"""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Sequence

import numpy as np

if __package__ in (None, ""):  # Support direct execution and ``python -m``.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from alphazero_training.ddqk_adapter import DDQKAdapter
else:
    from .ddqk_adapter import DDQKAdapter


BOARD_SIZE = 19
BLACK = 1
WHITE = 2
EMPTY = 0
DIRECTIONS = ((1, 0), (0, 1), (1, 1), (1, -1))
COMPLETE_TERMINATIONS = frozenset(("win", "full_board_draw"))
REPORT_TYPE = "ddqk_teacher_selfplay"

_WORKER_DDQK: DDQKAdapter | None = None


@dataclass(frozen=True)
class OpeningMember:
    """One deterministic member of a correlated opening group."""

    group_index: int
    pair_index: int
    member_index: int
    group_seed: int
    game_seed: int
    symmetry: int
    opening: list[list[int]]


@dataclass
class TeacherGameRecord:
    """A complete audit record for one attempted teacher game."""

    group_index: int
    pair_index: int
    member_index: int
    group_seed: int
    game_seed: int
    opening_symmetry: int
    opening: list[list[int]]
    moves: list[list[int]]
    winner: int
    plies: int
    ddqk_seconds: float
    ddqk_moves: int
    termination: str
    complete: bool
    error: str | None
    final_board: list[str]
    final_board_sha256: str


def other(player: int) -> int:
    return WHITE if player == BLACK else BLACK


def is_win(board: np.ndarray, x: int, y: int, player: int) -> bool:
    """Return whether the last stone makes freestyle five-or-more."""

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


def _nearby_empty(board: np.ndarray, radius: int = 2) -> list[tuple[int, int]]:
    occupied = np.argwhere(board != EMPTY)
    if occupied.size == 0:
        center = BOARD_SIZE // 2
        return [(center, center)]
    points: set[tuple[int, int]] = set()
    for y, x in occupied:
        for ny in range(max(0, int(y) - radius), min(BOARD_SIZE, int(y) + radius + 1)):
            for nx in range(max(0, int(x) - radius), min(BOARD_SIZE, int(x) + radius + 1)):
                if board[ny, nx] == EMPTY:
                    points.add((nx, ny))
    return sorted(points)


def transform_point(x: int, y: int, symmetry: int) -> tuple[int, int]:
    """Apply one of the eight D4 symmetries to a board coordinate."""

    if not 0 <= symmetry < 8:
        raise ValueError("symmetry must be in [0, 8)")
    n = BOARD_SIZE - 1
    if symmetry >= 4:
        x = n - x
        symmetry -= 4
    for _ in range(symmetry):
        x, y = n - y, x
    return x, y


def generate_opening(rng: np.random.Generator, plies: int) -> list[tuple[int, int]]:
    """Generate a legal alternating opening with no already-won position."""

    if not 0 <= plies <= BOARD_SIZE * BOARD_SIZE:
        raise ValueError(f"opening plies must be in [0, {BOARD_SIZE * BOARD_SIZE}]")
    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
    moves: list[tuple[int, int]] = []
    player = BLACK
    for _ in range(plies):
        choices: list[tuple[int, int]] = []
        for x, y in _nearby_empty(board):
            board[y, x] = player
            terminal = is_win(board, x, y, player)
            board[y, x] = EMPTY
            if not terminal:
                choices.append((x, y))
        if not choices:
            raise RuntimeError("could not construct the requested non-terminal opening")
        x, y = choices[int(rng.integers(len(choices)))]
        board[y, x] = player
        moves.append((x, y))
        player = other(player)
    return moves


def opening_board(moves: Iterable[Sequence[int]]) -> tuple[np.ndarray, int]:
    """Validate and replay an opening, returning board and side to move."""

    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
    player = BLACK
    for raw_move in moves:
        if len(raw_move) != 2:
            raise ValueError(f"malformed opening move: {raw_move!r}")
        x, y = map(int, raw_move)
        if not (0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE):
            raise ValueError(f"opening move outside board: {(x, y)}")
        if board[y, x] != EMPTY:
            raise ValueError(f"duplicate opening move: {(x, y)}")
        board[y, x] = player
        if is_win(board, x, y, player):
            raise ValueError("opening is already terminal")
        player = other(player)
    return board, player


def derive_seed(master_seed: int, *parts: object) -> int:
    """Derive a stable unsigned 63-bit seed without Python hash randomisation."""

    material = "|".join((str(int(master_seed)), *(str(part) for part in parts)))
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def build_opening_manifest(
    *,
    seed: int,
    groups: int,
    games_per_group: int,
    opening_plies: int,
) -> list[OpeningMember]:
    """Build grouped, deterministic D4 variants of legal base openings."""

    if groups <= 0:
        raise ValueError("groups must be positive")
    if not 1 <= games_per_group <= 8:
        raise ValueError("games_per_group must be in [1, 8]")
    members: list[OpeningMember] = []
    for group_index in range(groups):
        group_seed = derive_seed(seed, "group", group_index)
        base = generate_opening(np.random.default_rng(group_seed), opening_plies)
        symmetry_rng = np.random.default_rng(derive_seed(group_seed, "symmetries"))
        symmetries = [int(value) for value in symmetry_rng.permutation(8)]
        for member_index in range(games_per_group):
            symmetry = symmetries[member_index]
            opening = [list(transform_point(x, y, symmetry)) for x, y in base]
            # Validate every transformed member rather than relying on the
            # mathematical symmetry argument alone.
            opening_board(opening)
            members.append(
                OpeningMember(
                    group_index=group_index,
                    pair_index=group_index,
                    member_index=member_index,
                    group_seed=group_seed,
                    game_seed=derive_seed(group_seed, "member", member_index),
                    symmetry=symmetry,
                    opening=opening,
                )
            )
    return members


def board_rows(board: np.ndarray) -> list[str]:
    """Encode a final board compactly and human-readably."""

    return ["".join(str(int(cell)) for cell in row) for row in board]


def board_sha256(rows: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(rows).encode("ascii")).hexdigest()


def record_key(record: TeacherGameRecord | OpeningMember) -> tuple[int, int]:
    return int(record.group_index), int(record.member_index)


def validate_record(record: TeacherGameRecord, member: OpeningMember) -> None:
    """Validate a saved record against its manifest and reconstructed board."""

    key = record_key(record)
    if (
        record.pair_index != member.pair_index
        or record.group_seed != member.group_seed
        or record.game_seed != member.game_seed
        or record.opening_symmetry != member.symmetry
        or record.opening != member.opening
    ):
        raise ValueError(f"report game {key} does not match its opening manifest")
    if record.plies != len(record.moves):
        raise ValueError(f"report game {key} plies does not match its move list")
    if record.winner not in (EMPTY, BLACK, WHITE):
        raise ValueError(f"report game {key} has an invalid winner")

    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
    observed_winner = EMPTY
    for ply_index, raw_move in enumerate(record.moves):
        if len(raw_move) != 3:
            raise ValueError(f"report game {key} has a malformed move")
        x, y, player = map(int, raw_move)
        expected_player = BLACK if ply_index % 2 == 0 else WHITE
        if player != expected_player:
            raise ValueError(f"report game {key} has non-alternating players")
        if not (0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE):
            raise ValueError(f"report game {key} has an out-of-board move")
        if board[y, x] != EMPTY:
            raise ValueError(f"report game {key} repeats a move")
        if observed_winner != EMPTY:
            raise ValueError(f"report game {key} contains moves after a win")
        if ply_index < len(member.opening):
            if member.opening[ply_index] != [x, y]:
                raise ValueError(f"report game {key} opening prefix does not match")
        board[y, x] = player
        if is_win(board, x, y, player):
            observed_winner = player

    derived_complete = (
        record.error is None and record.termination in COMPLETE_TERMINATIONS
    )
    if record.complete != derived_complete:
        raise ValueError(f"report game {key} has a false completion flag")
    if record.termination == "win":
        if observed_winner == EMPTY or record.winner != observed_winner:
            raise ValueError(f"report game {key} winner does not match its board")
    elif record.termination == "full_board_draw":
        if record.winner != EMPTY or observed_winner != EMPTY or np.any(board == EMPTY):
            raise ValueError(f"report game {key} is not a genuine full-board draw")
    elif record.winner != EMPTY:
        raise ValueError(f"incomplete report game {key} declares a winner")

    expected_rows = board_rows(board)
    if record.final_board != expected_rows:
        raise ValueError(f"report game {key} final board does not match its moves")
    if board_sha256(record.final_board) != record.final_board_sha256:
        raise ValueError(f"report game {key} final-board checksum does not match")


def play_teacher_game(
    ddqk: Any,
    member: OpeningMember,
    max_moves: int,
) -> TeacherGameRecord:
    """Play one DDQK-vs-DDQK game, preserving all failure evidence."""

    opening = [(int(move[0]), int(move[1])) for move in member.opening]
    board, player = opening_board(opening)
    ddqk.reset()
    ddqk.sync_opening(opening, starting_player=BLACK)
    last_move = opening[-1] if opening else None
    moves = [
        [x, y, BLACK if index % 2 == 0 else WHITE]
        for index, (x, y) in enumerate(opening)
    ]
    winner = EMPTY
    ddqk_seconds = 0.0
    ddqk_moves = 0
    termination = "truncated"
    error: str | None = None

    while len(moves) < max_moves and np.any(board == EMPTY):
        started = time.perf_counter()
        try:
            move = ddqk.choose_move(board.tolist(), player, last_move)
            ddqk_moves += 1
            if getattr(ddqk, "last_engine_error", None) is not None:
                raise RuntimeError(
                    "DDQK used fallback after engine failure: "
                    f"{ddqk.last_engine_error}"
                )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            termination = "engine_error"
            break
        finally:
            ddqk_seconds += time.perf_counter() - started

        if move is None:
            error = f"player {player} returned no move"
            termination = "invalid_move"
            break
        try:
            x, y = map(int, move)
        except Exception as exc:
            error = f"malformed move {move!r}: {type(exc).__name__}: {exc}"
            termination = "invalid_move"
            break
        if not (0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE):
            error = f"player {player} returned out-of-board move {(x, y)}"
            termination = "invalid_move"
            break
        if board[y, x] != EMPTY:
            error = f"player {player} returned occupied move {(x, y)}"
            termination = "invalid_move"
            break

        board[y, x] = player
        moves.append([x, y, player])
        last_move = (x, y)
        if is_win(board, x, y, player):
            winner = player
            termination = "win"
            break
        player = other(player)

    if error is None and winner == EMPTY and not np.any(board == EMPTY):
        termination = "full_board_draw"

    complete = error is None and termination in COMPLETE_TERMINATIONS
    rows = board_rows(board)
    return TeacherGameRecord(
        group_index=member.group_index,
        pair_index=member.pair_index,
        member_index=member.member_index,
        group_seed=member.group_seed,
        game_seed=member.game_seed,
        opening_symmetry=member.symmetry,
        opening=[list(move) for move in member.opening],
        moves=moves,
        winner=winner,
        plies=len(moves),
        ddqk_seconds=ddqk_seconds,
        ddqk_moves=ddqk_moves,
        termination=termination,
        complete=complete,
        error=error,
        final_board=rows,
        final_board_sha256=board_sha256(rows),
    )


def worker_failure_record(
    member: OpeningMember,
    exc: BaseException,
) -> TeacherGameRecord:
    """Turn an out-of-game worker failure into an explicit unusable record."""

    board, _ = opening_board(member.opening)
    rows = board_rows(board)
    moves = [
        [int(x), int(y), BLACK if index % 2 == 0 else WHITE]
        for index, (x, y) in enumerate(member.opening)
    ]
    return TeacherGameRecord(
        group_index=member.group_index,
        pair_index=member.pair_index,
        member_index=member.member_index,
        group_seed=member.group_seed,
        game_seed=member.game_seed,
        opening_symmetry=member.symmetry,
        opening=[list(move) for move in member.opening],
        moves=moves,
        winner=EMPTY,
        plies=len(moves),
        ddqk_seconds=0.0,
        ddqk_moves=0,
        termination="worker_error",
        complete=False,
        error=f"{type(exc).__name__}: {exc}",
        final_board=rows,
        final_board_sha256=board_sha256(rows),
    )


def initialize_worker(ddqk_source: str) -> None:
    """Load exactly one native DDQK engine in each spawned process."""

    global _WORKER_DDQK
    if _WORKER_DDQK is not None:
        raise RuntimeError("teacher worker was initialized more than once")
    _WORKER_DDQK = DDQKAdapter(Path(ddqk_source))


def run_worker_game(member_payload: dict[str, object], max_moves: int) -> TeacherGameRecord:
    if _WORKER_DDQK is None:
        raise RuntimeError("teacher worker has not been initialized")
    member = OpeningMember(**member_payload)  # type: ignore[arg-type]
    return play_teacher_game(_WORKER_DDQK, member, max_moves)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def summarize(
    records: Sequence[TeacherGameRecord], expected_games: int, games_per_group: int
) -> dict[str, object]:
    complete = [record for record in records if record.complete]
    wins = [record for record in complete if record.termination == "win"]
    draws = [record for record in complete if record.termination == "full_board_draw"]
    recorded_keys = {record_key(record) for record in records}
    groups: dict[int, list[TeacherGameRecord]] = {}
    for record in complete:
        groups.setdefault(record.group_index, []).append(record)
    complete_groups = sum(
        1
        for group_records in groups.values()
        if len(group_records)
        == games_per_group
        and len({record.member_index for record in group_records}) == games_per_group
    )
    if len(records) < expected_games:
        status = "in_progress"
    elif len(complete) == expected_games:
        status = "complete"
    else:
        status = "finished_with_failures"
    return {
        "collection_status": status,
        "expected_games": expected_games,
        "recorded_games": len(records),
        "distinct_recorded_games": len(recorded_keys),
        "usable_complete_games": len(complete),
        "failed_or_truncated_games": len(records) - len(complete),
        "wins": len(wins),
        "black_wins": sum(record.winner == BLACK for record in wins),
        "white_wins": sum(record.winner == WHITE for record in wins),
        "draws": len(draws),
        "engine_errors": sum(record.termination == "engine_error" for record in records),
        "worker_errors": sum(record.termination == "worker_error" for record in records),
        "invalid_moves": sum(record.termination == "invalid_move" for record in records),
        "truncated": sum(record.termination == "truncated" for record in records),
        "mean_plies": (
            float(np.mean([record.plies for record in complete])) if complete else 0.0
        ),
        "ddqk_seconds_per_move": (
            sum(record.ddqk_seconds for record in records)
            / sum(record.ddqk_moves for record in records)
            if sum(record.ddqk_moves for record in records)
            else 0.0
        ),
        # This is informational; exact expected group completeness is exposed
        # by the top-level manifest and per-record completion fields.
        "groups_with_usable_games": len(groups),
        "groups_whose_recorded_games_are_usable": complete_groups,
    }


def manifest_payload(members: Sequence[OpeningMember]) -> list[dict[str, object]]:
    grouped: dict[int, list[OpeningMember]] = {}
    for member in members:
        grouped.setdefault(member.group_index, []).append(member)
    result: list[dict[str, object]] = []
    for group_index, group_members in sorted(grouped.items()):
        ordered = sorted(group_members, key=lambda member: member.member_index)
        result.append(
            {
                "group_index": group_index,
                "pair_index": group_index,
                "group_seed": ordered[0].group_seed,
                "members": [asdict(member) for member in ordered],
            }
        )
    return result


def build_report(
    *,
    signature: dict[str, object],
    members: Sequence[OpeningMember],
    records: Sequence[TeacherGameRecord],
    workers: int,
) -> dict[str, object]:
    manifest = manifest_payload(members)
    return {
        "format_version": 2,
        "report_type": REPORT_TYPE,
        "game_mode": "ddqk_vs_ddqk",
        "rules": {"board_size": BOARD_SIZE, "win_length": 5, "freestyle": True},
        "signature": signature,
        "ddqk_source": signature["ddqk_source"],
        "groups": signature["groups"],
        "games_per_group": signature["games_per_group"],
        "opening_plies": signature["opening_plies"],
        "max_moves": signature["max_moves"],
        "seed": signature["seed"],
        "workers": workers,
        "opening_manifest": manifest,
        # Flat openings preserve the convenient shape used by benchmark v2.
        "openings": [member.opening for member in members],
        "summary": summarize(
            records,
            expected_games=len(members),
            games_per_group=int(signature["games_per_group"]),
        ),
        "games": [asdict(record) for record in records],
    }


def load_resume_records(
    report_path: Path,
    *,
    signature: dict[str, object],
    members: Sequence[OpeningMember],
) -> dict[tuple[int, int], TeacherGameRecord]:
    """Load and strictly validate a durable partial report."""

    previous = json.loads(report_path.read_text(encoding="utf-8"))
    if previous.get("format_version") != 2 or previous.get("report_type") != REPORT_TYPE:
        raise ValueError("existing output is not a DDQK teacher format-v2 report")
    if previous.get("signature") != signature:
        raise ValueError("existing report signature does not match")
    expected_manifest = manifest_payload(members)
    if previous.get("opening_manifest") != expected_manifest:
        raise ValueError("existing opening manifest does not match")

    expected = {record_key(member): member for member in members}
    records: dict[tuple[int, int], TeacherGameRecord] = {}
    for raw in previous.get("games", []):
        record = TeacherGameRecord(**raw)
        key = record_key(record)
        if key not in expected:
            raise ValueError(f"report contains unexpected game {key}")
        if key in records:
            raise ValueError(f"report contains duplicate game {key}")
        member = expected[key]
        validate_record(record, member)
        records[key] = record
    return records


def print_game_status(record: TeacherGameRecord, total: int) -> None:
    status = record.termination if record.error is None else f"{record.termination}: {record.error}"
    print(
        f"group={record.group_index + 1} member={record.member_index + 1} "
        f"recorded={status} plies={record.plies} ({total} total requested)",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ddqk-source", type=Path, default=None)
    parser.add_argument("--groups", type=int, default=1)
    parser.add_argument("--games-per-group", type=int, default=2)
    parser.add_argument("--opening-plies", type=int, default=6)
    parser.add_argument("--max-moves", type=int, default=BOARD_SIZE * BOARD_SIZE)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing report instead of refusing to destroy it",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "ddqk_teacher_selfplay.json",
    )
    args = parser.parse_args()
    if args.resume and args.overwrite:
        raise SystemExit("--resume and --overwrite are mutually exclusive")
    if args.groups <= 0:
        raise SystemExit("--groups must be positive")
    if not 1 <= args.games_per_group <= 8:
        raise SystemExit("--games-per-group must be in [1, 8]")
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    if not 0 <= args.opening_plies <= BOARD_SIZE * BOARD_SIZE:
        raise SystemExit(f"--opening-plies must be in [0, {BOARD_SIZE * BOARD_SIZE}]")
    if not args.opening_plies <= args.max_moves <= BOARD_SIZE * BOARD_SIZE:
        raise SystemExit(
            f"--max-moves must be between opening plies and {BOARD_SIZE * BOARD_SIZE}"
        )

    members = build_opening_manifest(
        seed=args.seed,
        groups=args.groups,
        games_per_group=args.games_per_group,
        opening_plies=args.opening_plies,
    )

    # Validate and fingerprint the native engine in the coordinator before any
    # workers are spawned.  Each spawned worker then creates its own singleton.
    coordinator_ddqk = DDQKAdapter(args.ddqk_source)
    source_path = coordinator_ddqk.source_path.resolve()
    dll_path = (coordinator_ddqk.asset_dir / "dll.so").resolve()
    manifest = manifest_payload(members)
    signature: dict[str, object] = {
        "generator_sha256": sha256_file(Path(__file__).resolve()),
        "ddqk_source": str(source_path),
        "ddqk_source_sha256": sha256_file(source_path),
        "ddqk_dll": str(dll_path),
        "ddqk_dll_sha256": sha256_file(dll_path),
        "ddqk_depth": int(coordinator_ddqk._engine.DEPTH),
        "board_size": BOARD_SIZE,
        "win_length": 5,
        "groups": args.groups,
        "games_per_group": args.games_per_group,
        "opening_plies": args.opening_plies,
        "max_moves": args.max_moves,
        "seed": args.seed,
        "opening_manifest_sha256": canonical_sha256(manifest),
    }

    records_by_key: dict[tuple[int, int], TeacherGameRecord] = {}
    if args.resume and args.output.is_file():
        try:
            records_by_key = load_resume_records(
                args.output, signature=signature, members=members
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise SystemExit(f"refusing to resume: {exc}") from exc
    elif args.output.exists() and not args.overwrite:
        raise SystemExit(
            f"refusing to overwrite existing report: {args.output}; "
            "use --resume or --overwrite"
        )

    task_order = [record_key(member) for member in members]
    member_by_key = {record_key(member): member for member in members}
    pending = [key for key in task_order if key not in records_by_key]

    # Persist the manifest before the first potentially long native-engine
    # call.  A crash before game one is then still safely resumable.
    atomic_write_json(
        args.output,
        build_report(
            signature=signature,
            members=members,
            records=[
                records_by_key[key] for key in task_order if key in records_by_key
            ],
            workers=args.workers,
        ),
    )

    if pending:
        spawn_context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=spawn_context,
            initializer=initialize_worker,
            initargs=(str(source_path),),
        ) as executor:
            futures: dict[Future[TeacherGameRecord], tuple[int, int]] = {}
            for key in pending:
                member = member_by_key[key]
                future = executor.submit(
                    run_worker_game, asdict(member), args.max_moves
                )
                futures[future] = key

            for future in as_completed(futures):
                expected_key = futures[future]
                member = member_by_key[expected_key]
                try:
                    record = future.result()
                except Exception as exc:
                    # A failed initializer, serialization error, or broken
                    # worker is evidence too.  Preserve it as unusable rather
                    # than losing the rest of the run's durable progress.
                    record = worker_failure_record(member, exc)
                if record_key(record) != expected_key:
                    raise RuntimeError(
                        f"worker returned game {record_key(record)}, expected {expected_key}"
                    )
                records_by_key[expected_key] = record
                ordered = [
                    records_by_key[key]
                    for key in task_order
                    if key in records_by_key
                ]
                atomic_write_json(
                    args.output,
                    build_report(
                        signature=signature,
                        members=members,
                        records=ordered,
                        workers=args.workers,
                    ),
                )
                print_game_status(record, len(members))

    ordered = [records_by_key[key] for key in task_order if key in records_by_key]
    report = build_report(
        signature=signature,
        members=members,
        records=ordered,
        workers=args.workers,
    )
    atomic_write_json(args.output, report)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"report={args.output.resolve()}")


if __name__ == "__main__":
    main()

"""Paired-opening benchmark between the AlphaZero agent and DDQK.

This benchmark deliberately uses the DDQK Python engine rather than driving
its Pygame executable.  Each generated opening is played twice with colors
swapped, and every move is retained in the JSON report for later curriculum
training and failure analysis.
"""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Iterable, TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from .ddqk_adapter import DDQKAdapter
    from .play_agent import AlphaZeroGomokuAgent


BOARD_SIZE = 19
BLACK = 1
WHITE = 2
EMPTY = 0
DIRECTIONS = ((1, 0), (0, 1), (1, 1), (1, -1))

DEVELOPMENT_MODE = "development"
FINAL_CERTIFICATION_MODE = "final-certification"
FINAL_MIN_PAIRS = 600
FINAL_MIN_SCORE = 0.995
FINAL_MIN_COLOR_SCORE = 0.99
FINAL_MIN_EXACT_PAIR_SWEEP_LOWER95 = 0.995
CONFIDENCE_ALPHA = 0.05
EVALUATION_CODE_FILES = (
    "play_agent.py",
    "train_alphazero.py",
    "v3_search.py",
    "tactical_solver.py",
    "ddqk_adapter.py",
    "benchmark_ddqk.py",
)
DDQK_DECISION_ASSET_FILES = (
    "dll.so",
    "guess_data.txt",
    "black_calculated_value_19.txt",
    "white_calculated_value_19.txt",
)

_WORKER_AGENT: AlphaZeroGomokuAgent | None = None
_WORKER_DDQK: DDQKAdapter | None = None


def _runtime_classes() -> tuple[type[AlphaZeroGomokuAgent], type[DDQKAdapter]]:
    """Import engine classes lazily so statistics/provenance helpers stay lightweight."""

    if __package__ in (None, ""):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from alphazero_training.ddqk_adapter import DDQKAdapter as Adapter
        from alphazero_training.play_agent import AlphaZeroGomokuAgent as Agent
    else:
        from .ddqk_adapter import DDQKAdapter as Adapter
        from .play_agent import AlphaZeroGomokuAgent as Agent
    return Agent, Adapter


@dataclass
class GameRecord:
    pair_index: int
    model_color: int
    opening: list[list[int]]
    moves: list[list[int]]
    winner: int
    model_result: float | None
    plies: int
    model_seconds: float
    ddqk_seconds: float
    model_moves: int
    ddqk_moves: int
    termination: str
    error: str | None = None
    model_decision_reasons: list[str] = field(default_factory=list)


def initialize_worker(
    checkpoint: str,
    simulations: int,
    ddqk_source: str,
) -> None:
    """Initialize exactly one model and DDQK engine in this worker process."""

    global _WORKER_AGENT, _WORKER_DDQK
    if _WORKER_AGENT is not None or _WORKER_DDQK is not None:
        raise RuntimeError("benchmark worker was initialized more than once")
    agent_class, adapter_class = _runtime_classes()
    agent = agent_class(Path(checkpoint), simulations=simulations)
    if agent.config.board_size != BOARD_SIZE or agent.config.win_length != 5:
        raise RuntimeError(
            "benchmark requires a 19x19 freestyle-five checkpoint; got "
            f"{agent.config.board_size}x{agent.config.board_size}, "
            f"win_length={agent.config.win_length}"
        )
    _WORKER_AGENT = agent
    _WORKER_DDQK = adapter_class(Path(ddqk_source))


def run_worker_game(
    pair_index: int,
    model_color: int,
    opening: list[tuple[int, int]],
    max_moves: int,
) -> GameRecord:
    """Play one game using the process-local, persistent worker engines."""

    if _WORKER_AGENT is None or _WORKER_DDQK is None:
        raise RuntimeError("benchmark worker has not been initialized")
    return play_game(
        _WORKER_AGENT,
        _WORKER_DDQK,
        opening,
        model_color,
        pair_index,
        max_moves,
    )


def other(player: int) -> int:
    return WHITE if player == BLACK else BLACK


def is_win(board: np.ndarray, x: int, y: int, player: int) -> bool:
    """Freestyle Gomoku: a contiguous line of at least five wins."""
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


def _transform(x: int, y: int, symmetry: int) -> tuple[int, int]:
    """Apply one of the eight square-board symmetries."""
    n = BOARD_SIZE - 1
    if symmetry >= 4:
        x = n - x
        symmetry -= 4
    for _ in range(symmetry):
        x, y = n - y, x
    return x, y


def generate_opening(rng: np.random.Generator, plies: int) -> list[tuple[int, int]]:
    if not 0 <= plies <= BOARD_SIZE * BOARD_SIZE:
        raise ValueError(
            f"opening plies must be between 0 and {BOARD_SIZE * BOARD_SIZE}"
        )
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
            break
        x, y = choices[int(rng.integers(len(choices)))]
        board[y, x] = player
        moves.append((x, y))
        player = other(player)

    symmetry = int(rng.integers(8))
    return [_transform(x, y, symmetry) for x, y in moves]


def opening_board(moves: Iterable[tuple[int, int]]) -> tuple[np.ndarray, int]:
    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
    player = BLACK
    for x, y in moves:
        if not (0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE):
            raise ValueError(f"opening move outside board: {(x, y)}")
        if board[y, x] != EMPTY:
            raise ValueError(f"duplicate opening move: {(x, y)}")
        board[y, x] = player
        if is_win(board, x, y, player):
            raise ValueError("opening is already terminal")
        player = other(player)
    return board, player


def play_game(
    agent: AlphaZeroGomokuAgent,
    ddqk: DDQKAdapter,
    opening: list[tuple[int, int]],
    model_color: int,
    pair_index: int,
    max_moves: int,
) -> GameRecord:
    board, player = opening_board(opening)
    ddqk.reset()
    ddqk.sync_opening(opening, starting_player=BLACK)
    last_move = opening[-1] if opening else None
    moves = [[x, y, BLACK if index % 2 == 0 else WHITE] for index, (x, y) in enumerate(opening)]
    model_seconds = 0.0
    ddqk_seconds = 0.0
    model_moves = 0
    ddqk_moves = 0
    model_decision_reasons: list[str] = []
    winner = EMPTY
    termination = "truncated"
    error: str | None = None

    while len(moves) < max_moves and np.any(board == EMPTY):
        started = time.perf_counter()
        try:
            if player == model_color:
                move = agent.choose_move(board.tolist(), last_move, ai_color=player)
                model_moves += 1
                model_decision_reasons.append(agent.last_decision_reason or "unknown")
            else:
                move = ddqk.choose_move(board.tolist(), player, last_move)
                ddqk_moves += 1
                if ddqk.last_engine_error is not None:
                    raise RuntimeError(
                        "DDQK used fallback after engine failure: "
                        f"{ddqk.last_engine_error}"
                    )
        except Exception as exc:  # Preserve the game that exposed an adapter/engine failure.
            error = f"{type(exc).__name__}: {exc}"
            termination = "engine_error"
            break
        finally:
            elapsed = time.perf_counter() - started
            if player == model_color:
                model_seconds += elapsed
            else:
                ddqk_seconds += elapsed

        if move is None:
            error = f"player {player} returned no move"
            break
        x, y = map(int, move)
        if not (0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE):
            error = f"player {player} returned out-of-board move {(x, y)}"
            break
        if board[y, x] != EMPTY:
            error = f"player {player} returned occupied move {(x, y)}"
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

    if error is not None or termination == "truncated":
        model_result = None
    elif winner == EMPTY:
        model_result = 0.5
    else:
        model_result = 1.0 if winner == model_color else 0.0

    return GameRecord(
        pair_index=pair_index,
        model_color=model_color,
        opening=[[x, y] for x, y in opening],
        moves=moves,
        winner=winner,
        model_result=model_result,
        plies=len(moves),
        model_seconds=model_seconds,
        ddqk_seconds=ddqk_seconds,
        model_moves=model_moves,
        ddqk_moves=ddqk_moves,
        termination=termination,
        error=error,
        model_decision_reasons=model_decision_reasons,
    )


def paired_bootstrap_ci95(values: list[float], seed: int = 20260722) -> list[float]:
    """Return a deterministic percentile CI over color-swapped pair scores."""
    if not values:
        return [0.0, 1.0]
    if len(values) == 1:
        return [0.0, 1.0]
    samples = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(samples), size=(10_000, len(samples)))
    means = samples[indices].mean(axis=1)
    low, high = np.quantile(means, (0.025, 0.975))
    return [float(low), float(high)]


def bounded_mean_one_sided_lower95(values: list[float]) -> float:
    """Distribution-free one-sided 95% lower bound for a [0, 1] mean.

    The paired-opening score is the independent sampling unit.  Hoeffding's
    inequality is intentionally conservative and, unlike a percentile
    bootstrap, never reports an all-win sample as having lower bound 1.0.
    """

    if not values:
        return 0.0
    mean = statistics.fmean(values)
    radius = math.sqrt(math.log(1.0 / CONFIDENCE_ALPHA) / (2.0 * len(values)))
    return max(0.0, float(mean - radius))


def _binomial_upper_tail_probability(
    successes: int,
    trials: int,
    probability: float,
) -> float:
    """Return P[X >= successes] for X ~ Binomial(trials, probability).

    Terms are accumulated with log-sum-exp so the exact-binomial inversion is
    stable near zero and one without requiring SciPy.
    """

    if successes <= 0:
        return 1.0
    if successes > trials:
        return 0.0
    if probability <= 0.0:
        return 0.0
    if probability >= 1.0:
        return 1.0

    log_probability = math.log(probability)
    log_failure = math.log1p(-probability)
    log_terms = [
        math.lgamma(trials + 1)
        - math.lgamma(outcome + 1)
        - math.lgamma(trials - outcome + 1)
        + outcome * log_probability
        + (trials - outcome) * log_failure
        for outcome in range(successes, trials + 1)
    ]
    largest = max(log_terms)
    if largest < math.log(sys.float_info.min):
        return 0.0
    tail = math.exp(largest) * sum(math.exp(term - largest) for term in log_terms)
    return min(1.0, float(tail))


def exact_binomial_one_sided_lower95(successes: int, trials: int) -> float:
    """Exact one-sided 95% Clopper-Pearson lower confidence bound.

    ``successes`` is a Bernoulli count, not a fractional game score.  For this
    benchmark one success means that the model won *both* color-swapped games
    from an opening.  The returned value inverts the exact binomial upper-tail
    test, ``P_p[X >= successes] = 0.05``.
    """

    if isinstance(successes, bool) or not isinstance(successes, int):
        raise TypeError("successes must be an integer")
    if isinstance(trials, bool) or not isinstance(trials, int):
        raise TypeError("trials must be an integer")
    if trials < 0:
        raise ValueError("trials must be non-negative")
    if not 0 <= successes <= trials:
        raise ValueError("successes must be between zero and trials")
    if trials == 0 or successes == 0:
        return 0.0
    if successes == trials:
        # P[X = n] = p**n, so the exact lower limit has a closed form.
        return float(CONFIDENCE_ALPHA ** (1.0 / trials))

    low = 0.0
    high = successes / trials
    # The tail probability is monotone in p.  Eighty iterations are ample for
    # double precision; returning low keeps rounding conservative.
    for _ in range(80):
        midpoint = (low + high) / 2.0
        tail = _binomial_upper_tail_probability(successes, trials, midpoint)
        if tail < CONFIDENCE_ALPHA:
            low = midpoint
        else:
            high = midpoint
    return float(low)


def stable_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def evaluation_code_signature(base_dir: Path | None = None) -> dict[str, object]:
    """Hash every decision-making source file and a stable aggregate bundle."""

    root = (base_dir or Path(__file__).resolve().parent).resolve()
    file_hashes: dict[str, str] = {}
    for name in EVALUATION_CODE_FILES:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"evaluation source file not found: {path}")
        file_hashes[name] = sha256_file(path)
    return {
        "files": file_hashes,
        "bundle_sha256": stable_json_sha256(file_hashes),
    }


def ddqk_asset_signature(asset_dir: Path) -> dict[str, object]:
    """Hash every DDQK native/table asset required by the adapter."""

    root = asset_dir.resolve()
    file_hashes: dict[str, str] = {}
    for name in DDQK_DECISION_ASSET_FILES:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"DDQK decision asset not found: {path}")
        file_hashes[name] = sha256_file(path)
    return {
        "files": file_hashes,
        "bundle_sha256": stable_json_sha256(file_hashes),
    }


def summarize(
    records: list[GameRecord], requested_pairs: int | None = None
) -> dict[str, object]:
    completed = [
        record
        for record in records
        if record.error is None
        and record.termination in ("win", "full_board_draw")
        and record.model_result is not None
    ]
    grouped: dict[int, list[GameRecord]] = {}
    for record in completed:
        grouped.setdefault(record.pair_index, []).append(record)
    complete_pairs = {
        pair_index: pair_records
        for pair_index, pair_records in grouped.items()
        if len(pair_records) == 2
        and {record.model_color for record in pair_records} == {BLACK, WHITE}
    }
    scored = [record for pair_records in complete_pairs.values() for record in pair_records]
    pair_scores = [
        sum(float(record.model_result) for record in pair_records) / 2.0
        for _, pair_records in sorted(complete_pairs.items())
    ]
    pair_sweep_successes = sum(
        all(record.model_result == 1.0 for record in pair_records)
        for _, pair_records in sorted(complete_pairs.items())
    )
    pair_sweep_trials = len(complete_pairs)
    observed_score = statistics.fmean(pair_scores) if pair_scores else 0.0
    hoeffding_pair_score_lower95 = bounded_mean_one_sided_lower95(pair_scores)
    observed_pair_sweep_rate = (
        pair_sweep_successes / pair_sweep_trials if pair_sweep_trials else 0.0
    )
    exact_pair_sweep_lower95 = exact_binomial_one_sided_lower95(
        pair_sweep_successes,
        pair_sweep_trials,
    )
    wins = sum(record.model_result == 1.0 for record in scored)
    losses = sum(record.model_result == 0.0 for record in scored)
    draws = sum(record.model_result == 0.5 for record in scored)
    by_color: dict[str, dict[str, float | int]] = {}
    for color, name in ((BLACK, "black"), (WHITE, "white")):
        subset = [record for record in scored if record.model_color == color]
        score = (
            sum(float(record.model_result) for record in subset) / len(subset)
            if subset
            else 0.0
        )
        color_results = [float(record.model_result) for record in subset]
        by_color[name] = {
            "games": len(subset),
            "score": score,
            "one_sided_95_lower_bound": bounded_mean_one_sided_lower95(color_results),
        }
    model_move_count = sum(record.model_moves for record in records)
    ddqk_move_count = sum(record.ddqk_moves for record in records)
    decision_reasons: dict[str, int] = {}
    for record in records:
        for reason in record.model_decision_reasons:
            decision_reasons[reason] = decision_reasons.get(reason, 0) + 1
    if requested_pairs is None:
        requested_pairs = len({record.pair_index for record in records})
    return {
        "games": len(records),
        "completed_games": len(completed),
        "scored_games": len(scored),
        "requested_pairs": requested_pairs,
        "complete_pairs": len(complete_pairs),
        "incomplete_pairs": requested_pairs - len(complete_pairs),
        "errors": sum(record.error is not None for record in records),
        "truncated": sum(record.termination == "truncated" for record in records),
        "wins": wins,
        "losses": losses,
        "draws": draws,
        # Keep ``score`` and ``one_sided_95_lower_bound`` as compatibility
        # aliases.  The explicit fields below distinguish the three different
        # quantities used in reporting/certification.
        "score": observed_score,
        "observed_score": observed_score,
        "paired_bootstrap_ci95": paired_bootstrap_ci95(pair_scores),
        "one_sided_95_lower_bound": hoeffding_pair_score_lower95,
        "one_sided_95_lower_bound_method": {
            "name": "hoeffding_bounded_mean",
            "alpha": CONFIDENCE_ALPHA,
            "independent_unit": "paired_opening",
            "sample_size": len(pair_scores),
        },
        "hoeffding_bounded_pair_score_lower95": hoeffding_pair_score_lower95,
        "hoeffding_bounded_pair_score_lower95_method": {
            "name": "hoeffding_bounded_mean",
            "alpha": CONFIDENCE_ALPHA,
            "quantity": "mean_color_swapped_pair_score",
            "independent_unit": "paired_opening",
            "sample_size": len(pair_scores),
        },
        "pair_sweep_successes": pair_sweep_successes,
        "pair_sweep_trials": pair_sweep_trials,
        "observed_pair_sweep_rate": observed_pair_sweep_rate,
        "exact_pair_sweep_lower95": exact_pair_sweep_lower95,
        "exact_pair_sweep_lower95_method": {
            "name": "clopper_pearson_exact_binomial",
            "alpha": CONFIDENCE_ALPHA,
            "success_definition": "model_wins_both_color_swapped_games",
            "independent_unit": "paired_opening",
            "successes": pair_sweep_successes,
            "trials": pair_sweep_trials,
        },
        "by_color": by_color,
        "mean_plies": statistics.fmean(record.plies for record in completed) if completed else 0.0,
        "model_seconds_per_move": (
            sum(record.model_seconds for record in records) / model_move_count
            if model_move_count
            else 0.0
        ),
        "ddqk_seconds_per_move": (
            sum(record.ddqk_seconds for record in records) / ddqk_move_count
            if ddqk_move_count
            else 0.0
        ),
        "model_decision_reasons": dict(sorted(decision_reasons.items())),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_checkpoint(path: Path) -> None:
    """Fail before spawning workers when the checkpoint has the wrong rules."""

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    raw_config = checkpoint.get("config")
    if not isinstance(raw_config, dict):
        raise SystemExit("checkpoint does not contain a valid config")
    board_size = int(raw_config.get("board_size", -1))
    win_length = int(raw_config.get("win_length", -1))
    if board_size != BOARD_SIZE or win_length != 5:
        raise SystemExit(
            "benchmark requires a 19x19 freestyle-five checkpoint; got "
            f"{board_size}x{board_size}, win_length={win_length}"
        )


def atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def build_report(
    *,
    args: argparse.Namespace,
    signature: dict[str, object],
    openings: list[list[tuple[int, int]]],
    records: list[GameRecord],
) -> dict[str, object]:
    summary = summarize(records, requested_pairs=args.pairs)
    final_requirements_passed = (
        args.certification_mode == FINAL_CERTIFICATION_MODE
        and summary["complete_pairs"] == args.pairs
        and summary["incomplete_pairs"] == 0
        and summary["errors"] == 0
        and summary["truncated"] == 0
        and args.pairs >= FINAL_MIN_PAIRS
        and float(summary["score"]) >= FINAL_MIN_SCORE
        and float(summary["by_color"]["black"]["score"]) >= FINAL_MIN_COLOR_SCORE
        and float(summary["by_color"]["white"]["score"]) >= FINAL_MIN_COLOR_SCORE
        and float(summary["exact_pair_sweep_lower95"])
        >= FINAL_MIN_EXACT_PAIR_SWEEP_LOWER95
    )
    return {
        "format_version": 3,
        "signature": signature,
        "checkpoint": str(args.checkpoint.resolve()),
        "ddqk_source": signature["ddqk_source"],
        "pairs": args.pairs,
        "opening_plies": args.opening_plies,
        "simulations": args.simulations,
        "max_moves": args.max_moves,
        "seed": args.seed,
        "workers": args.workers,
        "openings": [[[x, y] for x, y in opening] for opening in openings],
        "summary": summary,
        "certification": {
            "mode": args.certification_mode,
            "status": (
                "benchmark_final_requirements_passed"
                if final_requirements_passed
                else "not_final_certified"
            ),
            "final_certified": bool(final_requirements_passed),
            "requirements": {
                "minimum_independent_paired_openings": FINAL_MIN_PAIRS,
                "minimum_observed_score": FINAL_MIN_SCORE,
                "minimum_observed_black_score": FINAL_MIN_COLOR_SCORE,
                "minimum_observed_white_score": FINAL_MIN_COLOR_SCORE,
                "minimum_exact_pair_sweep_one_sided_95_lower_bound": (
                    FINAL_MIN_EXACT_PAIR_SWEEP_LOWER95
                ),
                "pair_sweep_success_definition": (
                    "model_wins_both_color_swapped_games"
                ),
                "requires_zero_errors": True,
                "requires_zero_truncated_games": True,
            },
        },
        "games": [asdict(record) for record in records],
    }


def print_game_status(record: GameRecord, pairs: int) -> None:
    status = (
        f"error={record.error}"
        if record.error
        else record.termination
        if record.model_result is None
        else f"result={record.model_result:g}"
    )
    print(
        f"pair={record.pair_index + 1}/{pairs} "
        f"model_color={'black' if record.model_color == BLACK else 'white'} "
        f"{status} plies={record.plies}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(__file__).resolve().parent / "latest.pt",
    )
    parser.add_argument("--ddqk-source", type=Path, default=None)
    parser.add_argument("--pairs", type=int, default=1)
    parser.add_argument(
        "--certification-mode",
        choices=(DEVELOPMENT_MODE, FINAL_CERTIFICATION_MODE),
        default=DEVELOPMENT_MODE,
        help=(
            "development permits small screening runs; final-certification "
            f"requires at least {FINAL_MIN_PAIRS} independent paired openings"
        ),
    )
    parser.add_argument("--opening-plies", type=int, default=4)
    parser.add_argument("--simulations", type=int, default=64)
    parser.add_argument("--max-moves", type=int, default=BOARD_SIZE * BOARD_SIZE)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="worker processes; each loads one model and one DDQK engine (default: 1)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume an interrupted report only when its immutable signature matches",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "ddqk_benchmark.json",
    )
    args = parser.parse_args()
    if args.pairs <= 0:
        raise SystemExit("--pairs must be positive")
    if (
        args.certification_mode == FINAL_CERTIFICATION_MODE
        and args.pairs < FINAL_MIN_PAIRS
    ):
        raise SystemExit(
            "--certification-mode final-certification requires "
            f"--pairs >= {FINAL_MIN_PAIRS}"
        )
    if args.simulations <= 0:
        raise SystemExit("--simulations must be positive")
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    if not 0 <= args.opening_plies <= BOARD_SIZE * BOARD_SIZE:
        raise SystemExit(f"--opening-plies must be in [0, {BOARD_SIZE * BOARD_SIZE}]")
    if not args.opening_plies <= args.max_moves <= BOARD_SIZE * BOARD_SIZE:
        raise SystemExit(
            f"--max-moves must be between opening plies and {BOARD_SIZE * BOARD_SIZE}"
        )
    args.checkpoint = args.checkpoint.resolve()
    if not args.checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")
    validate_checkpoint(args.checkpoint)

    rng = np.random.default_rng(args.seed)
    openings = [generate_opening(rng, args.opening_plies) for _ in range(args.pairs)]
    serialized_openings = [[[x, y] for x, y in opening] for opening in openings]
    if (
        args.certification_mode == FINAL_CERTIFICATION_MODE
        and len({stable_json_sha256(opening) for opening in serialized_openings})
        != len(serialized_openings)
    ):
        raise SystemExit(
            "final-certification requires distinct independently generated openings; "
            "choose another seed or opening length"
        )
    # Resolve and validate DDQK once in the coordinator.  Spawned Windows
    # workers do not inherit this singleton; each initializer creates its own.
    _, adapter_class = _runtime_classes()
    coordinator_ddqk = adapter_class(args.ddqk_source)
    resolved_ddqk_source = coordinator_ddqk.source_path
    ddqk_assets = ddqk_asset_signature(coordinator_ddqk.asset_dir)
    signature: dict[str, object] = {
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "ddqk_source": str(resolved_ddqk_source),
        "ddqk_source_sha256": sha256_file(resolved_ddqk_source),
        "ddqk_dll_sha256": sha256_file(coordinator_ddqk.asset_dir / "dll.so"),
        "ddqk_assets": ddqk_assets,
        "ddqk_depth": int(coordinator_ddqk._engine.DEPTH),
        "evaluation_code": evaluation_code_signature(),
        "opening_manifest_sha256": stable_json_sha256(serialized_openings),
        "certification_mode": args.certification_mode,
        "pairs": args.pairs,
        "opening_plies": args.opening_plies,
        "simulations": args.simulations,
        "max_moves": args.max_moves,
        "seed": args.seed,
    }
    records: list[GameRecord] = []
    if args.resume and args.output.is_file():
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        if previous.get("signature") != signature:
            raise SystemExit("refusing to resume: existing report signature does not match")
        if previous.get("openings") != [
            [[x, y] for x, y in opening] for opening in openings
        ]:
            raise SystemExit("refusing to resume: existing opening manifest does not match")
        try:
            if __package__ in (None, ""):
                from ddqk_replay_export import validate_benchmark_report
            else:
                from .ddqk_replay_export import validate_benchmark_report

            validate_benchmark_report(previous)
        except ValueError as exc:
            raise SystemExit(
                f"refusing to resume: existing game records failed replay audit: {exc}"
            ) from exc
        records = [GameRecord(**item) for item in previous.get("games", [])]

    task_order = [
        (pair_index, model_color)
        for pair_index in range(args.pairs)
        for model_color in (BLACK, WHITE)
    ]
    expected_keys = set(task_order)
    records_by_key: dict[tuple[int, int], GameRecord] = {}
    for record in records:
        key = (record.pair_index, record.model_color)
        if key not in expected_keys:
            raise SystemExit(f"refusing to resume: report contains unexpected game {key}")
        if key in records_by_key:
            raise SystemExit(f"refusing to resume: report contains duplicate game {key}")
        records_by_key[key] = record
    resumed_keys = set(records_by_key)
    pending = [key for key in task_order if key not in records_by_key]

    next_print_index = 0
    while (
        next_print_index < len(task_order)
        and task_order[next_print_index] in resumed_keys
    ):
        next_print_index += 1

    if pending:
        spawn_context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=spawn_context,
            initializer=initialize_worker,
            initargs=(
                str(args.checkpoint),
                args.simulations,
                str(resolved_ddqk_source),
            ),
        ) as executor:
            futures: dict[Future[GameRecord], tuple[int, int]] = {}
            for pair_index, model_color in pending:
                future = executor.submit(
                    run_worker_game,
                    pair_index,
                    model_color,
                    openings[pair_index],
                    args.max_moves,
                )
                futures[future] = (pair_index, model_color)

            for future in as_completed(futures):
                expected_key = futures[future]
                record = future.result()
                actual_key = (record.pair_index, record.model_color)
                if actual_key != expected_key:
                    raise RuntimeError(
                        f"worker returned game {actual_key}, expected {expected_key}"
                    )
                records_by_key[actual_key] = record
                ordered_records = [
                    records_by_key[key]
                    for key in task_order
                    if key in records_by_key
                ]
                # Only the coordinator writes.  os.replace makes each finished
                # game a durable, resume-safe progress checkpoint.
                report = build_report(
                    args=args,
                    signature=signature,
                    openings=openings,
                    records=ordered_records,
                )
                atomic_write_json(args.output, report)

                # Worker completion order is nondeterministic.  Delay console
                # output until all earlier task keys are available.
                while (
                    next_print_index < len(task_order)
                    and task_order[next_print_index] in records_by_key
                ):
                    print_key = task_order[next_print_index]
                    if print_key not in resumed_keys:
                        print_game_status(records_by_key[print_key], args.pairs)
                    next_print_index += 1

    records = [records_by_key[key] for key in task_order if key in records_by_key]
    report = build_report(
        args=args,
        signature=signature,
        openings=openings,
        records=records,
    )
    atomic_write_json(args.output, report)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"report={args.output.resolve()}")


if __name__ == "__main__":
    main()

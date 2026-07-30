#!/usr/bin/env python3
"""Build a leakage-safe white-defense curriculum from DDQK format-3 games.

Only completed benchmark games in which the model played white and lost are
eligible.  Positions are sampled *before* selected late white decisions.  The
target policy is uniform over every action in ``GomokuGame.search_actions``
that is not proven to lose by the configured bounded tactical checks:

* no immediate black win after the white move;
* no exact black win in at most three plies; and
* no black VCF proven within the configured 5/7-ply budgets.

``UNKNOWN_BUDGET`` is deliberately retained as a policy action and recorded in
``vcf_unknown_mask``.  It is not evidence that an action is unsafe.  Likewise,
the resulting label is a bounded non-loss filter over the radius-limited search
candidate set; it is not a proof against arbitrary VCT play.

Train/eval assignment is made by original paired-opening group before any game
is replayed or tactically labelled.  Separate NPZ files are written so the eval
archive cannot accidentally be consumed by the supervised trainer.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

try:
    from .ddqk_replay_export import validate_benchmark_report
    from .tactical_solver import (
        BLACK as TACTICAL_BLACK,
        WHITE as TACTICAL_WHITE,
        FreestyleBoard,
        SolveLimits,
        SolveStatus,
        TacticalSolver,
    )
    from .train_alphazero import (
        BLACK as TRAIN_BLACK,
        EMPTY as TRAIN_EMPTY,
        WHITE as TRAIN_WHITE,
        GomokuGame,
    )
except ImportError:  # pragma: no cover - direct script execution convenience.
    from ddqk_replay_export import validate_benchmark_report  # type: ignore
    from tactical_solver import (  # type: ignore
        BLACK as TACTICAL_BLACK,
        WHITE as TACTICAL_WHITE,
        FreestyleBoard,
        SolveLimits,
        SolveStatus,
        TacticalSolver,
    )
    from train_alphazero import (  # type: ignore
        BLACK as TRAIN_BLACK,
        EMPTY as TRAIN_EMPTY,
        WHITE as TRAIN_WHITE,
        GomokuGame,
    )


BOARD_SIZE = 19
ACTION_COUNT = BOARD_SIZE * BOARD_SIZE
WIN_LENGTH = 5
BENCHMARK_BLACK = 1
BENCHMARK_WHITE = 2
BENCHMARK_EMPTY = 0
TRAIN_SPLIT = "train"
EVAL_SPLIT = "eval"
SCHEMA_VERSION = 1
DEFAULT_DECISION_DISTANCES = (2, 4, 6, 8, 10, 12)
DEFAULT_VCF_PLIES = (5, 7)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _unicode(values: Sequence[object]) -> np.ndarray:
    strings = [str(value) for value in values]
    width = max(1, *(len(value) for value in strings))
    return np.asarray(strings, dtype=f"<U{width}")


@dataclass(frozen=True)
class WhiteDefenseConfig:
    """Deterministic extraction and bounded-oracle settings."""

    decision_distances: tuple[int, ...] = DEFAULT_DECISION_DISTANCES
    eval_fraction: float = 0.20
    split_seed: int = 20260802
    candidate_radius: int = 2
    vcf_plies: tuple[int, ...] = DEFAULT_VCF_PLIES
    vcf_max_nodes: int = 50_000
    # Published curriculum generation defaults to a deterministic node budget.
    # A non-zero wall-clock limit is useful only for exploratory runs because
    # host load can otherwise make identical states receive different UNKNOWN
    # outcomes across independently generated reports.
    vcf_time_ms: float = 0.0

    def validate(self) -> None:
        distances = tuple(map(int, self.decision_distances))
        if not distances or any(distance <= 0 for distance in distances):
            raise ValueError("decision distances must be positive")
        if len(set(distances)) != len(distances):
            raise ValueError("decision distances must be unique")
        if not 0.0 <= float(self.eval_fraction) < 0.5:
            raise ValueError("eval_fraction must be in [0, 0.5)")
        if self.candidate_radius < 1:
            raise ValueError("candidate_radius must be positive")
        depths = tuple(map(int, self.vcf_plies))
        if not depths or any(depth <= 0 for depth in depths):
            raise ValueError("VCF ply limits must be positive")
        if tuple(sorted(set(depths))) != depths:
            raise ValueError("VCF ply limits must be unique and increasing")
        if self.vcf_max_nodes <= 0:
            raise ValueError("vcf_max_nodes must be positive")
        if not math.isfinite(float(self.vcf_time_ms)) or self.vcf_time_ms < 0:
            raise ValueError("vcf_time_ms must be finite and non-negative")


@dataclass(frozen=True)
class ActionLabels:
    candidate_actions: tuple[int, ...]
    safe_actions: tuple[int, ...]
    unknown_actions: tuple[int, ...]
    unsafe_immediate_actions: tuple[int, ...]
    unsafe_three_ply_actions: tuple[int, ...]
    unsafe_vcf_actions: tuple[int, ...]
    vcf_nodes: int
    vcf_queries: int


@dataclass(frozen=True)
class WhiteDefenseDataset:
    """Trainer arrays plus sufficient provenance to independently audit them."""

    states: np.ndarray
    policies: np.ndarray
    values: np.ndarray
    policy_weights: np.ndarray
    value_weights: np.ndarray
    source: np.ndarray
    priority: np.ndarray
    group_id: np.ndarray
    split: np.ndarray
    report_sha256: np.ndarray
    opening_sha256: np.ndarray
    game_index: np.ndarray
    pair_index: np.ndarray
    ply_index: np.ndarray
    white_decision_distance: np.ndarray
    original_action: np.ndarray
    original_action_in_candidates: np.ndarray
    original_action_safe: np.ndarray
    last_action: np.ndarray
    move_count: np.ndarray
    move_history: np.ndarray
    state_hash: np.ndarray
    candidate_mask: np.ndarray
    safe_mask: np.ndarray
    vcf_unknown_mask: np.ndarray
    unsafe_immediate_mask: np.ndarray
    unsafe_three_ply_mask: np.ndarray
    unsafe_vcf_mask: np.ndarray
    candidate_count: np.ndarray
    safe_count: np.ndarray
    unsafe_count: np.ndarray
    vcf_unknown_count: np.ndarray
    unsafe_immediate_count: np.ndarray
    unsafe_three_ply_count: np.ndarray
    unsafe_vcf_count: np.ndarray
    vcf_nodes: np.ndarray
    vcf_queries: np.ndarray
    candidate_radius: np.ndarray
    vcf_max_plies: np.ndarray
    summary: dict[str, object]

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if field.name != "summary"
        }

    def subset(self, split: str) -> "WhiteDefenseDataset":
        if split not in (TRAIN_SPLIT, EVAL_SPLIT):
            raise ValueError(f"unknown split: {split}")
        mask = self.split == split
        arrays = {name: array[mask] for name, array in self.arrays().items()}
        summary = dict(self.summary)
        summary["exported_split"] = split
        summary["exported_records"] = int(mask.sum())
        return WhiteDefenseDataset(summary=summary, **arrays)


@dataclass(frozen=True)
class _DecisionSnapshot:
    game: GomokuGame
    ply_index: int
    original_action: int
    move_history: tuple[int, ...]


def _benchmark_to_training_player(player: int) -> int:
    if player == BENCHMARK_BLACK:
        return TRAIN_BLACK
    if player == BENCHMARK_WHITE:
        return TRAIN_WHITE
    raise ValueError(f"invalid benchmark player: {player}")


def _eligible_white_loss(raw_game: object) -> bool:
    if not isinstance(raw_game, dict):
        return False
    try:
        return (
            int(raw_game.get("model_color", -1)) == BENCHMARK_WHITE
            and int(raw_game.get("winner", -1)) == BENCHMARK_BLACK
            and float(raw_game.get("model_result", float("nan"))) == 0.0
            and raw_game.get("termination") == "win"
            and raw_game.get("error") is None
        )
    except (TypeError, ValueError):
        return False


def _assign_group_splits(
    group_ids: Sequence[str],
    *,
    eval_fraction: float,
    seed: int,
) -> dict[str, str]:
    """Assign whole opening groups before replay/tactical processing."""

    unique = np.asarray(sorted(set(group_ids)), dtype=str)
    if unique.size == 0:
        return {}
    rng = np.random.default_rng(seed)
    shuffled = unique[rng.permutation(len(unique))]
    if len(shuffled) <= 1 or eval_fraction == 0:
        eval_count = 0
    else:
        eval_count = max(1, round(len(shuffled) * eval_fraction))
        eval_count = min(eval_count, len(shuffled) - 1)
    eval_groups = set(map(str, shuffled[:eval_count]))
    return {
        str(group): EVAL_SPLIT if str(group) in eval_groups else TRAIN_SPLIT
        for group in unique
    }


def _replay_losing_white_game(
    raw_game: Mapping[str, object],
    *,
    game_index: int,
) -> list[_DecisionSnapshot]:
    """Strictly replay one eligible game and retain pre-white-move snapshots."""

    if not _eligible_white_loss(raw_game):
        raise ValueError(f"game {game_index} is not a completed white-model loss")
    raw_moves = raw_game.get("moves")
    opening = raw_game.get("opening")
    if not isinstance(raw_moves, list) or not isinstance(opening, list):
        raise ValueError(f"game {game_index}: moves/opening must be lists")
    try:
        reported_plies = int(raw_game["plies"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"game {game_index}: invalid plies") from exc
    if reported_plies != len(raw_moves):
        raise ValueError(f"game {game_index}: plies do not match move history")

    game = GomokuGame(BOARD_SIZE, WIN_LENGTH)
    history: list[int] = []
    decisions: list[_DecisionSnapshot] = []
    for ply_index, raw_move in enumerate(raw_moves):
        if not isinstance(raw_move, (list, tuple)) or len(raw_move) != 3:
            raise ValueError(f"game {game_index} ply {ply_index}: malformed move")
        try:
            x, y, benchmark_player = map(int, raw_move)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"game {game_index} ply {ply_index}: non-integer move"
            ) from exc
        expected_benchmark_player = (
            BENCHMARK_BLACK if ply_index % 2 == 0 else BENCHMARK_WHITE
        )
        if benchmark_player != expected_benchmark_player:
            raise ValueError(
                f"game {game_index} ply {ply_index}: expected player "
                f"{expected_benchmark_player}, got {benchmark_player}"
            )
        if not (0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE):
            raise ValueError(
                f"game {game_index} ply {ply_index}: point outside board {(x, y)}"
            )
        if ply_index < len(opening):
            raw_opening_move = opening[ply_index]
            if (
                not isinstance(raw_opening_move, (list, tuple))
                or len(raw_opening_move) != 2
                or list(map(int, raw_opening_move)) != [x, y]
            ):
                raise ValueError(
                    f"game {game_index} ply {ply_index}: opening prefix mismatch"
                )
        if game.terminal:
            raise ValueError(f"game {game_index}: contains moves after terminal state")
        training_player = _benchmark_to_training_player(benchmark_player)
        if game.player != training_player:
            raise ValueError(
                f"game {game_index} ply {ply_index}: replay side-to-move mismatch"
            )
        action = y * BOARD_SIZE + x
        if int(game.board[y, x]) != TRAIN_EMPTY:
            raise ValueError(
                f"game {game_index} ply {ply_index}: repeated point {(x, y)}"
            )
        if ply_index >= len(opening) and training_player == TRAIN_WHITE:
            decisions.append(
                _DecisionSnapshot(
                    game=game.clone(),
                    ply_index=ply_index,
                    original_action=action,
                    move_history=tuple(history),
                )
            )
        game.play(action)
        history.append(action)

    if not game.terminal or game.winner != TRAIN_BLACK:
        raise ValueError(f"game {game_index}: terminal replay is not a black win")
    if int(raw_game["winner"]) != BENCHMARK_BLACK:
        raise ValueError(f"game {game_index}: reported winner disagrees with replay")
    return decisions


def label_safe_actions(
    game: GomokuGame,
    config: WhiteDefenseConfig,
    *,
    solver: TacticalSolver | object | None = None,
) -> ActionLabels:
    """Classify every radius-limited root candidate with bounded proofs.

    A VCF budget exhaustion leaves the candidate in ``safe_actions`` and also
    records it in ``unknown_actions``.  Here "safe" therefore means "not
    proven unsafe by these configured checks", not a full-game guarantee.
    """

    config.validate()
    if game.size != BOARD_SIZE or game.win_length != WIN_LENGTH:
        raise ValueError("white-defense labels require 19x19 freestyle-five")
    if game.terminal or game.player != TRAIN_WHITE:
        raise ValueError("white-defense labels require a non-terminal white turn")
    solver = solver or TacticalSolver(board_size=BOARD_SIZE, win_length=WIN_LENGTH)
    candidate_actions = tuple(
        map(int, game.search_actions(config.candidate_radius).tolist())
    )
    if not candidate_actions:
        raise ValueError("search candidate set is empty")
    position = FreestyleBoard.from_board(
        game.board, size=BOARD_SIZE, win_length=WIN_LENGTH
    )

    safe: list[int] = []
    unknown: list[int] = []
    immediate: list[int] = []
    three_ply: list[int] = []
    vcf: list[int] = []
    total_nodes = 0
    query_count = 0
    for action in candidate_actions:
        after = position.with_move(action, TACTICAL_WHITE)
        # An immediate white win ends the game before black can reply.
        if after.has_five(TACTICAL_WHITE):
            safe.append(action)
            continue
        if tuple(solver.immediate_wins(after, TACTICAL_BLACK)):  # type: ignore[attr-defined]
            immediate.append(action)
            continue
        if tuple(solver.forced_wins_in_three(after, TACTICAL_BLACK)):  # type: ignore[attr-defined]
            three_ply.append(action)
            continue

        action_unknown = False
        action_loses = False
        for depth in config.vcf_plies:
            limits = SolveLimits(
                max_plies=int(depth),
                max_nodes=config.vcf_max_nodes,
                time_ms=config.vcf_time_ms,
            )
            result = solver.solve_vcf(after, TACTICAL_BLACK, limits)  # type: ignore[attr-defined]
            query_count += 1
            total_nodes += int(result.nodes)
            status = SolveStatus(result.status)
            if status is SolveStatus.PROVEN_WIN:
                vcf.append(action)
                action_loses = True
                break
            if status is SolveStatus.UNKNOWN_BUDGET:
                action_unknown = True
        if action_loses:
            continue
        safe.append(action)
        if action_unknown:
            unknown.append(action)

    return ActionLabels(
        candidate_actions=tuple(candidate_actions),
        safe_actions=tuple(safe),
        unknown_actions=tuple(unknown),
        unsafe_immediate_actions=tuple(immediate),
        unsafe_three_ply_actions=tuple(three_ply),
        unsafe_vcf_actions=tuple(vcf),
        vcf_nodes=total_nodes,
        vcf_queries=query_count,
    )


def _action_mask(actions: Sequence[int]) -> np.ndarray:
    mask = np.zeros(ACTION_COUNT, dtype=np.uint8)
    if actions:
        mask[np.asarray(actions, dtype=np.int32)] = 1
    return mask


def _policy(actions: Sequence[int]) -> np.ndarray:
    unique = tuple(sorted(set(map(int, actions))))
    if not unique:
        raise ValueError("cannot make a policy from an empty safe set")
    result = np.zeros(ACTION_COUNT, dtype=np.float32)
    result[np.asarray(unique, dtype=np.int32)] = np.float32(1.0 / len(unique))
    residue = np.float32(1.0 - float(result.sum(dtype=np.float64)))
    result[unique[0]] += residue
    return result


def _record_arrays(records: list[dict[str, object]]) -> dict[str, np.ndarray]:
    if not records:
        raise ValueError("no salvageable white-defense positions were found")

    def numeric(name: str, dtype: object) -> np.ndarray:
        return np.asarray([record[name] for record in records], dtype=dtype)

    return {
        "states": np.stack([record["state"] for record in records]).astype(np.uint8),
        "policies": np.stack([record["policy"] for record in records]).astype(np.float16),
        "values": numeric("value", np.float32),
        "policy_weights": numeric("policy_weight", np.float32),
        "value_weights": numeric("value_weight", np.float32),
        "source": _unicode([record["source"] for record in records]),
        "priority": numeric("priority", np.float32),
        "group_id": _unicode([record["group_id"] for record in records]),
        "split": _unicode([record["split"] for record in records]),
        "report_sha256": _unicode([record["report_sha256"] for record in records]),
        "opening_sha256": _unicode([record["opening_sha256"] for record in records]),
        "game_index": numeric("game_index", np.int32),
        "pair_index": numeric("pair_index", np.int32),
        "ply_index": numeric("ply_index", np.int16),
        "white_decision_distance": numeric("white_decision_distance", np.int16),
        "original_action": numeric("original_action", np.int16),
        "original_action_in_candidates": numeric(
            "original_action_in_candidates", np.uint8
        ),
        "original_action_safe": numeric("original_action_safe", np.uint8),
        "last_action": numeric("last_action", np.int16),
        "move_count": numeric("move_count", np.uint16),
        "move_history": np.stack([record["move_history"] for record in records]).astype(
            np.int16
        ),
        "state_hash": _unicode([record["state_hash"] for record in records]),
        "candidate_mask": np.stack([record["candidate_mask"] for record in records]).astype(
            np.uint8
        ),
        "safe_mask": np.stack([record["safe_mask"] for record in records]).astype(
            np.uint8
        ),
        "vcf_unknown_mask": np.stack(
            [record["vcf_unknown_mask"] for record in records]
        ).astype(np.uint8),
        "unsafe_immediate_mask": np.stack(
            [record["unsafe_immediate_mask"] for record in records]
        ).astype(np.uint8),
        "unsafe_three_ply_mask": np.stack(
            [record["unsafe_three_ply_mask"] for record in records]
        ).astype(np.uint8),
        "unsafe_vcf_mask": np.stack(
            [record["unsafe_vcf_mask"] for record in records]
        ).astype(np.uint8),
        "candidate_count": numeric("candidate_count", np.int16),
        "safe_count": numeric("safe_count", np.int16),
        "unsafe_count": numeric("unsafe_count", np.int16),
        "vcf_unknown_count": numeric("vcf_unknown_count", np.int16),
        "unsafe_immediate_count": numeric("unsafe_immediate_count", np.int16),
        "unsafe_three_ply_count": numeric("unsafe_three_ply_count", np.int16),
        "unsafe_vcf_count": numeric("unsafe_vcf_count", np.int16),
        "vcf_nodes": numeric("vcf_nodes", np.uint64),
        "vcf_queries": numeric("vcf_queries", np.uint32),
        "candidate_radius": numeric("candidate_radius", np.uint8),
        "vcf_max_plies": numeric("vcf_max_plies", np.uint8),
    }


def build_dataset_from_validated_report(
    report: Mapping[str, object],
    *,
    report_sha256: str,
    config: WhiteDefenseConfig | None = None,
    solver: TacticalSolver | object | None = None,
    benchmark_audit: Mapping[str, object] | None = None,
) -> WhiteDefenseDataset:
    """Build arrays from an already signature-validated format-3 report.

    Normal callers should use :func:`generate_white_defense_dataset`, which
    performs format-3 signature and summary validation before entering here.
    This lower-level function remains public to allow deterministic unit tests
    and trusted in-memory pipelines without writing a temporary report.
    """

    config = config or WhiteDefenseConfig()
    config.validate()
    if int(report.get("format_version", -1)) != 3:
        raise ValueError("white-defense extraction requires benchmark format_version 3")
    if len(report_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in report_sha256.lower()
    ):
        raise ValueError("report_sha256 must be a SHA256 hex digest")
    raw_games = report.get("games")
    if not isinstance(raw_games, list):
        raise ValueError("benchmark games must be a list")

    eligible: list[tuple[int, Mapping[str, object], str, str, int]] = []
    for game_index, raw_game in enumerate(raw_games):
        if not _eligible_white_loss(raw_game):
            continue
        assert isinstance(raw_game, dict)
        try:
            pair_index = int(raw_game["pair_index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"game {game_index}: invalid pair_index") from exc
        opening = raw_game.get("opening")
        if not isinstance(opening, list):
            raise ValueError(f"game {game_index}: opening must be a list")
        opening_sha = stable_json_sha256(opening)
        group_id = (
            f"report={report_sha256.lower()}|pair={pair_index}|opening={opening_sha}"
        )
        eligible.append((game_index, raw_game, group_id, opening_sha, pair_index))
    if not eligible:
        raise ValueError("report contains no completed white-model losses")

    # This must happen before replay, tactical filtering, or any future
    # augmentation.  Even a game whose every sampled state is later discarded
    # has already had its immutable opening group assigned to one split.
    # Identical opening coordinates are one leakage unit even if a development
    # benchmark happened to generate them under two different pair indices.
    split_unit_by_group = {
        group_id: f"report={report_sha256.lower()}|opening={opening_sha}"
        for _, _, group_id, opening_sha, _ in eligible
    }
    split_by_unit = _assign_group_splits(
        list(split_unit_by_group.values()),
        eval_fraction=config.eval_fraction,
        seed=config.split_seed,
    )
    split_by_group = {
        group_id: split_by_unit[split_unit]
        for group_id, split_unit in split_unit_by_group.items()
    }
    solver = solver or TacticalSolver(board_size=BOARD_SIZE, win_length=WIN_LENGTH)

    records: list[dict[str, object]] = []
    # A position can be reached by more than one recorded move order.  Tactical
    # wall-clock budgets must not let the same encoded state receive mutually
    # contradictory labels inside one artifact, so label each exact state once
    # and reuse that result for every duplicate occurrence.
    label_cache: dict[str, ActionLabels] = {}
    label_cache_hits = 0
    skipped_short_distances = 0
    dropped_no_safe_within_bounded_candidates = 0
    labelled_positions = 0
    for game_index, raw_game, group_id, opening_sha, pair_index in eligible:
        decisions = _replay_losing_white_game(raw_game, game_index=game_index)
        for distance in config.decision_distances:
            if distance > len(decisions):
                skipped_short_distances += 1
                continue
            snapshot = decisions[-distance]
            state = snapshot.game.encode()
            state_hash = _sha256_bytes(state.tobytes())
            labels = label_cache.get(state_hash)
            if labels is None:
                labels = label_safe_actions(snapshot.game, config, solver=solver)
                label_cache[state_hash] = labels
                record_vcf_nodes = labels.vcf_nodes
                record_vcf_queries = labels.vcf_queries
            else:
                label_cache_hits += 1
                # Reused proof work costs no additional solver nodes/queries.
                record_vcf_nodes = 0
                record_vcf_queries = 0
            labelled_positions += 1
            if not labels.safe_actions:
                dropped_no_safe_within_bounded_candidates += 1
                continue
            padded_history = np.full(ACTION_COUNT, -1, dtype=np.int16)
            padded_history[: len(snapshot.move_history)] = np.asarray(
                snapshot.move_history, dtype=np.int16
            )
            candidate_mask = _action_mask(labels.candidate_actions)
            safe_mask = _action_mask(labels.safe_actions)
            unknown_mask = _action_mask(labels.unknown_actions)
            immediate_mask = _action_mask(labels.unsafe_immediate_actions)
            three_mask = _action_mask(labels.unsafe_three_ply_actions)
            vcf_mask = _action_mask(labels.unsafe_vcf_actions)
            split = split_by_group[group_id]
            source = (
                f"white_defense|report={report_sha256[:16]}|pair={pair_index}"
                f"|game={game_index}|ply={snapshot.ply_index}|distance={distance}"
            )
            records.append(
                {
                    "state": state,
                    "policy": _policy(labels.safe_actions),
                    "value": 0.0,
                    "policy_weight": 1.0,
                    # A bounded safety filter supplies no honest game value.
                    "value_weight": 0.0,
                    "source": source,
                    # Late, directly actionable rescues receive more sampling
                    # mass while every original game remains group-balanced.
                    "priority": 1.0 + 1.0 / float(distance),
                    "group_id": group_id,
                    "split": split,
                    "report_sha256": report_sha256.lower(),
                    "opening_sha256": opening_sha,
                    "game_index": game_index,
                    "pair_index": pair_index,
                    "ply_index": snapshot.ply_index,
                    "white_decision_distance": distance,
                    "original_action": snapshot.original_action,
                    "original_action_in_candidates": int(
                        snapshot.original_action in labels.candidate_actions
                    ),
                    "original_action_safe": int(
                        snapshot.original_action in labels.safe_actions
                    ),
                    "last_action": snapshot.game.last_action,
                    "move_count": snapshot.game.move_count,
                    "move_history": padded_history,
                    "state_hash": state_hash,
                    "candidate_mask": candidate_mask,
                    "safe_mask": safe_mask,
                    "vcf_unknown_mask": unknown_mask,
                    "unsafe_immediate_mask": immediate_mask,
                    "unsafe_three_ply_mask": three_mask,
                    "unsafe_vcf_mask": vcf_mask,
                    "candidate_count": len(labels.candidate_actions),
                    "safe_count": len(labels.safe_actions),
                    "unsafe_count": (
                        len(labels.unsafe_immediate_actions)
                        + len(labels.unsafe_three_ply_actions)
                        + len(labels.unsafe_vcf_actions)
                    ),
                    "vcf_unknown_count": len(labels.unknown_actions),
                    "unsafe_immediate_count": len(labels.unsafe_immediate_actions),
                    "unsafe_three_ply_count": len(labels.unsafe_three_ply_actions),
                    "unsafe_vcf_count": len(labels.unsafe_vcf_actions),
                    "vcf_nodes": record_vcf_nodes,
                    "vcf_queries": record_vcf_queries,
                    "candidate_radius": config.candidate_radius,
                    "vcf_max_plies": max(config.vcf_plies),
                }
            )

    arrays = _record_arrays(records)
    assigned_counts = {
        split: sum(value == split for value in split_by_group.values())
        for split in (TRAIN_SPLIT, EVAL_SPLIT)
    }
    exported_counts = {
        split: int(np.count_nonzero(arrays["split"] == split))
        for split in (TRAIN_SPLIT, EVAL_SPLIT)
    }
    summary: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "source": "ddqk_benchmark_format3_white_model_losses",
        "report_sha256": report_sha256.lower(),
        "benchmark_audit": dict(benchmark_audit or {}),
        "rules": {
            "board_size": BOARD_SIZE,
            "win_length": WIN_LENGTH,
            "freestyle": True,
            "side_to_move": "white",
        },
        "claim_boundary": {
            "label": "bounded_non_loss_within_search_candidates",
            "candidate_scope": (
                f"GomokuGame.search_actions(radius={config.candidate_radius}); "
                "local legal candidates, narrowed to immediate wins or blocks when present"
            ),
            "checks": [
                "opponent immediate win",
                "opponent exact forced win in at most 3 plies",
                f"opponent bounded VCF at plies {list(config.vcf_plies)}",
            ],
            "unknown_budget_policy": "retained_as_safe_and_recorded_not_unsafe",
            "does_not_prove": [
                "safety outside the search_actions candidate set",
                "absence of arbitrary VCT",
                "eventual draw or win",
            ],
        },
        "config": {
            "decision_distances": list(config.decision_distances),
            "decision_distance_semantics": (
                "1 is the final non-opening white model decision before game end"
            ),
            "eval_fraction": config.eval_fraction,
            "split_seed": config.split_seed,
            "candidate_radius": config.candidate_radius,
            "vcf_plies": list(config.vcf_plies),
            "vcf_max_nodes_per_query": config.vcf_max_nodes,
            "vcf_time_ms_per_query": config.vcf_time_ms,
            "deterministic_node_budget_only": config.vcf_time_ms == 0.0,
        },
        "split": {
            "unit": (
                "report_sha256_plus_opening_sha256; group_id additionally retains "
                "pair_index provenance"
            ),
            "assigned_before_replay_or_tactical_labelling": True,
            "augmentation": "none",
            "assigned_groups": assigned_counts,
            "exported_records": exported_counts,
        },
        "counts": {
            "input_games": len(raw_games),
            "eligible_white_losses": len(eligible),
            "requested_positions_reached": labelled_positions,
            "exported_positions": len(records),
            "dropped_no_safe_action_within_bounded_candidate_set": (
                dropped_no_safe_within_bounded_candidates
            ),
            "skipped_distance_beyond_game_history": skipped_short_distances,
            "candidate_actions": int(arrays["candidate_count"].sum()),
            "safe_actions": int(arrays["safe_count"].sum()),
            "unknown_vcf_actions_retained": int(arrays["vcf_unknown_count"].sum()),
            "unsafe_immediate_actions": int(arrays["unsafe_immediate_count"].sum()),
            "unsafe_three_ply_actions": int(arrays["unsafe_three_ply_count"].sum()),
            "unsafe_vcf_actions": int(arrays["unsafe_vcf_count"].sum()),
            "vcf_queries": int(arrays["vcf_queries"].sum()),
            "vcf_nodes": int(arrays["vcf_nodes"].sum()),
            "unique_labelled_states": len(label_cache),
            "state_label_cache_hits": label_cache_hits,
        },
        "value_target": {
            "values": 0.0,
            "value_weights": 0.0,
            "reason": "bounded defensive labels do not justify a terminal value target",
        },
    }
    dataset = WhiteDefenseDataset(summary=summary, **arrays)
    summary["validation"] = validate_dataset(dataset)
    return dataset


def generate_white_defense_dataset(
    report_path: Path,
    *,
    config: WhiteDefenseConfig | None = None,
    solver: TacticalSolver | object | None = None,
) -> WhiteDefenseDataset:
    """Load, authenticate, replay, and label a DDQK format-3 report."""

    report_path = report_path.resolve()
    payload = report_path.read_bytes()
    try:
        report = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid UTF-8 JSON report: {report_path}") from exc
    if not isinstance(report, dict):
        raise ValueError("DDQK report must be a JSON object")
    if int(report.get("format_version", -1)) != 3:
        raise ValueError("white-defense extraction requires benchmark format_version 3")
    benchmark_audit = validate_benchmark_report(report)
    dataset = build_dataset_from_validated_report(
        report,
        report_sha256=_sha256_bytes(payload),
        config=config,
        solver=solver,
        benchmark_audit=benchmark_audit,
    )
    dataset.summary["report"] = str(report_path)
    return dataset


def validate_dataset(dataset: WhiteDefenseDataset) -> dict[str, object]:
    """Deeply validate schema, replay encoding, masks, and split isolation."""

    arrays = dataset.arrays()
    count = len(dataset.states)
    if count <= 0:
        raise ValueError("dataset must contain at least one sample")
    expected_vector_fields = set(arrays) - {
        "states",
        "policies",
        "move_history",
        "candidate_mask",
        "safe_mask",
        "vcf_unknown_mask",
        "unsafe_immediate_mask",
        "unsafe_three_ply_mask",
        "unsafe_vcf_mask",
    }
    if dataset.states.shape != (count, 4, BOARD_SIZE, BOARD_SIZE):
        raise ValueError("invalid states shape")
    if dataset.policies.shape != (count, ACTION_COUNT):
        raise ValueError("invalid policies shape")
    for name in (
        "candidate_mask",
        "safe_mask",
        "vcf_unknown_mask",
        "unsafe_immediate_mask",
        "unsafe_three_ply_mask",
        "unsafe_vcf_mask",
    ):
        if arrays[name].shape != (count, ACTION_COUNT):
            raise ValueError(f"invalid {name} shape")
    if dataset.move_history.shape != (count, ACTION_COUNT):
        raise ValueError("invalid move_history shape")
    for name in expected_vector_fields:
        if arrays[name].shape != (count,):
            raise ValueError(f"invalid {name} shape")
    for name in (
        "states",
        "candidate_mask",
        "safe_mask",
        "vcf_unknown_mask",
        "unsafe_immediate_mask",
        "unsafe_three_ply_mask",
        "unsafe_vcf_mask",
    ):
        if not np.all((arrays[name] == 0) | (arrays[name] == 1)):
            raise ValueError(f"{name} must be binary")
    if not np.all(np.isfinite(dataset.policies)) or np.any(dataset.policies < 0):
        raise ValueError("policies must be finite and non-negative")
    if not np.allclose(dataset.policies.sum(axis=1), 1.0, atol=2e-3):
        raise ValueError("policy rows must sum to one")
    if np.any(dataset.safe_count <= 0):
        raise ValueError("every exported sample must have a safe action")
    if not np.all(dataset.values == 0) or not np.all(dataset.value_weights == 0):
        raise ValueError("white-defense data must not fabricate value targets")
    if np.any(dataset.policy_weights <= 0) or np.any(dataset.priority <= 0):
        raise ValueError("policy weights and priorities must be positive")
    if not np.all(np.isfinite(dataset.policy_weights)) or not np.all(
        np.isfinite(dataset.priority)
    ):
        raise ValueError("weights/priorities must be finite")

    candidate = dataset.candidate_mask.astype(bool)
    safe = dataset.safe_mask.astype(bool)
    unknown = dataset.vcf_unknown_mask.astype(bool)
    immediate = dataset.unsafe_immediate_mask.astype(bool)
    three = dataset.unsafe_three_ply_mask.astype(bool)
    vcf = dataset.unsafe_vcf_mask.astype(bool)
    unsafe_sum = immediate.astype(np.uint8) + three.astype(np.uint8) + vcf.astype(np.uint8)
    if np.any(unsafe_sum > 1):
        raise ValueError("unsafe reason masks overlap")
    if np.any(safe & (immediate | three | vcf)):
        raise ValueError("safe and unsafe masks overlap")
    if not np.array_equal(candidate, safe | immediate | three | vcf):
        raise ValueError("candidate mask is not fully classified")
    if np.any(unknown & ~safe):
        raise ValueError("UNKNOWN_BUDGET actions must be retained as safe")
    if not np.array_equal(candidate.sum(axis=1), dataset.candidate_count):
        raise ValueError("candidate_count mismatch")
    if not np.array_equal(unsafe_sum.sum(axis=1), dataset.unsafe_count):
        raise ValueError("unsafe_count mismatch")
    if not np.array_equal(
        dataset.safe_count + dataset.unsafe_count, dataset.candidate_count
    ):
        raise ValueError("safe/unsafe counts do not cover the candidates")
    for mask, counts, name in (
        (safe, dataset.safe_count, "safe_count"),
        (unknown, dataset.vcf_unknown_count, "vcf_unknown_count"),
        (immediate, dataset.unsafe_immediate_count, "unsafe_immediate_count"),
        (three, dataset.unsafe_three_ply_count, "unsafe_three_ply_count"),
        (vcf, dataset.unsafe_vcf_count, "unsafe_vcf_count"),
    ):
        if not np.array_equal(mask.sum(axis=1), counts):
            raise ValueError(f"{name} mismatch")
    policy_support = dataset.policies > 0
    if not np.array_equal(policy_support, safe):
        raise ValueError("policy support must equal the complete safe set")
    expected_probabilities = safe / dataset.safe_count[:, None]
    if not np.allclose(dataset.policies, expected_probabilities, atol=5e-4):
        raise ValueError("policy must be uniform over safe actions")

    splits = set(map(str, dataset.split.tolist()))
    if not splits.issubset({TRAIN_SPLIT, EVAL_SPLIT}):
        raise ValueError(f"invalid split values: {sorted(splits)}")
    group_splits: dict[str, str] = {}
    opening_splits: dict[str, set[str]] = {}
    for group, split, opening_hash in zip(
        map(str, dataset.group_id),
        map(str, dataset.split),
        map(str, dataset.opening_sha256),
    ):
        previous = group_splits.setdefault(group, split)
        if previous != split:
            raise ValueError("one original game/opening group crosses train/eval")
        opening_splits.setdefault(opening_hash, set()).add(split)
    if any(len(values) > 1 for values in opening_splits.values()):
        raise ValueError("one opening hash crosses train/eval")
    train_hashes = set(map(str, dataset.state_hash[dataset.split == TRAIN_SPLIT]))
    eval_hashes = set(map(str, dataset.state_hash[dataset.split == EVAL_SPLIT]))
    if train_hashes & eval_hashes:
        raise ValueError("identical state leaks across train/eval")

    labels_by_state: dict[str, tuple[bytes, ...]] = {}
    for index in range(count):
        move_count = int(dataset.move_count[index])
        if not 0 < move_count < ACTION_COUNT:
            raise ValueError(f"sample {index}: invalid move_count")
        history = dataset.move_history[index]
        if np.any(history[:move_count] < 0) or np.any(history[move_count:] != -1):
            raise ValueError(f"sample {index}: malformed padded move history")
        game = GomokuGame(BOARD_SIZE, WIN_LENGTH)
        for action in map(int, history[:move_count]):
            game.play(action)
        if game.terminal or game.player != TRAIN_WHITE:
            raise ValueError(f"sample {index}: history is not a live white turn")
        if not np.array_equal(game.encode(), dataset.states[index]):
            raise ValueError(f"sample {index}: state differs from replay encoding")
        if game.last_action != int(dataset.last_action[index]):
            raise ValueError(f"sample {index}: last_action mismatch")
        expected_candidates = np.zeros(ACTION_COUNT, dtype=bool)
        expected_candidates[
            game.search_actions(int(dataset.candidate_radius[index]))
        ] = True
        if not np.array_equal(expected_candidates, candidate[index]):
            raise ValueError(f"sample {index}: candidate scope mismatch")
        original_action = int(dataset.original_action[index])
        if not 0 <= original_action < ACTION_COUNT or game.board.ravel()[original_action] != 0:
            raise ValueError(f"sample {index}: original action is not legal")
        if int(dataset.original_action_in_candidates[index]) != int(
            candidate[index, original_action]
        ):
            raise ValueError(f"sample {index}: original candidate flag mismatch")
        if int(dataset.original_action_safe[index]) != int(safe[index, original_action]):
            raise ValueError(f"sample {index}: original safe flag mismatch")
        expected_hash = _sha256_bytes(dataset.states[index].tobytes())
        if str(dataset.state_hash[index]) != expected_hash:
            raise ValueError(f"sample {index}: state hash mismatch")
        label_signature = tuple(
            np.ascontiguousarray(arrays[name][index]).tobytes()
            for name in (
                "candidate_mask",
                "safe_mask",
                "vcf_unknown_mask",
                "unsafe_immediate_mask",
                "unsafe_three_ply_mask",
                "unsafe_vcf_mask",
            )
        )
        previous_signature = labels_by_state.setdefault(expected_hash, label_signature)
        if previous_signature != label_signature:
            raise ValueError(
                f"sample {index}: identical state has conflicting tactical labels"
            )
        if str(dataset.report_sha256[index]) not in str(dataset.group_id[index]):
            raise ValueError(f"sample {index}: group_id omits report provenance")

    return {
        "records": count,
        "train_records": int(np.count_nonzero(dataset.split == TRAIN_SPLIT)),
        "eval_records": int(np.count_nonzero(dataset.split == EVAL_SPLIT)),
        "groups": len(group_splits),
        "state_overlap": 0,
        "opening_group_overlap": 0,
        "unknown_actions_retained": int(dataset.vcf_unknown_count.sum()),
        "replayed_positions": count,
    }


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    token = f"{os.getpid()}.{os.urandom(6).hex()}"
    temporary = path.with_name(f"{path.name}.{token}.tmp.npz")
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_text(path: Path, content: str) -> None:
    token = f"{os.getpid()}.{os.urandom(6).hex()}"
    temporary = path.with_name(f"{path.name}.{token}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_split_archives(
    dataset: WhiteDefenseDataset,
    train_output: Path,
    eval_output: Path,
    manifest_json: Path,
    *,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Atomically write trainer-safe train/eval archives and SHA manifest."""

    validation = validate_dataset(dataset)
    train_output = train_output.resolve()
    eval_output = eval_output.resolve()
    manifest_json = manifest_json.resolve()
    manifest_sha256 = manifest_json.with_suffix(manifest_json.suffix + ".sha256")
    paths = (train_output, eval_output, manifest_json, manifest_sha256)
    if len(set(paths)) != len(paths):
        raise ValueError("train, eval, manifest, and hash paths must be different")
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite:
        existing = [str(path) for path in paths if path.exists()]
        if existing:
            raise FileExistsError("refusing to overwrite: " + ", ".join(existing))

    train = dataset.subset(TRAIN_SPLIT)
    evaluation = dataset.subset(EVAL_SPLIT)
    if len(train.states) == 0:
        raise ValueError(
            "split assignment left no salvageable training samples; choose another "
            "split seed or add reports"
        )
    eval_fraction = float(dataset.summary.get("config", {}).get("eval_fraction", 0.0))
    if eval_fraction > 0 and len(evaluation.states) == 0:
        raise ValueError(
            "split assignment requested evaluation data but tactical filtering left "
            "no salvageable eval samples; choose another split seed or add reports"
        )
    _atomic_npz(train_output, train.arrays())
    _atomic_npz(eval_output, evaluation.arrays())
    expected_fields = set(dataset.arrays())
    for path in (train_output, eval_output):
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != expected_fields:
                raise ValueError(f"round-trip archive schema mismatch: {path}")

    try:
        train_manifest_path = Path(os.path.relpath(train_output, manifest_json.parent))
        eval_manifest_path = Path(os.path.relpath(eval_output, manifest_json.parent))
    except ValueError as error:
        raise ValueError(
            "train/eval artifacts must be on the same filesystem volume as the manifest"
        ) from error
    manifest = dict(dataset.summary)
    manifest["validation"] = validation
    manifest["artifacts"] = {
        TRAIN_SPLIT: {
            "path": train_manifest_path.as_posix(),
            "records": len(train.states),
            "bytes": train_output.stat().st_size,
            "sha256": sha256_file(train_output),
        },
        EVAL_SPLIT: {
            "path": eval_manifest_path.as_posix(),
            "records": len(evaluation.states),
            "bytes": eval_output.stat().st_size,
            "sha256": sha256_file(eval_output),
            "training_prohibition": "never pass this archive to a trainer",
        },
    }
    canonical_payload = json.dumps(
        manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    manifest["manifest_payload_sha256"] = _sha256_bytes(canonical_payload)
    text = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    _atomic_text(manifest_json, text)
    actual_manifest_hash = sha256_file(manifest_json)
    _atomic_text(
        manifest_sha256,
        f"{actual_manifest_hash}  {manifest_json.name}\n",
    )
    return {
        "train": train_output,
        "eval": eval_output,
        "manifest": manifest_json,
        "manifest_sha256": manifest_sha256,
    }


def _parse_positive_csv(value: str, *, name: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must be comma-separated integers") from exc
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError(f"{name} must contain positive integers")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--eval-output", type=Path, required=True)
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument(
        "--decision-distances",
        default=",".join(map(str, DEFAULT_DECISION_DISTANCES)),
        help="late white decision ranks; 1 is the final white model decision",
    )
    parser.add_argument("--eval-fraction", type=float, default=0.20)
    parser.add_argument("--split-seed", type=int, default=20260802)
    parser.add_argument("--candidate-radius", type=int, default=2)
    parser.add_argument(
        "--vcf-plies", default=",".join(map(str, DEFAULT_VCF_PLIES))
    )
    parser.add_argument("--vcf-max-nodes", type=int, default=50_000)
    parser.add_argument("--vcf-time-ms", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    args.decision_distances = _parse_positive_csv(
        args.decision_distances, name="decision-distances"
    )
    args.vcf_plies = _parse_positive_csv(args.vcf_plies, name="vcf-plies")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = WhiteDefenseConfig(
        decision_distances=args.decision_distances,
        eval_fraction=args.eval_fraction,
        split_seed=args.split_seed,
        candidate_radius=args.candidate_radius,
        vcf_plies=args.vcf_plies,
        vcf_max_nodes=args.vcf_max_nodes,
        vcf_time_ms=args.vcf_time_ms,
    )
    dataset = generate_white_defense_dataset(args.report, config=config)
    artifacts = write_split_archives(
        dataset,
        args.train_output,
        args.eval_output,
        args.manifest_json,
        overwrite=args.overwrite,
    )
    result = {
        "records": len(dataset.states),
        "train_records": int(np.count_nonzero(dataset.split == TRAIN_SPLIT)),
        "eval_records": int(np.count_nonzero(dataset.split == EVAL_SPLIT)),
        "unknown_actions_retained": int(dataset.vcf_unknown_count.sum()),
        "artifacts": {name: str(path) for name, path in artifacts.items()},
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

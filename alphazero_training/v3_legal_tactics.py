#!/usr/bin/env python3
"""Generate leakage-resistant legal tactics and evaluate Gomoku V3 models.

The older synthetic curriculum contained useful patterns but not reachable game
states: stone counts did not alternate legally and its last-move plane was
always empty.  This module builds every sample from an explicit, replayable
black-first history.  Training and held-out evaluation use different tactical
families and domain-separated random seeds, then pass both exact-state and
D4/translation-canonical overlap gates.

Two evaluation results are intentionally kept separate:

``raw_network``
    Legal-move-masked policy top-1 and best-oracle rank from the network alone.

``v3_search_with_exact_oracle``
    The deployed tactical-first search result.  This verifies integration but
    must not be interpreted as learned tactical skill.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import sys
from typing import Iterable, Mapping, Sequence

import numpy as np


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from alphazero_training.tactical_solver import FreestyleBoard, TacticalSolver
from alphazero_training.train_alphazero import (
    BLACK,
    EMPTY,
    WHITE,
    Config,
    GomokuGame,
)


BOARD_SIZE = 19
ACTION_COUNT = BOARD_SIZE * BOARD_SIZE
SCHEMA_VERSION = 2
TRAIN_SPLIT = "train"
EVAL_SPLIT = "eval"
ORACLE_KINDS = (
    "immediate_win",
    "immediate_block",
    "forced_win_in_3",
    "prevent_forced_win_in_3",
)
CHECKPOINT_MODEL_KEYS = ("best_model", "candidate_model", "train_model")


@dataclass(frozen=True)
class FamilySpec:
    family_name: str
    split: str
    oracle_kind: str
    side_to_move: int
    stones: tuple[tuple[int, int, int], ...]


@dataclass(frozen=True)
class LegalTacticsConfig:
    train_seed: int = 20260723
    eval_seed: int = 20260823
    train_samples_per_family: int = 16
    eval_samples_per_family: int = 8
    distractor_pairs: int = 1
    core_margin: int = 2
    distractor_min_distance: int = 3
    distractor_spacing: int = 2
    max_sample_attempts: int = 128

    def validate(self) -> None:
        if self.train_seed == self.eval_seed:
            raise ValueError("train and eval seeds must be different")
        if self.train_samples_per_family <= 0 or self.eval_samples_per_family <= 0:
            raise ValueError("samples per family must be positive")
        if self.distractor_pairs < 1:
            raise ValueError("at least one distractor pair is required for a real last move")
        if not 0 <= self.core_margin < BOARD_SIZE // 2:
            raise ValueError("invalid core margin")
        if self.distractor_min_distance < 0 or self.distractor_spacing < 0:
            raise ValueError("distractor distances cannot be negative")
        if self.max_sample_attempts <= 0:
            raise ValueError("max_sample_attempts must be positive")


@dataclass(frozen=True)
class LegalTacticsDataset:
    states: np.ndarray
    policies: np.ndarray
    values: np.ndarray
    policy_weight: np.ndarray
    value_weight: np.ndarray
    priority: np.ndarray
    source: np.ndarray
    group_id: np.ndarray
    family: np.ndarray
    split: np.ndarray
    oracle_kind: np.ndarray
    oracle_mask: np.ndarray
    side_to_move: np.ndarray
    last_action: np.ndarray
    move_count: np.ndarray
    move_history: np.ndarray
    state_hash: np.ndarray
    sample_fingerprint: np.ndarray
    generation_seed: np.ndarray
    summary: dict[str, object]

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            name: getattr(self, name)
            for name in (
                "states",
                "policies",
                "values",
                "policy_weight",
                "value_weight",
                "priority",
                "source",
                "group_id",
                "family",
                "split",
                "oracle_kind",
                "oracle_mask",
                "side_to_move",
                "last_action",
                "move_count",
                "move_history",
                "state_hash",
                "sample_fingerprint",
                "generation_seed",
            )
        }

    def subset(self, split: str) -> "LegalTacticsDataset":
        if split not in (TRAIN_SPLIT, EVAL_SPLIT):
            raise ValueError(f"unknown split: {split}")
        mask = self.split == split
        if not np.any(mask):
            raise ValueError(f"dataset contains no {split} samples")
        arrays = {name: array[mask] for name, array in self.arrays().items()}
        summary = dict(self.summary)
        summary["exported_split"] = split
        summary["exported_records"] = int(mask.sum())
        return LegalTacticsDataset(summary=summary, **arrays)


def _line(stone: int, points: Iterable[tuple[int, int]]) -> tuple[tuple[int, int, int], ...]:
    return tuple((x, y, stone) for x, y in points)


def family_catalog() -> tuple[FamilySpec, ...]:
    """Return structurally distinct train and held-out tactical families."""

    train = (
        FamilySpec(
            "train_capped_four_win",
            TRAIN_SPLIT,
            "immediate_win",
            BLACK,
            _line(BLACK, ((6, 9), (7, 9), (8, 9), (9, 9))) + ((5, 9, WHITE),),
        ),
        FamilySpec(
            "train_capped_four_block",
            TRAIN_SPLIT,
            "immediate_block",
            BLACK,
            _line(WHITE, ((6, 9), (7, 9), (8, 9), (9, 9))) + ((5, 9, BLACK),),
        ),
        FamilySpec(
            "train_open_three_attack",
            TRAIN_SPLIT,
            "forced_win_in_3",
            BLACK,
            _line(BLACK, ((7, 9), (8, 9), (9, 9))),
        ),
        FamilySpec(
            "train_open_three_defense",
            TRAIN_SPLIT,
            "prevent_forced_win_in_3",
            BLACK,
            _line(WHITE, ((7, 9), (8, 9), (9, 9))),
        ),
        FamilySpec(
            "train_orthogonal_fork_attack",
            TRAIN_SPLIT,
            "forced_win_in_3",
            BLACK,
            _line(BLACK, ((6, 9), (7, 9), (8, 9), (9, 6), (9, 7), (9, 8)))
            + _line(WHITE, ((5, 9), (9, 5))),
        ),
        FamilySpec(
            "train_orthogonal_fork_defense",
            TRAIN_SPLIT,
            "prevent_forced_win_in_3",
            BLACK,
            _line(WHITE, ((6, 9), (7, 9), (8, 9), (9, 6), (9, 7), (9, 8)))
            + _line(BLACK, ((5, 9), (9, 5))),
        ),
    )
    held_out = (
        FamilySpec(
            "eval_broken_four_win",
            EVAL_SPLIT,
            "immediate_win",
            BLACK,
            _line(BLACK, ((6, 9), (7, 9), (9, 9), (10, 9)))
            + _line(WHITE, ((5, 9), (11, 9))),
        ),
        FamilySpec(
            "eval_broken_four_block",
            EVAL_SPLIT,
            "immediate_block",
            BLACK,
            _line(WHITE, ((6, 9), (7, 9), (9, 9), (10, 9)))
            + _line(BLACK, ((5, 9), (11, 9))),
        ),
        FamilySpec(
            "eval_broken_three_attack",
            EVAL_SPLIT,
            "forced_win_in_3",
            BLACK,
            _line(BLACK, ((6, 9), (7, 9), (9, 9))),
        ),
        FamilySpec(
            "eval_broken_three_defense",
            EVAL_SPLIT,
            "prevent_forced_win_in_3",
            BLACK,
            _line(WHITE, ((6, 9), (7, 9), (9, 9))),
        ),
        FamilySpec(
            "eval_diagonal_fork_attack",
            EVAL_SPLIT,
            "forced_win_in_3",
            BLACK,
            _line(BLACK, ((6, 6), (7, 7), (8, 8), (6, 12), (7, 11), (8, 10)))
            + _line(WHITE, ((5, 5), (5, 13))),
        ),
        FamilySpec(
            "eval_diagonal_fork_defense",
            EVAL_SPLIT,
            "prevent_forced_win_in_3",
            BLACK,
            _line(WHITE, ((6, 6), (7, 7), (8, 8), (6, 12), (7, 11), (8, 10)))
            + _line(BLACK, ((5, 5), (5, 13))),
        ),
    )
    return train + held_out


def _seed64(master: int, *parts: object) -> int:
    message = ":".join((str(master), *(str(part) for part in parts)))
    return int.from_bytes(hashlib.sha256(message.encode("utf-8")).digest()[:8], "little")


def _d4_point_board(x: int, y: int, symmetry: int) -> tuple[int, int]:
    if symmetry >= 4:
        x = BOARD_SIZE - 1 - x
        symmetry -= 4
    for _ in range(symmetry):
        x, y = BOARD_SIZE - 1 - y, x
    return x, y


def _d4_point_unbounded(x: int, y: int, symmetry: int) -> tuple[int, int]:
    transforms = (
        (x, y),
        (-y, x),
        (-x, -y),
        (y, -x),
        (-x, y),
        (-y, -x),
        (x, -y),
        (y, x),
    )
    return transforms[symmetry]


def _board_from_stones(stones: Sequence[tuple[int, int, int]]) -> FreestyleBoard:
    return FreestyleBoard.from_stones(stones, size=BOARD_SIZE, win_length=5)


def _oracle_actions(
    board: FreestyleBoard,
    side_to_move: int,
    oracle_kind: str,
    solver: TacticalSolver,
) -> tuple[int, ...]:
    own_immediate = solver.immediate_wins(board, side_to_move)
    opponent_immediate = solver.immediate_wins(board, -side_to_move)
    if oracle_kind == "immediate_win":
        actions = own_immediate
    elif oracle_kind == "immediate_block":
        if own_immediate:
            return ()
        # A single placement cannot solve two independent immediate wins.
        actions = opponent_immediate if len(opponent_immediate) == 1 else ()
    elif oracle_kind == "forced_win_in_3":
        if own_immediate or opponent_immediate:
            return ()
        actions = solver.forced_wins_in_three(board, side_to_move)
    elif oracle_kind == "prevent_forced_win_in_3":
        if own_immediate or opponent_immediate:
            return ()
        opponent_attacks = solver.forced_wins_in_three(board, -side_to_move)
        actions = solver.exact_defenses(board, side_to_move) if opponent_attacks else ()
    else:
        raise ValueError(f"unsupported oracle kind: {oracle_kind}")
    return tuple(sorted(set(map(int, actions))))


def _canonical_payload(
    stones: Sequence[tuple[int, int, int]],
    side_to_move: int,
    last_action: int | None,
    actions: Sequence[int] = (),
    *,
    role_colors: bool,
    include_actions: bool,
) -> bytes:
    representations: list[str] = []
    for symmetry in range(8):
        transformed_stones = []
        all_points: list[tuple[int, int]] = []
        for x, y, stone in stones:
            tx, ty = _d4_point_unbounded(x, y, symmetry)
            role = stone * side_to_move if role_colors else stone
            transformed_stones.append((tx, ty, role))
            all_points.append((tx, ty))
        transformed_actions = []
        if include_actions:
            for action in actions:
                y, x = divmod(int(action), BOARD_SIZE)
                tx, ty = _d4_point_unbounded(x, y, symmetry)
                transformed_actions.append((tx, ty))
                all_points.append((tx, ty))
        transformed_last: tuple[int, int] | None = None
        if last_action is not None:
            y, x = divmod(int(last_action), BOARD_SIZE)
            transformed_last = _d4_point_unbounded(x, y, symmetry)
            all_points.append(transformed_last)
        if not all_points:
            raise ValueError("cannot fingerprint an empty position")
        min_x = min(point[0] for point in all_points)
        min_y = min(point[1] for point in all_points)
        normalized_stones = sorted(
            (x - min_x, y - min_y, stone) for x, y, stone in transformed_stones
        )
        normalized_actions = sorted(
            (x - min_x, y - min_y) for x, y in transformed_actions
        )
        normalized_last = (
            None
            if transformed_last is None
            else (transformed_last[0] - min_x, transformed_last[1] - min_y)
        )
        absolute_side = side_to_move if not role_colors else 1
        representations.append(
            json.dumps(
                [absolute_side, normalized_stones, normalized_last, normalized_actions],
                separators=(",", ":"),
                sort_keys=False,
            )
        )
    return min(representations).encode("utf-8")


def family_fingerprint(spec: FamilySpec, solver: TacticalSolver | None = None) -> str:
    solver = solver or TacticalSolver(board_size=BOARD_SIZE, win_length=5)
    board = _board_from_stones(spec.stones)
    actions = _oracle_actions(board, spec.side_to_move, spec.oracle_kind, solver)
    if not actions:
        raise ValueError(f"family {spec.family_name} has no independent oracle answer")
    payload = spec.oracle_kind.encode("ascii") + b"|" + _canonical_payload(
        spec.stones,
        spec.side_to_move,
        None,
        actions,
        role_colors=True,
        include_actions=True,
    )
    return hashlib.sha256(payload).hexdigest()


def sample_fingerprint(
    stones: Sequence[tuple[int, int, int]],
    side_to_move: int,
    last_action: int,
    actions: Sequence[int],
) -> str:
    payload = _canonical_payload(
        stones,
        side_to_move,
        last_action,
        actions,
        role_colors=False,
        include_actions=True,
    )
    return hashlib.sha256(payload).hexdigest()


def _translate_and_transform(
    spec: FamilySpec,
    rng: np.random.Generator,
    margin: int,
    colour_swap: bool,
) -> tuple[tuple[int, int, int], ...]:
    xs = [stone[0] for stone in spec.stones]
    ys = [stone[1] for stone in spec.stones]
    min_dx = margin - min(xs)
    max_dx = BOARD_SIZE - 1 - margin - max(xs)
    min_dy = margin - min(ys)
    max_dy = BOARD_SIZE - 1 - margin - max(ys)
    if min_dx > max_dx or min_dy > max_dy:
        raise ValueError(f"family {spec.family_name} does not fit with margin {margin}")
    dx = int(rng.integers(min_dx, max_dx + 1))
    dy = int(rng.integers(min_dy, max_dy + 1))
    symmetry = int(rng.integers(0, 8))
    transformed = []
    for x, y, stone in spec.stones:
        tx, ty = _d4_point_board(x + dx, y + dy, symmetry)
        transformed.append((tx, ty, -stone if colour_swap else stone))
    return tuple(sorted(transformed, key=lambda item: (item[1], item[0], item[2])))


def _chebyshev(a: int, b: int) -> int:
    ay, ax = divmod(int(a), BOARD_SIZE)
    by, bx = divmod(int(b), BOARD_SIZE)
    return max(abs(ax - bx), abs(ay - by))


def _augment_to_legal_counts(
    stones: tuple[tuple[int, int, int], ...],
    side_to_move: int,
    oracle_kind: str,
    baseline_actions: tuple[int, ...],
    rng: np.random.Generator,
    config: LegalTacticsConfig,
    solver: TacticalSolver,
) -> tuple[tuple[tuple[int, int, int], ...], int]:
    current = list(stones)
    black_count = sum(stone == BLACK for _x, _y, stone in current)
    white_count = sum(stone == WHITE for _x, _y, stone in current)
    desired_delta = 0 if side_to_move == BLACK else 1
    delta = black_count - white_count
    additions: list[int] = []
    if delta > desired_delta:
        additions.extend([WHITE] * (delta - desired_delta))
    elif delta < desired_delta:
        additions.extend([BLACK] * (desired_delta - delta))
    for _ in range(config.distractor_pairs):
        additions.extend((BLACK, WHITE))
    rng.shuffle(additions)

    important = {
        y * BOARD_SIZE + x for x, y, _stone in stones
    } | set(map(int, baseline_actions))
    added_actions: list[int] = []
    occupied = {y * BOARD_SIZE + x for x, y, _stone in current}
    candidates = np.asarray(
        [
            action
            for action in range(ACTION_COUNT)
            if action not in occupied
            and action not in important
            and all(
                _chebyshev(action, anchor) >= config.distractor_min_distance
                for anchor in important
            )
        ],
        dtype=np.int32,
    )
    rng.shuffle(candidates)

    for stone in additions:
        accepted = False
        for action in candidates:
            action = int(action)
            if action in occupied:
                continue
            if any(
                _chebyshev(action, previous) < config.distractor_spacing
                for previous in added_actions
            ):
                continue
            y, x = divmod(action, BOARD_SIZE)
            current.append((x, y, stone))
            occupied.add(action)
            added_actions.append(action)
            accepted = True
            break
        if not accepted:
            raise RuntimeError("could not place every oracle-preserving balancing stone")

    # Distance is only a proposal heuristic.  Safety is decided semantically
    # on the complete final position; a failed set makes the outer generator
    # retry with a new domain-separated seed.  One final exact query is both
    # sufficient and much faster than re-proving the same tactic after each
    # prefix of filler placements.
    final_board = _board_from_stones(tuple(current))
    if final_board.has_five(BLACK) or final_board.has_five(WHITE):
        raise RuntimeError("balancing stones created a terminal position")
    calculated = _oracle_actions(final_board, side_to_move, oracle_kind, solver)
    if calculated != baseline_actions:
        raise RuntimeError("balancing stones changed the independent oracle answer")

    opponent_additions = [
        action
        for action in added_actions
        if next(stone for x, y, stone in current if y * BOARD_SIZE + x == action)
        == -side_to_move
    ]
    if not opponent_additions:
        raise RuntimeError("legalization did not create an opponent last-move candidate")
    last_action = int(opponent_additions[int(rng.integers(0, len(opponent_additions)))])
    return tuple(current), last_action


def _history_for_position(
    stones: Sequence[tuple[int, int, int]],
    side_to_move: int,
    last_action: int,
    rng: np.random.Generator,
) -> tuple[int, ...]:
    black = [y * BOARD_SIZE + x for x, y, stone in stones if stone == BLACK]
    white = [y * BOARD_SIZE + x for x, y, stone in stones if stone == WHITE]
    rng.shuffle(black)
    rng.shuffle(white)
    if side_to_move == BLACK:
        if len(black) != len(white) or last_action not in white:
            raise ValueError("illegal counts or last move for black to move")
        white.remove(last_action)
        history: list[int] = []
        for index in range(len(white)):
            history.extend((black[index], white[index]))
        history.extend((black[-1], last_action))
    elif side_to_move == WHITE:
        if len(black) != len(white) + 1 or last_action not in black:
            raise ValueError("illegal counts or last move for white to move")
        black.remove(last_action)
        history = []
        for black_action, white_action in zip(black, white):
            history.extend((black_action, white_action))
        history.append(last_action)
    else:
        raise ValueError(f"invalid side to move: {side_to_move}")
    return tuple(history)


def _replay_history(history: Sequence[int]) -> GomokuGame:
    game = GomokuGame(BOARD_SIZE, 5)
    for index, action in enumerate(history):
        if game.terminal:
            raise ValueError(f"history became terminal before move {index + 1}")
        game.play(int(action))
    if game.terminal:
        raise ValueError("tactical input position is already terminal")
    return game


def _policy(actions: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    actions = tuple(sorted(set(map(int, actions))))
    if not actions:
        raise ValueError("oracle action set must not be empty")
    policy = np.zeros(ACTION_COUNT, dtype=np.float16)
    policy[list(actions)] = np.float16(1.0 / len(actions))
    policy[actions[0]] = np.float16(
        policy[actions[0]] + np.float16(1.0 - float(policy.sum(dtype=np.float32)))
    )
    mask = np.zeros(ACTION_COUNT, dtype=np.uint8)
    mask[list(actions)] = 1
    return policy, mask


def _labels(oracle_kind: str) -> tuple[int, float, float, float]:
    if oracle_kind == "immediate_win":
        return 1, 1.0, 1.0, 3.0
    if oracle_kind == "forced_win_in_3":
        return 1, 1.0, 1.0, 4.0
    if oracle_kind == "immediate_block":
        return 0, 1.0, 0.0, 3.0
    if oracle_kind == "prevent_forced_win_in_3":
        return 0, 1.0, 0.0, 5.0
    raise ValueError(oracle_kind)


def _unicode(values: Sequence[object]) -> np.ndarray:
    text = [str(value) for value in values]
    width = max(1, *(len(value) for value in text))
    return np.asarray(text, dtype=f"<U{width}")


def generate_legal_tactics(
    config: LegalTacticsConfig | None = None,
    families: Sequence[FamilySpec] | None = None,
) -> LegalTacticsDataset:
    """Generate deterministic legal train/eval samples and enforce isolation."""

    config = config or LegalTacticsConfig()
    config.validate()
    families = tuple(family_catalog() if families is None else families)
    if not families:
        raise ValueError("family catalog must not be empty")
    solver = TacticalSolver(board_size=BOARD_SIZE, win_length=5)

    family_ids: dict[str, str] = {}
    families_by_split: dict[str, set[str]] = {TRAIN_SPLIT: set(), EVAL_SPLIT: set()}
    for spec in sorted(families, key=lambda item: item.family_name):
        if spec.split not in families_by_split:
            raise ValueError(f"invalid family split: {spec.split}")
        if spec.oracle_kind not in ORACLE_KINDS:
            raise ValueError(f"invalid oracle kind: {spec.oracle_kind}")
        if spec.family_name in family_ids:
            raise ValueError(f"duplicate family name: {spec.family_name}")
        fingerprint = family_fingerprint(spec, solver)
        if fingerprint in families_by_split[spec.split]:
            raise ValueError(f"duplicate canonical family in {spec.split}: {spec.family_name}")
        family_ids[spec.family_name] = fingerprint
        families_by_split[spec.split].add(fingerprint)
    family_overlap = families_by_split[TRAIN_SPLIT] & families_by_split[EVAL_SPLIT]
    if family_overlap:
        raise ValueError(f"train/eval canonical family overlap: {sorted(family_overlap)}")
    for split in (TRAIN_SPLIT, EVAL_SPLIT):
        kinds = {spec.oracle_kind for spec in families if spec.split == split}
        missing = set(ORACLE_KINDS) - kinds
        if missing:
            raise ValueError(f"{split} is missing oracle kinds: {sorted(missing)}")

    records: list[dict[str, object]] = []
    exact_seen: set[str] = set()
    canonical_seen: set[str] = set()
    rejected = Counter()
    generation_seeds: dict[str, set[int]] = {TRAIN_SPLIT: set(), EVAL_SPLIT: set()}

    for spec in sorted(families, key=lambda item: (item.split, item.family_name)):
        count = (
            config.train_samples_per_family
            if spec.split == TRAIN_SPLIT
            else config.eval_samples_per_family
        )
        master_seed = config.train_seed if spec.split == TRAIN_SPLIT else config.eval_seed
        family_id = family_ids[spec.family_name]
        for variant in range(count):
            emitted = False
            for attempt in range(config.max_sample_attempts):
                sample_seed = _seed64(
                    master_seed, spec.split, family_id, variant, attempt
                )
                if sample_seed in generation_seeds[spec.split]:
                    raise RuntimeError("domain-separated generation seed collision")
                rng = np.random.default_rng(sample_seed)
                colour_swap = bool(variant % 2)
                side_to_move = -spec.side_to_move if colour_swap else spec.side_to_move
                try:
                    transformed = _translate_and_transform(
                        spec, rng, config.core_margin, colour_swap
                    )
                    core_board = _board_from_stones(transformed)
                    core_actions = _oracle_actions(
                        core_board, side_to_move, spec.oracle_kind, solver
                    )
                    if not core_actions:
                        raise RuntimeError("transformed core lost its oracle answer")
                    augmented, last_action = _augment_to_legal_counts(
                        transformed,
                        side_to_move,
                        spec.oracle_kind,
                        core_actions,
                        rng,
                        config,
                        solver,
                    )
                    history = _history_for_position(
                        augmented, side_to_move, last_action, rng
                    )
                    game = _replay_history(history)
                    if game.player != side_to_move or game.last_action != last_action:
                        raise RuntimeError("replayed state disagrees with side/last action")
                    final_stones = tuple(
                        (int(x), int(y), int(game.board[y, x]))
                        for y, x in np.argwhere(game.board != EMPTY)
                    )
                    final_board = _board_from_stones(final_stones)
                    final_actions = _oracle_actions(
                        final_board, side_to_move, spec.oracle_kind, solver
                    )
                    if final_actions != core_actions:
                        raise RuntimeError("final oracle differs from transformed core")
                    state = game.encode()
                    exact_hash = hashlib.sha256(state.tobytes()).hexdigest()
                    canonical_hash = sample_fingerprint(
                        final_stones, side_to_move, last_action, final_actions
                    )
                    if exact_hash in exact_seen:
                        rejected["exact_state_duplicate"] += 1
                        continue
                    if canonical_hash in canonical_seen:
                        rejected["d4_translation_duplicate"] += 1
                        continue
                    policy, oracle_mask = _policy(final_actions)
                    value, p_weight, v_weight, priority = _labels(spec.oracle_kind)
                except (RuntimeError, ValueError) as exc:
                    rejected[str(exc)] += 1
                    continue

                exact_seen.add(exact_hash)
                canonical_seen.add(canonical_hash)
                generation_seeds[spec.split].add(sample_seed)
                padded_history = np.full(ACTION_COUNT, -1, dtype=np.int16)
                padded_history[: len(history)] = np.asarray(history, dtype=np.int16)
                records.append(
                    {
                        "state": state,
                        "policy": policy,
                        "value": value,
                        "policy_weight": p_weight,
                        "value_weight": v_weight,
                        "priority": priority,
                        "source": (
                            f"{family_id}|{spec.family_name}|variant={variant}"
                            f"|seed={sample_seed}"
                        ),
                        "group_id": family_id,
                        "family": spec.family_name,
                        "split": spec.split,
                        "oracle_kind": spec.oracle_kind,
                        "oracle_mask": oracle_mask,
                        "side_to_move": side_to_move,
                        "last_action": last_action,
                        "move_count": len(history),
                        "move_history": padded_history,
                        "state_hash": exact_hash,
                        "sample_fingerprint": canonical_hash,
                        "generation_seed": sample_seed,
                    }
                )
                emitted = True
                break
            if not emitted:
                raise RuntimeError(
                    f"failed to generate {spec.family_name} variant {variant} after "
                    f"{config.max_sample_attempts} attempts"
                )

    if not records:
        raise RuntimeError("no legal tactical samples were generated")
    arrays = {
        "states": np.stack([record["state"] for record in records]).astype(np.uint8),
        "policies": np.stack([record["policy"] for record in records]).astype(np.float16),
        "values": np.asarray([record["value"] for record in records], dtype=np.int8),
        "policy_weight": np.asarray(
            [record["policy_weight"] for record in records], dtype=np.float16
        ),
        "value_weight": np.asarray(
            [record["value_weight"] for record in records], dtype=np.float16
        ),
        "priority": np.asarray([record["priority"] for record in records], dtype=np.float16),
        "source": _unicode([record["source"] for record in records]),
        "group_id": _unicode([record["group_id"] for record in records]),
        "family": _unicode([record["family"] for record in records]),
        "split": _unicode([record["split"] for record in records]),
        "oracle_kind": _unicode([record["oracle_kind"] for record in records]),
        "oracle_mask": np.stack([record["oracle_mask"] for record in records]).astype(np.uint8),
        "side_to_move": np.asarray(
            [record["side_to_move"] for record in records], dtype=np.int8
        ),
        "last_action": np.asarray(
            [record["last_action"] for record in records], dtype=np.int16
        ),
        "move_count": np.asarray(
            [record["move_count"] for record in records], dtype=np.uint16
        ),
        "move_history": np.stack([record["move_history"] for record in records]).astype(
            np.int16
        ),
        "state_hash": _unicode([record["state_hash"] for record in records]),
        "sample_fingerprint": _unicode(
            [record["sample_fingerprint"] for record in records]
        ),
        "generation_seed": np.asarray(
            [record["generation_seed"] for record in records], dtype=np.uint64
        ),
    }
    summary: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "dataset": "v3_legal_freestyle_gomoku_tactics",
        "config": vars(config),
        "records": len(records),
        "records_by_split": dict(Counter(map(str, arrays["split"]))),
        "records_by_oracle": dict(Counter(map(str, arrays["oracle_kind"]))),
        "families_by_split": {
            split: sorted(
                spec.family_name for spec in families if spec.split == split
            )
            for split in (TRAIN_SPLIT, EVAL_SPLIT)
        },
        "rejections": dict(sorted(rejected.items())),
        "isolation": {
            "train_eval_master_seed_overlap": False,
            "train_eval_family_overlap": 0,
            "train_eval_state_hash_overlap": 0,
            "train_eval_d4_translation_overlap": 0,
            "train_eval_generation_seed_overlap": 0,
        },
        "label_contract": {
            "policy": "uniform over independently recomputed complete oracle actions",
            "last_move_plane": "exactly one opponent stone from a replayed alternating history",
            "attack_value": "+1 with value_weight=1",
            "defense_value": "0 with value_weight=0",
        },
    }
    dataset = LegalTacticsDataset(summary=summary, **arrays)
    validation = validate_dataset(dataset)
    dataset.summary["validation"] = validation
    return dataset


def _stones_from_state(state: np.ndarray, side_to_move: int) -> tuple[tuple[int, int, int], ...]:
    stones: list[tuple[int, int, int]] = []
    for y, x in np.argwhere(state[0] != 0):
        stones.append((int(x), int(y), int(side_to_move)))
    for y, x in np.argwhere(state[1] != 0):
        stones.append((int(x), int(y), int(-side_to_move)))
    return tuple(stones)


def validate_dataset(dataset: LegalTacticsDataset) -> dict[str, object]:
    """Deeply validate histories, states, labels, and split isolation."""

    arrays = dataset.arrays()
    count = len(dataset.states)
    if count == 0:
        raise ValueError("dataset is empty")
    expected_shapes = {
        "states": (count, 4, BOARD_SIZE, BOARD_SIZE),
        "policies": (count, ACTION_COUNT),
        "oracle_mask": (count, ACTION_COUNT),
        "move_history": (count, ACTION_COUNT),
    }
    for name, shape in expected_shapes.items():
        if arrays[name].shape != shape:
            raise ValueError(f"{name} shape {arrays[name].shape} != {shape}")
    for name, array in arrays.items():
        if name not in expected_shapes and array.shape != (count,):
            raise ValueError(f"{name} shape {array.shape} != {(count,)}")
        if array.dtype.kind == "O":
            raise ValueError(f"{name} uses pickle-requiring object dtype")
    if dataset.states.dtype != np.uint8 or dataset.policies.dtype != np.float16:
        raise ValueError("states/policies dtypes do not match the schema")
    if dataset.oracle_mask.dtype != np.uint8:
        raise ValueError("oracle_mask must be uint8")
    declared_splits = set(map(str, dataset.split))
    unexpected_splits = declared_splits - {TRAIN_SPLIT, EVAL_SPLIT}
    if unexpected_splits:
        raise ValueError(f"dataset contains invalid split labels: {sorted(unexpected_splits)}")
    if not np.allclose(dataset.policies.sum(axis=1, dtype=np.float32), 1.0, atol=2e-3):
        raise ValueError("policy rows must sum to one")

    solver = TacticalSolver(board_size=BOARD_SIZE, win_length=5)
    for index in range(count):
        side = int(dataset.side_to_move[index])
        move_count = int(dataset.move_count[index])
        history = tuple(map(int, dataset.move_history[index, :move_count]))
        if np.any(dataset.move_history[index, move_count:] != -1):
            raise ValueError(f"sample {index}: non-padding history after move_count")
        if not history or history[-1] != int(dataset.last_action[index]):
            raise ValueError(f"sample {index}: last action does not end history")
        if len(set(history)) != len(history):
            raise ValueError(f"sample {index}: history repeats a point")
        game = _replay_history(history)
        if game.player != side or game.last_action != int(dataset.last_action[index]):
            raise ValueError(f"sample {index}: replay metadata mismatch")
        encoded = game.encode()
        if not np.array_equal(encoded, dataset.states[index]):
            raise ValueError(f"sample {index}: state is not the replay encoding")
        last_y, last_x = divmod(game.last_action, BOARD_SIZE)
        if int(encoded[2].sum()) != 1 or not encoded[2, last_y, last_x]:
            raise ValueError(f"sample {index}: invalid last-move plane")
        if not encoded[1, last_y, last_x]:
            raise ValueError(f"sample {index}: last move is not an opponent stone")
        black_count = int(np.count_nonzero(game.board == BLACK))
        white_count = int(np.count_nonzero(game.board == WHITE))
        if side == BLACK and black_count != white_count:
            raise ValueError(f"sample {index}: illegal counts for black to move")
        if side == WHITE and black_count != white_count + 1:
            raise ValueError(f"sample {index}: illegal counts for white to move")
        expected = _oracle_actions(
            _board_from_stones(_stones_from_state(encoded, side)),
            side,
            str(dataset.oracle_kind[index]),
            solver,
        )
        stored = tuple(map(int, np.flatnonzero(dataset.oracle_mask[index])))
        policy_support = tuple(map(int, np.flatnonzero(dataset.policies[index] > 0)))
        if not expected or expected != stored or stored != policy_support:
            raise ValueError(f"sample {index}: independent oracle label mismatch")
        occupied = (encoded[0] | encoded[1]).reshape(-1).astype(bool)
        if np.any(dataset.policies[index, occupied] > 0):
            raise ValueError(f"sample {index}: policy targets an occupied point")
        exact_hash = hashlib.sha256(encoded.tobytes()).hexdigest()
        if exact_hash != str(dataset.state_hash[index]):
            raise ValueError(f"sample {index}: state hash mismatch")
        canonical = sample_fingerprint(
            _stones_from_state(encoded, side), side, game.last_action, stored
        )
        if canonical != str(dataset.sample_fingerprint[index]):
            raise ValueError(f"sample {index}: sample fingerprint mismatch")

    train = dataset.split == TRAIN_SPLIT
    held_out = dataset.split == EVAL_SPLIT
    if not np.any(train) or not np.any(held_out):
        # Split archives are valid artifacts even though isolation was already
        # proven on the combined generator output.
        return {
            "records": count,
            "alternating_histories": count,
            "real_last_move_planes": count,
            "single_split_archive": True,
        }
    train_families = set(map(str, dataset.group_id[train]))
    eval_families = set(map(str, dataset.group_id[held_out]))
    train_states = set(map(str, dataset.state_hash[train]))
    eval_states = set(map(str, dataset.state_hash[held_out]))
    train_canonical = set(map(str, dataset.sample_fingerprint[train]))
    eval_canonical = set(map(str, dataset.sample_fingerprint[held_out]))
    train_seeds = set(map(int, dataset.generation_seed[train]))
    eval_seeds = set(map(int, dataset.generation_seed[held_out]))
    overlaps = {
        "family": len(train_families & eval_families),
        "state_hash": len(train_states & eval_states),
        "d4_translation": len(train_canonical & eval_canonical),
        "generation_seed": len(train_seeds & eval_seeds),
    }
    if any(overlaps.values()):
        raise ValueError(f"train/eval leakage detected: {overlaps}")
    return {
        "records": count,
        "alternating_histories": count,
        "real_last_move_planes": count,
        "train_records": int(train.sum()),
        "eval_records": int(held_out.sum()),
        "overlap": overlaps,
    }


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path = path.resolve()
    if path.suffix.lower() != ".npz":
        raise ValueError(f"NPZ output must end in .npz: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_split_archives(
    dataset: LegalTacticsDataset,
    train_output: Path,
    eval_output: Path,
    manifest_json: Path | None = None,
) -> dict[str, object]:
    """Write separate trainer-safe train and sealed evaluation NPZ files."""

    validate_dataset(dataset)
    train_output = train_output.resolve()
    eval_output = eval_output.resolve()
    if train_output == eval_output:
        raise ValueError("train and eval outputs must be different files")
    train_dataset = dataset.subset(TRAIN_SPLIT)
    eval_dataset = dataset.subset(EVAL_SPLIT)
    _atomic_npz(train_output, train_dataset.arrays())
    _atomic_npz(eval_output, eval_dataset.arrays())
    # Pickle-free loading is an explicit artifact gate.
    for path in (train_output, eval_output):
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(dataset.arrays()):
                raise ValueError(f"round-trip schema mismatch in {path}")
    manifest = dict(dataset.summary)
    manifest["artifacts"] = {
        TRAIN_SPLIT: {
            "path": str(train_output),
            "records": len(train_dataset.states),
            "bytes": train_output.stat().st_size,
            "sha256": _sha256_file(train_output),
        },
        EVAL_SPLIT: {
            "path": str(eval_output),
            "records": len(eval_dataset.states),
            "bytes": eval_output.stat().st_size,
            "sha256": _sha256_file(eval_output),
            "training_prohibition": "never pass this archive to train_v3_supervised",
        },
    }
    manifest_path = (
        manifest_json.resolve()
        if manifest_json is not None
        else train_output.with_name("v3_legal_tactics_manifest.json")
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, manifest_path)
    return {"train": train_output, "eval": eval_output, "manifest": manifest_path}


def load_archive(path: Path) -> LegalTacticsDataset:
    path = path.resolve()
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    required = set(LegalTacticsDataset.__dataclass_fields__) - {"summary"}
    if set(arrays) != required:
        raise ValueError(f"archive schema mismatch: {sorted(set(arrays) ^ required)}")
    dataset = LegalTacticsDataset(
        summary={"loaded_from": str(path)},
        **arrays,
    )
    validate_dataset(dataset)
    return dataset


def _metric_group(records: Sequence[dict[str, float]]) -> dict[str, float | int]:
    if not records:
        return {"samples": 0, "top1": 0.0, "mean_best_rank": 0.0, "mrr": 0.0}
    return {
        "samples": len(records),
        "top1": float(np.mean([record["top1"] for record in records])),
        "mean_best_rank": float(np.mean([record["rank"] for record in records])),
        "median_best_rank": float(np.median([record["rank"] for record in records])),
        "mrr": float(np.mean([1.0 / record["rank"] for record in records])),
        "mean_oracle_mass": float(np.mean([record["mass"] for record in records])),
        "mean_cross_entropy": float(np.mean([record["cross_entropy"] for record in records])),
    }


def evaluate_model(
    model: object,
    model_config: Config,
    dataset: LegalTacticsDataset,
    *,
    device: str | object = "cpu",
    simulations: int = 64,
    split: str = EVAL_SPLIT,
    seed: int = 20260723,
    limit: int | None = None,
) -> dict[str, object]:
    """Report raw policy skill separately from oracle-assisted V3 search."""

    import torch

    from alphazero_training.v3_search import V3RootSearch

    validate_dataset(dataset)
    indices = np.flatnonzero(dataset.split == split)
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        indices = indices[:limit]
    if not len(indices):
        raise ValueError(f"dataset contains no samples for split {split}")
    torch_device = torch.device(device)
    model.to(torch_device)
    model.eval()

    states = torch.from_numpy(dataset.states[indices]).to(
        device=torch_device, dtype=torch.float32
    )
    with torch.inference_mode():
        logits_parts = []
        value_parts = []
        for start in range(0, len(states), 256):
            logits, values = model(states[start : start + 256])
            logits_parts.append(logits.float().cpu().numpy())
            value_parts.append(values.float().cpu().numpy())
    logits_array = np.concatenate(logits_parts)
    values_array = np.concatenate(value_parts)

    raw_records: list[dict[str, float]] = []
    raw_by_kind: dict[str, list[dict[str, float]]] = defaultdict(list)
    raw_by_family: dict[str, list[dict[str, float]]] = defaultdict(list)
    failures: list[dict[str, object]] = []
    games: list[GomokuGame] = []
    for local_index, dataset_index in enumerate(indices):
        state = dataset.states[dataset_index]
        occupied = (state[0] | state[1]).reshape(-1).astype(bool)
        legal_logits = logits_array[local_index].astype(np.float64, copy=True)
        legal_logits[occupied] = -np.inf
        expected = np.flatnonzero(dataset.oracle_mask[dataset_index]).astype(np.int32)
        chosen = int(np.argmax(legal_logits))
        best_expected = float(np.max(legal_logits[expected]))
        rank = 1 + int(np.count_nonzero(legal_logits > best_expected))
        finite_legal = legal_logits[~occupied]
        shifted = finite_legal - float(np.max(finite_legal))
        denominator = float(np.exp(shifted).sum())
        action_probabilities = np.zeros(ACTION_COUNT, dtype=np.float64)
        action_probabilities[~occupied] = np.exp(shifted) / denominator
        mass = float(action_probabilities[expected].sum())
        cross_entropy = float(
            -np.mean(np.log(np.maximum(action_probabilities[expected], 1e-300)))
        )
        record = {
            "top1": float(chosen in set(map(int, expected))),
            "rank": float(rank),
            "mass": mass,
            "cross_entropy": cross_entropy,
        }
        raw_records.append(record)
        kind = str(dataset.oracle_kind[dataset_index])
        family = str(dataset.family[dataset_index])
        raw_by_kind[kind].append(record)
        raw_by_family[family].append(record)
        if not record["top1"] and len(failures) < 50:
            failures.append(
                {
                    "source": str(dataset.source[dataset_index]),
                    "family": family,
                    "oracle_kind": kind,
                    "last_action": int(dataset.last_action[dataset_index]),
                    "oracle_actions": list(map(int, expected)),
                    "network_action": chosen,
                    "best_oracle_rank": rank,
                    "oracle_probability_mass": mass,
                }
            )

        side = int(dataset.side_to_move[dataset_index])
        game = GomokuGame(BOARD_SIZE, 5)
        game.player = side
        game.board[state[0].astype(bool)] = side
        game.board[state[1].astype(bool)] = -side
        game.move_count = int(dataset.move_count[dataset_index])
        game.last_action = int(dataset.last_action[dataset_index])
        games.append(game)

    raw_summary = _metric_group(raw_records)
    family_summaries = {
        family: _metric_group(records) for family, records in sorted(raw_by_family.items())
    }
    raw_summary["family_macro_top1"] = float(
        np.mean([float(metrics["top1"]) for metrics in family_summaries.values()])
    )
    raw_summary["value_mae_on_proven_attacks"] = float(
        np.mean(
            [
                abs(float(values_array[local]) - float(dataset.values[index]))
                for local, index in enumerate(indices)
                if float(dataset.value_weight[index]) > 0
            ]
        )
    )

    searcher = V3RootSearch(
        model,
        model_config,
        torch_device,
        rng=np.random.default_rng(seed),
    )
    search_correct = 0
    proven_count = 0
    reasons = Counter()
    search_by_kind: dict[str, list[float]] = defaultdict(list)
    for game, dataset_index in zip(games, indices):
        decision = searcher.decide(
            game,
            simulations=simulations,
            add_noise=False,
            temperature=0.0,
        )
        expected = set(map(int, np.flatnonzero(dataset.oracle_mask[dataset_index])))
        correct = float(decision.action in expected)
        search_correct += int(correct)
        proven_count += int(decision.proven)
        reasons[decision.reason] += 1
        search_by_kind[str(dataset.oracle_kind[dataset_index])].append(correct)

    return {
        "split": split,
        "samples": len(indices),
        "raw_network": {
            **raw_summary,
            "by_oracle_kind": {
                kind: _metric_group(records)
                for kind, records in sorted(raw_by_kind.items())
            },
            "by_family": family_summaries,
            "failures": failures,
        },
        "v3_search_with_exact_oracle": {
            "accuracy": search_correct / len(indices),
            "proven_rate": proven_count / len(indices),
            "reasons": dict(sorted(reasons.items())),
            "by_oracle_kind": {
                kind: float(np.mean(results))
                for kind, results in sorted(search_by_kind.items())
            },
        },
        "interpretation": (
            "raw_network measures learned generalization; v3_search_with_exact_oracle "
            "measures system routing and can be perfect even for an untrained network"
        ),
    }


def evaluate_checkpoint(
    checkpoint_path: Path,
    dataset_path: Path,
    *,
    model_key: str = "best_model",
    split: str = EVAL_SPLIT,
    simulations: int = 64,
    device: str | None = None,
    limit: int | None = None,
) -> dict[str, object]:
    import torch

    from alphazero_training.train_alphazero import PolicyValueNet

    checkpoint_path = checkpoint_path.resolve()
    if model_key not in CHECKPOINT_MODEL_KEYS:
        raise ValueError(
            f"unsupported checkpoint model key {model_key!r}; "
            f"choose one of {CHECKPOINT_MODEL_KEYS}"
        )
    torch_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    checkpoint_bytes = checkpoint_path.read_bytes()
    checkpoint_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
    checkpoint = torch.load(
        io.BytesIO(checkpoint_bytes),
        map_location=torch_device,
        weights_only=False,
    )
    if model_key not in checkpoint:
        raise ValueError(
            f"checkpoint {checkpoint_path} does not contain requested model key "
            f"{model_key!r}"
        )
    config = Config(**checkpoint["config"])
    model = PolicyValueNet(
        config.board_size, config.channels, config.residual_blocks
    ).to(torch_device)
    model.load_state_dict(checkpoint[model_key])
    dataset = load_archive(dataset_path)
    result = evaluate_model(
        model,
        config,
        dataset,
        device=torch_device,
        simulations=simulations,
        split=split,
        limit=limit,
    )
    result["checkpoint"] = str(checkpoint_path)
    # Promotion binds an independent report to immutable checkpoint bytes,
    # not merely to a mutable pathname.
    result["checkpoint_sha256"] = checkpoint_sha256
    result["checkpoint_iteration"] = int(checkpoint.get("iteration", -1))
    result["checkpoint_model_key"] = model_key
    result["dataset"] = str(dataset_path.resolve())
    result["dataset_sha256"] = _sha256_file(dataset_path.resolve())
    return result


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate", help="build isolated train/eval NPZ files")
    generate.add_argument("--train-output", type=Path, required=True)
    generate.add_argument("--eval-output", type=Path, required=True)
    generate.add_argument("--manifest-json", type=Path)
    generate.add_argument("--train-seed", type=int, default=LegalTacticsConfig.train_seed)
    generate.add_argument("--eval-seed", type=int, default=LegalTacticsConfig.eval_seed)
    generate.add_argument(
        "--train-samples-per-family",
        type=int,
        default=LegalTacticsConfig.train_samples_per_family,
    )
    generate.add_argument(
        "--eval-samples-per-family",
        type=int,
        default=LegalTacticsConfig.eval_samples_per_family,
    )
    generate.add_argument(
        "--distractor-pairs", type=int, default=LegalTacticsConfig.distractor_pairs
    )
    generate.add_argument(
        "--distractor-min-distance",
        type=int,
        default=LegalTacticsConfig.distractor_min_distance,
    )

    evaluate = commands.add_parser("evaluate", help="evaluate raw network and V3Search")
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--dataset", type=Path, required=True)
    evaluate.add_argument(
        "--model-key",
        choices=CHECKPOINT_MODEL_KEYS,
        default="best_model",
        help="checkpoint weights to evaluate (V3 candidates use candidate_model)",
    )
    evaluate.add_argument("--split", choices=(TRAIN_SPLIT, EVAL_SPLIT), default=EVAL_SPLIT)
    evaluate.add_argument("--simulations", type=int, default=64)
    evaluate.add_argument("--device")
    evaluate.add_argument("--limit", type=int)
    evaluate.add_argument("--json-out", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "generate":
        config = LegalTacticsConfig(
            train_seed=args.train_seed,
            eval_seed=args.eval_seed,
            train_samples_per_family=args.train_samples_per_family,
            eval_samples_per_family=args.eval_samples_per_family,
            distractor_pairs=args.distractor_pairs,
            distractor_min_distance=args.distractor_min_distance,
        )
        dataset = generate_legal_tactics(config)
        artifacts = write_split_archives(
            dataset, args.train_output, args.eval_output, args.manifest_json
        )
        result = {
            "records": len(dataset.states),
            "train_records": int(np.count_nonzero(dataset.split == TRAIN_SPLIT)),
            "eval_records": int(np.count_nonzero(dataset.split == EVAL_SPLIT)),
            "train_output": str(artifacts["train"]),
            "eval_output": str(artifacts["eval"]),
            "manifest": str(artifacts["manifest"]),
            "validation": dataset.summary["validation"],
        }
    else:
        result = evaluate_checkpoint(
            args.checkpoint,
            args.dataset,
            model_key=args.model_key,
            split=args.split,
            simulations=args.simulations,
            device=args.device,
            limit=args.limit,
        )
        if args.json_out is not None:
            _write_json(args.json_out, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

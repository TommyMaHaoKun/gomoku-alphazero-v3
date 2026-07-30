#!/usr/bin/env python3
"""Build a deterministic supervised tactical curriculum for Gomoku V3.

The source positions and the bounded oracle come from :mod:`v3_tactical_suite`.
Every emitted record is independently re-checked after translation, D4
symmetry, colour swap, and optional distractor placement.  A record is never
kept when its board is terminal or when the transformed oracle answer differs
from its policy target.

The compressed ``npz`` schema is deliberately simple and pickle-free::

    states          uint8   [N, 4, 19, 19]
    policies        float16 [N, 361]
    values          int8    [N]
    policy_weight   float16 [N]
    value_weight    float16 [N]
    source          unicode [N]
    priority        float16 [N]

Planes use the existing trainer contract: side-to-move stones, opponent
stones, last-move (zero because these are unordered tactical diagrams), and a
constant black-to-move plane.  Policy mass is uniform across every oracle move.
Only positions whose bounded oracle proves a win receive a supervised value of
``+1``.  Defensive positions have an unknown game outcome, so their value loss
is disabled; in particular every unique-defense record has
``value_weight == 0``.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Iterable, Sequence

import numpy as np


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from alphazero_training.v3_tactical_suite import (
    BLACK,
    BOARD_SIZE,
    EMPTY,
    WHITE,
    TacticalBoard,
    TacticalCase,
    action_of,
    built_in_cases,
    oracle_actions,
    self_check,
)


SCHEMA_VERSION = 1
ACTION_COUNT = BOARD_SIZE * BOARD_SIZE
D4_NAMES = (
    "identity",
    "rot90",
    "rot180",
    "rot270",
    "mirror_x",
    "mirror_x_rot90",
    "mirror_x_rot180",
    "mirror_x_rot270",
)


@dataclass(frozen=True)
class CurriculumConfig:
    """Deterministic augmentation controls."""

    seed: int = 20260722
    translation_stride: int = 4
    distractor_variants: int = 1
    distractors_per_variant: int = 2
    distractor_min_distance: int = 3
    distractor_max_attempts: int = 2000

    def validate(self) -> None:
        if self.translation_stride <= 0:
            raise ValueError("translation_stride must be positive")
        if self.distractor_variants < 0:
            raise ValueError("distractor_variants cannot be negative")
        if self.distractors_per_variant < 0:
            raise ValueError("distractors_per_variant cannot be negative")
        if self.distractor_min_distance < 0:
            raise ValueError("distractor_min_distance cannot be negative")
        if self.distractor_max_attempts <= 0:
            raise ValueError("distractor_max_attempts must be positive")


@dataclass(frozen=True)
class CurriculumDataset:
    states: np.ndarray
    policies: np.ndarray
    values: np.ndarray
    policy_weight: np.ndarray
    value_weight: np.ndarray
    source: np.ndarray
    priority: np.ndarray
    summary: dict[str, object]

    def arrays(self) -> dict[str, np.ndarray]:
        """Return the exact arrays written to the compressed archive."""

        return {
            "states": self.states,
            "policies": self.policies,
            "values": self.values,
            "policy_weight": self.policy_weight,
            "value_weight": self.value_weight,
            "source": self.source,
            "priority": self.priority,
        }


def d4_point(x: int, y: int, symmetry: int, size: int = BOARD_SIZE) -> tuple[int, int]:
    """Map a point through one of the eight square-board D4 symmetries."""

    if not (0 <= symmetry < 8):
        raise ValueError(f"symmetry must be in [0, 7], got {symmetry}")
    if not (0 <= x < size and 0 <= y < size):
        raise ValueError(f"point out of range: ({x}, {y})")
    if symmetry >= 4:
        x = size - 1 - x
        symmetry -= 4
    for _ in range(symmetry):
        x, y = size - 1 - y, x
    return x, y


def d4_action(action: int, symmetry: int) -> int:
    y, x = divmod(int(action), BOARD_SIZE)
    tx, ty = d4_point(x, y, symmetry)
    return action_of(tx, ty)


def _axis_offsets(minimum: int, maximum: int, stride: int) -> tuple[int, ...]:
    """Sample a legal inclusive offset range, retaining origin and both edges."""

    if minimum > maximum:
        return ()
    offsets = set(range(minimum, maximum + 1, stride))
    offsets.update((minimum, maximum))
    if minimum <= 0 <= maximum:
        offsets.add(0)
    return tuple(sorted(offsets))


def translation_offsets(case: TacticalCase, stride: int) -> tuple[tuple[int, int], ...]:
    """Return deterministic legal translations of stones and all target points."""

    if stride <= 0:
        raise ValueError("stride must be positive")
    points = [(x, y) for x, y, _stone in case.stones]
    points.extend((action % BOARD_SIZE, action // BOARD_SIZE) for action in case.declared_actions)
    if not points:
        raise ValueError(f"{case.case_id}: cannot translate an empty case")
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    dxs = _axis_offsets(-min(xs), BOARD_SIZE - 1 - max(xs), stride)
    dys = _axis_offsets(-min(ys), BOARD_SIZE - 1 - max(ys), stride)
    return tuple((dx, dy) for dy in dys for dx in dxs)


def transform_case(
    case: TacticalCase,
    dx: int,
    dy: int,
    symmetry: int,
    colour_swap: bool,
) -> TacticalCase:
    """Translate, transform, and optionally swap both stones and side-to-move."""

    stones: list[tuple[int, int, int]] = []
    for x, y, stone in case.stones:
        tx, ty = d4_point(x + dx, y + dy, symmetry)
        stones.append((tx, ty, -stone if colour_swap else stone))
    actions: list[int] = []
    for action in case.declared_actions:
        y, x = divmod(int(action), BOARD_SIZE)
        tx, ty = d4_point(x + dx, y + dy, symmetry)
        actions.append(action_of(tx, ty))
    side_to_move = -case.side_to_move if colour_swap else case.side_to_move
    return TacticalCase(
        case_id=case.case_id,
        category=case.category,
        side_to_move=side_to_move,
        stones=tuple(stones),
        oracle_kind=case.oracle_kind,
        declared_actions=tuple(sorted(actions)),
        description=case.description,
        max_plies=case.max_plies,
    )


def _case_validation_error(case: TacticalCase) -> str | None:
    """Return an explanatory validation error, or ``None`` for a sound record."""

    try:
        board = case.board
    except (IndexError, ValueError) as exc:
        return f"invalid board: {exc}"
    if board.has_existing_five(BLACK) or board.has_existing_five(WHITE):
        return "input board is terminal"
    if not case.declared_actions:
        return "policy target is empty"
    if len(set(case.declared_actions)) != len(case.declared_actions):
        return "policy target contains duplicates"
    for action in case.declared_actions:
        if not 0 <= int(action) < ACTION_COUNT:
            return f"target action out of range: {action}"
        if board.cells[int(action)] != EMPTY:
            return f"target action is occupied: {action}"
    try:
        calculated = tuple(sorted(map(int, oracle_actions(case))))
    except (IndexError, ValueError) as exc:
        return f"oracle failed: {exc}"
    declared = tuple(sorted(map(int, case.declared_actions)))
    if calculated != declared:
        return f"oracle actions {calculated} differ from target {declared}"
    return None


def _stable_rng(seed: int, source: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{seed}:{source}".encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little", signed=False))


def _chebyshev_distance(a: int, b: int) -> int:
    ay, ax = divmod(int(a), BOARD_SIZE)
    by, bx = divmod(int(b), BOARD_SIZE)
    return max(abs(ax - bx), abs(ay - by))


def add_safe_distractors(
    case: TacticalCase,
    count: int,
    *,
    seed: int,
    source: str,
    min_distance: int = 3,
    max_attempts: int = 2000,
) -> tuple[TacticalCase, tuple[tuple[int, int, int], ...]]:
    """Add deterministic random stones while preserving the exact oracle label.

    Safety is semantic, not assumed from distance alone: after every accepted
    stone the bounded oracle is recomputed and must still equal the declared
    policy target.  Distance merely makes successful, non-interacting samples
    much more likely.
    """

    if count < 0:
        raise ValueError("count cannot be negative")
    if count == 0:
        return case, ()
    baseline_error = _case_validation_error(case)
    if baseline_error is not None:
        raise ValueError(f"cannot distract invalid case {case.case_id}: {baseline_error}")

    rng = _stable_rng(seed, source)
    current = case
    added: list[tuple[int, int, int]] = []
    important = {
        action_of(x, y) for x, y, _stone in case.stones
    } | set(map(int, case.declared_actions))
    available = [
        action
        for action in range(ACTION_COUNT)
        if current.board.cells[action] == EMPTY
        and action not in important
        and all(_chebyshev_distance(action, other) >= min_distance for other in important)
    ]
    rng.shuffle(available)
    first_colour = BLACK if int(rng.integers(0, 2)) == 0 else WHITE
    attempts = 0

    while available and len(added) < count and attempts < max_attempts:
        action = int(available.pop())
        attempts += 1
        if any(
            _chebyshev_distance(action, action_of(x, y)) < min_distance
            for x, y, _stone in added
        ):
            continue
        preferred = first_colour if len(added) % 2 == 0 else -first_colour
        colour_order = (preferred, -preferred)
        for stone in colour_order:
            y, x = divmod(action, BOARD_SIZE)
            candidate = TacticalCase(
                case_id=current.case_id,
                category=current.category,
                side_to_move=current.side_to_move,
                stones=current.stones + ((x, y, stone),),
                oracle_kind=current.oracle_kind,
                declared_actions=current.declared_actions,
                description=current.description,
                max_plies=current.max_plies,
            )
            if _case_validation_error(candidate) is None:
                current = candidate
                added.append((x, y, stone))
                break

    if len(added) != count:
        raise RuntimeError(
            f"{case.case_id}: found {len(added)}/{count} safe distractors "
            f"after {attempts} attempts"
        )
    return current, tuple(added)


def encode_case(case: TacticalCase) -> np.ndarray:
    """Encode a tactical diagram using the trainer's four-plane state contract."""

    board = np.asarray(case.board.cells, dtype=np.int8).reshape(BOARD_SIZE, BOARD_SIZE)
    planes = np.zeros((4, BOARD_SIZE, BOARD_SIZE), dtype=np.uint8)
    planes[0] = board == case.side_to_move
    planes[1] = board == -case.side_to_move
    # Plane 2 remains all-zero: tactical diagrams do not have an ordered history.
    if case.side_to_move == BLACK:
        planes[3].fill(1)
    return planes


def policy_target(actions: Iterable[int]) -> np.ndarray:
    actions = tuple(sorted(set(map(int, actions))))
    if not actions:
        raise ValueError("policy target requires at least one action")
    policy = np.zeros(ACTION_COUNT, dtype=np.float16)
    policy[list(actions)] = np.float16(1.0 / len(actions))
    # Correct rounding drift while retaining float16 storage.
    remainder = np.float16(1.0 - float(policy.sum(dtype=np.float32)))
    policy[actions[0]] = np.float16(policy[actions[0]] + remainder)
    return policy


def _labels(case: TacticalCase) -> tuple[int, float, float, float]:
    """Return value, policy weight, value weight, and sampling priority."""

    if case.oracle_kind == "immediate_win":
        return 1, 1.0, 1.0, 3.0
    if case.oracle_kind == "forced_win_in_3":
        return 1, 1.0, 1.0, 4.0
    if case.oracle_kind == "immediate_block":
        return 0, 1.0, 0.0, 3.0
    if case.oracle_kind == "prevent_forced_win_in_3":
        return 0, 1.0, 0.0, 5.0
    raise ValueError(f"unsupported oracle kind: {case.oracle_kind}")


def _validate_dataset_arrays(arrays: dict[str, np.ndarray]) -> None:
    required = {
        "states",
        "policies",
        "values",
        "policy_weight",
        "value_weight",
        "source",
        "priority",
    }
    if set(arrays) != required:
        raise ValueError(f"dataset fields {sorted(arrays)} != {sorted(required)}")
    count = len(arrays["states"])
    if count == 0:
        raise ValueError("dataset is empty")
    if arrays["states"].shape != (count, 4, BOARD_SIZE, BOARD_SIZE):
        raise ValueError(f"unexpected states shape: {arrays['states'].shape}")
    if arrays["policies"].shape != (count, ACTION_COUNT):
        raise ValueError(f"unexpected policies shape: {arrays['policies'].shape}")
    for name in ("values", "policy_weight", "value_weight", "source", "priority"):
        if arrays[name].shape != (count,):
            raise ValueError(f"unexpected {name} shape: {arrays[name].shape}")
    if arrays["states"].dtype != np.uint8:
        raise ValueError("states must be uint8")
    if arrays["policies"].dtype != np.float16:
        raise ValueError("policies must be float16")
    if arrays["values"].dtype != np.int8:
        raise ValueError("values must be int8")
    if arrays["source"].dtype.kind != "U":
        raise ValueError("source must be a pickle-free Unicode array")
    if not np.allclose(arrays["policies"].sum(axis=1, dtype=np.float32), 1.0, atol=2e-3):
        raise ValueError("policy rows must sum to one")
    occupied = arrays["states"][:, 0] | arrays["states"][:, 1]
    if np.any(occupied.sum(axis=(1, 2)) == 0):
        raise ValueError("every record must contain stones")
    if np.any((arrays["states"][:, 0] & arrays["states"][:, 1]) != 0):
        raise ValueError("friendly and opponent planes overlap")
    if np.any(arrays["states"][:, 2] != 0):
        raise ValueError("last-move plane must be zero for diagram data")
    policy_grid = arrays["policies"].reshape(count, BOARD_SIZE, BOARD_SIZE)
    if np.any(policy_grid[occupied.astype(bool)] > 0):
        raise ValueError("policy assigns mass to an occupied action")
    if np.any(arrays["policy_weight"] <= 0):
        raise ValueError("policy weights must be positive")
    if np.any(arrays["value_weight"] < 0):
        raise ValueError("value weights cannot be negative")
    if np.any(arrays["priority"] <= 0):
        raise ValueError("priorities must be positive")


def generate_curriculum(
    cases: Sequence[TacticalCase] | None = None,
    config: CurriculumConfig | None = None,
) -> CurriculumDataset:
    """Generate, validate, and state-deduplicate the tactical curriculum."""

    config = config or CurriculumConfig()
    config.validate()
    using_complete_builtin_suite = cases is None
    cases = tuple(built_in_cases() if using_complete_builtin_suite else cases)
    if not cases:
        raise ValueError("at least one source tactical case is required")
    if using_complete_builtin_suite:
        checks = self_check(cases)
        if not checks["passed"]:
            raise ValueError(f"source tactical suite failed self-check: {checks['errors']}")
    else:
        # Unit tests and focused failure-mining jobs may intentionally request a
        # subset which cannot satisfy self_check's whole-suite category quota.
        for case in cases:
            error = _case_validation_error(case)
            if error is not None:
                raise ValueError(f"invalid source case {case.case_id}: {error}")

    records: list[tuple[np.ndarray, np.ndarray, int, float, float, str, float]] = []
    state_index: dict[bytes, int] = {}
    rejected = Counter()
    attempted = Counter()
    emitted_by_case = Counter()
    emitted_by_oracle = Counter()
    duplicate_count = 0

    for original in cases:
        for dx, dy in translation_offsets(original, config.translation_stride):
            for symmetry, symmetry_name in enumerate(D4_NAMES):
                for colour_swap in (False, True):
                    clean = transform_case(original, dx, dy, symmetry, colour_swap)
                    clean_error = _case_validation_error(clean)
                    attempted["clean_transforms"] += 1
                    if clean_error is not None:
                        rejected[f"clean:{clean_error}"] += 1
                        continue

                    for variant in range(config.distractor_variants + 1):
                        attempted["records"] += 1
                        source = (
                            f"{original.case_id}|dx={dx}|dy={dy}|sym={symmetry_name}"
                            f"|swap={int(colour_swap)}|d={variant}"
                        )
                        augmented = clean
                        if variant:
                            try:
                                augmented, _added = add_safe_distractors(
                                    clean,
                                    config.distractors_per_variant,
                                    seed=config.seed,
                                    source=source,
                                    min_distance=config.distractor_min_distance,
                                    max_attempts=config.distractor_max_attempts,
                                )
                            except RuntimeError as exc:
                                rejected[f"distractor:{exc}"] += 1
                                continue
                        error = _case_validation_error(augmented)
                        if error is not None:
                            rejected[f"final:{error}"] += 1
                            continue

                        state = encode_case(augmented)
                        policy = policy_target(augmented.declared_actions)
                        value, p_weight, v_weight, priority = _labels(augmented)
                        key = state.tobytes()
                        existing_index = state_index.get(key)
                        if existing_index is not None:
                            existing = records[existing_index]
                            labels_match = (
                                np.array_equal(existing[1], policy)
                                and existing[2] == value
                                and existing[3] == p_weight
                                and existing[4] == v_weight
                            )
                            if not labels_match:
                                raise ValueError(
                                    "conflicting labels for an identical encoded state: "
                                    f"{existing[5]} versus {source}"
                                )
                            # Preserve the strongest requested sampling priority.
                            if priority > existing[6]:
                                records[existing_index] = existing[:-1] + (priority,)
                            duplicate_count += 1
                            continue

                        state_index[key] = len(records)
                        records.append(
                            (state, policy, value, p_weight, v_weight, source, priority)
                        )
                        emitted_by_case[original.case_id] += 1
                        emitted_by_oracle[original.oracle_kind] += 1

    if not records:
        raise RuntimeError("augmentation produced no valid records")

    sources = [record[5] for record in records]
    source_width = max(map(len, sources))
    arrays = {
        "states": np.stack([record[0] for record in records]).astype(np.uint8, copy=False),
        "policies": np.stack([record[1] for record in records]).astype(np.float16, copy=False),
        "values": np.asarray([record[2] for record in records], dtype=np.int8),
        "policy_weight": np.asarray([record[3] for record in records], dtype=np.float16),
        "value_weight": np.asarray([record[4] for record in records], dtype=np.float16),
        "source": np.asarray(sources, dtype=f"<U{source_width}"),
        "priority": np.asarray([record[6] for record in records], dtype=np.float16),
    }
    _validate_dataset_arrays(arrays)

    value_supervised = int(np.count_nonzero(arrays["value_weight"]))
    summary: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "dataset": "v3_freestyle_gomoku_tactical_curriculum",
        "rules": {"board_size": BOARD_SIZE, "win_length": 5, "overline_wins": True},
        "config": {
            "seed": config.seed,
            "translation_stride": config.translation_stride,
            "d4_symmetries": list(D4_NAMES),
            "colour_swaps": [False, True],
            "distractor_variants": config.distractor_variants,
            "distractors_per_variant": config.distractors_per_variant,
            "distractor_min_distance": config.distractor_min_distance,
        },
        "counts": {
            "source_cases": len(cases),
            "clean_transform_attempts": attempted["clean_transforms"],
            "record_attempts": attempted["records"],
            "records": len(records),
            "duplicates_removed": duplicate_count,
            "rejected": int(sum(rejected.values())),
            "value_supervised": value_supervised,
            "policy_only": len(records) - value_supervised,
        },
        "emitted_by_case": dict(sorted(emitted_by_case.items())),
        "emitted_by_oracle": dict(sorted(emitted_by_oracle.items())),
        "rejections": dict(sorted(rejected.items())),
        "label_contract": {
            "policy": "uniform over the complete bounded-oracle action set",
            "forcing_win_value": "+1 from the side-to-move perspective",
            "defense_value": "target 0 with value_weight 0 because outcome is not proven",
            "unique_defense_value_weight": 0,
            "last_move_plane": "all zero because source diagrams have no ordered history",
        },
        "arrays": {
            name: {"shape": list(array.shape), "dtype": str(array.dtype)}
            for name, array in arrays.items()
        },
    }
    return CurriculumDataset(summary=summary, **arrays)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_curriculum(
    dataset: CurriculumDataset,
    output: Path,
    summary_json: Path | None = None,
) -> tuple[Path, Path]:
    """Atomically write the compressed archive and its human-readable summary."""

    output = output.resolve()
    if output.suffix.lower() != ".npz":
        raise ValueError("output filename must end in .npz")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **dataset.arrays())
    temporary.replace(output)

    summary_path = (
        summary_json.resolve()
        if summary_json is not None
        else output.with_suffix(".summary.json")
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = dict(dataset.summary)
    summary["artifact"] = {
        "path": str(output),
        "bytes": output.stat().st_size,
        "sha256": sha256_file(output),
    }
    payload = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    summary_temporary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    summary_temporary.write_text(payload, encoding="utf-8")
    summary_temporary.replace(summary_path)
    return output, summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="destination compressed .npz")
    parser.add_argument("--summary-json", type=Path, help="optional summary JSON path")
    parser.add_argument("--seed", type=int, default=CurriculumConfig.seed)
    parser.add_argument(
        "--translation-stride", type=int, default=CurriculumConfig.translation_stride
    )
    parser.add_argument(
        "--distractor-variants", type=int, default=CurriculumConfig.distractor_variants
    )
    parser.add_argument(
        "--distractors-per-variant",
        type=int,
        default=CurriculumConfig.distractors_per_variant,
    )
    parser.add_argument(
        "--distractor-min-distance",
        type=int,
        default=CurriculumConfig.distractor_min_distance,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = CurriculumConfig(
        seed=args.seed,
        translation_stride=args.translation_stride,
        distractor_variants=args.distractor_variants,
        distractors_per_variant=args.distractors_per_variant,
        distractor_min_distance=args.distractor_min_distance,
    )
    dataset = generate_curriculum(config=config)
    output, summary_path = write_curriculum(dataset, args.output, args.summary_json)
    result = {
        "records": len(dataset.states),
        "output": str(output),
        "summary_json": str(summary_path),
        "sha256": sha256_file(output),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

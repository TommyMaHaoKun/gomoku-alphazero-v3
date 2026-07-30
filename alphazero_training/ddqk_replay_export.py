"""Convert complete DDQK benchmark or teacher games into replay samples.

Only moves actually selected by DDQK are exported.  Random opening moves and,
for model-vs-engine benchmarks, AlphaZero moves are never mislabeled as expert
decisions.  DDQK-vs-DDQK teacher reports export both sides with the genuine
terminal value from the side-to-move perspective.  The four input planes
exactly match :meth:`train_alphazero.GomokuGame.encode`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

try:
    from .benchmark_ddqk import (
        DEVELOPMENT_MODE,
        DDQK_DECISION_ASSET_FILES,
        EVALUATION_CODE_FILES,
        FINAL_CERTIFICATION_MODE,
        FINAL_MIN_EXACT_PAIR_SWEEP_LOWER95,
        FINAL_MIN_COLOR_SCORE,
        FINAL_MIN_PAIRS,
        FINAL_MIN_SCORE,
        GameRecord as BenchmarkGameRecord,
        generate_opening as generate_benchmark_opening,
        stable_json_sha256,
        summarize as summarize_benchmark,
    )
except ImportError:  # Support direct execution from this directory.
    from benchmark_ddqk import (  # type: ignore[no-redef]
        DEVELOPMENT_MODE,
        DDQK_DECISION_ASSET_FILES,
        EVALUATION_CODE_FILES,
        FINAL_CERTIFICATION_MODE,
        FINAL_MIN_EXACT_PAIR_SWEEP_LOWER95,
        FINAL_MIN_COLOR_SCORE,
        FINAL_MIN_PAIRS,
        FINAL_MIN_SCORE,
        GameRecord as BenchmarkGameRecord,
        generate_opening as generate_benchmark_opening,
        stable_json_sha256,
        summarize as summarize_benchmark,
    )

try:
    from .generate_ddqk_teacher import (
        OpeningMember,
        TeacherGameRecord,
        build_opening_manifest,
        canonical_sha256,
        manifest_payload,
        record_key,
        summarize as summarize_teacher,
        validate_record,
    )
except ImportError:  # Support direct execution from this directory.
    from generate_ddqk_teacher import (  # type: ignore[no-redef]
        OpeningMember,
        TeacherGameRecord,
        build_opening_manifest,
        canonical_sha256,
        manifest_payload,
        record_key,
        summarize as summarize_teacher,
        validate_record,
    )


BOARD_SIZE = 19
BLACK = 1
WHITE = 2
EMPTY = 0
DIRECTIONS = ((1, 0), (0, 1), (1, 1), (1, -1))
TEACHER_REPORT_TYPE = "ddqk_teacher_selfplay"
SHA256_HEX_LENGTH = 64
LEGACY4_EVALUATION_CODE_FILES = (
    "play_agent.py",
    "v3_search.py",
    "tactical_solver.py",
    "benchmark_ddqk.py",
)
CURRENT6_EVALUATION_CODE_FILES = tuple(EVALUATION_CODE_FILES)
if len(CURRENT6_EVALUATION_CODE_FILES) != 6:  # pragma: no cover - contract guard
    raise RuntimeError("current DDQK evaluation-code provenance must contain six files")

LEGACY4_SUMMARY_FIELDS = (
    "games",
    "completed_games",
    "scored_games",
    "requested_pairs",
    "complete_pairs",
    "incomplete_pairs",
    "errors",
    "truncated",
    "wins",
    "losses",
    "draws",
    "score",
    "paired_bootstrap_ci95",
    "one_sided_95_lower_bound",
    "one_sided_95_lower_bound_method",
    "by_color",
    "mean_plies",
    "model_seconds_per_move",
    "ddqk_seconds_per_move",
    "model_decision_reasons",
)


def rebuild_benchmark_openings(
    *,
    seed: int,
    pairs: int,
    opening_plies: int,
) -> list[list[list[int]]]:
    """Rebuild the deterministic benchmark manifest from its signed recipe."""

    rng = np.random.default_rng(seed)
    openings = [
        [list(move) for move in generate_benchmark_opening(rng, opening_plies)]
        for _ in range(pairs)
    ]
    if any(len(opening) != opening_plies for opening in openings):
        raise ValueError(
            "benchmark opening generator could not produce the signed opening_plies"
        )
    return openings


def _require_sha256(value: object, *, field: str) -> str:
    text = str(value)
    if len(text) != SHA256_HEX_LENGTH or any(
        character not in "0123456789abcdef" for character in text.lower()
    ):
        raise ValueError(f"teacher report {field} is not a SHA256 hex digest")
    return text.lower()


def _summary_matches(
    recorded: object,
    expected: dict[str, object],
) -> bool:
    if not isinstance(recorded, dict) or set(recorded) != set(expected):
        return False
    for key, expected_value in expected.items():
        recorded_value = recorded[key]
        if isinstance(expected_value, float):
            try:
                if not math.isclose(
                    float(recorded_value), expected_value, rel_tol=1e-12, abs_tol=1e-12
                ):
                    return False
            except (TypeError, ValueError):
                return False
        elif recorded_value != expected_value:
            return False
    return True


def validate_teacher_report(
    report: dict[str, object],
    *,
    allow_partial: bool,
) -> dict[str, object]:
    """Strictly authenticate and replay a DDQK-vs-DDQK teacher report.

    The report is treated as an untrusted serialization.  In particular, a
    ``complete`` flag alone is never sufficient to make a game exportable.
    The deterministic opening manifest is rebuilt from its signed recipe and
    every recorded game is replayed against the corresponding member.
    """

    if report.get("game_mode") != "ddqk_vs_ddqk":
        raise ValueError("teacher report game_mode must be ddqk_vs_ddqk")
    rules = report.get("rules")
    if rules != {"board_size": BOARD_SIZE, "win_length": 5, "freestyle": True}:
        raise ValueError("teacher report rules do not match 19x19 freestyle Gomoku")
    signature = report.get("signature")
    if not isinstance(signature, dict):
        raise ValueError("teacher report signature must be an object")

    integer_fields = (
        "groups",
        "games_per_group",
        "opening_plies",
        "max_moves",
        "seed",
    )
    values: dict[str, int] = {}
    for field in integer_fields:
        try:
            value = int(signature[field])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"teacher report signature has invalid {field}") from exc
        if report.get(field) != value:
            raise ValueError(f"teacher report top-level {field} disagrees with signature")
        values[field] = value
    if values["groups"] <= 0 or not 1 <= values["games_per_group"] <= 8:
        raise ValueError("teacher report has invalid group dimensions")
    if not 0 <= values["opening_plies"] <= values["max_moves"] <= BOARD_SIZE**2:
        raise ValueError("teacher report has invalid opening/max-move limits")
    try:
        workers = int(report["workers"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("teacher report has invalid workers") from exc
    if workers <= 0:
        raise ValueError("teacher report workers must be positive")

    for field in (
        "generator_sha256",
        "ddqk_source_sha256",
        "ddqk_dll_sha256",
        "opening_manifest_sha256",
    ):
        try:
            _require_sha256(signature[field], field=f"signature.{field}")
        except KeyError as exc:
            raise ValueError(f"teacher report signature is missing {field}") from exc
    if not isinstance(signature.get("ddqk_source"), str) or not signature["ddqk_source"]:
        raise ValueError("teacher report signature is missing ddqk_source")
    if report.get("ddqk_source") != signature["ddqk_source"]:
        raise ValueError("teacher report ddqk_source disagrees with signature")
    if not isinstance(signature.get("ddqk_dll"), str) or not signature["ddqk_dll"]:
        raise ValueError("teacher report signature is missing ddqk_dll")
    try:
        if int(signature["ddqk_depth"]) <= 0:
            raise ValueError("teacher report signature has invalid ddqk_depth")
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("teacher report signature has invalid ddqk_depth") from exc
    if int(signature.get("board_size", -1)) != BOARD_SIZE or int(
        signature.get("win_length", -1)
    ) != 5:
        raise ValueError("teacher report signature has incompatible board rules")

    members = build_opening_manifest(
        seed=values["seed"],
        groups=values["groups"],
        games_per_group=values["games_per_group"],
        opening_plies=values["opening_plies"],
    )
    rebuilt_manifest = manifest_payload(members)
    if report.get("opening_manifest") != rebuilt_manifest:
        raise ValueError("teacher report opening_manifest does not match its recipe")
    if canonical_sha256(rebuilt_manifest) != str(
        signature["opening_manifest_sha256"]
    ).lower():
        raise ValueError("teacher report opening_manifest SHA256 does not match")
    expected_openings = [member.opening for member in members]
    if report.get("openings") != expected_openings:
        raise ValueError("teacher report flat openings do not match opening_manifest")

    raw_games = report.get("games")
    if not isinstance(raw_games, list):
        raise ValueError("teacher report games must be a list")
    member_by_key = {record_key(member): member for member in members}
    records: list[TeacherGameRecord] = []
    seen: set[tuple[int, int]] = set()
    for game_index, raw_game in enumerate(raw_games):
        if not isinstance(raw_game, dict):
            raise ValueError(f"teacher report game {game_index} must be an object")
        try:
            record = TeacherGameRecord(**raw_game)
        except TypeError as exc:
            raise ValueError(
                f"teacher report game {game_index} has an invalid schema: {exc}"
            ) from exc
        key = record_key(record)
        if key not in member_by_key:
            raise ValueError(f"teacher report contains unexpected game {key}")
        if key in seen:
            raise ValueError(f"teacher report contains duplicate game {key}")
        seen.add(key)
        validate_record(record, member_by_key[key])
        successful_engine_moves = record.plies - len(record.opening)
        if record.complete:
            if record.ddqk_moves != successful_engine_moves:
                raise ValueError(
                    f"teacher report game {key} ddqk_moves does not match its history"
                )
        elif not successful_engine_moves <= record.ddqk_moves <= successful_engine_moves + 1:
            raise ValueError(
                f"teacher report game {key} has impossible ddqk_moves accounting"
            )
        if (
            not math.isfinite(float(record.ddqk_seconds))
            or float(record.ddqk_seconds) < 0.0
        ):
            raise ValueError(f"teacher report game {key} has invalid ddqk_seconds")
        if record.plies > values["max_moves"]:
            raise ValueError(f"teacher report game {key} exceeds max_moves")
        records.append(record)

    expected_order = [record_key(member) for member in members]
    recorded_order = [record_key(record) for record in records]
    if recorded_order != [key for key in expected_order if key in seen]:
        raise ValueError("teacher report games are not in opening-manifest order")

    expected_summary = summarize_teacher(
        records,
        expected_games=len(members),
        games_per_group=values["games_per_group"],
    )
    if not _summary_matches(report.get("summary"), expected_summary):
        raise ValueError("teacher report summary does not match replayed records")
    collection_status = str(expected_summary["collection_status"])
    if not allow_partial and collection_status != "complete":
        raise ValueError(
            "teacher report collection_status is not complete; "
            "use allow_partial=True only for an intentional partial export"
        )

    return {
        "collection_status": collection_status,
        "expected_games": int(expected_summary["expected_games"]),
        "recorded_games": int(expected_summary["recorded_games"]),
        "usable_complete_games": int(expected_summary["usable_complete_games"]),
        "failed_or_truncated_games": int(
            expected_summary["failed_or_truncated_games"]
        ),
        "complete_groups": int(
            expected_summary["groups_whose_recorded_games_are_usable"]
        ),
    }


def validate_benchmark_report(report: dict[str, object]) -> dict[str, object]:
    """Authenticate a completed or resumable paired DDQK benchmark report.

    Format 3 adds the immutable evaluation-code bundle and certification
    contract.  Those fields are checked rather than discarded so a report
    cannot be converted into trusted training data after its engine, search
    budget, opening manifest, or summary has been edited.  Legacy format 2 is
    retained for the already collected baseline corpus.
    """

    try:
        format_version = int(report["format_version"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("benchmark report has invalid format_version") from exc
    if format_version not in (2, 3):
        raise ValueError("expected DDQK benchmark format_version 2 or 3")
    signature = report.get("signature")
    if not isinstance(signature, dict):
        raise ValueError("benchmark report signature must be an object")

    # Format 2 predates the signed opening recipe and code-bundle contract.
    # Preserve import compatibility for that historical corpus; the replay
    # loop below still validates every exported move and terminal result.
    if format_version == 2:
        raw_games = report.get("games")
        if not isinstance(raw_games, list):
            raise ValueError("benchmark games must be a list")
        for field in ("ddqk_source_sha256", "ddqk_dll_sha256"):
            if field in signature:
                _require_sha256(signature[field], field=f"signature.{field}")
        return {
            "format_version": 2,
            "certification_mode": "legacy_unsigned",
            "requested_pairs": report.get("pairs"),
            "recorded_games": len(raw_games),
            "complete_pairs": report.get("summary", {}).get("complete_pairs")
            if isinstance(report.get("summary"), dict)
            else None,
        }

    integer_fields = ("pairs", "opening_plies", "simulations", "max_moves", "seed")
    values: dict[str, int] = {}
    for field in integer_fields:
        try:
            value = int(signature[field])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"benchmark signature has invalid {field}") from exc
        if report.get(field) != value:
            raise ValueError(f"benchmark top-level {field} disagrees with signature")
        values[field] = value
    if values["pairs"] <= 0 or values["simulations"] <= 0:
        raise ValueError("benchmark pairs and simulations must be positive")
    if not 0 <= values["opening_plies"] <= values["max_moves"] <= BOARD_SIZE**2:
        raise ValueError("benchmark has invalid opening/max-move limits")
    try:
        workers = int(report["workers"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("benchmark has invalid workers") from exc
    if workers <= 0:
        raise ValueError("benchmark workers must be positive")

    for field in ("checkpoint_sha256", "ddqk_source_sha256", "ddqk_dll_sha256"):
        try:
            _require_sha256(signature[field], field=f"signature.{field}")
        except KeyError as exc:
            raise ValueError(f"benchmark signature is missing {field}") from exc
    if not isinstance(signature.get("ddqk_source"), str) or not signature["ddqk_source"]:
        raise ValueError("benchmark signature is missing ddqk_source")
    if report.get("ddqk_source") != signature["ddqk_source"]:
        raise ValueError("benchmark ddqk_source disagrees with signature")
    try:
        if int(signature["ddqk_depth"]) <= 0:
            raise ValueError
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("benchmark signature has invalid ddqk_depth") from exc

    expected_openings = rebuild_benchmark_openings(
        seed=values["seed"],
        pairs=values["pairs"],
        opening_plies=values["opening_plies"],
    )
    if report.get("openings") != expected_openings:
        raise ValueError("benchmark opening manifest does not match its signed recipe")

    provenance_generation: str | None = None
    asset_bundle_sha: str | None = None
    if format_version == 3:
        try:
            manifest_sha = _require_sha256(
                signature["opening_manifest_sha256"],
                field="signature.opening_manifest_sha256",
            )
        except KeyError as exc:
            raise ValueError("benchmark signature is missing opening_manifest_sha256") from exc
        if manifest_sha != stable_json_sha256(expected_openings):
            raise ValueError("benchmark opening manifest SHA256 does not match")
        code = signature.get("evaluation_code")
        if not isinstance(code, dict) or not isinstance(code.get("files"), dict):
            raise ValueError("benchmark evaluation_code signature is invalid")
        file_hashes = code["files"]
        if set(file_hashes) == set(LEGACY4_EVALUATION_CODE_FILES):
            provenance_generation = "legacy4"
            provenance_files = LEGACY4_EVALUATION_CODE_FILES
        elif set(file_hashes) == set(CURRENT6_EVALUATION_CODE_FILES):
            provenance_generation = "current6"
            provenance_files = CURRENT6_EVALUATION_CODE_FILES
        else:
            raise ValueError("benchmark evaluation_code file set is invalid")
        normalized_hashes = {
            name: _require_sha256(file_hashes[name], field=f"evaluation_code.{name}")
            for name in provenance_files
        }
        bundle_sha = _require_sha256(
            code.get("bundle_sha256"), field="evaluation_code.bundle_sha256"
        )
        if bundle_sha != stable_json_sha256(normalized_hashes):
            raise ValueError("benchmark evaluation_code bundle SHA256 does not match")

        raw_assets = signature.get("ddqk_assets")
        if provenance_generation == "current6" and not isinstance(raw_assets, dict):
            raise ValueError(
                "current6 benchmark signature is missing the DDQK asset bundle"
            )
        if raw_assets is not None:
            if not isinstance(raw_assets, dict) or not isinstance(
                raw_assets.get("files"), dict
            ):
                raise ValueError("benchmark DDQK asset signature is invalid")
            asset_hashes = raw_assets["files"]
            if set(asset_hashes) != set(DDQK_DECISION_ASSET_FILES):
                raise ValueError("benchmark DDQK asset file set is invalid")
            normalized_asset_hashes = {
                name: _require_sha256(
                    asset_hashes[name], field=f"ddqk_assets.{name}"
                )
                for name in DDQK_DECISION_ASSET_FILES
            }
            asset_bundle_sha = _require_sha256(
                raw_assets.get("bundle_sha256"),
                field="ddqk_assets.bundle_sha256",
            )
            if asset_bundle_sha != stable_json_sha256(normalized_asset_hashes):
                raise ValueError("benchmark DDQK asset bundle SHA256 does not match")
            if normalized_asset_hashes["dll.so"] != str(
                signature["ddqk_dll_sha256"]
            ).lower():
                raise ValueError(
                    "benchmark DDQK DLL SHA256 disagrees with its asset bundle"
                )

    raw_games = report.get("games")
    if not isinstance(raw_games, list):
        raise ValueError("benchmark games must be a list")
    records: list[BenchmarkGameRecord] = []
    seen: set[tuple[int, int]] = set()
    expected_order = [
        (pair_index, model_color)
        for pair_index in range(values["pairs"])
        for model_color in (BLACK, WHITE)
    ]
    for game_index, raw_game in enumerate(raw_games):
        if not isinstance(raw_game, dict):
            raise ValueError(f"benchmark game {game_index} must be an object")
        try:
            record = BenchmarkGameRecord(**raw_game)
        except TypeError as exc:
            raise ValueError(f"benchmark game {game_index} has invalid schema: {exc}") from exc
        key = (int(record.pair_index), int(record.model_color))
        if key not in expected_order:
            raise ValueError(f"benchmark contains unexpected game {key}")
        if key in seen:
            raise ValueError(f"benchmark contains duplicate game {key}")
        seen.add(key)
        if record.opening != expected_openings[record.pair_index]:
            raise ValueError(f"benchmark game {key} opening does not match manifest")
        validate_benchmark_game_history(
            raw_game,
            expected_opening=expected_openings[record.pair_index],
            expected_opening_plies=values["opening_plies"],
            max_moves=values["max_moves"],
            label=f"benchmark game {key}",
        )
        for timing_name, timing_value in (
            ("model_seconds", record.model_seconds),
            ("ddqk_seconds", record.ddqk_seconds),
        ):
            if not math.isfinite(float(timing_value)) or float(timing_value) < 0.0:
                raise ValueError(f"benchmark game {key} has invalid {timing_name}")
        records.append(record)
    recorded_order = [(record.pair_index, record.model_color) for record in records]
    if recorded_order != [key for key in expected_order if key in seen]:
        raise ValueError("benchmark games are not in opening-manifest order")

    current_summary = summarize_benchmark(records, requested_pairs=values["pairs"])
    expected_summary = (
        {field: current_summary[field] for field in LEGACY4_SUMMARY_FIELDS}
        if provenance_generation == "legacy4"
        else current_summary
    )
    if not _summary_matches(report.get("summary"), expected_summary):
        raise ValueError("benchmark summary does not match replayed records")

    certification_mode = str(signature.get("certification_mode", DEVELOPMENT_MODE))
    if format_version == 3:
        if certification_mode not in (DEVELOPMENT_MODE, FINAL_CERTIFICATION_MODE):
            raise ValueError("benchmark certification_mode is invalid")
        if (
            provenance_generation == "legacy4"
            and certification_mode != DEVELOPMENT_MODE
        ):
            raise ValueError(
                "legacy4 benchmark provenance is permitted only for development export"
            )
        certification = report.get("certification")
        if not isinstance(certification, dict) or certification.get("mode") != certification_mode:
            raise ValueError("benchmark certification disagrees with signature")
        final_passed = (
            certification_mode == FINAL_CERTIFICATION_MODE
            and int(expected_summary["complete_pairs"]) == values["pairs"]
            and int(expected_summary["incomplete_pairs"]) == 0
            and int(expected_summary["errors"]) == 0
            and int(expected_summary["truncated"]) == 0
            and values["pairs"] >= FINAL_MIN_PAIRS
            and float(expected_summary["score"]) >= FINAL_MIN_SCORE
            and float(expected_summary["by_color"]["black"]["score"])
            >= FINAL_MIN_COLOR_SCORE
            and float(expected_summary["by_color"]["white"]["score"])
            >= FINAL_MIN_COLOR_SCORE
            and provenance_generation == "current6"
            and float(expected_summary["exact_pair_sweep_lower95"])
            >= FINAL_MIN_EXACT_PAIR_SWEEP_LOWER95
        )
        requirements = (
            {
                "minimum_independent_paired_openings": FINAL_MIN_PAIRS,
                "minimum_observed_score": FINAL_MIN_SCORE,
                "minimum_observed_black_score": FINAL_MIN_COLOR_SCORE,
                "minimum_observed_white_score": FINAL_MIN_COLOR_SCORE,
                "requires_conservative_one_sided_95_lower_bound": True,
            }
            if provenance_generation == "legacy4"
            else {
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
            }
        )
        expected_certification = {
            "mode": certification_mode,
            "status": (
                "benchmark_final_requirements_passed"
                if final_passed
                else "not_final_certified"
            ),
            "final_certified": bool(final_passed),
            "requirements": requirements,
        }
        if certification != expected_certification:
            raise ValueError("benchmark certification metadata does not match results")

    return {
        "format_version": format_version,
        "certification_mode": certification_mode,
        "requested_pairs": values["pairs"],
        "recorded_games": len(records),
        "complete_pairs": int(expected_summary["complete_pairs"]),
        "provenance_generation": provenance_generation,
        "ddqk_assets_bundle_sha256": asset_bundle_sha,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def encode_state(
    board: np.ndarray,
    player: int,
    last_action: int,
) -> np.ndarray:
    state = np.zeros((4, BOARD_SIZE, BOARD_SIZE), dtype=np.uint8)
    state[0] = board == player
    state[1] = (board != EMPTY) & (board != player)
    if last_action >= 0:
        y, x = divmod(last_action, BOARD_SIZE)
        state[2, y, x] = 1
    if player == BLACK:
        state[3].fill(1)
    return state


def policy_target(board: np.ndarray, action: int, smoothing: float) -> np.ndarray:
    legal = np.flatnonzero(board.ravel() == EMPTY)
    if action not in legal:
        raise ValueError(f"expert action {action} is not legal")
    policy = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32)
    policy[legal] = smoothing / len(legal)
    policy[action] += 1.0 - smoothing
    return policy


def is_win(board: np.ndarray, x: int, y: int, player: int) -> bool:
    """Freestyle Gomoku terminal check used to audit imported records."""

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


def _history_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _history_number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def validate_benchmark_game_history(
    game: Mapping[str, object],
    *,
    expected_opening: list[list[int]],
    expected_opening_plies: int,
    max_moves: int,
    label: str,
) -> dict[str, object]:
    """Replay one benchmark game and derive its trusted terminal result.

    The JSON summary and ``model_result`` are untrusted inputs.  Promotion and
    replay export both call this routine so the result is derived from the
    legal move history, not merely copied from mutable report fields.
    """

    if len(expected_opening) != expected_opening_plies:
        raise ValueError(
            f"{label}: opening length {len(expected_opening)} does not match "
            f"signed opening_plies {expected_opening_plies}"
        )
    if game.get("opening") != expected_opening:
        raise ValueError(f"{label}: opening does not match its signed manifest")

    model_color = _history_int(game.get("model_color"), label=f"{label} model_color")
    if model_color not in (BLACK, WHITE):
        raise ValueError(f"{label}: invalid model_color {model_color}")
    winner = _history_int(game.get("winner"), label=f"{label} winner")
    if winner not in (EMPTY, BLACK, WHITE):
        raise ValueError(f"{label}: invalid winner {winner}")
    plies = _history_int(game.get("plies"), label=f"{label} plies")
    model_moves = _history_int(
        game.get("model_moves"), label=f"{label} model_moves"
    )
    ddqk_moves = _history_int(
        game.get("ddqk_moves"), label=f"{label} ddqk_moves"
    )
    if min(plies, model_moves, ddqk_moves) < 0:
        raise ValueError(f"{label}: move counts must be non-negative")
    for timing_name in ("model_seconds", "ddqk_seconds"):
        if _history_number(game.get(timing_name), label=f"{label} {timing_name}") < 0.0:
            raise ValueError(f"{label}: {timing_name} must be non-negative")

    raw_moves = game.get("moves")
    if not isinstance(raw_moves, list):
        raise ValueError(f"{label}: moves must be a list")
    if plies != len(raw_moves):
        raise ValueError(f"{label}: plies do not match move history")
    if not expected_opening_plies <= plies <= max_moves:
        raise ValueError(f"{label}: move history violates opening/max-move limits")

    reasons = game.get("model_decision_reasons", [])
    if not isinstance(reasons, list) or any(not isinstance(item, str) for item in reasons):
        raise ValueError(f"{label}: model_decision_reasons must be a string list")
    if len(reasons) != model_moves:
        raise ValueError(f"{label}: model decision reasons do not match model_moves")

    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
    observed_winner = EMPTY
    successful_model_moves = 0
    successful_ddqk_moves = 0
    for ply_index, raw_move in enumerate(raw_moves):
        if not isinstance(raw_move, list) or len(raw_move) != 3:
            raise ValueError(f"{label} ply {ply_index}: malformed move")
        x = _history_int(raw_move[0], label=f"{label} ply {ply_index} x")
        y = _history_int(raw_move[1], label=f"{label} ply {ply_index} y")
        player = _history_int(
            raw_move[2], label=f"{label} ply {ply_index} player"
        )
        expected_player = BLACK if ply_index % 2 == 0 else WHITE
        if player != expected_player:
            raise ValueError(
                f"{label} ply {ply_index}: expected player {expected_player}, got {player}"
            )
        if not (0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE):
            raise ValueError(f"{label} ply {ply_index}: move outside board {(x, y)}")
        if board[y, x] != EMPTY:
            raise ValueError(f"{label} ply {ply_index}: repeated move {(x, y)}")
        if observed_winner != EMPTY:
            raise ValueError(f"{label}: contains moves after a win")
        if ply_index < expected_opening_plies:
            if expected_opening[ply_index] != [x, y]:
                raise ValueError(f"{label} ply {ply_index}: opening manifest mismatch")
        elif player == model_color:
            successful_model_moves += 1
        else:
            successful_ddqk_moves += 1
        board[y, x] = player
        if is_win(board, x, y, player):
            if ply_index < expected_opening_plies:
                raise ValueError(f"{label}: signed opening is already terminal")
            observed_winner = player

    termination = game.get("termination")
    if not isinstance(termination, str):
        raise ValueError(f"{label}: termination must be a string")
    error = game.get("error")
    if error is not None and (not isinstance(error, str) or not error):
        raise ValueError(f"{label}: error must be null or a non-empty string")

    if observed_winner != EMPTY:
        expected_termination = "win"
        expected_result: float | None = (
            1.0 if observed_winner == model_color else 0.0
        )
        if error is not None:
            raise ValueError(f"{label}: winning game cannot contain an engine error")
    elif not np.any(board == EMPTY):
        expected_termination = "full_board_draw"
        expected_result = 0.5
        if error is not None:
            raise ValueError(f"{label}: full-board draw cannot contain an engine error")
    else:
        expected_termination = "truncated" if error is None else termination
        expected_result = None
        if error is None and plies != max_moves:
            raise ValueError(f"{label}: unexplained non-terminal early stop")
        if error is not None and termination not in ("engine_error", "truncated"):
            raise ValueError(f"{label}: errored game has an invalid termination")

    if winner != observed_winner:
        raise ValueError(f"{label}: winner does not match replayed terminal board")
    if termination != expected_termination:
        raise ValueError(
            f"{label}: termination {termination!r} does not match replayed "
            f"{expected_termination!r}"
        )
    raw_result = game.get("model_result")
    if expected_result is None:
        if raw_result is not None:
            raise ValueError(f"{label}: incomplete game must not have model_result")
    else:
        reported_result = _history_number(raw_result, label=f"{label} model_result")
        if reported_result != expected_result:
            raise ValueError(f"{label}: model_result does not match replayed winner")

    expected_model_moves = successful_model_moves
    expected_ddqk_moves = successful_ddqk_moves
    if error is None:
        if model_moves != expected_model_moves or ddqk_moves != expected_ddqk_moves:
            raise ValueError(f"{label}: engine move counts do not match move history")
    else:
        next_player = BLACK if plies % 2 == 0 else WHITE
        model_allowance = 1 if next_player == model_color else 0
        ddqk_allowance = 1 - model_allowance
        if model_moves not in (
            expected_model_moves,
            expected_model_moves + model_allowance,
        ) or ddqk_moves not in (
            expected_ddqk_moves,
            expected_ddqk_moves + ddqk_allowance,
        ):
            raise ValueError(f"{label}: errored engine move counts are impossible")

    return {
        "winner": observed_winner,
        "model_result": expected_result,
        "termination": expected_termination,
        "plies": plies,
        "model_moves": model_moves,
        "ddqk_moves": ddqk_moves,
    }


def d4_examples(
    state: np.ndarray,
    policy: np.ndarray,
) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    policy_board = policy.reshape(BOARD_SIZE, BOARD_SIZE)
    for rotations in range(4):
        rotated_state = np.rot90(state, rotations, axes=(-2, -1))
        rotated_policy = np.rot90(policy_board, rotations)
        yield (
            np.ascontiguousarray(rotated_state),
            np.ascontiguousarray(rotated_policy).reshape(-1),
        )
        yield (
            np.ascontiguousarray(np.flip(rotated_state, axis=-1)),
            np.ascontiguousarray(np.flip(rotated_policy, axis=-1)).reshape(-1),
        )


def export_report(
    report_path: Path,
    *,
    smoothing: float = 0.05,
    include_ddqk_losses: bool = False,
    augment: bool = False,
    policy_only: bool = False,
    allow_partial: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    if not 0.0 <= smoothing < 1.0:
        raise ValueError("smoothing must be in [0, 1)")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("DDQK report must be a JSON object")
    teacher_selfplay = report.get("report_type") == TEACHER_REPORT_TYPE
    signature = report.get("signature", {})
    teacher_audit = (
        validate_teacher_report(report, allow_partial=allow_partial)
        if teacher_selfplay
        else None
    )
    benchmark_audit = (
        None if teacher_selfplay else validate_benchmark_report(report)
    )

    states: list[np.ndarray] = []
    policies: list[np.ndarray] = []
    values: list[float] = []
    policy_weights: list[float] = []
    value_weights: list[float] = []
    priorities: list[float] = []
    game_indices: list[int] = []
    pair_indices: list[int] = []
    ply_indices: list[int] = []
    skipped_incomplete = 0
    skipped_ddqk_losses = 0

    for game_index, game in enumerate(report.get("games", [])):
        if teacher_selfplay:
            derived_complete = game.get("error") is None and game.get(
                "termination"
            ) in ("win", "full_board_draw")
            if bool(game.get("complete")) != derived_complete:
                raise ValueError(
                    f"game {game_index}: false or inconsistent completion flag"
                )
        if game.get("error") is not None or game.get("termination") not in (
            "win",
            "full_board_draw",
        ):
            skipped_incomplete += 1
            continue
        winner = int(game["winner"])
        if winner not in (EMPTY, BLACK, WHITE):
            raise ValueError(f"game {game_index}: invalid winner {winner}")
        if teacher_selfplay:
            # Both colours are demonstrations.  Losing-side moves still receive
            # their genuine -1 terminal value rather than a fabricated win.
            expert_players = {BLACK, WHITE}
            ddqk_color: int | None = None
        else:
            model_color = int(game["model_color"])
            if model_color not in (BLACK, WHITE):
                raise ValueError(f"game {game_index}: invalid model_color {model_color}")
            ddqk_color = WHITE if model_color == BLACK else BLACK
            expert_players = {ddqk_color}
            if winner != ddqk_color and not include_ddqk_losses:
                skipped_ddqk_losses += 1
                continue

        opening_plies = len(game.get("opening", []))
        board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        last_action = -1
        observed_winner = EMPTY
        for ply_index, raw_move in enumerate(game["moves"]):
            if len(raw_move) != 3:
                raise ValueError(f"game {game_index} ply {ply_index}: malformed move")
            x, y, player = map(int, raw_move)
            if player not in (BLACK, WHITE):
                raise ValueError(
                    f"game {game_index} ply {ply_index}: invalid player {player}"
                )
            expected_player = BLACK if ply_index % 2 == 0 else WHITE
            if player != expected_player:
                raise ValueError(
                    f"game {game_index} ply {ply_index}: expected player "
                    f"{expected_player}, got {player}"
                )
            if not (0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE):
                raise ValueError(
                    f"game {game_index} ply {ply_index}: move outside board {(x, y)}"
                )
            if board[y, x] != EMPTY:
                raise ValueError(
                    f"game {game_index} ply {ply_index}: repeated move {(x, y)}"
                )
            if observed_winner != EMPTY:
                raise ValueError(f"game {game_index}: contains moves after a win")
            if ply_index < opening_plies:
                expected_opening = game["opening"][ply_index]
                if list(map(int, expected_opening)) != [x, y]:
                    raise ValueError(
                        f"game {game_index} ply {ply_index}: opening manifest mismatch"
                    )
            action = y * BOARD_SIZE + x
            if ply_index >= opening_plies and player in expert_players:
                state = encode_state(board, player, last_action)
                policy = policy_target(board, action, smoothing)
                examples = d4_examples(state, policy) if augment else ((state, policy),)
                value = 0.0 if winner == EMPTY else (1.0 if winner == player else -1.0)
                for example_state, example_policy in examples:
                    states.append(example_state)
                    policies.append(example_policy)
                    values.append(value)
                    policy_weights.append(1.0)
                    value_weights.append(0.0 if policy_only else 1.0)
                    if teacher_selfplay:
                        priorities.append(1.0)
                    else:
                        priorities.append(2.0 if winner == ddqk_color else 0.5)
                    game_indices.append(game_index)
                    pair_indices.append(int(game["pair_index"]))
                    ply_indices.append(ply_index)
            board[y, x] = player
            last_action = action
            if is_win(board, x, y, player):
                observed_winner = player

        termination = game["termination"]
        if termination == "win":
            if observed_winner == EMPTY or winner != observed_winner:
                raise ValueError(
                    f"game {game_index}: winner/terminal board does not match"
                )
        elif termination == "full_board_draw":
            if winner != EMPTY or observed_winner != EMPTY or np.any(board == EMPTY):
                raise ValueError(f"game {game_index}: invalid full-board draw")

    if not states:
        raise ValueError("no eligible DDQK expert moves found")
    arrays = {
        "states": np.stack(states).astype(np.uint8, copy=False),
        "policies": np.stack(policies).astype(np.float16, copy=False),
        "values": np.asarray(values, dtype=np.float32),
        "policy_weights": np.asarray(policy_weights, dtype=np.float32),
        "value_weights": np.asarray(value_weights, dtype=np.float32),
        "source": np.full(len(states), 1, dtype=np.uint8),
        "priority": np.asarray(priorities, dtype=np.float32),
        "game_index": np.asarray(game_indices, dtype=np.int32),
        "pair_index": np.asarray(pair_indices, dtype=np.int32),
        "ply_index": np.asarray(ply_indices, dtype=np.int16),
    }
    metadata: dict[str, object] = {
        "schema_version": 1,
        "source": (
            "ddqk_teacher_selfplay_complete_games"
            if teacher_selfplay
            else "ddqk_benchmark_complete_games"
        ),
        "report_type": report.get("report_type", "ddqk_benchmark"),
        "report": str(report_path.resolve()),
        "report_sha256": sha256_file(report_path),
        "samples": len(states),
        "augmented": augment,
        "smoothing": smoothing,
        "include_ddqk_losses": include_ddqk_losses,
        "policy_only": policy_only,
        "allow_partial": bool(allow_partial),
        "skipped_incomplete_games": skipped_incomplete,
        "skipped_ddqk_loss_or_draw_games": skipped_ddqk_losses,
        "ddqk_source_sha256": (
            signature.get("ddqk_source_sha256")
            if isinstance(signature, dict)
            else None
        ),
        "ddqk_dll_sha256": (
            signature.get("ddqk_dll_sha256")
            if isinstance(signature, dict)
            else None
        ),
    }
    if teacher_audit is not None:
        metadata["teacher_collection"] = teacher_audit
    if benchmark_audit is not None:
        metadata["benchmark_collection"] = benchmark_audit
        if "provenance_generation" in benchmark_audit:
            metadata["provenance_generation"] = benchmark_audit[
                "provenance_generation"
            ]
    return arrays, metadata


def write_dataset_bundle(
    output: Path,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, object],
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Write an NPZ plus a SHA-bound sidecar without silent clobbering.

    Each payload is first completed under a process-specific temporary name.
    The sidecar is committed last and contains the exact NPZ digest and size,
    so an interrupted two-file commit is detectable by every consumer.
    """

    output = output.resolve()
    metadata_path = output.with_suffix(output.suffix + ".json")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite and (output.exists() or metadata_path.exists()):
        existing = [str(path) for path in (output, metadata_path) if path.exists()]
        raise FileExistsError(
            "refusing to overwrite existing dataset bundle: " + ", ".join(existing)
        )

    token = f"{os.getpid()}.{os.urandom(6).hex()}"
    temporary_npz = output.with_name(f"{output.name}.{token}.tmp.npz")
    temporary_metadata = metadata_path.with_name(
        f"{metadata_path.name}.{token}.tmp"
    )
    try:
        np.savez_compressed(temporary_npz, **arrays)
        committed_metadata = dict(metadata)
        committed_metadata.update(
            {
                "dataset": str(output),
                "dataset_sha256": sha256_file(temporary_npz),
                "dataset_bytes": temporary_npz.stat().st_size,
            }
        )
        temporary_metadata.write_text(
            json.dumps(committed_metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        # Close the race between the initial no-clobber check and commit.
        if not overwrite and (output.exists() or metadata_path.exists()):
            raise FileExistsError("dataset bundle appeared while export was running")
        os.replace(temporary_npz, output)
        os.replace(temporary_metadata, metadata_path)
        return committed_metadata
    finally:
        temporary_npz.unlink(missing_ok=True)
        temporary_metadata.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoothing", type=float, default=0.05)
    parser.add_argument("--include-ddqk-losses", action="store_true")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help=(
            "allow an in-progress/failed teacher collection; every recorded "
            "game is still strictly audited and only complete games are exported"
        ),
    )
    parser.add_argument(
        "--policy-only",
        action="store_true",
        help="mask value loss (recommended when exporting only DDQK wins)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="explicitly replace an existing NPZ/sidecar bundle",
    )
    args = parser.parse_args()

    arrays, metadata = export_report(
        args.report.resolve(),
        smoothing=args.smoothing,
        include_ddqk_losses=args.include_ddqk_losses,
        augment=args.augment,
        policy_only=args.policy_only,
        allow_partial=args.allow_partial,
    )
    metadata = write_dataset_bundle(
        args.output, arrays, metadata, overwrite=args.overwrite
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"dataset={args.output.resolve()}")


if __name__ == "__main__":
    main()

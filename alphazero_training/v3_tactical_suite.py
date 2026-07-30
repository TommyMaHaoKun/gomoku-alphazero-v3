#!/usr/bin/env python3
"""Small, reproducible multi-ply tactical smoke suite for freestyle Gomoku.

This module deliberately keeps its oracle separate from ``GomokuGame``.  The
oracle never calls ``candidate_actions()``, ``search_actions()``,
``move_heuristic()``, MCTS, or the neural network.  It operates on all legal
points of a minimal immutable board and proves only two modest properties:

* a one-ply win (or the only move which blocks one), and
* a win by the attacker's next turn: attack, any legal reply, win.

The latter is enough to verify open fours and multiple simultaneous fours.  It
is *not* a general VCF/VCT solver and the JSON output calls it a bounded
three-ply oracle accordingly.

Run the self-check without loading a model::

    python v3_tactical_suite.py --self-check-only

Run the desktop agent and write machine-readable results::

    python v3_tactical_suite.py --checkpoint latest.pt --simulations 64 \
        --json-out v3_tactical_result.json

Coordinates in JSON use both a human ``A1`` form and zero-based ``x``/``y``.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Iterable, Sequence

import numpy as np

# Direct execution needs the parent of ``alphazero_training`` on sys.path so
# play_agent's package-relative import continues to work.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from alphazero_training.play_agent import AlphaZeroGomokuAgent
from alphazero_training.train_alphazero import (  # used only by leakage audit
    BLACK,
    EMPTY,
    WHITE,
    GomokuGame,
    evaluate_positions,
)


BOARD_SIZE = 19
WIN_LENGTH = 5
DIRECTIONS = ((1, 0), (0, 1), (1, 1), (1, -1))
SCHEMA_VERSION = 1


def action_of(x: int, y: int) -> int:
    if not (0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE):
        raise ValueError(f"point out of range: ({x}, {y})")
    return y * BOARD_SIZE + x


def point_of(action: int) -> dict[str, int | str]:
    y, x = divmod(int(action), BOARD_SIZE)
    return {"action": int(action), "x": x, "y": y, "coord": f"{chr(65 + x)}{y + 1}"}


@dataclass(frozen=True)
class TacticalBoard:
    """Independent freestyle board used exclusively by the bounded oracle."""

    cells: tuple[int, ...]

    @classmethod
    def from_stones(cls, stones: Iterable[tuple[int, int, int]]) -> "TacticalBoard":
        cells = [EMPTY] * (BOARD_SIZE * BOARD_SIZE)
        for x, y, stone in stones:
            if stone not in (BLACK, WHITE):
                raise ValueError(f"invalid stone {stone} at ({x}, {y})")
            action = action_of(x, y)
            if cells[action] != EMPTY:
                raise ValueError(f"duplicate stone at ({x}, {y})")
            cells[action] = stone
        return cls(tuple(cells))

    def legal_actions(self) -> tuple[int, ...]:
        return tuple(index for index, value in enumerate(self.cells) if value == EMPTY)

    def with_move(self, action: int, player: int) -> "TacticalBoard":
        if player not in (BLACK, WHITE):
            raise ValueError(f"invalid player: {player}")
        if self.cells[action] != EMPTY:
            raise ValueError(f"occupied action: {action}")
        cells = list(self.cells)
        cells[action] = player
        return TacticalBoard(tuple(cells))

    def would_win(self, action: int, player: int) -> bool:
        if self.cells[action] != EMPTY:
            return False
        y, x = divmod(int(action), BOARD_SIZE)
        for dx, dy in DIRECTIONS:
            length = 1
            for sign in (-1, 1):
                nx, ny = x + sign * dx, y + sign * dy
                while (
                    0 <= nx < BOARD_SIZE
                    and 0 <= ny < BOARD_SIZE
                    and self.cells[ny * BOARD_SIZE + nx] == player
                ):
                    length += 1
                    nx += sign * dx
                    ny += sign * dy
            if length >= WIN_LENGTH:
                return True
        return False

    def winning_actions(self, player: int) -> tuple[int, ...]:
        # A winning placement must be collinear with an existing friendly stone
        # no more than four intersections away.  This is an exact reduction,
        # not the trainer's radius-based candidate heuristic.
        return tuple(
            action for action in self.potential_line_actions(player) if self.would_win(action, player)
        )

    def potential_line_actions(self, player: int) -> tuple[int, ...]:
        candidates: set[int] = set()
        for action, stone in enumerate(self.cells):
            if stone != player:
                continue
            y, x = divmod(action, BOARD_SIZE)
            for dx, dy in DIRECTIONS:
                for step in range(-WIN_LENGTH + 1, WIN_LENGTH):
                    if step == 0:
                        continue
                    nx, ny = x + step * dx, y + step * dy
                    if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE:
                        candidate = ny * BOARD_SIZE + nx
                        if self.cells[candidate] == EMPTY:
                            candidates.add(candidate)
        return tuple(sorted(candidates))

    def has_existing_five(self, player: int) -> bool:
        for action, stone in enumerate(self.cells):
            if stone != player:
                continue
            y, x = divmod(action, BOARD_SIZE)
            for dx, dy in DIRECTIONS:
                previous_x, previous_y = x - dx, y - dy
                if (
                    0 <= previous_x < BOARD_SIZE
                    and 0 <= previous_y < BOARD_SIZE
                    and self.cells[previous_y * BOARD_SIZE + previous_x] == player
                ):
                    continue
                length = 0
                nx, ny = x, y
                while (
                    0 <= nx < BOARD_SIZE
                    and 0 <= ny < BOARD_SIZE
                    and self.cells[ny * BOARD_SIZE + nx] == player
                ):
                    length += 1
                    nx += dx
                    ny += dy
                if length >= WIN_LENGTH:
                    return True
        return False


@dataclass(frozen=True)
class TacticalCase:
    case_id: str
    category: str
    side_to_move: int
    stones: tuple[tuple[int, int, int], ...]
    oracle_kind: str
    declared_actions: tuple[int, ...]
    description: str
    max_plies: int

    @property
    def board(self) -> TacticalBoard:
        return TacticalBoard.from_stones(self.stones)


def forced_win_in_three_actions(board: TacticalBoard, attacker: int) -> tuple[int, ...]:
    """Return attacks proven to win on the attacker's immediately following turn.

    A candidate is accepted only if it is not already a one-ply win, gives the
    opponent no immediate win, and creates at least two distinct immediate win
    points.  One reply can occupy at most one of those points, so another win
    remains.  Candidate reduction is exact here: a newly created five must
    contain the newly placed stone.
    """

    proven: list[int] = []
    opponent = -attacker
    for attack in board.potential_line_actions(attacker):
        if board.would_win(attack, attacker):
            continue
        after_attack = board.with_move(attack, attacker)
        threats = after_attack.winning_actions(attacker)
        if len(threats) < 2:
            continue
        if after_attack.winning_actions(opponent):
            continue
        proven.append(attack)
    return tuple(proven)


def defenses_against_three_ply_win(board: TacticalBoard, defender: int) -> tuple[int, ...]:
    """Enumerate moves which remove every opponent win of the bounded kind."""

    opponent = -defender
    root_attacks = forced_win_in_three_actions(board, opponent)
    if not root_attacks:
        return board.legal_actions()

    # A pre-emptive stone can alter an attack/reply/win proof only by occupying
    # its attack point or one of its immediate win points.  Immediate defender
    # wins are also always safe.  Every other empty point leaves at least one
    # original proof unchanged, so this is an exact occupancy reduction.
    relevant: set[int] = set(root_attacks)
    relevant.update(board.winning_actions(defender))
    for attack in root_attacks:
        after_attack = board.with_move(attack, opponent)
        relevant.update(after_attack.winning_actions(opponent))

    safe: list[int] = []
    for defense in sorted(relevant):
        if board.cells[defense] != EMPTY:
            continue
        if board.would_win(defense, defender):
            safe.append(defense)
            continue
        after_defense = board.with_move(defense, defender)
        if after_defense.winning_actions(opponent):
            continue
        if not forced_win_in_three_actions(after_defense, opponent):
            safe.append(defense)
    return tuple(safe)


def oracle_actions(case: TacticalCase) -> tuple[int, ...]:
    board = case.board
    if case.oracle_kind == "immediate_win":
        return board.winning_actions(case.side_to_move)
    if case.oracle_kind == "immediate_block":
        if board.winning_actions(case.side_to_move):
            raise ValueError(f"{case.case_id}: side to move also has a win; block is ambiguous")
        return board.winning_actions(-case.side_to_move)
    if case.oracle_kind == "forced_win_in_3":
        return forced_win_in_three_actions(board, case.side_to_move)
    if case.oracle_kind == "prevent_forced_win_in_3":
        return defenses_against_three_ply_win(board, case.side_to_move)
    raise ValueError(f"unknown oracle kind: {case.oracle_kind}")


def _actions(*points: tuple[int, int]) -> tuple[int, ...]:
    return tuple(sorted(action_of(x, y) for x, y in points))


def _line(
    player: int,
    points: Iterable[tuple[int, int]],
) -> tuple[tuple[int, int, int], ...]:
    return tuple((x, y, player) for x, y in points)


def built_in_cases() -> tuple[TacticalCase, ...]:
    """Return a fixed smoke suite; positions intentionally stay small and legible."""

    cases: list[TacticalCase] = []

    cases.append(
        TacticalCase(
            "immediate_attack_black",
            "immediate_attack",
            BLACK,
            _line(BLACK, ((6, 9), (7, 9), (8, 9), (9, 9))) + ((5, 9, WHITE),),
            "immediate_win",
            _actions((10, 9)),
            "Black has one legal winning endpoint.",
            1,
        )
    )
    cases.append(
        TacticalCase(
            "immediate_attack_white",
            "immediate_attack",
            WHITE,
            _line(WHITE, ((9, 6), (9, 7), (9, 8), (9, 9))) + ((9, 5, BLACK),),
            "immediate_win",
            _actions((9, 10)),
            "White has one legal winning endpoint.",
            1,
        )
    )
    cases.append(
        TacticalCase(
            "immediate_defense_black",
            "immediate_defense",
            BLACK,
            _line(WHITE, ((6, 6), (7, 7), (8, 8), (9, 9))) + ((5, 5, BLACK),),
            "immediate_block",
            _actions((10, 10)),
            "Black must block White's only immediate win.",
            1,
        )
    )
    cases.append(
        TacticalCase(
            "immediate_defense_white",
            "immediate_defense",
            WHITE,
            _line(BLACK, ((6, 12), (7, 11), (8, 10), (9, 9))) + ((5, 13, WHITE),),
            "immediate_block",
            _actions((10, 8)),
            "White must block Black's only immediate win.",
            1,
        )
    )

    cases.append(
        TacticalCase(
            "open_four_black",
            "open_four",
            BLACK,
            _line(BLACK, ((7, 9), (8, 9), (9, 9))) + ((3, 3, WHITE), (14, 14, WHITE)),
            "forced_win_in_3",
            _actions((6, 9), (10, 9)),
            "Either extension creates a two-ended open four.",
            3,
        )
    )
    cases.append(
        TacticalCase(
            "open_four_white",
            "open_four",
            WHITE,
            _line(WHITE, ((7, 7), (8, 8), (9, 9))) + ((3, 14, BLACK), (14, 3, BLACK)),
            "forced_win_in_3",
            _actions((6, 6), (10, 10)),
            "Either diagonal extension creates a two-ended open four.",
            3,
        )
    )

    cases.append(
        TacticalCase(
            "double_closed_four_black",
            "forcing_four",
            BLACK,
            _line(BLACK, ((6, 9), (7, 9), (8, 9), (9, 6), (9, 7), (9, 8)))
            + _line(WHITE, ((5, 9), (9, 5))),
            "forced_win_in_3",
            _actions((9, 9)),
            "The pivot creates two one-ended fours (a double forcing-four threat).",
            3,
        )
    )
    cases.append(
        TacticalCase(
            "double_closed_four_white",
            "double_threat",
            WHITE,
            _line(WHITE, ((6, 6), (7, 7), (8, 8), (6, 12), (7, 11), (8, 10)))
            + _line(BLACK, ((5, 5), (5, 13))),
            "forced_win_in_3",
            _actions((9, 9)),
            "The center creates two independent diagonal forcing fours.",
            3,
        )
    )

    white_triple = (
        _line(WHITE, ((6, 9), (7, 9), (8, 9), (9, 10), (9, 11), (9, 12)))
        + _line(WHITE, ((6, 6), (7, 7), (8, 8)))
        + _line(BLACK, ((5, 9), (9, 13), (5, 5)))
    )
    cases.append(
        TacticalCase(
            "unique_defense_black",
            "unique_defense",
            BLACK,
            white_triple,
            "prevent_forced_win_in_3",
            _actions((9, 9)),
            "Only occupying the common pivot prevents White's triple forcing-four fork.",
            4,
        )
    )

    black_triple = (
        _line(BLACK, ((6, 9), (7, 9), (8, 9), (9, 10), (9, 11), (9, 12)))
        + _line(BLACK, ((6, 6), (7, 7), (8, 8)))
        + _line(WHITE, ((5, 9), (9, 13), (5, 5)))
    )
    cases.append(
        TacticalCase(
            "unique_defense_white",
            "unique_defense",
            WHITE,
            black_triple,
            "prevent_forced_win_in_3",
            _actions((9, 9)),
            "Only occupying the common pivot prevents Black's triple forcing-four fork.",
            4,
        )
    )
    return tuple(cases)


def _as_gomoku_game(case: TacticalCase) -> GomokuGame:
    game = GomokuGame(BOARD_SIZE, WIN_LENGTH)
    for x, y, stone in case.stones:
        game.board[y, x] = stone
    game.move_count = len(case.stones)
    game.player = case.side_to_move
    return game


def root_leakage_audit(case: TacticalCase) -> dict[str, object]:
    """Show whether the old root ``search_actions`` can directly reveal an answer."""

    game = _as_gomoku_game(case)
    own_wins = set(map(int, game.winning_actions(case.side_to_move, radius=BOARD_SIZE)))
    opponent_wins = set(map(int, game.winning_actions(-case.side_to_move, radius=BOARD_SIZE)))
    searched = set(map(int, game.search_actions(radius=2)))
    candidates = set(map(int, game.candidate_actions(radius=2)))
    action_filter_active = searched != candidates
    direct_answer_eligible = case.oracle_kind in ("immediate_win", "immediate_block")
    return {
        "root_own_immediate_wins": [point_of(action) for action in sorted(own_wins)],
        "root_opponent_immediate_wins": [point_of(action) for action in sorted(opponent_wins)],
        "search_actions_filter_active": action_filter_active,
        "direct_answer_eligible": direct_answer_eligible,
        "attribution": (
            "one_ply_safety_wrapper_eligible"
            if direct_answer_eligible
            else "root_requires_policy_or_search"
        ),
    }


def self_check(cases: Sequence[TacticalCase]) -> dict[str, object]:
    errors: list[str] = []
    details: list[dict[str, object]] = []
    ids: set[str] = set()
    for case in cases:
        if case.case_id in ids:
            errors.append(f"duplicate case id: {case.case_id}")
        ids.add(case.case_id)
        board = case.board
        if board.has_existing_five(BLACK) or board.has_existing_five(WHITE):
            errors.append(f"{case.case_id}: input board is already terminal")
        calculated = tuple(sorted(oracle_actions(case)))
        declared = tuple(sorted(case.declared_actions))
        if calculated != declared:
            errors.append(
                f"{case.case_id}: oracle {list(map(point_of, calculated))} "
                f"!= declared {list(map(point_of, declared))}"
            )
        audit = root_leakage_audit(case)
        if case.oracle_kind not in ("immediate_win", "immediate_block"):
            if audit["root_own_immediate_wins"] or audit["root_opponent_immediate_wins"]:
                errors.append(f"{case.case_id}: multi-ply root contains a one-ply win/block")
            if audit["search_actions_filter_active"]:
                errors.append(f"{case.case_id}: search_actions directly filters the root")
        details.append(
            {
                "case_id": case.case_id,
                "oracle_matches_declared": calculated == declared,
                "expected_actions": [point_of(action) for action in calculated],
                "root_leakage_audit": audit,
            }
        )
    required = {
        "immediate_attack",
        "immediate_defense",
        "open_four",
        "forcing_four",
        "double_threat",
        "unique_defense",
    }
    categories = {case.category for case in cases}
    missing = sorted(required - categories)
    if missing:
        errors.append(f"missing required categories: {missing}")
    return {
        "passed": not errors,
        "case_count": len(cases),
        "categories": sorted(categories),
        "errors": errors,
        "cases": details,
    }


def _grid_for_agent(case: TacticalCase) -> list[list[int]]:
    grid = [[0 for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    for x, y, stone in case.stones:
        grid[y][x] = 1 if stone == BLACK else 2
    return grid


def evaluate_agent(
    agent: AlphaZeroGomokuAgent,
    cases: Sequence[TacticalCase],
) -> dict[str, object]:
    case_results: list[dict[str, object]] = []
    by_category: dict[str, dict[str, int]] = defaultdict(lambda: {"passed": 0, "total": 0})
    by_capability: dict[str, dict[str, int]] = defaultdict(lambda: {"passed": 0, "total": 0})
    raw_policy = {
        "all": {"passed": 0, "total": 0},
        "multi_ply": {"passed": 0, "total": 0},
    }

    for case in cases:
        expected = set(oracle_actions(case))
        policy_game = _as_gomoku_game(case)
        raw_logits, _ = evaluate_positions(agent.model, [policy_game], agent.device)
        legal = policy_game.legal_actions()
        order = legal[np.argsort(-raw_logits[0, legal], kind="stable")]
        raw_choice = int(order[0])
        raw_rank = min(
            int(np.flatnonzero(order == expected_action)[0]) + 1
            for expected_action in expected
        )
        ai_color = 1 if case.side_to_move == BLACK else 2
        move = agent.choose_move(_grid_for_agent(case), last_move=None, ai_color=ai_color)
        decision_reason = agent.last_decision_reason
        chosen = None if move is None else action_of(move[0], move[1])
        passed = chosen in expected
        capability = (
            "one_ply_safety_wrapper"
            if case.oracle_kind in ("immediate_win", "immediate_block")
            else "multi_ply_root_choice"
        )
        by_category[case.category]["total"] += 1
        by_capability[capability]["total"] += 1
        raw_policy["all"]["total"] += 1
        raw_policy["all"]["passed"] += int(raw_choice in expected)
        if capability == "multi_ply_root_choice":
            raw_policy["multi_ply"]["total"] += 1
            raw_policy["multi_ply"]["passed"] += int(raw_choice in expected)
        if passed:
            by_category[case.category]["passed"] += 1
            by_capability[capability]["passed"] += 1
        case_results.append(
            {
                "case_id": case.case_id,
                "category": case.category,
                "description": case.description,
                "side_to_move": "black" if case.side_to_move == BLACK else "white",
                "oracle": {
                    "kind": case.oracle_kind,
                    "max_plies": case.max_plies,
                    "expected_actions": [point_of(action) for action in sorted(expected)],
                },
                "agent_capability_bucket": capability,
                "agent_decision_reason": decision_reason,
                "raw_network_policy": {
                    "top_action": point_of(raw_choice),
                    "best_expected_rank": raw_rank,
                    "top1_passed": raw_choice in expected,
                },
                "chosen_action": None if chosen is None else point_of(chosen),
                "passed": passed,
                "root_leakage_audit": root_leakage_audit(case),
            }
        )

    total = len(case_results)
    passed = sum(int(result["passed"]) for result in case_results)
    return {
        "summary": {
            "passed": passed,
            "total": total,
            "score": passed / total if total else 0.0,
            "by_category": dict(sorted(by_category.items())),
            "by_capability": dict(sorted(by_capability.items())),
            "raw_network_policy": raw_policy,
        },
        "cases": case_results,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="AlphaZero checkpoint for agent run")
    parser.add_argument("--simulations", type=int, default=64, help="MCTS simulations per case")
    parser.add_argument("--json-out", type=Path, help="also write the JSON report to this path")
    parser.add_argument(
        "--self-check-only",
        action="store_true",
        help="validate the cases and bounded oracle without loading an agent",
    )
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="exit non-zero if the evaluated agent misses any case",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cases = built_in_cases()
    checks = self_check(cases)
    report: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "suite": "v3_freestyle_gomoku_tactical_smoke",
        "rules": {"board_size": BOARD_SIZE, "win_length": WIN_LENGTH, "overline_wins": True},
        "reproducibility": {
            "fixed_case_order": [case.case_id for case in cases],
            "random_openings": False,
            "oracle_uses_search_actions": False,
        },
        "scope_note": (
            "Small smoke suite only. Multi-ply cases prove at most attack/reply/win; "
            "this is not a complete VCF/VCT benchmark."
        ),
        "self_check": checks,
    }

    exit_code = 0
    if not checks["passed"]:
        exit_code = 2
    elif not args.self_check_only:
        if args.checkpoint is None:
            raise SystemExit("--checkpoint is required unless --self-check-only is used")
        checkpoint = args.checkpoint.resolve()
        agent = AlphaZeroGomokuAgent(checkpoint, simulations=args.simulations)
        report["agent"] = {
            "label": agent.label,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "simulations": args.simulations,
            "important_attribution_note": (
                "agent_decision_reason records whether the independent tactical solver or "
                "MCTS selected each action. Root leakage audit remains separate."
            ),
        }
        evaluation = evaluate_agent(agent, cases)
        report["evaluation"] = evaluation
        if args.require_all and evaluation["summary"]["passed"] != evaluation["summary"]["total"]:
            exit_code = 1

    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(payload)
    if args.json_out is not None:
        destination = args.json_out.resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(payload + "\n", encoding="utf-8")
        temporary.replace(destination)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

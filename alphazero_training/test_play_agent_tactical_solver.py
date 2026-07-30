#!/usr/bin/env python3
"""Regression checks for exact short-tactics routing in the desktop agent."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alphazero_training.play_agent import AlphaZeroGomokuAgent
from alphazero_training.train_alphazero import BLACK
from alphazero_training.v3_tactical_suite import built_in_cases, oracle_actions


EXPECTED_REASONS = {
    "immediate_attack": "immediate_win",
    "immediate_defense": "immediate_block",
    "open_four": "win_in_3",
    "forcing_four": "win_in_3",
    "double_threat": "win_in_3",
    "unique_defense": "block_win_in_3",
}


def main() -> int:
    checkpoint = Path(__file__).resolve().parent / "latest.pt"
    agent = AlphaZeroGomokuAgent(checkpoint, simulations=1)
    checked = 0

    for case in built_in_cases():
        grid = [[0 for _ in range(19)] for _ in range(19)]
        for x, y, stone in case.stones:
            grid[y][x] = 1 if stone == BLACK else 2

        ai_color = 1 if case.side_to_move == BLACK else 2
        move = agent.choose_move(grid, ai_color=ai_color)
        if move is None:
            raise AssertionError(f"{case.case_id}: agent returned no move")
        chosen = move[1] * 19 + move[0]
        expected_actions = set(oracle_actions(case))
        if chosen not in expected_actions:
            raise AssertionError(
                f"{case.case_id}: chose {move}, expected one of {sorted(expected_actions)}"
            )

        expected_reason = EXPECTED_REASONS[case.category]
        if agent.last_decision_reason != expected_reason:
            raise AssertionError(
                f"{case.case_id}: reason {agent.last_decision_reason!r}, "
                f"expected {expected_reason!r}"
            )
        checked += 1

    print(f"play-agent tactical solver verified: {checked}/10 decisions and reasons passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

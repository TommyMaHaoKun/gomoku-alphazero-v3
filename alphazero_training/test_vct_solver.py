from __future__ import annotations

import unittest

from alphazero_training.tactical_solver import (
    BLACK,
    EMPTY,
    WHITE,
    FreestyleBoard,
    SolveStatus,
    ThreatSolveLimits,
    solve_vct,
)


SIZE = 19


def action(x: int, y: int) -> int:
    return y * SIZE + x


def transform_xy(x: int, y: int, symmetry: int) -> tuple[int, int]:
    n = SIZE - 1
    transforms = (
        (x, y), (n - y, x), (n - x, n - y), (y, n - x),
        (n - x, y), (x, n - y), (y, x), (n - y, n - x),
    )
    return transforms[symmetry]


def transformed(board: FreestyleBoard, symmetry: int, swap: bool) -> FreestyleBoard:
    stones: list[tuple[int, int, int]] = []
    for source, stone in enumerate(board.cells):
        if stone == EMPTY:
            continue
        y, x = divmod(source, SIZE)
        tx, ty = transform_xy(x, y, symmetry)
        stones.append((tx, ty, -stone if swap else stone))
    return FreestyleBoard.from_stones(stones, size=SIZE)


def transformed_action(source: int, symmetry: int) -> int:
    y, x = divmod(source, SIZE)
    tx, ty = transform_xy(x, y, symmetry)
    return action(tx, ty)


class ConservativeVCTTests(unittest.TestCase):
    def limits(self, **overrides: object) -> ThreatSolveLimits:
        values: dict[str, object] = {
            "max_plies": 9,
            "max_nodes": 50_000,
            "time_ms": 2_000.0,
            "max_attack_candidates": 4,
            "max_defenses": 64,
        }
        values.update(overrides)
        return ThreatSolveLimits(**values)

    def test_open_three_is_a_proven_three_ply_win(self) -> None:
        board = FreestyleBoard.from_stones(
            ((7, 9, BLACK), (8, 9, BLACK), (9, 9, BLACK)), size=SIZE
        )
        result = solve_vct(board, BLACK, self.limits(max_plies=3))
        self.assertEqual(result.status, SolveStatus.PROVEN_WIN)
        self.assertEqual(len(result.principal_variation), 3)

    def test_double_three_is_proven_and_pv_replays_to_five(self) -> None:
        board = FreestyleBoard.from_stones(
            (
                (8, 9, BLACK), (10, 9, BLACK),
                (9, 8, BLACK), (9, 10, BLACK),
            ),
            size=SIZE,
        )
        result = solve_vct(
            board,
            BLACK,
            self.limits(max_attack_candidates=1),
        )
        self.assertEqual(result.status, SolveStatus.PROVEN_WIN)
        self.assertEqual(result.winning_actions, (action(9, 9),))
        self.assertEqual(len(result.principal_variation), 5)
        replay = board
        side = BLACK
        for move in result.principal_variation:
            replay = replay.with_move(move, side)
            side = -side
        self.assertTrue(replay.has_five(BLACK))

    def test_four_three_and_continuous_threats_are_proven(self) -> None:
        four_three = FreestyleBoard.from_stones(
            (
                (7, 9, BLACK), (8, 9, BLACK), (9, 9, BLACK),
                (10, 8, BLACK), (10, 10, BLACK),
            ),
            size=SIZE,
        )
        result = solve_vct(four_three, BLACK, self.limits())
        self.assertEqual(result.status, SolveStatus.PROVEN_WIN)
        self.assertIn(action(10, 9), result.winning_actions)

        chain_stones = (
            tuple((x, 8, BLACK) for x in (6, 7, 8))
            + tuple((x, 9, BLACK) for x in (6, 7, 8))
            + tuple((9, y, BLACK) for y in (6, 7))
            + (
                (5, 8, WHITE), (5, 9, WHITE), (9, 5, WHITE),
                (10, 6, WHITE), (10, 5, WHITE),
            )
        )
        chain = FreestyleBoard.from_stones(chain_stones, size=SIZE)
        chained = solve_vct(chain, BLACK, self.limits())
        self.assertEqual(chained.status, SolveStatus.PROVEN_WIN)
        self.assertGreaterEqual(len(chained.principal_variation), 5)

    def test_node_and_candidate_cutoffs_are_unknown_not_losses(self) -> None:
        double_three = FreestyleBoard.from_stones(
            ((8, 9, BLACK), (10, 9, BLACK), (9, 8, BLACK), (9, 10, BLACK)),
            size=SIZE,
        )
        exhausted = solve_vct(
            double_three,
            BLACK,
            self.limits(max_nodes=1, max_attack_candidates=1),
        )
        self.assertEqual(exhausted.status, SolveStatus.UNKNOWN_BUDGET)

        # A single stone has many line-relevant setup points but no checked
        # forcing move.  Capping that incomplete set must not return NO_VCT.
        quiet = FreestyleBoard.from_stones(((9, 9, BLACK),), size=SIZE)
        capped = solve_vct(
            quiet,
            BLACK,
            self.limits(max_attack_candidates=1),
        )
        self.assertEqual(capped.status, SolveStatus.UNKNOWN_BUDGET)

    def test_wall_clock_cutoff_is_unknown_not_a_loss(self) -> None:
        board = FreestyleBoard.from_stones(
            ((8, 9, BLACK), (10, 9, BLACK), (9, 8, BLACK), (9, 10, BLACK)),
            size=SIZE,
        )
        timed_out = solve_vct(
            board,
            BLACK,
            self.limits(time_ms=0.0001, max_attack_candidates=1),
        )
        self.assertEqual(timed_out.status, SolveStatus.UNKNOWN_BUDGET)
        self.assertEqual(timed_out.winning_actions, ())

    def test_rotation_and_color_symmetry(self) -> None:
        board = FreestyleBoard.from_stones(
            ((8, 9, BLACK), (10, 9, BLACK), (9, 8, BLACK), (9, 10, BLACK)),
            size=SIZE,
        )
        root = action(9, 9)
        for symmetry in range(8):
            for swap in (False, True):
                with self.subTest(symmetry=symmetry, swap=swap):
                    position = transformed(board, symmetry, swap)
                    side = WHITE if swap else BLACK
                    result = solve_vct(
                        position,
                        side,
                        self.limits(max_attack_candidates=1),
                    )
                    self.assertEqual(result.status, SolveStatus.PROVEN_WIN)
                    self.assertEqual(
                        result.winning_actions,
                        (transformed_action(root, symmetry),),
                    )


if __name__ == "__main__":
    unittest.main()

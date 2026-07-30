from __future__ import annotations

import unittest

try:
    from .tactical_solver import (
        BLACK,
        EMPTY,
        WHITE,
        FreestyleBoard,
        SolveLimits,
        SolveStatus,
        defenses_against_forced_win_in_three,
        forced_win_in_three_actions,
        immediate_winning_actions,
        solve_vcf,
    )
    from .v3_tactical_suite import built_in_cases, oracle_actions
except ImportError:  # pragma: no cover - supports direct execution
    from tactical_solver import (
        BLACK,
        EMPTY,
        WHITE,
        FreestyleBoard,
        SolveLimits,
        SolveStatus,
        defenses_against_forced_win_in_three,
        forced_win_in_three_actions,
        immediate_winning_actions,
        solve_vcf,
    )
    from v3_tactical_suite import built_in_cases, oracle_actions


SIZE = 19


def action(x: int, y: int) -> int:
    return y * SIZE + x


def board_from_stones(stones: tuple[tuple[int, int, int], ...]) -> FreestyleBoard:
    return FreestyleBoard.from_stones(stones, size=SIZE)


def transform_xy(x: int, y: int, symmetry: int) -> tuple[int, int]:
    """Eight D4 transforms on a square board."""

    n = SIZE - 1
    if symmetry == 0:
        return x, y
    if symmetry == 1:
        return n - y, x
    if symmetry == 2:
        return n - x, n - y
    if symmetry == 3:
        return y, n - x
    if symmetry == 4:
        return n - x, y
    if symmetry == 5:
        return x, n - y
    if symmetry == 6:
        return y, x
    if symmetry == 7:
        return n - y, n - x
    raise ValueError(symmetry)


def transform_board(board: FreestyleBoard, symmetry: int, color_swap: bool = False) -> FreestyleBoard:
    stones: list[tuple[int, int, int]] = []
    for source_action, stone in enumerate(board.cells):
        if stone == EMPTY:
            continue
        y, x = divmod(source_action, SIZE)
        tx, ty = transform_xy(x, y, symmetry)
        stones.append((tx, ty, -stone if color_swap else stone))
    return FreestyleBoard.from_stones(stones, size=SIZE)


def transform_actions(actions: tuple[int, ...], symmetry: int) -> tuple[int, ...]:
    transformed = []
    for source_action in actions:
        y, x = divmod(source_action, SIZE)
        tx, ty = transform_xy(x, y, symmetry)
        transformed.append(action(tx, ty))
    return tuple(sorted(transformed))


class ExactShortTacticsTests(unittest.TestCase):
    def test_reproduces_all_ten_current_oracles(self) -> None:
        cases = built_in_cases()
        self.assertEqual(len(cases), 10)
        for case in cases:
            board = board_from_stones(case.stones)
            expected = tuple(sorted(oracle_actions(case)))
            with self.subTest(case=case.case_id):
                if case.oracle_kind == "immediate_win":
                    actual = immediate_winning_actions(board, case.side_to_move)
                elif case.oracle_kind == "immediate_block":
                    actual = immediate_winning_actions(board, -case.side_to_move)
                elif case.oracle_kind == "forced_win_in_3":
                    actual = forced_win_in_three_actions(board, case.side_to_move)
                elif case.oracle_kind == "prevent_forced_win_in_3":
                    actual = defenses_against_forced_win_in_three(board, case.side_to_move)
                else:  # pragma: no cover
                    self.fail(f"unhandled oracle: {case.oracle_kind}")
                self.assertEqual(tuple(sorted(actual)), expected)

    def test_overline_is_an_immediate_win(self) -> None:
        # Filling the gap makes six contiguous stones, which wins in freestyle.
        board = FreestyleBoard.from_stones(
            tuple((x, 9, BLACK) for x in (5, 6, 7, 9, 10)), size=SIZE
        )
        self.assertIn(action(8, 9), immediate_winning_actions(board, BLACK))

    def test_d4_and_color_symmetry_for_forced_win_in_three(self) -> None:
        source = next(case for case in built_in_cases() if case.case_id == "double_closed_four_black")
        board = board_from_stones(source.stones)
        expected = forced_win_in_three_actions(board, BLACK)
        self.assertTrue(expected)
        for symmetry in range(8):
            for color_swap in (False, True):
                with self.subTest(symmetry=symmetry, color_swap=color_swap):
                    transformed = transform_board(board, symmetry, color_swap)
                    player = WHITE if color_swap else BLACK
                    actual = forced_win_in_three_actions(transformed, player)
                    self.assertEqual(actual, transform_actions(expected, symmetry))

    def test_d4_and_color_symmetry_for_exact_defense(self) -> None:
        source = next(case for case in built_in_cases() if case.case_id == "unique_defense_black")
        board = board_from_stones(source.stones)
        expected = defenses_against_forced_win_in_three(board, BLACK)
        self.assertEqual(expected, (action(9, 9),))
        for symmetry in range(8):
            for color_swap in (False, True):
                with self.subTest(symmetry=symmetry, color_swap=color_swap):
                    transformed = transform_board(board, symmetry, color_swap)
                    defender = WHITE if color_swap else BLACK
                    actual = defenses_against_forced_win_in_three(transformed, defender)
                    self.assertEqual(actual, transform_actions(expected, symmetry))


class BoundedVCFTests(unittest.TestCase):
    def setUp(self) -> None:
        source = next(case for case in built_in_cases() if case.case_id == "open_four_black")
        self.board = board_from_stones(source.stones)
        self.expected = tuple(sorted(source.declared_actions))

    def test_vcf_proves_all_three_ply_roots_and_pv(self) -> None:
        result = solve_vcf(
            self.board,
            BLACK,
            SolveLimits(max_plies=3, max_nodes=10_000, time_ms=2_000),
        )
        self.assertEqual(result.status, SolveStatus.PROVEN_WIN)
        self.assertEqual(tuple(sorted(result.winning_actions)), self.expected)
        self.assertEqual(len(result.principal_variation), 3)
        self.assertIn(result.principal_variation[0], self.expected)
        self.assertEqual(len(result.required_defenses), 2)
        self.assertGreater(result.nodes, 0)

    def test_too_shallow_is_proven_no_vcf_within_bound(self) -> None:
        result = solve_vcf(
            self.board,
            BLACK,
            SolveLimits(max_plies=2, max_nodes=10_000, time_ms=2_000),
        )
        self.assertEqual(result.status, SolveStatus.PROVEN_NO_VCF)
        self.assertEqual(result.winning_actions, ())

    def test_node_cutoff_is_unknown_not_a_loss(self) -> None:
        result = solve_vcf(
            self.board,
            BLACK,
            SolveLimits(max_plies=7, max_nodes=1, time_ms=2_000),
        )
        self.assertEqual(result.status, SolveStatus.UNKNOWN_BUDGET)
        self.assertEqual(result.nodes, 1)

    def test_empty_board_has_no_bounded_vcf(self) -> None:
        board = FreestyleBoard((EMPTY,) * (SIZE * SIZE), SIZE, 5)
        result = solve_vcf(
            board,
            BLACK,
            SolveLimits(max_plies=9, max_nodes=100, time_ms=2_000),
        )
        self.assertEqual(result.status, SolveStatus.PROVEN_NO_VCF)

    def test_dfs_finds_a_five_ply_vcf_not_visible_at_three_plies(self) -> None:
        # The added white stones suppress every direct double-four.  Black must
        # first make a single four, accept its forced block, then fork.
        stones = (
            tuple((x, 8, BLACK) for x in (6, 7, 8))
            + tuple((x, 9, BLACK) for x in (6, 7, 8))
            + tuple((9, y, BLACK) for y in (6, 7))
            + ((5, 8, WHITE), (5, 9, WHITE), (9, 5, WHITE), (10, 6, WHITE), (10, 5, WHITE))
        )
        board = FreestyleBoard.from_stones(stones, size=SIZE)
        self.assertEqual(forced_win_in_three_actions(board, BLACK), ())
        shallow = solve_vcf(
            board,
            BLACK,
            SolveLimits(max_plies=4, max_nodes=20_000, time_ms=2_000),
        )
        deep = solve_vcf(
            board,
            BLACK,
            SolveLimits(max_plies=5, max_nodes=20_000, time_ms=2_000),
        )
        self.assertEqual(shallow.status, SolveStatus.PROVEN_NO_VCF)
        self.assertEqual(deep.status, SolveStatus.PROVEN_WIN)
        self.assertEqual(len(deep.principal_variation), 5)

    def test_vcf_color_and_rotation_symmetry(self) -> None:
        baseline = solve_vcf(
            self.board,
            BLACK,
            SolveLimits(max_plies=3, max_nodes=10_000, time_ms=2_000),
        )
        for symmetry in range(8):
            transformed = transform_board(self.board, symmetry, color_swap=True)
            result = solve_vcf(
                transformed,
                WHITE,
                SolveLimits(max_plies=3, max_nodes=10_000, time_ms=2_000),
            )
            with self.subTest(symmetry=symmetry):
                self.assertEqual(result.status, SolveStatus.PROVEN_WIN)
                self.assertEqual(
                    tuple(sorted(result.winning_actions)),
                    transform_actions(baseline.winning_actions, symmetry),
                )


if __name__ == "__main__":
    unittest.main()

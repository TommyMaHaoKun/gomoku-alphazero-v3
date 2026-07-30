"""Minimal local smoke tests for the headless DDQK adapter."""

from __future__ import annotations

import unittest

try:
    from .ddqk_adapter import BLACK, BOARD_SIZE, DDQKAdapter, EMPTY, WHITE
except ImportError:  # Allow direct execution from this directory.
    from ddqk_adapter import BLACK, BOARD_SIZE, DDQKAdapter, EMPTY, WHITE


class DDQKAdapterSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.adapter = DDQKAdapter()

    def setUp(self) -> None:
        self.adapter.reset()

    def test_empty_board_move_is_legal_and_reset_is_deterministic(self) -> None:
        board = [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        first = self.adapter.choose_move(board, BLACK)
        self.assertEqual(first, (9, 9))
        self.assertEqual(board[9][9], EMPTY, "adapter must not mutate caller grid")

        self.adapter.reset()
        second = self.adapter.choose_move(board, BLACK)
        self.assertEqual(second, first)
        self.assertIs(self.adapter, DDQKAdapter(), "adapter must be a singleton")

    def test_white_reply_accepts_alphazero_minus_one_colour(self) -> None:
        board = [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        board[9][9] = BLACK
        move = self.adapter.choose_move(board, -1, last_move=(9, 9))
        x, y = move
        self.assertTrue(0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE)
        self.assertEqual(board[y][x], EMPTY)

    def test_ordered_opening_sync_produces_a_legal_reply(self) -> None:
        board = self.adapter.sync_opening([(9, 9), (9, 8)])
        self.assertEqual(board[9][9], BLACK)
        self.assertEqual(board[8][9], WHITE)
        move = self.adapter.choose_move(None, BLACK, last_move=(9, 8))
        x, y = move
        self.assertEqual(board[y][x], EMPTY)


if __name__ == "__main__":
    unittest.main(verbosity=2)

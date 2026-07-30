from __future__ import annotations

import unittest

import numpy as np
import torch

from alphazero_training.tactical_solver import SolveStatus, ThreatSolveResult
from alphazero_training.train_alphazero import BLACK, Config, GomokuGame
from alphazero_training.v3_search import V3RootSearch, VCTRootOptions


SIZE = 19
COUNT = SIZE * SIZE


class _Model(torch.nn.Module):
    def __init__(self, scores: dict[int, float]) -> None:
        super().__init__()
        self.scores = scores

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = torch.full((states.shape[0], COUNT), -40.0, device=states.device)
        for action, score in self.scores.items():
            logits[:, action] = score
        return logits, torch.zeros(states.shape[0], device=states.device)


def result(
    status: SolveStatus,
    winning_actions: tuple[int, ...] = (),
) -> ThreatSolveResult:
    return ThreatSolveResult(status, winning_actions, (), 1, 0, 0.1)


class VCTRootGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.occupied = 9 * SIZE + 9
        self.unsafe = 9 * SIZE + 8
        self.safe = 9 * SIZE + 10
        self.model = _Model({self.unsafe: 8.0, self.safe: 7.0})
        self.config = Config(
            simulations=0,
            candidate_radius=2,
            heuristic_prior_weight=0.0,
            vcf_root_filter=False,
        )
        self.options = VCTRootOptions(
            enabled=True,
            attack_priority=False,
            root_candidates=2,
            minimum_probability=0.0,
        )

    def game(self, player: int = BLACK) -> GomokuGame:
        game = GomokuGame(SIZE, 5)
        game.board.ravel()[self.occupied] = -player
        game.player = player
        game.move_count = 1
        return game

    def test_default_is_disabled_and_has_zero_solver_calls(self) -> None:
        search = V3RootSearch(self.model, self.config, "cpu")

        def forbidden(*_args: object, **_kwargs: object) -> ThreatSolveResult:
            raise AssertionError("disabled VCT must not run")

        search.tactical_solver.solve_vct = forbidden  # type: ignore[method-assign]
        decision = search.decide(self.game(), simulations=0)
        self.assertEqual(decision.reason, "mcts")

    def test_only_proven_win_is_filtered_unknown_is_retained(self) -> None:
        for unsafe_status, expected_reason, expected_mass in (
            (SolveStatus.PROVEN_WIN, "mcts_vct_safe", 0.0),
            (SolveStatus.UNKNOWN_BUDGET, "mcts", None),
            (SolveStatus.PROVEN_NO_VCT, "mcts", None),
        ):
            with self.subTest(status=unsafe_status):
                search = V3RootSearch(
                    self.model, self.config, "cpu", vct_options=self.options
                )

                def fake(board: np.ndarray, _side: int, _limits: object) -> ThreatSolveResult:
                    status = (
                        unsafe_status
                        if int(board.ravel()[self.unsafe]) != 0
                        else SolveStatus.PROVEN_NO_VCT
                    )
                    return result(status)

                search.tactical_solver.solve_vct = fake  # type: ignore[method-assign]
                decision = search.decide(self.game(), simulations=0)
                self.assertEqual(decision.reason, expected_reason)
                if expected_mass is None:
                    self.assertGreater(float(decision.policy[self.unsafe]), 0.0)
                else:
                    self.assertEqual(float(decision.policy[self.unsafe]), expected_mass)
                self.assertAlmostEqual(float(decision.policy.sum()), 1.0, places=6)

    def test_all_proven_unsafe_falls_back_instead_of_claiming_safety(self) -> None:
        baseline = V3RootSearch(self.model, self.config, "cpu").decide(
            self.game(), simulations=0
        )
        search = V3RootSearch(
            self.model, self.config, "cpu", vct_options=self.options
        )
        search.tactical_solver.solve_vct = (  # type: ignore[method-assign]
            lambda *_args: result(SolveStatus.PROVEN_WIN)
        )
        decision = search.decide(self.game(), simulations=0)
        self.assertEqual(decision.reason, "mcts")
        np.testing.assert_array_equal(decision.policy, baseline.policy)

    def test_attack_priority_requires_proof_and_no_immediate_counterkill(self) -> None:
        options = VCTRootOptions(
            enabled=True,
            attack_priority=True,
            root_candidates=2,
            minimum_probability=0.0,
        )
        game = self.game()
        attack = self.safe
        search = V3RootSearch(self.model, self.config, "cpu", vct_options=options)

        def fake(board: np.ndarray, side: int, _limits: object) -> ThreatSolveResult:
            if int(np.count_nonzero(board)) == 1 and side == game.player:
                return result(SolveStatus.PROVEN_WIN, (attack,))
            return result(SolveStatus.PROVEN_NO_VCT)

        search.tactical_solver.solve_vct = fake  # type: ignore[method-assign]
        decision = search.decide(game, simulations=0)
        self.assertEqual(decision.reason, "mcts_vct_attack")
        self.assertTrue(decision.proven)
        self.assertEqual(decision.action, attack)

        blocked = V3RootSearch(self.model, self.config, "cpu", vct_options=options)
        blocked.tactical_solver.solve_vct = fake  # type: ignore[method-assign]
        original_immediate = blocked.tactical_solver.immediate_wins

        def immediate(board: np.ndarray, side: int) -> tuple[int, ...]:
            if int(board.ravel()[attack]) == game.player and side == -game.player:
                return (self.unsafe,)
            return original_immediate(board, side)

        blocked.tactical_solver.immediate_wins = immediate  # type: ignore[method-assign]
        rejected = blocked.decide(game, simulations=0)
        self.assertNotEqual(rejected.reason, "mcts_vct_attack")
        self.assertFalse(rejected.proven)

    def test_filter_is_black_white_symmetric(self) -> None:
        decisions = []
        for player in (BLACK, -BLACK):
            search = V3RootSearch(
                self.model, self.config, "cpu", vct_options=self.options
            )

            def fake(board: np.ndarray, _side: int, _limits: object) -> ThreatSolveResult:
                return result(
                    SolveStatus.PROVEN_WIN
                    if int(board.ravel()[self.unsafe]) != 0
                    else SolveStatus.UNKNOWN_BUDGET
                )

            search.tactical_solver.solve_vct = fake  # type: ignore[method-assign]
            decisions.append(search.decide(self.game(player), simulations=0))
        self.assertEqual(decisions[0].action, decisions[1].action)
        np.testing.assert_array_equal(decisions[0].policy, decisions[1].policy)


if __name__ == "__main__":
    unittest.main()

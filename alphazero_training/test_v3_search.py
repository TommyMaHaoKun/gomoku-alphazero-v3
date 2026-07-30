from __future__ import annotations

import threading
import unittest

import numpy as np
import torch

from alphazero_training.train_alphazero import BLACK, Config, GomokuGame
from alphazero_training.play_agent import AlphaZeroGomokuAgent
from alphazero_training.tactical_solver import SolveResult, SolveStatus
from alphazero_training.v3_search import SearchDecision, V3RootSearch, search_root
from alphazero_training.v3_tactical_suite import built_in_cases, oracle_actions


SIZE = 19
ACTION_COUNT = SIZE * SIZE


class _ZeroModel(torch.nn.Module):
    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = states.shape[0]
        return (
            torch.zeros((batch, ACTION_COUNT), device=states.device),
            torch.zeros(batch, device=states.device),
        )


class _PreferredActionModel(torch.nn.Module):
    def __init__(self, preferred_action: int) -> None:
        super().__init__()
        self.preferred_action = preferred_action

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = states.shape[0]
        logits = torch.zeros((batch, ACTION_COUNT), device=states.device)
        logits[:, self.preferred_action] = 12.0
        return logits, torch.zeros(batch, device=states.device)


class _MappedLogitModel(torch.nn.Module):
    def __init__(self, logits_by_action: dict[int, float]) -> None:
        super().__init__()
        self.logits_by_action = logits_by_action

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = states.shape[0]
        logits = torch.full(
            (batch, ACTION_COUNT),
            -40.0,
            device=states.device,
        )
        for action, value in self.logits_by_action.items():
            logits[:, action] = value
        return logits, torch.zeros(batch, device=states.device)


def _solve_result(
    status: SolveStatus,
    winning_actions: tuple[int, ...] = (),
) -> SolveResult:
    return SolveResult(status, winning_actions, (), (), 1)


def _game_for_case(case: object) -> GomokuGame:
    game = GomokuGame(SIZE, 5)
    game.player = case.side_to_move
    for x, y, stone in case.stones:
        game.board[y, x] = stone
    game.move_count = len(case.stones)
    return game


def _assert_valid_decision(
    test: unittest.TestCase,
    game: GomokuGame,
    decision: SearchDecision,
) -> None:
    test.assertEqual(decision.policy.shape, (ACTION_COUNT,))
    test.assertEqual(decision.policy.dtype, np.float32)
    test.assertTrue(np.all(np.isfinite(decision.policy)))
    test.assertTrue(np.all(decision.policy >= 0))
    test.assertAlmostEqual(float(decision.policy.sum(dtype=np.float64)), 1.0, places=6)
    test.assertEqual(int(game.board.ravel()[decision.action]), 0)
    occupied = np.flatnonzero(game.board.ravel() != 0)
    test.assertTrue(np.all(decision.policy[occupied] == 0))


class TacticalRootSearchTests(unittest.TestCase):
    def setUp(self) -> None:
        # radius=0 deliberately makes the old spatial root filter empty on
        # non-empty positions.  Exact tactical actions must still survive.
        self.config = Config(
            simulations=2,
            inference_batch_per_game=1,
            candidate_radius=0,
            heuristic_prior_weight=0.0,
        )
        self.search = V3RootSearch(
            _ZeroModel(),
            self.config,
            "cpu",
            rng=np.random.default_rng(20260722),
        )

    def test_all_ten_current_tactical_cases(self) -> None:
        reason_for_oracle = {
            "immediate_win": "immediate_win",
            "immediate_block": "immediate_block",
            "forced_win_in_3": "win_in_3",
            "prevent_forced_win_in_3": "block_win_in_3",
        }
        cases = built_in_cases()
        self.assertEqual(len(cases), 10)
        for case in cases:
            with self.subTest(case=case.case_id):
                game = _game_for_case(case)
                expected = tuple(sorted(oracle_actions(case)))
                decision = self.search.decide(game, simulations=0)
                _assert_valid_decision(self, game, decision)
                self.assertIn(decision.action, expected)
                self.assertTrue(decision.proven)
                self.assertEqual(decision.reason, reason_for_oracle[case.oracle_kind])
                self.assertEqual(
                    tuple(map(int, np.flatnonzero(decision.policy))),
                    expected,
                )
                masses = decision.policy[np.asarray(expected, dtype=np.int32)]
                np.testing.assert_allclose(
                    masses,
                    np.full(len(expected), 1.0 / len(expected)),
                    rtol=0,
                    atol=2e-7,
                )

    def test_all_equally_proven_attacks_are_labeled(self) -> None:
        case = next(case for case in built_in_cases() if case.case_id == "open_four_black")
        game = _game_for_case(case)
        decision = self.search.decide(game, temperature=0.0)
        expected = tuple(sorted(oracle_actions(case)))
        self.assertEqual(len(expected), 2)
        self.assertEqual(tuple(map(int, np.flatnonzero(decision.policy))), expected)
        self.assertEqual(decision.action, expected[0])
        np.testing.assert_allclose(decision.policy[list(expected)], (0.5, 0.5))

    def test_two_opponent_immediate_wins_are_unavoidable_not_a_proven_block(self) -> None:
        # White has an open four.  Black can occupy either endpoint, but one
        # move cannot cover both; claiming ``immediate_block`` here previously
        # turned a forced loss into a false positive tactical label.
        left = 9 * SIZE + 6
        right = 9 * SIZE + 11
        game = GomokuGame(SIZE, 5)
        for x in (7, 8, 9, 10):
            game.board[9, x] = -BLACK
        game.move_count = 4
        game.player = BLACK

        decision = self.search.decide(game, simulations=0, temperature=0.0)
        _assert_valid_decision(self, game, decision)
        self.assertEqual(decision.action, left)
        self.assertEqual(decision.reason, "unavoidable_immediate_loss")
        self.assertFalse(decision.proven)
        self.assertEqual(
            tuple(map(int, np.flatnonzero(decision.policy))),
            (left, right),
        )
        np.testing.assert_allclose(decision.policy[[left, right]], (0.5, 0.5))

    def test_multiple_exact_defenses_are_all_kept_but_mcts_ranks_move(self) -> None:
        # White's open three has two win-in-three roots.  Black can avert the
        # bounded proof at either endpoint.  The neural prior strongly prefers
        # the right endpoint, which MCTS must be allowed to rank even though
        # candidate_radius=0 would prune both in the original root expander.
        left = 9 * SIZE + 6
        right = 9 * SIZE + 10
        game = GomokuGame(SIZE, 5)
        for x in (7, 8, 9):
            game.board[9, x] = -BLACK
        game.move_count = 3
        game.player = BLACK

        search = V3RootSearch(
            _PreferredActionModel(right),
            self.config,
            "cpu",
            rng=np.random.default_rng(7),
        )
        decision = search.decide(game, simulations=24, temperature=0.0)
        _assert_valid_decision(self, game, decision)
        self.assertEqual(decision.action, right)
        self.assertEqual(decision.reason, "block_win_in_3_mcts")
        self.assertTrue(decision.proven)
        self.assertEqual(tuple(map(int, np.flatnonzero(decision.policy))), (left, right))
        np.testing.assert_allclose(decision.policy[[left, right]], (0.5, 0.5))


class GeneralRootSearchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Config(
            simulations=3,
            inference_batch_per_game=1,
            heuristic_prior_weight=0.0,
        )

    def test_empty_board_mcts_and_one_shot_wrapper(self) -> None:
        game = GomokuGame(SIZE, 5)
        decision = search_root(
            _ZeroModel(),
            game,
            self.config,
            "cpu",
            simulations=0,
            rng=np.random.default_rng(1),
        )
        _assert_valid_decision(self, game, decision)
        self.assertEqual(decision.action, 9 * SIZE + 9)
        self.assertEqual(decision.reason, "mcts")
        self.assertFalse(decision.proven)

    def test_random_reachable_boards_always_return_legal_normalized_policy(self) -> None:
        board_rng = np.random.default_rng(12345)
        search = V3RootSearch(
            _ZeroModel(),
            self.config,
            "cpu",
            rng=np.random.default_rng(54321),
        )
        checked = 0
        for requested_plies in (1, 4, 8, 12, 18, 24):
            game = GomokuGame(SIZE, 5)
            for _ in range(requested_plies):
                if game.terminal:
                    break
                legal = game.legal_actions()
                game.play(int(board_rng.choice(legal)))
            if game.terminal:
                continue
            decision = search.decide(
                game,
                simulations=2,
                add_noise=(checked % 2 == 1),
                temperature=1.0 if checked % 2 else 0.0,
            )
            _assert_valid_decision(self, game, decision)
            checked += 1
        self.assertGreaterEqual(checked, 5)

    def test_policy_is_detached_normalized_float32(self) -> None:
        raw = np.zeros(ACTION_COUNT, dtype=np.float64)
        raw[1] = 2.0
        raw[2] = 2.0
        decision = SearchDecision(1, raw, "test", False)
        raw[1] = 0.0
        self.assertEqual(decision.policy.dtype, np.float32)
        self.assertAlmostEqual(float(decision.policy.sum(dtype=np.float64)), 1.0, places=7)
        np.testing.assert_allclose(decision.policy[[1, 2]], (0.5, 0.5))

    def test_rejects_terminal_or_invalid_search_parameters(self) -> None:
        search = V3RootSearch(_ZeroModel(), self.config, "cpu")
        game = GomokuGame(SIZE, 5)
        game.terminal = True
        with self.assertRaisesRegex(ValueError, "terminal"):
            search.decide(game)

        game.terminal = False
        with self.assertRaisesRegex(ValueError, "non-negative"):
            search.decide(game, simulations=-1)
        with self.assertRaisesRegex(ValueError, "temperature"):
            search.decide(game, temperature=float("nan"))

    def test_batched_mcts_matches_sequential_without_noise(self) -> None:
        config = Config(
            simulations=4,
            inference_batch_per_game=2,
            heuristic_prior_weight=0.0,
        )
        games: list[GomokuGame] = []
        for first_action in (9 * SIZE + 9, 8 * SIZE + 9, 9 * SIZE + 8):
            game = GomokuGame(SIZE, 5)
            game.play(first_action)
            games.append(game)

        batched_search = V3RootSearch(
            _ZeroModel(), config, "cpu", rng=np.random.default_rng(44)
        )
        batched, stats = batched_search.decide_batch(
            games,
            simulations=4,
            add_noise=False,
            temperature=[0.0, 0.0, 0.0],
        )
        sequential = tuple(
            V3RootSearch(
                _ZeroModel(), config, "cpu", rng=np.random.default_rng(100 + index)
            ).decide(game, simulations=4, add_noise=False, temperature=0.0)
            for index, game in enumerate(games)
        )
        self.assertEqual(
            [decision.action for decision in sequential],
            [decision.action for decision in batched],
        )
        for expected, actual in zip(sequential, batched):
            self.assertEqual(expected.reason, actual.reason)
            np.testing.assert_allclose(expected.policy, actual.policy, rtol=0, atol=0)
        self.assertEqual(3, stats.root_batch_size)
        self.assertEqual(3, stats.mcts_positions)
        self.assertEqual(0, stats.direct_positions)
        self.assertGreaterEqual(stats.max_inference_batch_size, 3)
        self.assertGreater(stats.evaluated_positions, stats.inference_calls)

    def test_direct_tactics_are_excluded_from_neural_batch(self) -> None:
        tactical_case = next(
            case for case in built_in_cases() if case.case_id == "immediate_attack_black"
        )
        tactical_game = _game_for_case(tactical_case)
        ordinary_game = GomokuGame(SIZE, 5)
        search = V3RootSearch(
            _ZeroModel(), self.config, "cpu", rng=np.random.default_rng(8)
        )
        decisions, stats = search.decide_batch(
            [tactical_game, ordinary_game],
            simulations=2,
            temperature=[0.0, 0.0],
        )
        self.assertEqual("immediate_win", decisions[0].reason)
        self.assertEqual("mcts", decisions[1].reason)
        self.assertEqual(1, stats.direct_positions)
        self.assertEqual(1, stats.mcts_positions)
        self.assertEqual(1, stats.root_batch_size)

    def test_single_item_batch_preserves_decide_semantics(self) -> None:
        game = GomokuGame(SIZE, 5)
        first = V3RootSearch(
            _ZeroModel(), self.config, "cpu", rng=np.random.default_rng(77)
        ).decide(game, simulations=3, add_noise=True, temperature=1.0)
        second_search = V3RootSearch(
            _ZeroModel(), self.config, "cpu", rng=np.random.default_rng(77)
        )
        batched, stats = second_search.decide_batch(
            [game], simulations=3, add_noise=True, temperature=[1.0]
        )
        self.assertEqual(first.action, batched[0].action)
        self.assertEqual(first.reason, batched[0].reason)
        np.testing.assert_allclose(first.policy, batched[0].policy, rtol=0, atol=0)
        self.assertEqual(second_search.last_batch_stats, stats)

        with self.assertRaisesRegex(ValueError, "temperatures"):
            second_search.decide_batch([game], temperature=[0.0, 1.0])


class RootVCFSafetyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.center = 9 * SIZE + 9
        self.unsafe = 9 * SIZE + 8
        self.safe = 9 * SIZE + 10
        self.model = _MappedLogitModel({self.unsafe: 8.0, self.safe: 7.0})

    def _config(self, **overrides: object) -> Config:
        values: dict[str, object] = {
            "simulations": 0,
            "candidate_radius": 2,
            "heuristic_prior_weight": 0.0,
            "vcf_root_filter": True,
            "vcf_attack_priority": False,
            "vcf_root_candidates": 2,
            "vcf_min_policy": 0.0,
            "vcf_max_plies": 5,
            "vcf_max_nodes": 100,
            "vcf_time_ms": 0.0,
        }
        values.update(overrides)
        return Config(**values)

    def _game(self, side_to_move: int = -BLACK) -> GomokuGame:
        game = GomokuGame(SIZE, 5)
        game.board.ravel()[self.center] = -side_to_move
        game.player = side_to_move
        game.move_count = 1
        return game

    @staticmethod
    def _real_five_ply_vcf_game(
        *, extra_white_blocks: tuple[int, ...] = ()
    ) -> GomokuGame:
        """Return a reachable-count fixture with a real black five-ply VCF.

        The base threat is the deterministic fixture used by the tactical
        solver tests.  It has no immediate or three-ply win, so it reaches the
        ordinary MCTS root before the bounded VCF guard is applied.
        """

        black_stones = (
            tuple((x, 8, BLACK) for x in (6, 7, 8))
            + tuple((x, 9, BLACK) for x in (6, 7, 8))
            + tuple((9, y, BLACK) for y in (6, 7))
        )
        white_stones = (
            (5, 8, -BLACK),
            (5, 9, -BLACK),
            (9, 5, -BLACK),
            (10, 6, -BLACK),
            (10, 5, -BLACK),
        )
        game = GomokuGame(SIZE, 5)
        for x, y, stone in black_stones + white_stones:
            game.board[y, x] = stone
        if len(extra_white_blocks) > 2:
            raise ValueError("VCF fixture accepts at most two local white blocks")
        # Black has eight stones and the base fixture has five white stones.
        # Fill any unused local-block slots with remote inert stones so every
        # returned board has legal 8:7 counts with White to move.
        balancing_stones = (0, SIZE * SIZE - 1)[: 2 - len(extra_white_blocks)]
        for action in extra_white_blocks + balancing_stones:
            if int(game.board.ravel()[action]) != 0:
                raise ValueError("extra VCF fixture block is occupied")
            game.board.ravel()[action] = -BLACK
        game.player = -BLACK
        game.move_count = int(np.count_nonzero(game.board))
        return game

    def _baseline(self, game: GomokuGame) -> SearchDecision:
        return V3RootSearch(
            self.model,
            self._config(vcf_root_filter=False),
            "cpu",
            rng=np.random.default_rng(10),
        ).decide(game, simulations=0)

    def test_proven_unsafe_move_is_zeroed_and_policy_is_normalized(self) -> None:
        game = self._game()
        search = V3RootSearch(
            self.model,
            self._config(),
            "cpu",
            rng=np.random.default_rng(10),
        )

        def fake_solve(board: np.ndarray, _attacker: int, _limits: object) -> SolveResult:
            status = (
                SolveStatus.PROVEN_WIN
                if int(board.ravel()[self.unsafe]) != 0
                else SolveStatus.PROVEN_NO_VCF
            )
            return _solve_result(status)

        search.tactical_solver.solve_vcf = fake_solve  # type: ignore[method-assign]
        decision = search.decide(game, simulations=0)

        _assert_valid_decision(self, game, decision)
        self.assertEqual(decision.action, self.safe)
        self.assertEqual(decision.reason, "mcts_vcf_safe")
        self.assertFalse(decision.proven)
        self.assertEqual(float(decision.policy[self.unsafe]), 0.0)
        self.assertGreater(float(decision.policy[self.safe]), 0.0)
        self.assertAlmostEqual(
            float(decision.policy.sum(dtype=np.float64)),
            1.0,
            places=7,
        )

    def test_real_solver_filters_move_that_allows_five_ply_vcf(self) -> None:
        # White blocks two of the four VCF roots in the base fixture.  Black's
        # only remaining five-ply VCF starts at (6, 10): occupying that point
        # is safe within the same bound, while the high-prior move (10, 10)
        # leaves the proof intact.  No solve_vcf method is mocked here.
        safe = 10 * SIZE + 6
        unsafe = 10 * SIZE + 10
        game = self._real_five_ply_vcf_game(
            extra_white_blocks=(7 * SIZE + 8, 8 * SIZE + 9)
        )
        model = _MappedLogitModel({unsafe: 8.0, safe: 7.0})
        search = V3RootSearch(
            model,
            self._config(vcf_max_nodes=5_000),
            "cpu",
            rng=np.random.default_rng(11),
        )

        decision = search.decide(game, simulations=0)

        _assert_valid_decision(self, game, decision)
        self.assertEqual(decision.reason, "mcts_vcf_safe")
        self.assertEqual(decision.action, safe)
        self.assertEqual(float(decision.policy[unsafe]), 0.0)
        self.assertGreater(float(decision.policy[safe]), 0.0)

    def test_unknown_budget_is_never_filtered(self) -> None:
        game = self._game()
        baseline = self._baseline(game)
        search = V3RootSearch(self.model, self._config(), "cpu")

        def fake_solve(board: np.ndarray, _attacker: int, _limits: object) -> SolveResult:
            status = (
                SolveStatus.UNKNOWN_BUDGET
                if int(board.ravel()[self.unsafe]) != 0
                else SolveStatus.PROVEN_NO_VCF
            )
            return _solve_result(status)

        search.tactical_solver.solve_vcf = fake_solve  # type: ignore[method-assign]
        decision = search.decide(game, simulations=0)

        self.assertEqual(decision.reason, "mcts")
        self.assertGreater(float(decision.policy[self.unsafe]), 0.0)
        np.testing.assert_allclose(decision.policy, baseline.policy, rtol=0, atol=0)

    def test_all_checked_candidates_unsafe_falls_back_to_original_policy(self) -> None:
        game = self._game()
        baseline = self._baseline(game)
        search = V3RootSearch(self.model, self._config(), "cpu")
        search.tactical_solver.solve_vcf = (  # type: ignore[method-assign]
            lambda _board, _attacker, _limits: _solve_result(SolveStatus.PROVEN_WIN)
        )

        decision = search.decide(game, simulations=0)

        self.assertEqual(decision.reason, "mcts")
        np.testing.assert_allclose(decision.policy, baseline.policy, rtol=0, atol=0)
        self.assertEqual(decision.action, baseline.action)

    def test_real_solver_all_unsafe_candidates_never_empty_the_root(self) -> None:
        # On the unblocked five-ply fixture, both preferred white moves leave
        # a proven black VCF.  The guard must retain the original normalized
        # root rather than zeroing every checked candidate or pretending an
        # unchecked low-prior move was proved safe.
        first = 10 * SIZE + 10
        second = 10 * SIZE + 7
        game = self._real_five_ply_vcf_game()
        model = _MappedLogitModel({first: 8.0, second: 7.0})
        baseline = V3RootSearch(
            model,
            self._config(vcf_root_filter=False, vcf_max_nodes=5_000),
            "cpu",
            rng=np.random.default_rng(12),
        ).decide(game, simulations=0)
        guarded = V3RootSearch(
            model,
            self._config(vcf_max_nodes=5_000),
            "cpu",
            rng=np.random.default_rng(12),
        ).decide(game, simulations=0)

        _assert_valid_decision(self, game, guarded)
        self.assertEqual(guarded.reason, "mcts")
        self.assertEqual(guarded.action, baseline.action)
        np.testing.assert_allclose(guarded.policy, baseline.policy, rtol=0, atol=0)

    def test_safety_filter_is_black_white_symmetric(self) -> None:
        decisions: list[SearchDecision] = []
        for side_to_move in (-BLACK, BLACK):
            game = self._game(side_to_move)
            search = V3RootSearch(self.model, self._config(), "cpu")

            def fake_solve(
                board: np.ndarray,
                _attacker: int,
                _limits: object,
            ) -> SolveResult:
                status = (
                    SolveStatus.PROVEN_WIN
                    if int(board.ravel()[self.unsafe]) != 0
                    else SolveStatus.PROVEN_NO_VCF
                )
                return _solve_result(status)

            search.tactical_solver.solve_vcf = fake_solve  # type: ignore[method-assign]
            decisions.append(search.decide(game, simulations=0))

        self.assertEqual(decisions[0].action, decisions[1].action)
        self.assertEqual(decisions[0].reason, decisions[1].reason)
        np.testing.assert_allclose(
            decisions[0].policy,
            decisions[1].policy,
            rtol=0,
            atol=0,
        )

    def test_low_risk_proven_attack_overrides_higher_mcts_prior(self) -> None:
        game = self._game()
        attack = self.safe
        search = V3RootSearch(
            self.model,
            self._config(vcf_attack_priority=True),
            "cpu",
        )

        def fake_solve(board: np.ndarray, attacker: int, _limits: object) -> SolveResult:
            if int(np.count_nonzero(board)) == 1 and attacker == game.player:
                return _solve_result(SolveStatus.PROVEN_WIN, (attack,))
            return _solve_result(SolveStatus.PROVEN_NO_VCF)

        search.tactical_solver.solve_vcf = fake_solve  # type: ignore[method-assign]
        decision = search.decide(game, simulations=0)

        self.assertEqual(decision.action, attack)
        self.assertEqual(decision.reason, "mcts_vcf_attack")
        self.assertTrue(decision.proven)
        self.assertEqual(
            tuple(map(int, np.flatnonzero(decision.policy))),
            (attack,),
        )

    def test_unknown_attack_reply_does_not_get_low_risk_priority(self) -> None:
        game = self._game()
        attack = self.safe
        baseline = self._baseline(game)
        search = V3RootSearch(
            self.model,
            self._config(vcf_attack_priority=True),
            "cpu",
        )

        def fake_solve(board: np.ndarray, attacker: int, _limits: object) -> SolveResult:
            if int(np.count_nonzero(board)) == 1 and attacker == game.player:
                return _solve_result(SolveStatus.PROVEN_WIN, (attack,))
            if int(board.ravel()[attack]) != 0:
                return _solve_result(SolveStatus.UNKNOWN_BUDGET)
            return _solve_result(SolveStatus.PROVEN_NO_VCF)

        search.tactical_solver.solve_vcf = fake_solve  # type: ignore[method-assign]
        decision = search.decide(game, simulations=0)

        self.assertEqual(decision.reason, "mcts")
        self.assertEqual(decision.action, baseline.action)
        np.testing.assert_allclose(decision.policy, baseline.policy, rtol=0, atol=0)


class DesktopSearchIntegrationTests(unittest.TestCase):
    def test_desktop_choose_move_delegates_to_shared_v3_root_search(self) -> None:
        class RecordingSearch:
            def __init__(self) -> None:
                self.calls: list[tuple[GomokuGame, dict[str, object]]] = []

            def decide(self, game: GomokuGame, **kwargs: object) -> SearchDecision:
                self.calls.append((game, kwargs))
                policy = np.zeros(ACTION_COUNT, dtype=np.float32)
                policy[1] = 1.0
                return SearchDecision(1, policy, "delegated_test", False)

        # Bypass checkpoint I/O: this test targets only the public desktop
        # adapter's routing contract, while verify_play_integration exercises
        # a real checkpoint end to end.
        agent = AlphaZeroGomokuAgent.__new__(AlphaZeroGomokuAgent)
        agent.config = Config()
        agent.simulations = 17
        agent._search_lock = threading.Lock()
        agent.last_decision_reason = None
        recording_search = RecordingSearch()
        agent.root_search = recording_search

        grid = [[0] * SIZE for _ in range(SIZE)]
        grid[0][0] = 1
        move = agent.choose_move(grid, last_move=(0, 0), ai_color=2)

        self.assertEqual(move, (1, 0))
        self.assertEqual(agent.last_decision_reason, "delegated_test")
        self.assertEqual(len(recording_search.calls), 1)
        game, kwargs = recording_search.calls[0]
        self.assertEqual(game.player, -BLACK)
        self.assertEqual(game.last_action, 0)
        self.assertEqual(int(game.board[0, 0]), BLACK)
        self.assertEqual(
            kwargs,
            {"simulations": 17, "add_noise": False, "temperature": 0.0},
        )


if __name__ == "__main__":
    unittest.main()

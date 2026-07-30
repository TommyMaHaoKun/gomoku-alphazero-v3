"""Shared tactical-first root search for Gomoku V3.

The original trainer and desktop player had separate root-selection logic.  In
particular, a move found by an exact tactical oracle could be dropped by the
trainer's spatial candidate filter, and the desktop player emitted a one-hot
choice even when several moves had the same proof.  This module provides one
small search contract that both paths can use:

* exact one-ply wins and blocks;
* exact wins in three plies;
* exact defenses against an opponent win in three, with MCTS used to rank a
  multi-move safe set; and
* ordinary policy/value MCTS when no short proof applies; and
* a bounded root VCF guard that removes only moves proven to give the
  opponent a continuous-four win, while never treating a budget cutoff as a
  loss.

``SearchDecision.policy`` is a normalized training target.  For a direct
proof it is uniform over the complete set of equally proven root actions, not
an arbitrary one-hot vector.  For ordinary search it is the MCTS visit policy.
When several exact defensive moves exist, all of them are kept as explicit
root children and MCTS ranks the actual move; the returned tactical label is
still uniform over the exact safe set.  If the opponent already has multiple
immediate winning points, the fallback is uniform over the legal one-point
blocks but is explicitly marked as an unproven, unavoidable loss.

Architecture / 代码架构
-----------------------
``V3RootSearch`` is the shared decision engine used by both training and the
desktop player. It first routes each position through short exact tactics,
then sends unresolved positions into one batched policy-value MCTS wave.
``SearchDecision`` returns both the played action and the normalized policy
target needed by replay training.

``V3RootSearch`` 是训练端与桌面端共用的决策引擎。它先让每个局面经过短期精确
战术路由，再把仍未解决的局面合并进一轮批量策略-价值 MCTS。``SearchDecision``
同时返回实际落子和回放训练所需的归一化策略目标。

Key algorithms / 重要算法
-------------------------
Priority is: immediate win, mandatory block, win in three, exact safe defense,
then ordinary MCTS. Optional bounded VCF/VCT guards may reject only moves that
are proven losing; ``UNKNOWN_BUDGET`` is never treated as a loss. In ordinary
search, 256 simulations mean 256 tree traversals, not a fixed depth of 256
moves, and the root action with the strongest visit evidence is selected.

优先级依次为：一步必胜、必须封堵、三手强制胜、安全防守集合、普通 MCTS。可选的
有限 VCF/VCT 保护只排除已经证明会输的走法，``UNKNOWN_BUDGET`` 绝不当作失败。
普通搜索中的 256 次模拟表示 256 次树遍历，不是固定向后搜索 256 手；最终根据
根节点访问证据选择落子。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

import numpy as np
import torch

from .tactical_solver import (
    SolveLimits,
    SolveStatus,
    TacticalSolver,
    ThreatSolveLimits,
)
from .train_alphazero import (
    BLACK,
    WHITE,
    Config,
    GomokuGame,
    Node,
    evaluate_positions,
    expand_node,
    root_policy,
    run_mcts_batch,
)


@dataclass(frozen=True)
class SearchDecision:
    """One root-search result.

    ``action`` is row-major (``y * board_size + x``).  ``policy`` is a
    ``float32`` vector with one entry per board point and sums to one.
    ``proven`` means the root action set came from an exact short-tactical
    oracle; it does not claim that a defensive position is a forced win.
    """

    action: int
    policy: np.ndarray
    reason: str
    proven: bool

    def __post_init__(self) -> None:
        # Prevent a caller from mutating a replay target after it has been
        # handed to another component.  ``copy`` also detaches array views.
        policy = np.asarray(self.policy, dtype=np.float32).reshape(-1).copy()
        if policy.size == 0:
            raise ValueError("policy must not be empty")
        if not np.all(np.isfinite(policy)) or np.any(policy < 0):
            raise ValueError("policy must contain finite, non-negative values")
        total = float(policy.sum(dtype=np.float64))
        if not math.isfinite(total) or total <= 0:
            raise ValueError("policy must have positive mass")
        policy /= np.float32(total)
        # Correct the tiny float32 rounding residue so downstream checks can
        # use a tight tolerance even for 361-way policies.
        residue = np.float32(1.0 - float(policy.sum(dtype=np.float64)))
        policy[int(np.argmax(policy))] += residue
        policy.setflags(write=False)
        object.__setattr__(self, "policy", policy)


@dataclass(frozen=True)
class BatchSearchStats:
    """Inference utilization for one tactical-routing/search wave."""

    positions: int
    direct_positions: int
    mcts_positions: int
    root_batch_size: int
    inference_calls: int
    evaluated_positions: int
    max_inference_batch_size: int
    mean_inference_batch_size: float
    inference_batch_histogram: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class VCTRootOptions:
    """Opt-in conservative VCT root guard.

    It is disabled by default because threat-space enumeration is materially
    more expensive than VCF.  Enabling it never changes the meaning of a
    budget cutoff: only ``PROVEN_WIN`` can filter a move.
    """

    enabled: bool = False
    attack_priority: bool = True
    root_candidates: int = 4
    minimum_probability: float = 0.02
    limits: ThreatSolveLimits = field(
        default_factory=lambda: ThreatSolveLimits(
            max_plies=9,
            max_nodes=5_000,
            time_ms=100.0,
            max_attack_candidates=4,
            max_defenses=32,
        )
    )

    def __post_init__(self) -> None:
        if self.root_candidates <= 0:
            raise ValueError("VCT root_candidates must be positive")
        if not math.isfinite(float(self.minimum_probability)) or not (
            0.0 <= self.minimum_probability <= 1.0
        ):
            raise ValueError("VCT minimum_probability must be between zero and one")


@dataclass(frozen=True)
class _SearchRoute:
    direct: SearchDecision | None
    legal: np.ndarray
    root_actions: tuple[int, ...] | None = None
    target_policy: np.ndarray | None = None
    reason: str = "mcts"
    proven: bool = False


class _ObservedModel(torch.nn.Module):
    """Transparent forward proxy that records real inference batch sizes."""

    def __init__(self, wrapped: torch.nn.Module, batch_sizes: list[int]) -> None:
        super().__init__()
        self.wrapped = wrapped
        self.batch_sizes = batch_sizes

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.batch_sizes.append(int(states.shape[0]))
        return self.wrapped(states)


def _validate_search_inputs(
    game: GomokuGame,
    config: Config,
    simulations: int,
    temperature: float,
) -> np.ndarray:
    if game.size != config.board_size:
        raise ValueError(
            f"game size {game.size} disagrees with config size {config.board_size}"
        )
    if game.win_length != config.win_length:
        raise ValueError(
            "game win length "
            f"{game.win_length} disagrees with config {config.win_length}"
        )
    if game.player not in (BLACK, WHITE):
        raise ValueError(f"invalid side to move: {game.player!r}")
    if game.terminal:
        raise ValueError("cannot search a terminal position")
    if not isinstance(simulations, int) or isinstance(simulations, bool):
        raise TypeError("simulations must be an integer")
    if simulations < 0:
        raise ValueError("simulations must be non-negative")
    if not math.isfinite(float(temperature)) or temperature < 0:
        raise ValueError("temperature must be finite and non-negative")
    legal = game.legal_actions()
    if legal.size == 0:
        raise ValueError("cannot search a position with no legal actions")
    return legal


def _uniform_policy(actions: tuple[int, ...], action_count: int) -> np.ndarray:
    unique = tuple(sorted(set(map(int, actions))))
    if not unique:
        raise ValueError("cannot build a policy from an empty action set")
    if unique[0] < 0 or unique[-1] >= action_count:
        raise ValueError("tactical action is outside the board")
    policy = np.zeros(action_count, dtype=np.float32)
    policy[np.asarray(unique, dtype=np.int32)] = np.float32(1.0 / len(unique))
    residue = np.float32(1.0 - float(policy.sum(dtype=np.float64)))
    policy[unique[0]] += residue
    return policy


def _sample_policy(
    policy: np.ndarray,
    temperature: float,
    rng: np.random.Generator,
) -> int:
    support = np.flatnonzero(policy > 0).astype(np.int32)
    if support.size == 0:
        raise RuntimeError("search produced an empty policy")
    weights = policy[support].astype(np.float64)
    if temperature == 0:
        # ``support`` is sorted, so ties are deterministic and reproducible.
        return int(support[int(np.argmax(weights))])

    # Compute p**(1/T) in log space.  This remains stable for very small
    # temperatures and highly concentrated visit distributions.
    log_weights = np.log(weights) / float(temperature)
    log_weights -= float(log_weights.max())
    probabilities = np.exp(log_weights)
    total = float(probabilities.sum())
    if not math.isfinite(total) or total <= 0:
        probabilities = np.full(support.size, 1.0 / support.size, dtype=np.float64)
    else:
        probabilities /= total
    return int(rng.choice(support, p=probabilities))


def _renormalized_policy(
    policy: np.ndarray,
    *,
    keep_actions: tuple[int, ...] | None = None,
    drop_actions: tuple[int, ...] = (),
) -> np.ndarray:
    """Return a detached normalized policy after a root safety edit.

    A proven attacking action can sit outside the spatially-pruned MCTS root.
    In that case restricting to it would otherwise have zero mass, so a
    uniform distribution over the requested legal actions is the sound
    fallback.
    """

    result = np.asarray(policy, dtype=np.float32).reshape(-1).copy()
    if keep_actions is not None:
        keep = tuple(sorted(set(map(int, keep_actions))))
        if not keep:
            raise ValueError("keep_actions must not be empty")
        if keep[0] < 0 or keep[-1] >= result.size:
            raise ValueError("kept action is outside the policy")
        mask = np.zeros(result.size, dtype=bool)
        mask[np.asarray(keep, dtype=np.int32)] = True
        result[~mask] = 0.0
        if float(result.sum(dtype=np.float64)) <= 0:
            result[np.asarray(keep, dtype=np.int32)] = np.float32(1.0 / len(keep))

    if drop_actions:
        drop = tuple(sorted(set(map(int, drop_actions))))
        if drop[0] < 0 or drop[-1] >= result.size:
            raise ValueError("dropped action is outside the policy")
        result[np.asarray(drop, dtype=np.int32)] = 0.0

    total = float(result.sum(dtype=np.float64))
    if not math.isfinite(total) or total <= 0:
        raise ValueError("root safety edit removed all policy mass")
    result /= np.float32(total)
    residue = np.float32(1.0 - float(result.sum(dtype=np.float64)))
    result[int(np.argmax(result))] += residue
    return result


def _ranked_policy_actions(
    policy: np.ndarray,
    *,
    count: int,
    minimum_probability: float,
) -> tuple[int, ...]:
    """Choose deterministic high-probability actions for bounded VCF checks."""

    support = np.flatnonzero(policy > 0).astype(np.int32)
    if support.size == 0:
        raise RuntimeError("cannot rank an empty root policy")
    # ``lexsort`` makes equal-probability ordering reproducible by action id.
    order = np.lexsort((support, -policy[support].astype(np.float64)))
    ranked = support[order]
    eligible = ranked[policy[ranked] >= minimum_probability]
    if eligible.size == 0:
        eligible = ranked[:1]
    return tuple(map(int, eligible[:count]))


def _expand_explicit_root(
    root: Node,
    game: GomokuGame,
    logits: np.ndarray,
    actions: tuple[int, ...],
    config: Config,
) -> None:
    """Expand exactly ``actions`` without passing through candidate pruning."""

    action_count = game.size * game.size
    legal_mask = game.board.ravel() == 0
    unique = tuple(sorted(set(map(int, actions))))
    if not unique:
        raise ValueError("explicit root action set must not be empty")
    if any(action < 0 or action >= action_count for action in unique):
        raise ValueError("explicit root action is outside the board")
    if any(not bool(legal_mask[action]) for action in unique):
        raise ValueError("explicit root action set contains an occupied point")

    indices = np.asarray(unique, dtype=np.int32)
    scores = np.asarray(logits[indices], dtype=np.float64)
    heuristic_weight = float(config.heuristic_prior_weight)
    if heuristic_weight:
        heuristic = np.asarray(
            [game.move_heuristic(action, game.player) for action in unique],
            dtype=np.float64,
        )
        scores += heuristic_weight * np.log1p(heuristic)

    # A corrupt network output must not make a forced tactical branch vanish.
    # Replacing non-finite scores with a neutral value keeps every proof move.
    finite = np.isfinite(scores)
    if not np.any(finite):
        probabilities = np.full(len(unique), 1.0 / len(unique), dtype=np.float64)
    else:
        floor = float(np.min(scores[finite]))
        scores = np.nan_to_num(scores, nan=floor, posinf=float(np.max(scores[finite])), neginf=floor)
        scores -= float(scores.max())
        probabilities = np.exp(scores)
        probabilities /= float(probabilities.sum())

    root.children = {
        action: Node(float(prior), -game.player)
        for action, prior in zip(unique, probabilities)
    }


class V3RootSearch:
    """Reusable tactical-first root search over the existing batched MCTS."""

    def __init__(
        self,
        model: torch.nn.Module,
        config: Config,
        device: torch.device | str,
        rng: np.random.Generator | None = None,
        tactical_solver: TacticalSolver | None = None,
        vct_options: VCTRootOptions | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.device = torch.device(device)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.tactical_solver = tactical_solver or TacticalSolver(
            board_size=config.board_size,
            win_length=config.win_length,
        )
        if self.tactical_solver.board_size != config.board_size:
            raise ValueError("tactical solver board size disagrees with config")
        if self.tactical_solver.win_length != config.win_length:
            raise ValueError("tactical solver win length disagrees with config")
        self.vct_options = vct_options or VCTRootOptions()
        if config.vcf_root_candidates <= 0:
            raise ValueError("vcf_root_candidates must be positive")
        if not math.isfinite(float(config.vcf_min_policy)) or not (
            0.0 <= config.vcf_min_policy <= 1.0
        ):
            raise ValueError("vcf_min_policy must be between zero and one")
        # SolveLimits performs the remaining bound validation once, rather
        # than once for every candidate in every move.
        self.vcf_limits = SolveLimits(
            max_plies=config.vcf_max_plies,
            max_nodes=config.vcf_max_nodes,
            time_ms=config.vcf_time_ms,
        )
        self.last_batch_stats = BatchSearchStats(
            positions=0,
            direct_positions=0,
            mcts_positions=0,
            root_batch_size=0,
            inference_calls=0,
            evaluated_positions=0,
            max_inference_batch_size=0,
            mean_inference_batch_size=0.0,
            inference_batch_histogram=(),
        )

    def _route(
        self,
        game: GomokuGame,
        *,
        simulations: int,
        temperature: float,
    ) -> _SearchRoute:
        """Resolve exact tactics without invoking the neural network."""

        legal = _validate_search_inputs(game, self.config, simulations, temperature)
        action_count = game.size * game.size
        player = game.player
        opponent = -player
        board = game.board

        own_wins = self.tactical_solver.immediate_wins(board, player)
        if own_wins:
            policy = _uniform_policy(own_wins, action_count)
            action = _sample_policy(policy, temperature, self.rng)
            return _SearchRoute(
                SearchDecision(action, policy, "immediate_win", True), legal
            )

        opponent_wins = self.tactical_solver.immediate_wins(board, opponent)
        if opponent_wins:
            policy = _uniform_policy(opponent_wins, action_count)
            action = _sample_policy(policy, temperature, self.rng)
            if len(opponent_wins) > 1:
                return _SearchRoute(
                    SearchDecision(
                        action,
                        policy,
                        "unavoidable_immediate_loss",
                        False,
                    ),
                    legal,
                )
            return _SearchRoute(
                SearchDecision(action, policy, "immediate_block", True), legal
            )

        forced_wins = self.tactical_solver.forced_wins_in_three(board, player)
        if forced_wins:
            policy = _uniform_policy(forced_wins, action_count)
            action = _sample_policy(policy, temperature, self.rng)
            return _SearchRoute(
                SearchDecision(action, policy, "win_in_3", True), legal
            )

        opponent_forced_wins = self.tactical_solver.forced_wins_in_three(
            board, opponent
        )
        if opponent_forced_wins:
            exact_defenses = self.tactical_solver.exact_defenses(board, player)
            if exact_defenses:
                tactical_policy = _uniform_policy(exact_defenses, action_count)
                if len(exact_defenses) == 1:
                    return _SearchRoute(
                        SearchDecision(
                            exact_defenses[0],
                            tactical_policy,
                            "block_win_in_3",
                            True,
                        ),
                        legal,
                    )
                return _SearchRoute(
                    None,
                    legal,
                    root_actions=exact_defenses,
                    target_policy=tactical_policy,
                    reason="block_win_in_3_mcts",
                    proven=True,
                )

        return _SearchRoute(None, legal)

    def _vcf_postprocess(
        self,
        game: GomokuGame,
        visit_policy: np.ndarray,
        *,
        base_reason: str,
    ) -> tuple[np.ndarray, str, bool]:
        """Apply bounded attacking priority and opponent-VCF safety filtering.

        Only ``PROVEN_WIN`` makes a candidate unsafe.  In particular,
        ``UNKNOWN_BUDGET`` retains its probability.  A root attacking proof is
        allowed to override MCTS only when the reply position is separately
        proven to contain no opponent VCF within the same bound; an unknown
        reply stays in the ordinary policy but is not advertised as low risk.
        """

        if not self.config.vcf_root_filter:
            return visit_policy, base_reason, False

        candidates = _ranked_policy_actions(
            visit_policy,
            count=self.config.vcf_root_candidates,
            minimum_probability=self.config.vcf_min_policy,
        )
        player = game.player
        action_count = game.size * game.size
        legal_mask = game.board.ravel() == 0
        reply_status: dict[int, SolveStatus] = {}

        def opponent_vcf_after(action: int) -> SolveStatus:
            cached = reply_status.get(action)
            if cached is not None:
                return cached
            if action < 0 or action >= action_count or not bool(legal_mask[action]):
                raise ValueError(f"VCF root action is illegal: {action}")
            after = game.board.copy()
            after.ravel()[action] = player
            status = self.tactical_solver.solve_vcf(
                after,
                -player,
                self.vcf_limits,
            ).status
            reply_status[action] = status
            return status

        if self.config.vcf_attack_priority:
            own_vcf = self.tactical_solver.solve_vcf(
                game.board,
                player,
                self.vcf_limits,
            )
            if own_vcf.status is SolveStatus.PROVEN_WIN:
                proven_actions = tuple(
                    action
                    for action in sorted(set(map(int, own_vcf.winning_actions)))
                    if 0 <= action < action_count and bool(legal_mask[action])
                )
                low_risk_attacks = tuple(
                    action
                    for action in proven_actions
                    if opponent_vcf_after(action) is SolveStatus.PROVEN_NO_VCF
                )
                if low_risk_attacks:
                    return (
                        _renormalized_policy(
                            visit_policy,
                            keep_actions=low_risk_attacks,
                        ),
                        "mcts_vcf_attack",
                        True,
                    )

        statuses = {
            action: opponent_vcf_after(action)
            for action in candidates
        }
        unsafe = tuple(
            action
            for action in candidates
            if statuses[action] is SolveStatus.PROVEN_WIN
        )
        # UNKNOWN_BUDGET deliberately belongs to the non-unsafe side here.
        # If every checked candidate is proven unsafe, retaining the original
        # distribution is safer than pretending an unchecked low-prior move
        # was proven good.
        has_non_unsafe = any(
            statuses[action] is not SolveStatus.PROVEN_WIN
            for action in candidates
        )
        if unsafe and has_non_unsafe:
            return (
                _renormalized_policy(visit_policy, drop_actions=unsafe),
                "mcts_vcf_safe",
                False,
            )
        return visit_policy, base_reason, False

    def _vct_postprocess(
        self,
        game: GomokuGame,
        policy: np.ndarray,
        *,
        base_reason: str,
    ) -> tuple[np.ndarray, str, bool]:
        """Apply the opt-in VCT guard without converting UNKNOWN into a loss."""

        options = self.vct_options
        if not options.enabled:
            return policy, base_reason, False

        player = game.player
        action_count = game.size * game.size
        legal_mask = game.board.ravel() == 0
        candidates = _ranked_policy_actions(
            policy,
            count=options.root_candidates,
            minimum_probability=options.minimum_probability,
        )
        reply_status: dict[int, SolveStatus] = {}

        def board_after(action: int) -> np.ndarray:
            if action < 0 or action >= action_count or not bool(legal_mask[action]):
                raise ValueError(f"VCT root action is illegal: {action}")
            after = game.board.copy()
            after.ravel()[action] = player
            return after

        def opponent_vct_after(action: int) -> SolveStatus:
            cached = reply_status.get(action)
            if cached is not None:
                return cached
            status = self.tactical_solver.solve_vct(
                board_after(action),
                -player,
                options.limits,
            ).status
            reply_status[action] = status
            return status

        if options.attack_priority:
            own = self.tactical_solver.solve_vct(
                game.board,
                player,
                options.limits,
            )
            if own.status is SolveStatus.PROVEN_WIN:
                attacks: list[int] = []
                for action in sorted(set(map(int, own.winning_actions))):
                    if not (0 <= action < action_count and bool(legal_mask[action])):
                        continue
                    after = board_after(action)
                    # The VCT proof already quantifies defender replies, but
                    # retain this explicit faster-counterkill gate at the root.
                    if self.tactical_solver.immediate_wins(after, -player):
                        continue
                    attacks.append(action)
                if attacks:
                    return (
                        _uniform_policy(tuple(attacks), action_count),
                        "mcts_vct_attack",
                        True,
                    )

        statuses = {action: opponent_vct_after(action) for action in candidates}
        unsafe = tuple(
            action
            for action in candidates
            if statuses[action] is SolveStatus.PROVEN_WIN
        )
        # UNKNOWN_BUDGET and either bounded negative status remain playable.
        has_non_unsafe = any(
            statuses[action] is not SolveStatus.PROVEN_WIN for action in candidates
        )
        if unsafe and has_non_unsafe:
            return (
                _renormalized_policy(policy, drop_actions=unsafe),
                "mcts_vct_safe",
                False,
            )
        return policy, base_reason, False

    @staticmethod
    def _temperatures(
        temperature: float | Sequence[float], count: int
    ) -> tuple[float, ...]:
        if np.isscalar(temperature):
            return (float(temperature),) * count
        values = tuple(float(value) for value in temperature)
        if len(values) != count:
            raise ValueError(
                f"expected {count} temperatures, received {len(values)}"
            )
        return values

    def decide_batch(
        self,
        games: Sequence[GomokuGame],
        *,
        simulations: int | None = None,
        add_noise: bool = False,
        temperature: float | Sequence[float] = 0.0,
    ) -> tuple[tuple[SearchDecision, ...], BatchSearchStats]:
        """Route tactics and search all remaining roots in one GPU batch.

        Direct exact moves never touch the network.  Ordinary roots and
        multi-defense roots share one root evaluation and one call to the
        existing multi-game MCTS, so its leaf evaluations have a real batch of
        up to ``len(games) * inference_batch_per_game`` positions.
        """

        if not games:
            raise ValueError("decide_batch requires at least one game")
        simulation_count = self.config.simulations if simulations is None else simulations
        temperatures = self._temperatures(temperature, len(games))
        routes = [
            self._route(
                game,
                simulations=simulation_count,
                temperature=temperatures[index],
            )
            for index, game in enumerate(games)
        ]
        decisions: list[SearchDecision | None] = [route.direct for route in routes]
        mcts_indices = [
            index for index, route in enumerate(routes) if route.direct is None
        ]
        batch_sizes: list[int] = []

        if mcts_indices:
            observed_model = _ObservedModel(self.model, batch_sizes)
            mcts_games = [games[index] for index in mcts_indices]
            logits_batch, _ = evaluate_positions(
                observed_model, mcts_games, self.device
            )
            roots: list[Node] = []
            for game, logits, route in zip(
                mcts_games,
                logits_batch,
                (routes[index] for index in mcts_indices),
            ):
                root = Node(1.0, game.player)
                if route.root_actions is None:
                    expand_node(root, game, logits, self.config)
                    if not root.children:
                        fallback = tuple(map(int, game.legal_actions()))
                        _expand_explicit_root(
                            root, game, logits, fallback, self.config
                        )
                else:
                    _expand_explicit_root(
                        root,
                        game,
                        logits,
                        route.root_actions,
                        self.config,
                    )
                roots.append(root)

            run_mcts_batch(
                observed_model,
                mcts_games,
                roots,
                simulation_count,
                self.config,
                self.device,
                self.rng,
                add_noise=add_noise,
            )
            for index, game, root in zip(mcts_indices, mcts_games, roots):
                route = routes[index]
                visit_policy = root_policy(root, game.size * game.size)
                selected_policy = visit_policy
                selected_reason = route.reason
                selected_proven = route.proven
                # Multi-defense 3-ply routes intentionally retain their exact
                # tactical target and existing MCTS ranking semantics.
                if route.target_policy is None:
                    selected_policy, selected_reason, selected_proven = (
                        self._vcf_postprocess(
                            game,
                            visit_policy,
                            base_reason=route.reason,
                        )
                    )
                    # Preserve a shorter exact VCF attack.  Otherwise the
                    # optional VCT layer may add proofs or safety filtering.
                    if not selected_proven:
                        selected_policy, selected_reason, selected_proven = (
                            self._vct_postprocess(
                                game,
                                selected_policy,
                                base_reason=selected_reason,
                            )
                        )
                action = _sample_policy(
                    selected_policy, temperatures[index], self.rng
                )
                if not bool(np.any(route.legal == action)):
                    raise RuntimeError(f"MCTS selected illegal root action {action}")
                target_policy = (
                    route.target_policy
                    if route.target_policy is not None
                    else selected_policy
                )
                decisions[index] = SearchDecision(
                    action,
                    target_policy,
                    selected_reason,
                    selected_proven,
                )

        finalized = tuple(decision for decision in decisions if decision is not None)
        if len(finalized) != len(games):
            raise RuntimeError("batched search failed to produce every decision")
        histogram: dict[int, int] = {}
        for batch_size in batch_sizes:
            histogram[batch_size] = histogram.get(batch_size, 0) + 1
        stats = BatchSearchStats(
            positions=len(games),
            direct_positions=len(games) - len(mcts_indices),
            mcts_positions=len(mcts_indices),
            root_batch_size=len(mcts_indices),
            inference_calls=len(batch_sizes),
            evaluated_positions=sum(batch_sizes),
            max_inference_batch_size=max(batch_sizes, default=0),
            mean_inference_batch_size=(
                sum(batch_sizes) / len(batch_sizes) if batch_sizes else 0.0
            ),
            inference_batch_histogram=tuple(sorted(histogram.items())),
        )
        self.last_batch_stats = stats
        return finalized, stats

    def decide(
        self,
        game: GomokuGame,
        *,
        simulations: int | None = None,
        add_noise: bool = False,
        temperature: float = 0.0,
    ) -> SearchDecision:
        """Return one action and its normalized root training target.

        ``temperature=0`` chooses deterministically.  Positive temperatures
        sample from the tactical-uniform or MCTS visit distribution.  Root
        Dirichlet noise is controlled independently with ``add_noise`` so the
        same function serves deterministic play and exploratory self-play.
        """

        decisions, _ = self.decide_batch(
            [game],
            simulations=simulations,
            add_noise=add_noise,
            temperature=temperature,
        )
        return decisions[0]


def search_root(
    model: torch.nn.Module,
    game: GomokuGame,
    config: Config,
    device: torch.device | str,
    *,
    simulations: int | None = None,
    add_noise: bool = False,
    temperature: float = 0.0,
    rng: np.random.Generator | None = None,
    tactical_solver: TacticalSolver | None = None,
    vct_options: VCTRootOptions | None = None,
) -> SearchDecision:
    """One-shot convenience wrapper around :class:`V3RootSearch`."""

    return V3RootSearch(
        model,
        config,
        device,
        rng=rng,
        tactical_solver=tactical_solver,
        vct_options=vct_options,
    ).decide(
        game,
        simulations=simulations,
        add_noise=add_noise,
        temperature=temperature,
    )


__all__ = [
    "BatchSearchStats",
    "SearchDecision",
    "V3RootSearch",
    "VCTRootOptions",
    "search_root",
]

#!/usr/bin/env python3
"""AlphaZero-Lite self-play trainer for 19x19 freestyle Gomoku.

The trainer is intentionally self-contained so it can run unattended on a
single GPU. It uses batched MCTS, a policy/value residual network, a circular
replay buffer, arena gating, compressed replay chunks, and atomic checkpoints.

Architecture / 代码架构
-----------------------
``Config`` defines one reproducible run. ``GomokuGame`` owns rules and the
four-plane state encoding. ``PolicyValueNet`` predicts a policy and value.
``Node`` plus the MCTS functions produce improved move targets. Self-play
feeds ``ReplayBuffer``; ``train_steps`` updates the network; ``arena`` decides
promotion; replay chunks and checkpoints make the run resumable.

``Config`` 保存可复现的训练设置；``GomokuGame`` 负责规则和四平面输入编码；
``PolicyValueNet`` 输出策略与价值；``Node`` 及 MCTS 函数生成更强的落子目标；
自我对弈数据进入 ``ReplayBuffer``，``train_steps`` 更新网络，``arena`` 决定是否
晋级，回放分块与检查点保证训练可恢复。

Key algorithms / 重要算法
-------------------------
The model is a residual policy-value network. Batched MCTS uses PUCT, neural
priors, value backup, root Dirichlet noise, and temperature sampling. Training
minimizes policy cross-entropy plus value mean-squared error with AdamW. A
candidate replaces the champion only after paired arena evaluation; failed
candidates roll back. Checkpoints are written atomically.

模型采用残差策略-价值网络。批量 MCTS 使用 PUCT、神经网络先验、价值回传、根节点
Dirichlet 噪声和温度采样。AdamW 最小化策略交叉熵与价值均方误差之和。候选模型
只有通过交换黑白的竞技场评估才会替换冠军；失败则回滚。检查点采用原子写入。
"""

from __future__ import annotations

import argparse
import copy
import logging
import math
import os
from pathlib import Path
import random
import signal
import time
from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

try:
    from .training_audit import TrainingAudit, add_audit_arguments
except ImportError:  # Support direct ``python train_alphazero.py`` execution.
    from training_audit import TrainingAudit, add_audit_arguments  # type: ignore


BLACK = 1
WHITE = -1
EMPTY = 0
DIRECTIONS = ((1, 0), (0, 1), (1, 1), (1, -1))
STOP_REQUESTED = False


@dataclass
class Config:
    board_size: int = 19
    win_length: int = 5
    channels: int = 96
    residual_blocks: int = 8
    simulations: int = 256
    selfplay_games: int = 128
    parallel_games: int = 32
    inference_batch_per_game: int = 2
    temperature_moves: int = 15
    c_puct: float = 2.0
    dirichlet_alpha: float = 0.10
    dirichlet_epsilon: float = 0.25
    candidate_radius: int = 2
    heuristic_prior_weight: float = 0.35
    # Conservative root-only VCF guard used by the shared V3 search.  It
    # checks only the most relevant MCTS candidates and has a hard per-query
    # budget, so UNKNOWN_BUDGET remains playable instead of being mistaken for
    # a proven loss.
    vcf_root_filter: bool = True
    vcf_attack_priority: bool = True
    vcf_root_candidates: int = 8
    vcf_min_policy: float = 0.01
    vcf_max_plies: int = 5
    vcf_max_nodes: int = 5_000
    vcf_time_ms: float = 25.0
    replay_capacity: int = 500_000
    min_replay_size: int = 2_048
    batch_size: int = 512
    train_steps: int = 200
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    eval_interval: int = 5
    arena_games: int = 40
    arena_simulations: int = 192
    arena_opening_plies: int = 4
    promotion_score: float = 0.55
    max_iterations: int = 10_000
    seed: int = 20260716
    max_replay_chunks: int = 200


class GomokuGame:
    __slots__ = (
        "size",
        "win_length",
        "board",
        "player",
        "last_action",
        "move_count",
        "terminal",
        "winner",
    )

    def __init__(self, size: int = 19, win_length: int = 5):
        self.size = size
        self.win_length = win_length
        self.board = np.zeros((size, size), dtype=np.int8)
        self.player = BLACK
        self.last_action = -1
        self.move_count = 0
        self.terminal = False
        self.winner = EMPTY

    def clone(self) -> "GomokuGame":
        game = object.__new__(GomokuGame)
        game.size = self.size
        game.win_length = self.win_length
        game.board = self.board.copy()
        game.player = self.player
        game.last_action = self.last_action
        game.move_count = self.move_count
        game.terminal = self.terminal
        game.winner = self.winner
        return game

    def legal_actions(self) -> np.ndarray:
        return np.flatnonzero(self.board.ravel() == EMPTY).astype(np.int32)

    def candidate_actions(self, radius: int = 2) -> np.ndarray:
        """Return relevant local points while preserving every tactical move."""
        if self.move_count == 0:
            center = self.size // 2
            return np.asarray([center * self.size + center], dtype=np.int32)
        occupied = np.argwhere(self.board != EMPTY)
        candidates: set[int] = set()
        for y, x in occupied:
            for candidate_y in range(max(0, y - radius), min(self.size, y + radius + 1)):
                for candidate_x in range(max(0, x - radius), min(self.size, x + radius + 1)):
                    if self.board[candidate_y, candidate_x] == EMPTY:
                        candidates.add(candidate_y * self.size + candidate_x)
        return np.asarray(sorted(candidates), dtype=np.int32)

    def winning_actions(self, player: int, radius: int = 2) -> np.ndarray:
        candidates = self.candidate_actions(radius)
        return np.asarray(
            [int(action) for action in candidates if self.would_win(int(action), player)],
            dtype=np.int32,
        )

    def would_win(self, action: int, player: int) -> bool:
        y, x = divmod(int(action), self.size)
        if self.board[y, x] != EMPTY:
            return False
        for dx, dy in DIRECTIONS:
            count = 1
            for sign in (1, -1):
                nx, ny = x + sign * dx, y + sign * dy
                while (
                    0 <= nx < self.size
                    and 0 <= ny < self.size
                    and self.board[ny, nx] == player
                ):
                    count += 1
                    nx += sign * dx
                    ny += sign * dy
            if count >= self.win_length:
                return True
        return False

    def search_actions(self, radius: int = 2) -> np.ndarray:
        candidates = self.candidate_actions(radius)
        wins = [int(action) for action in candidates if self.would_win(int(action), self.player)]
        if wins:
            return np.asarray(wins, dtype=np.int32)
        forced_blocks = [
            int(action) for action in candidates if self.would_win(int(action), -self.player)
        ]
        if forced_blocks:
            return np.asarray(forced_blocks, dtype=np.int32)
        return candidates

    def move_heuristic(self, action: int, player: int) -> float:
        """Pattern prior for attack and defence; learning still controls Q values."""
        y, x = divmod(int(action), self.size)
        if self.board[y, x] != EMPTY:
            return 0.0

        def shape_score(stone: int) -> float:
            total = 0.0
            for dx, dy in DIRECTIONS:
                length = 1
                open_ends = 0
                for sign in (1, -1):
                    nx, ny = x + sign * dx, y + sign * dy
                    while (
                        0 <= nx < self.size
                        and 0 <= ny < self.size
                        and self.board[ny, nx] == stone
                    ):
                        length += 1
                        nx += sign * dx
                        ny += sign * dy
                    if 0 <= nx < self.size and 0 <= ny < self.size and self.board[ny, nx] == EMPTY:
                        open_ends += 1
                if length >= 5:
                    total += 1_000_000
                elif length == 4:
                    total += 120_000 if open_ends == 2 else 18_000 if open_ends == 1 else 0
                elif length == 3:
                    total += 8_000 if open_ends == 2 else 900 if open_ends == 1 else 0
                elif length == 2:
                    total += 500 if open_ends == 2 else 80 if open_ends == 1 else 0
                elif open_ends == 2:
                    total += 10
            return total

        return shape_score(player) + 1.15 * shape_score(-player)

    def play(self, action: int) -> None:
        if self.terminal:
            raise ValueError("cannot play in a terminal position")
        y, x = divmod(int(action), self.size)
        if self.board[y, x] != EMPTY:
            raise ValueError(f"illegal move: {action}")

        stone = self.player
        self.board[y, x] = stone
        self.last_action = int(action)
        self.move_count += 1
        if self._is_win(x, y, stone):
            self.terminal = True
            self.winner = stone
        elif self.move_count == self.size * self.size:
            self.terminal = True
            self.winner = EMPTY
        self.player = -self.player

    def _is_win(self, x: int, y: int, stone: int) -> bool:
        for dx, dy in DIRECTIONS:
            count = 1
            for sign in (1, -1):
                nx, ny = x + sign * dx, y + sign * dy
                while (
                    0 <= nx < self.size
                    and 0 <= ny < self.size
                    and self.board[ny, nx] == stone
                ):
                    count += 1
                    nx += sign * dx
                    ny += sign * dy
            if count >= self.win_length:
                return True
        return False

    def encode(self) -> np.ndarray:
        planes = np.zeros((4, self.size, self.size), dtype=np.uint8)
        planes[0] = self.board == self.player
        planes[1] = self.board == -self.player
        if self.last_action >= 0:
            y, x = divmod(self.last_action, self.size)
            planes[2, y, x] = 1
        if self.player == BLACK:
            planes[3].fill(1)
        return planes


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual, inplace=True)


class PolicyValueNet(nn.Module):
    def __init__(self, board_size: int, channels: int, blocks: int):
        super().__init__()
        actions = board_size * board_size
        self.stem = nn.Sequential(
            nn.Conv2d(4, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.tower = nn.Sequential(*(ResidualBlock(channels) for _ in range(blocks)))
        self.policy_conv = nn.Conv2d(channels, 2, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * actions, actions)
        self.value_conv = nn.Conv2d(channels, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(actions, 128)
        self.value_fc2 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.tower(self.stem(x))
        policy = F.relu(self.policy_bn(self.policy_conv(x)), inplace=True)
        policy = self.policy_fc(policy.flatten(1))
        value = F.relu(self.value_bn(self.value_conv(x)), inplace=True)
        value = F.relu(self.value_fc1(value.flatten(1)), inplace=True)
        value = torch.tanh(self.value_fc2(value)).squeeze(1)
        return policy, value


class Node:
    __slots__ = (
        "prior",
        "to_play",
        "visit_count",
        "value_sum",
        "children",
        "noise_applied",
    )

    def __init__(self, prior: float, to_play: int):
        self.prior = float(prior)
        self.to_play = int(to_play)
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: dict[int, Node] = {}
        self.noise_applied = False

    @property
    def expanded(self) -> bool:
        return bool(self.children)

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


def select_child(node: Node, c_puct: float) -> tuple[int, Node]:
    sqrt_parent = math.sqrt(node.visit_count + 1.0)
    best_score = -float("inf")
    best_action = -1
    best_child: Node | None = None
    for action, child in node.children.items():
        q_value = -child.value
        exploration = c_puct * child.prior * sqrt_parent / (1 + child.visit_count)
        score = q_value + exploration
        if score > best_score:
            best_score = score
            best_action = action
            best_child = child
    if best_child is None:
        raise RuntimeError("selection requested on an unexpanded node")
    return best_action, best_child


def expand_node(
    node: Node,
    game: GomokuGame,
    logits: np.ndarray,
    config: Config | None = None,
) -> None:
    radius = config.candidate_radius if config is not None else 2
    heuristic_weight = config.heuristic_prior_weight if config is not None else 0.35
    legal = game.search_actions(radius)
    if legal.size == 0:
        return
    legal_logits = logits[legal].astype(np.float64)
    if heuristic_weight:
        heuristic = np.asarray(
            [game.move_heuristic(int(action), node.to_play) for action in legal],
            dtype=np.float64,
        )
        legal_logits += heuristic_weight * np.log1p(heuristic)
    legal_logits -= legal_logits.max()
    probabilities = np.exp(legal_logits)
    probabilities /= probabilities.sum()
    next_player = -node.to_play
    node.children = {
        int(action): Node(float(probability), next_player)
        for action, probability in zip(legal, probabilities)
    }


def add_root_noise(node: Node, alpha: float, epsilon: float, rng: np.random.Generator) -> None:
    if node.noise_applied or not node.children:
        return
    actions = list(node.children)
    noise = rng.dirichlet(np.full(len(actions), alpha, dtype=np.float64))
    for action, sample in zip(actions, noise):
        child = node.children[action]
        child.prior = (1.0 - epsilon) * child.prior + epsilon * float(sample)
    node.noise_applied = True


def backup(path: list[Node], value: float, leaf_to_play: int) -> None:
    for node in path:
        node.visit_count += 1
        node.value_sum += value if node.to_play == leaf_to_play else -value


def reserve_path(path: list[Node]) -> None:
    """Apply a virtual loss so one inference batch explores distinct leaves."""
    for node in path:
        node.visit_count += 1
        node.value_sum += 1.0


def release_path(path: list[Node]) -> None:
    for node in path:
        node.visit_count -= 1
        node.value_sum -= 1.0


@torch.inference_mode()
def evaluate_positions(
    model: PolicyValueNet,
    games: list[GomokuGame],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    states = np.stack([game.encode() for game in games])
    tensor = torch.from_numpy(states).to(device=device, dtype=torch.float32)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        logits, values = model(tensor)
    return logits.float().cpu().numpy(), values.float().cpu().numpy()


def run_mcts_batch(
    model: PolicyValueNet,
    games: list[GomokuGame],
    roots: list[Node],
    simulations: int,
    config: Config,
    device: torch.device,
    rng: np.random.Generator,
    add_noise: bool,
) -> None:
    missing = [index for index, root in enumerate(roots) if not root.expanded]
    if missing:
        root_games = [games[index] for index in missing]
        logits_batch, _ = evaluate_positions(model, root_games, device)
        for index, logits in zip(missing, logits_batch):
            expand_node(roots[index], games[index], logits, config)

    if add_noise:
        for root in roots:
            add_root_noise(
                root,
                config.dirichlet_alpha,
                config.dirichlet_epsilon,
                rng,
            )

    completed_simulations = 0
    while completed_simulations < simulations:
        pending_games: list[GomokuGame] = []
        pending_nodes: list[Node] = []
        pending_paths: list[list[Node]] = []

        inference_group = min(
            config.inference_batch_per_game,
            simulations - completed_simulations,
        )
        for _ in range(inference_group):
            for game, root in zip(games, roots):
                simulation_game = game.clone()
                node = root
                path = [node]
                while node.expanded and not simulation_game.terminal:
                    action, node = select_child(node, config.c_puct)
                    simulation_game.play(action)
                    path.append(node)

                if simulation_game.terminal:
                    if simulation_game.winner == EMPTY:
                        terminal_value = 0.0
                    else:
                        terminal_value = (
                            1.0 if simulation_game.winner == simulation_game.player else -1.0
                        )
                    backup(path, terminal_value, simulation_game.player)
                else:
                    reserve_path(path)
                    pending_games.append(simulation_game)
                    pending_nodes.append(node)
                    pending_paths.append(path)

        if pending_games:
            logits_batch, values = evaluate_positions(model, pending_games, device)
            for game, node, path, logits, value in zip(
                pending_games,
                pending_nodes,
                pending_paths,
                logits_batch,
                values,
            ):
                release_path(path)
                if not node.expanded:
                    expand_node(node, game, logits, config)
                backup(path, float(value), game.player)
        completed_simulations += inference_group


def root_policy(root: Node, action_count: int) -> np.ndarray:
    policy = np.zeros(action_count, dtype=np.float32)
    for action, child in root.children.items():
        policy[action] = child.visit_count
    total = float(policy.sum())
    if total > 0:
        policy /= total
    else:
        for action, child in root.children.items():
            policy[action] = child.prior
        policy /= max(float(policy.sum()), 1e-12)
    return policy


def sample_action(
    root: Node,
    move_count: int,
    temperature_moves: int,
    rng: np.random.Generator,
) -> int:
    actions = np.fromiter(root.children.keys(), dtype=np.int32)
    visits = np.array([root.children[int(action)].visit_count for action in actions])
    if move_count >= temperature_moves:
        return int(actions[np.argmax(visits)])
    probabilities = visits.astype(np.float64)
    if probabilities.sum() == 0:
        probabilities = np.array(
            [root.children[int(action)].prior for action in actions],
            dtype=np.float64,
        )
    probabilities /= probabilities.sum()
    return int(rng.choice(actions, p=probabilities))


def generate_selfplay(
    model: PolicyValueNet,
    config: Config,
    device: torch.device,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    model.eval()
    all_states: list[np.ndarray] = []
    all_policies: list[np.ndarray] = []
    all_values: list[float] = []
    results = {"black": 0, "white": 0, "draw": 0}
    completed = 0
    started_at = time.time()
    slot_count = min(config.parallel_games, config.selfplay_games)
    if slot_count <= 0:
        raise ValueError("selfplay_games and parallel_games must be positive")
    games = [GomokuGame(config.board_size, config.win_length) for _ in range(slot_count)]
    roots = [Node(1.0, BLACK) for _ in range(slot_count)]
    histories: list[list[tuple[np.ndarray, np.ndarray, int]]] = [
        [] for _ in range(slot_count)
    ]
    active = list(range(slot_count))
    launched = slot_count
    rounds = 0

    # Refill a slot as soon as its game ends. This keeps inference batches full
    # instead of making an entire wave wait for one unusually long game.
    while active:
        active_games = [games[index] for index in active]
        active_roots = [roots[index] for index in active]
        run_mcts_batch(
            model,
            active_games,
            active_roots,
            config.simulations,
            config,
            device,
            rng,
            add_noise=True,
        )
        retired: list[int] = []
        for index in active:
            game = games[index]
            root = roots[index]
            policy = root_policy(root, config.board_size * config.board_size)
            histories[index].append((game.encode(), policy, game.player))
            action = sample_action(root, game.move_count, config.temperature_moves, rng)
            game.play(action)
            if not game.terminal:
                roots[index] = root.children[action]
                continue

            completed += 1
            if game.winner == BLACK:
                results["black"] += 1
            elif game.winner == WHITE:
                results["white"] += 1
            else:
                results["draw"] += 1
            for state, target_policy, player in histories[index]:
                all_states.append(state)
                all_policies.append(target_policy)
                if game.winner == EMPTY:
                    all_values.append(0.0)
                else:
                    all_values.append(1.0 if game.winner == player else -1.0)

            if launched < config.selfplay_games:
                games[index] = GomokuGame(config.board_size, config.win_length)
                roots[index] = Node(1.0, BLACK)
                histories[index] = []
                launched += 1
            else:
                retired.append(index)
        if retired:
            retired_set = set(retired)
            active = [index for index in active if index not in retired_set]
        rounds += 1
        if rounds % 10 == 0:
            logging.info(
                "selfplay progress: completed=%d/%d launched=%d active=%d rounds=%d positions=%d",
                completed,
                config.selfplay_games,
                launched,
                len(active),
                rounds,
                len(all_states),
            )

    elapsed = time.time() - started_at
    states = np.stack(all_states).astype(np.uint8, copy=False)
    policies = np.stack(all_policies).astype(np.float16, copy=False)
    values = np.asarray(all_values, dtype=np.int8)
    stats = {
        "games": float(config.selfplay_games),
        "positions": float(len(states)),
        "seconds": elapsed,
        "positions_per_second": len(states) / max(elapsed, 1e-9),
        "black_wins": float(results["black"]),
        "white_wins": float(results["white"]),
        "draws": float(results["draw"]),
    }
    return states, policies, values, stats


class ReplayBuffer:
    def __init__(self, capacity: int, board_size: int, seed: int):
        self.capacity = capacity
        self.board_size = board_size
        self.states = np.empty((capacity, 4, board_size, board_size), dtype=np.uint8)
        self.policies = np.empty((capacity, board_size * board_size), dtype=np.float16)
        self.values = np.empty(capacity, dtype=np.int8)
        self.position = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def add(self, states: np.ndarray, policies: np.ndarray, values: np.ndarray) -> None:
        count = len(states)
        if count >= self.capacity:
            states = states[-self.capacity :]
            policies = policies[-self.capacity :]
            values = values[-self.capacity :]
            count = self.capacity
        first = min(count, self.capacity - self.position)
        self.states[self.position : self.position + first] = states[:first]
        self.policies[self.position : self.position + first] = policies[:first]
        self.values[self.position : self.position + first] = values[:first]
        remaining = count - first
        if remaining:
            self.states[:remaining] = states[first:]
            self.policies[:remaining] = policies[first:]
            self.values[:remaining] = values[first:]
        self.position = (self.position + count) % self.capacity
        self.size = min(self.capacity, self.size + count)

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        indices = self.rng.integers(0, self.size, size=batch_size)
        states = self.states[indices].copy()
        policies = self.policies[indices].astype(np.float32)
        values = self.values[indices].astype(np.float32)
        return augment_batch(states, policies, self.rng), policies, values


def augment_batch(
    states: np.ndarray,
    policies: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    size = states.shape[-1]
    for index in range(len(states)):
        rotations = int(rng.integers(0, 4))
        flip = bool(rng.integers(0, 2))
        state = np.rot90(states[index], rotations, axes=(-2, -1))
        policy = np.rot90(policies[index].reshape(size, size), rotations)
        if flip:
            state = np.flip(state, axis=-1)
            policy = np.flip(policy, axis=-1)
        states[index] = np.ascontiguousarray(state)
        policies[index] = np.ascontiguousarray(policy).reshape(-1)
    return states


def train_steps(
    model: PolicyValueNet,
    optimizer: torch.optim.Optimizer,
    replay: ReplayBuffer,
    config: Config,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0}
    started_at = time.time()
    for step in range(1, config.train_steps + 1):
        states, target_policies, target_values = replay.sample(config.batch_size)
        state_tensor = torch.from_numpy(states).to(device=device, dtype=torch.float32)
        policy_tensor = torch.from_numpy(target_policies).to(device=device)
        value_tensor = torch.from_numpy(target_values).to(device=device)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits, values = model(state_tensor)
            policy_loss = -(policy_tensor * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
            value_loss = F.mse_loss(values, value_tensor)
            loss = policy_loss + value_loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        totals["loss"] += float(loss.detach())
        totals["policy_loss"] += float(policy_loss.detach())
        totals["value_loss"] += float(value_loss.detach())
        if step % 50 == 0:
            logging.info(
                "training step %d/%d loss=%.4f policy=%.4f value=%.4f",
                step,
                config.train_steps,
                totals["loss"] / step,
                totals["policy_loss"] / step,
                totals["value_loss"] / step,
            )
    elapsed = time.time() - started_at
    for key in totals:
        totals[key] /= config.train_steps
    totals["seconds"] = elapsed
    totals["steps_per_second"] = config.train_steps / max(elapsed, 1e-9)
    return totals


def arena(
    candidate: PolicyValueNet,
    champion: PolicyValueNet,
    config: Config,
    device: torch.device,
    rng: np.random.Generator,
) -> dict[str, float]:
    candidate.eval()
    champion.eval()
    games: list[GomokuGame] = []
    candidate_colors: list[bool] = []
    pair_count = (config.arena_games + 1) // 2
    for _ in range(pair_count):
        opening = GomokuGame(config.board_size, config.win_length)
        for _ply in range(config.arena_opening_plies):
            choices = opening.candidate_actions(config.candidate_radius)
            opening.play(int(rng.choice(choices)))
            if opening.terminal:
                break
        games.append(opening.clone())
        candidate_colors.append(True)
        if len(games) < config.arena_games:
            games.append(opening.clone())
            candidate_colors.append(False)
    candidate_is_black = np.asarray(candidate_colors, dtype=bool)
    active = list(range(config.arena_games))

    while active:
        candidate_turn: list[int] = []
        champion_turn: list[int] = []
        for index in active:
            game = games[index]
            uses_candidate = (
                game.player == BLACK and candidate_is_black[index]
            ) or (
                game.player == WHITE and not candidate_is_black[index]
            )
            (candidate_turn if uses_candidate else champion_turn).append(index)

        for indices, model in ((candidate_turn, candidate), (champion_turn, champion)):
            if not indices:
                continue
            selected_games = [games[index] for index in indices]
            roots = [Node(1.0, game.player) for game in selected_games]
            run_mcts_batch(
                model,
                selected_games,
                roots,
                config.arena_simulations,
                config,
                device,
                rng,
                add_noise=False,
            )
            for index, root in zip(indices, roots):
                visits = {
                    action: child.visit_count for action, child in root.children.items()
                }
                action = max(visits, key=visits.get)
                games[index].play(action)
        active = [index for index in active if not games[index].terminal]

    wins = losses = draws = 0
    for index, game in enumerate(games):
        candidate_color = BLACK if candidate_is_black[index] else WHITE
        if game.winner == EMPTY:
            draws += 1
        elif game.winner == candidate_color:
            wins += 1
        else:
            losses += 1
    score = (wins + 0.5 * draws) / config.arena_games
    return {"wins": wins, "losses": losses, "draws": draws, "score": score}


def save_replay_chunk(
    data_dir: Path,
    iteration: int,
    states: np.ndarray,
    policies: np.ndarray,
    values: np.ndarray,
    max_chunks: int,
) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    destination = data_dir / f"replay_{iteration:06d}.npz"
    temporary = destination.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, states=states, policies=policies, values=values)
    os.replace(temporary, destination)
    chunks = sorted(data_dir.glob("replay_*.npz"))
    for old_chunk in chunks[:-max_chunks]:
        old_chunk.unlink(missing_ok=True)


def load_replay_chunks(data_dir: Path, replay: ReplayBuffer, max_chunks: int) -> int:
    loaded = 0
    chunks = sorted(data_dir.glob("replay_*.npz"))[-max_chunks:]
    for chunk in chunks:
        try:
            with np.load(chunk) as data:
                replay.add(data["states"], data["policies"], data["values"])
                loaded += len(data["values"])
        except Exception:
            logging.exception("failed to load replay chunk %s", chunk)
    return loaded


def atomic_torch_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def save_checkpoint(
    path: Path,
    iteration: int,
    train_model: PolicyValueNet,
    best_model: PolicyValueNet,
    optimizer: torch.optim.Optimizer,
    config: Config,
    replay_size: int,
) -> None:
    atomic_torch_save(
        {
            "iteration": iteration,
            "train_model": train_model.state_dict(),
            "best_model": best_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": asdict(config),
            "replay_size": replay_size,
        },
        path,
    )


def configure_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "metrics.log", encoding="utf-8"),
        ],
    )


def handle_stop(signum: int, _frame: object) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    logging.warning("received signal %s; stopping after the current iteration", signum)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("/root/autodl-tmp/gomoku_az/run"))
    parser.add_argument("--simulations", type=int, default=None)
    parser.add_argument("--selfplay-games", type=int, default=None)
    parser.add_argument("--parallel-games", type=int, default=None)
    parser.add_argument("--inference-batch-per-game", type=int, default=None)
    parser.add_argument("--train-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--no-arena", action="store_true")
    add_audit_arguments(parser)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audit = TrainingAudit.from_namespace(
        args,
        trainer="train_alphazero",
        config=vars(args),
    )
    config = Config()
    for argument, attribute in (
        (args.simulations, "simulations"),
        (args.selfplay_games, "selfplay_games"),
        (args.parallel_games, "parallel_games"),
        (args.inference_batch_per_game, "inference_batch_per_game"),
        (args.train_steps, "train_steps"),
        (args.batch_size, "batch_size"),
        (args.max_iterations, "max_iterations"),
    ):
        if argument is not None:
            setattr(config, attribute, argument)
    if args.no_arena:
        config.eval_interval = 0

    configure_logging(args.output)
    audit.event("phase", {"name": "initialization", "state": "started"}, force=True)
    signal.signal(signal.SIGTERM, handle_stop)
    signal.signal(signal.SIGINT, handle_stop)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this training configuration")
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    logging.info("starting AlphaZero-Lite Gomoku training")
    logging.info("device=%s", torch.cuda.get_device_name(0))
    logging.info("torch=%s cuda=%s", torch.__version__, torch.version.cuda)
    logging.info("config=%s", asdict(config))

    train_model = PolicyValueNet(
        config.board_size,
        config.channels,
        config.residual_blocks,
    ).to(device)
    best_model = copy.deepcopy(train_model).to(device).eval()
    optimizer = torch.optim.AdamW(
        train_model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    parameter_count = sum(parameter.numel() for parameter in train_model.parameters())
    logging.info("model parameters=%d", parameter_count)

    replay = ReplayBuffer(config.replay_capacity, config.board_size, config.seed)
    data_dir = args.output / "replay"
    checkpoint_path = args.output / "latest.pt"
    start_iteration = 1
    if checkpoint_path.exists():
        audit.record_artifact(checkpoint_path, role="resume_checkpoint")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        train_model.load_state_dict(checkpoint["train_model"])
        best_model.load_state_dict(checkpoint["best_model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_iteration = int(checkpoint["iteration"]) + 1
        logging.info("resumed checkpoint at iteration %d", start_iteration - 1)
    elif args.init_checkpoint is not None:
        audit.record_artifact(args.init_checkpoint, role="input_checkpoint")
        checkpoint = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        source_state = checkpoint.get("best_model", checkpoint.get("train_model"))
        train_model.load_state_dict(source_state)
        best_model.load_state_dict(source_state)
        logging.info("initialized both models from %s", args.init_checkpoint)

    loaded = load_replay_chunks(data_dir, replay, config.max_replay_chunks)
    logging.info("loaded replay positions=%d buffer_size=%d", loaded, replay.size)
    save_checkpoint(
        checkpoint_path,
        start_iteration - 1,
        train_model,
        best_model,
        optimizer,
        config,
        replay.size,
    )
    logging.info("initial checkpoint ready at %s", checkpoint_path)
    audit.record_artifact(checkpoint_path, role="training_checkpoint_initialized")

    audit.event("phase", {"name": "selfplay_training", "state": "started"}, force=True)
    completed_iteration = start_iteration - 1
    for iteration in range(start_iteration, config.max_iterations + 1):
        if audit.check_control():
            audit.finish(
                "stopped",
                {"iteration": completed_iteration, "reason": "control request"},
            )
            logging.info("training stopped by audit control")
            return 0
        iteration_start = time.time()
        logging.info("iteration %d: self-play started", iteration)
        states, policies, values, selfplay_stats = generate_selfplay(
            best_model,
            config,
            device,
            rng,
        )
        replay.add(states, policies, values)
        save_replay_chunk(
            data_dir,
            iteration,
            states,
            policies,
            values,
            config.max_replay_chunks,
        )
        logging.info("iteration %d: self-play stats=%s replay=%d", iteration, selfplay_stats, replay.size)

        if replay.size >= config.min_replay_size:
            training_stats = train_steps(train_model, optimizer, replay, config, device)
            logging.info("iteration %d: training stats=%s", iteration, training_stats)
        else:
            logging.info(
                "iteration %d: waiting for replay warm-up (%d/%d)",
                iteration,
                replay.size,
                config.min_replay_size,
            )

        if config.eval_interval and iteration % config.eval_interval == 0:
            logging.info("iteration %d: arena started", iteration)
            arena_stats = arena(train_model, best_model, config, device, rng)
            logging.info("iteration %d: arena stats=%s", iteration, arena_stats)
            if arena_stats["score"] >= config.promotion_score:
                best_model.load_state_dict(train_model.state_dict())
                logging.info("iteration %d: candidate promoted", iteration)
            else:
                train_model.load_state_dict(best_model.state_dict())
                optimizer = torch.optim.AdamW(
                    train_model.parameters(),
                    lr=config.learning_rate,
                    weight_decay=config.weight_decay,
                )
                logging.info("iteration %d: candidate rejected and reset", iteration)

        save_checkpoint(
            checkpoint_path,
            iteration,
            train_model,
            best_model,
            optimizer,
            config,
            replay.size,
        )
        logging.info(
            "iteration %d complete in %.1fs; checkpoint=%s",
            iteration,
            time.time() - iteration_start,
            checkpoint_path,
        )
        audit.event(
            "iteration_metrics",
            {
                "iteration": iteration,
                "selfplay": selfplay_stats,
                "training": training_stats if replay.size >= config.min_replay_size else None,
                "arena": arena_stats
                if config.eval_interval and iteration % config.eval_interval == 0
                else None,
                "replay_size": replay.size,
            },
        )
        audit.record_artifact(checkpoint_path, role="training_checkpoint")
        completed_iteration = iteration
        if STOP_REQUESTED:
            break

    logging.info("training stopped cleanly")
    audit.finish(
        "stopped" if STOP_REQUESTED else "completed",
        {"iteration": completed_iteration},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

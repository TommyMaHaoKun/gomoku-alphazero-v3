#!/usr/bin/env python3
"""Tactical and match-based acceptance tests for Gomoku checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train_alphazero import (
    BLACK,
    EMPTY,
    WHITE,
    Config,
    GomokuGame,
    Node,
    PolicyValueNet,
    run_mcts_batch,
)


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[PolicyValueNet, Config, int]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = Config(**checkpoint["config"])
    model = PolicyValueNet(config.board_size, config.channels, config.residual_blocks).to(device)
    model.load_state_dict(checkpoint["best_model"])
    model.eval()
    return model, config, int(checkpoint.get("iteration", -1))


def model_actions(
    model: PolicyValueNet,
    games: list[GomokuGame],
    config: Config,
    simulations: int,
    device: torch.device,
    rng: np.random.Generator,
) -> list[int]:
    roots = [Node(1.0, game.player) for game in games]
    run_mcts_batch(
        model,
        games,
        roots,
        simulations,
        config,
        device,
        rng,
        add_noise=False,
    )
    return [
        max(root.children, key=lambda action: root.children[action].visit_count)
        for root in roots
    ]


def heuristic_action(game: GomokuGame, config: Config, rng: np.random.Generator) -> int:
    candidates = game.search_actions(config.candidate_radius)
    center = game.size // 2
    scores = []
    for action in candidates:
        y, x = divmod(int(action), game.size)
        value = game.move_heuristic(int(action), game.player)
        value += max(0, game.size - abs(x - center) - abs(y - center))
        value += float(rng.uniform(0.0, 0.01))
        scores.append(value)
    return int(candidates[int(np.argmax(scores))])


def tactical_suite(
    model: PolicyValueNet,
    config: Config,
    device: torch.device,
    rng: np.random.Generator,
) -> dict[str, int]:
    cases: list[tuple[GomokuGame, set[int], str]] = []
    directions = ((1, 0), (0, 1), (1, 1), (1, -1))
    starts = ((5, 5), (8, 7))

    for player in (BLACK, WHITE):
        for dx, dy in directions:
            for start_x, start_y in starts:
                if dy == -1:
                    start_y += 5
                # Four consecutive stones: win now or block now.
                game = GomokuGame(config.board_size, config.win_length)
                stone = player
                for offset in range(4):
                    x = start_x + offset * dx
                    y = start_y + offset * dy
                    game.board[y, x] = stone
                game.move_count = 4
                game.player = player
                expected = set(map(int, game.winning_actions(player, config.candidate_radius)))
                cases.append((game, expected, "win"))

                block = game.clone()
                block.player = -player
                expected_block = set(map(int, block.winning_actions(player, config.candidate_radius)))
                cases.append((block, expected_block, "block"))

                # Broken four XX_XX: the gap is the only tactical answer.
                broken = GomokuGame(config.board_size, config.win_length)
                for offset in (0, 1, 3, 4):
                    x = start_x + offset * dx
                    y = start_y + offset * dy
                    broken.board[y, x] = stone
                broken.move_count = 4
                broken.player = player
                expected_broken = set(map(int, broken.winning_actions(player, config.candidate_radius)))
                cases.append((broken, expected_broken, "broken_win"))

                broken_block = broken.clone()
                broken_block.player = -player
                expected_broken_block = set(
                    map(int, broken_block.winning_actions(player, config.candidate_radius))
                )
                cases.append((broken_block, expected_broken_block, "broken_block"))

    passed = 0
    failures: dict[str, int] = {}
    batch_size = 16
    for offset in range(0, len(cases), batch_size):
        batch = cases[offset : offset + batch_size]
        actions = model_actions(
            model,
            [case[0] for case in batch],
            config,
            simulations=8,
            device=device,
            rng=rng,
        )
        for action, (_game, expected, name) in zip(actions, batch):
            if action in expected:
                passed += 1
            else:
                failures[name] = failures.get(name, 0) + 1
    return {"passed": passed, "total": len(cases), "failures": failures}


def benchmark_matches(
    model: PolicyValueNet,
    config: Config,
    game_count: int,
    simulations: int,
    device: torch.device,
    rng: np.random.Generator,
) -> dict[str, float]:
    games: list[GomokuGame] = []
    model_is_black: list[bool] = []
    for _ in range((game_count + 1) // 2):
        opening = GomokuGame(config.board_size, config.win_length)
        for _ply in range(config.arena_opening_plies):
            action = int(rng.choice(opening.candidate_actions(config.candidate_radius)))
            opening.play(action)
        games.append(opening.clone())
        model_is_black.append(True)
        if len(games) < game_count:
            games.append(opening.clone())
            model_is_black.append(False)

    active = list(range(len(games)))
    while active:
        model_turn: list[int] = []
        heuristic_turn: list[int] = []
        for index in active:
            uses_model = (games[index].player == BLACK) == model_is_black[index]
            (model_turn if uses_model else heuristic_turn).append(index)
        if model_turn:
            actions = model_actions(
                model,
                [games[index] for index in model_turn],
                config,
                simulations,
                device,
                rng,
            )
            for index, action in zip(model_turn, actions):
                games[index].play(action)
        for index in heuristic_turn:
            games[index].play(heuristic_action(games[index], config, rng))
        active = [index for index in active if not games[index].terminal]

    wins = losses = draws = 0
    color_stats = {
        "black": {"wins": 0, "losses": 0, "draws": 0},
        "white": {"wins": 0, "losses": 0, "draws": 0},
    }
    for index, game in enumerate(games):
        model_color = BLACK if model_is_black[index] else WHITE
        color_name = "black" if model_color == BLACK else "white"
        if game.winner == EMPTY:
            draws += 1
            color_stats[color_name]["draws"] += 1
        elif game.winner == model_color:
            wins += 1
            color_stats[color_name]["wins"] += 1
        else:
            losses += 1
            color_stats[color_name]["losses"] += 1
    black_games = sum(color_stats["black"].values())
    white_games = sum(color_stats["white"].values())
    black_score = (
        color_stats["black"]["wins"] + 0.5 * color_stats["black"]["draws"]
    ) / max(black_games, 1)
    white_score = (
        color_stats["white"]["wins"] + 0.5 * color_stats["white"]["draws"]
    ) / max(white_games, 1)
    return {
        "games": game_count,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "score": (wins + 0.5 * draws) / game_count,
        "black": {**color_stats["black"], "score": black_score},
        "white": {**color_stats["white"], "score": white_score},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--simulations", type=int, default=256)
    parser.add_argument("--minimum-score", type=float, default=0.65)
    parser.add_argument("--minimum-color-score", type=float, default=0.55)
    parser.add_argument("--seed", type=int, default=20260717)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    model, config, iteration = load_model(args.checkpoint, device)
    tactical = tactical_suite(model, config, device, rng)
    matches = benchmark_matches(
        model,
        config,
        args.games,
        args.simulations,
        device,
        rng,
    )
    result = {
        "checkpoint": str(args.checkpoint),
        "iteration": iteration,
        "device": str(device),
        "tactical": tactical,
        "matches": matches,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if tactical["passed"] != tactical["total"]:
        return 2
    if matches["score"] < args.minimum_score:
        return 3
    if min(matches["black"]["score"], matches["white"]["score"]) < args.minimum_color_score:
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

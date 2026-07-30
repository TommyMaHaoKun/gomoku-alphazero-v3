"""Inference adapter for using a trained AlphaZero checkpoint in the game UI."""

from __future__ import annotations

import os
from pathlib import Path
import threading
import warnings

import numpy as np
import torch

from .tactical_solver import TacticalSolver
from .train_alphazero import (
    BLACK,
    WHITE,
    Config,
    GomokuGame,
    PolicyValueNet,
)
from .v3_search import V3RootSearch


DEFAULT_PLAY_SIMULATIONS = 256
PLAY_SIMULATIONS_ENV = "GOMOKU_MCTS_SIMULATIONS"
MODEL_NAME = "Gargantua"


def configured_play_simulations(simulations: int | None = None) -> int:
    """Resolve the desktop MCTS budget, with a safe 256-search default."""
    if simulations is not None:
        value = int(simulations)
        if value <= 0:
            raise ValueError(f"simulations must be positive, got {value}")
        return value

    raw_value = os.environ.get(PLAY_SIMULATIONS_ENV)
    if raw_value is None or not raw_value.strip():
        return DEFAULT_PLAY_SIMULATIONS
    try:
        value = int(raw_value)
        if value <= 0:
            raise ValueError
    except ValueError:
        warnings.warn(
            f"Ignoring invalid {PLAY_SIMULATIONS_ENV}={raw_value!r}; "
            f"using {DEFAULT_PLAY_SIMULATIONS} simulations.",
            RuntimeWarning,
            stacklevel=2,
        )
        return DEFAULT_PLAY_SIMULATIONS
    return value


class AlphaZeroGomokuAgent:
    """Load the champion network and select moves with deterministic MCTS."""

    def __init__(
        self,
        checkpoint_path: Path | str,
        simulations: int | None = None,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {self.checkpoint_path}")

        # Keep one core available for the Pygame event/rendering thread.
        torch.set_num_threads(max(1, min(8, (os.cpu_count() or 4) - 1)))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        raw_config = checkpoint["config"]
        format_version = int(checkpoint.get("format_version", 0))
        self.training_version = (
            "v3"
            if format_version >= 3
            else "v2"
            if "candidate_radius" in raw_config
            else "v1"
        )
        self.config = Config(**raw_config)
        self.simulations = configured_play_simulations(simulations)
        self.iteration = int(checkpoint.get("iteration", -1))
        self.model = PolicyValueNet(
            self.config.board_size,
            self.config.channels,
            self.config.residual_blocks,
        ).to(self.device)
        self.model.load_state_dict(checkpoint["best_model"])
        self.model.eval()
        self.rng = np.random.default_rng(20260717)
        self._search_lock = threading.Lock()
        self.tactical_solver = TacticalSolver(
            board_size=self.config.board_size,
            win_length=self.config.win_length,
        )
        self.root_search = V3RootSearch(
            self.model,
            self.config,
            self.device,
            rng=self.rng,
            tactical_solver=self.tactical_solver,
        )
        self.last_decision_reason: str | None = None

    @property
    def model_label(self) -> str:
        return f"{MODEL_NAME} {self.training_version.upper()} i{self.iteration}"

    @property
    def search_label(self) -> str:
        device_name = "GPU" if self.device.type == "cuda" else "CPU"
        return f"{self.simulations} MCTS ({device_name})"

    @property
    def label(self) -> str:
        return f"{self.model_label} {self.search_label}"

    def _convert_game(
        self,
        grid: list[list[int]],
        last_move: tuple[int, int] | None,
        player: int,
    ) -> GomokuGame:
        game = GomokuGame(self.config.board_size, self.config.win_length)
        array = np.asarray(grid, dtype=np.int8)
        game.board[array == 1] = BLACK
        game.board[array == 2] = WHITE
        game.player = player
        game.move_count = int(np.count_nonzero(game.board))
        if last_move is not None:
            game.last_action = last_move[1] * game.size + last_move[0]
        return game

    def choose_move(
        self,
        grid: list[list[int]],
        last_move: tuple[int, int] | None = None,
        ai_color: int = 1,
    ) -> tuple[int, int] | None:
        with self._search_lock:
            self.last_decision_reason = None
            if ai_color not in (1, 2):
                raise ValueError(f"ai_color must be 1 (black) or 2 (white), got {ai_color}")
            player = BLACK if ai_color == 1 else WHITE
            game = self._convert_game(grid, last_move, player)
            legal = game.legal_actions()
            if legal.size == 0:
                self.last_decision_reason = "no_legal_moves"
                return None

            # Desktop play and V3 self-play now share the exact same tactical
            # routing, candidate handling, priors, MCTS, and tie-breaking.
            decision = self.root_search.decide(
                game,
                simulations=self.simulations,
                add_noise=False,
                temperature=0.0,
            )
            action = decision.action
            self.last_decision_reason = decision.reason

            y, x = divmod(int(action), game.size)
            return x, y

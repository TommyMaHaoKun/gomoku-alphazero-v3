"""Persist desktop games as AlphaZero-compatible replay data.

Every move is stored as the position before the move, a one-hot policy target
for the move that was actually played, and the final result from the current
player's viewpoint.  The five core arrays intentionally match V3 self-play
chunks so a reviewed game can be loaded by the existing replay code.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Sequence
import uuid

import numpy as np

BLACK = 1
WHITE = -1
EMPTY = 0
UI_BLACK = 1
UI_WHITE = 2
UI_DRAW = 0
SCHEMA_VERSION = 1
DIRECTIONS = ((1, 0), (0, 1), (1, 1), (1, -1))


class _ReplayGame:
    """Small dependency-free implementation of the training board contract."""

    def __init__(self, size: int = 19, win_length: int = 5):
        self.size = size
        self.win_length = win_length
        self.board = np.zeros((size, size), dtype=np.int8)
        self.player = BLACK
        self.last_action = -1
        self.move_count = 0
        self.terminal = False
        self.winner = EMPTY

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

    def play(self, action: int) -> None:
        y, x = divmod(int(action), self.size)
        if self.terminal or self.board[y, x] != EMPTY:
            raise ValueError(f"illegal replay action {action}")
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


@dataclass(frozen=True)
class SavedGame:
    """Paths created for one archived desktop game."""

    game_id: str
    replay_path: Path
    metadata_path: Path
    pending_replay_path: Path | None
    pending_metadata_path: Path | None


class GameReplayLogger:
    """Write completed and interrupted desktop games to replay archives."""

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.all_games_dir = self.root / "all_games"
        self.pending_training_dir = self.root / "pending_training" / "ai_losses"

    @staticmethod
    def _training_player(ui_color: int) -> int:
        if ui_color == UI_BLACK:
            return BLACK
        if ui_color == UI_WHITE:
            return WHITE
        raise ValueError(f"invalid stone color {ui_color!r}")

    @staticmethod
    def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp.npz")
        try:
            np.savez_compressed(temporary, **arrays)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _atomic_json(path: Path, payload: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _atomic_copy(source: Path, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(
            f".{destination.name}.{uuid.uuid4().hex}.tmp"
        )
        try:
            shutil.copyfile(source, temporary)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _build_replay(
        self,
        moves: Sequence[tuple[int, int, int]],
        winner: int | None,
        game_id: str,
    ) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
        if not moves:
            raise ValueError("cannot archive a game with no moves")
        if winner not in (None, UI_DRAW, UI_BLACK, UI_WHITE):
            raise ValueError(f"invalid winner {winner!r}")

        game = _ReplayGame()
        states: list[np.ndarray] = []
        policies: list[np.ndarray] = []
        players: list[int] = []
        actions: list[int] = []
        move_records: list[dict[str, object]] = []
        action_count = game.size * game.size

        for move_number, raw_move in enumerate(moves, start=1):
            if len(raw_move) != 3:
                raise ValueError(f"move {move_number} must contain x, y, and color")
            x, y, ui_color = map(int, raw_move)
            if not (0 <= x < game.size and 0 <= y < game.size):
                raise ValueError(f"move {move_number} is outside the board: {(x, y)}")
            player = self._training_player(ui_color)
            if game.terminal:
                raise ValueError(f"move {move_number} appears after the game ended")
            if game.player != player:
                raise ValueError(f"move {move_number} violates alternating color order")
            action = y * game.size + x
            if game.board[y, x] != EMPTY:
                raise ValueError(f"move {move_number} repeats occupied point {(x, y)}")

            states.append(game.encode())
            policy = np.zeros(action_count, dtype=np.float16)
            policy[action] = 1.0
            policies.append(policy)
            players.append(player)
            actions.append(action)
            move_records.append(
                {
                    "move_number": move_number,
                    "x": x,
                    "y": y,
                    "action": action,
                    "color": "black" if ui_color == UI_BLACK else "white",
                }
            )
            game.play(action)

        if winner is None:
            if game.terminal:
                raise ValueError("an interrupted game cannot already be terminal")
            training_winner = EMPTY
            value_weight = 0.0
        else:
            if not game.terminal:
                raise ValueError("a completed game must end in a win or full-board draw")
            training_winner = (
                EMPTY if winner == UI_DRAW else self._training_player(winner)
            )
            if game.winner != training_winner:
                raise ValueError("reported winner does not match replayed moves")
            value_weight = 1.0

        values = np.asarray(
            [
                0.0
                if training_winner == EMPTY
                else 1.0
                if player == training_winner
                else -1.0
                for player in players
            ],
            dtype=np.float32,
        )
        count = len(states)
        group_ids = np.asarray([game_id] * count)
        arrays = {
            "states": np.stack(states).astype(np.uint8, copy=False),
            "policies": np.stack(policies).astype(np.float16, copy=False),
            "values": values,
            "policy_weights": np.ones(count, dtype=np.float32),
            "value_weights": np.full(count, value_weight, dtype=np.float32),
            "priority": np.ones(count, dtype=np.float32),
            "actions": np.asarray(actions, dtype=np.int16),
            "players": np.asarray(players, dtype=np.int8),
            "move_numbers": np.arange(1, count + 1, dtype=np.int16),
            "game_id": group_ids,
            "group_id": group_ids.copy(),
            "source": np.asarray(["desktop_human_vs_ai"] * count),
            "split": np.asarray(["train"] * count),
        }
        return arrays, move_records

    def record_game(
        self,
        moves: Sequence[tuple[int, int, int]],
        *,
        winner: int | None,
        ai_color: int,
        model_label: str,
        search_label: str = "",
        termination: str,
    ) -> SavedGame:
        """Archive one game and separately enqueue a completed AI loss."""

        if ai_color not in (UI_BLACK, UI_WHITE):
            raise ValueError(f"invalid AI color {ai_color!r}")
        if not termination.strip():
            raise ValueError("termination must be non-empty")

        created_at = datetime.now(timezone.utc)
        timestamp = created_at.strftime("%Y%m%dT%H%M%S_%fZ")
        game_id = f"desktop_{timestamp}_{uuid.uuid4().hex[:8]}"
        arrays, move_records = self._build_replay(moves, winner, game_id)
        ai_lost = winner in (UI_BLACK, UI_WHITE) and winner != ai_color
        if winner is None:
            ai_result = "unfinished"
        elif winner == UI_DRAW:
            ai_result = "draw"
        elif ai_lost:
            ai_result = "loss"
        else:
            ai_result = "win"

        replay_path = self.all_games_dir / f"{game_id}.npz"
        metadata_path = self.all_games_dir / f"{game_id}.json"
        self._atomic_npz(replay_path, arrays)
        metadata: dict[str, object] = {
            "schema": "gargantua_desktop_replay",
            "schema_version": SCHEMA_VERSION,
            "game_id": game_id,
            "created_at_utc": created_at.isoformat(),
            "source": "desktop_human_vs_ai",
            "board_size": 19,
            "win_length": 5,
            "termination": termination,
            "completed": winner is not None,
            "winner": (
                None
                if winner is None
                else "draw"
                if winner == UI_DRAW
                else "black"
                if winner == UI_BLACK
                else "white"
            ),
            "ai_color": "black" if ai_color == UI_BLACK else "white",
            "ai_result": ai_result,
            "eligible_for_pending_training": ai_lost,
            "model_label": str(model_label),
            "search_label": str(search_label),
            "plies": len(moves),
            "replay_file": replay_path.name,
            "replay_sha256": self._sha256(replay_path),
            "core_selfplay_arrays": [
                "states",
                "policies",
                "values",
                "policy_weights",
                "value_weights",
            ],
            "policy_target": "one_hot_played_move",
            "moves": move_records,
        }
        self._atomic_json(metadata_path, metadata)

        pending_replay_path: Path | None = None
        pending_metadata_path: Path | None = None
        if ai_lost:
            pending_replay_path = self.pending_training_dir / replay_path.name
            pending_metadata_path = self.pending_training_dir / metadata_path.name
            self._atomic_copy(replay_path, pending_replay_path)
            self._atomic_copy(metadata_path, pending_metadata_path)

        return SavedGame(
            game_id=game_id,
            replay_path=replay_path,
            metadata_path=metadata_path,
            pending_replay_path=pending_replay_path,
            pending_metadata_path=pending_metadata_path,
        )

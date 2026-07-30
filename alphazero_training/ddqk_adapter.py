"""Headless adapter for the DDQK-CONQUER 6.16 Gomoku engine.

The original DDQK program keeps nearly all search state in module globals and
normally creates a Pygame user interface.  This module imports the original
source without calling its ``main()`` function, translates a normal 19x19
board into the engine's private representation, and exposes a small, stable
API for automated matches.

Public board convention
-----------------------
``grid[y][x]`` is row-major and uses ``0`` for empty, ``1`` for black and
``2`` for white.  ``-1`` is also accepted as white so AlphaZero boards can be
passed directly.  Coordinates passed to and returned by this module are
always ``(x, y)``.

The wrapped engine uses process-wide globals and is not re-entrant.  Therefore
``DDQKAdapter`` is a process singleton and serializes every engine operation.
For parallel evaluation, create one adapter per worker *process*.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import os
from pathlib import Path
import sys
import threading
from types import ModuleType
from typing import Iterable, Sequence
import warnings


BOARD_SIZE = 19
EMPTY = 0
BLACK = 1
WHITE = 2

_SOURCE_FILENAME = "DDQK-CONQUER-6-16-15-fixed.py"
_DEFAULT_SOURCE = (
    Path(r"D:\Tommy\RT\6.16 -发行版（无代码）\6.16 -发行版（无代码）")
    / _SOURCE_FILENAME
)


class DDQKError(RuntimeError):
    """Base exception raised by the adapter."""


class DDQKLoadError(DDQKError):
    """The original DDQK source, its assets, or its native library failed."""


class DDQKAdapter:
    """Process-local singleton wrapper around the original DDQK engine.

    Args:
        source_path: Path to ``DDQK-CONQUER-6-16-15-fixed.py``.  A directory
            containing that file is accepted too.  If omitted, the
            ``DDQK_SOURCE_PATH`` environment variable is checked before the
            known local release path is used.

    The original engine always represents its own stones with ``1`` and the
    opponent's stones with ``2``.  :meth:`choose_move` remaps real black/white
    colours on every call, so one process can benchmark either side safely.
    """

    _instance: DDQKAdapter | None = None
    _instance_guard = threading.Lock()

    def __new__(cls, source_path: os.PathLike[str] | str | None = None) -> DDQKAdapter:
        with cls._instance_guard:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, source_path: os.PathLike[str] | str | None = None) -> None:
        requested_source = self._resolve_source_path(source_path)
        if getattr(self, "_initialized", False):
            if requested_source != self.source_path:
                raise DDQKLoadError(
                    "DDQKAdapter is a process singleton and is already bound to "
                    f"{self.source_path}; start another process to use "
                    f"{requested_source}"
                )
            return

        self._lock = threading.RLock()
        self.source_path = requested_source
        self.release_dir = requested_source.parent
        self.asset_dir = self.release_dir / "img"
        self._dll_directory_handle = None
        self._prepared_player: int | None = None
        self._board = self._empty_board()
        self._history: list[tuple[int, int, int]] = []
        self._last_move: tuple[int, int] | None = None
        self.last_engine_error: Exception | None = None

        self._validate_installation()
        self._engine = self._load_engine_module()
        self._initialized = True
        self.reset()

    @staticmethod
    def _resolve_source_path(
        source_path: os.PathLike[str] | str | None,
    ) -> Path:
        raw_path = source_path or os.environ.get("DDQK_SOURCE_PATH") or _DEFAULT_SOURCE
        path = Path(raw_path).expanduser()
        if path.is_dir():
            path = path / _SOURCE_FILENAME
        return path.resolve()

    def _validate_installation(self) -> None:
        if os.name != "nt":
            raise DDQKLoadError(
                "DDQK 6.16 ships a Windows native library; this adapter must run "
                "in a Windows process"
            )
        if not self.source_path.is_file():
            raise DDQKLoadError(f"DDQK source not found: {self.source_path}")
        required_assets = (
            self.asset_dir / "dll.so",
            self.asset_dir / "guess_data.txt",
            self.asset_dir / "black_calculated_value_19.txt",
            self.asset_dir / "white_calculated_value_19.txt",
        )
        missing = [str(path) for path in required_assets if not path.is_file()]
        if missing:
            raise DDQKLoadError("DDQK assets are missing: " + ", ".join(missing))

    @contextlib.contextmanager
    def _release_working_directory(self):
        """Use the release CWD only while importing, then restore the caller's."""

        old_cwd = Path.cwd()
        os.chdir(self.release_dir)
        try:
            yield
        finally:
            os.chdir(old_cwd)

    def _load_engine_module(self) -> ModuleType:
        # The library is named dll.so but contains Windows-native code.  Keeping
        # this handle alive lets Windows resolve any adjacent dependencies.
        if hasattr(os, "add_dll_directory"):
            try:
                self._dll_directory_handle = os.add_dll_directory(str(self.asset_dir))
            except OSError as exc:
                raise DDQKLoadError(
                    f"could not add DDQK DLL directory {self.asset_dir}: {exc}"
                ) from exc

        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "hide")
        digest = hashlib.sha1(
            str(self.source_path).encode("utf-8"), usedforsecurity=False
        ).hexdigest()[:12]
        module_name = f"_ddqk_headless_{digest}"
        existing = sys.modules.get(module_name)
        if existing is not None:
            return existing

        spec = importlib.util.spec_from_file_location(module_name, self.source_path)
        if spec is None or spec.loader is None:
            raise DDQKLoadError(f"cannot create import spec for {self.source_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            # Importing defines the search functions and loads dll.so, but the
            # source's guarded main() is never called, so no window is created.
            with self._release_working_directory(), contextlib.redirect_stdout(io.StringIO()):
                spec.loader.exec_module(module)
        except Exception as exc:
            sys.modules.pop(module_name, None)
            raise DDQKLoadError(f"failed to load DDQK engine: {exc}") from exc
        return module

    @staticmethod
    def _empty_board() -> list[list[int]]:
        return [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]

    @staticmethod
    def _normalize_player(player: int) -> int:
        value = int(player)
        if value == BLACK:
            return BLACK
        if value in (WHITE, -1):
            return WHITE
        raise ValueError(f"player must be BLACK (1) or WHITE (2/-1), got {player!r}")

    @staticmethod
    def _normalize_cell(cell: int, x: int, y: int) -> int:
        value = int(cell)
        if value == EMPTY:
            return EMPTY
        if value == BLACK:
            return BLACK
        if value in (WHITE, -1):
            return WHITE
        raise ValueError(
            f"grid[{y}][{x}] must be 0, 1, 2, or -1; got {cell!r}"
        )

    @classmethod
    def _coerce_board(cls, grid: Sequence[Sequence[int]]) -> list[list[int]]:
        try:
            rows = list(grid)
        except TypeError as exc:
            raise ValueError("grid must be a 19x19 row-major sequence") from exc
        if len(rows) != BOARD_SIZE:
            raise ValueError(f"grid must have {BOARD_SIZE} rows, got {len(rows)}")
        board: list[list[int]] = []
        for y, row in enumerate(rows):
            try:
                cells = list(row)
            except TypeError as exc:
                raise ValueError(f"grid row {y} is not a sequence") from exc
            if len(cells) != BOARD_SIZE:
                raise ValueError(
                    f"grid row {y} must have {BOARD_SIZE} cells, got {len(cells)}"
                )
            board.append(
                [cls._normalize_cell(cell, x, y) for x, cell in enumerate(cells)]
            )
        return board

    @staticmethod
    def _coerce_coordinate(
        move: Sequence[int] | None,
        *,
        name: str,
    ) -> tuple[int, int] | None:
        if move is None:
            return None
        try:
            values = list(move)
        except TypeError as exc:
            raise ValueError(f"{name} must be an (x, y) coordinate") from exc
        if len(values) != 2:
            raise ValueError(f"{name} must contain exactly two values: (x, y)")
        x, y = int(values[0]), int(values[1])
        if not (0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE):
            raise ValueError(f"{name} is outside the 19x19 board: {(x, y)}")
        return x, y

    def _configure_19x19_engine(self) -> None:
        engine = self._engine
        engine.board_order = BOARD_SIZE
        engine.board_size = 30
        engine.xiestatic = BOARD_SIZE * 2 - 5
        engine.Board_order_5 = BOARD_SIZE - 5
        engine.hugeCoord = engine.hugeCoord19
        engine.hugesecCheck = engine.hugesecCheck19
        engine.largerCheck = engine.largerCheck19
        engine.grid_array_type = engine.c_short * BOARD_SIZE
        engine.grid_ctypes = (engine.grid_array_type * BOARD_SIZE)()
        engine.is_five = engine.lib.is_five19
        engine.c_get_single_chess_assess = engine.lib.get_single_chess_assess19

    def _reset_engine_globals(self) -> None:
        engine = self._engine
        self._configure_19x19_engine()
        engine.STEP = 1
        engine.is_black = False
        engine.current_special_number = 0
        engine.public_grid = self._empty_board()
        engine.huge_may_go = []
        engine.find_suit_hash_list = {}
        engine.value_hash_dict.clear()
        engine.zobrist_calculate_value.clear()
        engine.get_new_is_in_huge_may_go()

    def reset(self) -> None:
        """Reset adapter history and every mutable DDQK search global."""

        with self._lock:
            self._board = self._empty_board()
            self._history = []
            self._last_move = None
            self.last_engine_error = None
            self._prepared_player = None
            self._reset_engine_globals()

    def sync_opening(
        self,
        moves: Iterable[Sequence[int]],
        starting_player: int = BLACK,
    ) -> list[list[int]]:
        """Reset and synchronize an ordered opening.

        Each entry may be ``(x, y)`` (colours alternate from
        ``starting_player``) or ``(x, y, player)``.  Explicit colours must still
        alternate, which catches accidentally malformed benchmark openings.
        A detached board copy is returned for immediate use by a match driver.
        """

        with self._lock:
            self.reset()
            expected_player = self._normalize_player(starting_player)
            board = self._empty_board()
            history: list[tuple[int, int, int]] = []
            for ply, raw_move in enumerate(moves, start=1):
                try:
                    values = list(raw_move)
                except TypeError as exc:
                    raise ValueError(f"opening move {ply} is not a sequence") from exc
                if len(values) not in (2, 3):
                    raise ValueError(
                        f"opening move {ply} must be (x, y) or (x, y, player)"
                    )
                coord = self._coerce_coordinate(values[:2], name=f"opening move {ply}")
                assert coord is not None
                x, y = coord
                player = (
                    expected_player
                    if len(values) == 2
                    else self._normalize_player(values[2])
                )
                if player != expected_player:
                    raise ValueError(
                        f"opening move {ply} has player {player}, expected "
                        f"{expected_player} for alternating play"
                    )
                if board[y][x] != EMPTY:
                    raise ValueError(f"opening move {ply} repeats occupied point {(x, y)}")
                board[y][x] = player
                history.append((x, y, player))
                expected_player = WHITE if expected_player == BLACK else BLACK

            self._board = board
            self._history = history
            self._last_move = (history[-1][0], history[-1][1]) if history else None
            return [row[:] for row in board]

    def _history_board(self) -> list[list[int]] | None:
        board = self._empty_board()
        for x, y, player in self._history:
            if board[y][x] != EMPTY:
                return None
            board[y][x] = player
        return board

    def _adopt_board(
        self,
        board: list[list[int]],
        last_move: tuple[int, int] | None,
    ) -> None:
        """Preserve known move order when the caller only added stones."""

        old_board = self._history_board()
        additions: list[tuple[int, int, int]] = []
        can_extend = old_board is not None
        if can_extend:
            for y in range(BOARD_SIZE):
                for x in range(BOARD_SIZE):
                    old = old_board[y][x]
                    new = board[y][x]
                    if old != EMPTY and old != new:
                        can_extend = False
                        break
                    if old == EMPTY and new != EMPTY:
                        additions.append((x, y, new))
                if not can_extend:
                    break

        if can_extend:
            if last_move is not None:
                lx, ly = last_move
                additions.sort(key=lambda item: (item[0] == lx and item[1] == ly,))
            self._history.extend(additions)
        else:
            rebuilt = [
                (x, y, board[y][x])
                for y in range(BOARD_SIZE)
                for x in range(BOARD_SIZE)
                if board[y][x] != EMPTY
            ]
            if last_move is not None:
                lx, ly = last_move
                rebuilt.sort(key=lambda item: (item[0] == lx and item[1] == ly,))
            self._history = rebuilt
        self._board = [row[:] for row in board]
        self._last_move = last_move

    def _prepare_evaluation_cache(self, player: int) -> None:
        if self._prepared_player == player:
            return
        engine = self._engine
        with contextlib.redirect_stdout(io.StringIO()):
            engine.calculated_value_initial(player == BLACK, str(BOARD_SIZE))
        self._prepared_player = player

    def _sync_engine(self, board: list[list[int]], player: int) -> None:
        engine = self._engine
        self._configure_19x19_engine()
        self._prepare_evaluation_cache(player)
        engine.is_black = player == BLACK
        engine.STEP = sum(cell != EMPTY for row in board for cell in row) + 1
        engine.current_special_number = 0
        engine.public_grid = self._empty_board()
        engine.find_suit_hash_list = {}
        engine.value_hash_dict.clear()

        occupied: set[tuple[int, int]] = set()
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                cell = board[y][x]
                if cell == EMPTY:
                    continue
                mapped = 1 if cell == player else 2
                engine.public_grid[y][x] = mapped
                engine.grid_ctypes[y][x] = mapped
                occupied.add((y, x))
                if mapped == 1:
                    engine.current_special_number ^= engine.zobrist_b[(y, x)]
                else:
                    engine.current_special_number ^= engine.zobrist_w[(y, x)]

        engine.huge_may_go = []
        engine.get_new_is_in_huge_may_go()
        ordered = list(self._history)
        known = {(x, y) for x, y, _ in ordered}
        ordered.extend(
            (x, y, board[y][x])
            for y, x in sorted(occupied)
            if (x, y) not in known
        )
        for x, y, _ in ordered:
            engine.add_may_go(x, y)

        # Played points are removed explicitly in the UI path before search.
        if occupied:
            engine.huge_may_go = [
                pos for pos in engine.huge_may_go if pos not in occupied
            ]
            for pos in occupied:
                engine.is_in_huge_may_go[pos] = 0

    @staticmethod
    def _would_win(board: list[list[int]], x: int, y: int, player: int) -> bool:
        for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
            count = 1
            for direction in (-1, 1):
                nx, ny = x + dx * direction, y + dy * direction
                while (
                    0 <= nx < BOARD_SIZE
                    and 0 <= ny < BOARD_SIZE
                    and board[ny][nx] == player
                ):
                    count += 1
                    nx += dx * direction
                    ny += dy * direction
            if count >= 5:
                return True
        return False

    @classmethod
    def _fallback_legal_move(
        cls,
        board: list[list[int]],
        player: int,
    ) -> tuple[int, int]:
        legal = [
            (x, y)
            for y in range(BOARD_SIZE)
            for x in range(BOARD_SIZE)
            if board[y][x] == EMPTY
        ]
        if not legal:
            raise DDQKError("cannot choose a move on a full board")
        opponent = WHITE if player == BLACK else BLACK
        for candidate_player in (player, opponent):
            for x, y in legal:
                board[y][x] = candidate_player
                wins = cls._would_win(board, x, y, candidate_player)
                board[y][x] = EMPTY
                if wins:
                    return x, y
        center = BOARD_SIZE // 2
        return min(
            legal,
            key=lambda move: (
                (move[0] - center) ** 2 + (move[1] - center) ** 2,
                move[1],
                move[0],
            ),
        )

    def choose_move(
        self,
        grid: Sequence[Sequence[int]] | None,
        player: int,
        last_move: Sequence[int] | None = None,
    ) -> tuple[int, int]:
        """Return a legal DDQK move for ``player`` without opening a UI.

        Args:
            grid: Current 19x19 row-major board.  Pass ``None`` to use the
                board most recently created by :meth:`sync_opening`.
            player: Actual side to move: black ``1`` or white ``2``/``-1``.
            last_move: Opponent's latest ``(x, y)`` move, when available.

        The returned move is also recorded in the adapter's private history;
        the caller's grid is never mutated.  Passing the complete updated grid
        on every turn remains the safest integration pattern.
        """

        with self._lock:
            normalized_player = self._normalize_player(player)
            board = (
                [row[:] for row in self._board]
                if grid is None
                else self._coerce_board(grid)
            )
            coord = self._coerce_coordinate(last_move, name="last_move")
            if coord is not None and board[coord[1]][coord[0]] == EMPTY:
                raise ValueError(f"last_move {coord} points to an empty cell")
            if not any(cell == EMPTY for row in board for cell in row):
                raise DDQKError("cannot choose a move on a full board")

            self._adopt_board(board, coord)
            self._sync_engine(board, normalized_player)
            engine = self._engine
            self.last_engine_error = None

            if all(cell == EMPTY for row in board for cell in row):
                move = (BOARD_SIZE // 2, BOARD_SIZE // 2)
            else:
                lx, ly = coord if coord is not None else (-1, -1)
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        raw_move = engine.go(lx, ly)
                    move = int(raw_move[0]), int(raw_move[1])
                except Exception as exc:  # Original engine is global-heavy.
                    self.last_engine_error = exc
                    warnings.warn(
                        f"DDQK search failed ({exc!r}); using a legal tactical fallback",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    move = self._fallback_legal_move(board, normalized_player)

            x, y = move
            if not (
                0 <= x < BOARD_SIZE
                and 0 <= y < BOARD_SIZE
                and board[y][x] == EMPTY
            ):
                self.last_engine_error = DDQKError(
                    f"DDQK returned illegal move {(x, y)}"
                )
                move = self._fallback_legal_move(board, normalized_player)
                x, y = move

            board[y][x] = normalized_player
            self._board = board
            self._history.append((x, y, normalized_player))
            self._last_move = move
            return move


__all__ = [
    "BLACK",
    "BOARD_SIZE",
    "DDQKAdapter",
    "DDQKError",
    "DDQKLoadError",
    "EMPTY",
    "WHITE",
]

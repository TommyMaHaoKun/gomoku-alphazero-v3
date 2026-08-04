"""Piskvork protocol adapter for the Rapfi Gomoku engine.

The public board convention matches the rest of this project: ``grid[y][x]``
contains ``0`` for empty, ``1`` for black, and ``2`` (or ``-1``) for white.
Coordinates are always ``(x, y)``.  Rapfi is kept alive between calls, while
``BOARD`` rebuilds its position before every search so a failed or interrupted
game cannot leak search state into the next one.
"""

from __future__ import annotations

from collections.abc import Sequence
import os
from pathlib import Path
import queue
import re
import subprocess
import sys
import threading
import time


BOARD_SIZE = 19
EMPTY = 0
BLACK = 1
WHITE = 2
_MOVE_RE = re.compile(r"^\s*(\d+)\s*,\s*(\d+)\s*$")


class RapfiError(RuntimeError):
    """Base error raised by the Rapfi adapter."""


class RapfiProtocolError(RapfiError):
    """Rapfi returned an invalid or contradictory protocol response."""


class RapfiTimeoutError(RapfiError):
    """Rapfi did not finish a protocol operation before its deadline."""


class RapfiAdapter:
    """Persistent, process-local Rapfi subprocess.

    Args:
        engine_path: ``pbrain-rapfi`` executable.  A Python file is also
            accepted for protocol tests and is launched with this interpreter.
        timeout_turn_ms: Piskvork per-move budget advertised to Rapfi.
        max_nodes: Optional deterministic search-node ceiling; zero disables it.
        threads: Search threads used by Rapfi.
        max_memory_mb: Piskvork memory ceiling.
        response_timeout_s: Hard adapter deadline.  By default it is derived
            conservatively from ``timeout_turn_ms``.
    """

    def __init__(
        self,
        engine_path: os.PathLike[str] | str,
        *,
        board_size: int = BOARD_SIZE,
        timeout_turn_ms: int = 1_000,
        timeout_match_ms: int = 86_400_000,
        max_nodes: int = 0,
        threads: int = 1,
        max_memory_mb: int = 1024,
        response_timeout_s: float | None = None,
    ) -> None:
        self.engine_path = Path(engine_path).expanduser().resolve()
        self.board_size = int(board_size)
        self.timeout_turn_ms = int(timeout_turn_ms)
        self.timeout_match_ms = int(timeout_match_ms)
        self.max_nodes = int(max_nodes)
        self.threads = int(threads)
        self.max_memory_mb = int(max_memory_mb)
        self.response_timeout_s = float(
            response_timeout_s
            if response_timeout_s is not None
            else max(15.0, self.timeout_turn_ms / 1000.0 * 4.0 + 5.0)
        )
        if self.board_size != BOARD_SIZE:
            raise ValueError("this project and its checkpoints require a 19x19 board")
        if self.timeout_turn_ms <= 0 or self.timeout_match_ms <= 0:
            raise ValueError("Rapfi time limits must be positive")
        if self.max_nodes < 0 or self.threads <= 0 or self.max_memory_mb <= 0:
            raise ValueError("invalid Rapfi node/thread/memory setting")
        if self.response_timeout_s <= 0:
            raise ValueError("response_timeout_s must be positive")
        if not self.engine_path.is_file():
            raise FileNotFoundError(f"Rapfi executable not found: {self.engine_path}")

        command = [str(self.engine_path)]
        if self.engine_path.suffix.lower() == ".py":
            command.insert(0, sys.executable)
        self._stdout: queue.Queue[str | None] = queue.Queue()
        self._stderr: queue.Queue[str | None] = queue.Queue()
        self.messages: list[str] = []
        self._closed = False
        self._process = subprocess.Popen(
            command,
            cwd=self.engine_path.parent,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert self._process.stdout is not None
        assert self._process.stderr is not None
        self._stdout_thread = threading.Thread(
            target=self._read_stream,
            args=(self._process.stdout, self._stdout),
            daemon=True,
            name="rapfi-stdout",
        )
        self._stderr_thread = threading.Thread(
            target=self._read_stream,
            args=(self._process.stderr, self._stderr),
            daemon=True,
            name="rapfi-stderr",
        )
        self._stdout_thread.start()
        self._stderr_thread.start()
        try:
            self._initialize_protocol()
        except BaseException:
            self.close(force=True)
            raise

    @staticmethod
    def _read_stream(stream, output: queue.Queue[str | None]) -> None:
        try:
            for line in stream:
                output.put(line.rstrip("\r\n"))
        finally:
            output.put(None)

    def _send(self, line: str) -> None:
        if self._closed or self._process.poll() is not None:
            raise RapfiProtocolError(self._death_message("Rapfi is not running"))
        assert self._process.stdin is not None
        try:
            self._process.stdin.write(line + "\n")
            self._process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise RapfiProtocolError(self._death_message("could not write to Rapfi")) from exc

    def _death_message(self, prefix: str) -> str:
        stderr_lines: list[str] = []
        while True:
            try:
                line = self._stderr.get_nowait()
            except queue.Empty:
                break
            if line:
                stderr_lines.append(line)
        suffix = " | stderr: " + " | ".join(stderr_lines[-8:]) if stderr_lines else ""
        return f"{prefix} (exit={self._process.poll()}){suffix}"

    def _next_stdout(self, deadline: float) -> str:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RapfiTimeoutError(
                    self._death_message("timed out waiting for Rapfi response")
                )
            try:
                line = self._stdout.get(timeout=min(remaining, 0.25))
            except queue.Empty:
                if self._process.poll() is not None:
                    raise RapfiProtocolError(self._death_message("Rapfi exited"))
                continue
            if line is None:
                raise RapfiProtocolError(self._death_message("Rapfi closed stdout"))
            stripped = line.strip()
            if stripped:
                return stripped

    def _initialize_protocol(self) -> None:
        self._send(f"START {self.board_size}")
        deadline = time.monotonic() + self.response_timeout_s
        while True:
            line = self._next_stdout(deadline)
            upper = line.upper()
            if upper == "OK":
                break
            if upper.startswith(("ERROR", "UNKNOWN")):
                raise RapfiProtocolError(f"Rapfi rejected START: {line}")
            self.messages.append(line)

        options = (
            ("TIMEOUT_TURN", self.timeout_turn_ms),
            ("TIMEOUT_MATCH", self.timeout_match_ms),
            ("MAX_MEMORY", self.max_memory_mb * 1024 * 1024),
            ("RULE", 0),  # freestyle Gomoku
            ("THREAD_NUM", self.threads),
            ("MAX_NODE", self.max_nodes),
            ("SHOW_DETAIL", 0),
        )
        for name, value in options:
            self._send(f"INFO {name} {value}")

    @staticmethod
    def _normalize_player(player: int) -> int:
        value = int(player)
        if value == BLACK:
            return BLACK
        if value in (WHITE, -1):
            return WHITE
        raise ValueError(f"player must be black 1 or white 2/-1, got {player!r}")

    def _coerce_moves(
        self, moves: Sequence[Sequence[int]], target_player: int
    ) -> tuple[list[tuple[int, int, int]], set[tuple[int, int]]]:
        normalized: list[tuple[int, int, int]] = []
        occupied: set[tuple[int, int]] = set()
        expected = BLACK
        for index, move in enumerate(moves):
            values = list(move)
            if len(values) not in (2, 3):
                raise ValueError(f"move {index} must contain x,y or x,y,player")
            x, y = int(values[0]), int(values[1])
            player = expected if len(values) == 2 else self._normalize_player(int(values[2]))
            if player != expected:
                raise ValueError(
                    f"move {index} has player {player}, expected alternating player {expected}"
                )
            if not (0 <= x < self.board_size and 0 <= y < self.board_size):
                raise ValueError(f"move {index} is outside the board: {(x, y)}")
            if (x, y) in occupied:
                raise ValueError(f"move {index} repeats occupied point {(x, y)}")
            occupied.add((x, y))
            # Piskvork BOARD flags are SELF/OPPONENT relative to the engine,
            # not black/white colours.
            relative_side = 1 if player == target_player else 2
            normalized.append((x, y, relative_side))
            expected = WHITE if expected == BLACK else BLACK
        if expected != target_player:
            raise ValueError(
                f"history has {len(moves)} plies, so player {expected} is to move, "
                f"not requested player {target_player}"
            )
        return normalized, occupied

    def choose_move(
        self,
        moves: Sequence[Sequence[int]],
        player: int,
    ) -> tuple[int, int]:
        """Return Rapfi's legal move for ``player`` after the supplied history."""

        target_player = self._normalize_player(player)
        normalized, occupied = self._coerce_moves(moves, target_player)
        # A previous search always yields exactly one coordinate.  Treat any
        # queued output here as a protocol violation instead of consuming a
        # stale move for the new position.
        stale: list[str] = []
        while True:
            try:
                line = self._stdout.get_nowait()
            except queue.Empty:
                break
            if line is None:
                raise RapfiProtocolError(self._death_message("Rapfi closed stdout"))
            if line.strip():
                stale.append(line.strip())
        if stale:
            raise RapfiProtocolError(f"unexpected queued Rapfi output: {stale[-4:]}")

        self._send("BOARD")
        for x, y, side in normalized:
            self._send(f"{x},{y},{side}")
        self._send("DONE")

        deadline = time.monotonic() + self.response_timeout_s
        while True:
            line = self._next_stdout(deadline)
            match = _MOVE_RE.match(line)
            if match:
                x, y = int(match.group(1)), int(match.group(2))
                if not (0 <= x < self.board_size and 0 <= y < self.board_size):
                    raise RapfiProtocolError(f"Rapfi returned out-of-board move {(x, y)}")
                if (x, y) in occupied:
                    raise RapfiProtocolError(f"Rapfi returned occupied move {(x, y)}")
                return x, y
            upper = line.upper()
            if upper.startswith(("ERROR", "UNKNOWN")):
                raise RapfiProtocolError(f"Rapfi search failed: {line}")
            self.messages.append(line)

    def close(self, *, force: bool = False) -> None:
        if self._closed:
            return
        self._closed = True
        if not force and self._process.poll() is None and self._process.stdin is not None:
            try:
                self._process.stdin.write("END\n")
                self._process.stdin.flush()
                self._process.wait(timeout=2.0)
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                force = True
        if force and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=2.0)
        for stream in (self._process.stdin, self._process.stdout, self._process.stderr):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass

    def __enter__(self) -> "RapfiAdapter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close(force=exc is not None)

    def __del__(self) -> None:  # pragma: no cover - best-effort interpreter cleanup
        try:
            self.close(force=True)
        except Exception:
            pass

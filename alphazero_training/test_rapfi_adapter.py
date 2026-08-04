from __future__ import annotations

from pathlib import Path
import textwrap

import pytest

from alphazero_training.rapfi_adapter import (
    BLACK,
    WHITE,
    RapfiAdapter,
    RapfiProtocolError,
    RapfiTimeoutError,
)


def _write_fake_engine(path: Path, *, response: str = "move") -> Path:
    if response == "move":
        action = """
            occupied = {(int(row.split(',')[0]), int(row.split(',')[1])) for row in board}
            for y in range(19):
                for x in range(19):
                    if (x, y) not in occupied:
                        print(f"{x},{y}", flush=True)
                        break
                else:
                    continue
                break
        """
    elif response == "illegal":
        action = "print('0,0', flush=True)"
    else:
        action = "time.sleep(5)"
    path.write_text(
        textwrap.dedent(
            f"""
            import pathlib
            import sys
            import time

            log = pathlib.Path(__file__).with_suffix('.log')
            board = []
            in_board = False
            for raw in sys.stdin:
                line = raw.strip()
                with log.open('a', encoding='utf-8') as stream:
                    stream.write(line + '\\n')
                if line.startswith('START '):
                    print('MESSAGE fake engine ready', flush=True)
                    print('OK', flush=True)
                elif line == 'BOARD':
                    board = []
                    in_board = True
                elif in_board and line == 'DONE':
                    in_board = False
                    {textwrap.indent(textwrap.dedent(action).strip(), '                    ').lstrip()}
                elif in_board:
                    board.append(line)
                elif line == 'END':
                    break
            """
        ),
        encoding="utf-8",
    )
    return path


def test_adapter_maps_black_white_to_self_opponent(tmp_path: Path) -> None:
    engine = _write_fake_engine(tmp_path / "fake_rapfi.py")
    with RapfiAdapter(engine, response_timeout_s=2) as adapter:
        # Two plies mean black is next.  Relative Piskvork flags must identify
        # black as SELF even though white made the latest move.
        assert adapter.choose_move([[9, 9, BLACK], [8, 9, WHITE]], BLACK) == (0, 0)
        assert "fake engine ready" in adapter.messages[0]

    log = engine.with_suffix(".log").read_text(encoding="utf-8").splitlines()
    board_index = log.index("BOARD")
    assert log[board_index + 1 : board_index + 4] == ["9,9,1", "8,9,2", "DONE"]
    assert "INFO RULE 0" in log


def test_adapter_maps_white_as_self(tmp_path: Path) -> None:
    engine = _write_fake_engine(tmp_path / "fake_rapfi.py")
    with RapfiAdapter(engine, response_timeout_s=2) as adapter:
        assert adapter.choose_move([[9, 9, BLACK]], WHITE) == (0, 0)
    log = engine.with_suffix(".log").read_text(encoding="utf-8").splitlines()
    board_index = log.index("BOARD")
    assert log[board_index + 1 : board_index + 3] == ["9,9,2", "DONE"]


def test_adapter_rejects_turn_mismatch_and_illegal_engine_move(tmp_path: Path) -> None:
    engine = _write_fake_engine(tmp_path / "illegal_rapfi.py", response="illegal")
    with RapfiAdapter(engine, response_timeout_s=2) as adapter:
        with pytest.raises(ValueError, match="to move"):
            adapter.choose_move([], WHITE)
        with pytest.raises(RapfiProtocolError, match="occupied"):
            adapter.choose_move([[0, 0, BLACK]], WHITE)


def test_adapter_enforces_hard_timeout(tmp_path: Path) -> None:
    engine = _write_fake_engine(tmp_path / "slow_rapfi.py", response="timeout")
    with RapfiAdapter(engine, response_timeout_s=0.1) as adapter:
        with pytest.raises(RapfiTimeoutError, match="timed out"):
            adapter.choose_move([], BLACK)

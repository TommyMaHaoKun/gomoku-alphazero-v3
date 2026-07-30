#!/usr/bin/env python3
"""Regression checks for the Windows Pygame inference integration."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
GAME_PATH = ROOT / "Gomoku AI player V1.0.py"
CHECKPOINT_PATH = Path(__file__).resolve().parent / "latest.pt"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alphazero_training.play_agent import (
    AlphaZeroGomokuAgent,
    DEFAULT_PLAY_SIMULATIONS,
    configured_play_simulations,
)


def load_game_module():
    spec = importlib.util.spec_from_file_location("gomoku_pygame", GAME_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {GAME_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def empty_grid() -> list[list[int]]:
    return [[0] * 19 for _ in range(19)]


def tactical_cases() -> list[tuple[str, list[list[int]], set[tuple[int, int]]]]:
    cases: list[tuple[str, list[list[int]], set[tuple[int, int]]]] = []
    directions = ((1, 0), (0, 1), (1, 1), (1, -1))
    starts = ((5, 5), (8, 7))
    for stone, name in ((1, "win"), (2, "block")):
        for dx, dy in directions:
            for initial_x, initial_y in starts:
                start_y = initial_y + (5 if dy == -1 else 0)
                grid = empty_grid()
                for offset in range(4):
                    grid[start_y + offset * dy][initial_x + offset * dx] = stone
                expected: set[tuple[int, int]] = set()
                for x in range(19):
                    for y in range(19):
                        if grid[y][x] == 0:
                            grid[y][x] = stone
                            if load_game_module_cached.is_five(grid, x, y, stone):
                                expected.add((x, y))
                            grid[y][x] = 0
                cases.append((name, grid, expected))

                broken = empty_grid()
                for offset in (0, 1, 3, 4):
                    broken[start_y + offset * dy][initial_x + offset * dx] = stone
                gap = (initial_x + 2 * dx, start_y + 2 * dy)
                cases.append((f"broken_{name}", broken, {gap}))
    return cases


load_game_module_cached = load_game_module()


def main() -> int:
    if DEFAULT_PLAY_SIMULATIONS != 256 or configured_play_simulations(8) != 8:
        raise AssertionError("desktop strong-mode MCTS configuration regression")
    agent = AlphaZeroGomokuAgent(CHECKPOINT_PATH, simulations=8)
    if agent.simulations != 8 or "8 MCTS" not in agent.search_label:
        raise AssertionError("desktop MCTS label does not match the active budget")
    checked = 0
    for ai_color in (1, 2):
        for name, grid, expected in tactical_cases():
            ui_action = load_game_module_cached.choose_forced_move(grid, ai_color)
            model_action = agent.choose_move(grid, ai_color=ai_color)
            if ui_action not in expected:
                raise AssertionError(
                    f"UI guard failed {name}/{ai_color=}: {ui_action=} {expected=}"
                )
            if model_action not in expected:
                raise AssertionError(
                    f"agent failed {name}/{ai_color=}: {model_action=} {expected=}"
                )
            checked += 2

    # Exact board immediately before the missed block shown in the screenshot.
    screenshot_grid = empty_grid()
    for x, y in ((9, 9), (6, 6), (7, 6), (5, 8)):
        screenshot_grid[y][x] = 1
    for x, y in ((6, 8), (7, 8), (8, 8), (9, 8)):
        screenshot_grid[y][x] = 2
    expected_screenshot_block = (10, 8)
    if load_game_module_cached.choose_forced_move(screenshot_grid) != expected_screenshot_block:
        raise AssertionError("screenshot UI regression")
    if agent.choose_move(screenshot_grid, (9, 8)) != expected_screenshot_block:
        raise AssertionError("screenshot agent regression")
    checked += 2

    # The same mandatory block must work when the AI is white.
    reversed_grid = [
        [2 if cell == 1 else 1 if cell == 2 else 0 for cell in row]
        for row in screenshot_grid
    ]
    if load_game_module_cached.choose_forced_move(reversed_grid, 2) != expected_screenshot_block:
        raise AssertionError("white UI regression")
    if agent.choose_move(reversed_grid, (9, 8), ai_color=2) != expected_screenshot_block:
        raise AssertionError("white agent regression")
    checked += 2

    # Move numbers must follow the real play order, and every AI stone must
    # receive the square outline regardless of which color the AI controls.
    pygame = load_game_module_cached.pygame
    pygame.font.init()

    class FirstEmptyAgent:
        label = "test agent"

        @staticmethod
        def choose_move(grid, last_move=None, ai_color=1):
            del last_move, ai_color
            for y, row in enumerate(grid):
                for x, cell in enumerate(row):
                    if cell == 0:
                        return x, y
            return None

    ui = load_game_module_cached.CLS_gomoku(
        pygame.Surface((30, 30)),
        pygame.Surface((30, 30)),
        30,
        80,
        FirstEmptyAgent(),
    )
    for star_y in (3, 9, 15):
        for star_x in (3, 9, 15):
            pixel_x = load_game_module_cached.BOARD_X0 + star_x * load_game_module_cached.BOARD_SIZE + 2
            pixel_y = load_game_module_cached.BOARD_Y0 + star_y * load_game_module_cached.BOARD_SIZE + 2
            if ui.board.get_at((pixel_x, pixel_y))[:3] != (0, 0, 0):
                raise AssertionError(f"missing star point at {(star_x, star_y)}")

    def finish_ai_move():
        deadline = time.time() + 3
        while ui.ai_thinking and time.time() < deadline:
            ui.update()
            time.sleep(0.01)
        ui.update()
        if ui.ai_thinking:
            raise AssertionError("test AI did not finish")

    ui.select_color(load_game_module_cached.GRID_BLACK)
    center_x = ui.x0 + load_game_module_cached.BOARD_X0 + 9 * load_game_module_cached.BOARD_SIZE
    center_y = ui.y0 + load_game_module_cached.BOARD_Y0 + 9 * load_game_module_cached.BOARD_SIZE
    ui.mouse_down(center_x, center_y)
    finish_ai_move()
    if ui.move_numbers[9][9] != 1 or ui.move_numbers[0][0] != 2:
        raise AssertionError("black-first move numbering regression")

    rendered = pygame.Surface((800, 680))
    ui.draw(rendered)
    if rendered.get_at((ui.x0, ui.y0))[:3] != (235, 0, 180):
        raise AssertionError("white AI outline regression")

    next_x = ui.x0 + load_game_module_cached.BOARD_X0 + 10 * load_game_module_cached.BOARD_SIZE
    next_y = ui.y0 + load_game_module_cached.BOARD_Y0 + 9 * load_game_module_cached.BOARD_SIZE
    ui.mouse_down(next_x, next_y)
    finish_ai_move()
    rendered.fill((0, 0, 0))
    ui.draw(rendered)
    if rendered.get_at((ui.x0, ui.y0))[:3] == (235, 0, 180):
        raise AssertionError("old AI outline was not cleared")
    if rendered.get_at((ui.x0 + load_game_module_cached.BOARD_SIZE, ui.y0))[:3] != (235, 0, 180):
        raise AssertionError("latest AI outline did not move")

    # Undo removes one complete human turn, including the AI response, and
    # restores move numbers, turn ownership, and the latest-AI highlight.
    ui.mouse_down(*ui.undo_button.center)
    if ui.grid[9][10] != 0 or ui.grid[0][1] != 0:
        raise AssertionError("undo did not remove the latest human/AI pair")
    if ui.grid[9][9] != 1 or ui.grid[0][0] != 2:
        raise AssertionError("undo removed moves from an earlier turn")
    if ui.move_count != 2 or ui.move_numbers[9][9] != 1 or ui.move_numbers[0][0] != 2:
        raise AssertionError("undo did not restore move numbering")
    if ui.turn != load_game_module_cached.GRID_BLACK or ui.last_ai_move != (0, 0):
        raise AssertionError("undo did not restore turn/highlight state")

    ui.select_color(load_game_module_cached.GRID_WHITE)
    finish_ai_move()
    if ui.move_numbers[0][0] != 1:
        raise AssertionError("white-second move numbering regression")
    rendered.fill((0, 0, 0))
    ui.draw(rendered)
    if rendered.get_at((ui.x0, ui.y0))[:3] != (235, 0, 180):
        raise AssertionError("black AI outline regression")
    if ui.can_undo():
        raise AssertionError("AI opening move must not be undoable by itself")

    ui.mouse_down(center_x, center_y)
    finish_ai_move()
    ui.eventkey(pygame.K_BACKSPACE)
    if ui.move_count != 1 or ui.grid[0][0] != 1 or ui.grid[9][9] != 0:
        raise AssertionError("white-player undo did not preserve the AI opening")
    if ui.turn != load_game_module_cached.GRID_WHITE or ui.last_ai_move != (0, 0):
        raise AssertionError("white-player undo restored incorrect state")

    # A result produced by an AI worker for the pre-undo board must be ignored.
    ui.select_color(load_game_module_cached.GRID_BLACK)
    ui.mouse_down(center_x, center_y)
    if not ui.undo():
        raise AssertionError("undo was unavailable while the AI was thinking")
    time.sleep(0.05)
    for _ in range(4):
        ui.update()
    if any(cell != 0 for row in ui.grid for cell in row):
        raise AssertionError("stale AI result placed a stone after undo")
    if ui.move_count != 0 or ui.ai_thinking or ui.turn != load_game_module_cached.GRID_BLACK:
        raise AssertionError("undo during AI calculation restored incorrect state")
    checked += 27

    print(f"play integration verified: {checked} tactical decisions passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

from pathlib import Path

import numpy as np

from alphazero_training.build_desktop_loss_correction import build_correction
from alphazero_training.game_logger import GameReplayLogger, UI_BLACK, UI_WHITE


def test_builds_policy_only_hard_negative_from_reviewed_loss(tmp_path: Path) -> None:
    moves = [
        (9, 8, UI_BLACK), (8, 7, UI_WHITE),
        (8, 9, UI_BLACK), (7, 10, UI_WHITE),
        (10, 8, UI_BLACK), (11, 8, UI_WHITE),
        (11, 9, UI_BLACK), (9, 7, UI_WHITE),
        (10, 7, UI_BLACK), (10, 10, UI_WHITE),
        (8, 10, UI_BLACK), (10, 6, UI_WHITE),
        (8, 8, UI_BLACK), (8, 11, UI_WHITE),
        (9, 9, UI_BLACK), (7, 11, UI_WHITE),
        (10, 9, UI_BLACK), (7, 9, UI_WHITE),
        (12, 9, UI_BLACK),
    ]
    saved = GameReplayLogger(tmp_path / "logs").record_game(
        moves,
        winner=UI_BLACK,
        ai_color=UI_WHITE,
        model_label="test",
        termination="five_in_a_row",
    )
    output = tmp_path / "correction.npz"
    metadata = build_correction(
        saved.metadata_path,
        saved.replay_path,
        output,
        move_number=14,
        teacher_x=9,
        teacher_y=9,
        teacher_name="Rapfi",
        teacher_engine_sha256="a" * 64,
        teacher_node_budgets=[200_000, 1_000_000],
    )

    assert metadata["samples"] == 8
    assert metadata["recorded_mistake"]["action"] == 11 * 19 + 8
    assert metadata["teacher_correction"]["action"] == 9 * 19 + 9
    with np.load(output, allow_pickle=False) as archive:
        assert archive["states"].shape == (8, 4, 19, 19)
        assert archive["policies"].shape == (8, 361)
        assert archive["teacher_action"].tolist() == [180] * 8
        assert archive["mistake_action"][0] == 217
        assert np.all(archive["value_weights"] == 0)
        assert np.all(archive["priority"] == 8)
        assert np.all(
            archive["teacher_action"] != archive["mistake_action"]
        )

from __future__ import annotations

from pathlib import Path

import numpy as np

from alphazero_training.rapfi_counterfactual import (
    BranchMove,
    BranchRecord,
    BranchTask,
    export_branch_dataset,
)
from alphazero_training.rapfi_distill import BLACK, WHITE


def test_counterfactual_dataset_marks_first_move_as_hard_negative(
    tmp_path: Path,
) -> None:
    task = BranchTask(
        task_index=0,
        pair_index=7,
        game_index=2,
        source_ply=2,
        student_color=WHITE,
        player=BLACK,
        history_before=[[9, 9, BLACK], [8, 9, WHITE]],
        teacher_x=10,
        teacher_y=9,
        mistake_x=7,
        mistake_y=9,
        original_moves_to_end=6,
        priority=4.0,
    )
    branch = BranchRecord(
        task=task,
        moves=[
            BranchMove(0, 10, 9, BLACK, "teacher_correction", 0.01),
            BranchMove(1, 8, 8, WHITE, "rapfi_rollout", 0.01),
        ],
        winner=BLACK,
        termination="win",
    )
    output = tmp_path / "counterfactual.npz"
    metadata = export_branch_dataset([branch], output)
    assert metadata["samples"] == 2
    assert metadata["terminal_branches"] == 1
    with np.load(output) as data:
        assert data["mistake_action"].tolist() == [9 * 19 + 7, -1]
        assert np.argmax(data["policies"], axis=1).tolist() == [9 * 19 + 10, 8 * 19 + 8]
        assert data["values"].tolist() == [1.0, -1.0]
        assert np.all(data["value_weights"] > 0)
        assert set(data["group_id"].tolist()) == {7}

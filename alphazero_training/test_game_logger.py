from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from alphazero_training.game_logger import GameReplayLogger
from alphazero_training.train_v3_selfplay import StaticReplaySource


BLACK_WIN = [
    (0, 0, 1),
    (0, 1, 2),
    (1, 0, 1),
    (1, 1, 2),
    (2, 0, 1),
    (2, 1, 2),
    (3, 0, 1),
    (3, 1, 2),
    (4, 0, 1),
]


class GameReplayLoggerTests(unittest.TestCase):
    def test_ai_loss_is_saved_in_all_games_and_pending_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            logger = GameReplayLogger(Path(temporary))
            saved = logger.record_game(
                BLACK_WIN,
                winner=1,
                ai_color=2,
                model_label="Gargantua test",
                search_label="8 MCTS (CPU)",
                termination="completed",
            )

            self.assertTrue(saved.replay_path.is_file())
            self.assertTrue(saved.metadata_path.is_file())
            self.assertIsNotNone(saved.pending_replay_path)
            self.assertIsNotNone(saved.pending_metadata_path)
            self.assertTrue(saved.pending_replay_path.is_file())
            self.assertEqual(
                saved.replay_path.read_bytes(), saved.pending_replay_path.read_bytes()
            )

            with np.load(saved.replay_path, allow_pickle=False) as archive:
                self.assertEqual(archive["states"].shape, (9, 4, 19, 19))
                self.assertEqual(archive["policies"].shape, (9, 361))
                np.testing.assert_array_equal(
                    archive["actions"], np.asarray([0, 19, 1, 20, 2, 21, 3, 22, 4])
                )
                np.testing.assert_array_equal(
                    archive["values"], np.asarray([1, -1, 1, -1, 1, -1, 1, -1, 1])
                )
                np.testing.assert_array_equal(archive["value_weights"], np.ones(9))
                self.assertTrue(np.allclose(archive["policies"].sum(axis=1), 1.0))
                for index, action in enumerate(archive["actions"]):
                    self.assertEqual(float(archive["policies"][index, action]), 1.0)

            metadata = json.loads(saved.metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["ai_result"], "loss")
            self.assertTrue(metadata["eligible_for_pending_training"])
            self.assertEqual(metadata["plies"], 9)
            self.assertEqual(metadata["moves"][-1]["x"], 4)

            # The existing V3 static replay loader accepts the archive without
            # conversion, which enforces the self-play schema contract.
            source = StaticReplaySource("desktop", saved.replay_path, 19, 7)
            self.assertEqual(len(source), 9)
            self.assertEqual(source.group_key, "group_id")

    def test_interrupted_game_keeps_policy_but_masks_unknown_value(self):
        with tempfile.TemporaryDirectory() as temporary:
            saved = GameReplayLogger(temporary).record_game(
                [(9, 9, 1), (10, 9, 2)],
                winner=None,
                ai_color=2,
                model_label="Gargantua test",
                termination="restarted",
            )
            self.assertIsNone(saved.pending_replay_path)
            with np.load(saved.replay_path, allow_pickle=False) as archive:
                np.testing.assert_array_equal(archive["values"], np.zeros(2))
                np.testing.assert_array_equal(archive["value_weights"], np.zeros(2))
                np.testing.assert_array_equal(archive["policy_weights"], np.ones(2))
            metadata = json.loads(saved.metadata_path.read_text(encoding="utf-8"))
            self.assertFalse(metadata["completed"])
            self.assertEqual(metadata["ai_result"], "unfinished")

    def test_ai_win_is_not_added_to_pending_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            saved = GameReplayLogger(temporary).record_game(
                BLACK_WIN,
                winner=1,
                ai_color=1,
                model_label="Gargantua test",
                termination="completed",
            )
            self.assertIsNone(saved.pending_replay_path)
            self.assertFalse((Path(temporary) / "pending_training").exists())

    def test_rejects_a_completed_nonterminal_history(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "completed game"):
                GameReplayLogger(temporary).record_game(
                    [(9, 9, 1)],
                    winner=1,
                    ai_color=2,
                    model_label="Gargantua test",
                    termination="completed",
                )


if __name__ == "__main__":
    unittest.main()

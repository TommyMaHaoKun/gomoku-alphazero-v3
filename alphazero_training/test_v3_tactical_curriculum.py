from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from alphazero_training.v3_tactical_curriculum import (
    ACTION_COUNT,
    BOARD_SIZE,
    CurriculumConfig,
    _case_validation_error,
    _labels,
    add_safe_distractors,
    d4_action,
    encode_case,
    generate_curriculum,
    transform_case,
    write_curriculum,
)
from alphazero_training.v3_tactical_suite import (
    BLACK,
    WHITE,
    TacticalCase,
    built_in_cases,
    oracle_actions,
)


class TacticalCurriculumTests(unittest.TestCase):
    def test_all_d4_maps_are_board_bijections(self) -> None:
        actions = set(range(ACTION_COUNT))
        for symmetry in range(8):
            transformed = {d4_action(action, symmetry) for action in actions}
            self.assertEqual(actions, transformed)

    def test_transform_colour_swap_and_encoding(self) -> None:
        source = built_in_cases()[0]
        transformed = transform_case(source, dx=2, dy=-3, symmetry=1, colour_swap=True)
        self.assertIsNone(_case_validation_error(transformed))
        self.assertEqual(-source.side_to_move, transformed.side_to_move)
        self.assertEqual(
            tuple(sorted(d4_action(action + (-3 * BOARD_SIZE + 2), 1) for action in source.declared_actions)),
            transformed.declared_actions,
        )
        state = encode_case(transformed)
        self.assertEqual((4, BOARD_SIZE, BOARD_SIZE), state.shape)
        self.assertEqual(np.uint8, state.dtype)
        self.assertEqual(len(source.stones), int(state[0].sum() + state[1].sum()))
        self.assertFalse(state[2].any())
        self.assertTrue(state[3].all() if transformed.side_to_move == BLACK else not state[3].any())

    def test_safe_distractors_are_deterministic_and_preserve_oracle(self) -> None:
        case = built_in_cases()[4]  # non-trivial open-four oracle
        first, first_added = add_safe_distractors(
            case,
            2,
            seed=1234,
            source="unit-test",
            min_distance=2,
        )
        second, second_added = add_safe_distractors(
            case,
            2,
            seed=1234,
            source="unit-test",
            min_distance=2,
        )
        self.assertEqual(first, second)
        self.assertEqual(first_added, second_added)
        self.assertEqual(2, len(first_added))
        self.assertIsNone(_case_validation_error(first))
        self.assertEqual(tuple(sorted(case.declared_actions)), tuple(sorted(oracle_actions(first))))
        self.assertFalse(first.board.has_existing_five(BLACK))
        self.assertFalse(first.board.has_existing_five(WHITE))

    def test_generation_schema_dedup_and_atomic_round_trip(self) -> None:
        # A focused subset keeps the unit test quick while still exercising a
        # one-ply and a multi-ply forcing label under every D4/colour variant.
        cases = (built_in_cases()[0], built_in_cases()[4])
        config = CurriculumConfig(
            seed=77,
            translation_stride=99,
            distractor_variants=0,
        )
        dataset = generate_curriculum(cases=cases, config=config)
        arrays = dataset.arrays()
        count = len(dataset.states)
        self.assertGreater(count, 0)
        self.assertEqual((count, 4, BOARD_SIZE, BOARD_SIZE), arrays["states"].shape)
        self.assertEqual((count, ACTION_COUNT), arrays["policies"].shape)
        self.assertEqual(np.uint8, arrays["states"].dtype)
        self.assertEqual(np.float16, arrays["policies"].dtype)
        self.assertEqual(np.int8, arrays["values"].dtype)
        self.assertEqual("U", arrays["source"].dtype.kind)
        np.testing.assert_allclose(
            arrays["policies"].sum(axis=1, dtype=np.float32), 1.0, atol=2e-3
        )
        self.assertTrue(np.all(arrays["values"] == 1))
        self.assertTrue(np.all(arrays["value_weight"] == 1))
        state_keys = [state.tobytes() for state in arrays["states"]]
        self.assertEqual(len(state_keys), len(set(state_keys)))

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "curriculum.npz"
            archive, summary = write_curriculum(dataset, output)
            self.assertEqual(output.resolve(), archive)
            self.assertTrue(summary.exists())
            with np.load(archive, allow_pickle=False) as loaded:
                self.assertEqual(set(arrays), set(loaded.files))
                for name, expected in arrays.items():
                    np.testing.assert_array_equal(expected, loaded[name])
            metadata = json.loads(summary.read_text(encoding="utf-8"))
            self.assertEqual(count, metadata["counts"]["records"])
            self.assertEqual(64, len(metadata["artifact"]["sha256"]))

    def test_defense_labels_do_not_claim_a_game_value(self) -> None:
        immediate_defense = built_in_cases()[2]
        unique_defense = built_in_cases()[-2]
        for case in (immediate_defense, unique_defense):
            value, policy_weight, value_weight, priority = _labels(case)
            self.assertEqual(0, value)
            self.assertEqual(1.0, policy_weight)
            self.assertEqual(0.0, value_weight)
            self.assertGreater(priority, 0)

    def test_terminal_input_is_rejected(self) -> None:
        source = built_in_cases()[0]
        terminal = TacticalCase(
            case_id="terminal",
            category=source.category,
            side_to_move=source.side_to_move,
            stones=source.stones + tuple((x, 0, WHITE) for x in range(5)),
            oracle_kind=source.oracle_kind,
            declared_actions=source.declared_actions,
            description=source.description,
            max_plies=source.max_plies,
        )
        self.assertEqual("input board is terminal", _case_validation_error(terminal))


if __name__ == "__main__":
    unittest.main()

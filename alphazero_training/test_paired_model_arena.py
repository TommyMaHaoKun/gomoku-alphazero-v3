import unittest

from alphazero_training.paired_model_arena import exact_two_sided_sign_p, summarize


class PairedArenaTests(unittest.TestCase):
    def test_exact_sign_p(self) -> None:
        self.assertEqual(exact_two_sided_sign_p(0, 0), 1.0)
        self.assertLess(exact_two_sided_sign_p(10, 0), 0.05)

    def test_summary_uses_paired_and_color_results(self) -> None:
        records = [
            {"pair_index": 0, "candidate_color": "black", "candidate_result": 1.0},
            {"pair_index": 0, "candidate_color": "white", "candidate_result": 1.0},
            {"pair_index": 1, "candidate_color": "black", "candidate_result": 1.0},
            {"pair_index": 1, "candidate_color": "white", "candidate_result": 0.0},
        ]
        result = summarize(records, 2)
        self.assertEqual(result["paired_gains"], 1)
        self.assertEqual(result["paired_unchanged"], 1)
        self.assertEqual(result["candidate_by_color"]["black"]["wins"], 2)


if __name__ == "__main__":
    unittest.main()

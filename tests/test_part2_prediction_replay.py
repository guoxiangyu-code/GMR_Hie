import unittest

from training.flash_vtg_gmr.validate_part2_predictions import replay_prediction, validate_predictions


class Part2PredictionReplayTest(unittest.TestCase):
    def test_c1_replay_and_only_count_zero_is_empty(self):
        gt = [{"qid": 1, "D_decode": 20.0, "relevant_windows": []}]
        row = {
            "qid": 1,
            "variant": "C1",
            "pred_count": 0,
            "pred_count_probs": [0.6, 0.1, 0.1, 0.1, 0.1],
            "pred_exist_score": 0.4,
            "pred_relevant_windows": [],
            "oracle_mode_windows": [[1.0, 5.0, 20.0, 20.0]],
        }
        result = validate_predictions([row], gt, {"variant": "C1", "tau_mode": 0.5})
        self.assertEqual(result["status"], "PASS")

    def test_four_plus_falls_back_to_top_four(self):
        row = {
            "variant": "C2", "pred_count": 4,
            "oracle_mode_windows": [[i * 2.0, i * 2.0 + 2.0, -10.0, float(10 - i)] for i in range(6)],
        }
        result = replay_prediction(row, {"tau_mode": 0.9}, 20.0)
        self.assertEqual(len(result), 4)


if __name__ == "__main__":
    unittest.main()

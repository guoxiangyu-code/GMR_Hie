import types
import unittest

import torch

from training.flash_vtg_gmr.inference import select_predictions_for_inference


class InferenceSelectionContractTest(unittest.TestCase):
    def test_candidate_spans_decode_with_d_grid(self):
        outputs = {
            "candidate_span": torch.tensor([[[0.1, 0.2]]]),
            "candidate_logit": torch.tensor([[10.0]]),
            "candidate_mask": torch.tensor([[True]]),
        }
        opt = types.SimpleNamespace(variant="G0-Threshold", count_calibration=None)
        result = select_predictions_for_inference(
            outputs, opt, {"duration": 100.0, "D_grid": 80.0}
        )
        self.assertAlmostEqual(result[0][0], 8.0, places=5)
        self.assertAlmostEqual(result[0][1], 16.0, places=5)

    def test_count_zero_is_the_only_aec_empty_decision(self):
        outputs = {
            "pred_count_logits": torch.tensor([[9.0, 1.0, 0.0, 0.0, 0.0]]),
            "event_span": torch.rand(1, 10, 2),
            "event_logit": torch.full((1, 10), 20.0),
            "quality_logit": torch.full((1, 10), 20.0),
            "event_mask": torch.ones(1, 10, dtype=torch.bool),
        }
        opt = types.SimpleNamespace(variant="C1", count_calibration=None)
        self.assertEqual(select_predictions_for_inference(outputs, opt, {"duration": 10.0}), [])


if __name__ == "__main__":
    unittest.main()

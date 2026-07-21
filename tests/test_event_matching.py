import sys
import os
import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.flash_vtg_gmr.event_adapter import (
    hungarian_matching, compute_adapter_losses, ProposalToEventAdapter, focal_loss_binary
)
from models.flash_vtg_gmr.event_cardinality import (
    AdaptiveEventCardinality, compute_effective_number_weights, select_events_from_aec
)

class TestEventMatchingAndLosses(unittest.TestCase):

    def test_focal_loss_exact(self):
        """focal_loss_binary should decrease as confidence in correct label increases."""
        logits_pos = torch.tensor([2.0]) # high probability (~0.88)
        logits_neg = torch.tensor([-2.0]) # low probability (~0.12)
        
        target_pos = torch.tensor([1.0])
        target_neg = torch.tensor([0.0])
        
        loss_correct_pos = focal_loss_binary(logits_pos, target_pos)
        loss_incorrect_pos = focal_loss_binary(logits_neg, target_pos)
        
        self.assertLess(loss_correct_pos.item(), loss_incorrect_pos.item())

        loss_correct_neg = focal_loss_binary(logits_neg, target_neg)
        loss_incorrect_neg = focal_loss_binary(logits_pos, target_neg)
        
        self.assertLess(loss_correct_neg.item(), loss_incorrect_neg.item())

    def test_hungarian_matching_toy_case(self):
        """Hungarian matching should correctly match overlapping regions."""
        # Modes: Mode 0: [0.1, 0.4], Mode 1: [0.5, 0.9]
        # GTs: GT 0: [0.5, 0.9], GT 1: [0.1, 0.4]
        event_span = torch.tensor([[0.1, 0.4], [0.5, 0.9]], dtype=torch.float32)
        event_logit = torch.zeros(2)
        gt_spans = torch.tensor([[0.5, 0.9], [0.1, 0.4]], dtype=torch.float32)
        
        m_inds, g_inds = hungarian_matching(event_span, event_logit, gt_spans)
        
        # Expected matches: Mode 0 matches GT 1, Mode 1 matches GT 0
        self.assertEqual(m_inds[0], 0)
        self.assertEqual(g_inds[0], 1)
        self.assertEqual(m_inds[1], 1)
        self.assertEqual(g_inds[1], 0)

    def test_adapter_loss_computation_positive_and_null(self):
        """compute_adapter_losses should run without errors for both null and positive targets."""
        # Batch size 2
        event_logit = torch.randn(2, 10)
        quality_logit = torch.randn(2, 10)
        event_span = torch.rand(2, 10, 2)
        event_mask = torch.ones(2, 10, dtype=torch.bool)
        
        targets = [
            {"is_null": False, "relevant_windows_norm": [[0.1, 0.3], [0.5, 0.8]]}, # positive
            {"is_null": True, "relevant_windows_norm": []} # null
        ]
        
        losses = compute_adapter_losses(event_logit, quality_logit, event_span, event_mask, targets, variant="P0")
        self.assertIn("loss_event", losses)
        self.assertIn("loss_quality", losses)
        self.assertNotIn("loss_span", losses) # P0 should not have span loss
        
        # P0-R variant should have span loss
        losses_r = compute_adapter_losses(event_logit, quality_logit, event_span, event_mask, targets, variant="P0-R")
        self.assertIn("loss_span", losses_r)

    def test_effective_number_weighting_normalization(self):
        """compute_effective_number_weights should return normalized and clipped weights."""
        counts = {0: 1000, 1: 500, 2: 100, 3: 10, 4: 2}
        weights = compute_effective_number_weights(counts, beta_factor=0.9999)
        
        # Rare classes (3 and 4) should have higher weights than common classes (0 and 1)
        self.assertGreater(weights[4].item(), weights[0].item())
        self.assertGreater(weights[3].item(), weights[1].item())
        
        # All weights must be within [0.5, 2.0]
        for w in weights:
            self.assertTrue(0.5 <= w.item() <= 2.0)

    def test_aec_forward_shapes(self):
        """AEC forward pass should yield correct shape output tensors."""
        B, L, D_txt = 2, 40, 512
        D_set = 256
        aec = AdaptiveEventCardinality(text_dim=D_txt, event_dim=D_set, variant="C2")
        
        text_feat = torch.randn(B, L, D_txt)
        text_mask = torch.ones(B, L, dtype=torch.bool)
        set_feat = torch.randn(B, 10, D_set)
        set_mask = torch.ones(B, 10, dtype=torch.bool)
        
        count_class = torch.tensor([1, 4], dtype=torch.long)
        
        out = aec(text_feat, text_mask, set_feat, set_mask, count_class=count_class)
        
        self.assertEqual(out["count_logits"].shape, (B, 5))
        self.assertEqual(out["count_probs"].shape, (B, 5))
        self.assertIn("loss_count", out)
        self.assertIn("loss_count_con", out)
        self.assertIn("loss_count_total", out)

    def test_select_events_from_aec_behavior(self):
        """select_events_from_aec should adhere to §9.6 selection rules."""
        # 1. Count class 0 (empty set)
        probs_0 = torch.tensor([0.9, 0.05, 0.03, 0.01, 0.01])
        scores = torch.ones(10)
        mask = torch.ones(10, dtype=torch.bool)
        self.assertEqual(select_events_from_aec(probs_0, scores, mask), [])

        # 2. Count class 2 (Top-2)
        probs_2 = torch.tensor([0.05, 0.05, 0.8, 0.05, 0.05])
        scores = torch.tensor([0.1, 0.9, 0.3, 0.8, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0])
        sel_2 = select_events_from_aec(probs_2, scores, mask)
        self.assertEqual(len(sel_2), 2)
        # Should be index 1 (score 0.9) and index 3 (score 0.8)
        self.assertIn(1, sel_2)
        self.assertIn(3, sel_2)

        # 3. Count class 4+ (thresholded, min 4, max 10)
        probs_4 = torch.tensor([0.01, 0.01, 0.01, 0.07, 0.9])
        scores = torch.tensor([0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.2, 0.2, 0.2])
        # tau_mode = 0.5. Scores above threshold: indices 0-6 (7 items).
        # Should select all 7 items because it is >= 4 and <= 10.
        sel_4 = select_events_from_aec(probs_4, scores, mask, tau_mode=0.5, top_n_clip=10)
        self.assertEqual(len(sel_4), 7)
        self.assertNotIn(7, sel_4)

        # Fallback to Top-4 if fewer than 4 meet the threshold
        scores_low = torch.tensor([0.6, 0.6, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
        # Only indices 0,1 are >= 0.5. Fallback should choose Top-4 (which will include some below threshold).
        sel_fb = select_events_from_aec(probs_4, scores_low, mask, tau_mode=0.5, top_n_clip=10)
        self.assertEqual(len(sel_fb), 4)

if __name__ == "__main__":
    unittest.main()

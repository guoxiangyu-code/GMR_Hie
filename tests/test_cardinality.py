import unittest

import torch

from models.flash_vtg_gmr.event_cardinality import AdaptiveEventCardinality, CountContrastiveHead


class CardinalityContractTest(unittest.TestCase):
    def test_validation_does_not_mutate_contrastive_queue(self):
        head = CountContrastiveHead(feat_dim=8, proj_dim=4, queue_size_per_class=2)
        head.eval()
        queue = head.queue.clone()
        labels = head.queue_labels.clone()
        pointers = head.queue_ptr.clone()
        head(torch.randn(3, 8), torch.tensor([0, 1, 2]))
        self.assertTrue(torch.equal(queue, head.queue))
        self.assertTrue(torch.equal(labels, head.queue_labels))
        self.assertTrue(torch.equal(pointers, head.queue_ptr))

    def test_only_contrastive_variants_emit_contrastive_loss(self):
        text = torch.randn(3, 2, 512)
        text_mask = torch.ones(3, 2, dtype=torch.bool)
        events = torch.randn(3, 4, 256)
        event_mask = torch.ones(3, 4, dtype=torch.bool)
        labels = torch.tensor([0, 1, 2])
        c1 = AdaptiveEventCardinality(variant="C1")
        c2 = AdaptiveEventCardinality(variant="C2")
        self.assertNotIn("loss_count_con", c1(text, text_mask, events, event_mask, labels))
        self.assertIn("loss_count_con", c2(text, text_mask, events, event_mask, labels))


if __name__ == "__main__":
    unittest.main()

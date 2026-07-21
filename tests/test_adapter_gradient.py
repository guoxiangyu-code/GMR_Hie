import unittest

import torch

from models.flash_vtg_gmr.event_adapter import ProposalToEventAdapter


class AdapterGradientContractTest(unittest.TestCase):
    def test_relation_encoder_receives_event_gradient(self):
        adapter = ProposalToEventAdapter()
        output = adapter(
            torch.randn(1, 5, 256),
            torch.ones(1, 5, dtype=torch.bool),
            torch.rand(1, 5, 2),
            torch.randn(1, 5),
            torch.ones(1, 5),
            torch.randn(1, 256),
        )
        output["event_logit"][output["event_mask"]].sum().backward()
        total = sum(
            float(parameter.grad.abs().sum())
            for parameter in adapter.relation_encoder.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(total, 0.0)


if __name__ == "__main__":
    unittest.main()

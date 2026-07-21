import types
import unittest

import torch
from torch import nn

from models.flash_vtg_gmr.model import FlashVTG


class PublicAdapterContractTest(unittest.TestCase):
    def test_frozen_b0_and_p0_stay_in_eval_mode(self):
        model = FlashVTG.__new__(FlashVTG)
        nn.Module.__init__(model)
        model.args = types.SimpleNamespace(freeze_backbone=True, freeze_adapter=True)
        model.backbone_dropout = nn.Dropout(0.5)
        model.event_adapter = nn.Sequential(nn.Dropout(0.5), nn.Linear(2, 2))
        model.aec = nn.Linear(2, 2)
        model.train()
        self.assertFalse(model.backbone_dropout.training)
        self.assertFalse(model.event_adapter.training)
        self.assertTrue(model.aec.training)


if __name__ == "__main__":
    unittest.main()

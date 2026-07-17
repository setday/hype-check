"""CFRNet: TARNet + linear MMD representation balancing."""

from typing import Dict, Tuple

import torch
import torch.nn as nn

from src.models.neural.base import NeuralUpliftModel
from src.models.neural.losses import factual_outcome_loss, linear_mmd
from src.models.neural.tarnet import _TARNetCore


class CFRNet(NeuralUpliftModel):
    display_name = "CFRNet"

    def __init__(self, config: dict):
        super().__init__(config)
        self.mmd_weight = float(config.get("mmd_weight", 1.0))

    def build_network(self, input_dim: int) -> nn.Module:
        return _TARNetCore(input_dim, self.hidden_dim, self.n_layers, self.dropout)

    def predict_outputs(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.net(x)

    def training_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        out = self.net(batch["features"])
        reg = factual_outcome_loss(out["y0"], out["y1"], batch["outcome"], batch["treatment"])
        mmd = linear_mmd(out["rep"], batch["treatment"])
        loss = reg + self.mmd_weight * mmd
        return loss, {"reg": reg.item(), "mmd": mmd.item()}

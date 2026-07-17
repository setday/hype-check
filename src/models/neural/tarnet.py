"""TARNet: shared encoder with separate Y0/Y1 heads."""

from typing import Dict, Tuple

import torch
import torch.nn as nn

from src.models.neural.base import NeuralUpliftModel, SharedMLP
from src.models.neural.losses import factual_outcome_loss


class _TARNetCore(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int, dropout: float):
        super().__init__()
        self.shared = SharedMLP(input_dim, hidden_dim, n_layers, dropout)
        self.y0 = nn.Linear(hidden_dim, 1)
        self.y1 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.shared.encode(x)
        y0 = torch.sigmoid(self.y0(h))
        y1 = torch.sigmoid(self.y1(h))
        return {"y0": y0, "y1": y1, "cate_pred": y1 - y0, "rep": h}


class TARNet(NeuralUpliftModel):
    display_name = "TARNet"

    def build_network(self, input_dim: int) -> nn.Module:
        return _TARNetCore(input_dim, self.hidden_dim, self.n_layers, self.dropout)

    def predict_outputs(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.net(x)

    def training_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        out = self.net(batch["features"])
        loss = factual_outcome_loss(out["y0"], out["y1"], batch["outcome"], batch["treatment"])
        return loss, {"reg": loss.item()}

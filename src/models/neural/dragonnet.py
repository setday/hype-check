"""DragonNet: shared representation + Y0/Y1/propensity heads + targeted regularization."""

from typing import Dict, Tuple

import torch
import torch.nn as nn

from src.models.neural.base import NeuralUpliftModel, SharedMLP
from src.models.neural.losses import (
    dragonnet_targeted_regularization,
    factual_outcome_loss,
    propensity_loss,
)


class _DragonNetCore(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int, dropout: float):
        super().__init__()
        self.shared = SharedMLP(input_dim, hidden_dim, n_layers, dropout)
        self.y0 = nn.Linear(hidden_dim, 1)
        self.y1 = nn.Linear(hidden_dim, 1)
        self.t_logit = nn.Linear(hidden_dim, 1)
        self.epsilon = nn.Linear(1, 1, bias=False)
        nn.init.zeros_(self.epsilon.weight)

    def forward(self, x):
        h = self.shared.encode(x)
        y0 = torch.sigmoid(self.y0(h))
        y1 = torch.sigmoid(self.y1(h))
        t_logit = self.t_logit(h)
        t_prob = torch.sigmoid(t_logit)
        cate = y1 - y0
        return {"y0": y0, "y1": y1, "t_logit": t_logit, "t_prob": t_prob, "cate_pred": cate}


class DragonNet(NeuralUpliftModel):
    display_name = "DragonNet"

    def __init__(self, config: dict):
        super().__init__(config)
        self.alpha = float(config.get("alpha", 1.0))
        self.beta = float(config.get("beta", 1.0))

    def build_network(self, input_dim: int) -> nn.Module:
        return _DragonNetCore(input_dim, self.hidden_dim, self.n_layers, self.dropout)

    def predict_outputs(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.net(x)

    def training_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        out = self.net(batch["features"])
        y, t = batch["outcome"], batch["treatment"]
        reg = factual_outcome_loss(out["y0"], out["y1"], y, t)
        prop = propensity_loss(out["t_logit"], t)
        tr = dragonnet_targeted_regularization(out["y0"], out["y1"], out["t_prob"], y, t, self.net.epsilon.weight.view(()))
        loss = reg + self.alpha * prop + self.beta * tr
        return loss, {"reg": reg.item(), "prop": prop.item(), "tr": tr.item()}

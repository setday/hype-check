"""DESCN-style model: entire-space outcome network + selection propensity."""

from typing import Dict, Tuple

import torch
import torch.nn as nn

from src.models.neural.base import NeuralUpliftModel, SharedMLP
from src.models.neural.losses import factual_outcome_loss, propensity_loss


class _DESCNCore(nn.Module):
    """Simplified DESCN: shared encoder, entire-space mu0/mu1, propensity head."""

    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int, dropout: float):
        super().__init__()
        self.shared = SharedMLP(input_dim, hidden_dim, n_layers, dropout)
        self.esn = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.t_logit = nn.Linear(hidden_dim, 1)

    def _outcome(self, h, t):
        return torch.sigmoid(self.esn(torch.cat([h, t], dim=-1)))

    def forward(self, x):
        h = self.shared.encode(x)
        zeros = torch.zeros(len(x), 1, device=x.device)
        ones = torch.ones(len(x), 1, device=x.device)
        y0 = self._outcome(h, zeros)
        y1 = self._outcome(h, ones)
        t_logit = self.t_logit(h)
        return {"y0": y0, "y1": y1, "t_logit": t_logit, "cate_pred": y1 - y0}


class DESCN(NeuralUpliftModel):
    display_name = "DESCN"

    def __init__(self, config: dict):
        super().__init__(config)
        self.alpha = float(config.get("alpha", 0.5))

    def build_network(self, input_dim: int) -> nn.Module:
        return _DESCNCore(input_dim, self.hidden_dim, self.n_layers, self.dropout)

    def predict_outputs(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.net(x)

    def training_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        out = self.net(batch["features"])
        reg = factual_outcome_loss(out["y0"], out["y1"], batch["outcome"], batch["treatment"])
        prop = propensity_loss(out["t_logit"], batch["treatment"])
        loss = reg + self.alpha * prop
        return loss, {"reg": reg.item(), "prop": prop.item()}

"""EFIN-style model: feature encoder + treatment interaction for uplift."""

from typing import Dict, Tuple

import torch
import torch.nn as nn

from src.models.neural.base import NeuralUpliftModel, SharedMLP


class _EFINCore(nn.Module):
    """Simplified EFIN: treatment-aware interaction on shared embeddings."""

    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int, dropout: float):
        super().__init__()
        self.encoder = SharedMLP(input_dim, hidden_dim, n_layers, dropout)
        self.treatment_emb = nn.Embedding(2, hidden_dim)
        self.interaction = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
        )
        self.y0 = nn.Linear(hidden_dim, 1)
        self.y1 = nn.Linear(hidden_dim, 1)

    def forward(self, x, t):
        h = self.encoder.encode(x)
        t_idx = t.long().view(-1)
        t_emb = self.treatment_emb(t_idx)
        h_int = self.interaction(torch.cat([h, h * t_emb], dim=-1))
        y0 = torch.sigmoid(self.y0(h))
        y1 = torch.sigmoid(self.y1(h_int))
        return {"y0": y0, "y1": y1, "cate_pred": y1 - y0}


class EFIN(NeuralUpliftModel):
    display_name = "EFIN"

    def build_network(self, input_dim: int) -> nn.Module:
        return _EFINCore(input_dim, self.hidden_dim, self.n_layers, self.dropout)

    def predict_outputs(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        t0 = torch.zeros(len(x), 1, device=x.device)
        t1 = torch.ones(len(x), 1, device=x.device)
        out0 = self.net(x, t0)
        out1 = self.net(x, t1)
        return {"y0": out0["y0"], "y1": out1["y1"], "cate_pred": out1["y1"] - out0["y0"]}

    def training_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        out = self.net(batch["features"], batch["treatment"])
        y, t = batch["outcome"], batch["treatment"]
        pred = t * out["y1"] + (1.0 - t) * out["y0"]
        loss = nn.functional.binary_cross_entropy(pred, y)
        return loss, {"reg": loss.item()}

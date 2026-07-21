"""TARNetContinuous: simpler continuous treatment variant."""

from typing import Dict, Tuple

import torch
import torch.nn as nn

from src.models.continuous_treatment_model import ContinuousTreatmentModel
from src.models.neural.base import SharedMLP
from src.models.neural.losses import continuous_outcome_loss


class _TARNetContinuousCore(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, treatment_dim: int, n_layers: int, dropout: float):
        super().__init__()
        # Shared representation for features
        self.shared = SharedMLP(input_dim, hidden_dim, n_layers, dropout)

        # Treatment embedding
        self.t_embed = nn.Sequential(
            nn.Linear(1, treatment_dim),
            nn.ELU(),
            nn.Dropout(dropout),
        )

        # Outcome head: [X_rep, T_embed] -> Y_pred
        combined_dim = hidden_dim + treatment_dim
        self.y_pred = nn.Linear(combined_dim, 1)

    def forward(self, x, t):
        """
        Args:
            x: (batch_size, n_features) normalized features
            t: (batch_size, 1) normalized continuous treatment

        Returns:
            Dict with keys: y_pred, cate_pred
        """
        # Shared representation
        h = self.shared.encode(x)

        # Treatment embedding
        t_h = self.t_embed(t)

        # Combined representation
        combined = torch.cat([h, t_h], dim=1)

        # Outcome prediction
        y_pred = self.y_pred(combined)

        return {
            "y_pred": y_pred,
            "cate_pred": y_pred,
        }


class TARNetContinuous(ContinuousTreatmentModel):
    display_name = "TARNetContinuous"

    def build_network(self, input_dim: int) -> nn.Module:
        return _TARNetContinuousCore(
            input_dim,
            self.hidden_dim,
            self.treatment_dim,
            self.n_layers,
            self.dropout,
        )

    def predict_outputs(self, x: torch.Tensor, t: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.net(x, t)

    def training_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        x = batch["features"]
        t = batch["treatment"]
        y = batch["outcome"]

        out = self.net(x, t)
        y_pred = out["y_pred"]

        # Simple MSE outcome loss
        loss = continuous_outcome_loss(y_pred, y)

        return loss, {"outcome": loss.item()}

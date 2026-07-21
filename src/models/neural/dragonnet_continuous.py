"""DragonNetContinuous: continuous treatment variant with T embedding."""

from typing import Dict, Tuple

import torch
import torch.nn as nn

from src.models.continuous_treatment_model import ContinuousTreatmentModel
from src.models.neural.base import SharedMLP
from src.models.neural.losses import (
    continuous_outcome_loss,
    continuous_propensity_loss,
)


class _DragonNetContinuousCore(nn.Module):
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

        # Propensity head: X_rep -> μ(T), log_σ(T)
        self.t_mu = nn.Linear(hidden_dim, 1)
        self.t_log_sigma = nn.Linear(hidden_dim, 1)

    def forward(self, x, t):
        """
        Args:
            x: (batch_size, n_features) normalized features
            t: (batch_size, 1) normalized continuous treatment

        Returns:
            Dict with keys: y_pred, t_mu, t_log_sigma, cate_pred
        """
        # Shared representation
        h = self.shared.encode(x)

        # Treatment embedding
        t_h = self.t_embed(t)

        # Combined representation
        combined = torch.cat([h, t_h], dim=1)

        # Outcome prediction
        y_pred = self.y_pred(combined)

        # Propensity: predict treatment distribution p(T|X)
        t_mu = self.t_mu(h)
        t_log_sigma = self.t_log_sigma(h)

        return {
            "y_pred": y_pred,
            "t_mu": t_mu,
            "t_log_sigma": t_log_sigma,
            "cate_pred": y_pred,  # For compatibility with evaluation harness
        }


class DragonNetContinuous(ContinuousTreatmentModel):
    display_name = "DragonNetContinuous"

    def __init__(self, config: dict):
        super().__init__(config)
        self.alpha = float(config.get("alpha", 1.0))  # Propensity weight

    def build_network(self, input_dim: int) -> nn.Module:
        return _DragonNetContinuousCore(
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
        t_mu = out["t_mu"]
        t_log_sigma = out["t_log_sigma"]

        # Outcome loss
        outcome_loss = continuous_outcome_loss(y_pred, y)

        # Propensity loss
        prop_loss = continuous_propensity_loss(t, t_mu, t_log_sigma)

        # Total loss
        loss = outcome_loss + self.alpha * prop_loss

        return loss, {"outcome": outcome_loss.item(), "prop": prop_loss.item()}

"""Base class for continuous treatment models."""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from src.models.uplift_model import UpliftModel

logger = logging.getLogger(__name__)


def _select_device(preferred: Optional[str] = None) -> torch.device:
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ContinuousTreatmentModel(UpliftModel):
    """
    Base class for continuous treatment (dose-response) models.

    Extends UpliftModel with:
    - Continuous treatment T ∈ ℝ (not just binary 0/1)
    - Treatment embedding layer for learning dose-response relationships
    - Regression-based CATE prediction τ(X, t)
    """

    display_name = "ContinuousTreatment"

    def __init__(self, config: dict):
        super().__init__(config)
        self.seed = int(config.get("seed", 42))
        self.hidden_dim = int(config.get("hidden_dim", 200))
        self.treatment_dim = int(config.get("treatment_dim", 32))
        self.n_layers = int(config.get("n_layers", 3))
        self.dropout = float(config.get("dropout", 0.1))
        self.lr = float(config.get("lr", 1e-3))
        self.weight_decay = float(config.get("weight_decay", 1e-5))
        self.batch_size = int(config.get("batch_size", 256))
        self.max_epochs = int(config.get("max_epochs", 50))
        self.patience = int(config.get("patience", 5))
        self.val_fraction = float(config.get("val_fraction", 0.1))
        self.device = _select_device(config.get("device"))
        self.net: Optional[nn.Module] = None
        self.scaler = StandardScaler()
        self.treatment_scaler = StandardScaler()
        self._is_fitted = False
        self.train_time_s = 0.0
        self.inference_time_s = 0.0

    def build_network(self, input_dim: int) -> nn.Module:
        raise NotImplementedError

    def training_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        raise NotImplementedError

    def predict_outputs(self, x: torch.Tensor, t: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict CATE and other outputs given features and treatment.

        Args:
            x: (batch_size, n_features) normalized features
            t: (batch_size, 1) normalized continuous treatment

        Returns:
            Dict with key 'cate_pred': (batch_size,) predictions
        """
        raise NotImplementedError

    @staticmethod
    def _np(a):
        return a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else np.asarray(a)

    def fit(self, X, T, Y, X_val=None, T_val=None, Y_val=None):
        """
        Fit continuous treatment model.

        Args:
            X: (n_samples, n_features) covariates
            T: (n_samples,) continuous treatment values
            Y: (n_samples,) outcomes
            X_val, T_val, Y_val: optional validation set
        """
        import time
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.model_selection import train_test_split

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # Scale features and treatment
        x = self.scaler.fit_transform(self._np(X).astype(np.float32))
        t = self._np(T).astype(np.float32).ravel()
        t = self.treatment_scaler.fit_transform(t.reshape(-1, 1)).ravel()
        y = self._np(Y).astype(np.float32).ravel()

        if X_val is not None:
            x_va = self.scaler.transform(self._np(X_val).astype(np.float32))
            t_va = self._np(T_val).astype(np.float32).ravel()
            t_va = self.treatment_scaler.transform(t_va.reshape(-1, 1)).ravel()
            y_va = self._np(Y_val).astype(np.float32).ravel()
            x_tr, t_tr, y_tr = x, t, y
        elif self.val_fraction > 0:
            x_tr, x_va, t_tr, t_va, y_tr, y_va = train_test_split(
                x, t, y, test_size=self.val_fraction, random_state=self.seed,
            )
        else:
            x_tr, t_tr, y_tr = x, t, y
            x_va = t_va = y_va = None

        self.net = self.build_network(x.shape[1]).to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Create dataloaders
        ds_tr = TensorDataset(
            torch.as_tensor(x_tr, dtype=torch.float32),
            torch.as_tensor(t_tr, dtype=torch.float32).view(-1, 1),
            torch.as_tensor(y_tr, dtype=torch.float32).view(-1, 1),
        )
        train_loader = DataLoader(ds_tr, batch_size=self.batch_size, shuffle=True)

        val_loader = None
        if x_va is not None:
            ds_va = TensorDataset(
                torch.as_tensor(x_va, dtype=torch.float32),
                torch.as_tensor(t_va, dtype=torch.float32).view(-1, 1),
                torch.as_tensor(y_va, dtype=torch.float32).view(-1, 1),
            )
            val_loader = DataLoader(ds_va, batch_size=self.batch_size, shuffle=False)

        best_state, best_val, bad_epochs = None, float("inf"), 0
        t0 = time.perf_counter()

        for epoch in range(self.max_epochs):
            self.net.train()
            for xb, tb, yb in train_loader:
                xb, tb, yb = xb.to(self.device), tb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                loss, _ = self.training_loss({"features": xb, "treatment": tb, "outcome": yb})
                loss.backward()
                opt.step()

            if val_loader is not None:
                val_loss = self._eval_loss(val_loader)
                if val_loss < best_val - 1e-5:
                    best_val = val_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                    if bad_epochs >= self.patience:
                        break
            else:
                best_state = {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}
                best_val = float("nan")

        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.net.eval()
        self._is_fitted = True
        self.train_time_s = time.perf_counter() - t0
        logger.info("%s trained in %.1fs (best val=%.4f).", self.display_name, self.train_time_s, best_val)
        return self

    def _eval_loss(self, loader):
        self.net.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for xb, tb, yb in loader:
                xb, tb, yb = xb.to(self.device), tb.to(self.device), yb.to(self.device)
                loss, _ = self.training_loss({"features": xb, "treatment": tb, "outcome": yb})
                total += loss.item() * len(xb)
                n += len(xb)
        return total / max(n, 1)

    def predict_cate(self, X, T=None) -> np.ndarray:
        """
        Predict continuous treatment effect τ(X, t).

        If T is provided, predict CATE at specific treatment values.
        If T is not provided, use T=0.5 (default dose) for all samples.

        Args:
            X: (n_samples, n_features) covariates
            T: (n_samples,) optional treatment values. If None, use median treatment.

        Returns:
            (n_samples,) CATE predictions
        """
        import time
        from torch.utils.data import DataLoader, TensorDataset

        if not self._is_fitted:
            raise RuntimeError(f"{self.display_name} must be fitted before predict_cate().")

        x = self.scaler.transform(self._np(X).astype(np.float32))

        if T is None:
            # Default: predict at T=0.5 (median of typical [0, 1] range)
            t = np.full(len(x), 0.5, dtype=np.float32)
        else:
            t = self._np(T).astype(np.float32).ravel()

        # Normalize treatment
        t = self.treatment_scaler.transform(t.reshape(-1, 1)).ravel()

        ds = TensorDataset(
            torch.as_tensor(x, dtype=torch.float32),
            torch.as_tensor(t, dtype=torch.float32).view(-1, 1),
            torch.zeros(len(x), dtype=torch.float32).view(-1, 1),  # dummy Y
        )
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False)

        preds = []
        t0 = time.perf_counter()
        self.net.eval()
        with torch.no_grad():
            for xb, tb, _ in loader:
                xb, tb = xb.to(self.device), tb.to(self.device)
                out = self.predict_outputs(xb, tb)
                preds.append(out["cate_pred"].cpu().numpy())
        self.inference_time_s = time.perf_counter() - t0
        return np.concatenate(preds).ravel()

    def forward(self, features, treatment=None, **batch) -> Dict[str, torch.Tensor]:
        if not self._is_fitted:
            raise RuntimeError(f"{self.display_name} must be fitted before forward().")
        x = features
        t = treatment if treatment is not None else torch.full_like(x[:, :1], 0.5)

        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy()
        else:
            x_np = self._np(x)

        if isinstance(t, torch.Tensor):
            t_np = t.detach().cpu().numpy()
        else:
            t_np = self._np(t)

        x_scaled = torch.as_tensor(self.scaler.transform(x_np.astype(np.float32)), dtype=torch.float32, device=x.device if isinstance(x, torch.Tensor) else self.device)
        t_scaled = torch.as_tensor(self.treatment_scaler.transform(t_np.reshape(-1, 1)).ravel(), dtype=torch.float32, device=t.device if isinstance(t, torch.Tensor) else self.device).view(-1, 1)

        out = self.predict_outputs(x_scaled, t_scaled)
        device = x.device if isinstance(x, torch.Tensor) else self.device
        return {k: v.to(device) if torch.is_tensor(v) else v for k, v in out.items()}

    def calculate_loss(self, batch) -> torch.Tensor:
        if not self._is_fitted:
            raise RuntimeError(f"{self.display_name} uses offline fit(); call fit() first.")
        loss, _ = self.training_loss(batch)
        return loss

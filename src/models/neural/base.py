"""Base class for gradient-trained neural uplift models with offline fit()."""

import logging
import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

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


class SharedMLP(nn.Module):
    """Shared representation + optional multi-head outputs."""

    def __init__(self, input_dim: int, hidden_dim: int = 200, n_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        layers = []
        dim = input_dim
        for _ in range(n_layers):
            layers.extend([nn.Linear(dim, hidden_dim), nn.ELU(), nn.Dropout(dropout)])
            dim = hidden_dim
        self.encoder = nn.Sequential(*layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class NeuralUpliftModel(UpliftModel):
    """Offline fit/predict_cate wrapper around a PyTorch uplift network."""

    display_name = "NeuralUplift"

    def __init__(self, config: dict):
        super().__init__(config)
        self.seed = int(config.get("seed", 42))
        self.hidden_dim = int(config.get("hidden_dim", 200))
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
        self._is_fitted = False
        self.train_time_s = 0.0
        self.inference_time_s = 0.0

    def build_network(self, input_dim: int) -> nn.Module:
        raise NotImplementedError

    def training_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        raise NotImplementedError

    def predict_outputs(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    @staticmethod
    def _np(a):
        return a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else np.asarray(a)

    def _make_loader(self, x, t, y, shuffle: bool) -> DataLoader:
        ds = TensorDataset(
            torch.as_tensor(x, dtype=torch.float32),
            torch.as_tensor(t, dtype=torch.float32).view(-1, 1),
            torch.as_tensor(y, dtype=torch.float32).view(-1, 1),
        )
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle)

    def fit(self, X, T, Y, X_val=None, T_val=None, Y_val=None):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        x = self.scaler.fit_transform(self._np(X).astype(np.float32))
        t = self._np(T).astype(np.float32).ravel()
        y = self._np(Y).astype(np.float32).ravel()

        if X_val is not None:
            x_va = self.scaler.transform(self._np(X_val).astype(np.float32))
            t_va = self._np(T_val).astype(np.float32).ravel()
            y_va = self._np(Y_val).astype(np.float32).ravel()
            x_tr, t_tr, y_tr = x, t, y
        elif self.val_fraction > 0:
            x_tr, x_va, t_tr, t_va, y_tr, y_va = train_test_split(
                x, t, y, test_size=self.val_fraction, random_state=self.seed, stratify=t,
            )
        else:
            x_tr, t_tr, y_tr = x, t, y
            x_va = t_va = y_va = None

        self.net = self.build_network(x.shape[1]).to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        train_loader = self._make_loader(x_tr, t_tr, y_tr, shuffle=True)
        val_loader = None
        if x_va is not None:
            val_loader = self._make_loader(x_va, t_va, y_va, shuffle=False)

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

    def _eval_loss(self, loader: DataLoader) -> float:
        self.net.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for xb, tb, yb in loader:
                xb, tb, yb = xb.to(self.device), tb.to(self.device), yb.to(self.device)
                loss, _ = self.training_loss({"features": xb, "treatment": tb, "outcome": yb})
                total += loss.item() * len(xb)
                n += len(xb)
        return total / max(n, 1)

    def predict_cate(self, X) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError(f"{self.display_name} must be fitted before predict_cate().")
        x = self.scaler.transform(self._np(X).astype(np.float32))
        loader = self._make_loader(x, np.zeros(len(x)), np.zeros(len(x)), shuffle=False)
        preds = []
        t0 = time.perf_counter()
        self.net.eval()
        with torch.no_grad():
            for xb, _, _ in loader:
                out = self.predict_outputs(xb.to(self.device))
                preds.append(out["cate_pred"].cpu().numpy())
        self.inference_time_s = time.perf_counter() - t0
        return np.concatenate(preds).ravel()

    def forward(self, features, treatment=None, **batch) -> Dict[str, torch.Tensor]:
        if not self._is_fitted:
            raise RuntimeError(f"{self.display_name} must be fitted before forward().")
        x = features
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy()
        else:
            x_np = self._np(x)
        x_scaled = torch.as_tensor(self.scaler.transform(x_np.astype(np.float32)), dtype=torch.float32, device=features.device if isinstance(features, torch.Tensor) else self.device)
        out = self.predict_outputs(x_scaled)
        device = features.device if isinstance(features, torch.Tensor) else self.device
        return {k: v.to(device) if torch.is_tensor(v) else v for k, v in out.items()}

    def calculate_loss(self, batch) -> torch.Tensor:
        if not self._is_fitted:
            raise RuntimeError(f"{self.display_name} uses offline fit(); call fit() first.")
        loss, _ = self.training_loss(batch)
        return loss

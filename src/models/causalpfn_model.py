"""
CausalPFN Model Wrapper
=======================

Wraps the pretrained CausalPFN foundation model for inference.

Reference: https://github.com/vdblm/CausalPFN
Paper: https://arxiv.org/abs/2506.07918
"""

import logging
from typing import Dict

import numpy as np
import torch

from src.models.uplift_model import FrozenFoundationModel

logger = logging.getLogger(__name__)


class CausalPFNModel(FrozenFoundationModel):
    """
    Wrapper for CausalPFN inference.

    CausalPFN is a foundation model trained to estimate treatment effects
    directly from raw data without fine-tuning.

    Args:
        config: dict with:
            - model_name: HuggingFace model ID (e.g., "vdblm/CausalPFN-v1")
            - device: "cuda" or "cpu"
            - batch_size: inference batch size
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.model_name = config.get("model_name", "vdblm/causalpfn-v1")
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = config.get("batch_size", 32)

        # Lazy loading: model initialized on first forward pass
        self.model = None
        self._is_loaded = False

    def _load_model(self):
        """
        Load pretrained CausalPFN model from HuggingFace Hub.
        """
        if self._is_loaded:
            return

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            logger.info(f"Loading CausalPFN from {self.model_name}...")
            # Note: This is a placeholder; actual implementation depends on CausalPFN's API
            # The model may not be on HF Hub yet; in that case, load from GitHub
            self.model = self._load_from_github()
            self.model.to(self.device)
            self.model.eval()
            self._is_loaded = True
            logger.info("CausalPFN model loaded successfully.")

        except ImportError:
            logger.error("transformers library not installed. Install with: pip install transformers")
            raise

    def _load_from_github(self):
        """
        Load CausalPFN from GitHub (fallback if not on HF Hub).
        Placeholder for actual loading logic.
        """
        logger.warning("Loading CausalPFN from GitHub (placeholder).")
        # TODO: Implement actual loading from https://github.com/vdblm/CausalPFN
        # This requires CausalPFN to be installed as a submodule or dependency
        raise NotImplementedError(
            "CausalPFN loading from GitHub not yet implemented. "
            "Install CausalPFN from: https://github.com/vdblm/CausalPFN"
        )

    def forward(self, features: torch.Tensor, treatment: torch.Tensor = None, **batch) -> Dict[str, torch.Tensor]:
        """
        Predict CATE using CausalPFN.

        Args:
            features: (batch_size, n_features)
            treatment: (batch_size,) treatment indicator
            **batch: additional fields (outcome, etc.)

        Returns:
            dict with "cate_pred": (batch_size,) predicted CATE
        """
        # Load model on first call
        if not self._is_loaded:
            self._load_model()

        with torch.no_grad():
            # Prepare input: features + treatment
            if treatment is not None:
                treatment_1d = treatment.unsqueeze(-1) if treatment.dim() == 1 else treatment
                # CausalPFN expects features and treatment as separate inputs
                # Placeholder: actual API depends on CausalPFN implementation
                X_np = features.cpu().numpy() if isinstance(features, torch.Tensor) else features
                T_np = treatment.cpu().numpy() if isinstance(treatment, torch.Tensor) else treatment

                # Batch inference
                cate_preds = []
                for i in range(0, len(X_np), self.batch_size):
                    batch_X = X_np[i : i + self.batch_size]
                    batch_T = T_np[i : i + self.batch_size]

                    # Placeholder: call actual CausalPFN API
                    batch_cate = self._predict_batch(batch_X, batch_T)
                    cate_preds.append(batch_cate)

                cate_pred = np.concatenate(cate_preds, axis=0)
            else:
                # If no treatment given, assume T=1 vs T=0 comparison
                X_np = features.cpu().numpy() if isinstance(features, torch.Tensor) else features
                cate_preds = []
                for i in range(0, len(X_np), self.batch_size):
                    batch_X = X_np[i : i + self.batch_size]
                    batch_cate = self._predict_batch_default(batch_X)
                    cate_preds.append(batch_cate)
                cate_pred = np.concatenate(cate_preds, axis=0)

        return {"cate_pred": torch.from_numpy(cate_pred).float().to(features.device)}

    def _predict_batch(self, X: np.ndarray, T: np.ndarray) -> np.ndarray:
        """
        Predict CATE for a batch using CausalPFN.
        Placeholder for actual implementation.

        Args:
            X: (batch_size, n_features)
            T: (batch_size,)

        Returns:
            (batch_size,) CATE predictions
        """
        # TODO: Implement actual CausalPFN inference
        # Placeholder: random predictions for development
        logger.warning("CausalPFN._predict_batch not implemented. Returning placeholder predictions.")
        return np.random.normal(0, 0.1, len(X)).astype(np.float32)

    def _predict_batch_default(self, X: np.ndarray) -> np.ndarray:
        """
        Predict CATE assuming T=1 vs T=0.
        Placeholder for actual implementation.
        """
        logger.warning("CausalPFN._predict_batch_default not implemented. Returning placeholder predictions.")
        return np.random.normal(0, 0.1, len(X)).astype(np.float32)

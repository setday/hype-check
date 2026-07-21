"""
Uplift Dataset Base Class
==========================

Extends BaseDataset for causal/uplift modeling.
Standardizes treatment, control, and outcome representation.
Supports optional ground-truth CATE for semi-synthetic benchmarks.

Index format:
  {
    "features": array-like (n_features,),
    "treatment": int or float,
    "outcome": float,
    "cate_true": float (optional, for semi-synthetic),
  }
"""

import logging
import numpy as np
from typing import Optional, List, Dict, Any

import torch
from src.datasets.base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class UpliftDataset(BaseDataset):
    """
    Base class for uplift modeling datasets.

    Enforces index structure with treatment, outcome, and optional ground-truth CATE.
    Provides utility methods for treatment/control separation and validation.
    """

    def __init__(
        self,
        index: List[Dict[str, Any]],
        limit: Optional[int] = None,
        shuffle_index: bool = False,
        instance_transforms: Optional[Dict] = None,
        feature_dtype: type = np.float32,
    ):
        """
        Args:
            index: list of dicts with keys {features, treatment, outcome, [cate_true]}
            limit: max number of samples
            shuffle_index: whether to shuffle
            instance_transforms: per-key transforms (not yet implemented)
            feature_dtype: dtype for feature arrays (default float32)
        """
        self.feature_dtype = feature_dtype
        super().__init__(
            index=index,
            limit=limit,
            shuffle_index=shuffle_index,
            instance_transforms=instance_transforms,
        )

    def preprocess_data(self, instance_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert to torch tensors.
        - features: (n_features,) -> float32 tensor
        - treatment: scalar -> long tensor (0 or 1, binary) or float32 (continuous)
        - outcome: scalar -> float32 tensor
        - cate_true: scalar -> float32 tensor (if present)
        """
        result = {}

        if "features" in instance_data:
            features = instance_data["features"]
            if isinstance(features, np.ndarray):
                result["features"] = torch.from_numpy(
                    features.astype(self.feature_dtype)
                )
            elif isinstance(features, list):
                result["features"] = torch.tensor(
                    features, dtype=torch.float32
                )
            else:
                result["features"] = torch.tensor(
                    features, dtype=torch.float32
                )

        if "treatment" in instance_data:
            t = instance_data["treatment"]
            # Infer treatment type: if float in [0,1] range, treat as continuous; else binary
            t_float = float(t)
            if isinstance(t, float) or (isinstance(t, (int, np.integer)) and not (t == 0 or t == 1)):
                result["treatment"] = torch.tensor(t_float, dtype=torch.float32)
            else:
                result["treatment"] = torch.tensor(int(t), dtype=torch.long)

        if "outcome" in instance_data:
            y = instance_data["outcome"]
            result["outcome"] = torch.tensor(y, dtype=torch.float32)

        if "cate_true" in instance_data:
            cate = instance_data["cate_true"]
            result["cate_true"] = torch.tensor(cate, dtype=torch.float32)

        return result

    @staticmethod
    def _assert_index_is_valid(index: List[Dict[str, Any]]) -> None:
        """
        Validate uplift index structure.
        Supports both binary (0, 1) and continuous (float) treatments.
        """
        if not index:
            raise ValueError("Index cannot be empty.")

        required_keys = {"features", "treatment", "outcome"}
        for i, entry in enumerate(index):
            if not isinstance(entry, dict):
                raise TypeError(f"Index entry {i} must be a dict, got {type(entry)}")

            missing = required_keys - set(entry.keys())
            if missing:
                raise ValueError(
                    f"Index entry {i} missing required keys: {missing}"
                )

            t = entry.get("treatment")
            # Accept both binary (0, 1) and continuous (float) treatments
            if not isinstance(t, (int, float, np.integer, np.floating)):
                raise ValueError(
                    f"Index entry {i}: treatment must be numeric; got {type(t).__name__}"
                )

        logger.info(f"Index validated: {len(index)} entries")

    def get_treatment_control_split(self) -> tuple:
        """
        Split dataset into treatment and control groups.

        Returns:
            (treatment_indices, control_indices): lists of indices
        """
        treatment_idx = []
        control_idx = []
        for i, entry in enumerate(self._index):
            if entry["treatment"] == 1:
                treatment_idx.append(i)
            else:
                control_idx.append(i)
        return treatment_idx, control_idx

    def get_statistics(self) -> Dict[str, Any]:
        """
        Compute and return dataset statistics.
        """
        n = len(self._index)
        n_treatment = sum(1 for e in self._index if e["treatment"] == 1)
        n_control = n - n_treatment

        outcomes = [e["outcome"] for e in self._index]
        mean_outcome = np.mean(outcomes)
        std_outcome = np.std(outcomes)

        treatment_outcomes = [
            e["outcome"] for e in self._index if e["treatment"] == 1
        ]
        control_outcomes = [
            e["outcome"] for e in self._index if e["treatment"] == 0
        ]

        mean_y1 = np.mean(treatment_outcomes) if treatment_outcomes else None
        mean_y0 = np.mean(control_outcomes) if control_outcomes else None
        naive_ate = (mean_y1 - mean_y0) if (mean_y1 is not None and mean_y0 is not None) else None

        stats = {
            "n_samples": n,
            "n_treatment": n_treatment,
            "n_control": n_control,
            "treatment_ratio": n_treatment / n if n > 0 else 0,
            "mean_outcome": mean_outcome,
            "std_outcome": std_outcome,
            "mean_y1": mean_y1,
            "mean_y0": mean_y0,
            "naive_ate": naive_ate,
            "n_features": len(self._index[0]["features"]) if self._index else 0,
        }

        if any("cate_true" in e for e in self._index):
            cate_values = [
                e["cate_true"] for e in self._index if "cate_true" in e
            ]
            stats["has_cate_ground_truth"] = True
            stats["mean_cate_true"] = np.mean(cate_values)
            stats["std_cate_true"] = np.std(cate_values)
        else:
            stats["has_cate_ground_truth"] = False

        return stats

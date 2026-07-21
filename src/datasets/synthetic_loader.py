"""Synthetic dose-response dataset loader."""

import logging
from typing import Optional, List, Dict, Any

from src.datasets.synthetic_dose_response import (
    generate_linear_dose_response,
    generate_nonlinear_dose_response,
    generate_heterogeneous_dose_response,
    create_dataset_index,
)
from src.datasets.uplift_dataset import UpliftDataset

logger = logging.getLogger(__name__)


class SyntheticDoseResponseDataset(UpliftDataset):
    """
    Synthetic continuous-treatment dataset with known ground-truth CATE.

    Supports three data-generating processes:
    - linear: Y = base + α·t + β·t² + noise
    - nonlinear: Y = f(X) + g(t) + h(X)·t + noise
    - heterogeneous: Y = base + (1 + X₁)·t² + noise
    """

    def __init__(
        self,
        scenario: str = "linear",
        n_samples: int = 1000,
        n_features: int = 10,
        seed: int = 42,
        split: str = "train",
        limit: Optional[int] = None,
        shuffle_index: bool = False,
        feature_dtype: type = None,
    ):
        """
        Args:
            scenario: 'linear', 'nonlinear', or 'heterogeneous'
            n_samples: Number of samples to generate
            n_features: Number of covariates
            seed: Random seed
            split: 'train' (70%) or 'test' (30%), applied after generation
            limit: Max samples (applied after split)
            shuffle_index: Whether to shuffle the index
            feature_dtype: numpy dtype for features
        """
        import numpy as np

        if feature_dtype is None:
            feature_dtype = np.float32

        # Generate data
        if scenario == "linear":
            X, T, Y, tau_true = generate_linear_dose_response(
                n_samples=n_samples,
                n_features=n_features,
                seed=seed,
            )
        elif scenario == "nonlinear":
            X, T, Y, tau_true = generate_nonlinear_dose_response(
                n_samples=n_samples,
                n_features=n_features,
                seed=seed,
            )
        elif scenario == "heterogeneous":
            X, T, Y, tau_true = generate_heterogeneous_dose_response(
                n_samples=n_samples,
                n_features=n_features,
                seed=seed,
            )
        else:
            raise ValueError(f"Unknown scenario: {scenario}")

        # Create index
        index = create_dataset_index(X, T, Y, tau_true)

        # Apply split
        if split == "train":
            split_idx = int(0.7 * len(index))
            index = index[:split_idx]
        elif split == "test":
            split_idx = int(0.7 * len(index))
            index = index[split_idx:]
        elif split != "full":
            raise ValueError(f"Unknown split: {split}")

        logger.info(f"Generated {scenario} synthetic dataset: {len(index)} samples, {n_features} features")

        super().__init__(
            index=index,
            limit=limit,
            shuffle_index=shuffle_index,
            feature_dtype=feature_dtype,
        )

"""
Uplift Dataset Utilities
=========================

Functions for loading, caching, and processing real RCT and semi-synthetic datasets.
Supports controlled subsampling by regimes (control ratio, target rows, conversion).
"""

import logging
import os
import pickle
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Root data directory
DATA_ROOT = Path(__file__).parent.parent.parent / "data" / "raw"
DATA_ROOT.mkdir(parents=True, exist_ok=True)


def get_cache_path(dataset_name: str, **kwargs) -> Path:
    """
    Generate deterministic cache path for a dataset with given parameters.
    Hash kwargs to create unique identifier.
    """
    kwargs_str = "_".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
    hash_str = hashlib.md5(kwargs_str.encode()).hexdigest()[:8]
    cache_dir = DATA_ROOT / dataset_name / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{hash_str}.pkl"


def load_hillstrom_dataset(
    limit: Optional[int] = None,
    control_ratio: float = 1.0,
    target_rows: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Load Hillstrom email marketing RCT dataset.
    Reference: https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html

    Args:
        limit: max samples to load
        control_ratio: downsample control group by this ratio (for regime testing)
        target_rows: if set, resample to this exact number of rows

    Returns:
        list of dicts with {features, treatment, outcome, [cate_true]}
    """
    cache_path = get_cache_path("hillstrom", limit=limit, control_ratio=control_ratio, target_rows=target_rows)
    if cache_path.exists():
        logger.info(f"Loading Hillstrom from cache: {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    # Placeholder: actual URL loading would go here
    # For now, create synthetic data matching Hillstrom structure
    logger.warning("Hillstrom dataset loading not yet implemented. Using synthetic placeholder.")
    np.random.seed(42)

    n_samples = limit or 64000
    n_features = 10

    index = []
    for _ in range(n_samples):
        # Generate synthetic features
        features = np.random.normal(0, 1, n_features).astype(np.float32)
        treatment = np.random.binomial(1, 0.5)

        # Synthetic outcome with treatment effect
        cate_true = np.random.normal(0.1, 0.05) if treatment == 1 else 0
        outcome = float(np.random.binomial(1, 0.1 + cate_true * treatment))

        index.append({
            "features": features,
            "treatment": treatment,
            "outcome": outcome,
            "cate_true": cate_true,
        })

    # Apply subsampling
    if control_ratio < 1.0:
        control_indices = [i for i, e in enumerate(index) if e["treatment"] == 0]
        n_keep = int(len(control_indices) * control_ratio)
        keep_indices = set(np.random.choice(control_indices, n_keep, replace=False))
        index = [e for i, e in enumerate(index) if i not in keep_indices or index[i]["treatment"] == 1]

    if target_rows and len(index) > target_rows:
        indices = np.random.choice(len(index), target_rows, replace=False)
        index = [index[i] for i in indices]

    # Cache and return
    with open(cache_path, "wb") as f:
        pickle.dump(index, f)
    logger.info(f"Hillstrom dataset loaded: {len(index)} samples")
    return index


def load_criteo_dataset(
    limit: Optional[int] = None,
    control_ratio: float = 1.0,
    target_rows: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Load Criteo uplift modeling dataset.
    Reference: https://ailab.criteo.com/criteo-uplift-prediction-dataset/

    Args:
        limit: max samples to load
        control_ratio: downsample control group by this ratio
        target_rows: if set, resample to this exact number of rows

    Returns:
        list of dicts with {features, treatment, outcome}
    """
    cache_path = get_cache_path("criteo", limit=limit, control_ratio=control_ratio, target_rows=target_rows)
    if cache_path.exists():
        logger.info(f"Loading Criteo from cache: {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    logger.warning("Criteo dataset loading not yet implemented. Using synthetic placeholder.")
    np.random.seed(42)

    n_samples = limit or 100000
    n_features = 20

    index = []
    for _ in range(n_samples):
        features = np.random.normal(0, 1, n_features).astype(np.float32)
        treatment = np.random.binomial(1, 0.5)
        outcome = float(np.random.binomial(1, 0.05 + 0.02 * treatment))

        index.append({
            "features": features,
            "treatment": treatment,
            "outcome": outcome,
        })

    if control_ratio < 1.0:
        control_indices = [i for i, e in enumerate(index) if e["treatment"] == 0]
        n_keep = int(len(control_indices) * control_ratio)
        keep_indices = set(np.random.choice(control_indices, n_keep, replace=False))
        index = [e for i, e in enumerate(index) if i not in keep_indices or index[i]["treatment"] == 1]

    if target_rows and len(index) > target_rows:
        indices = np.random.choice(len(index), target_rows, replace=False)
        index = [index[i] for i in indices]

    with open(cache_path, "wb") as f:
        pickle.dump(index, f)
    logger.info(f"Criteo dataset loaded: {len(index)} samples")
    return index


def load_x5_dataset(
    limit: Optional[int] = None,
    control_ratio: float = 1.0,
    target_rows: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Load X5 retail dataset.
    Reference: https://www.uplift-modeling.com/

    Args:
        limit: max samples to load
        control_ratio: downsample control group by this ratio
        target_rows: if set, resample to this exact number of rows

    Returns:
        list of dicts with {features, treatment, outcome}
    """
    cache_path = get_cache_path("x5", limit=limit, control_ratio=control_ratio, target_rows=target_rows)
    if cache_path.exists():
        logger.info(f"Loading X5 from cache: {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    logger.warning("X5 dataset loading not yet implemented. Using synthetic placeholder.")
    np.random.seed(42)

    n_samples = limit or 50000
    n_features = 15

    index = []
    for _ in range(n_samples):
        features = np.random.normal(0, 1, n_features).astype(np.float32)
        treatment = np.random.binomial(1, 0.5)
        outcome = float(np.random.binomial(1, 0.08 + 0.03 * treatment))

        index.append({
            "features": features,
            "treatment": treatment,
            "outcome": outcome,
        })

    if control_ratio < 1.0:
        control_indices = [i for i, e in enumerate(index) if e["treatment"] == 0]
        n_keep = int(len(control_indices) * control_ratio)
        keep_indices = set(np.random.choice(control_indices, n_keep, replace=False))
        index = [e for i, e in enumerate(index) if i not in keep_indices or index[i]["treatment"] == 1]

    if target_rows and len(index) > target_rows:
        indices = np.random.choice(len(index), target_rows, replace=False)
        index = [index[i] for i in indices]

    with open(cache_path, "wb") as f:
        pickle.dump(index, f)
    logger.info(f"X5 dataset loaded: {len(index)} samples")
    return index


def load_ihdp_dataset(
    limit: Optional[int] = None,
    control_ratio: float = 1.0,
    target_rows: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Load IHDP semi-synthetic dataset with ground truth CATE.
    Reference: https://github.com/AMLab-Amsterdam/CEVAE

    Args:
        limit: max samples to load
        control_ratio: downsample control group by this ratio
        target_rows: if set, resample to this exact number of rows

    Returns:
        list of dicts with {features, treatment, outcome, cate_true}
    """
    cache_path = get_cache_path("ihdp", limit=limit, control_ratio=control_ratio, target_rows=target_rows)
    if cache_path.exists():
        logger.info(f"Loading IHDP from cache: {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    logger.warning("IHDP dataset loading not yet implemented. Using synthetic placeholder.")
    np.random.seed(42)

    n_samples = limit or 7000
    n_features = 25

    index = []
    for _ in range(n_samples):
        features = np.random.normal(0, 1, n_features).astype(np.float32)
        treatment = np.random.binomial(1, 0.5)

        # Synthetic CATE and outcome
        cate_true = np.random.normal(2.0, 1.0)
        outcome = float(5.0 + cate_true * treatment + np.random.normal(0, 1))

        index.append({
            "features": features,
            "treatment": treatment,
            "outcome": outcome,
            "cate_true": cate_true,
        })

    if control_ratio < 1.0:
        control_indices = [i for i, e in enumerate(index) if e["treatment"] == 0]
        n_keep = int(len(control_indices) * control_ratio)
        keep_indices = set(np.random.choice(control_indices, n_keep, replace=False))
        index = [e for i, e in enumerate(index) if i not in keep_indices or index[i]["treatment"] == 1]

    if target_rows and len(index) > target_rows:
        indices = np.random.choice(len(index), target_rows, replace=False)
        index = [index[i] for i in indices]

    with open(cache_path, "wb") as f:
        pickle.dump(index, f)
    logger.info(f"IHDP dataset loaded: {len(index)} samples")
    return index


def load_acic_dataset(
    limit: Optional[int] = None,
    control_ratio: float = 1.0,
    target_rows: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Load ACIC 2016 semi-synthetic dataset with ground truth CATE.
    Reference: https://github.com/BiomedSciAI/causallib/tree/master/causallib/datasets/data/acic_challenge_2016

    Args:
        limit: max samples to load
        control_ratio: downsample control group by this ratio
        target_rows: if set, resample to this exact number of rows

    Returns:
        list of dicts with {features, treatment, outcome, cate_true}
    """
    cache_path = get_cache_path("acic", limit=limit, control_ratio=control_ratio, target_rows=target_rows)
    if cache_path.exists():
        logger.info(f"Loading ACIC from cache: {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    logger.warning("ACIC dataset loading not yet implemented. Using synthetic placeholder.")
    np.random.seed(42)

    n_samples = limit or 10000
    n_features = 30

    index = []
    for _ in range(n_samples):
        features = np.random.normal(0, 1, n_features).astype(np.float32)
        treatment = np.random.binomial(1, 0.5)

        # Synthetic CATE and outcome
        cate_true = np.dot(features[:5], np.array([0.5, -0.3, 0.2, 0.1, -0.1]))
        outcome = float(0.0 + cate_true * treatment + np.random.normal(0, 0.5))

        index.append({
            "features": features,
            "treatment": treatment,
            "outcome": outcome,
            "cate_true": cate_true,
        })

    if control_ratio < 1.0:
        control_indices = [i for i, e in enumerate(index) if e["treatment"] == 0]
        n_keep = int(len(control_indices) * control_ratio)
        keep_indices = set(np.random.choice(control_indices, n_keep, replace=False))
        index = [e for i, e in enumerate(index) if i not in keep_indices or index[i]["treatment"] == 1]

    if target_rows and len(index) > target_rows:
        indices = np.random.choice(len(index), target_rows, replace=False)
        index = [index[i] for i in indices]

    with open(cache_path, "wb") as f:
        pickle.dump(index, f)
    logger.info(f"ACIC dataset loaded: {len(index)} samples")
    return index


def bootstrap_resample(
    index: List[Dict[str, Any]],
    n_resamples: int = 100,
    seed: int = 42,
) -> List[List[Dict[str, Any]]]:
    """
    Create bootstrap resamples for confidence interval computation.

    Args:
        index: original dataset index
        n_resamples: number of resamples to generate
        seed: random seed for reproducibility

    Returns:
        list of bootstrap resampled indices
    """
    np.random.seed(seed)
    n = len(index)
    resamples = []

    for _ in range(n_resamples):
        resample_indices = np.random.choice(n, n, replace=True)
        resample = [index[i] for i in resample_indices]
        resamples.append(resample)

    return resamples

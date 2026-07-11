"""Loaders for the cleaned uplift datasets (features.parquet + outcomes.parquet).

Set HYPECHECK_DATA_ROOT to the folder holding the dataset subdirs.
"""

import hashlib
import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DATA_ROOT = Path(os.environ.get("HYPECHECK_DATA_ROOT", Path(__file__).parents[2] / "data"))
CACHE_ROOT = Path(os.environ.get("HYPECHECK_CACHE_ROOT", DATA_ROOT / "cache"))

DATASETS: Dict[str, Dict[str, Any]] = {
    "hillstrom": {"folder": "Hillstrom", "outcome": "visit"},
    "retailhero": {"folder": "Retailhero-uplift", "outcome": "Y"},
    "lzd": {"folder": "LZD", "outcome": "Y"},
    "orange": {"folder": "Orange Telecom Churn", "outcome": "churn"},
    "criteo": {"folder": "Criteo-ITE-v2.1", "outcome": "visit"},
}
ALIASES = {"x5": "retailhero"}

_NON_FEATURES = {"epk_id", "T", "treatment_dt", "lag"}


def _cache_path(name: str, **kwargs) -> Path:
    key = "_".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
    h = hashlib.md5(f"{name}_{key}".encode()).hexdigest()[:10]
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    return CACHE_ROOT / f"{name}_{h}.pkl"


def _encode_features(df: pd.DataFrame, exclude: set) -> Tuple[np.ndarray, List[str]]:
    drop = _NON_FEATURES | set(exclude)
    frame = df[[c for c in df.columns if c not in drop]].copy()

    obj_cols = [c for c in frame.columns if frame[c].dtype == "object" or str(frame[c].dtype) == "category"]
    if obj_cols:
        frame = pd.get_dummies(frame, columns=obj_cols, dummy_na=False)

    frame = frame.apply(pd.to_numeric, errors="coerce")
    frame = frame.fillna(frame.median(numeric_only=True)).fillna(0.0)
    return frame.to_numpy(dtype=np.float32), list(frame.columns)


def load_uplift_arrays(name, outcome=None, data_root=None, limit=None, seed=42):
    """Load a dataset as (X, T, Y, feature_names)."""
    name = ALIASES.get(name, name)
    if name not in DATASETS:
        raise KeyError(f"Unknown dataset '{name}'. Known: {sorted(DATASETS)} (aliases: {sorted(ALIASES)})")

    root = Path(data_root) if data_root is not None else DATA_ROOT
    folder = DATASETS[name]["folder"]
    outcome = outcome or DATASETS[name]["outcome"]

    feats = pd.read_parquet(root / folder / "features.parquet")
    outs = pd.read_parquet(root / folder / "outcomes.parquet")
    if outcome not in outs.columns:
        raise KeyError(f"Outcome '{outcome}' not in {name}: {list(outs.columns)}")

    df = feats.merge(outs[["epk_id", outcome]], on="epk_id", how="inner").dropna(subset=[outcome, "T"])
    if limit is not None and len(df) > limit:
        df = df.sample(n=limit, random_state=seed).reset_index(drop=True)

    # outcome must stay out of X, otherwise CATE collapses to ~0
    X, feature_names = _encode_features(df, exclude={outcome})
    T = df["T"].to_numpy().astype(np.int8)
    Y = df[outcome].to_numpy().astype(np.float32)
    logger.info("Loaded %s: n=%d, features=%d, outcome=%s", name, len(X), X.shape[1], outcome)
    return X, T, Y, feature_names


def arrays_to_index(X, T, Y, cate_true=None) -> List[Dict[str, Any]]:
    """Convert arrays to the UpliftDataset index format."""
    index = []
    for i in range(len(X)):
        entry = {"features": X[i], "treatment": int(T[i]), "outcome": float(Y[i])}
        if cate_true is not None:
            entry["cate_true"] = float(cate_true[i])
        index.append(entry)
    return index


def load_uplift_dataset(name, outcome=None, data_root=None, limit=None, seed=42, use_cache=True):
    """Load a dataset as the UpliftDataset index (list of dicts)."""
    cache = _cache_path(name, outcome=outcome, limit=limit, seed=seed)
    if use_cache and cache.exists():
        with open(cache, "rb") as f:
            return pickle.load(f)

    X, T, Y, _ = load_uplift_arrays(name, outcome=outcome, data_root=data_root, limit=limit, seed=seed)
    index = arrays_to_index(X, T, Y)
    if use_cache:
        with open(cache, "wb") as f:
            pickle.dump(index, f)
    return index


def load_hillstrom_dataset(limit=None, **kwargs):
    return load_uplift_dataset("hillstrom", limit=limit, **kwargs)


def load_criteo_dataset(limit=None, **kwargs):
    return load_uplift_dataset("criteo", limit=limit, **kwargs)


def load_x5_dataset(limit=None, **kwargs):
    return load_uplift_dataset("retailhero", limit=limit, **kwargs)

"""Dataset loading aligned with the fixed W1 experimental protocol."""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DATA_ROOT = Path(os.environ.get("HYPECHECK_DATA_ROOT", Path(__file__).parents[2] / "data"))

DATASETS = {
    "hillstrom_mens": {
        "folder": "Hillstrom",
        "outcome": "visit",
        "task": "mens_vs_none",
    },
    "hillstrom_womens": {
        "folder": "Hillstrom",
        "outcome": "visit",
        "task": "womens_vs_none",
    },
    "x5": {"folder": "Retailhero-uplift", "outcome": "Y", "task": None},
    "retailhero": {"folder": "Retailhero-uplift", "outcome": "Y", "task": None},
    "criteo": {"folder": "Criteo-ITE-v2.1", "outcome": "visit", "task": None},
    "lzd": {"folder": "LZD", "outcome": "Y", "task": None},
}

_NON_FEATURES = {"epk_id", "T", "treatment_dt", "lag", "mens", "womens"}


def _encode_features(df: pd.DataFrame, outcome: str) -> Tuple[np.ndarray, list]:
    exclude = _NON_FEATURES | {outcome}
    frame = df.drop(columns=[c for c in df.columns if c in exclude], errors="ignore").copy()
    obj_cols = frame.select_dtypes(include=["object", "category"]).columns.tolist()
    if obj_cols:
        frame = pd.get_dummies(frame, columns=obj_cols, dummy_na=False)
    frame = frame.apply(pd.to_numeric, errors="coerce")
    frame = frame.fillna(frame.median(numeric_only=True)).fillna(0.0)
    return frame.to_numpy(dtype=np.float32), list(frame.columns)


def _apply_hillstrom_task(df: pd.DataFrame, task: Optional[str]) -> pd.DataFrame:
    if not task:
        return df
    if task == "mens_vs_none":
        # Exclude womens-only arm; do not merge mens and womens letters.
        return df[~((df["mens"] == 0) & (df["womens"] == 1))].copy()
    if task == "womens_vs_none":
        return df[~((df["mens"] == 1) & (df["womens"] == 0))].copy()
    raise ValueError(f"Unknown Hillstrom task '{task}'")


def _cohort_subsample(df: pd.DataFrame, limit: Optional[int], cohort_seed: int) -> pd.DataFrame:
    if limit is None or len(df) <= limit:
        return df.reset_index(drop=True)
    return df.sample(n=limit, random_state=cohort_seed).reset_index(drop=True)


def load_protocol_arrays(
    dataset: str,
    *,
    data_root: Optional[Path] = None,
    cohort_limit: Optional[int] = None,
    cohort_seed: int = 20260715,
    outcome_transform: Optional[str] = None,
    outcome_transform_eps: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Return (X, T, Y_fit, meta). Y_fit may be transformed; meta['y_eval'] is raw Y."""
    if dataset not in DATASETS:
        raise KeyError(f"Unknown dataset '{dataset}'. Known: {sorted(DATASETS)}")

    cfg = DATASETS[dataset]
    root = Path(data_root) if data_root else DATA_ROOT
    folder = cfg["folder"]
    outcome = cfg["outcome"]

    feats = pd.read_parquet(root / folder / "features.parquet")
    outs = pd.read_parquet(root / folder / "outcomes.parquet")
    if outcome not in outs.columns:
        raise KeyError(f"Outcome '{outcome}' missing in {dataset}")

    df = feats.merge(outs[["epk_id", outcome]], on="epk_id", how="inner")
    df = df.dropna(subset=[outcome, "T"])
    df = _apply_hillstrom_task(df, cfg.get("task"))
    df = _cohort_subsample(df, cohort_limit, cohort_seed)

    X, feature_names = _encode_features(df, outcome)
    T = pd.to_numeric(df["T"], errors="coerce").fillna(0).to_numpy().astype(np.int8)
    Y_raw = pd.to_numeric(df[outcome], errors="coerce").fillna(0).to_numpy().astype(np.float32)
    from src.datasets.instance_transforms import transform_outcome_array

    Y_fit = transform_outcome_array(Y_raw, outcome_transform, eps=outcome_transform_eps)

    meta = {
        "dataset": dataset,
        "n": len(X),
        "n_features": X.shape[1],
        "feature_names": feature_names,
        "outcome": outcome,
        "task": cfg.get("task"),
        "cohort_limit": cohort_limit,
        "cohort_seed": cohort_seed,
        "outcome_transform": outcome_transform or "identity",
        "y_eval": Y_raw,
    }
    logger.info(
        "Loaded %s: n=%d, d=%d, treat_share=%.3f, outcome=%s, transform=%s",
        dataset, len(X), X.shape[1], T.mean(), outcome, meta["outcome_transform"],
    )
    return X, T, Y_fit, meta


def split_cache_path(dataset: str, cohort_limit: Optional[int], cohort_seed: int, split_seed: int) -> Path:
    key = f"{dataset}|limit={cohort_limit}|cohort={cohort_seed}|split={split_seed}"
    h = hashlib.md5(key.encode()).hexdigest()[:12]
    cache = DATA_ROOT / "cache" / "splits"
    cache.mkdir(parents=True, exist_ok=True)
    return cache / f"split_{h}.npz"

"""Per-field instance transforms for uplift datasets.

Outcome transforms are applied on **train** only; ranking metrics always use the
original (untransformed) outcome on test / lockbox.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

TransformFn = Callable[[Any], Any]

TRANSFORM_NAMES = ("identity", "log1p", "sqrt", "logit")


def identity(x: Any) -> Any:
    return x


def log1p_outcome(x: Any, *, eps: float = 0.0) -> np.ndarray | float:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 0:
        return float(np.log1p(max(arr.item(), eps)))
    return np.log1p(np.clip(arr, eps, None)).astype(np.float32)


def sqrt_outcome(x: Any, *, eps: float = 0.0) -> np.ndarray | float:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 0:
        return float(np.sqrt(max(arr.item(), eps)))
    return np.sqrt(np.clip(arr, eps, None)).astype(np.float32)


def logit_outcome(x: Any, *, eps: float = 1e-3) -> np.ndarray | float:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 0:
        v = float(np.clip(arr.item(), eps, 1.0 - eps))
        return float(np.log(v / (1.0 - v)))
    clipped = np.clip(arr, eps, 1.0 - eps)
    return np.log(clipped / (1.0 - clipped)).astype(np.float32)


def get_outcome_transform(name: Optional[str], *, eps: float = 1e-3) -> TransformFn:
    if not name or name == "identity":
        return identity
    if name == "log1p":
        return lambda x: log1p_outcome(x)
    if name == "sqrt":
        return lambda x: sqrt_outcome(x)
    if name == "logit":
        return lambda x: logit_outcome(x, eps=eps)
    raise KeyError(f"Unknown outcome transform '{name}'. Known: {TRANSFORM_NAMES}")


def cate_in_transform_space(
    y0: np.ndarray,
    tau: np.ndarray,
    transform: Optional[str],
    *,
    eps: float = 1e-3,
) -> np.ndarray:
    """Structural ITE in transformed outcome space: h(y0 + tau) - h(y0)."""
    y0 = np.asarray(y0, dtype=np.float64)
    tau = np.asarray(tau, dtype=np.float64)
    y1 = y0 + tau
    h0 = transform_outcome_array(y0, transform, eps=eps).astype(np.float64)
    h1 = transform_outcome_array(y1, transform, eps=eps).astype(np.float64)
    return (h1 - h0).astype(np.float32)


def transform_outcome_array(y: np.ndarray, name: Optional[str], *, eps: float = 1e-3) -> np.ndarray:
    """Vectorized outcome transform for (n,) arrays."""
    if not name or name == "identity":
        return np.asarray(y, dtype=np.float32)
    fn = get_outcome_transform(name, eps=eps)
    out = fn(np.asarray(y))
    return np.asarray(out, dtype=np.float32)


def build_train_instance_transforms(
    outcome: Optional[str] = None,
    *,
    eps: float = 1e-3,
) -> Optional[Dict[str, TransformFn]]:
    """Hydra-friendly dict for ControlDataset.train `instance_transforms`."""
    if not outcome or outcome == "identity":
        return None
    fn = get_outcome_transform(outcome, eps=eps)
    return {"outcome": fn}


def apply_instance_transforms_to_arrays(
    features: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    transforms: Optional[Dict[str, TransformFn]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not transforms:
        return features, treatment, outcome
    x, t, y = features, treatment, outcome
    if "features" in transforms:
        x = np.asarray([transforms["features"](v) for v in x], dtype=np.float32)
    if "treatment" in transforms:
        t = np.asarray([transforms["treatment"](v) for v in t])
    if "outcome" in transforms:
        mapped = [transforms["outcome"](v) for v in y]
        y = np.asarray(mapped, dtype=np.float32)
    return x, t, y

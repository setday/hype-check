"""Synthetic dose-response dataset generators with known ground-truth CATE."""

import numpy as np
from typing import Tuple, Dict, List, Any


def generate_linear_dose_response(
    n_samples: int = 1000,
    n_features: int = 10,
    seed: int = 42,
    noise_scale: float = 0.1,
    alpha: float = 1.0,
    beta: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Linear dose-response: Y = β₀ + β₁X + τ(t) + ε
    where τ(t) = α·t + β·t²

    Returns:
        X: (n_samples, n_features) covariates
        T: (n_samples,) continuous treatment ∈ [0, 1]
        Y: (n_samples,) outcomes
        tau_true: (n_samples,) ground-truth CATE τ(X, t)
    """
    rng = np.random.RandomState(seed)

    X = rng.randn(n_samples, n_features)
    T = rng.uniform(0, 1, n_samples)

    # Linear effect on outcome
    base = np.sum(X[:, :3], axis=1)  # Use first 3 features

    # Dose-response effect: τ(t) = α·t + β·t²
    tau_true = alpha * T + beta * T**2

    # Outcome = base + dose-response + noise
    Y = base + tau_true + noise_scale * rng.randn(n_samples)

    return X.astype(np.float32), T.astype(np.float32), Y.astype(np.float32), tau_true.astype(np.float32)


def generate_nonlinear_dose_response(
    n_samples: int = 1000,
    n_features: int = 10,
    seed: int = 42,
    noise_scale: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Nonlinear dose-response: Y = f(X) + g(t) + h(X)·t + ε
    where f(X) = sin(X₁) + cos(X₂)
          g(t) = sin(π·t)
          h(X) = |X₃|

    Returns:
        X: (n_samples, n_features) covariates
        T: (n_samples,) continuous treatment ∈ [0, 1]
        Y: (n_samples,) outcomes
        tau_true: (n_samples,) ground-truth CATE τ(X, t)
    """
    rng = np.random.RandomState(seed)

    X = rng.randn(n_samples, n_features)
    T = rng.uniform(0, 1, n_samples)

    # Nonlinear base effects
    f_x = np.sin(X[:, 0]) + np.cos(X[:, 1])
    g_t = np.sin(np.pi * T)
    h_x = np.abs(X[:, 2])

    # Dose-response effect (nonlinear, heterogeneous)
    tau_true = g_t + h_x * T

    # Outcome
    Y = f_x + tau_true + noise_scale * rng.randn(n_samples)

    return X.astype(np.float32), T.astype(np.float32), Y.astype(np.float32), tau_true.astype(np.float32)


def generate_heterogeneous_dose_response(
    n_samples: int = 1000,
    n_features: int = 10,
    seed: int = 42,
    noise_scale: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Heterogeneous dose-response: τ(X, t) depends on X
    Y = X₀ + τ(X, t) + ε
    where τ(X, t) = (1 + X₁) * t^2

    Returns:
        X: (n_samples, n_features) covariates
        T: (n_samples,) continuous treatment ∈ [0, 1]
        Y: (n_samples,) outcomes
        tau_true: (n_samples,) ground-truth CATE τ(X, t)
    """
    rng = np.random.RandomState(seed)

    X = rng.randn(n_samples, n_features)
    T = rng.uniform(0, 1, n_samples)

    # Heterogeneous CATE: depends on X₁
    tau_true = (1 + X[:, 1]) * T**2

    # Outcome
    Y = X[:, 0] + tau_true + noise_scale * rng.randn(n_samples)

    return X.astype(np.float32), T.astype(np.float32), Y.astype(np.float32), tau_true.astype(np.float32)


def create_dataset_index(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    tau_true: np.ndarray,
) -> List[Dict[str, Any]]:
    """
    Convert arrays to uplift dataset index format.

    Args:
        X: (n_samples, n_features)
        T: (n_samples,) continuous treatment
        Y: (n_samples,) outcomes
        tau_true: (n_samples,) ground-truth CATE

    Returns:
        List of dicts with keys {features, treatment, outcome, cate_true}
    """
    index = []
    for i in range(len(X)):
        index.append({
            "features": X[i],
            "treatment": float(T[i]),
            "outcome": float(Y[i]),
            "cate_true": float(tau_true[i]),
        })
    return index


# Utility function to generate multiple scenarios for benchmarking
def generate_continuous_datasets(seed: int = 42, n_samples: int = 1000) -> Dict[str, Tuple]:
    """
    Generate multiple continuous-treatment datasets for evaluation.

    Returns:
        Dict with keys {linear, nonlinear, heterogeneous}
        Each value is (X, T, Y, tau_true) tuple
    """
    return {
        "linear": generate_linear_dose_response(n_samples=n_samples, seed=seed),
        "nonlinear": generate_nonlinear_dose_response(n_samples=n_samples, seed=seed),
        "heterogeneous": generate_heterogeneous_dose_response(n_samples=n_samples, seed=seed),
    }

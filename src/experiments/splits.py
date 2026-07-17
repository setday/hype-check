"""Fixed train/lockbox splits and CV folds for uplift experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split


@dataclass(frozen=True)
class SplitBundle:
    """Train/lockbox indices plus shared CV folds on train."""

    train_idx: np.ndarray
    test_idx: np.ndarray
    folds: Tuple[np.ndarray, ...]  # each element = val indices within train


def stratify_ty(treatment: np.ndarray, outcome: np.ndarray) -> np.ndarray:
    t = np.asarray(treatment).astype(int).ravel()
    y = np.asarray(outcome).astype(int).ravel()
    return t * 2 + y


def make_lockbox_split(
    n: int,
    treatment: np.ndarray,
    outcome: np.ndarray,
    *,
    test_size: float = 0.2,
    split_seed: int = 20260716,
) -> Tuple[np.ndarray, np.ndarray]:
    labels = stratify_ty(treatment, outcome)
    idx = np.arange(n)
    train_idx, test_idx = train_test_split(
        idx,
        test_size=test_size,
        random_state=split_seed,
        stratify=labels,
    )
    return train_idx, test_idx


def make_cv_folds(
    train_idx: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    *,
    n_splits: int = 3,
    cv_seed: int = 42,
) -> Tuple[np.ndarray, ...]:
    labels = stratify_ty(treatment[train_idx], outcome[train_idx])
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cv_seed)
    folds = []
    for _, val_rel in skf.split(train_idx, labels):
        folds.append(train_idx[val_rel])
    return tuple(folds)


def build_split_bundle(
    treatment: np.ndarray,
    outcome: np.ndarray,
    *,
    test_size: float = 0.2,
    split_seed: int = 20260716,
    n_splits: int = 3,
    cv_seed: int = 42,
) -> SplitBundle:
    n = len(treatment)
    train_idx, test_idx = make_lockbox_split(
        n, treatment, outcome, test_size=test_size, split_seed=split_seed,
    )
    folds = make_cv_folds(
        train_idx, treatment, outcome, n_splits=n_splits, cv_seed=cv_seed,
    )
    return SplitBundle(train_idx=train_idx, test_idx=test_idx, folds=folds)


def iter_cv_train_val(
    train_idx: np.ndarray,
    folds: Tuple[np.ndarray, ...],
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    train_set = set(train_idx.tolist())
    for val_idx in folds:
        val_set = set(val_idx.tolist())
        fold_train = np.array(sorted(train_set - val_set), dtype=int)
        yield fold_train, val_idx

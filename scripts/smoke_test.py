#!/usr/bin/env python3
"""
Smoke Test: Hillstrom Dataset with S-Learner and CausalPFN
============================================================

Quick validation that the pipeline works end-to-end:
1. Load Hillstrom dataset
2. Run S-Learner baseline
3. Run CausalPFN (in-context)
4. Verify Qini scores are computed

Usage:
    python scripts/smoke_test.py [--dataset hillstrom] [--models s_learner,causalpfn]
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import torch
from torch.utils.data import DataLoader

# Add repo root to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.datasets.uplift_dataset import UpliftDataset
from src.datasets.uplift_data_utils import (
    load_hillstrom_dataset,
    load_criteo_dataset,
)
from src.models.uplift_model import SLearnerWrapper, TLearnerWrapper
from src.models.causalpfn_model import CausalPFNModel
from src.metrics.ranking import qini_coefficient
from src.datasets.collate import collate_fn

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_dataset(dataset_name: str, split: str = "train") -> UpliftDataset:
    """
    Load dataset and return UpliftDataset.
    """
    logger.info(f"Loading {dataset_name} ({split} split)...")

    if dataset_name == "hillstrom":
        index = load_hillstrom_dataset(limit=1000)
    elif dataset_name == "criteo":
        index = load_criteo_dataset(limit=1000)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Train/val/test split
    n = len(index)
    if split == "train":
        indices = list(range(0, int(0.6 * n)))
    elif split == "val":
        indices = list(range(int(0.6 * n), int(0.8 * n)))
    else:  # test
        indices = list(range(int(0.8 * n), n))

    subset_index = [index[i] for i in indices]
    dataset = UpliftDataset(subset_index)

    logger.info(f"Loaded {len(dataset)} samples ({split} split)")
    stats = dataset.get_statistics()
    logger.info(f"Dataset stats: {stats}")

    return dataset


def test_s_learner(train_dataset: UpliftDataset, val_dataset: UpliftDataset) -> Dict[str, Any]:
    """
    Test S-Learner baseline.
    """
    logger.info("Testing S-Learner...")

    config = {
        "params": {
            "n_estimators": 50,
            "max_depth": 3,
            "learning_rate": 0.1,
            "random_state": 42,
        }
    }

    model = SLearnerWrapper(config)

    # Prepare training data
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        collate_fn=collate_fn,
        shuffle=True,
    )

    # Extract all training data for fitting
    X_list, T_list, Y_list = [], [], []
    for batch in train_loader:
        X_list.append(batch["features"].numpy())
        T_list.append(batch["treatment"].numpy())
        Y_list.append(batch["outcome"].numpy())

    X_train = np.concatenate(X_list, axis=0)
    T_train = np.concatenate(T_list, axis=0)
    Y_train = np.concatenate(Y_list, axis=0)

    # Fit model
    model.fit(
        X=torch.from_numpy(X_train).float(),
        T=torch.from_numpy(T_train).long(),
        Y=torch.from_numpy(Y_train).float(),
    )

    # Evaluate on validation set
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        collate_fn=collate_fn,
        shuffle=False,
    )

    qini = _eval_qini(model, val_loader)
    logger.info(f"S-Learner Qini: {qini:.4f}")

    return {"model": "s_learner", "qini": qini}


def _eval_qini(model, val_loader) -> float:
    """Collect CATE predictions over a loader and return the Qini coefficient."""
    cate, out, treat = [], [], []
    with torch.no_grad():
        for batch in val_loader:
            cate.append(model.forward(features=batch["features"])["cate_pred"].cpu().numpy())
            out.append(batch["outcome"].numpy())
            treat.append(batch["treatment"].numpy())
    return qini_coefficient(np.concatenate(cate), np.concatenate(out), np.concatenate(treat))


def test_causalpfn(train_dataset: UpliftDataset, val_dataset: UpliftDataset) -> Dict[str, Any]:
    """
    Test CausalPFN (frozen model, training-free in-context CATE).
    """
    logger.info("Testing CausalPFN...")

    config = {"device": "cpu", "max_context": 1000, "verbose": False}
    model = CausalPFNModel(config)

    # Provide the in-context sample (frozen model, no gradient training)
    train_loader = DataLoader(train_dataset, batch_size=64, collate_fn=collate_fn, shuffle=False)
    Xc, Tc, Yc = [], [], []
    for batch in train_loader:
        Xc.append(batch["features"].numpy())
        Tc.append(batch["treatment"].numpy())
        Yc.append(batch["outcome"].numpy())
    model.fit(np.concatenate(Xc), np.concatenate(Tc), np.concatenate(Yc))

    # Evaluate on validation set
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        collate_fn=collate_fn,
        shuffle=False,
    )

    qini = _eval_qini(model, val_loader)
    logger.info(f"CausalPFN Qini: {qini:.4f}")

    return {"model": "causalpfn", "qini": qini}


def main(args):
    """
    Run smoke test.
    """
    logger.info("=" * 70)
    logger.info("SMOKE TEST: Uplift Modeling Pipeline")
    logger.info("=" * 70)

    # Load datasets
    train_dataset = load_dataset(args.dataset, split="train")
    val_dataset = load_dataset(args.dataset, split="val")

    results = []

    # Test selected models
    if "s_learner" in args.models:
        result = test_s_learner(train_dataset, val_dataset)
        results.append(result)

    if "t_learner" in args.models:
        logger.info("T-Learner test not yet implemented (similar to S-Learner)")

    if "causalpfn" in args.models:
        try:
            result = test_causalpfn(train_dataset, val_dataset)
        except Exception as e:
            logger.warning(f"CausalPFN unavailable ({e}); skipping.")
            result = {"model": "causalpfn", "qini": float("nan"), "note": "unavailable"}
        results.append(result)

    # Summary
    logger.info("=" * 70)
    logger.info("SMOKE TEST RESULTS")
    logger.info("=" * 70)
    for result in results:
        logger.info(f"{result['model']:20s} Qini: {result.get('qini', 'N/A')}")
    logger.info("=" * 70)

    # Validation
    if any(result.get('qini') is None or np.isnan(result.get('qini', float('nan'))) for result in results):
        logger.warning("Some results are missing or NaN. Check implementations.")
        return 1

    logger.info("✓ Smoke test passed!")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smoke test for uplift pipeline")
    parser.add_argument(
        "--dataset",
        default="hillstrom",
        choices=["hillstrom", "criteo"],
        help="Dataset to use",
    )
    parser.add_argument(
        "--models",
        default="s_learner,causalpfn",
        help="Comma-separated list of models to test",
    )

    args = parser.parse_args()
    args.models = args.models.split(",")

    sys.exit(main(args))

"""Loaders for the cleaned uplift datasets (features.parquet + outcomes.parquet).

Set HYPECHECK_DATA_ROOT to the folder holding the dataset subdirs.
If the datasets are not found locally, they are auto-downloaded from Yandex Disk
(the public link in YANDEX_PUBLIC_URL, a zip of all cleaned datasets).
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Literal, Tuple

import numpy as np
import pandas as pd

from .uplift_dataset import UpliftDataset

logger = logging.getLogger(__name__)

DATA_ROOT = Path(os.environ.get("HYPECHECK_DATA_ROOT", Path(__file__).parents[2] / "data"))
CACHE_ROOT = Path(os.environ.get("HYPECHECK_CACHE_ROOT", DATA_ROOT / "cache"))


class ScikitDataset(UpliftDataset):
    """
    A dataset of control group examples only, for use in training a control model.

    This dataset is derived from the original uplift dataset by filtering out all
    treatment group examples. It is useful for training a model that predicts the
    outcome for control group examples, which can then be used to estimate the
    treatment effect by comparing the predicted outcomes for control and treatment
    groups.
    """

    def __init__(
        self,
        limit: Optional[int] = None,
        split: Literal['train', 'test', 'val'] = 'train',
        section: Literal['hillstrom', 'retailhero', 'lenta', 'megafon', 'criteo'] = 'hillstrom',
        instance_transforms: Optional[Dict] = None,
        convert_to_index: bool = True,
    ):
        """
        Args:
            limit (int | None): if not None, limit the total number of elements
                in the dataset to 'limit' elements.
            split (str): one of 'train', 'test', or 'val'. Determines whether
                the dataset is shuffled (train) or not (test/val).
            section (str): one of the known dataset sections (see _DATASETS_HANDLER).
            instance_transforms (dict | None): optional per-key transforms
                applied in preprocess_data.
            convert_to_index (bool): if True, convert the loaded arrays into
                the UpliftDataset index (list of dicts). If False, the dataset
                will hold the raw arrays (X, T, Y) instead of the index.
        """
        from sklift.datasets import fetch_x5, fetch_lenta, fetch_criteo, fetch_hillstrom, fetch_megafon

        _DATASETS_HANDLER: Dict[str, Dict[str, Any]] = {
            "hillstrom": fetch_hillstrom,
            "retailhero": fetch_x5,
            "lenta": fetch_lenta,
            "megafon": fetch_megafon,
            "criteo": fetch_criteo,
        }
        _DATASETS_HANDLER["x5"] = _DATASETS_HANDLER["retailhero"] # alias

        if section not in _DATASETS_HANDLER:
            raise KeyError(f"Unknown dataset '{section}'. Known: {_DATASETS_HANDLER.keys()}")
        
        self._convert_to_index = convert_to_index
        
        X, y, t = _DATASETS_HANDLER[section](return_X_y_t=True, data_home=DATA_ROOT, dest_subdir=section)

        train_fraction = 0.7
        val_fraction = 0.15
        test_fraction = 1.0 - train_fraction - val_fraction
        if split == "train":
            X = X.head(int(len(X) * train_fraction))
            y = y.head(int(len(y) * train_fraction))
            t = t.head(int(len(t) * train_fraction))
        elif split == "val":
            X = X.slice(int(len(X) * train_fraction), int(len(X) * val_fraction))
            y = y.slice(int(len(y) * train_fraction), int(len(y) * val_fraction))
            t = t.slice(int(len(t) * train_fraction), int(len(t) * val_fraction))
        elif split == "test":
            X = X.tail(int(len(X) * test_fraction))
            y = y.tail(int(len(y) * test_fraction))
            t = t.tail(int(len(t) * test_fraction))

        if limit and len(X) > limit:
            X = X.head(limit)
            y = y.head(limit)
            t = t.head(limit)

        X = pd.get_dummies(X, dummy_na=False)
        if section == "megafon":
            t = t == 'treatment'

        X = X.to_numpy()
        T = t.to_numpy()
        Y = y.to_numpy()

        logger.info("Loaded %s: n=%d, features=%d", section, len(X), X.shape[1])

        if convert_to_index:
            index = [
                {
                    "features": x,
                    "treatment": t,
                    "outcome": y
                }
                for x, t, y in zip(
                    X.astype(np.float32),
                    T.astype(np.int8),
                    Y.astype(np.float32)
                )
            ]
        
            super().__init__(
                index=index,
                limit=None,
                shuffle_index=False,
                instance_transforms=instance_transforms,
                feature_dtype=np.float32,
            )
        else:
            self.instance_transforms = instance_transforms
            self._index: dict[str, list] = {
                "features": X.astype(np.float32),
                "treatment": T.astype(np.int8),
                "outcome": Y.astype(np.float32)
            }


    def get_all_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the full dataset as (X, T, Y)."""
        assert not self._convert_to_index, "get_all_data() is only available when convert_to_index=False"
        index = {
            "features": self._index["features"],
            "treatment": self._index["treatment"],
            "outcome": self._index["outcome"]
        }
        if self.instance_transforms:
            for key, transform in self.instance_transforms.items():
                if key in self._index:
                    index[key] = [transform(x) for x in self._index[key]]
        return self._index["features"], self._index["treatment"], self._index["outcome"]

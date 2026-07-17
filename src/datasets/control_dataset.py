"""Loaders for the cleaned uplift datasets (features.parquet + outcomes.parquet).

Set HYPECHECK_DATA_ROOT to the folder holding the dataset subdirs.
If the datasets are not found locally, they are auto-downloaded from Yandex Disk
(the public link in YANDEX_PUBLIC_URL, a zip of all cleaned datasets).
"""

import hashlib
import io
import logging
import os
import pickle
import shutil
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Literal

import numpy as np
import polars as pl
import polars.selectors as cs
import requests
from tqdm.auto import tqdm

from .uplift_dataset import UpliftDataset

logger = logging.getLogger(__name__)

DATA_ROOT = Path(os.environ.get("HYPECHECK_DATA_ROOT", Path(__file__).parents[2] / "data"))
CACHE_ROOT = Path(os.environ.get("HYPECHECK_CACHE_ROOT", DATA_ROOT / "cache"))

# Yandex Disk — public link and zip name for the full cleaned dataset bundle
_YANDEX_API_URL = "https://cloud-api.yandex.net/v1/disk/public/resources/download"
_YANDEX_DATASET_KEY = os.environ.get("YANDEX_DATASET_KEY")
_YANDEX_DISK_URL = f"https://disk.yandex.ru/d/{_YANDEX_DATASET_KEY}"
_ZIP_NAME = "data_A_cleaned.zip"

_DATASETS_CONFIG: Dict[str, Dict[str, Any]] = {
    "hillstrom": {"folder": "Hillstrom", "outcome": "visit"},
    "retailhero": {"folder": "Retailhero-uplift", "outcome": "Y"},
    "lzd": {"folder": "LZD", "outcome": "Y"},
    "orange": {"folder": "Orange Telecom Churn", "outcome": "churn"},
    "criteo": {"folder": "Criteo-ITE-v2.1", "outcome": "visit"},
}
_DATASETS_CONFIG["x5"] = _DATASETS_CONFIG["retailhero"] # alias


class ControlDataset(UpliftDataset):
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
        section: Literal['hillstrom', 'retailhero', 'lzd', 'orange', 'criteo'] = 'hillstrom',
        outcome_column_name: Optional[str] = None,
        instance_transforms: Optional[Dict] = None,
        convert_to_index: bool = True,
    ):
        """
        Args:
            limit (int | None): if not None, limit the total number of elements
                in the dataset to 'limit' elements.
            split (str): one of 'train', 'test', or 'val'. Determines whether
                the dataset is shuffled (train) or not (test/val).
            section (str): one of the known dataset sections (see _DATASETS_CONFIG).
            instance_transforms (dict | None): optional per-key transforms
                applied in preprocess_data.
            convert_to_index (bool): if True, convert the loaded arrays into
                the UpliftDataset index (list of dicts). If False, the dataset
                will hold the raw arrays (X, T, Y) instead of the index.
        """
        if section not in _DATASETS_CONFIG:
            raise KeyError(f"Unknown dataset '{section}'. Known: {_DATASETS_CONFIG.keys()}")
        
        self._convert_to_index = convert_to_index
        
        index = self.load_uplift_dataset(
            section,
            outcome=outcome_column_name,
            convert_to_index=convert_to_index,
            limit=limit,
            split=split,
            use_cache=True
        )

        if convert_to_index:
            super().__init__(
                index=index,
                limit=None,
                shuffle_index=False,
                instance_transforms=instance_transforms,
                feature_dtype=np.float32,
            )
        else:
            self.instance_transforms = instance_transforms
            self._index: dict[str, list] = index


    def _cache_path(self, name: str, **kwargs) -> Path:
        key = "_".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
        h = hashlib.md5(f"{name}_{key}".encode()).hexdigest()[:10]
        CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        return CACHE_ROOT / f"{name}_{h}.pkl"


    def _encode_features(self, df: pl.DataFrame, exclude: set) -> Tuple[np.ndarray, List[str]]:
        frame = df.drop(exclude.intersection(df.columns))

        categorical_selector = cs.string() | cs.by_dtype(pl.Object)
        categorical_cols = frame.select(categorical_selector).columns
        if categorical_cols:
            frame = frame.to_dummies(columns=categorical_cols, separator="_", drop_nulls=False)

        frame = frame.with_columns(
            pl.all()
            .cast(pl.Float32, strict=False)
            .fill_null(pl.all().median().fill_null(0.0))
        )

        return frame.to_numpy().astype(np.float32), frame.columns


    # ──────────────────────────────────────────────────────────────────
    # Yandex Disk download helpers
    # ──────────────────────────────────────────────────────────────────

    def _download_and_extract(self, data_root: Path) -> None:
        """Download *data_A_cleaned.zip* from Yandex Disk and extract into *data_root*.

        The bundle on Yandex Disk is a zip-of-zip: the outer archive contains
        ``Данные uplift/data_A_cleaned.zip``, whose inner paths are prefixed with
        ``data_A_cleaned/``.  This function flattens the nesting so the dataset
        folders sit directly under *data_root*.
        """
        resp = requests.get(_YANDEX_API_URL, params={"public_key": _YANDEX_DISK_URL}, timeout=30)
        resp.raise_for_status()
        url = resp.json()["href"]
        logger.info("Downloading %s from %s …", _ZIP_NAME, _YANDEX_DISK_URL)
        resp = requests.get(url, stream=True, timeout=600)
        resp.raise_for_status()

        total = int(resp.headers.get("content-length", 0))
        chunks: List[bytes] = []
        with tqdm(total=total, unit="B", unit_scale=True, desc="Downloading datasets") as pbar:
            for chunk in resp.iter_content(chunk_size=8192):
                chunks.append(chunk)
                pbar.update(len(chunk))

        data_root.mkdir(parents=True, exist_ok=True)

        # ── 1. Extract outer zip into a temp scratch folder ──
        tmp = data_root / ".yandex_tmp"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(io.BytesIO(b"".join(chunks))) as zf:
            zf.extractall(tmp)

        chunks = []  # free memory

        # ── 2. Find and extract the inner data_A_cleaned.zip ──
        inner_zips = list(tmp.rglob("data_A_cleaned.zip"))
        if not inner_zips:
            raise FileNotFoundError(
                "Expected inner data_A_cleaned.zip inside the Yandex Disk bundle, "
                f"but none was found under {tmp}. Contents: {list(tmp.rglob('*'))[:20]}"
            )

        inner_zip_path = inner_zips[0]
        # Strip the data_A_cleaned/ prefix from every member of the inner zip
        prefix = "data_A_cleaned/"
        with zipfile.ZipFile(inner_zip_path) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                # Remove the top-level prefix
                name = info.filename
                if name.startswith(prefix):
                    name = name[len(prefix):]
                else:
                    # if there's no prefix, keep the relative path as-is
                    pass
                target = data_root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info.filename) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)

        # ── 3. Clean up ──
        shutil.rmtree(tmp, ignore_errors=True)

        logger.info("Extracted %s → %s (datasets: %s)", _ZIP_NAME, data_root, ", ".join(_DATASETS_CONFIG.keys()))


    def _resolve_data_root(self, data_root: Optional[Path]) -> Path:
        """Return *data_root* if given, else *DATA_ROOT*.

        If the target directory is missing the expected dataset folders,
        download and extract the Yandex Disk bundle automatically.
        """
        root = data_root if data_root is not None else DATA_ROOT

        # Quick check — does at least one dataset folder exist?
        if not root.is_dir() or not any((root / info["folder"]).is_dir() for info in _DATASETS_CONFIG.values()):
            logger.info("Dataset folders not found under %s – downloading from Yandex Disk …", root)
            self._download_and_extract(root)

        return root


    # ──────────────────────────────────────────────────────────────────
    # Main loaders
    # ──────────────────────────────────────────────────────────────────


    def load_uplift_arrays(self, name, outcome=None, split=None, limit=None, data_root=None):
        """Load a dataset as (X, T, Y, feature_names)."""
        root = self._resolve_data_root(Path(data_root) if data_root is not None else None)

        folder = _DATASETS_CONFIG[name]["folder"]

        feats = pl.read_parquet(root / folder / "features.parquet")
        outs = pl.read_parquet(root / folder / "outcomes.parquet").select(["epk_id", outcome])

        df = feats.join(outs, on="epk_id", how="inner").drop_nulls(subset=[outcome, "T"])

        df = df.sample(fraction=1.0, seed=42, shuffle=True)

        train_fraction = 0.7
        val_fraction = 0.15
        test_fraction = 1.0 - train_fraction - val_fraction
        if split == "train":
            df = df.head(int(len(df) * train_fraction))
        elif split == "val":
            df = df.slice(int(len(df) * train_fraction), int(len(df) * val_fraction))
        elif split == "test":
            df = df.tail(int(len(df) * test_fraction))

        if limit and len(df) > limit:
            df = df.head(limit)

        # outcome must stay out of X, otherwise CATE collapses to ~0
        X, feature_names = self._encode_features(df, exclude={"epk_id", "T", "treatment_dt", "lag", outcome})
        T = df["T"].to_numpy()
        Y = df[outcome].to_numpy()
        logger.info("Loaded %s: n=%d, features=%d, outcome=%s", name, len(X), X.shape[1], outcome)
        return X, T, Y, feature_names


    def load_uplift_dataset(self, name, outcome=None, data_root=None, convert_to_index=True, split=None, limit=None, use_cache=False):
        """Load a dataset as the UpliftDataset index (list of dicts)."""
        cache = self._cache_path(name, outcome=outcome, split=split, limit=limit, convert_to_index=convert_to_index)
        if use_cache and cache.exists():
            with open(cache, "rb") as f:
                return pickle.load(f)

        X, T, Y, _ = self.load_uplift_arrays(
            name,
            outcome=outcome,
            split=split,
            limit=limit,
            data_root=data_root,
        )
        if not convert_to_index:
            return {
                "features": X.astype(np.float32),
                "treatment": T.astype(np.int8),
                "outcome": Y.astype(np.float32)
            }
        else:
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

        if use_cache:
            with open(cache, "wb") as f:
                pickle.dump(index, f)

        return index

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
        return index["features"], index["treatment"], index["outcome"]

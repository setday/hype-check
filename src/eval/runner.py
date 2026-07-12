"""Offline W1 harness: split -> fit learners -> ranking metrics + Qini plots."""

import logging
import time
from pathlib import Path
from typing import List, Optional
from tqdm.auto import tqdm

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torchmetrics.wrappers import BootStrapper

from hydra.utils import instantiate

from src.datasets.uplift_data_utils import DATASETS, load_uplift_arrays
from src.metrics import utils

logger = logging.getLogger(__name__)

def run_dataset(ds_name, model_factories, metrics, *, limit=None, eval_limit=None,
                test_size=0.3, seed=42, outpath):
    t0 = time.perf_counter()
    X, T, Y, _ = load_uplift_arrays(ds_name, limit=limit, seed=seed)
    print(f"\n=== {ds_name} === n={len(X)} d={X.shape[1]} treat_share={T.mean():.3f} "
          f"ATE_naive={Y[T==1].mean()-Y[T==0].mean():+.4f}")

    X_tr, X_te, T_tr, T_te, Y_tr, Y_te = train_test_split(
        X, T, Y, test_size=test_size, random_state=seed, stratify=T)
    if eval_limit is not None and len(X_te) > eval_limit:
        sel = np.random.default_rng(seed).choice(len(X_te), size=eval_limit, replace=False)
        X_te, T_te, Y_te = X_te[sel], T_te[sel], Y_te[sel]
    print(f"  split: train={len(X_tr)} test={len(X_te)} (load {time.perf_counter()-t0:.1f}s)")

    rows, curves = [], []
    for factory in tqdm(model_factories.values(), "models", leave=False):
        model = factory()
        tf = time.perf_counter()
        model.fit(X_tr, T_tr, Y_tr)
        cate = model.predict_cate(X_te)
        fit_s = time.perf_counter() - tf

        metrics_args = {
            "cate_pred": torch.from_numpy(cate),
            "outcome": torch.from_numpy(Y_te),
            "treatment": torch.from_numpy(T_te),
        }

        metrics_values = {}
        for name, metric in metrics.items():
            metric.update(**metrics_args)
            result = metric.compute()
            metrics_values[name] = result["mean"].item()
            metrics_values[f"{name}_std"] = result["std"].item()
            metrics_values[f"{name}_q5"] = result["quantile"][0].item()
            metrics_values[f"{name}_q95"] = result["quantile"][1].item()

        disp = getattr(model, "display_name", model.__class__.__name__)
        curves.append((disp, utils.compute_qini_curve(cate, Y_te, T_te)))
        metrics_values.update({"dataset": ds_name, "model": disp, "fit_time_s": round(fit_s, 2), "n_test": len(X_te)})
        rows.append(metrics_values)
    
    x = np.linspace(0.0, 1.0, len(curves[0][1]))

    all_curves = np.vstack([x]+[e[1] for e in curves]).T
    df = pd.DataFrame(all_curves, columns=["x"]+[e[0] for e in curves])
    df.to_csv(outpath / f"curves_{ds_name}.csv", index=False)
    
    return rows


def run_experiments(datasets: List[str], models_config: dict, metrics_config: dict, *, outdir="results",
                    limit: Optional[int] = None, eval_limit: Optional[int] = None,
                    n_boot=200, test_size=0.3, seed=42) -> pd.DataFrame:
    outpath = Path(outdir)
    outpath.mkdir(parents=True, exist_ok=True)
    ds_names = list(DATASETS) if datasets == ["all"] else datasets

    make_factory = lambda cls: (lambda *args, **kwargs: instantiate(cls, *args, **kwargs))

    model_factories = {
        key: make_factory(value)
        for key, value in models_config.items()
    } # _build_models(models, seed, with_causalpfn, pfn_context)
    metrics = {
        value.name: BootStrapper(
            instantiate(value),
            num_bootstraps=n_boot, 
            mean=True,
            std=True,
            quantile=torch.tensor([0.05, 0.95])
        )
        for value in metrics_config
    }

    all_rows = []
    for name in tqdm(ds_names, "datasets"):
        rows = run_dataset(name, model_factories, metrics, limit=limit, eval_limit=eval_limit,
                                   test_size=test_size, seed=seed, outpath=outpath)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df.to_csv(outpath / "metrics.csv", index=False)
    try:
        (outpath / "metrics.md").write_text(df.round(5).to_markdown(index=False))
    except Exception:
        pass
    print("\n=== SUMMARY ===")
    print(df.round(5).to_string(index=False))
    print(f"\nSaved: {outpath/'metrics.csv'}")
    return df

"""Offline W1 harness: split -> fit learners -> ranking metrics + Qini plots."""

import logging
import time
import sys
from pathlib import Path

from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import torch
from torchmetrics.wrappers import BootStrapper

import hydra
from hydra.utils import instantiate

from src.metrics import utils
from src.utils.init_utils import ROOT_PATH



logger = logging.getLogger(__name__)

def run_dataset(dataset, model_factories, metrics, *, outpath):
    ds_name, dataset = dataset
    
    X_tr, T_tr, Y_tr = dataset['train'].get_all_data()
    X_te, T_te, Y_te = dataset['test'].get_all_data()

    print(f"\n=== {ds_name} === n={len(X_tr)} d={X_tr.shape[1]} treat_share={T_tr.mean():.3f} "
          f"ATE_naive={Y_tr[T_tr==1].mean()-Y_tr[T_tr==0].mean():+.4f}")

    print(f"  split: train={len(X_tr)} test={len(X_te)}")

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


@hydra.main(version_base=None, config_path="config", config_name="baselines_eval")
def main(config):
    outpath = Path(ROOT_PATH / "results")
    outpath.mkdir(parents=True, exist_ok=True)

    make_factory = lambda cls: (lambda *args, **kwargs: instantiate(cls, *args, **kwargs))

    model_factories = {
        key: make_factory(value)
        for key, value in config.model.items()
    }
    metrics = {
        value.name: BootStrapper(
            instantiate(value),
            num_bootstraps=200, 
            mean=True,
            std=True,
            quantile=torch.tensor([0.05, 0.95])
        )
        for value in config.metrics.val
    }

    all_rows = []
    for name, dataset in tqdm(config.datasets.items(), "datasets"):
        dataset = {
            split: instantiate(data, convert_to_index=False)
            for split, data in dataset.items()
        }
        rows = run_dataset((name, dataset), model_factories, metrics, outpath=outpath)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df.to_csv(outpath / "metrics.csv", index=False)
    print("\n=== SUMMARY ===")
    print(df.round(5).to_string(index=False))

if __name__ == '__main__':
    main()

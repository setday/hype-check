"""Offline eval harness: fit models -> ranking metrics + Qini curves (+ optional W&B)."""

import logging
import os
import time
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torchmetrics.wrappers import BootStrapper
from tqdm.auto import tqdm

from src.metrics import utils
from src.utils.init_utils import ROOT_PATH

logger = logging.getLogger(__name__)


def _make_metrics(metric_cfgs, num_bootstraps: int, quantile):
    """Fresh metric objects — never reuse across models without reset."""
    return {
        cfg.name: BootStrapper(
            instantiate(cfg),
            num_bootstraps=num_bootstraps,
            mean=True,
            std=True,
            quantile=torch.tensor(quantile),
        )
        for cfg in metric_cfgs
    }


def _evaluate(cate, y_te, t_te, metric_cfgs, num_bootstraps, quantile):
    metrics = _make_metrics(metric_cfgs, num_bootstraps, quantile)
    args = {
        "cate_pred": torch.from_numpy(cate),
        "outcome": torch.from_numpy(y_te),
        "treatment": torch.from_numpy(t_te),
    }
    out = {}
    for name, metric in metrics.items():
        metric.update(**args)
        result = metric.compute()
        out[name] = result["mean"].item()
        out[f"{name}_std"] = result["std"].item()
        out[f"{name}_q5"] = result["quantile"][0].item()
        out[f"{name}_q95"] = result["quantile"][1].item()
        metric.reset()
    return out


def _maybe_init_wandb(cfg: DictConfig):
    wb = cfg.get("wandb", {})
    if not wb.get("enabled", False):
        return None
    import wandb

    run_name = wb.get("run_name") or os.environ.get("WANDB_RUN_NAME")
    return wandb.init(
        project=wb.get("project", "hype-check"),
        entity=wb.get("entity"),
        name=run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        reinit=True,
    )


def run_dataset(dataset, model_factories, metric_cfgs, eval_cfg, *, outpath, wandb_run=None):
    ds_name, dataset = dataset
    X_tr, T_tr, Y_tr = dataset["train"].get_all_data()
    X_te, T_te, Y_te = dataset["test"].get_all_data()

    print(
        f"\n=== {ds_name} === n={len(X_tr)} d={X_tr.shape[1]} treat_share={T_tr.mean():.3f} "
        f"ATE_naive={Y_tr[T_tr == 1].mean() - Y_tr[T_tr == 0].mean():+.4f}"
    )
    print(f"  split: train={len(X_tr)} test={len(X_te)}")

    rows, curves = [], []
    num_boot = int(eval_cfg.get("num_bootstraps", 200))
    quantile = list(eval_cfg.get("quantile", [0.05, 0.95]))

    for factory in tqdm(model_factories.values(), "models", leave=False):
        model = factory()
        disp = getattr(model, "display_name", model.__class__.__name__)

        t0 = time.perf_counter()
        model.fit(X_tr, T_tr, Y_tr)
        train_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        cate = model.predict_cate(X_te)
        infer_s = time.perf_counter() - t1

        metrics_values = _evaluate(cate, Y_te, T_te, metric_cfgs, num_boot, quantile)
        curves.append((disp, utils.compute_qini_curve(cate, Y_te, T_te)))
        metrics_values.update({
            "dataset": ds_name,
            "model": disp,
            "train_time_s": round(getattr(model, "train_time_s", train_s), 2),
            "inference_time_s": round(getattr(model, "inference_time_s", infer_s), 2),
            "fit_time_s": round(train_s + infer_s, 2),
            "n_test": len(X_te),
        })
        rows.append(metrics_values)

        if wandb_run is not None:
            import wandb
            wandb.log({f"{ds_name}/{disp}/{k}": v for k, v in metrics_values.items() if isinstance(v, (int, float))})

    x = np.linspace(0.0, 1.0, len(curves[0][1]))
    all_curves = np.vstack([x] + [e[1] for e in curves]).T
    pd.DataFrame(all_curves, columns=["x"] + [e[0] for e in curves]).to_csv(outpath / f"curves_{ds_name}.csv", index=False)
    return rows


@hydra.main(version_base=None, config_path="config", config_name="baselines_eval")
def main(config: DictConfig):
    outpath = Path(ROOT_PATH / "results")
    outpath.mkdir(parents=True, exist_ok=True)

    make_factory = lambda cls: (lambda *args, **kwargs: instantiate(cls, *args, **kwargs))
    model_factories = {key: make_factory(value) for key, value in config.model.items()}

    metric_cfgs = list(config.metrics.val)
    eval_cfg = config.get("eval", {"num_bootstraps": 200, "quantile": [0.05, 0.95]})
    wandb_run = _maybe_init_wandb(config)

    all_rows = []
    for name, dataset in tqdm(config.datasets.items(), "datasets"):
        dataset = {split: instantiate(data, convert_to_index=False) for split, data in dataset.items()}
        rows = run_dataset(
            (name, dataset), model_factories, metric_cfgs, eval_cfg,
            outpath=outpath, wandb_run=wandb_run,
        )
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df.to_csv(outpath / "metrics.csv", index=False)
    print("\n=== SUMMARY ===")
    print(df.round(5).to_string(index=False))

    if wandb_run is not None:
        import wandb
        wandb.log({"metrics_table": wandb.Table(dataframe=df)})
        wandb.finish()


if __name__ == "__main__":
    main()

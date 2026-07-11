"""Offline W1 harness: split -> fit learners -> ranking metrics + Qini plots."""

import logging
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.datasets.uplift_data_utils import DATASETS, load_uplift_arrays
from src.metrics import ranking
from src.models import BASELINES
from src.models.causalpfn_model import CausalPFNModel, causalpfn_available

logger = logging.getLogger(__name__)

METRIC_COLUMNS = ["dataset", "model", "qini", "qini_lo", "qini_hi", "auuc",
                  "uplift@10", "uplift@30", "fit_time_s", "n_test"]


def evaluate_predictions(cate_pred, y_test, t_test, n_boot=200, seed=0) -> dict:
    q = ranking.bootstrap_metric(ranking.qini_coefficient, cate_pred, y_test, t_test, n_boot=n_boot, seed=seed)
    out = {"qini": q["mean"], "qini_lo": q["lo"], "qini_hi": q["hi"],
           "auuc": ranking.bootstrap_metric(ranking.auuc, cate_pred, y_test, t_test, n_boot=n_boot, seed=seed)["mean"]}
    for k in (0.1, 0.3):
        out[f"uplift@{int(k*100)}"] = ranking.bootstrap_metric(
            ranking.uplift_at_k, cate_pred, y_test, t_test, n_boot=n_boot, seed=seed, k=k)["mean"]
    return out


def _build_models(model_keys, seed, with_causalpfn, pfn_context):
    models = {}
    for key in model_keys:
        if key in BASELINES:
            models[key] = lambda cls=BASELINES[key]: cls({"seed": seed})
        else:
            logger.warning("Unknown baseline key '%s' (skipped).", key)
    if with_causalpfn:
        if causalpfn_available():
            models["causalpfn"] = lambda: CausalPFNModel({"max_context": pfn_context})
        else:
            logger.warning("--with-causalpfn set but 'causalpfn' not installed; skipping.")
    return models


def run_dataset(name, model_factories, *, limit=None, eval_limit=None,
                test_size=0.3, n_boot=200, seed=42):
    t0 = time.time()
    X, T, Y, _ = load_uplift_arrays(name, limit=limit, seed=seed)
    print(f"\n=== {name} === n={len(X)} d={X.shape[1]} treat_share={T.mean():.3f} "
          f"ATE_naive={Y[T==1].mean()-Y[T==0].mean():+.4f}")

    X_tr, X_te, T_tr, T_te, Y_tr, Y_te = train_test_split(
        X, T, Y, test_size=test_size, random_state=seed, stratify=T)
    if eval_limit is not None and len(X_te) > eval_limit:
        sel = np.random.default_rng(seed).choice(len(X_te), size=eval_limit, replace=False)
        X_te, T_te, Y_te = X_te[sel], T_te[sel], Y_te[sel]
    print(f"  split: train={len(X_tr)} test={len(X_te)} (load {time.time()-t0:.1f}s)")

    rows, curves = [], {}
    for factory in model_factories.values():
        model = factory()
        disp = getattr(model, "display_name", model.__class__.__name__)
        tf = time.time()
        model.fit(X_tr, T_tr, Y_tr)
        cate = model.predict_cate(X_te)
        fit_s = time.time() - tf
        m = evaluate_predictions(cate, Y_te, T_te, n_boot=n_boot, seed=seed)
        m.update({"dataset": name, "model": disp, "fit_time_s": round(fit_s, 2), "n_test": len(X_te)})
        rows.append(m)
        curves[disp] = ranking.qini_curve(cate, Y_te, T_te)
        print(f"  {disp:<12} qini={m['qini']:+.5f} [{m['qini_lo']:+.5f},{m['qini_hi']:+.5f}] "
              f"uplift@30={m['uplift@30']:+.5f} ({fit_s:.1f}s)")
    return rows, curves


def plot_qini(name, curves, outdir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 5))
    for model, (x, q) in curves.items():
        plt.plot(x, q, linewidth=1.8, label=model)
    last = next(iter(curves.values()))[1][-1]
    plt.plot([0, 1], [0, last], "k--", alpha=0.6, label="random")
    plt.xlabel("Fraction targeted")
    plt.ylabel("Cumulative incremental responders (Qini)")
    plt.title(f"Qini — {name}")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    out = outdir / f"qini_{name.replace('/', '_')}.png"
    plt.savefig(out, dpi=130); plt.close()
    print(f"  saved {out}")


def run_experiments(datasets: List[str], models: List[str], *, outdir="results",
                    limit: Optional[int] = None, eval_limit: Optional[int] = None,
                    with_causalpfn=False, pfn_context=5000,
                    test_size=0.3, n_boot=200, seed=42) -> pd.DataFrame:
    outpath = Path(outdir)
    outpath.mkdir(parents=True, exist_ok=True)
    ds_names = list(DATASETS) if datasets == ["all"] else datasets
    model_factories = _build_models(models, seed, with_causalpfn, pfn_context)

    all_rows = []
    for name in ds_names:
        rows, curves = run_dataset(name, model_factories, limit=limit, eval_limit=eval_limit,
                                   test_size=test_size, n_boot=n_boot, seed=seed)
        all_rows.extend(rows)
        plot_qini(name, curves, outpath)

    df = pd.DataFrame(all_rows)
    df = df[[c for c in METRIC_COLUMNS if c in df.columns]]
    df.to_csv(outpath / "metrics.csv", index=False)
    try:
        (outpath / "metrics.md").write_text(df.round(5).to_markdown(index=False))
    except Exception:
        pass
    print("\n=== SUMMARY ===")
    print(df.round(5).to_string(index=False))
    print(f"\nSaved: {outpath/'metrics.csv'}")
    return df

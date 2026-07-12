"""W1 baselines vs untuned CausalPFN. Set HYPECHECK_DATA_ROOT first.

    python scripts/run_baselines.py --datasets all --limit 300000
    python scripts/run_baselines.py --datasets all --with-causalpfn --eval-limit 6000 --outdir results_with_pfn
"""

import argparse
import sys
from pathlib import Path

import hydra

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.eval.runner import run_experiments


@hydra.main(version_base=None, config_path="../config", config_name="baselines_eval")
def main(config):
    p = argparse.ArgumentParser(description="W1 baselines vs CausalPFN")
    # p.add_argument("--datasets", default="all",
    #                help="comma-separated dataset keys or 'all' (hillstrom, retailhero, lzd, orange, criteo)")
    # p.add_argument("--with-causalpfn", action="store_true", help="also run CausalPFN if installed")
    # p.add_argument("--pfn-context", type=int, default=5000, help="CausalPFN in-context sample cap")
    # p.add_argument("--limit", type=int, default=300000, help="subsample rows before split (e.g. Criteo)")
    # p.add_argument("--eval-limit", type=int, default=None, help="cap the test set (fast fair comparison)")
    # p.add_argument("--test-size", type=float, default=0.3)
    # p.add_argument("--n-boot", type=int, default=200)
    # p.add_argument("--seed", type=int, default=42)
    # p.add_argument("--outdir", default=str(REPO_ROOT / "results"))
    # args = p.parse_args()

    run_experiments(
        datasets=["all"],
        models_config=config.model,
        metrics_config=config.metrics.val,
        outdir=str(REPO_ROOT / "results"),
        limit=300000,
        eval_limit=None,
        test_size=0.3,
        seed=42,
    )


if __name__ == "__main__":
    main()

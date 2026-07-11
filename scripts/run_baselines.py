"""W1 baselines vs untuned CausalPFN. Set HYPECHECK_DATA_ROOT first.

    python scripts/run_baselines.py --datasets all --limit 300000
    python scripts/run_baselines.py --datasets all --with-causalpfn --eval-limit 6000 --outdir results_with_pfn
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.eval.runner import run_experiments


def main():
    p = argparse.ArgumentParser(description="W1 baselines vs CausalPFN")
    p.add_argument("--datasets", default="hillstrom",
                   help="comma-separated dataset keys or 'all' (hillstrom, retailhero, lzd, orange, criteo)")
    p.add_argument("--models", default="s_learner,t_learner,x_learner,dr_learner",
                   help="comma-separated baseline keys")
    p.add_argument("--with-causalpfn", action="store_true", help="also run CausalPFN if installed")
    p.add_argument("--pfn-context", type=int, default=5000, help="CausalPFN in-context sample cap")
    p.add_argument("--limit", type=int, default=None, help="subsample rows before split (e.g. Criteo)")
    p.add_argument("--eval-limit", type=int, default=None, help="cap the test set (fast fair comparison)")
    p.add_argument("--test-size", type=float, default=0.3)
    p.add_argument("--n-boot", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", default=str(REPO_ROOT / "results"))
    args = p.parse_args()

    run_experiments(
        datasets=args.datasets.split(","),
        models=args.models.split(","),
        outdir=args.outdir,
        limit=args.limit,
        eval_limit=args.eval_limit,
        with_causalpfn=args.with_causalpfn,
        pfn_context=args.pfn_context,
        test_size=args.test_size,
        n_boot=args.n_boot,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

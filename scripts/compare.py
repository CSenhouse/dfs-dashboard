"""Paired comparison of two named models over identical weeks.

    python scripts/compare.py two_stage two_stage_calibrated

Separate evaluation runs are not comparable in this project -- the lineup
metric has a ~25 point weekly standard deviation and has given a misleading
answer three times. Anything worth acting on has to win pairwise.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from scipy import stats

from dfs import backtest, baseline, model
from dfs.config import CURATED

REGISTRY = {
    "baseline": baseline.expected,
    "salary": baseline.expected_salary,
    "two_stage": model.two_stage,
    "two_stage_floor": model.two_stage_floor,
    "two_stage_ceiling": model.two_stage_ceiling,
    "two_stage_calibrated": model.two_stage_calibrated,
    "single_stage": model.single_stage,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs=2, choices=sorted(REGISTRY))
    ap.add_argument("--min-train-weeks", type=int, default=8)
    ap.add_argument("--lineups", type=int, default=1)
    args = ap.parse_args()

    f = pd.read_parquet(CURATED / "features.parquet")
    print(f"dataset: {len(f):,} rows, "
          f"{f.groupby(['season', 'week']).ngroups} weeks, "
          f"seasons {sorted(f['season'].unique())}")

    sp, lu, cal = {}, {}, {}
    for name in args.models:
        pred = backtest.walk_forward(
            f, REGISTRY[name], min_train_weeks=args.min_train_weeks
        )
        acc = backtest.accuracy(pred)
        L = backtest.lineup_results(pred, n_lineups=args.lineups)
        print(backtest.summarize(acc, L, name), flush=True)
        sp[name] = acc.set_index(["season", "week"])["spearman"]
        lu[name] = L[L["lineup"] == 0].set_index(["season", "week"])["actual"]
        cal[name] = backtest.calibration(pred)

    a, b = args.models
    S, Lf = pd.DataFrame(sp), pd.DataFrame(lu)
    ds = (S[a] - S[b]).dropna()
    dl = (Lf[a] - Lf[b]).dropna()

    print()
    print(f"=== paired: {a} minus {b} ===")
    print(f"spearman {ds.mean():+.5f}  {a} better in {int((ds > 0).sum())}/{len(ds)}"
          f"  p={stats.ttest_rel(S[a], S[b]).pvalue:.4f}")
    print(f"lineup   {dl.mean():+.2f}  {a} better in {int((dl > 0).sum())}/{len(dl)}"
          f"  p={stats.ttest_rel(Lf[a], Lf[b]).pvalue:.4f}"
          f"  (se {dl.std() / np.sqrt(len(dl)):.2f})")

    print()
    print("=== top-decile calibration gap ===")
    for name in args.models:
        c = cal[name]
        if c is None or c.empty:
            continue
        top = c.iloc[-1]
        print(f"  {name:<22} predicted {top['predicted']:6.2f}  "
              f"actual {top['actual']:6.2f}  gap {top['gap']:+.2f} "
              f"({top['gap'] / top['predicted']:+.1%})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

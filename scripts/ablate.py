"""Paired feature-group ablation.

Comparing two separate evaluation runs is not a valid test in this project:
the lineup metric has a ~25 point weekly standard deviation and its sign has
already flipped between runs. A feature group is only worth keeping if it wins
on the SAME weeks, tested pairwise.

    python scripts/ablate.py --group pbp
"""
from __future__ import annotations

import argparse
import functools
import sys

import numpy as np
import pandas as pd
from scipy import stats

from dfs import backtest, model
from dfs.config import CURATED

# Substrings identifying each ablatable feature group.
GROUPS = {
    "pbp": ("rz_", "gl_", "pbp_", "team_plays", "team_pass_rate", "team_rz"),
    "prior": ("prior_",),
    "matchup": ("def_allowed",),
    "vegas": ("implied_total", "spread", "game_total"),
    "position": ("pos_",),
    "snaps": ("offense_pct",),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True, choices=sorted(GROUPS))
    ap.add_argument("--min-train-weeks", type=int, default=8)
    args = ap.parse_args()

    f = pd.read_parquet(CURATED / "features.parquet")
    markers = GROUPS[args.group]
    print(f"dataset: {len(f):,} rows, "
          f"{f.groupby(['season', 'week']).ngroups} weeks, "
          f"seasons {sorted(f['season'].unique())}")
    print(f"ablating group '{args.group}' -> markers {markers}")

    variants = {
        "with": model.two_stage,
        "without": functools.partial(model.two_stage, exclude=markers),
    }
    sp, lu = {}, {}
    for name, fn in variants.items():
        pred = backtest.walk_forward(f, fn, min_train_weeks=args.min_train_weeks)
        acc = backtest.accuracy(pred)
        L = backtest.lineup_results(pred, n_lineups=1)
        print(backtest.summarize(acc, L, f"{args.group} {name}"), flush=True)
        sp[name] = acc.set_index(["season", "week"])["spearman"]
        lu[name] = L.set_index(["season", "week"])["actual"]

    S, Lf = pd.DataFrame(sp), pd.DataFrame(lu)
    ds = (S["with"] - S["without"]).dropna()
    dl = (Lf["with"] - Lf["without"]).dropna()

    print()
    print(f"=== paired effect of keeping '{args.group}' ===")
    print(f"spearman {ds.mean():+.5f}  better in {int((ds > 0).sum())}/{len(ds)} "
          f"weeks  p={stats.ttest_rel(S['with'], S['without']).pvalue:.4f}")
    print(f"lineup   {dl.mean():+.2f}  better in {int((dl > 0).sum())}/{len(dl)} "
          f"weeks  p={stats.ttest_rel(Lf['with'], Lf['without']).pvalue:.4f}  "
          f"(se {dl.std() / np.sqrt(len(dl)):.2f})")

    verdict = "KEEP" if ds.mean() > 0 and (ds > 0).mean() > 0.6 else "QUESTIONABLE"
    print(f"\nverdict on ranking: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

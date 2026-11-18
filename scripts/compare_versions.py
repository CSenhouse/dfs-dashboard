"""Paired comparison of two VERSIONS of the feature table.

ablate.py answers "is this group worth having?" by dropping it. This answers
"is this version of the features better than that one?" -- the same model
walked forward over two parquet files that differ in how a feature is
computed, scored on identical weeks (rule 2) and selected on ranking (rule
3).

The two tables must hold the same rows. Beyond the whole-slate numbers it
reports accuracy on just the rows whose features differ, because a change
that touches a few dozen players a season cannot move a slate-wide
spearman far, and whether it is RIGHT about those players is the point.

    PYTHONPATH=src .venv/Scripts/python.exe scripts/compare_versions.py \
        features_v1.parquet features.parquet
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from scipy import stats

from dfs import backtest, model
from dfs.config import CURATED

KEY = ["season", "week", "player_dk_id"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a", help="baseline parquet, in data/curated/")
    ap.add_argument("b", help="candidate parquet, in data/curated/")
    ap.add_argument("--min-train-weeks", type=int, default=8)
    ap.add_argument("--cols", default="vac_",
                    help="prefix of the columns whose differences mark the "
                         "affected rows")
    args = ap.parse_args()

    fa = pd.read_parquet(CURATED / args.a)
    fb = pd.read_parquet(CURATED / args.b)
    if len(fa) != len(fb):
        print(f"row counts differ: {len(fa)} vs {len(fb)}")
        return 1
    cols = [c for c in fb.columns if c.startswith(args.cols)]
    m = fa[KEY + cols].merge(fb[KEY + cols], on=KEY, suffixes=("_a", "_b"))
    diff = np.zeros(len(m), dtype=bool)
    for c in cols:
        x, y = m[f"{c}_a"].fillna(-1), m[f"{c}_b"].fillna(-1)
        diff |= ~np.isclose(x, y)
    changed = m.loc[diff, KEY]
    print(f"{len(fb):,} rows; {len(changed):,} differ in {cols}")

    preds, sp, lu = {}, {}, {}
    for name, f in (("a", fa), ("b", fb)):
        p = backtest.walk_forward(f, model.two_stage,
                                  min_train_weeks=args.min_train_weeks)
        acc = backtest.accuracy(p)
        L = backtest.lineup_results(p, n_lineups=1)
        print(backtest.summarize(acc, L, f"{name}: {getattr(args, name)}"),
              flush=True)
        preds[name] = p
        sp[name] = acc.set_index(["season", "week"])["spearman"]
        lu[name] = L.set_index(["season", "week"])["actual"]

    S, Lf = pd.DataFrame(sp), pd.DataFrame(lu)
    ds, dl = (S["b"] - S["a"]).dropna(), (Lf["b"] - Lf["a"]).dropna()
    print(f"\nspearman {ds.mean():+.5f}  b better in {int((ds > 0).sum())}"
          f"/{len(ds)} weeks  p={stats.ttest_rel(S['b'], S['a']).pvalue:.4f}")
    print(f"lineup   {dl.mean():+.2f}  b better in {int((dl > 0).sum())}"
          f"/{len(dl)} weeks  p={stats.ttest_rel(Lf['b'], Lf['a']).pvalue:.4f}")

    if len(changed):
        pa = preds["a"].merge(changed, on=KEY)
        pb = preds["b"].merge(changed, on=KEY)
        j = pa[KEY + ["projection", "dk_actual"]].merge(
            pb[KEY + ["projection"]], on=KEY, suffixes=("_a", "_b"))
        j = j.dropna(subset=["dk_actual"])
        ea = (j["projection_a"] - j["dk_actual"]).abs()
        eb = (j["projection_b"] - j["dk_actual"]).abs()
        print(f"\nthe {len(j)} predicted rows whose features differ:")
        print(f"  mean actual {j['dk_actual'].mean():.2f}, "
              f"projection a {j['projection_a'].mean():.2f}, "
              f"b {j['projection_b'].mean():.2f}")
        print(f"  abs error a {ea.mean():.2f}, b {eb.mean():.2f}, "
              f"b closer on {int((eb < ea).sum())} of {len(j)}  "
              f"p={stats.wilcoxon(ea, eb).pvalue:.4f}" if len(j) > 5 else "")
    return 0


if __name__ == "__main__":
    sys.exit(main())

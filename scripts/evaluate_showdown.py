"""Build Showdown features and evaluate a Showdown-specific model.

Showdown is its own pricing regime. The same player is priced against roughly
45 others instead of 800, so salary rank has to be computed inside the slate,
and the model has to be trained on Showdown rows rather than borrowed from
Classic. It is also the only format that prices kickers.

Lineups are evaluated per slate with the real rules: 1 Captain at 1.5x points
and 1.5x salary, 5 FLEX, both teams represented.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy import stats

from dfs import backtest, baseline, features, model, optimizer
from dfs.config import CURATED

MIN_TRAIN_WEEKS = 8


def showdown_lineups(pred: pd.DataFrame, objective: str = "projection"
                     ) -> pd.DataFrame:
    """Best legal Showdown lineup per slate, scored on what happened."""
    rows = []
    for (season, week, dgid), g in pred.groupby(
        ["season", "week", "draft_group_id"]
    ):
        pool = g.dropna(subset=[objective])
        if len(pool) < 12:
            continue
        built = optimizer.optimize_showdown(pool, objective=objective,
                                            n_lineups=1)
        ceiling = optimizer.optimize_showdown(
            pool.assign(_a=pool["dk_scored"]), objective="_a", n_lineups=1
        )
        if not built or not ceiling:
            continue
        lu = built[0]
        actual = float(
            (lu["dk_scored"] * np.where(lu["slot"] == "CPT", 1.5, 1.0)).sum()
        )
        best = ceiling[0]
        best_actual = float(
            (best["dk_scored"] * np.where(best["slot"] == "CPT", 1.5, 1.0)).sum()
        )
        rows.append({"season": season, "week": week, "draft_group_id": dgid,
                     "actual": actual, "ceiling": best_actual})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["pct_of_ceiling"] = out["actual"] / out["ceiling"]
    return out


def main() -> int:
    raw = pd.read_parquet(CURATED / "showdown_player_slate.parquet")
    print(f"{len(raw):,} showdown player-slate rows, "
          f"{raw['draft_group_id'].nunique()} slates", flush=True)

    print("building features (salary ranked inside each slate) ...", flush=True)
    f = features.build(raw, salary_within_slate=True)
    f.to_parquet(CURATED / "showdown_features.parquet", index=False)
    print(f"features: {len(features.feature_columns(f))}", flush=True)
    print("kicker rows:", int((f["position"] == "K").sum()))
    print()

    sp, lu = {}, {}
    for name, fn in (("baseline", baseline.expected),
                     ("showdown model", model.two_stage)):
        pred = backtest.walk_forward(f, fn, min_train_weeks=MIN_TRAIN_WEEKS)
        acc = backtest.accuracy(pred)
        L = showdown_lineups(pred)
        print(f"=== {name} ===")
        print(f"  weeks={len(acc)}  slates={len(L)}  "
              f"spearman={acc['spearman'].mean():.4f}  "
              f"rmse={acc['rmse'].mean():.3f}")
        print(f"  lineup: mean={L['actual'].mean():.2f}  "
              f"ceiling={L['ceiling'].mean():.2f}  "
              f"pct of ceiling={L['pct_of_ceiling'].mean():.1%}", flush=True)
        sp[name] = acc.set_index(["season", "week"])["spearman"]
        lu[name] = L.set_index(["season", "week", "draft_group_id"])["actual"]

        if name == "showdown model":
            k = pred[pred["position"] == "K"].dropna(
                subset=["projection", "dk_scored"])
            if not k.empty:
                print(f"  kickers: n={len(k)}  "
                      f"mean projected={k['projection'].mean():.2f}  "
                      f"mean actual={k['dk_scored'].mean():.2f}  "
                      f"spearman={stats.spearmanr(k['projection'], k['dk_scored']).statistic:.3f}")
            print()

    S = pd.DataFrame(sp)
    d = (S["showdown model"] - S["baseline"]).dropna()
    print("=== paired, showdown model minus baseline ===")
    print(f"spearman {d.mean():+.4f}  better in {int((d > 0).sum())}/{len(d)} "
          f"weeks  p={stats.ttest_rel(S['showdown model'], S['baseline']).pvalue:.2e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

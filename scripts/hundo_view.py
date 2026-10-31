"""Which projection view and construction suits a Hundo?

A Hundo is a 100-entry, single-entry contest paying the top 22 on a
moderately top-heavy table (90% of fees returned). It sits between a
double-up, which pays for clearing a line, and a large tournament, which
pays for the extreme tail -- so neither the floor view nor the ceiling view
is obviously right.

For every walk-forward week (sim_predictions.parquet: each week projected
only from earlier ones), candidate lineups are built from five views of the
same distribution -- q30 (floor), q50, mean, q70, q85 (ceiling) -- each
with no stack, a QB stack, and a stack plus bring-back. Each is dropped into
a 100-entry contest against 99 opponents who also optimize, on the mean
projection with 30% noise: an optimizing field, because docs/MODELING.md 8
found a sampling field far too soft to compare against. Ranks are paid from
the real $3 Hundo table, per $1 staked.

Every candidate faces the same opponents in the same week, so the comparison
is paired (rule 2). The field is modelled, not counted; read the RANKING of
views, not the absolute return.

    PYTHONPATH=src .venv/Scripts/python.exe scripts/hundo_view.py
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

from dfs import field, model, optimizer
from dfs.config import CURATED

ENTRIES = 100
DISAGREEMENT = 0.30
# The $3 Hundo table, per $1 staked: rank range -> multiple of the entry.
PAYOUT = [(1, 1, 14.0), (2, 2, 10.0), (3, 4, 8.0), (5, 6, 6.0), (7, 8, 4.0),
          (9, 10, 3.0), (11, 14, 2.5), (15, 18, 2.0), (19, 22, 1.5)]
VIEWS = {"q30 floor": "q30", "q50": "q50", "mean": "points_mean",
         "q70": "q70", "q85 ceiling": "q85"}
BUILDS = {"no stack": dict(), "stack": dict(stack_qb_with=1),
          "stack+bring-back": dict(stack_qb_with=1, bring_back=1)}


def pay(rank: float) -> float:
    for lo, hi, v in PAYOUT:
        if lo <= rank <= hi:
            return v
    return 0.0


def payout(score: float, field_scores: np.ndarray) -> float:
    """Our return per $1 against 99 opponents; ties share the places."""
    above = int((field_scores > score).sum())
    ties = int((field_scores == score).sum())
    places = range(above + 1, above + ties + 2)
    return float(np.mean([pay(r) for r in places]))


def main() -> int:
    pred = pd.read_parquet(CURATED / "sim_predictions.parquet")
    pred = pred[pred["salary"].fillna(0) > 0].copy()
    for name, col in VIEWS.items():
        pred[f"v_{col}"] = pred["p_play"] * pred[col].clip(lower=0)
    rows, t0 = [], time.time()
    weeks = pred[["season", "week"]].drop_duplicates().sort_values(
        ["season", "week"]).to_records(index=False)
    for k, (s, w) in enumerate(weeks, 1):
        pool = pred[(pred["season"] == s) & (pred["week"] == w)]
        if len(pool) < 100:
            continue
        opp = field.optimizer_entries(
            pool.assign(projection=pool["v_points_mean"]), n=ENTRIES - 1,
            disagreement=DISAGREEMENT, seed=int(s) * 100 + int(w))
        if opp.size < ENTRIES - 1:
            continue
        for vname, col in VIEWS.items():
            for bname, kw in BUILDS.items():
                lus = optimizer.optimize(pool, objective=f"v_{col}",
                                         n_lineups=1, **kw)
                if not lus:
                    continue
                score = float(lus[0]["dk_scored"].sum())
                rows.append({"season": s, "week": w, "view": vname,
                             "build": bname, "score": score,
                             "ret": payout(score, opp),
                             "cash": float(payout(score, opp) > 0)})
        if k % 10 == 0:
            print(f"  {k}/{len(weeks)} weeks  {time.time() - t0:.0f}s",
                  flush=True)
    r = pd.DataFrame(rows)
    r.to_parquet(CURATED / "hundo_view.parquet", index=False)

    t = r.groupby(["view", "build"]).agg(
        weeks=("ret", "size"), return_per_1=("ret", "mean"),
        cash_rate=("cash", "mean"), mean_score=("score", "mean")
    ).sort_values("return_per_1", ascending=False)
    print(f"\n{r[['season', 'week']].drop_duplicates().shape[0]} weeks, "
          f"{ENTRIES}-entry field of optimizing opponents\n")
    print(t.round(3).to_string())

    best = t.index[0]
    piv = r.pivot_table(index=["season", "week"], columns=["view", "build"],
                        values="ret")
    print(f"\npaired against the best ({best[0]}, {best[1]}):")
    for other in t.index[1:6]:
        d = (piv[best] - piv[other]).dropna()
        p = stats.wilcoxon(piv[best].loc[d.index],
                           piv[other].loc[d.index]).pvalue if (d != 0).any() \
            else float("nan")
        print(f"  vs {other[0]:<12} {other[1]:<17} {d.mean():+.3f} per $1, "
              f"better in {int((d > 0).sum())}, worse in {int((d < 0).sum())}"
              f"  p={p:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Which projection view should a double-up lineup be built on?

The dashboard builds cash lineups on the floor (q30). Six tracked weeks of
2026 said that may be wrong: the floor build cleared the real double-up line
in 1 of 12 lineups, the Hundo build (q70) in 6 of 12 and the ceiling build
in 4 of 12, and the floor build scored LOWER than the upside builds in 10 of
12. Week 11's lines were 169-175 and the floor builds scored 122 and 128,
while a q70 build scored 182. A converged double-up field scores what its
chalk scores, and the chalk is chosen for upside.

Six weeks is a hint, not a measurement. This is the measurement, built like
scripts/hundo_view.py: for every walk-forward week (sim_predictions.parquet,
each week projected only from earlier ones) lineups are built on five views
of the same distribution -- q30 floor, q50, mean, q70, q85 ceiling -- with
and without a QB stack, and each is dropped into a 100-entry contest against
99 opponents who also optimize (the mean projection with 30% noise, the
field docs/MODELING.md 8 found realistic where a sampling field was not).
A double-up pays 2x to the top 43: return per $1 is 2.0 or 0, ties sharing.

Paired by week (rule 2): every view faces the same opponents in the same
week. The field is modelled, not counted, so read the RANKING of views, and
the real-contest lines in data/observed_cash_lines.csv beside it.

    PYTHONPATH=src .venv/Scripts/python.exe scripts/doubleup_view.py
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

from dfs import field, optimizer
from dfs.config import CURATED

ENTRIES = 100
PAID = 43
DISAGREEMENT = 0.30
VIEWS = {"q30 floor": "q30", "q50": "q50", "mean": "points_mean",
         "q70": "q70", "q85 ceiling": "q85"}
BUILDS = {"no stack": dict(), "stack": dict(stack_qb_with=1)}


def payout(score: float, field_scores: np.ndarray) -> float:
    """Return per $1 against the 99 opponents; ties share the places."""
    above = int((field_scores > score).sum())
    ties = int((field_scores == score).sum())
    places = np.arange(above + 1, above + ties + 2)
    return float(np.mean(np.where(places <= PAID, 2.0, 0.0)))


def main() -> int:
    pred = pd.read_parquet(CURATED / "sim_predictions.parquet")
    pred = pred[pred["salary"].fillna(0) > 0].copy()
    for col in VIEWS.values():
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
        line = np.sort(opp)[::-1][PAID - 1]
        for vname, col in VIEWS.items():
            for bname, kw in BUILDS.items():
                lus = optimizer.optimize(pool, objective=f"v_{col}",
                                         n_lineups=1, **kw)
                if not lus:
                    continue
                score = float(lus[0]["dk_scored"].fillna(0).sum())
                ret = payout(score, opp)
                rows.append({"season": s, "week": w, "view": vname,
                             "build": bname, "score": score, "ret": ret,
                             "cash": float(ret > 0), "margin": score - line})
        if k % 10 == 0:
            print(f"  {k}/{len(weeks)} weeks  {time.time() - t0:.0f}s",
                  flush=True)
    r = pd.DataFrame(rows)
    r.to_parquet(CURATED / "doubleup_view.parquet", index=False)

    t = r.groupby(["view", "build"]).agg(
        weeks=("ret", "size"), return_per_1=("ret", "mean"),
        cash_rate=("cash", "mean"), mean_score=("score", "mean"),
        mean_margin=("margin", "mean")
    ).sort_values("return_per_1", ascending=False)
    print(f"\n{r[['season', 'week']].drop_duplicates().shape[0]} weeks, "
          f"{ENTRIES}-entry double-up, {PAID} paid at 2x, "
          "optimizing opponents\n")
    print(t.round(3).to_string())

    piv = r.pivot_table(index=["season", "week"], columns=["view", "build"],
                        values="score")
    cash = r.pivot_table(index=["season", "week"], columns=["view", "build"],
                         values="cash")
    base = ("q30 floor", "no stack")
    print(f"\npaired against the current cash build {base}:")
    for other in t.index:
        if other == base:
            continue
        d = (piv[other] - piv[base]).dropna()
        c = (cash[other] - cash[base]).dropna()
        p = stats.wilcoxon(piv[other].loc[d.index],
                           piv[base].loc[d.index]).pvalue
        won, lost = int((c > 0).sum()), int((c < 0).sum())
        pc = stats.binomtest(won, won + lost).pvalue if won + lost else 1.0
        print(f"  {other[0]:<12} {other[1]:<9} score {d.mean():+6.2f} "
              f"(higher {int((d > 0).sum())}/{len(d)}, p={p:.4f})  cash "
              f"{c.mean():+.3f}: cashed where floor missed {won}, "
              f"the reverse {lost} (p={pc:.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

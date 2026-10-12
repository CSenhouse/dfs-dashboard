"""Precompute everything the dashboard's Model page shows.

    python scripts/diagnostics.py

A walk-forward run takes about twenty minutes, so it cannot happen on demand
inside the app. This does it once and writes the results to data/curated.

The most useful output is `worst_weeks.parquet`: for each week the model did
badly, both the lineup it built and the lineup that was actually available,
side by side. Aggregate metrics say how well it does; that file says what it
got wrong, which is the thing a football person can look at and diagnose.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from dfs import backtest, baseline, features, model, optimizer
from dfs.config import CURATED

MODELS = {
    "two_stage": model.two_stage,
    "two_stage_floor": model.two_stage_floor,
    "baseline": baseline.expected,
}
WORST_N = 12


def main() -> int:
    f = pd.read_parquet(CURATED / "features.parquet")
    print(f"{len(f):,} rows, {f.groupby(['season', 'week']).ngroups} weeks",
          flush=True)

    weekly, preds = [], {}
    for name, fn in MODELS.items():
        print(f"walk-forward: {name} ...", flush=True)
        pred = backtest.walk_forward(f, fn, min_train_weeks=8)
        preds[name] = pred
        acc = backtest.accuracy(pred)
        lu = backtest.lineup_results(pred, n_lineups=1)
        merged = acc.merge(lu, on=["season", "week"], how="outer")
        merged["model"] = name
        weekly.append(merged)

    weekly_df = pd.concat(weekly, ignore_index=True)
    weekly_df.to_parquet(CURATED / "weekly.parquet", index=False)
    print(f"wrote weekly.parquet ({len(weekly_df)} rows)")

    calib = backtest.calibration(preds["two_stage"])
    calib.to_parquet(CURATED / "calibration.parquet", index=False)
    print("wrote calibration.parquet")

    imp = model.importance(backtest.prepare(f), top=40)
    imp.to_parquet(CURATED / "importance.parquet", index=False)
    print("wrote importance.parquet")

    # --- the weeks it got wrong ---------------------------------------------
    pred = preds["two_stage"]
    ours = weekly_df[weekly_df["model"] == "two_stage"].dropna(
        subset=["pct_of_ceiling"]
    )
    worst = ours.nsmallest(WORST_N, "pct_of_ceiling")

    rows = []
    for r in worst.itertuples():
        g = pred[(pred["season"] == r.season) & (pred["week"] == r.week)]
        if g.empty:
            continue
        built = optimizer.optimize(g, objective="projection", n_lineups=1)
        best = optimizer.optimize(
            g.assign(_a=g["dk_scored"]), objective="_a", n_lineups=1
        )
        if not built or not best:
            continue
        for tag, lineup in (("picked", built[0]), ("best available", best[0])):
            out = lineup[["player_name", "position", "team", "salary",
                          "projection", "dk_scored"]].copy()
            out["which"] = tag
            out["season"] = r.season
            out["week"] = r.week
            out["lineup_actual"] = r.actual
            out["ceiling"] = r.ceiling
            rows.append(out)
        print(f"  {int(r.season)} wk{int(r.week):>2}: "
              f"scored {r.actual:.0f} of {r.ceiling:.0f} available", flush=True)

    if rows:
        pd.concat(rows, ignore_index=True).to_parquet(
            CURATED / "worst_weeks.parquet", index=False
        )
        print(f"wrote worst_weeks.parquet ({WORST_N} weeks)")

    # --- biggest individual misses ------------------------------------------
    p = pred.dropna(subset=["projection", "dk_scored"]).copy()
    p["error"] = p["dk_scored"] - p["projection"]
    cols = ["season", "week", "player_name", "position", "team", "salary",
            "projection", "dk_scored", "error"]
    misses = pd.concat([
        p.nlargest(150, "error")[cols].assign(kind="under-projected"),
        p.nsmallest(150, "error")[cols].assign(kind="over-projected"),
    ], ignore_index=True)
    misses.to_parquet(CURATED / "misses.parquet", index=False)
    print(f"wrote misses.parquet ({len(misses)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

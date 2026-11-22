"""Does the Showdown cash build do better WITH its punt plays or without them?

The optimizer usually fills one or two FLEX spots with a $200-$1,600 player
projected near zero, the price of four stars. That has looked fine in real
contests (four of five cashed in week 10 of 2026, one of them with two
punts that both scored 0), but five contests is not a measurement.

For every finished Showdown slate, the dashboard's cash lineup is built
twice from walk-forward floor projections (each week from earlier weeks
only), with the same rule the dashboard applies (at least one QB):

    as built   punts allowed -- what the dashboard does
    no punts   every player whose floor is under PUNT removed first

Both are scored on what happened (captain 1.5x) and ranked inside a field of
lineups sampled the way entrants draft (field.sample_showdown_indices, as in
scripts/showdown_rules.py). Paired by slate. Cash = beating 56.5% of the
field, a double-up line. The field is modelled, not counted; read the
DIFFERENCE between the two builds, not the absolute rates.

    PYTHONPATH=src .venv/Scripts/python.exe scripts/showdown_punts.py
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

from dfs import backtest, field, model, optimizer
from dfs.config import CURATED

N_FIELD = 3000
FIELD_SKILL = 3.0
LINE = 0.565
PUNT = 2.0            # a floor under two points is a punt


def main() -> int:
    f = pd.read_parquet(CURATED / "showdown_features.parquet")
    print(f"{len(f):,} rows, {f['draft_group_id'].nunique()} slates; "
          "walk-forward projections ...", flush=True)
    mean = backtest.walk_forward(f, model.two_stage, min_train_weeks=8)
    floor = backtest.walk_forward(f, model.two_stage_floor, min_train_weeks=8)
    key = ["draft_group_id", "player_dk_id"]
    pred = mean.merge(floor[key + ["projection"]].rename(
        columns={"projection": "floor"}), on=key, how="inner")
    pred = pred.dropna(subset=["projection", "floor"])
    pred = pred[pred["salary"] > 0]

    rows, t0 = [], time.time()
    slates = list(pred.groupby("draft_group_id"))
    for k, (dgid, pool) in enumerate(slates, 1):
        pool = pool.reset_index(drop=True)
        if len(pool) < 15 or pool["team"].nunique() != 2:
            continue
        idx, w = field.sample_showdown_indices(
            pool, n=N_FIELD, field_skill=FIELD_SKILL, seed=int(dgid) % 100000)
        if idx.size == 0:
            continue
        pts = pool["dk_scored"].fillna(0).to_numpy(float)
        field_scores = np.sort((pts[idx] * w).sum(axis=1))
        line = np.quantile(field_scores, LINE)

        built = {}
        for name, p in (("as built", pool),
                        ("no punts", pool[pool["floor"] >= PUNT])):
            lus = optimizer.optimize_showdown(
                p.assign(projection=p["floor"]), objective="projection",
                n_lineups=1, min_qb=1)
            if not lus:
                break
            lu = lus[0]
            mult = np.where(lu["slot"] == "CPT", 1.5, 1.0)
            built[name] = dict(
                score=float((lu["dk_scored"].fillna(0) * mult).sum()),
                floor=float(lu["lineup_points"].sum()),
                punts=int(((lu["floor"] < PUNT) & (lu["slot"] != "CPT")).sum()))
        if len(built) < 2:
            continue
        a, b = built["as built"], built["no punts"]
        rows.append(dict(dgid=dgid, punts=a["punts"],
                         score_a=a["score"], score_b=b["score"],
                         floor_a=a["floor"], floor_b=b["floor"],
                         cash_a=a["score"] > line, cash_b=b["score"] > line))
        if k % 200 == 0:
            print(f"  {k}/{len(slates)} slates  {time.time() - t0:.0f}s",
                  flush=True)

    r = pd.DataFrame(rows)
    r.to_parquet(CURATED / "showdown_punts.parquet", index=False)
    print(f"\n{len(r)} slates. The as-built lineup used punts on "
          f"{(r['punts'] > 0).mean():.0%} of them "
          f"({(r['punts'] == 1).mean():.0%} one, {(r['punts'] >= 2).mean():.0%} "
          "two or more).\n")
    for label, d in (("all slates", r),
                     ("slates where it punted", r[r["punts"] > 0]),
                     ("one punt", r[r["punts"] == 1]),
                     ("two or more", r[r["punts"] >= 2])):
        if len(d) < 20:
            continue
        diff = d["score_a"] - d["score_b"]
        changed = d["cash_a"] != d["cash_b"]
        p = stats.binomtest(int((d["cash_a"] & ~d["cash_b"]).sum()),
                            int(changed.sum())).pvalue if changed.any() \
            else float("nan")
        print(f"{label:<24} n={len(d):>4}  cash {d['cash_a'].mean():.1%} "
              f"as built vs {d['cash_b'].mean():.1%} no punts  "
              f"(differ on {int(changed.sum())}, p={p:.3f})  |  score "
              f"{diff.mean():+.2f}, as built higher on "
              f"{int((diff > 0).sum())}/{int((diff != 0).sum())}  |  floor "
              f"{(d['floor_a'] - d['floor_b']).mean():+.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Estimate the cash line and our win rate against it.

    python scripts/cash_line.py

Every result so far has compared our lineup against a hindsight ceiling nobody
reaches, or against an assumed cash line of "130-150" that was never measured.
Neither answers the question that decides whether this is worth playing: what
does the rest of the field score?

Real contest results are paid data. Absent them we simulate a field, read the
median off it as the double-up threshold, and report how often we clear it.
Two things about the field are assumptions, not observations, so both are
swept rather than fixed:

  projection source   what entrants know: salary-implied (naive) through to
                      our own model's output (sharp)
  field_skill         how tightly they concentrate on their best options

Read the spread, not any single number.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from dfs import backtest, baseline, field, model
from dfs.config import CURATED

# A double-up pays about 1.8x, so break-even needs better than a coin flip.
PAYOUT_MULTIPLIER = 1.8
BREAK_EVEN = 1.0 / PAYOUT_MULTIPLIER

SKILLS = (1.0, 2.0, 3.0, 4.0)
FIELDS = {
    "naive (salary-implied)": baseline.expected_salary,
    "informed (recent form)": baseline.expected,
    "sharp (our model)": None,  # reuse the walk-forward projection
}
N_ENTRIES = 3000


def main() -> int:
    f = pd.read_parquet(CURATED / "features.parquet")
    print(f"dataset: {len(f):,} rows, "
          f"{f.groupby(['season', 'week']).ngroups} weeks")
    print("projecting (walk-forward) ...", flush=True)
    pred = backtest.walk_forward(f, model.two_stage, min_train_weeks=8)
    if pred.empty:
        print("no predictions")
        return 1

    ours = backtest.lineup_results(pred, n_lineups=1)
    ours = ours.set_index(["season", "week"])["actual"]
    print(f"weeks evaluated: {len(ours)}   our mean: {ours.mean():.1f}")
    print()

    data = backtest.prepare(f)
    weeks = list(pred.groupby(["season", "week"]))

    rows = []
    for fname, fn in FIELDS.items():
        for skill in SKILLS:
            lines, beats = {}, {}
            for (season, week), g in weeks:
                if (season, week) not in ours.index:
                    continue
                g = g.copy()
                if fn is not None:
                    past = data[(data["season"] < season)
                                | ((data["season"] == season)
                                   & (data["week"] < week))]
                    if past.empty:
                        continue
                    g["projection"] = fn(past, g).to_numpy()
                line = field.cash_line(
                    g, n=N_ENTRIES, field_skill=skill,
                    seed=int(season) * 100 + int(week),
                )
                if np.isnan(line):
                    continue
                lines[(season, week)] = line
                beats[(season, week)] = ours.loc[(season, week)] > line
            if not lines:
                continue
            L, B = pd.Series(lines), pd.Series(beats)
            roi = B.mean() * PAYOUT_MULTIPLIER - 1.0
            rows.append({
                "field": fname, "skill": skill,
                "cash_line": round(L.mean(), 1),
                "line_sd": round(L.std(), 1),
                "our_score": round(ours.reindex(L.index).mean(), 1),
                "win_rate": round(B.mean(), 3),
                "roi": round(roi, 3),
            })
            print(f"  {fname:<24} skill {skill}: line {L.mean():6.1f}  "
                  f"win {B.mean():5.1%}  ROI {roi:+6.1%}", flush=True)

    out = pd.DataFrame(rows)
    print()
    print("=== estimated double-up economics ===")
    print(out.to_string(index=False))
    print()
    print(f"break-even win rate at {PAYOUT_MULTIPLIER}x: {BREAK_EVEN:.1%}")
    print(f"cash line ranges {out['cash_line'].min():.0f} to "
          f"{out['cash_line'].max():.0f} across assumptions; "
          f"we average {ours.mean():.1f}")
    bad = out[out["win_rate"] <= BREAK_EVEN]
    if bad.empty:
        print("profitable against every field simulated here")
    else:
        print("NOT profitable against: "
              + ", ".join(f"{r.field}@{r.skill}" for r in bad.itertuples()))
    out.to_csv(CURATED / "cash_line.csv", index=False)
    print(f"\nwrote {CURATED / 'cash_line.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

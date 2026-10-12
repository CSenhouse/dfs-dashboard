"""How does our edge hold up against opponents who also optimize?

    python scripts/sharp_field.py --weeks 25 --entries 30

scripts/cash_line.py found us profitable against every simulated field, but
that field only SAMPLED players. Our entry solves a MILP and theirs did not,
so part of the measured edge was the optimizer rather than the projections.

Two opponent types, and the distinction decides what the result is worth:

  mirror        opponents hold our own projection plus noise. We are the
                centre of that field and they are degraded copies, so we win
                by construction. Reported only as an optimistic bound and to
                check the boundary case: at zero noise they reproduce our
                lineup exactly and we can only tie.

  independent   opponents hold a genuinely different model's projection. That
                model can be better than ours on a given week, so this test
                can actually go against us. It is the one to believe.

One MILP solve per opponent per week, so this runs on a sample of weeks.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from dfs import backtest, baseline, field, model
from dfs.config import CURATED

PAYOUT_MULTIPLIER = 1.8
BREAK_EVEN = 1.0 / PAYOUT_MULTIPLIER


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weeks", type=int, default=25)
    ap.add_argument("--entries", type=int, default=30)
    ap.add_argument("--min-train-weeks", type=int, default=8)
    args = ap.parse_args()

    f = pd.read_parquet(CURATED / "features.parquet")
    print("projecting ours (walk-forward) ...", flush=True)
    pred = backtest.walk_forward(f, model.two_stage,
                                 min_train_weeks=args.min_train_weeks)
    ours = backtest.lineup_results(pred, n_lineups=1)
    ours = ours.set_index(["season", "week"])["actual"]

    print("projecting the independent opponent model ...", flush=True)
    rival = backtest.walk_forward(f, baseline.expected,
                                  min_train_weeks=args.min_train_weeks)
    rival = rival.set_index(["season", "week", "player_dk_id"])["projection"]

    groups = [(k, g) for k, g in pred.groupby(["season", "week"])
              if k in ours.index]
    step = max(1, len(groups) // args.weeks)
    groups = groups[::step][:args.weeks]
    print(f"sampling {len(groups)} weeks, {args.entries} optimizing opponents each")
    print()

    scenarios = [
        ("mirror", None, 0.0),
        ("mirror", None, 0.15),
        ("mirror", None, 0.30),
        ("independent", "rival", 0.0),
        ("independent", "rival", 0.15),
        ("independent", "rival", 0.30),
    ]

    rows = []
    for label, col, dis in scenarios:
        lines, beats, ties = {}, {}, {}
        for (season, week), g in groups:
            g = g.copy()
            if col:
                idx = pd.MultiIndex.from_arrays(
                    [g["season"], g["week"], g["player_dk_id"]]
                )
                g["rival"] = rival.reindex(idx).to_numpy()
                if g["rival"].notna().sum() < 50:
                    continue
            scores = field.optimizer_entries(
                g, n=args.entries, disagreement=dis, opponent_col=col,
                seed=int(season) * 100 + int(week),
            )
            if scores.size == 0:
                continue
            us = ours.loc[(season, week)]
            lines[(season, week)] = float(np.median(scores))
            beats[(season, week)] = float((us > scores).mean())
            ties[(season, week)] = float(np.isclose(us, scores).mean())
        if not lines:
            continue
        L, B, T = pd.Series(lines), pd.Series(beats), pd.Series(ties)
        roi = B.mean() * PAYOUT_MULTIPLIER - 1.0
        rows.append({
            "opponents": label, "noise": dis,
            "opp_median": round(L.mean(), 1),
            "our_score": round(ours.reindex(L.index).mean(), 1),
            "beat_rate": round(B.mean(), 3),
            "tie_rate": round(T.mean(), 3),
            "roi": round(roi, 3),
        })
        print(f"  {label:<12} noise {dis:.2f}: opp median {L.mean():6.1f}   "
              f"we score {ours.reindex(L.index).mean():6.1f}   "
              f"beat {B.mean():5.1%}   ties {T.mean():4.1%}   "
              f"ROI {roi:+6.1%}", flush=True)

    out = pd.DataFrame(rows)
    print()
    print("=== versus opponents who also optimize ===")
    print(out.to_string(index=False))
    print()
    print(f"break-even beat rate at {PAYOUT_MULTIPLIER}x: {BREAK_EVEN:.1%}")
    print("Believe the 'independent' rows. The 'mirror' rows cannot go "
          "against us by construction.")
    out.to_csv(CURATED / "sharp_field.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())

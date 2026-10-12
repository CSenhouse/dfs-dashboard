"""Rebuild features and run the full model comparison.

    python scripts/evaluate.py            # rebuild features, then evaluate
    python scripts/evaluate.py --no-build # evaluate the existing feature table
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from scipy import stats

from dfs import backtest, baseline, features, model
from dfs.config import CURATED

MODELS = (
    ("baseline", baseline.expected),
    ("two_stage", model.two_stage),
    ("two_stage_floor", model.two_stage_floor),
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-build", action="store_true")
    ap.add_argument("--min-train-weeks", type=int, default=8)
    args = ap.parse_args()

    if not args.no_build:
        df = pd.read_parquet(CURATED / "player_week.parquet")
        print(f"building features over {len(df):,} player-slate rows ...",
              flush=True)
        f = features.build(df)
        f.to_parquet(CURATED / "features.parquet", index=False)
    else:
        f = pd.read_parquet(CURATED / "features.parquet")

    cols = features.feature_columns(f)
    print(f"rows={len(f):,}  features={len(cols)}  "
          f"seasons={sorted(f['season'].unique())}  "
          f"weeks={f.groupby(['season', 'week']).ngroups}")
    print()

    sp, lu, calib = {}, {}, {}
    for name, fn in MODELS:
        pred = backtest.walk_forward(f, fn, min_train_weeks=args.min_train_weeks)
        acc = backtest.accuracy(pred)
        L = backtest.lineup_results(pred, n_lineups=1)
        print(backtest.summarize(acc, L, name), flush=True)
        sp[name] = acc.set_index(["season", "week"])["spearman"]
        lu[name] = L.set_index(["season", "week"])["actual"]
        calib[name] = backtest.calibration(pred)

    S, Lf = pd.DataFrame(sp), pd.DataFrame(lu)
    print()
    for k in [m[0] for m in MODELS[1:]]:
        d = (S[k] - S["baseline"]).dropna()
        d2 = (Lf[k] - Lf["baseline"]).dropna()
        print(f"{k:<16} "
              f"spearman {d.mean():+.4f} {int((d > 0).sum())}/{len(d)} "
              f"p={stats.ttest_rel(S[k], S['baseline']).pvalue:.2e}  |  "
              f"lineup {d2.mean():+6.2f} {int((d2 > 0).sum())}/{len(d2)} "
              f"p={stats.ttest_rel(Lf[k], Lf['baseline']).pvalue:.4f}")

    best = Lf["two_stage"].dropna()
    print()
    print(f"two_stage lineup: n={len(best)} weeks  mean={best.mean():.1f}  "
          f"sd={best.std():.1f}  se={best.std() / np.sqrt(len(best)):.1f}")
    for t in (120, 130, 140, 150):
        print(f"  weeks >= {t}: {int((best >= t).sum())}/{len(best)} "
              f"({(best >= t).mean():.0%})")

    # Persist headline numbers so the dashboard can show them without
    # re-running an hour of walk-forward training.
    import json
    lu_two = Lf["two_stage"].dropna()
    sp_two = S["two_stage"].dropna()
    (CURATED / "metrics.json").write_text(json.dumps({
        "seasons": f"{int(f['season'].min())}-{int(f['season'].max())}",
        "weeks": int(len(lu_two)),
        "spearman": round(float(sp_two.mean()), 4),
        "spearman_baseline": round(float(S["baseline"].mean()), 4),
        "lineup_mean": round(float(lu_two.mean()), 2),
        "lineup_edge": round(float((lu_two - Lf["baseline"]).mean()), 2),
        "lineup_se": round(float(lu_two.std() / np.sqrt(len(lu_two))), 2),
        "pct_weeks_over_130": round(float((lu_two >= 130).mean()), 3),
    }, indent=2))

    print()
    print("=== calibration by projection decile (two_stage) ===")
    c = calib.get("two_stage")
    if c is not None and not c.empty:
        print(c.round(3).to_string(index=False))
        top = c.iloc[-1]
        print(f"top decile over-projects by {top['gap']:+.2f} points "
              f"({top['gap'] / top['predicted']:+.1%}) -- this is the bucket "
              f"the optimizer rosters from")

    print()
    print("=== top features (conditional points model) ===")
    print(model.importance(backtest.prepare(f), top=20).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

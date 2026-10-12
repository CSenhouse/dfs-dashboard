"""Three-way paired test of how absences should enter the rolling windows.

  old       window rolls over appearances, so it silently skips missed weeks
            and reaches back to whenever the player last played
  zeroed    window rolls over calendar weeks, absences counted as zeros
  staleness window rolls over appearances (form still means "when he plays")
            but a weeks_since_played feature tells the model how old it is

Same model, same weeks, compared pairwise. The obvious fix is `zeroed`, and
the obvious fix is not automatically the right one: the model already has a
separate P(play) stage, so zeroing form makes the regressor pay for absence
a second time.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy import stats

from dfs import backtest, features, model
from dfs.config import CURATED

VARIANTS = {
    "old": dict(calendar_aware=False, zero_absences=True),
    "zeroed": dict(calendar_aware=True, zero_absences=True),
    "staleness": dict(calendar_aware=True, zero_absences=False),
}


def main() -> int:
    raw = pd.read_parquet(CURATED / "player_week.parquet")
    sp, lu = {}, {}
    for name, kw in VARIANTS.items():
        print(f"building features: {name} ...", flush=True)
        f = features.build(raw, **kw)
        f.to_parquet(CURATED / f"features_{name}.parquet", index=False)
        print("  walk-forward ...", flush=True)
        pred = backtest.walk_forward(f, model.two_stage, min_train_weeks=8)
        acc = backtest.accuracy(pred)
        L = backtest.lineup_results(pred, n_lineups=1)
        print(backtest.summarize(acc, L, name), flush=True)
        sp[name] = acc.set_index(["season", "week"])["spearman"]
        lu[name] = L.set_index(["season", "week"])["actual"]

    S, Lf = pd.DataFrame(sp), pd.DataFrame(lu)
    print()
    print("=== paired against 'old' ===")
    for k in ("zeroed", "staleness"):
        ds, dl = (S[k] - S["old"]).dropna(), (Lf[k] - Lf["old"]).dropna()
        print(f"{k:<10} spearman {ds.mean():+.5f} {int((ds > 0).sum())}/{len(ds)}"
              f" p={stats.ttest_rel(S[k], S['old']).pvalue:.4f}  |  "
              f"lineup {dl.mean():+6.2f} {int((dl > 0).sum())}/{len(dl)}"
              f" p={stats.ttest_rel(Lf[k], Lf['old']).pvalue:.4f}"
              f" (se {dl.std() / np.sqrt(len(dl)):.2f})")
    ds = (S["staleness"] - S["zeroed"]).dropna()
    dl = (Lf["staleness"] - Lf["zeroed"]).dropna()
    print()
    print("=== staleness minus zeroed ===")
    print(f"spearman {ds.mean():+.5f} {int((ds > 0).sum())}/{len(ds)}"
          f" p={stats.ttest_rel(S['staleness'], S['zeroed']).pvalue:.4f}  |  "
          f"lineup {dl.mean():+6.2f} {int((dl > 0).sum())}/{len(dl)}"
          f" p={stats.ttest_rel(Lf['staleness'], Lf['zeroed']).pvalue:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

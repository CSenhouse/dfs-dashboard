"""Do the hand edits carry signal the model does not have?

This is the reason `data/overrides/` exists. Whether the dashboard's LINEUPS
beat a hand-built one cannot be settled by playing -- weekly lineup scores
carry about a 25-point standard deviation, so even paired on one slate it needs
dozens of weeks, which is more than a season. An override is a far smaller unit
of judgement: one player, one number, against one actual score. Twenty a week
reaches a usable sample in a month.

Three questions, in increasing order of how much they ask of the data.

    direction   of the edits made, how many moved TOWARD what happened?
                A coin flip is 50%. This is a binomial test and it is the one
                worth reading first, because it costs one bit per edit and
                needs no distributional assumption.

    ranking     on the overridden players only, does the edited number rank
                outcomes better than the model's own did?

    residual    does the edit carry signal AFTER the model's number is removed
                from both sides? Same partial rank correlation that found
                ownership holds +0.29 the model lacks (docs/MODELING.md 21).

Read `direction` as the headline and the other two as colour until there are a
few hundred rows. A rank correlation over twenty players is almost no evidence.

Overridden players are a SELECTED sample by construction -- they are the ones
that looked wrong enough to touch -- so none of this says anything about
players left alone. The claim being tested is narrow and precise: when you
decide to overrule the model, are you right more often than not?

    PYTHONPATH=src .venv/Scripts/python.exe scripts/override_value.py
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy import stats

from dfs import overrides, slates


def partial(x: np.ndarray, y: np.ndarray, given: np.ndarray) -> tuple[float, float]:
    """Correlation of x and y once `given` is removed from both, on ranks."""
    if len(x) < 5 or np.ptp(given) == 0 or np.ptp(x) == 0:
        # A constant leaves nothing to remove and nothing to rank, and numpy
        # answers with a RankWarning and a nan rather than saying so.
        return float("nan"), float("nan")
    rx, ry, rg = (stats.rankdata(v) for v in (x, y, given))
    res = lambda a: a - np.poly1d(np.polyfit(rg, a, 1))(rg)
    r = stats.pearsonr(res(rx), res(ry))
    return float(r.statistic), float(r.pvalue)


def collect() -> pd.DataFrame:
    """Every recorded override joined to what the player actually scored."""
    if not overrides.DIR.exists():
        return pd.DataFrame()
    rows = []
    for path in sorted(overrides.DIR.glob("*.csv")):
        try:
            dgid = int(path.stem)
        except ValueError:
            continue
        edits = overrides.load(dgid)
        if edits.empty:
            continue
        up = slates.assemble(dgid)
        if up.empty:
            print(f"  slate {dgid} not cached, skipping", file=sys.stderr)
            continue
        season, week = int(up["season"].iloc[0]), int(up["week"].iloc[0])
        if not slates.week_is_complete(season, week):
            print(f"  {season} week {week} is not finished, skipping "
                  f"{len(edits)} edit(s)")
            continue
        # A rostered player with no box-score row scored zero (rule 5).
        actual = up[["player_dk_id", "dk_actual"]].copy()
        actual["dk_actual"] = actual["dk_actual"].fillna(0.0)
        j = edits.merge(actual, on="player_dk_id", how="inner")
        j["season"], j["week"] = season, week
        rows.append(j)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main(argv: list[str]) -> int:
    d = collect()
    if d.empty:
        print("No overrides to judge yet.")
        print()
        print("Edit a projection on the Adjust projections tab, play the week,")
        print("then run this once the box scores land. Each edit is one paired")
        print("observation -- model number, your number, what happened.")
        return 0

    d = d.dropna(subset=["model_value", "user_value", "dk_actual"])
    print(f"{len(d)} edit(s) across "
          f"{d.groupby(['season', 'week']).ngroups} finished week(s)\n")

    # Direction. An edit that moved toward the outcome is a hit. Edits that
    # landed exactly on the model's number cannot be scored and are already
    # excluded by `record`.
    toward = np.sign(d["user_value"] - d["model_value"]) == np.sign(
        d["dk_actual"] - d["model_value"])
    n, k = len(d), int(toward.sum())
    p = stats.binomtest(k, n, 0.5).pvalue if n else float("nan")
    print(f"  direction  {k}/{n} edits moved toward what happened "
          f"({k / n:.1%}, p={p:.3f})")

    # Ranking, on the overridden players only.
    if n >= 5:
        rm = stats.spearmanr(d["model_value"], d["dk_actual"]).statistic
        ru = stats.spearmanr(d["user_value"], d["dk_actual"]).statistic
        if np.isfinite(rm) and np.isfinite(ru):
            print(f"  ranking    model {rm:+.4f} vs yours {ru:+.4f} "
                  f"({ru - rm:+.4f})")
        else:
            print("  ranking    not defined -- one of the two numbers is the "
                  "same for every edit")
        r, pv = partial(d["user_value"].to_numpy(), d["dk_actual"].to_numpy(),
                        d["model_value"].to_numpy())
        if np.isfinite(r):
            print(f"  residual   your edit holds {r:+.4f} after the model's "
                  f"own number is removed (p={pv:.4f})")

    # Size of the edits, because a hit rate means little if every edit was 0.1.
    moved = (d["user_value"] - d["model_value"]).abs()
    print(f"\n  edits move the projection by {moved.mean():.2f} points on "
          f"average (median {moved.median():.2f}, max {moved.max():.2f})")

    if p >= 0.05:
        print(f"\n  Not settled: {n} edits, direction at p={p:.3f}. If the "
              f"true hit rate\n  is 65%, that test reaches p<0.05 at around 60 "
              f"edits -- roughly three\n  weeks of steady use.")
    else:
        print(f"\n  Direction is significant at p={p:.3f} on {n} edits. The "
              f"other two\n  lines need a few hundred before they mean much; "
              f"this one does not.")

    by = d.groupby(["season", "week"]).apply(
        lambda g: pd.Series({
            "edits": len(g),
            "toward": int((np.sign(g["user_value"] - g["model_value"])
                           == np.sign(g["dk_actual"] - g["model_value"])).sum()),
        }), include_groups=False)
    print("\n=== by week ===")
    print(by.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

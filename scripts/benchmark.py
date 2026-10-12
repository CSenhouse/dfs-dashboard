"""Are our projections good, or merely not obviously bad?

This is the question `docs/MODELING.md` has listed as an open gap since the
beginning: everything measured here compares the model to a baseline, to the
crowd, or to itself. None of that says whether a professional service does
better. A Daily Fantasy Fuel export does, in one slate.

Three things come out of it, in increasing order of what they would change.

    ranking     spearman against what actually happened, for ours, theirs,
                salary alone and the two averaged. Salary is in because it
                beat our projection on the only Showdown slate ever measured,
                and a benchmark without a dumb baseline is not a benchmark.

    residual    does their number carry signal after ours is removed from both
                sides, and does ours after theirs? Both can be true. If only
                theirs survives, the model is strictly behind. This is the
                same partial rank correlation that found ownership holding
                +0.29 the model lacks.

    ownership   their `ownership_projection` against what the field really
                rostered, from a standings export. Ownership is the one input
                measured to carry signal we do not have and the one that
                cannot be derived from a box score, so whether their estimate
                of it is any good decides whether buying it is worth anything.

Run it on a FINISHED slate, with a DFF export taken before lock and a contest
standings export taken after:

    PYTHONPATH=src .venv/Scripts/python.exe scripts/benchmark.py \\
        154078 data/projections/DFF_NFL_cheatsheet_2026-10-04.csv \\
        --contest 196169740

A caution that applies to every number here: one slate is one draw. The
projection comparison needs several before a difference of a few hundredths
means anything, and the pattern this project keeps hitting is that a
single-slate result looks decisive and then reverses.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from dfs import backtest, contests, features, model, slates
from dfs.config import CURATED, DATA
from dfs.sources import dff, dk_api


def partial(x: np.ndarray, y: np.ndarray, given: np.ndarray) -> tuple[float, float]:
    """Correlation of x and y once `given` is removed from both, on ranks."""
    if len(x) < 10 or np.ptp(given) == 0 or np.ptp(x) == 0:
        return float("nan"), float("nan")
    rx, ry, rg = (stats.rankdata(v) for v in (x, y, given))
    res = lambda a: a - np.poly1d(np.polyfit(rg, a, 1))(rg)
    r = stats.pearsonr(res(rx), res(ry))
    return float(r.statistic), float(r.pvalue)


def our_projection(dgid: int, showdown: bool) -> pd.DataFrame:
    up = slates.assemble(dgid)
    if up.empty:
        raise SystemExit(f"draft group {dgid} is not cached")
    season, week = int(up["season"].iloc[0]), int(up["week"].iloc[0])
    name = "showdown_features" if showdown else "features"
    past = backtest.prepare(pd.read_parquet(CURATED / f"{name}.parquet"))
    past = past[(past["season"] < season)
                | ((past["season"] == season) & (past["week"] < week))]
    cur = features.build(up, salary_within_slate=showdown)
    cur = cur[cur["salary"].fillna(0) > 0].reset_index(drop=True)
    return cur.merge(model.two_stage_all(past, cur), on="player_dk_id",
                     how="left")


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    dgid = int(argv[0])
    export = Path(argv[1])
    contest = argv[argv.index("--contest") + 1] if "--contest" in argv else None

    try:
        theirs = dff.load(export)      # refuses an export taken before posting
    except (ValueError, FileNotFoundError) as exc:
        print(f"cannot use this export: {exc}", file=sys.stderr)
        return 1
    print(f"{export.name}: {len(theirs)} players, ownership "
          f"{'INCLUDED' if dff.has_ownership(theirs) else 'absent'}")

    up = slates.assemble(dgid)
    showdown = up["team"].nunique() == 2 and len(up) < 120
    ours = our_projection(dgid, showdown)
    out = dk_api.unavailable(dk_api.draftables(dgid))
    ours = ours[~ours["player_dk_id"].isin(out["player_dk_id"])]

    j = dff.match(theirs, ours[["player_dk_id", "player_name", "position",
                                "team", "salary"]])
    rate = dff.match_rate(j)
    on = j[j["on_slate"]]
    print(f"matched {on['matched'].sum()}/{len(on)} ({rate:.1%}) of players "
          f"on a team this slate contains; {(~j['on_slate']).sum()} rows in "
          f"the export belong to other games that day")
    if rate < 0.95:
        print("  below 95% -- is this export the same slate as that draft "
              "group?", file=sys.stderr)

    j = j[j["matched"]].merge(
        ours[["player_dk_id", "proj_mean", "dk_actual"]],
        on="player_dk_id", how="left")
    j["actual"] = j["dk_actual"].fillna(0.0)
    j = j.dropna(subset=["proj_mean", "dff_proj"])
    if j["dk_actual"].notna().sum() == 0:
        print("\nno actual scores yet -- run this after the games", file=sys.stderr)
        return 1

    j["blend"] = 0.5 * (
        j["proj_mean"] / max(j["proj_mean"].mean(), 1e-9)
        + j["dff_proj"] / max(j["dff_proj"].mean(), 1e-9))

    print(f"\n=== ranking {len(j)} players against what happened ===")
    for label, col in (("ours", "proj_mean"), ("DFF", "dff_proj"),
                       ("salary alone", "salary"), ("the two averaged", "blend")):
        r = stats.spearmanr(j[col], j["actual"])
        print(f"  {label:<18} spearman {r.statistic:+.4f}  (p={r.pvalue:.2g})")

    print("\n=== what each knows that the other does not ===")
    r, p = partial(j["dff_proj"].to_numpy(), j["actual"].to_numpy(),
                   j["proj_mean"].to_numpy())
    print(f"  DFF after ours is removed   {r:+.4f}  (p={p:.4f})")
    r, p = partial(j["proj_mean"].to_numpy(), j["actual"].to_numpy(),
                   j["dff_proj"].to_numpy())
    print(f"  ours after DFF is removed   {r:+.4f}  (p={p:.4f})")
    print("  both positive means two genuinely different opinions, and the")
    print("  average should beat either. Only theirs surviving means the")
    print("  model is strictly behind.")

    if contest and dff.has_ownership(theirs):
        path = next(iter(sorted(
            (DATA / "contests").glob(f"contest-standings-{contest}.*"))), None)
        if path is None:
            print(f"\nno export for contest {contest}", file=sys.stderr)
            return 0
        _, players = contests.load(path)
        real = contests.ownership(players)
        o = j.merge(real, left_on="player_name_dk", right_on="player_name",
                    how="inner")
        o = o.dropna(subset=["dff_own", "pct_drafted"])
        if len(o) >= 10:
            r = stats.spearmanr(o["dff_own"], o["pct_drafted"])
            err = (o["dff_own"] / 100.0 - o["pct_drafted"]).abs().mean()
            print(f"\n=== their projected ownership vs what the field did "
                  f"({len(o)} players) ===")
            print(f"  spearman {r.statistic:+.4f}  (p={r.pvalue:.2g}), "
                  f"mean absolute error {100 * err:.1f} points of ownership")
            print("  this is the only input measured to carry signal the model")
            print("  lacks, and the only one that cannot be derived here.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

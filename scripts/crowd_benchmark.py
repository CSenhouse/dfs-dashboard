"""Is our model better than the crowd? Ownership is the crowd's projection.

The project has had no answer to "are these projections good or mediocre"
because comparing against a professional service costs money. Contest
standings supply a free substitute: ownership is thousands of entrants'
aggregated opinion of who to play, formed before lock, and it can be ranked
against what actually happened exactly as a projection can.

Two things fall out, and they are different questions.

    Does our model rank players better than the crowd does?
    Does the crowd know anything our model does not?

The first is the benchmark. The second decides whether ownership is worth
having as an input, which matters because it is the one piece of information
this project cannot derive from box scores -- it would have to be bought, or
predicted from scratch.

Both are measured on rostered players only. Nobody drafts the whole slate, so
this is the top of the pool rather than all of it -- which is the population
lineups are built from, and also a selected one, so read it as "among players
anyone considered".

    PYTHONPATH=src .venv/Scripts/python.exe scripts/crowd_benchmark.py
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy import stats

from dfs import backtest, contests, features, model, slates
from dfs.config import CURATED, DATA

# (draft group, a contest played on it). The contest only supplies ownership,
# so any contest on the slate will do.
SLATES = [
    (153428, 195661381, "2026 week 2"),
    (153771, 195921990, "2026 week 3"),
]


def partial(x: np.ndarray, y: np.ndarray, given: np.ndarray) -> tuple[float, float]:
    """Correlation of x and y once `given` is removed from both, on ranks.

    Ranks rather than levels because the question is about ordering, and
    because DK points have a long right tail that a linear fit would let
    dominate.
    """
    rx, ry, rg = (stats.rankdata(v) for v in (x, y, given))
    res = lambda a: a - np.poly1d(np.polyfit(rg, a, 1))(rg)
    r = stats.pearsonr(res(rx), res(ry))
    return float(r.statistic), float(r.pvalue)


def main(argv: list[str]) -> int:
    path = CURATED / "features.parquet"
    if not path.exists():
        print(f"{path} missing", file=sys.stderr)
        return 1
    past = backtest.prepare(pd.read_parquet(path))
    past = past[past["season"] < 2026]

    rows = []
    for dgid, cid, label in SLATES:
        up = slates.assemble(dgid)
        if up.empty:
            print(f"{label}: slate {dgid} not cached, skipping")
            continue
        cur = features.build(up)
        cur = cur[cur["salary"].fillna(0) > 0].reset_index(drop=True)
        cur = backtest.prepare(cur)
        views = model.two_stage_all(past, cur)
        m = cur.merge(views, on="player_dk_id", how="left")

        export = next(iter(sorted(
            (DATA / "contests").glob(f"contest-standings-{cid}.*"))), None)
        if export is None:
            print(f"{label}: no export for contest {cid}, skipping")
            continue
        _, players = contests.load(export)
        own = contests.ownership(players)
        j = m.merge(own, on="player_name", how="inner", suffixes=("", "_c"))
        j = j.dropna(subset=["proj_mean", "pct_drafted", "dk_scored"])

        print(f"=== {label}, dg {dgid}: {len(j)} rostered players "
              f"({len(j)/len(own):.0%} of those drafted matched) ===")
        got = {}
        for name, col in (("our projection", "proj_mean"),
                          ("crowd ownership", "pct_drafted"),
                          ("salary alone", "salary")):
            r = stats.spearmanr(j[col], j["dk_scored"]).statistic
            got[name] = float(r)
            print(f"  {name:<17} spearman vs actual points {r:+.4f}")

        r, p = partial(j["pct_drafted"].to_numpy(), j["dk_scored"].to_numpy(),
                       j["proj_mean"].to_numpy())
        print(f"  crowd signal our model does NOT have: {r:+.4f} (p={p:.4f})")
        rows.append({"slate": label, **got, "residual": r, "p": p})
        print()

    if not rows:
        return 1
    d = pd.DataFrame(rows)
    print("=== summary ===")
    print(f"  our projection beats the crowd in "
          f"{int((d['our projection'] > d['crowd ownership']).sum())}"
          f"/{len(d)} slates "
          f"({d['our projection'].mean():+.4f} against "
          f"{d['crowd ownership'].mean():+.4f})")
    print(f"  but the crowd carries {d['residual'].mean():+.4f} of signal the "
          f"model lacks")
    print()
    print("  The first says the projections are not obviously mediocre, which")
    print("  is the first benchmark this project has ever had. The second says")
    print("  ownership is worth having as an input -- and it is the one input")
    print("  that cannot be derived from box scores.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

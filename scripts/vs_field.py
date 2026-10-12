"""Where would our lineup have finished in a field that was actually counted?

Every cash-rate figure this project has produced came from a simulated field,
and the first real export showed that field was wrong in both directions at
once -- 16.8 points too low and 1.26x too narrow. A standings export removes
the model from the loop entirely: it lists every lineup that was submitted and
what it scored, so our lineup can be dropped into the real distribution and
ranked.

    PYTHONPATH=src .venv/Scripts/python.exe scripts/vs_field.py 154084 \\
        196281323 196281475

Scoring uses nflverse box scores through `slates.assemble`, not the export's
own FPTS column, so the lineup is scored by the same engine the backtest uses.
The two are cross-checked anyway and must agree within a tenth of a point --
DK's numbers are the authority on DK's own contest, and a mismatch means the
scoring engine has drifted.

Rule 1 says never join on player names, and this script does, because a
standings export carries no DK player ids. It is the one place where there is
no alternative, so the join is exact-match only and the unmatched are printed
rather than dropped quietly. A single-game pool of about fifty players from two
named teams is the narrowest case a name join ever faces here; on a main slate
of 800 it would not be defensible.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy import stats

from dfs import backtest, contests, features, model, optimizer, slates
from dfs.config import CURATED, DATA

CAPTAIN = 1.5
VIEWS = {"cash": "proj_floor", "gpp": "proj_ceiling", "neutral": "proj_mean"}


def project(dgid: int, showdown: bool) -> pd.DataFrame:
    """Floor/mean/ceiling for a slate, trained only on weeks before it."""
    up = slates.assemble(dgid)
    if up.empty:
        raise SystemExit(f"draft group {dgid} not cached")
    season, week = int(up["season"].iloc[0]), int(up["week"].iloc[0])

    name = "showdown_features" if showdown else "features"
    past = backtest.prepare(pd.read_parquet(CURATED / f"{name}.parquet"))
    past = past[(past["season"] < season)
                | ((past["season"] == season) & (past["week"] < week))]
    if past.empty:
        raise SystemExit("no training rows before this week")

    cur = features.build(up, salary_within_slate=showdown)
    cur = cur[cur["salary"].fillna(0) > 0].reset_index(drop=True)
    views = model.two_stage_all(past, cur)
    return cur.merge(views, on="player_dk_id", how="left")


def score(lineup: pd.DataFrame) -> float:
    """What the lineup really scored. A missing box-score row is a zero here:
    the player was rostered and did not produce (rule 5)."""
    pts = lineup["dk_actual"].fillna(0.0).to_numpy(dtype=float)
    mult = np.where(lineup["slot"].to_numpy() == "CPT", CAPTAIN, 1.0)
    return float((pts * mult).sum())


def place(points: float, field: np.ndarray, paid: int | None) -> str:
    beat = int((field < points).sum())
    rank = int((field > points).sum()) + 1
    out = (f"rank {rank:,} of {field.size:,} "
           f"(beat {beat / field.size:.1%} of the field)")
    if paid:
        line = float(np.sort(field)[::-1][min(paid, field.size) - 1])
        out += f"  line {line:.2f} -> {'CASHED' if points >= line else 'missed'}"
        if points < line:
            out += f" by {line - points:.2f}"
    return out


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    dgid = int(argv[0])
    cids = [a for a in argv[1:] if not a.startswith("-")]

    up = slates.assemble(dgid)
    showdown = up["team"].nunique() == 2 and len(up) < 120
    fmt = "showdown" if showdown else "classic"
    print(f"draft group {dgid}: {fmt}, {len(up)} priced, "
          f"{int(up['season'].iloc[0])} week {int(up['week'].iloc[0])}, "
          f"{', '.join(sorted(up['team'].dropna().unique()))}\n")

    cur = project(dgid, showdown)
    pool = cur[cur["proj_mean"].notna()].copy()

    hindsight = None
    pool["perfect"] = pool["dk_actual"].fillna(0.0)
    if showdown:
        best = optimizer.optimize_showdown(pool, objective="perfect")
        hindsight = score(best[0]) if best else None

    fields = {}
    for cid in cids:
        path = next((p for p in (DATA / "contests").glob(
            f"contest-standings-{cid}.*")), None)
        if path is None:
            print(f"  no export for contest {cid}", file=sys.stderr)
            continue
        entries, players = contests.load(path)
        fields[cid] = (entries, players)

    paid = {}
    f = pd.read_csv(DATA / "contest_fields.csv")
    f["contest_id"] = f["contest_id"].astype(str)
    for cid in fields:
        row = f[f["contest_id"] == cid]
        if len(row):
            paid[cid] = int(row["paid_positions"].iloc[0])

    for label, col in VIEWS.items():
        built = (optimizer.optimize_showdown(pool, objective=col) if showdown
                 else optimizer.optimize(pool, objective=col))
        if not built:
            print(f"{label}: no lineup")
            continue
        lu = built[0]
        pts = score(lu)
        print(f"=== {label} build (optimizes {col}) -> scored {pts:.2f} ===")
        for r in lu.sort_values("slot").itertuples():
            print(f"    {r.slot:<5} {r.player_name:<22} {r.team:<4} "
                  f"${r.lineup_salary:>6,.0f}  proj {r.lineup_points:>5.2f}  "
                  f"actual {0.0 if pd.isna(r.dk_actual) else r.dk_actual:>5.2f}")
        for cid, (entries, _) in fields.items():
            field = entries["points"].to_numpy(dtype=float)
            print(f"    {cid}: {place(pts, field, paid.get(cid))}")
        print()

    if hindsight is not None:
        print(f"best possible lineup on this slate, in hindsight: "
              f"{hindsight:.2f}\n")

    # The crowd benchmark, on the players the field actually rostered.
    for cid, (entries, players) in fields.items():
        own = contests.ownership(players)
        j = cur.merge(own, on="player_name", how="inner", suffixes=("", "_c"))
        missing = sorted(set(own["player_name"]) - set(cur["player_name"]))
        j = j.dropna(subset=["proj_mean"])
        j["actual"] = j["dk_actual"].fillna(0.0)
        print(f"=== {cid} crowd benchmark: {len(j)} of {len(own)} rostered "
              f"players matched ===")
        if missing:
            print(f"    unmatched: {', '.join(missing)}")
        if len(j) < 5:
            print("    too few to rank")
            continue
        for name, col in (("our projection", "proj_mean"),
                          ("crowd ownership", "pct_drafted"),
                          ("salary alone", "salary")):
            r = stats.spearmanr(j[col], j["actual"])
            print(f"    {name:<17} spearman vs actual {r.statistic:+.4f} "
                  f"(p={r.pvalue:.3f})")
        # Cross-check the scoring engine against DK's own numbers.
        d = (j["actual"] - j["dk_scored"].fillna(0.0)).abs()
        print(f"    scoring agrees with DK's export to {d.max():.2f} points "
              f"at worst")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

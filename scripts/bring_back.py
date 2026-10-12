"""Do bring-back lineups actually finish higher? Counted entries, no simulator.

Section 13 measured stacking inside the simulation and was explicit about the
limit: a simulator fed a correlation table will report whatever that table
implies, so it cannot confirm the idea it was built from. The outcome test
available at the time -- our own lineup, 86 walk-forward weeks -- had no power
either, needing something like 3,700 seasons to separate the tail rates that
matter.

Contest standings sidestep both problems. Seven exports hold about 44,000
lineups that real people submitted, each with the score it actually posted.
Every one is an independent opinion about how to build, already graded. No
projection, no correlation table and no simulation enters this file.

Each entry is classified by what its QB is connected to:

    none          no pass catcher from the QB's own team
    stack         >=1 WR/TE from the QB's own team
    bring-back    a skill player from the OPPOSING team, but no team-mate
    both          a team-mate AND an opponent -- the full game stack

Scores are standardised within each contest before pooling, because a week 3
double-up and a week 2 tournament are different scales and different fields.
One z unit is one contest standard deviation.

Two caveats that cannot be designed away. Entrants who stack are not a random
sample of entrants -- they are more likely to be the ones who read about
stacking, and that is confounded with everything else they know. And a
tournament pays the tail, so the mean is the wrong summary; the top-1% rate
is the number to read, which is why it is computed per contest and pooled.

Rule 1 says never join on player names, and this joins on player names,
because a standings export carries no DK ids. It is exact-match only and the
rate is printed per contest; anything below about 95% should not be read.

    PYTHONPATH=src .venv/Scripts/python.exe scripts/bring_back.py
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy import stats

from dfs import contests
from dfs.sources import dk_api
from dfs.config import DATA

# Every export we hold, and the slate it was played on.
CONTESTS = {
    "195661381": 153428, "195661399": 153428,
    "195921990": 153771, "195921991": 153771, "195922009": 153771,
    "195922010": 153771, "195922011": 153771,
    # Week 4. Worth keeping separable: this was a chalk-bust week, the one
    # where the double-up premium went negative (docs/MODELING.md 27), so it
    # is a real stress test rather than more of the same.
    "196169740": 154078, "196169741": 154078, "196169758": 154078,
    "196169760": 154078, "196169761": 154078,
}
WEEK = {153428: 2, 153771: 3, 154078: 4}
# Paid fraction per contest, read off what the contest actually paid.
_f = pd.read_csv(DATA / "contest_fields.csv")
PAID = dict(zip(_f["contest_id"].astype(str),
                _f["paid_positions"] / _f["entries"]))

CATCHERS = {"WR", "TE"}
SKILL = {"WR", "TE", "RB"}


def slate_lookup(dgid: int) -> dict[str, tuple[str, str, str]]:
    """name -> (position, team, opponent) for one draft group.

    The opponent comes from `competition_id`: the two teams sharing one are
    playing each other. `slates.assemble` does not carry it, and deriving it
    from the schedule would add a join this does not need.
    """
    d = dk_api.draftables(dgid)
    if d.empty:
        return {}
    d = d.dropna(subset=["player_name", "position", "team", "competition_id"])
    opp = {}
    for comp, g in d.groupby("competition_id"):
        teams = sorted(set(g["team"].astype(str)))
        if len(teams) == 2:
            opp[teams[0]], opp[teams[1]] = teams[1], teams[0]
    return {str(n).strip(): (str(p), str(tm), opp.get(str(tm), ""))
            for n, p, tm in zip(d["player_name"], d["position"], d["team"])
            if str(tm) in opp}


def classify(names: list[str], look: dict) -> tuple[str, int, int]:
    """(class, matched, total) for one entry's nine players."""
    hit = [look[n] for n in names if n in look]
    qb = next((r for r in hit if r[0] == "QB"), None)
    if qb is None:
        return "no qb", len(hit), len(names)
    _, team, opp = qb
    mates = sum(1 for p, t, _ in hit if t == team and p in CATCHERS)
    backs = sum(1 for p, t, _ in hit if t == opp and p in SKILL)
    if mates and backs:
        return "both", len(hit), len(names)
    if mates:
        return "stack", len(hit), len(names)
    if backs:
        return "bring-back", len(hit), len(names)
    return "none", len(hit), len(names)


def main(argv: list[str]) -> int:
    looks = {d: slate_lookup(d) for d in set(CONTESTS.values())}
    if not any(looks.values()):
        print("no slates cached", file=sys.stderr)
        return 1

    rows = []
    for cid, dgid in CONTESTS.items():
        path = next(iter(sorted(
            (DATA / "contests").glob(f"contest-standings-{cid}.*"))), None)
        if path is None:
            continue
        entries, _ = contests.load(path)
        look = looks[dgid]
        pts = entries["points"].to_numpy(dtype=float)
        kinds, matched, total = [], 0, 0
        for text in entries["lineup"]:
            names = [n for _, n in contests.parse_lineup(text)]
            k, m, t = classify(names, look)
            kinds.append(k)
            matched += m
            total += t
        d = pd.DataFrame({"contest": cid, "week": WEEK.get(dgid, 0),
                          "kind": kinds, "points": pts})
        d["z"] = (pts - pts.mean()) / pts.std()
        d["top1"] = pts >= np.quantile(pts, 0.99)
        d["top10"] = pts >= np.quantile(pts, 0.90)
        # The contest's OWN pay line, not a fixed decile: a double-up pays
        # 43.5% and a tournament 25%, so "did it cash" is a different
        # question in each and the decile answers neither.
        d["cash"] = pts >= np.quantile(pts, 1.0 - PAID.get(cid, 0.25))
        print(f"{cid}: {len(d):,} entries, names matched {matched/total:.1%}")
        if matched / total < 0.95:
            print("   match rate too low, skipping this contest")
            continue
        rows.append(d)

    if not rows:
        return 1
    d = pd.concat(rows, ignore_index=True)
    d = d[d["kind"] != "no qb"]

    print(f"\n=== {len(d):,} counted entries across {d['contest'].nunique()} "
          f"contests ===")
    print(f"{'construction':<12} {'entries':>8} {'share':>6} {'mean z':>8} "
          f"{'top 1%':>8} {'top 10%':>8}")
    base = d[d["kind"] == "none"]
    for kind in ("none", "stack", "bring-back", "both"):
        g = d[d["kind"] == kind]
        if g.empty:
            continue
        print(f"{kind:<12} {len(g):>8,} {len(g)/len(d):>5.1%} "
              f"{g['z'].mean():>+8.4f} {g['top1'].mean():>7.2%} "
              f"{g['top10'].mean():>7.2%}")

    print("\n=== against the unstacked entries in the same contests ===")
    for kind in ("stack", "bring-back", "both"):
        g = d[d["kind"] == kind]
        if g.empty or base.empty:
            continue
        t = stats.ttest_ind(g["z"], base["z"], equal_var=False)
        # Tail rate is the number that matters for a tournament.
        tab = [[int(g["top1"].sum()), len(g) - int(g["top1"].sum())],
               [int(base["top1"].sum()), len(base) - int(base["top1"].sum())]]
        odds, pf = stats.fisher_exact(tab)
        print(f"  {kind:<11} mean {g['z'].mean() - base['z'].mean():+.4f} z "
              f"(p={t.pvalue:.4f})   top-1% rate {odds:.2f}x "
              f"(p={pf:.4f})")

    # The cleanest comparison in the file. "stack" and "both" are both groups
    # of people who deliberately stacked, so the usual objection -- that
    # stackers are simply better players -- applies about equally to each, and
    # what separates them is the bring-back itself.
    print("\n=== the controlled comparison: bring-back AMONG stackers ===")
    s, b = d[d["kind"] == "stack"], d[d["kind"] == "both"]
    tt = stats.ttest_ind(b["z"], s["z"], equal_var=False)
    tab = [[int(b["top1"].sum()), len(b) - int(b["top1"].sum())],
           [int(s["top1"].sum()), len(s) - int(s["top1"].sum())]]
    odds, pf = stats.fisher_exact(tab)
    print(f"  both vs stack   mean {b['z'].mean() - s['z'].mean():+.4f} z "
          f"(p={tt.pvalue:.4f})   top-1% rate {odds:.2f}x (p={pf:.4f})")
    print("  Both groups stacked. What separates them is the opponent.")

    # If the bring-back works by buying tail, it should help where the tail is
    # paid and not where it is not. A double-up pays a flat amount for
    # clearing a bar, so there it should be useless or worse.
    print("\n=== where the tail is paid, and where it is not ===")
    d["structure"] = ["double-up" if PAID.get(c, 0) > 0.35 else "tournament"
                      for c in d["contest"]]
    for struct, g in d.groupby("structure"):
        gs, gb = g[g["kind"] == "stack"], g[g["kind"] == "both"]
        if gs.empty or gb.empty:
            continue
        tab = [[int(gb["top1"].sum()), len(gb) - int(gb["top1"].sum())],
               [int(gs["top1"].sum()), len(gs) - int(gs["top1"].sum())]]
        odds, pf = stats.fisher_exact(tab)
        print(f"  {struct:<11} n={len(g):>6,}  both vs stack: "
              f"mean {gb['z'].mean() - gs['z'].mean():+.4f} z, "
              f"top-1% {odds:.2f}x (p={pf:.4f}), "
              f"cash rate {gb['cash'].mean():.2%} vs {gs['cash'].mean():.2%}")

    # Week 4 is the stress test: it was the chalk-bust week, the one where the
    # double-up premium went negative (docs/MODELING.md 27). If the bring-back
    # only works when the chalk works, it should fail here.
    print("\n=== does it replicate week to week? (tournaments only) ===")
    for wk, g in d[d["structure"] == "tournament"].groupby("week"):
        gs, gb = g[g["kind"] == "stack"], g[g["kind"] == "both"]
        if gs.empty or gb.empty:
            continue
        tab = [[int(gb["top1"].sum()), len(gb) - int(gb["top1"].sum())],
               [int(gs["top1"].sum()), len(gs) - int(gs["top1"].sum())]]
        odds, pf = stats.fisher_exact(tab)
        print(f"  week {wk}  n={len(g):>6,}  both vs stack: "
              f"top-1% {odds:.2f}x (p={pf:.4f}), "
              f"mean {gb['z'].mean() - gs['z'].mean():+.4f} z")

    print("\n  Stacking is not randomly assigned. An entrant who stacks is an")
    print("  entrant who has read about stacking, and that travels with")
    print("  everything else they know about the slate. Read these as what")
    print("  the people who build each way actually scored, not as the effect")
    print("  of the construction on a fixed lineup.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

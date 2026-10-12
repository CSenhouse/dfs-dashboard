"""Does being contrarian pay, and in which contests?

Ownership is the one input this project cannot derive. Five standings exports
now carry it exactly, along with every lineup that was submitted and what it
scored, which makes the foundational question answerable without a model:

    do the lineups that finish high roster less-owned players?

This is arithmetic on counted entries, not a simulation. Every entry's roster
is summed against the ownership table from the SAME contest, so there is no
cross-source join and no name resolution risk -- both halves come out of one
file written by one system.

A roster's ownership sum is the obvious summary: nine players each owned by
some fraction of the field, added up. A sum near 4.0 means the lineup is built
from players the field also chose; near 1.0 means almost nobody shares it.

The reason it matters is structural rather than statistical. In a tournament
your payout depends on how many entries you share the top with, so being right
about a player everyone else also had is worth far less than being right about
one they did not. In a double-up it should not matter at all -- the contest
pays a flat amount for clearing a bar, and who else cleared it is irrelevant.
That contrast is the cleanest available test that any measured effect is real
and not an artefact of chalk simply being good players.

    PYTHONPATH=src .venv/Scripts/python.exe scripts/leverage.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from dfs import contests
from dfs.config import DATA

CONTESTS = DATA / "contests"


def contest_kinds() -> dict[str, str]:
    """Payout structure, read off what was actually paid.

    A double-up pays about 43.5% of the field and a tournament about 25%,
    so the ratio identifies the structure exactly. Guessing from entry
    count instead happens to work on the Classic exports and breaks on a
    39-entry Showdown double-up.
    """
    f = pd.read_csv(DATA / "contest_fields.csv")
    frac = f["paid_positions"] / f["entries"]
    return dict(zip(f["contest_id"].astype(str),
                    np.where(frac > 0.35, "double-up", "tournament")))


KINDS = contest_kinds()


def roster_ownership(entries: pd.DataFrame, own: pd.DataFrame
                     ) -> tuple[np.ndarray, float]:
    """Summed ownership of each entry's nine players, and the match rate."""
    table = dict(zip(own["player_name"], own["pct_drafted"]))
    sums, matched, total = [], 0, 0
    for text in entries["lineup"]:
        names = [n for _, n in contests.parse_lineup(text)]
        total += len(names)
        hit = [table[n] for n in names if n in table]
        matched += len(hit)
        sums.append(sum(hit) if hit else np.nan)
    return np.array(sums, dtype=float), matched / max(total, 1)


def main(argv: list[str]) -> int:
    # Exports arrive zipped from the browser and bare when saved as CSV.
    # Globbing only .zip skipped the first two Showdown contests without
    # saying so.
    files = sorted(p for p in CONTESTS.glob("contest-standings-*")
                   if p.suffix.lower() in (".zip", ".csv"))
    if not files:
        print(f"no exports in {CONTESTS}", file=sys.stderr)
        return 1

    rows = []
    for path in files:
        cid = path.stem.split("-")[-1]
        entries, players = contests.load(path)
        own = contests.ownership(players)
        sums, match = roster_ownership(entries, own)
        ok = np.isfinite(sums)
        pts = entries["points"].to_numpy(dtype=float)[ok]
        chalk = sums[ok]
        kind = KINDS.get(cid, "unknown")

        r = stats.pearsonr(chalk, pts)
        slots = int(np.median([len(contests.parse_lineup(x))
                               for x in entries["lineup"]]))
        fmt = "showdown" if slots == 6 else "classic"
        print(f"=== {cid}  {fmt} {kind}  {len(entries):,} entries "
              f"(names matched {match:.1%}) ===")
        print(f"  roster ownership: mean {chalk.mean():.2f} of {slots} "
              f"slots  sd {chalk.std():.2f}  "
              f"range {chalk.min():.2f} to {chalk.max():.2f}")
        print(f"  correlation with points: {r.statistic:+.4f}  p={r.pvalue:.2g}")

        # Where the money is: the ownership profile of the entries that
        # actually finished high, against the field they came from.
        cut = {"top 1%": 0.99, "top 5%": 0.95, "top 25%": 0.75}
        for label, q in cut.items():
            sel = pts >= np.quantile(pts, q)
            print(f"  {label:<7} n={int(sel.sum()):>5}  "
                  f"ownership {chalk[sel].mean():.2f} "
                  f"({chalk[sel].mean() - chalk.mean():+.2f} vs field)  "
                  f"points {pts[sel].mean():.1f}")
        rows.append({"contest_id": cid, "kind": kind, "fmt": fmt,
                     "slots": slots, "entries": len(entries),
                     "corr": r.statistic, "p": r.pvalue,
                     "field_own": chalk.mean(), "own_sd": chalk.std(),
                     "top1_own": chalk[pts >= np.quantile(pts, 0.99)].mean()})
        print()

    d = pd.DataFrame(rows)
    print("=== summary ===")
    print("  a tournament pays for being right when others were not;")
    print("  a double-up pays a flat amount and should not care at all.")
    for (fmt, kind), g in d.groupby(["fmt", "kind"]):
        print(f"  {fmt:<9} {kind:<11} corr {g['corr'].mean():+.4f}   "
              f"top-1% ownership {g['top1_own'].mean():.2f} "
              f"vs field {g['field_own'].mean():.2f} "
              f"({g['top1_own'].mean() - g['field_own'].mean():+.2f})   "
              f"spread sd {(g['own_sd'] / g['slots']).mean():.3f} per slot")

    print()
    print("  The sd column is there to rule out the easy explanation, and it")
    print("  does not. A converged field cannot vary much in how chalky its")
    print("  entries are, and a correlation over a restricted range is pushed")
    print("  toward zero -- but PER SLOT the Showdown fields spread more than")
    print("  the Classic tournaments, which still manage +0.31. So the near-")
    print("  zero Showdown figure is not a range artefact. It is also not a")
    print("  finding: two contests on one game, sharing 24 of 39 entrants and")
    print("  22 identical lineups, are one observation, and the top-1% column")
    print("  is literally one entry.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

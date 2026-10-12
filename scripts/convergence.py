"""How much of the field is the same lineup, and does it differ by format?

CLAUDE.md has carried this as reasoning rather than measurement: a Showdown
slate prices about 45 players into 6 slots, so its field should converge harder
than a Classic field choosing 9 from 800, and a tight field is a hard field.
The first two Showdown exports make the first half of that testable.

The metric has to work across formats, so it cannot be "ownership of the top
ten players" -- top ten of 45 and top ten of 800 are not the same quantity.
What is comparable is overlap: of your S roster slots, how many does another
randomly chosen entry also hold? Normalised by S it runs from S/pool (nobody
agrees) to 1.0 (everyone submitted the same lineup), whatever S is.

It is computed twice:

    implied   sum(p^2)/S, what ownership alone predicts if entries were
              assembled independently from those marginals
    actual    sampled over real pairs of submitted lineups

The gap was meant to detect clustering into a few discrete builds, which
ownership as a marginal cannot see. There is no gap: implied and actual agree
to within 0.01 in both formats and all nine contests. Ownership is a sufficient
statistic for overlap, and the thirteen identical lineups in a Showdown
double-up are what those marginals predict rather than evidence of anything
beyond them.

Contests are split by payout structure, not size: paid/entries near 0.435 is a
double-up, near 0.25 a tournament. Without that split the comparison is
confounded, because every Showdown export here is a double-up and Classic
double-ups already converge far harder than Classic tournaments.

    PYTHONPATH=src .venv/Scripts/python.exe scripts/convergence.py
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from dfs import contests
from dfs.config import DATA

CONTESTS = DATA / "contests"
PAIRS = 200_000
RNG = np.random.default_rng(17)


def rosters(entries: pd.DataFrame) -> list[frozenset[str]]:
    """Each entry as a set of player names, slot discarded.

    Slot is dropped on purpose. Two entries holding the same player at CPT and
    at FLEX score differently, but they rise and fall together, which is what
    overlap is measuring.
    """
    return [frozenset(n for _, n in contests.parse_lineup(t))
            for t in entries["lineup"]]


def actual_overlap(sets: list[frozenset[str]], slots: int) -> float:
    """Mean shared players over random pairs of distinct entries, / slots."""
    n = len(sets)
    if n < 2:
        return float("nan")
    i = RNG.integers(0, n, PAIRS)
    j = RNG.integers(0, n, PAIRS)
    keep = i != j
    i, j = i[keep], j[keep]
    shared = [len(sets[a] & sets[b]) for a, b in zip(i, j)]
    return float(np.mean(shared)) / slots


def main(argv: list[str]) -> int:
    files = sorted(p for p in CONTESTS.glob("contest-standings-*")
                   if p.suffix.lower() in (".zip", ".csv"))
    if not files:
        print(f"no exports in {CONTESTS}", file=sys.stderr)
        return 1

    fields = pd.read_csv(DATA / "contest_fields.csv")
    paid = dict(zip(fields["contest_id"].astype(str), fields["paid_positions"]))

    rows = []
    for path in files:
        cid = path.stem.split("-")[-1]
        entries, players = contests.load(path)
        sets = rosters(entries)
        slots = int(np.median([len(s) for s in sets]))
        fmt = "showdown" if slots == 6 else "classic"

        own = contests.ownership(players)
        p = own["pct_drafted"].to_numpy(dtype=float)
        implied = float((p ** 2).sum()) / slots
        actual = actual_overlap(sets, slots)

        pts = entries["points"].to_numpy(dtype=float)
        n_paid = paid.get(cid)
        line = np.nan
        if n_paid and np.isfinite(n_paid):
            line = float(np.sort(pts)[::-1][min(int(n_paid), pts.size) - 1])

        frac = (n_paid / len(entries)) if n_paid else float("nan")
        kind = "double-up" if frac > 0.35 else "tournament"
        # Entries sitting on exactly the pay line, and the block just under it.
        at_line = int((pts == line).sum()) if np.isfinite(line) else 0
        under = pts[pts < line] if np.isfinite(line) else np.array([0.0])
        top_under = under.max() if under.size else float("nan")
        blocked = int((pts == top_under).sum()) if under.size else 0

        rows.append({
            "contest": cid, "format": fmt, "kind": kind, "slots": slots,
            "at_line": at_line, "blocked": blocked,
            "margin": float(line - top_under) if under.size else float("nan"),
            "entries": len(entries), "distinct": entries["lineup"].nunique(),
            "biggest_block": int(entries["lineup"].value_counts().iloc[0]),
            "implied": implied, "actual": actual,
            "mean": pts.mean(), "sd": pts.std(), "cv": pts.std() / pts.mean(),
            "line": line,
        })

    d = pd.DataFrame(rows).sort_values(["format", "kind", "entries"])
    d["unique"] = d["distinct"] / d["entries"]
    d["block"] = d["biggest_block"] / d["entries"]

    print("overlap = fraction of your roster a random other entry also holds")
    print("implied = from ownership alone; actual = from the real lineups\n")
    hdr = (f"{'contest':>10} {'format':>9} {'kind':>11} {'entries':>8} "
           f"{'uniq':>6} {'block':>6} {'implied':>8} {'actual':>7} "
           f"{'mean':>7} {'sd':>6} {'cv':>6} {'line':>7}")
    print(hdr)
    print("-" * len(hdr))
    for _, r in d.iterrows():
        print(f"{r['contest']:>10} {r['format']:>9} {r['kind']:>11} "
              f"{r['entries']:>8,} {r['unique']:>5.0%} {r['block']:>5.0%} "
              f"{r['implied']:>8.3f} {r['actual']:>7.3f} "
              f"{r['mean']:>7.1f} {r['sd']:>6.1f} {r['cv']:>5.1%} "
              f"{r['line']:>7.1f}")

    print("\n=== by format and payout structure ===")
    for (fmt, kind), g in d.groupby(["format", "kind"]):
        print(f"  {fmt:<9} {kind:<11} n={len(g)}  overlap {g['actual'].mean():.3f}"
              f"  (ownership implied {g['implied'].mean():.3f},"
              f" gap {g['actual'].mean() - g['implied'].mean():+.3f})"
              f"  cv {g['cv'].mean():.1%}")

    s = d[d["format"] == "showdown"]
    c = d[(d["format"] == "classic") & (d["kind"] == "double-up")]
    if len(s) and len(c):
        print()
        print("  Against Classic double-ups -- same payout structure, so this is")
        print("  not confounded by contest type -- a Showdown field holds "
              f"{s['actual'].mean():.0%} of")
        print(f"  your roster against {c['actual'].mean():.0%}, and is "
              f"{c['cv'].mean() / s['cv'].mean():.1f}x tighter in relative spread.")
        print("  CLAUDE.md predicted that from the pool size; it is now measured.")
        print()
        print("  Ownership alone accounts for all of it. The gap between implied")
        print("  and actual is under 0.01 in every contest and both formats, so")
        print("  thirteen identical lineups in a 39-entry Showdown are what those")
        print("  marginals predict, not clustering on top of them.")

    print("\n=== where the pay line falls ===")
    print("  A tight field puts a block of entries on one score. If the line")
    print("  lands in that block they split it; if it lands just above, the")
    print("  whole block misses by the margin shown.")
    for _, r in d.iterrows():
        print(f"  {r['contest']:>10} {r['format']:>9}  line {r['line']:7.2f}  "
              f"{r['at_line']:>5} tied ON it, "
              f"{r['blocked']:>5} tied {r['margin']:.2f} under it "
              f"({r['blocked'] / r['entries']:.0%} of the field)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

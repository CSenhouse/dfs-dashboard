"""Turn a DraftKings standings export into ground truth the project keeps.

Export a completed contest from its gamecenter while logged in, drop the file
in data/contests/, and run this. It writes two things that are committed and
one that is not:

    data/contest_fields.csv        the field's real score distribution
    data/ownership/<contest>.csv   what share of entries rostered each player
    data/contests/<file>           the raw export, gitignored -- it carries
                                   other entrants' usernames

The first two are the point. Until the first export arrived, the field in this
project was a model with nothing to check it against, and it was wrong in both
directions at once: too low by 16.8 points and 1.26x too narrow against the
first contest measured. Every cash probability ever printed here rested on it.

Export promptly. Both week 2 contests exported cleanly six days after they
ran; both week 1 contests returned an empty file at thirteen days, including
one with the same entry count as a week 2 export that worked. The evidence
points at DraftKings dropping full standings after a week or two rather than
at anything to do with size. These files are the most valuable data this
project has and they appear to be perishable.

Ownership is the other half, and it cannot be derived from anything. No amount
of projection accuracy tells you what fraction of the field rostered a player,
and in a tournament that is most of what decides whether being right pays.

    PYTHONPATH=src .venv/Scripts/python.exe scripts/ingest_contest.py \\
        data/contests/contest-standings-195661381.zip --paid 2011
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

from dfs import contests
from dfs.config import DATA

FIELDS = DATA / "contest_fields.csv"
OWNERSHIP = DATA / "ownership"


def contest_id(path: Path) -> str:
    m = re.search(r"(\d{6,})", path.name)
    return m.group(1) if m else path.stem


def main(argv: list[str]) -> int:
    if not argv or argv[0].startswith("-"):
        print(__doc__, file=sys.stderr)
        return 2
    path = Path(argv[0])
    if not path.exists():
        print(f"{path} not found", file=sys.stderr)
        return 1
    paid = int(argv[argv.index("--paid") + 1]) if "--paid" in argv else None

    entries, players = contests.load(path)
    cid = contest_id(path)
    # Roster size identifies the format, and it has to be recorded: a 6-man
    # Showdown field is not a reference for a 9-man Classic lineup, and
    # without this column the dashboard cannot tell them apart.
    slots = int(pd.Series(
        [len(contests.parse_lineup(x)) for x in entries["lineup"]]).median())
    summary = contests.field_summary(entries, paid_positions=paid)
    summary = {"contest_id": cid, "format": "showdown" if slots == 6
               else "classic", "slots": slots, **summary}
    print(f"{cid}: {summary['entries']:,} entries, "
          f"mean {summary['mean']:.1f}, sd {summary['sd']:.1f}")
    if paid:
        print(f"  cash line at rank {paid}: {summary['cash_line']:.1f}")

    row = pd.DataFrame([summary])
    if FIELDS.exists():
        old = pd.read_csv(FIELDS)
        old = old[old["contest_id"].astype(str) != cid]
        row = pd.concat([old, row], ignore_index=True)
    row.to_csv(FIELDS, index=False)

    OWNERSHIP.mkdir(parents=True, exist_ok=True)
    own = contests.ownership(players)
    own.to_csv(OWNERSHIP / f"{cid}.csv", index=False)
    print(f"  {len(own)} players, most owned "
          f"{own.iloc[0]['player_name']} at {own.iloc[0]['pct_drafted']:.1%}")
    print(f"  -> {FIELDS.name}, ownership/{cid}.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

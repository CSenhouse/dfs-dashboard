"""Build the curated player-week table from cached DraftKings slates.

Reads only from the on-disk cache, so it runs offline and can be re-run at any
point during a long sweep. For each (season, week) we take the *main* slate --
the classic slate covering the most games -- because salary differs between the
main slate and the many concurrent mini-slates DK runs.

    python scripts/build_dataset.py --lo 53600 --hi 62900
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from dfs import slates
from dfs.config import CURATED
from dfs.sources import dk_api


INDEX_PATH = CURATED / "slate_index.parquet"


def scan(lo: int, hi: int, rescan: bool = False) -> pd.DataFrame:
    """Catalog every cached NFL classic slate in an id range.

    Results are persisted per draft group id, so extending the range after the
    sweep collects another season only pays for the new ids.
    """
    known = pd.DataFrame()
    if INDEX_PATH.exists() and not rescan:
        known = pd.read_parquet(INDEX_PATH)

    seen = set(known["draft_group_id"]) if not known.empty else set()
    todo = [d for d in range(lo, hi + 1) if d not in seen]
    print(f"scan range {lo}..{hi}: {len(seen & set(range(lo, hi + 1)))} indexed, "
          f"{len(todo)} to read")

    rows = []
    for n, dgid in enumerate(todo, 1):
        info = dk_api.peek(dgid)
        if info is not None:
            rows.append(info)
        if n % 5000 == 0:
            print(f"  read {n}/{len(todo)}, {len(rows)} draft groups", flush=True)

    fresh = pd.DataFrame(rows)
    index = (
        pd.concat([known, fresh], ignore_index=True)
        if not fresh.empty else known
    )
    if not index.empty:
        index = index.drop_duplicates("draft_group_id")
        index.to_parquet(INDEX_PATH, index=False)

    if index.empty:
        return index
    out = index[
        index["is_nfl"] & (index["kind"] == "classic")
        & index["draft_group_id"].between(lo, hi)
    ]
    return out.copy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lo", type=int, required=True)
    ap.add_argument("--hi", type=int, required=True)
    ap.add_argument("--min-games", type=int, default=6,
                    help="ignore mini-slates below this many games")
    args = ap.parse_args()

    cat = scan(args.lo, args.hi)
    if cat.empty:
        print("no cached classic NFL slates in range")
        return 1
    print(f"cached classic slates: {len(cat)}")

    cat = cat[cat["n_games"] >= args.min_games].copy()
    print(f"slates with >= {args.min_games} games: {len(cat)}")

    sw = cat["start_time"].apply(slates.season_week_for)
    cat = cat[sw.notna()].copy()
    cat["season"] = [x[0] for x in sw[sw.notna()]]
    cat["week"] = [x[1] for x in sw[sw.notna()]]

    # One main slate per week: most games, then most players.
    main = (cat.sort_values(["n_games", "n_players"], ascending=False)
               .drop_duplicates(subset=["season", "week"])
               .sort_values(["season", "week"]))
    print(f"main slates identified: {len(main)}")
    print(main.groupby("season").size().to_string())

    # Reject slates whose matchups are not real NFL games. DK runs simulated
    # "Madden Stream" contests with real teams, real players and a real DST
    # that are otherwise indistinguishable from a genuine slate.
    keep = []
    for r in main.itertuples():
        if not slates.week_is_complete(r.season, r.week):
            print(f"  SKIP   dg={r.draft_group_id} {r.season} wk{r.week} "
                  f"-- games not finished; no actual scores to train on",
                  flush=True)
            continue
        dk = dk_api.draftables(r.draft_group_id, cache_only=True)
        if slates.matches_schedule(dk, r.season, r.week):
            keep.append(r)
        else:
            print(f"  REJECT dg={r.draft_group_id} {r.season} wk{r.week} "
                  f"-- matchups absent from the real schedule", flush=True)
    print(f"slates passing validation: {len(keep)}/{len(main)}")

    frames = []
    for r in keep:
        df = slates.assemble(r.draft_group_id, season=r.season, week=r.week)
        if not df.empty:
            frames.append(df)
            print(f"  {r.season} wk{r.week:>2}  dg={r.draft_group_id} "
                  f"players={len(df):>3} games={r.n_games}", flush=True)
    if not frames:
        print("nothing assembled")
        return 1

    out = pd.concat(frames, ignore_index=True)
    path = CURATED / "player_week.parquet"
    out.to_parquet(path, index=False)
    print(f"\nwrote {len(out):,} player-slate rows -> {path}")
    print(f"seasons: {sorted(out['season'].unique())}")
    print(f"with actual score: {out['dk_actual'].notna().sum():,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

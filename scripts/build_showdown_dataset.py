"""Build the Showdown training set.

Showdown is its own pricing regime, not a roster-rules variant of Classic.
The same player is priced against a 45-player pool instead of an 800-player
one, so a Classic-trained model has no idea what a $5,000 Showdown salary
means. It also has no idea what a kicker is, because Classic never prices one.

Two filters matter here:

  partial games   DK runs second-half and fourth-quarter contests on the same
                  game, with the same players and the same salaries. Scoring
                  those with full-game results would be badly wrong. The real
                  slate is the one that opens at kickoff, so we keep the
                  earliest draft group per game.

  fake games      the usual Madden Stream and schedule validation.
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from dfs import slates
from dfs.config import CURATED
from dfs.sources import dk_api


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-players", type=int, default=28)
    args = ap.parse_args()

    idx = pd.read_parquet(CURATED / "slate_index.parquet")
    sd = idx[(idx["kind"] == "showdown") & idx["is_nfl"]].copy()
    sd["start_time"] = pd.to_datetime(sd["start_time"], utc=True,
                                      errors="coerce")
    sd = sd.dropna(subset=["start_time"])
    sd = sd[sd["n_players"] >= args.min_players]
    print(f"NFL showdown draft groups: {len(sd)}")

    sw = sd["start_time"].apply(slates.season_week_for)
    sd = sd[sw.notna()].copy()
    sd["season"] = [x[0] for x in sw[sw.notna()]]
    sd["week"] = [x[1] for x in sw[sw.notna()]]
    print(f"mapped to a scheduled week: {len(sd)}")

    complete = {
        (s, w) for s, w in sd[["season", "week"]].drop_duplicates().itertuples(
            index=False)
        if slates.week_is_complete(int(s), int(w))
    }
    sd = sd[[(s, w) in complete
             for s, w in zip(sd["season"], sd["week"])]]
    print(f"in a finished week: {len(sd)}")

    # One slate per game: the earliest draft group is the full-game contest,
    # the later ones are second-half and fourth-quarter derivatives.
    rows = []
    for r in sd.sort_values("start_time").itertuples():
        df = dk_api.draftables(r.draft_group_id, cache_only=True)
        if df.empty:
            continue
        comp = df["competition_name"].dropna()
        rows.append({
            "draft_group_id": r.draft_group_id,
            "season": int(r.season), "week": int(r.week),
            "game": comp.iloc[0] if len(comp) else "",
            "start_time": r.start_time,
            "n_players": r.n_players,
        })
    cat = pd.DataFrame(rows)
    if cat.empty:
        print("nothing cached")
        return 1
    before = len(cat)
    cat = (cat.sort_values("start_time")
              .drop_duplicates(subset=["season", "week", "game"], keep="first"))
    print(f"one per game: {len(cat)} (dropped {before - len(cat)} "
          f"half/quarter derivatives)")

    frames, rejected = [], 0
    for i, r in enumerate(cat.itertuples(), 1):
        dk = dk_api.draftables(r.draft_group_id, cache_only=True)
        if not slates.matches_schedule(dk, r.season, r.week, min_fraction=1.0):
            rejected += 1
            continue
        df = slates.assemble(r.draft_group_id, season=r.season, week=r.week)
        if df.empty:
            continue
        df["game"] = r.game
        frames.append(df)
        if i % 200 == 0:
            print(f"  {i}/{len(cat)} assembled, {len(frames)} kept",
                  flush=True)

    if not frames:
        print("nothing assembled")
        return 1
    out = pd.concat(frames, ignore_index=True)
    path = CURATED / "showdown_player_slate.parquet"
    out.to_parquet(path, index=False)
    print(f"\nrejected as not real games: {rejected}")
    print(f"wrote {len(out):,} player-slate rows -> {path}")
    print(out.groupby("season").size().to_string())
    print()
    print("positions:", out["position"].value_counts().to_dict())
    print(f"with an actual score: {out['dk_actual'].notna().sum():,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

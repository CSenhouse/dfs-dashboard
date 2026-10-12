"""Build lineups for a live DraftKings slate.

    python scripts/lineups.py --list
    python scripts/lineups.py --slate main --lineups 5
    python scripts/lineups.py --slate main --contest cash

Trains on every historical week available, projects the upcoming slate, and
runs the optimizer. Cash play optimizes a floor (30th percentile) because
double-ups pay for clearing a threshold, not for upside; GPP optimizes a
ceiling and forces a QB stack, because tournaments pay for the right tail.

Salaries and the injury report both move up to kickoff. Re-run this after the
inactives land 90 minutes before the first game -- a projection built on a
Wednesday depth chart is stale by Sunday.
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from dfs import backtest, features, model, optimizer, slates
from dfs.config import CURATED
from dfs.sources import dk_api

CONTESTS = {
    "cash": dict(fn=model.two_stage_floor, stack=0, min_diff=1,
                 label="cash / double-up (floor)"),
    "gpp": dict(fn=model.two_stage_ceiling, stack=1, min_diff=3,
                label="GPP / tournament (ceiling + stack)"),
    # Calibrated for the neutral view: it does not build better lineups, but
    # it makes the projected total an honest number, which is what you want
    # when deciding which contest to enter.
    "mean": dict(fn=model.two_stage_calibrated, stack=0, min_diff=1,
                 label="neutral (calibrated mean projection)"),
}


def list_slates() -> pd.DataFrame:
    live = dk_api.live_draftgroups("NFL")
    rows = []
    for r in live.itertuples():
        df = dk_api.draftables(r.draft_group_id, refresh=True)
        if df.empty:
            continue
        info = dk_api.classify(df)
        if not info["is_nfl"] or info["kind"] == "draft":
            continue
        rows.append({
            "draft_group_id": r.draft_group_id,
            "kind": info["kind"],
            "games": info["n_games"],
            "players": info["n"],
            "start": info["start_time"],
            "suffix": r.suffix,
        })
    return pd.DataFrame(rows).sort_values(["kind", "games"], ascending=[True, False])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="show live slates")
    ap.add_argument("--slate", default="main",
                    help="'main' for the largest classic slate, or a draft group id")
    ap.add_argument("--contest", default="cash", choices=sorted(CONTESTS))
    ap.add_argument("--lineups", type=int, default=1)
    ap.add_argument("--max-exposure", type=float, default=1.0)
    ap.add_argument("--out", default=None, help="write lineups to CSV")
    args = ap.parse_args()

    live = list_slates()
    if live.empty:
        print("no live NFL slates in the lobby")
        return 1
    if args.list:
        print(live.to_string(index=False))
        return 0

    if args.slate == "main":
        classic = live[live["kind"] == "classic"]
        if classic.empty:
            print("no classic slate live")
            return 1
        row = classic.iloc[0]
    else:
        hit = live[live["draft_group_id"] == int(args.slate)]
        if hit.empty:
            print(f"draft group {args.slate} is not live; use --list")
            return 1
        row = hit.iloc[0]

    dgid = int(row["draft_group_id"])
    print(f"slate: dg={dgid}  {row['kind']}  {row['games']} games  "
          f"{row['players']} players  start={row['start']}")

    upcoming = slates.assemble(dgid)
    if upcoming.empty:
        print("could not assemble slate (schedule mapping failed)")
        return 1
    season = int(upcoming["season"].iloc[0])
    week = int(upcoming["week"].iloc[0])
    print(f"resolved to season {season} week {week}")

    hist_path = CURATED / "features.parquet"
    if not hist_path.exists():
        print("no training data; run scripts/build_dataset.py first")
        return 1
    hist = pd.read_parquet(hist_path)

    # Train on everything strictly before this slate.
    past = backtest.prepare(hist)
    past = past[(past["season"] < season) |
                ((past["season"] == season) & (past["week"] < week))]
    if past.empty:
        print("no historical weeks before this slate")
        return 1
    print(f"training rows: {len(past):,} "
          f"({past['season'].min()}-{past['season'].max()})")

    cur = features.build(upcoming)
    cur = cur.dropna(subset=["salary"])
    cur = cur[cur["salary"] > 0].reset_index(drop=True)

    # DraftKings still prices players on injured reserve, usually at the $300
    # minimum, which makes them look like a free roster spot to an optimizer.
    out_players = dk_api.unavailable(dk_api.draftables(dgid))
    if not out_players.empty:
        cur = cur[~cur["player_dk_id"].isin(out_players["player_dk_id"])]
        cur = cur.reset_index(drop=True)
        names = ", ".join(
            f"{r.player_name} ({r.status})" for r in out_players.itertuples()
        )
        print(f"excluded {len(out_players)} unable to play: {names}")

    cfg = CONTESTS[args.contest]
    cur["projection"] = cfg["fn"](past, cur).to_numpy()

    built = optimizer.optimize(
        cur, objective="projection", n_lineups=args.lineups,
        stack_qb_with=cfg["stack"], min_difference=cfg["min_diff"],
        max_exposure=args.max_exposure,
    )
    if not built:
        print("optimizer found no feasible lineup")
        return 1

    print(f"\n=== {cfg['label']} ===")
    for lu in built:
        print(f"\n-- lineup {lu['lineup'].iloc[0] + 1} --")
        print(optimizer.summarize(lu, "projection"))

    print("\n=== top projected values ===")
    top = cur.nlargest(15, "projection")[
        ["player_name", "position", "team", "salary", "projection"]
    ]
    top = top.assign(pts_per_1k=(top["projection"] / (top["salary"] / 1000)).round(2))
    print(top.to_string(index=False))

    if args.out:
        pd.concat(built, ignore_index=True).to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Validate our DK scoring engine against RotoGuru's published DK points.

RotoGuru carries DraftKings' own scored totals for 2011-2021. Recomputing those
seasons from nflverse box scores and comparing is a free, independent
correctness test on the scoring engine. If this doesn't agree, nothing
downstream can be trusted.
"""
from __future__ import annotations

import sys

import pandas as pd

from dfs.scoring import score_offense
from dfs.sources import nflverse, rotoguru

SEASONS = (2019, 2020, 2021)
OFFENSE = ("QB", "RB", "WR", "TE")
TOL = 0.011  # RotoGuru publishes 2dp; allow rounding only


def normalize_name(s: pd.Series) -> pd.Series:
    return (
        s.str.normalize("NFKD")
        .str.encode("ascii", "ignore")
        .str.decode("ascii")
        .str.lower()
        .str.replace(r"\b(jr|sr|ii|iii|iv|v)\b\.?", "", regex=True)
        .str.replace(r"[^a-z ]", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def main() -> int:
    all_rows = []
    for season in SEASONS:
        rg = rotoguru.fetch_season(season)
        if rg.empty:
            print(f"[{season}] rotoguru: NO DATA")
            continue
        nv = nflverse.player_week(season)
        if nv.empty:
            print(f"[{season}] nflverse: NO DATA")
            continue

        nv = nv[nv["season_type"] == "REG"].copy()
        nv["dk_points_calc"] = score_offense(nv)
        nv["key_name"] = normalize_name(nv["player_display_name"])

        rg = rg[rg["position"].isin(OFFENSE)].copy()
        rg["key_name"] = normalize_name(rg["player_name"])

        # Join on position too: "Michael Thomas" was both a Saints WR and a
        # defensive back in 2019, and a name-only join silently pairs the WR's
        # salary with the DB's (zero) offensive stats. The real pipeline must
        # use an ID crosswalk, not names.
        nv_off = nv[nv["position"].isin(OFFENSE)]
        merged = rg.merge(
            nv_off[["season", "week", "key_name", "position", "dk_points_calc"]],
            on=["season", "week", "key_name", "position"],
            how="inner",
        )
        merged = merged.drop_duplicates(subset=["season", "week", "key_name"])
        merged["position_rg"] = merged["position"]

        merged["diff"] = (
            merged["dk_points_calc"] - merged["dk_points_rotoguru"]
        ).round(4)
        merged["agree"] = merged["diff"].abs() <= TOL

        n, ok = len(merged), int(merged["agree"].sum())
        print(
            f"[{season}] rotoguru={len(rg):>5}  matched={n:>5} "
            f"({n / len(rg):6.1%})  agree={ok:>5} ({ok / n:6.2%})  "
            f"maxdiff={merged['diff'].abs().max():.3f}"
        )
        all_rows.append(merged)

    if not all_rows:
        print("NO DATA -- cannot validate")
        return 1

    full = pd.concat(all_rows, ignore_index=True)
    n, ok = len(full), int(full["agree"].sum())
    print(f"\nTOTAL matched={n}  agree={ok} ({ok / n:.3%})")

    bad = full[~full["agree"]]
    if not bad.empty:
        print(f"\n--- {len(bad)} disagreements, worst 15 ---")
        cols = ["season", "week", "player_name", "position_rg",
                "dk_points_rotoguru", "dk_points_calc", "diff"]
        print(bad.reindex(bad["diff"].abs().sort_values(ascending=False).index)
                 .head(15)[cols].to_string(index=False))
        print("\n--- diff value counts (top 10) ---")
        print(bad["diff"].round(2).value_counts().head(10).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())

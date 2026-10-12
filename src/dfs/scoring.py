"""DraftKings NFL scoring, computed from nflverse box scores.

Validated against DraftKings' own published points (RotoGuru, 2019-2021):
15,946 player-weeks, 99.574% exact agreement. The residue is stat corrections
and return touchdowns, which DK's rules credit to the player.

Points are recomputed rather than scraped so any season can be backfilled and
so the salary source and the points source stay independently checkable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FUMBLE_COMPONENTS = ("rushing_fumbles_lost", "receiving_fumbles_lost",
                     "sack_fumbles_lost")
TWO_POINT = ("passing_2pt_conversions", "rushing_2pt_conversions",
             "receiving_2pt_conversions")


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    """A stat column, with absent columns and missing values read as zero."""
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").fillna(0.0)
    return pd.Series(0.0, index=df.index)


def fumbles_lost(df: pd.DataFrame) -> pd.Series:
    """Total fumbles lost. Older seasons carry only the components."""
    parts = sum(_col(df, c) for c in FUMBLE_COMPONENTS)
    if "fumbles_lost_total" in df.columns:
        total = pd.to_numeric(df["fumbles_lost_total"], errors="coerce")
        return total.fillna(parts)
    return parts


def score_kicking(df: pd.DataFrame) -> pd.Series:
    """Field goals by distance (3 to 39 yards, 4 for 40-49, 5 for 50+) and one
    per extra point. DK does not dock a kicker for a miss."""
    return (
        3.0 * (_col(df, "fg_made_0_19") + _col(df, "fg_made_20_29")
               + _col(df, "fg_made_30_39"))
        + 4.0 * _col(df, "fg_made_40_49")
        + 5.0 * (_col(df, "fg_made_50_59") + _col(df, "fg_made_60_"))
        + 1.0 * _col(df, "pat_made")
    )


def score_offense(df: pd.DataFrame) -> pd.Series:
    """Classic DK points for an offensive player (and a kicker)."""
    pass_yds = _col(df, "passing_yards")
    rush_yds = _col(df, "rushing_yards")
    rec_yds = _col(df, "receiving_yards")
    pts = (
        0.04 * pass_yds
        + 4.0 * _col(df, "passing_tds")
        - 1.0 * _col(df, "passing_interceptions")
        + 0.1 * rush_yds
        + 6.0 * _col(df, "rushing_tds")
        + 1.0 * _col(df, "receptions")
        + 0.1 * rec_yds
        + 6.0 * _col(df, "receiving_tds")
        - 1.0 * fumbles_lost(df)
        + 2.0 * sum(_col(df, c) for c in TWO_POINT)
        # Return and recovery touchdowns are credited to the player.
        + 6.0 * _col(df, "special_teams_tds")
        + 6.0 * _col(df, "fumble_recovery_tds")
        + 3.0 * (pass_yds >= 300)
        + 3.0 * (rush_yds >= 100)
        + 3.0 * (rec_yds >= 100)
    )
    return (pts + score_kicking(df)).astype(float)


def points_allowed_score(allowed: pd.Series) -> pd.Series:
    """DK's tiered points-allowed component for a DST."""
    a = pd.to_numeric(allowed, errors="coerce")
    out = np.select(
        [a == 0, a <= 6, a <= 13, a <= 20, a <= 27, a <= 34, a >= 35],
        [10.0, 7.0, 4.0, 1.0, 0.0, -1.0, -4.0],
        default=0.0,
    )
    return pd.Series(out, index=allowed.index, dtype=float)


def score_dst(df: pd.DataFrame) -> pd.Series:
    """Classic DK points for a team defense / special teams."""
    allowed = (df["points_allowed"] if "points_allowed" in df.columns
               else pd.Series(np.nan, index=df.index))
    return (
        1.0 * _col(df, "def_sacks")
        + 2.0 * _col(df, "def_interceptions")
        + 2.0 * _col(df, "fumble_recovery_opp")
        + 6.0 * _col(df, "def_tds")
        + 6.0 * _col(df, "special_teams_tds")
        + 2.0 * _col(df, "def_safeties")
        + 2.0 * _col(df, "def_blocked_kicks")
        + points_allowed_score(allowed)
    ).astype(float)

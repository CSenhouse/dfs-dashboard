"""nflverse: box scores, schedules and Vegas lines, rosters, snaps, injuries,
depth charts and play-by-play. Free, and back to 1999 for most of it.

nflverse rewrites the CURRENT season's files every week at the same URL
(rule 12). Cached forever, a live slate would be projected on a snapshot taken
mid-week-1 and degrade every week after. So the season in progress expires
after LIVE_MAX_AGE; completed seasons are final and stay cached permanently.
"""
from __future__ import annotations

import datetime as dt
import io

import pandas as pd

from dfs import http

RELEASE = "https://github.com/nflverse/nflverse-data/releases/download"
GAMES = "https://github.com/nflverse/nfldata/raw/master/data/games.csv"

LIVE_MAX_AGE = 6 * 3600


def current_season(today: dt.date | None = None) -> int:
    """A season is named for the year it started; January and February
    belong to the previous one."""
    today = today or dt.date.today()
    return today.year if today.month >= 3 else today.year - 1


def _max_age(season: int | None) -> float | None:
    if season is None:
        return None
    return LIVE_MAX_AGE if int(season) == current_season() else None


def _fetch(url: str, season: int | None, cache_only: bool = False
           ) -> bytes | None:
    return http.get(url, "." + url.rsplit(".", 1)[-1], min_interval=0.5,
                    max_age=_max_age(season), cache_only=cache_only)


def _parquet(url: str, season: int | None, columns: list[str] | None = None,
             cache_only: bool = False) -> pd.DataFrame:
    body = _fetch(url, season, cache_only)
    if not body:
        return pd.DataFrame()
    try:
        return pd.read_parquet(io.BytesIO(body), columns=columns)
    except Exception:  # noqa: BLE001 -- a truncated download is just missing
        return pd.DataFrame()


# --- box scores ---------------------------------------------------------------

_OLD_NAMES = {
    "recent_team": "team",
    "interceptions": "passing_interceptions",
    "sacks": "sacks_suffered",
    "sack_yards": "sack_yards_lost",
    "opponent_team": "opponent_team",
}


def player_week(season: int, cache_only: bool = False) -> pd.DataFrame:
    """One row per player per game he recorded a stat in.

    nflverse emits NO row for a player who recorded nothing -- a scratch and a
    zero look the same here (rule 5), which slates.assemble keeps apart.
    """
    df = _parquet(f"{RELEASE}/stats_player/stats_player_week_{season}.parquet",
                  season, cache_only=cache_only)
    if df.empty:
        df = _parquet(f"{RELEASE}/player_stats/player_stats_{season}.parquet",
                      season, cache_only=cache_only)
        df = df.rename(columns={k: v for k, v in _OLD_NAMES.items()
                                if k in df.columns and v not in df.columns})
    if df.empty:
        return df
    if "season_type" not in df.columns:
        df["season_type"] = "REG"
    if "player_display_name" not in df.columns and "player_name" in df.columns:
        df["player_display_name"] = df["player_name"]
    return df


def team_week(season: int, cache_only: bool = False) -> pd.DataFrame:
    """Team totals per game, including the defensive stats a DST scores on."""
    return _parquet(f"{RELEASE}/stats_team/stats_team_week_{season}.parquet",
                    season, cache_only=cache_only)


# --- schedule and lines ------------------------------------------------------

def games(cache_only: bool = False) -> pd.DataFrame:
    """Every game 1999 to the end of the published schedule, with Vegas lines,
    roof, surface, temperature, wind and rest."""
    body = http.get(GAMES, ".csv", min_interval=0, max_age=LIVE_MAX_AGE,
                    cache_only=cache_only)
    if not body:
        return pd.DataFrame()
    df = pd.read_csv(io.BytesIO(body), low_memory=False)
    df["gameday"] = pd.to_datetime(df["gameday"], errors="coerce")
    for col in ("home_team", "away_team"):
        df[col] = df[col].replace({"OAK": "LV", "SD": "LAC", "STL": "LA"})
    return df


# --- people ------------------------------------------------------------------

def rosters(season: int, cache_only: bool = False) -> pd.DataFrame:
    df = _parquet(f"{RELEASE}/weekly_rosters/roster_weekly_{season}.parquet",
                  season, cache_only=cache_only)
    if df.empty:
        df = _parquet(f"{RELEASE}/rosters/roster_{season}.parquet", season,
                      cache_only=cache_only)
    return df


def players(cache_only: bool = False) -> pd.DataFrame:
    """The master player table, which carries the pfr and gsis id mapping."""
    return _parquet(f"{RELEASE}/players/players.parquet", None,
                    cache_only=cache_only)


def snap_counts(season: int, cache_only: bool = False) -> pd.DataFrame:
    return _parquet(f"{RELEASE}/snap_counts/snap_counts_{season}.parquet",
                    season, cache_only=cache_only)


def injuries(season: int, cache_only: bool = False) -> pd.DataFrame:
    return _parquet(f"{RELEASE}/injuries/injuries_{season}.parquet", season,
                    cache_only=cache_only)


def depth_charts(season: int, cache_only: bool = False) -> pd.DataFrame:
    return _parquet(f"{RELEASE}/depth_charts/depth_charts_{season}.parquet",
                    season, cache_only=cache_only)


PBP_COLUMNS = [
    "season", "week", "season_type", "posteam", "defteam", "play_type",
    "yardline_100", "pass_attempt", "rush_attempt", "receiver_player_id",
    "rusher_player_id", "passer_player_id", "air_yards", "complete_pass",
    "touchdown", "two_point_attempt", "qb_scramble",
]


def pbp(season: int, columns: list[str] | None = None,
        cache_only: bool = False) -> pd.DataFrame:
    """Play-by-play. Large (~20-40MB a season), so read only what is used."""
    cols = columns or PBP_COLUMNS
    df = _parquet(f"{RELEASE}/pbp/play_by_play_{season}.parquet", season,
                  cache_only=cache_only)
    if df.empty:
        return df
    return df[[c for c in cols if c in df.columns]]

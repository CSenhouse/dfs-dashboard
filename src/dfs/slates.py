"""Assemble a slate: DraftKings salaries joined to the schedule and to what
each player actually scored.

Three guards live here, each because something silently contaminated the
dataset:

- matches_schedule: DraftKings runs simulated "Madden Stream" contests with
  real teams, real players and a real DST. Nothing in the payload gives them
  away; the schedule does (rule 6).
- week_is_complete: a live week has no actual scores, and prepare() would
  read every one of them as a real zero (rule 7).
- no_stat_line: nflverse emits no row for a player who recorded nothing. For
  lineup scoring that is 0; for training it may be a missing observation, so
  the fact is kept rather than collapsed (rule 5).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from dfs import crosswalk
from dfs.scoring import score_dst, score_offense
from dfs.sources import dk_api, nflverse

_cache: dict[str, tuple[float, pd.DataFrame]] = {}
_TTL = 3600


def _games() -> pd.DataFrame:
    hit = _cache.get("games")
    if hit and time.time() - hit[0] < _TTL:
        return hit[1]
    g = nflverse.games()
    _cache["games"] = (time.time(), g)
    return g


def _memo(key: str, fn):
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < _TTL:
        return hit[1]
    val = fn()
    _cache[key] = (time.time(), val)
    return val


# --- schedule ----------------------------------------------------------------

def _eastern_date(ts) -> pd.Timestamp | None:
    t = pd.to_datetime(ts, utc=True, errors="coerce")
    if pd.isna(t):
        return None
    return pd.Timestamp(t.tz_convert("America/New_York").date())


def season_week_for(start_time) -> tuple[int, int] | None:
    """(season, week) of the NFL week a slate starting at `start_time` belongs
    to, or None if it falls outside any scheduled week."""
    day = _eastern_date(start_time)
    if day is None:
        return None
    g = _games()
    if g.empty:
        return None
    span = g.groupby(["season", "week"])["gameday"].agg(["min", "max"])
    span = span.reset_index()
    ok = span[(span["min"] - pd.Timedelta(days=6) <= day) & (span["max"] >= day)]
    if ok.empty:
        return None
    first = ok.sort_values(["season", "week"]).iloc[0]
    return int(first["season"]), int(first["week"])


def _week_games(season: int, week: int) -> pd.DataFrame:
    g = _games()
    if g.empty:
        return g
    return g[(g["season"] == season) & (g["week"] == week)]


def matches_schedule(dk: pd.DataFrame, season: int, week: int,
                     min_fraction: float = 0.75) -> bool:
    """True when the slate's matchups are games actually played that week."""
    if dk is None or dk.empty or "competition_id" not in dk.columns:
        return False
    sched = _week_games(season, week)
    if sched.empty:
        return False
    real = {frozenset(crosswalk.normalize_team(pd.Series([h, a])))
            for h, a in zip(sched["home_team"], sched["away_team"])}
    teams = dk.assign(team=crosswalk.normalize_team(dk["team"]))
    pairs = teams.groupby("competition_id")["team"].agg(
        lambda s: frozenset(x for x in s.dropna() if x))
    pairs = [p for p in pairs if len(p) == 2]
    if not pairs:
        return False
    hit = sum(p in real for p in pairs)
    return hit / len(pairs) >= min_fraction


def week_is_complete(season: int, week: int) -> bool:
    """Every game in the week has a final score."""
    sched = _week_games(season, week)
    if sched.empty:
        return False
    return bool(sched["result"].notna().all())


# --- actual scores -----------------------------------------------------------

def _player_points(season: int, week: int) -> pd.DataFrame:
    pw = _memo(f"pw{season}", lambda: nflverse.player_week(season))
    if pw.empty:
        return pd.DataFrame(columns=["gsis_id", "dk_actual"])
    w = pw[pw["week"] == week].copy()
    w["dk_actual"] = score_offense(w)
    w = w.rename(columns={"player_id": "gsis_id"})
    return w.groupby("gsis_id", as_index=False)["dk_actual"].sum()


def _dst_points(season: int, week: int) -> pd.DataFrame:
    tw = _memo(f"tw{season}", lambda: nflverse.team_week(season))
    sched = _week_games(season, week)
    if tw.empty or sched.empty:
        return pd.DataFrame(columns=["team", "dk_actual"])
    w = tw[tw["week"] == week].copy()
    w["team"] = crosswalk.normalize_team(w["team"])
    allowed = {}
    for r in sched.itertuples():
        if pd.isna(r.result):
            continue
        h, a = crosswalk.normalize_team(pd.Series([r.home_team, r.away_team]))
        allowed[h] = r.away_score
        allowed[a] = r.home_score
    w["points_allowed"] = w["team"].map(allowed)
    w = w.dropna(subset=["points_allowed"])
    w["dk_actual"] = score_dst(w)
    return w[["team", "dk_actual"]]


def _crosswalk(dk: pd.DataFrame, season: int) -> pd.DataFrame:
    roster = _memo(f"ro{season}", lambda: nflverse.rosters(season))
    if roster.empty:
        return pd.DataFrame(columns=["player_dk_id", "gsis_id", "matched_on"])
    r = roster.copy()
    r["season"] = season
    keep = ["season", "team", "position", "full_name", "gsis_id"]
    r = r[[c for c in keep if c in r.columns]].drop_duplicates()
    d = dk.assign(season=season)
    return crosswalk.build(d, r)


def assemble(dgid: int, season: int | None = None, week: int | None = None
             ) -> pd.DataFrame:
    """Every priced player on a draft group, with season, week, opponent,
    gsis_id, and -- once the week is played -- the points he scored."""
    dk = dk_api.draftables(dgid)
    if dk.empty:
        return pd.DataFrame()
    if dk_api.is_live(dk):
        dk = dk_api.draftables(dgid, max_age=dk_api.LIVE_MAX_AGE)
    dk = dk[dk["salary"].fillna(0) > 0].copy()
    if dk.empty:
        return pd.DataFrame()

    if season is None or week is None:
        sw = season_week_for(dk["start_time"].min())
        if sw is None:
            return pd.DataFrame()
        season, week = sw
    dk["season"], dk["week"] = int(season), int(week)

    xw = _crosswalk(dk, int(season))
    dk = dk.merge(xw, on="player_dk_id", how="left")

    complete = week_is_complete(int(season), int(week))
    dk["dk_actual"] = np.nan
    dk["no_stat_line"] = pd.Series(pd.NA, index=dk.index, dtype="boolean")
    if complete:
        pts = _player_points(int(season), int(week)).set_index("gsis_id")
        people = dk["position"] != "DST"
        got = dk.loc[people, "gsis_id"].map(pts["dk_actual"])
        dk.loc[people, "dk_actual"] = got
        dk.loc[people, "no_stat_line"] = got.isna() & dk.loc[people,
                                                              "gsis_id"].notna()
        dst = _dst_points(int(season), int(week)).set_index("team")
        is_dst = dk["position"] == "DST"
        dk.loc[is_dst, "dk_actual"] = dk.loc[is_dst, "team"].map(dst["dk_actual"])
        dk.loc[is_dst, "no_stat_line"] = dk.loc[is_dst, "dk_actual"].isna()
        # A matched player with no stat row scored 0 for a lineup; prepare()
        # keeps "played" separate.
    dk["no_stat_line"] = dk["no_stat_line"].astype("boolean")
    return dk.reset_index(drop=True)

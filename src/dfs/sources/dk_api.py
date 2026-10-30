"""DraftKings salaries: the lobby, and the players priced on a draft group.

Unauthenticated. Draft group ids are shared across every sport and issued in
time order, and DK serves historical draft groups as well as live ones -- back
to about id 10000 (mid 2016).

Two player endpoints exist. `api.draftkings.com/.../draftables` returned 403
to this project after it was probed too hard (the contest API went the same
way), so the primary source is the older lobby endpoint
`lineup/getavailableplayers`, which carries salary, status, team, opponent and
the game for every priced player -- one row per player, at the FLEX price on a
Showdown slate. The newer endpoint is kept as a fallback.

Gotchas, all found by probing:
- salary 0, or a salary that is a draft RANK (1..N), means a snake/best-ball
  draft group, not a salary-cap slate;
- an id that resolves tells you nothing about the sport: check the positions
  (a DST means NFL) and the teams (college slates share QB/RB/WR/TE);
- DK prices players on injured reserve at the minimum salary. Filter them on
  a LIVE slate with `playable()`; on a historical draft group the status is
  whatever DK held when the payload was cached and must never be a feature.
"""
from __future__ import annotations

import json
import re

import pandas as pd

from dfs import http
from dfs.config import CAPTAIN_MULTIPLIER
from dfs.crosswalk import NFL_TEAMS, normalize_team

LOBBY = "https://www.draftkings.com/lobby/getcontests?sport={sport}"
LEGACY = "https://www.draftkings.com/lineup/getavailableplayers?draftGroupId={dgid}"
DRAFTABLES = ("https://api.draftkings.com/draftgroups/v1/draftgroups/{dgid}"
              "/draftables?format=json")

CONTEST_TYPE_CLASSIC = 21
CONTEST_TYPE_SHOWDOWN = 96

# How long a live payload is trusted. Salaries and statuses move until lock.
LIVE_MAX_AGE = 15 * 60

# What DraftKings prices on a standard NFL salary-cap slate. IDP variants
# (LB, CB, S, DE, DT) are a different game with different pricing, and a
# Classic slate that prices kickers is a variant too. An IDP Classic slate
# has MORE players than the real main slate, so "most players" picked it.
CLASSIC_POSITIONS = frozenset({"QB", "RB", "WR", "TE", "DST"})
SHOWDOWN_POSITIONS = CLASSIC_POSITIONS | {"K"}

UNAVAILABLE = frozenset({"IR", "OUT", "O", "NA", "SUSP", "PUP", "DNP",
                         "INACTIVE", "NFI", "SUS", "EXEMPT", "RES"})

COLUMNS = ["draft_group_id", "player_dk_id", "draftable_id", "player_name",
           "position", "team", "opponent", "salary", "captain_salary",
           "status", "competition_id", "competition_name", "start_time"]

_DATE = re.compile(r"/Date\((-?\d+)\)/")


def _ms(value, eastern: bool = False) -> pd.Timestamp:
    """A DraftKings timestamp. The legacy endpoint writes Eastern WALL-CLOCK
    time as if it were UTC (a 1:00 PM ET kickoff arrives as 13:00Z), so
    `eastern=True` re-reads it in America/New_York. Read as UTC it put every
    game four hours early, and a slate looked started at 9 AM -- which
    stopped the live refresh before Sunday's inactives landed."""
    if value is None:
        return pd.NaT
    if isinstance(value, (int, float)):
        ts = pd.to_datetime(int(value), unit="ms")
    else:
        m = _DATE.search(str(value))
        if not m:
            return pd.to_datetime(value, utc=True, errors="coerce")
        ts = pd.to_datetime(int(m.group(1)), unit="ms")
    if eastern:
        return ts.tz_localize("America/New_York", ambiguous=True,
                              nonexistent="shift_forward").tz_convert("UTC")
    return ts.tz_localize("UTC")


def _from_legacy(dgid: int, payload: dict) -> pd.DataFrame:
    players = payload.get("playerList") or []
    if not players:
        return pd.DataFrame(columns=COLUMNS)
    games = payload.get("teamList") or {}
    rows = []
    for p in players:
        home, away = p.get("htabbr"), p.get("atabbr")
        tid, htid = p.get("tid"), p.get("htid")
        team = home if tid == htid else away
        opp = away if tid == htid else home
        g = games.get(str(p.get("tsid")), {})
        start = (_ms(g.get("tz"), eastern=True) if g.get("tz")
                 else _ms(p.get("dgst"), eastern=True))
        first, last = (p.get("fn") or "").strip(), (p.get("ln") or "").strip()
        rows.append({
            "draft_group_id": int(dgid),
            "player_dk_id": p.get("pdkid") or p.get("pid"),
            "draftable_id": p.get("pid"),
            "player_name": f"{first} {last}".strip(),
            "position": p.get("pn"),
            "team": team,
            "opponent": opp,
            "salary": p.get("s"),
            "status": p.get("i") or None,
            "competition_id": p.get("tsid"),
            "competition_name": f"{away} @ {home}" if home and away else None,
            "start_time": start,
        })
    return pd.DataFrame(rows)


def _from_draftables(dgid: int, payload: dict) -> pd.DataFrame:
    players = payload.get("draftables") or []
    if not players:
        return pd.DataFrame(columns=COLUMNS)
    rows = []
    for p in players:
        comp = p.get("competition") or {}
        name = comp.get("name") or ""
        rows.append({
            "draft_group_id": int(dgid),
            "player_dk_id": p.get("playerDkId") or p.get("playerId"),
            "draftable_id": p.get("draftableId"),
            "player_name": p.get("displayName"),
            "position": p.get("position"),
            "team": p.get("teamAbbreviation"),
            "opponent": None,
            "salary": p.get("salary"),
            "status": p.get("status"),
            "roster_slot_id": p.get("rosterSlotId"),
            "competition_id": comp.get("competitionId"),
            "competition_name": name or None,
            "start_time": _ms(comp.get("startTime")),
        })
    df = pd.DataFrame(rows)
    # One row per roster slot a player can fill; keep the cheapest, which on
    # a Showdown slate is FLEX. Keying on the lowest slot id picks Captain and
    # silently prices the whole slate at 1.5x.
    df["salary"] = pd.to_numeric(df["salary"], errors="coerce")
    df = df.sort_values("salary").drop_duplicates("player_dk_id", keep="first")
    return df.drop(columns=["roster_slot_id"])


def _opponents(df: pd.DataFrame) -> pd.Series:
    """Fill the opponent from the other team in the same competition."""
    teams = df.groupby("competition_id")["team"].agg(
        lambda s: sorted(set(s.dropna())))
    def other(r):
        t = teams.get(r["competition_id"], [])
        rest = [x for x in t if x != r["team"]]
        return rest[0] if len(rest) == 1 else None
    return df.apply(other, axis=1)


def draftables(dgid: int, *, cache_only: bool = False, refresh: bool = False,
               min_interval: float = 0.5, max_age: float | None = None,
               retry_miss: bool = True) -> pd.DataFrame:
    """Every player priced on draft group `dgid`, one row each."""
    df = pd.DataFrame(columns=COLUMNS)
    body = http.get(LEGACY.format(dgid=int(dgid)), ".json",
                    min_interval=min_interval, cache_only=cache_only,
                    refresh=refresh, max_age=max_age, retry_miss=retry_miss)
    if body:
        try:
            df = _from_legacy(dgid, json.loads(body))
        except (ValueError, TypeError, AttributeError):
            df = pd.DataFrame(columns=COLUMNS)
    # DK creates featured main slates weeks ahead, and a sweep that reached one
    # before it was priced cached a body with no players. Cached forever, the
    # live main slate then vanished from the dashboard (week 8, dg 153728).
    # An empty player list is never final while we are online: ask again.
    if df.empty and body and not cache_only and not refresh:
        fresh = http.get(LEGACY.format(dgid=int(dgid)), ".json",
                         min_interval=min_interval, refresh=True)
        if fresh:
            try:
                df = _from_legacy(dgid, json.loads(fresh))
            except (ValueError, TypeError, AttributeError):
                df = pd.DataFrame(columns=COLUMNS)
    if df.empty:
        status = http.last_status       # what the live request said
        body = http.get(DRAFTABLES.format(dgid=int(dgid)), ".json",
                        min_interval=min_interval, cache_only=True)
        http.last_status = status
        if body:
            try:
                df = _from_draftables(dgid, json.loads(body))
            except (ValueError, TypeError, AttributeError):
                df = pd.DataFrame(columns=COLUMNS)
    if df.empty:
        return pd.DataFrame(columns=COLUMNS)

    df["salary"] = pd.to_numeric(df["salary"], errors="coerce")
    df["player_dk_id"] = pd.to_numeric(df["player_dk_id"], errors="coerce")
    df = df.dropna(subset=["player_dk_id"])
    df["player_dk_id"] = df["player_dk_id"].astype("int64")
    df = df.drop_duplicates("player_dk_id", keep="first")
    df["team"] = normalize_team(df["team"])
    if df["opponent"].isna().all():
        df["opponent"] = _opponents(df)
    else:
        df["opponent"] = normalize_team(df["opponent"])
    # Not "D": that is a hockey defenseman, and mapping it made two NHL
    # Showdowns look like NFL slates with a DST. DK writes DST for the NFL.
    df["position"] = df["position"].astype(str).str.upper().replace(
        {"DEF": "DST"})
    df["status"] = df["status"].where(df["status"].astype(str).str.strip()
                                      .ne(""), None)
    df["captain_salary"] = df["salary"] * CAPTAIN_MULTIPLIER
    df["start_time"] = pd.to_datetime(df["start_time"], utc=True,
                                      errors="coerce")
    return df[COLUMNS].reset_index(drop=True)


def classify(df: pd.DataFrame) -> dict:
    """What kind of draft group this is.

    kind is one of empty / draft / variant / classic / showdown. is_nfl
    requires a DST (DK never prices a college DST), every team to be an NFL
    team, and only standard NFL positions -- a hockey slate and an IDP slate
    both passed the first two checks.
    """
    if df is None or df.empty or "salary" not in df.columns:
        return {"kind": "empty", "is_nfl": False, "rank_like": False, "n": 0,
                "n_games": 0, "start_time": pd.NaT}
    sal = pd.to_numeric(df["salary"], errors="coerce").fillna(0)
    n = len(df)
    n_games = int(df["competition_id"].nunique()) \
        if "competition_id" in df.columns else 0
    start = (pd.to_datetime(df["start_time"], utc=True, errors="coerce").min()
             if "start_time" in df.columns else pd.NaT)

    positions = set(df["position"].dropna().astype(str).str.upper()) \
        if "position" in df.columns else set()
    teams = set(normalize_team(df["team"]).dropna()) - {""} \
        if "team" in df.columns else set()
    is_nfl = ("DST" in positions and bool(teams) and teams <= NFL_TEAMS
              and positions <= SHOWDOWN_POSITIONS)

    priced = sal[sal > 0]
    # A draft rank is a small integer that is not a multiple of $100. Real
    # salaries -- Showdown captains included -- always are multiples of 100.
    rank_like = (priced.empty or not (priced % 100 == 0).all()
                 or priced.max() <= max(2 * n, 100))
    if rank_like:
        kind = "draft"
    elif n_games == 1:
        kind = "showdown"
    elif positions <= CLASSIC_POSITIONS:
        kind = "classic"
    else:
        kind = "variant"
    return {"kind": kind, "is_nfl": bool(is_nfl), "rank_like": bool(rank_like),
            "n": n, "n_games": n_games, "start_time": start}


def peek(dgid: int, cache_only: bool = True) -> dict | None:
    """classify() plus identifiers, or None if the draft group is empty."""
    df = draftables(dgid, cache_only=cache_only)
    if df.empty:
        return None
    info = classify(df)
    info.update({"draft_group_id": int(dgid), "n_players": info["n"]})
    return info


def live_draftgroups(sport: str = "NFL") -> pd.DataFrame:
    """Draft groups open in the lobby right now."""
    body = http.get(LOBBY.format(sport=sport), ".json", min_interval=0,
                    refresh=True)
    cols = ["draft_group_id", "contest_type_id", "start_time", "suffix",
            "n_games", "tag"]
    if not body:
        return pd.DataFrame(columns=cols)
    try:
        groups = json.loads(body).get("DraftGroups") or []
    except ValueError:
        return pd.DataFrame(columns=cols)
    rows = [{
        "draft_group_id": int(g["DraftGroupId"]),
        "contest_type_id": g.get("ContestTypeId"),
        "start_time": pd.to_datetime(g.get("StartDate"), utc=True,
                                     errors="coerce"),
        "suffix": (g.get("ContestStartTimeSuffix") or "").strip(),
        "n_games": g.get("GameCount"),
        "tag": g.get("DraftGroupTag"),
    } for g in groups if g.get("DraftGroupId")]
    return pd.DataFrame(rows, columns=cols)


def _status(df: pd.DataFrame) -> pd.Series:
    return df["status"].astype(str).str.strip().str.upper()


def unavailable(df: pd.DataFrame) -> pd.DataFrame:
    """Players DraftKings flags as unable to play (IR, OUT, suspended...)."""
    if df is None or df.empty or "status" not in df.columns:
        return pd.DataFrame(columns=getattr(df, "columns", []))
    return df[_status(df).isin(UNAVAILABLE)].copy()


def playable(df: pd.DataFrame) -> pd.DataFrame:
    """The pool minus players who cannot take a snap. Q and D stay -- they
    often play, and the projection handles them. LIVE SLATES ONLY."""
    if df is None or df.empty or "status" not in df.columns:
        return df.copy() if df is not None else pd.DataFrame()
    return df[~_status(df).isin(UNAVAILABLE)].copy()


def is_live(df: pd.DataFrame) -> bool:
    """True when the slate has not started yet."""
    if df.empty or "start_time" not in df.columns:
        return False
    start = pd.to_datetime(df["start_time"], utc=True, errors="coerce").min()
    return bool(pd.notna(start) and start > pd.Timestamp.now(tz="UTC"))

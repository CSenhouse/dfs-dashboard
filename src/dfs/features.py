"""Features for every priced player on a slate.

Every feature is classified by when it becomes knowable (docs/MODELING.md 3):

    LAGGED   rolling volume, snap share, red-zone usage, form, opponent
             defense -- shift BEFORE aggregating (rule 4); week W sees only
             weeks < W. `_roll` does this.
    PREGAME  Vegas lines, injury report, depth chart, rest, venue -- the
             current week is legal.

Rolling windows run over the weeks the player's TEAM played (a bye is not a
missed week), keep form as "how good when he plays", go NaN once the window
holds only absences, and report `weeks_since_played` alongside -- zeroing
missed weeks measured 6.6 lineup points worse (section 9).

Rule 11: a lagged feature needs a row for the week it predicts. On a live
slate that week is not in the box scores, so `_with_upcoming` appends a
statistics-free placeholder row that `_roll` fills from the weeks before it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from dfs import crosswalk
from dfs.scoring import score_dst, score_offense
from dfs.sources import nflverse

POSITIONS = ("QB", "RB", "WR", "TE", "DST", "K")
OFFENSE = ("QB", "RB", "WR", "TE", "K", "FB")
WINDOWS = (3, 6)

ROLL_COLS = ["dk_points", "targets", "carries", "receptions",
             "receiving_yards", "rushing_yards", "passing_yards", "attempts",
             "passing_tds", "target_share", "air_yards_share", "wopr",
             "offense_pct", "rz_targets", "rz_carries", "gl_carries",
             "pbp_air_yards", "appeared"]
SHORT_ONLY = {"receptions", "receiving_yards", "rushing_yards",
              "passing_yards", "attempts", "passing_tds", "target_share",
              "air_yards_share", "wopr", "rz_targets", "rz_carries",
              "gl_carries", "pbp_air_yards"}

FEATURES = (
    ["salary", "salary_pct", "salary_team_rank", "value"]
    + [f"pos_{p}" for p in POSITIONS]
    + ["is_home", "spread", "game_total", "implied_total",
       "opp_implied_total", "temp", "wind", "dome", "rest_days", "div_game"]
    + [f"{c}_r{w}" for c in ROLL_COLS for w in WINDOWS
       if not (w == 6 and c in SHORT_ONLY)]
    + ["weeks_since_played"]
    + ["prior_ppg", "prior_best", "prior_games", "prior_targets_pg",
       "prior_carries_pg"]
    + ["team_plays_r3", "team_pass_rate_r3", "team_rz_plays_r3"]
    + ["def_allowed_r3", "def_allowed_r6"]
    + ["injury_prob", "injury_code", "practice_code"]
    + ["depth_team", "is_starter"]
)

INJURY_PROB = {"out": 0.0, "doubtful": 0.15, "questionable": 0.75}
INJURY_CODE = {"out": 3, "doubtful": 2, "questionable": 1}
PRACTICE_CODE = {"did not participate in practice": 2,
                 "limited participation in practice": 1,
                 "full participation in practice": 0}


def feature_columns(df: pd.DataFrame) -> list[str]:
    """The model's inputs that are present in `df`, in a fixed order."""
    return [c for c in FEATURES if c in df.columns]


# --- rolling ------------------------------------------------------------------

def _roll(df: pd.DataFrame, cols: list[str], by: str = "gsis_id",
          windows=WINDOWS, calendar: pd.DataFrame | None = None,
          zero_absences: bool = False) -> pd.DataFrame:
    """Shifted rolling means of `cols` within (player, season).

    Without a calendar the window runs over rows that exist -- appearances.
    With one it runs over calendar weeks: `appeared` is 0 in a missed week,
    other columns are NaN there (or 0 with zero_absences), and
    weeks_since_played counts calendar weeks since the last appearance.
    """
    keys = [by, "season", "week"]
    if calendar is not None:
        cal = calendar[keys].drop_duplicates()
        d = cal.merge(df, on=keys, how="left")
        if "appeared" in cols and "appeared" in d.columns:
            d["appeared"] = d["appeared"].fillna(0.0)
        if zero_absences:
            for c in cols:
                d[c] = d[c].fillna(0.0)
    else:
        d = df.copy()
    d = d.sort_values(keys, kind="stable").reset_index(drop=True)
    grp = d.groupby([by, "season"], sort=False).ngroup()

    for c in cols:
        shifted = d[c].astype(float).groupby(grp).shift(1)
        for w in windows:
            rolled = shifted.groupby(grp).rolling(w, min_periods=1).mean()
            d[f"{c}_r{w}"] = rolled.reset_index(level=0, drop=True).sort_index()

    if calendar is not None:
        if "appeared" in d.columns:
            played = d["appeared"].fillna(0) > 0
        else:
            played = d[cols[0]].notna()
        k = d.groupby(grp).cumcount().astype(float)
        last = pd.Series(np.where(played, k, np.nan), index=d.index)
        last = last.groupby(grp).shift(1)
        last = last.groupby(grp).ffill()
        d["weeks_since_played"] = k - last
    return d


def _with_upcoming(hist: pd.DataFrame, slate: pd.DataFrame,
                   by: str = "gsis_id") -> pd.DataFrame:
    """Append a placeholder row for every (id, season, week) on the slate that
    the history lacks, so a lagged feature has a row to land on (rule 11)."""
    keys = [by, "season", "week"]
    want = slate.dropna(subset=[by])[keys].drop_duplicates()
    have = hist[keys].drop_duplicates() if not hist.empty else want.iloc[:0]
    new = want.merge(have, on=keys, how="left", indicator=True)
    new = new[new["_merge"] == "left_only"].drop(columns="_merge")
    if new.empty:
        return hist
    return pd.concat([hist, new], ignore_index=True)


# --- loading ----------------------------------------------------------------

_mem: dict = {}


def _load(name: str, season: int | None = None) -> pd.DataFrame:
    key = (name, season)
    if key not in _mem:
        fn = {"player_week": nflverse.player_week, "team_week": nflverse.team_week,
              "rosters": nflverse.rosters, "snaps": nflverse.snap_counts,
              "injuries": nflverse.injuries, "depth": nflverse.depth_charts,
              "pbp": nflverse.pbp}.get(name)
        if name == "games":
            _mem[key] = nflverse.games()
        elif name == "players":
            _mem[key] = nflverse.players()
        else:
            _mem[key] = fn(season)
    return _mem[key]


def _team_games(games: pd.DataFrame) -> pd.DataFrame:
    """One row per team per game, from that team's point of view."""
    g = games.copy()
    home = pd.DataFrame({
        "season": g["season"], "week": g["week"], "team": g["home_team"],
        "opponent": g["away_team"], "is_home": 1.0,
        "margin": g["spread_line"], "game_total": g["total_line"],
        "rest_days": g["home_rest"], "points_for": g["home_score"],
        "points_against": g["away_score"], "gameday": g["gameday"],
        "gametime": g["gametime"], "roof": g["roof"], "temp": g["temp"],
        "wind": g["wind"], "div_game": g["div_game"],
    })
    away = home.copy()
    away["team"], away["opponent"] = g["away_team"], g["home_team"]
    away["is_home"] = 0.0
    away["margin"] = -g["spread_line"]
    away["rest_days"] = g["away_rest"]
    away["points_for"], away["points_against"] = g["away_score"], g["home_score"]
    out = pd.concat([home, away], ignore_index=True)
    out["team"] = crosswalk.normalize_team(out["team"])
    out["opponent"] = crosswalk.normalize_team(out["opponent"])
    return out


# --- per-season history tables ------------------------------------------------

def _player_history(season: int, games_tg: pd.DataFrame) -> pd.DataFrame:
    """One row per player per game he appeared in: stats, snaps, red zone."""
    pw = _load("player_week", season)
    if pw.empty:
        return pd.DataFrame()
    pw = pw[pw["season_type"].isin(["REG", "POST"])].copy()
    h = pd.DataFrame({
        "gsis_id": pw["player_id"], "season": pw["season"].astype(int),
        "week": pw["week"].astype(int),
        "team": crosswalk.normalize_team(pw["team"]),
        "opponent": crosswalk.normalize_team(pw["opponent_team"]),
        "position": crosswalk.normalize_position(pw["position"]),
        "dk_points": score_offense(pw),
    })
    for c in ("targets", "carries", "receptions", "receiving_yards",
              "rushing_yards", "passing_yards", "attempts", "passing_tds",
              "target_share", "air_yards_share", "wopr"):
        h[c] = pd.to_numeric(pw[c], errors="coerce") if c in pw.columns \
            else np.nan

    snaps = _load("snaps", season)
    players = _load("players")
    if not snaps.empty and not players.empty and "pfr_id" in players.columns:
        pmap = players.dropna(subset=["pfr_id"]).drop_duplicates("pfr_id")
        pmap = pmap.set_index("pfr_id")["gsis_id"]
        s = snaps[snaps["game_type"].isin(["REG", "POST"])].copy()
        s["gsis_id"] = s["pfr_player_id"].map(pmap)
        s = s.dropna(subset=["gsis_id"])
        s = s.groupby(["gsis_id", "season", "week"], as_index=False).agg(
            offense_pct=("offense_pct", "max"),
            offense_snaps=("offense_snaps", "max"))
        s["season"], s["week"] = s["season"].astype(int), s["week"].astype(int)
        h = h.merge(s, on=["gsis_id", "season", "week"], how="outer")
    else:
        h["offense_pct"] = np.nan
        h["offense_snaps"] = np.nan

    pbp = _load("pbp", season)
    if not pbp.empty:
        p = pbp[pbp["season_type"].isin(["REG", "POST"])]
        rz = p["yardline_100"] <= 20
        gl = p["yardline_100"] <= 5
        tgt = p[p["pass_attempt"] == 1].dropna(subset=["receiver_player_id"])
        a = tgt.groupby(["receiver_player_id", "season", "week"]).agg(
            pbp_air_yards=("air_yards", "sum"),
            rz_targets=("yardline_100", lambda s: (s <= 20).sum()))
        car = p[p["rush_attempt"] == 1].dropna(subset=["rusher_player_id"])
        b = car.groupby(["rusher_player_id", "season", "week"]).agg(
            rz_carries=("yardline_100", lambda s: (s <= 20).sum()),
            gl_carries=("yardline_100", lambda s: (s <= 5).sum()))
        a.index.names = b.index.names = ["gsis_id", "season", "week"]
        red = a.join(b, how="outer").reset_index()
        red["season"], red["week"] = red["season"].astype(int), red["week"].astype(int)
        h = h.merge(red, on=["gsis_id", "season", "week"], how="left")
        del rz, gl
    for c in ("rz_targets", "rz_carries", "gl_carries", "pbp_air_yards"):
        if c not in h.columns:
            h[c] = np.nan
        else:
            h[c] = h[c].where(h["dk_points"].notna() | h[c].notna(), np.nan)

    played = h["dk_points"].notna() | (h["offense_snaps"].fillna(0) > 0)
    h["appeared"] = played.astype(float)
    h["dk_points"] = h["dk_points"].where(h["dk_points"].notna(),
                                          np.where(played, 0.0, np.nan))
    return h


def _calendar(season: int, games_tg: pd.DataFrame) -> pd.DataFrame:
    """(gsis_id, season, week) for every week a rostered offensive player's
    team had a game: the weeks a rolling window should count."""
    ro = _load("rosters", season)
    if ro.empty:
        return pd.DataFrame(columns=["gsis_id", "season", "week"])
    r = ro.dropna(subset=["gsis_id"]).copy()
    if "week" not in r.columns:
        return pd.DataFrame(columns=["gsis_id", "season", "week"])
    r = r[crosswalk.normalize_position(r["position"]).isin(OFFENSE)]
    r["team"] = crosswalk.normalize_team(r["team"])
    r["season"], r["week"] = r["season"].astype(int), r["week"].astype(int)
    tg = games_tg[["season", "week", "team"]].drop_duplicates()
    cal = r[["gsis_id", "season", "week", "team"]].merge(
        tg, on=["season", "week", "team"], how="inner")
    return cal[["gsis_id", "season", "week"]].drop_duplicates()


def _team_context(season: int) -> pd.DataFrame:
    pbp = _load("pbp", season)
    if pbp.empty:
        return pd.DataFrame()
    p = pbp[pbp["season_type"].isin(["REG", "POST"])
            & pbp["play_type"].isin(["pass", "run"])].copy()
    p["rz"] = (p["yardline_100"] <= 20).astype(float)
    p["is_pass"] = (p["play_type"] == "pass").astype(float)
    t = p.groupby(["posteam", "season", "week"], as_index=False).agg(
        team_plays=("is_pass", "size"), team_pass_rate=("is_pass", "mean"),
        team_rz_plays=("rz", "sum"))
    t = t.rename(columns={"posteam": "team"})
    t["team"] = crosswalk.normalize_team(t["team"])
    t["season"], t["week"] = t["season"].astype(int), t["week"].astype(int)
    return t


def _dst_history(season: int, games_tg: pd.DataFrame) -> pd.DataFrame:
    tw = _load("team_week", season)
    if tw.empty:
        return pd.DataFrame()
    t = tw[tw["season_type"].isin(["REG", "POST"])].copy()
    t["team"] = crosswalk.normalize_team(t["team"])
    t["season"], t["week"] = t["season"].astype(int), t["week"].astype(int)
    pa = games_tg[["season", "week", "team", "opponent", "points_against"]]
    t = t.merge(pa, on=["season", "week", "team"], how="inner")
    t["points_allowed"] = t["points_against"]
    t = t.dropna(subset=["points_allowed"])
    t["dk_points"] = score_dst(t)
    return t[["season", "week", "team", "opponent", "dk_points"]]


# --- pregame tables ------------------------------------------------------------

def _injuries(season: int) -> pd.DataFrame:
    inj = _load("injuries", season)
    if inj.empty:
        return pd.DataFrame(columns=["gsis_id", "season", "week"])
    i = inj.dropna(subset=["gsis_id"]).copy()
    status = i["report_status"].fillna("").astype(str).str.strip().str.lower()
    practice = i["practice_status"].fillna("").astype(str).str.strip().str.lower()
    i["injury_prob"] = status.map(INJURY_PROB).fillna(0.95)
    i["injury_code"] = status.map(INJURY_CODE).fillna(0)
    i["practice_code"] = practice.map(PRACTICE_CODE).fillna(0)
    i["season"], i["week"] = i["season"].astype(int), i["week"].astype(int)
    return (i.sort_values("injury_prob")
             .drop_duplicates(["gsis_id", "season", "week"])
             [["gsis_id", "season", "week", "injury_prob", "injury_code",
               "practice_code"]])


def _kickoffs(games: pd.DataFrame, season: int) -> pd.DataFrame:
    g = games[games["season"] == season].copy()
    when = pd.to_datetime(g["gameday"].dt.strftime("%Y-%m-%d") + " "
                          + g["gametime"].fillna("13:00").astype(str),
                          errors="coerce")
    g["kickoff"] = when.dt.tz_localize("America/New_York",
                                       ambiguous="NaT",
                                       nonexistent="NaT").dt.tz_convert("UTC")
    return g.groupby("week", as_index=False)["kickoff"].min()


def _depth(season: int, games: pd.DataFrame) -> pd.DataFrame:
    """Depth chart rank per player-week, using only what was published
    before that week kicked off. nflverse switched in 2025 from one row per
    player-week to dated snapshots; a snapshot taken after kickoff knows who
    started, so each snapshot is mapped to the next week not yet started."""
    dc = _load("depth", season)
    cols = ["gsis_id", "season", "week", "depth_team"]
    if dc.empty:
        return pd.DataFrame(columns=cols)
    if "week" in dc.columns and "depth_team" in dc.columns:
        d = dc.dropna(subset=["gsis_id", "week"]).copy()
        if "formation" in d.columns:
            d = d[d["formation"].isin(["Offense", "Special Teams"])]
        d["depth_team"] = pd.to_numeric(d["depth_team"], errors="coerce")
        d["season"], d["week"] = d["season"].astype(int), d["week"].astype(int)
        return d.groupby(["gsis_id", "season", "week"], as_index=False)[
            "depth_team"].min()
    if "dt" not in dc.columns:
        return pd.DataFrame(columns=cols)
    d = dc.dropna(subset=["gsis_id"]).copy()
    d = d[d["pos_abb"].isin(["QB", "RB", "WR", "TE", "FB", "PK", "K"])]
    d["dt"] = pd.to_datetime(d["dt"], utc=True, errors="coerce")
    ko = _kickoffs(games, season).dropna().sort_values("kickoff")
    if ko.empty:
        return pd.DataFrame(columns=cols)
    snaps = pd.Series(d["dt"].dropna().unique()).sort_values()
    rows = []
    for r in ko.itertuples():
        before = snaps[snaps < r.kickoff]
        if before.empty:
            continue
        sub = d[d["dt"] == before.iloc[-1]]
        g = sub.groupby("gsis_id", as_index=False)["pos_rank"].min()
        g["week"] = int(r.week)
        rows.append(g)
    if not rows:
        return pd.DataFrame(columns=cols)
    out = pd.concat(rows, ignore_index=True).rename(
        columns={"pos_rank": "depth_team"})
    out["season"] = season
    out["depth_team"] = pd.to_numeric(out["depth_team"], errors="coerce")
    return out[cols]


# --- assembly ----------------------------------------------------------------

def _salary_features(f: pd.DataFrame, within_slate: bool) -> pd.DataFrame:
    group = (["draft_group_id", "position"] if within_slate
             and "draft_group_id" in f.columns
             else ["season", "week", "position"])
    f["salary_pct"] = f.groupby(group)["salary"].rank(pct=True)
    tgroup = (["draft_group_id", "team", "position"] if within_slate
              and "draft_group_id" in f.columns
              else ["season", "week", "team", "position"])
    f["salary_team_rank"] = f.groupby(tgroup)["salary"].rank(
        ascending=False, method="first")
    f["value"] = f["salary"] / 1000.0
    for p in POSITIONS:
        f[f"pos_{p}"] = (f["position"] == p).astype(float)
    return f


def build(df: pd.DataFrame, salary_within_slate: bool = False,
          calendar_aware: bool = True, zero_absences: bool = False
          ) -> pd.DataFrame:
    """Every feature for every row of a slate frame (slates.assemble)."""
    if df.empty:
        return df.copy()
    f = df.copy().reset_index(drop=True)
    f["season"], f["week"] = f["season"].astype(int), f["week"].astype(int)
    f["position"] = crosswalk.normalize_position(f["position"])
    f["team"] = crosswalk.normalize_team(f["team"])
    if "gsis_id" not in f.columns:
        f["gsis_id"] = np.nan
    f["salary"] = pd.to_numeric(f["salary"], errors="coerce")
    f = _salary_features(f, salary_within_slate)

    games = _load("games")
    tg = _team_games(games)
    seasons = sorted(set(f["season"]))
    hist_seasons = sorted(set(seasons) | {s - 1 for s in seasons})

    # Vegas and venue: pregame, current week legal.
    ctx = tg[["season", "week", "team", "opponent", "is_home", "margin",
              "game_total", "rest_days", "roof", "temp", "wind", "div_game"]]
    ctx = ctx.drop_duplicates(["season", "week", "team"])
    f = f.drop(columns=[c for c in ("is_home", "game_total", "temp", "wind")
                        if c in f.columns])
    f = f.merge(ctx.rename(columns={"opponent": "_sched_opp"}),
                on=["season", "week", "team"], how="left")
    if "opponent" not in f.columns:
        f["opponent"] = f["_sched_opp"]
    else:
        f["opponent"] = f["opponent"].where(f["opponent"].notna()
                                            & (f["opponent"] != ""),
                                            f["_sched_opp"])
    f["spread"] = -f["margin"]
    f["implied_total"] = (f["game_total"] + f["margin"]) / 2
    f["opp_implied_total"] = (f["game_total"] - f["margin"]) / 2
    f["dome"] = f["roof"].isin(["dome", "closed"]).astype(float)
    f = f.drop(columns=["_sched_opp", "margin", "roof"])

    # Player history: lagged.
    hist = pd.concat([_player_history(s, tg) for s in hist_seasons],
                     ignore_index=True)
    people = f[f["position"] != "DST"]
    if not hist.empty:
        hist = _with_upcoming(hist, people)
        cal = None
        if calendar_aware:
            cal = pd.concat([_calendar(s, tg) for s in hist_seasons],
                            ignore_index=True)
            cal = pd.concat([cal, people.dropna(subset=["gsis_id"])
                             [["gsis_id", "season", "week"]]],
                            ignore_index=True).drop_duplicates()
        rolled = _roll(hist, ROLL_COLS, by="gsis_id", windows=WINDOWS,
                       calendar=cal, zero_absences=zero_absences)
        keep = ["gsis_id", "season", "week"] + [
            c for c in FEATURES if c in rolled.columns
            and (c.endswith(("_r3", "_r6")) or c == "weeks_since_played")]
        rolled = rolled[keep].drop_duplicates(["gsis_id", "season", "week"])
        f = f.merge(rolled, on=["gsis_id", "season", "week"], how="left")

        # Prior season: covers rookies' second year and every week-1 slate.
        seen = hist[hist["appeared"] > 0]
        prior = seen.groupby(["gsis_id", "season"]).agg(
            prior_ppg=("dk_points", "mean"), prior_best=("dk_points", "max"),
            prior_games=("dk_points", "size"),
            prior_targets_pg=("targets", "mean"),
            prior_carries_pg=("carries", "mean")).reset_index()
        prior["season"] = prior["season"] + 1
        f = f.merge(prior, on=["gsis_id", "season"], how="left")

        # Opponent defense: DK points it allowed to this position, lagged.
        allow = seen.dropna(subset=["opponent"]).groupby(
            ["opponent", "position", "season", "week"], as_index=False)[
            "dk_points"].sum().rename(columns={"opponent": "def_team"})
    else:
        allow = pd.DataFrame(columns=["def_team", "position", "season", "week",
                                      "dk_points"])

    # DST: team defense form, and what each offense has allowed to DSTs.
    dst = pd.concat([_dst_history(s, tg) for s in hist_seasons],
                    ignore_index=True)
    if not dst.empty:
        d = dst.rename(columns={"team": "gsis_id"}).assign(appeared=1.0)
        d = _with_upcoming(d, f[f["position"] == "DST"].assign(
            gsis_id=lambda x: x["team"]))
        dr = _roll(d, ["dk_points"], by="gsis_id", windows=WINDOWS)
        dr = dr[["gsis_id", "season", "week", "dk_points_r3", "dk_points_r6"]]
        dr = dr.drop_duplicates(["gsis_id", "season", "week"]).rename(
            columns={"gsis_id": "team"})
        is_dst = f["position"] == "DST"
        m = f.loc[is_dst, ["team", "season", "week"]].merge(
            dr, on=["team", "season", "week"], how="left")
        for c in ("dk_points_r3", "dk_points_r6"):
            if c not in f.columns:
                f[c] = np.nan
            f.loc[is_dst, c] = m[c].to_numpy()
        for c in ("appeared_r3", "appeared_r6"):
            if c in f.columns:
                f.loc[is_dst, c] = 1.0
        vs = dst.dropna(subset=["opponent"]).rename(
            columns={"opponent": "def_team"})[["def_team", "season", "week",
                                                "dk_points"]]
        allow = pd.concat([allow, vs.assign(position="DST")], ignore_index=True)

    if not allow.empty:
        a = allow.rename(columns={"def_team": "gsis_id"})
        a["_k"] = a["gsis_id"].astype(str) + "|" + a["position"].astype(str)
        up = f[["opponent", "position", "season", "week"]].dropna().copy()
        up["_k"] = up["opponent"].astype(str) + "|" + up["position"].astype(str)
        a2 = _with_upcoming(a[["_k", "season", "week", "dk_points"]],
                            up[["_k", "season", "week"]], by="_k")
        ar = _roll(a2, ["dk_points"], by="_k", windows=WINDOWS)
        ar = ar.rename(columns={"dk_points_r3": "def_allowed_r3",
                                "dk_points_r6": "def_allowed_r6"})
        ar = ar[["_k", "season", "week", "def_allowed_r3", "def_allowed_r6"]]
        ar = ar.drop_duplicates(["_k", "season", "week"])
        f["_k"] = f["opponent"].astype(str) + "|" + f["position"].astype(str)
        f = f.merge(ar, on=["_k", "season", "week"], how="left").drop(
            columns="_k")

    # Team play volume: lagged.
    team = pd.concat([_team_context(s) for s in hist_seasons], ignore_index=True)
    if not team.empty:
        t = team.rename(columns={"team": "gsis_id"})
        t = _with_upcoming(t, f[["team", "season", "week"]].rename(
            columns={"team": "gsis_id"}))
        tr = _roll(t, ["team_plays", "team_pass_rate", "team_rz_plays"],
                   by="gsis_id", windows=(3,))
        tr = tr[["gsis_id", "season", "week", "team_plays_r3",
                 "team_pass_rate_r3", "team_rz_plays_r3"]].rename(
            columns={"gsis_id": "team"}).drop_duplicates(
            ["team", "season", "week"])
        f = f.merge(tr, on=["team", "season", "week"], how="left")

    # Injury report and depth chart: pregame.
    inj = pd.concat([_injuries(s) for s in seasons], ignore_index=True)
    if not inj.empty:
        f = f.merge(inj, on=["gsis_id", "season", "week"], how="left")
        matched = f["gsis_id"].notna()
        f.loc[matched, "injury_prob"] = f.loc[matched, "injury_prob"].fillna(1.0)
        for c in ("injury_code", "practice_code"):
            f.loc[matched, c] = f.loc[matched, c].fillna(0)
    dep = pd.concat([_depth(s, games) for s in seasons], ignore_index=True)
    if not dep.empty:
        f = f.merge(dep, on=["gsis_id", "season", "week"], how="left")
        f["is_starter"] = (f["depth_team"] == 1).astype(float).where(
            f["depth_team"].notna())

    for c in FEATURES:
        if c not in f.columns:
            f[c] = np.nan
        f[c] = pd.to_numeric(f[c], errors="coerce").astype(float)
    return f

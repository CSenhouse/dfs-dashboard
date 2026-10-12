"""RotoGuru: DraftKings salaries and DK's own scored points, 2011-2021 only.

    http://rotoguru1.com/cgi-bin/fyday.pl?week={W}&year={Y}&game=dk&scsv=1

HTTP only -- port 443 refuses connections. Semicolon-delimited inside <pre>.
2022 onward returns no rows. Its value here is the "DK points" column: an
independent check on the scoring engine (scripts/validate_scoring.py).
"""
from __future__ import annotations

import io
import re

import pandas as pd

from dfs import http
from dfs.crosswalk import normalize_team

URL = "http://rotoguru1.com/cgi-bin/fyday.pl?week={week}&year={year}&game=dk&scsv=1"

TEAMS = {
    "ari": "ARI", "atl": "ATL", "bal": "BAL", "buf": "BUF", "car": "CAR",
    "chi": "CHI", "cin": "CIN", "cle": "CLE", "dal": "DAL", "den": "DEN",
    "det": "DET", "gnb": "GB", "hou": "HOU", "ind": "IND", "jac": "JAX",
    "kan": "KC", "lac": "LAC", "lar": "LA", "lvr": "LV", "mia": "MIA",
    "min": "MIN", "nor": "NO", "nwe": "NE", "nyg": "NYG", "nyj": "NYJ",
    "oak": "LV", "phi": "PHI", "pit": "PIT", "sdg": "LAC", "sea": "SEA",
    "sfo": "SF", "stl": "LA", "tam": "TB", "ten": "TEN", "was": "WAS",
}

_PRE = re.compile(r"<pre>(.*?)</pre>", re.S | re.I)


def _flip(name: str) -> str:
    """'Brady, Tom' -> 'Tom Brady'."""
    if "," in name:
        last, first = [x.strip() for x in name.split(",", 1)]
        return f"{first} {last}".strip()
    return name.strip()


def fetch_week(year: int, week: int) -> pd.DataFrame:
    body = http.get(URL.format(week=week, year=year), ".html", min_interval=1.0)
    if not body:
        return pd.DataFrame()
    m = _PRE.search(body.decode("latin-1"))
    if not m or ";" not in m.group(1):
        return pd.DataFrame()
    df = pd.read_csv(io.StringIO(m.group(1).strip()), sep=";")
    if df.empty:
        return df
    df.columns = [c.strip() for c in df.columns]
    out = pd.DataFrame({
        "season": pd.to_numeric(df["Year"], errors="coerce"),
        "week": pd.to_numeric(df["Week"], errors="coerce"),
        "player_name": df["Name"].astype(str).map(_flip),
        "position": df["Pos"].astype(str).str.strip().replace(
            {"Def": "DST", "PK": "K"}),
        "team": normalize_team(df["Team"].astype(str).str.lower().map(TEAMS)
                               .fillna(df["Team"].astype(str))),
        "dk_points_rotoguru": pd.to_numeric(df["DK points"], errors="coerce"),
        "salary": pd.to_numeric(df["DK salary"], errors="coerce"),
    })
    return out.dropna(subset=["season", "week"])


def fetch_season(year: int, weeks: range | None = None) -> pd.DataFrame:
    weeks = weeks or range(1, 19)
    frames = [fetch_week(year, w) for w in weeks]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["season"] = out["season"].astype(int)
    out["week"] = out["week"].astype(int)
    return out

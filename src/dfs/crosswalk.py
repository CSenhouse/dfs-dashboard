"""DraftKings player -> nflverse gsis_id.

Rule 1: never join on player names alone. A name-only join paired the 2019
Saints WR Michael Thomas with a same-named defensive back and scored his
44-point week as 0.00. Every pass here keys on season, team and position as
well, and a key that maps to more than one player resolves to NEITHER rather
than to whichever row happened to come first.
"""
from __future__ import annotations

import pandas as pd

SUFFIX = r"\b(jr|sr|ii|iii|iv|v)\b\.?"

TEAM_ALIASES = {
    "OAK": "LV", "LVR": "LV", "SD": "LAC", "STL": "LA", "LAR": "LA",
    "WSH": "WAS", "JAC": "JAX", "ARZ": "ARI", "BLT": "BAL", "CLV": "CLE",
    "HST": "HOU", "GNB": "GB", "KAN": "KC", "NWE": "NE", "NOR": "NO",
    "SFO": "SF", "TAM": "TB", "SL": "LA",
}

NFL_TEAMS = frozenset({
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN",
    "DET", "GB", "HOU", "IND", "JAX", "KC", "LA", "LAC", "LV", "MIA", "MIN",
    "NE", "NO", "NYG", "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS",
})

# nflverse roster positions that DraftKings prices under another label.
POSITION_ALIASES = {"FB": "RB", "HB": "RB", "PK": "K", "DEF": "DST", "D": "DST"}


def normalize_name(s: pd.Series) -> pd.Series:
    """'A.J. Brown' -> 'aj brown'; 'Michael Woods II' -> 'michael woods'."""
    return (
        s.fillna("").astype(str)
        .str.normalize("NFKD")
        .str.encode("ascii", "ignore")
        .str.decode("ascii")
        .str.lower()
        .str.replace(SUFFIX, " ", regex=True)
        .str.replace(r"[^a-z ]", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def normalize_team(s: pd.Series) -> pd.Series:
    t = s.fillna("").astype(str).str.strip().str.upper()
    return t.replace(TEAM_ALIASES)


def normalize_position(s: pd.Series) -> pd.Series:
    p = s.fillna("").astype(str).str.strip().str.upper()
    return p.replace(POSITION_ALIASES)


def initial_last(key_name: pd.Series) -> pd.Series:
    """'joshua palmer' and 'josh palmer' both -> 'j palmer'."""
    parts = key_name.fillna("").str.split()
    return parts.apply(lambda p: f"{p[0][0]} {p[-1]}" if len(p) >= 2 else
                       (p[0] if p else ""))


def _attempt(dk: pd.DataFrame, roster: pd.DataFrame, keys: list[str],
             tag: str) -> pd.DataFrame:
    """Match DK rows to roster ids on `keys`, refusing any ambiguity.

    A key that resolves to more than one gsis_id on the roster side, or that
    is claimed by more than one DK player, matches nobody.
    """
    if dk.empty or roster.empty:
        return pd.DataFrame(columns=["player_dk_id", "gsis_id", "matched_on"])
    r = roster.dropna(subset=["gsis_id"])[keys + ["gsis_id"]].drop_duplicates()
    r = r[(r[keys] != "").all(axis=1)]
    counts = r.groupby(keys)["gsis_id"].transform("nunique")
    r = r[counts == 1].drop_duplicates(subset=keys)

    d = dk[keys + ["player_dk_id"]].drop_duplicates()
    claims = d.groupby(keys)["player_dk_id"].transform("nunique")
    d = d[claims == 1]

    hit = d.merge(r, on=keys, how="inner")
    hit = hit.drop_duplicates(subset=["player_dk_id"])
    hit["matched_on"] = tag
    return hit[["player_dk_id", "gsis_id", "matched_on"]].reset_index(drop=True)


def _prepare_dk(dk: pd.DataFrame) -> pd.DataFrame:
    out = dk.copy()
    out["team"] = normalize_team(out["team"])
    out["position"] = normalize_position(out["position"])
    out["key_name"] = normalize_name(out["player_name"])
    out["key_init"] = initial_last(out["key_name"])
    return out


def _prepare_roster(roster: pd.DataFrame) -> pd.DataFrame:
    out = roster.copy()
    out["team"] = normalize_team(out["team"])
    out["position"] = normalize_position(out["position"])
    out["key_name"] = normalize_name(out["full_name"])
    out["key_init"] = initial_last(out["key_name"])
    return out


PASSES = (
    (["season", "team", "position", "key_name"], "name"),
    (["season", "team", "position", "key_init"], "initial"),
    (["season", "position", "key_name"], "name, any team"),
    (["season", "team", "key_name"], "name, any position"),
)


def build(dk: pd.DataFrame, roster: pd.DataFrame) -> pd.DataFrame:
    """player_dk_id -> gsis_id for one season of DK players.

    `dk` needs season, player_dk_id, player_name, position, team. `roster`
    needs season, team, position, full_name, gsis_id. DSTs are not people and
    are skipped; they key on team instead.
    """
    if dk.empty or roster.empty:
        return pd.DataFrame(columns=["player_dk_id", "gsis_id", "matched_on"])
    d = _prepare_dk(dk)
    d = d[d["position"] != "DST"]
    r = _prepare_roster(roster)
    found: list[pd.DataFrame] = []
    left = d
    for keys, tag in PASSES:
        if left.empty:
            break
        hit = _attempt(left, r, keys, tag)
        if not hit.empty:
            # A gsis_id already taken by an earlier, stricter pass is not
            # available to a looser one.
            taken = set().union(*[set(f["gsis_id"]) for f in found]) if found \
                else set()
            hit = hit[~hit["gsis_id"].isin(taken)]
            found.append(hit)
            left = left[~left["player_dk_id"].isin(hit["player_dk_id"])]
    if not found:
        return pd.DataFrame(columns=["player_dk_id", "gsis_id", "matched_on"])
    out = pd.concat(found, ignore_index=True)
    return out.drop_duplicates(subset=["player_dk_id"]).reset_index(drop=True)

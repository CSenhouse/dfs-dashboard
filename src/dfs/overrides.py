"""Hand edits to a projection, logged as an experiment.

Lineups are the wrong unit to learn from: a 25-point weekly sd needs ~70
weeks to settle a 5-point edge. One edit against one actual score is a
binomial that reaches p<0.05 at about 60 edits, so every edit is written down
with the model's own number beside it (scripts/override_value.py reads it).

Rule 13: an override moves the whole distribution, not the column on screen.
The optimizer reads proj_*, the simulator reads the quantile grid, so one
factor moves every points-like column and p_play is left alone -- an override
says what a player is worth, not whether he suits up. Below MIN_BASE the edit
shifts instead of scaling, because a ratio against nothing is noise.
"""
from __future__ import annotations

import datetime as dt
import re

import numpy as np
import pandas as pd

from dfs.config import DATA

DIR = DATA / "overrides"
MAX_RATIO = 20.0
MIN_BASE = 1.0
FIELDS = ["draft_group_id", "player_dk_id", "player_name", "view",
          "model_value", "user_value", "cleared", "recorded_at"]
_POINTS = re.compile(r"^(q\d\d|points_\w+|proj_\w+)$")


def path(dgid: int):
    return DIR / f"{int(dgid)}.csv"


def _raw(dgid: int) -> pd.DataFrame:
    p = path(dgid)
    if not p.exists():
        return pd.DataFrame(columns=FIELDS)
    df = pd.read_csv(p)
    for c in FIELDS:
        if c not in df.columns:
            df[c] = False if c == "cleared" else np.nan
    return df


def _append(dgid: int, rows: pd.DataFrame) -> None:
    DIR.mkdir(parents=True, exist_ok=True)
    p = path(dgid)
    rows = rows[FIELDS]
    rows.to_csv(p, mode="a", header=not p.exists(), index=False)


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def record(dgid: int, edits: pd.DataFrame) -> int:
    """Append edits that actually change something. Returns rows written."""
    if edits is None or edits.empty:
        return 0
    e = edits.copy()
    moved = (e["user_value"].astype(float) - e["model_value"].astype(float)
             ).abs() > 0.005
    e = e[moved]
    if e.empty:
        return 0
    e["draft_group_id"] = int(dgid)
    e["cleared"] = False
    e["recorded_at"] = _now()
    _append(dgid, e)
    return len(e)


def load(dgid: int) -> pd.DataFrame:
    """The overrides in force: the latest entry per player and view."""
    raw = _raw(dgid)
    if raw.empty:
        return pd.DataFrame(columns=[c for c in FIELDS if c != "cleared"])
    raw["_order"] = np.arange(len(raw))
    last = raw.sort_values("_order").groupby(
        ["player_dk_id", "view"], as_index=False).tail(1)
    cleared = last["cleared"].astype(str).str.lower().isin(("true", "1"))
    last = last[~cleared].drop(columns=["_order", "cleared"])
    last["player_dk_id"] = last["player_dk_id"].astype(int)
    last["draft_group_id"] = last["draft_group_id"].astype(int)
    return last.sort_values("player_dk_id").reset_index(drop=True)


def forget(dgid: int, player_ids, view: str) -> int:
    """Drop the override in force for these players on this view. History
    stays on disk. Returns how many were in force."""
    live = load(dgid)
    if live.empty:
        return 0
    want = {int(x) for x in player_ids}
    hit = live[(live["view"] == view) & live["player_dk_id"].isin(want)]
    if hit.empty:
        return 0
    marks = hit.copy()
    marks["user_value"] = np.nan
    marks["cleared"] = True
    marks["recorded_at"] = _now()
    _append(dgid, marks)
    return len(hit)


def apply(pool: pd.DataFrame, edits: pd.DataFrame | None, view: str
          ) -> pd.DataFrame:
    """The pool with every edit for `view` applied to the whole distribution."""
    if edits is None or edits.empty:
        return pool
    e = edits[edits["view"] == view] if "view" in edits.columns else edits
    if e.empty or view not in pool.columns:
        return pool
    out = pool.copy()
    cols = [c for c in out.columns if _POINTS.match(c)]
    want = e.groupby("player_dk_id")["user_value"].last()
    for pid, target in want.items():
        rows = out.index[out["player_dk_id"] == pid]
        if len(rows) == 0 or pd.isna(target):
            continue
        for r in rows:
            base = float(out.at[r, view])
            if not np.isfinite(base):
                continue
            vals = out.loc[r, cols].astype(float)
            if base >= MIN_BASE:
                factor = min(max(float(target) / base, 0.0), MAX_RATIO)
                vals = vals * factor
            else:
                vals = vals + (float(target) - base)
            out.loc[r, cols] = vals.clip(lower=0.0)
    return out

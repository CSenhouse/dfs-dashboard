"""Pre-lock snapshots of a slate's projections.

The method tracker (scripts/track_methods.py) used to rebuild each week's
lineups after the games. By then the features carry the week's final injury
report and refreshed nflverse files, so the rebuild is not what the dashboard
showed: on week 9 of 2026 the rebuilt Model cash lineup held Kyren Williams,
Kittle and Caleb Douglas where the live one held Lamb, Hubbard and Dulcich,
and the tracker was scoring lineups nobody could have built.

A snapshot is the dashboard's own numbers, written while the slate is still
live and overwritten until lock, so the last one written IS the pre-lock
state. It stores the model's views and, instead of DFF's raw projection
(their product, kept out of git), the factor the 50/50 blend applies:
blended view = model view x (0.5 + 0.5 x dff / model mean), exactly what
`overrides.apply` does for the dashboard's Blend control.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from dfs.config import DATA

DIR = DATA / "snapshots"
VIEWS = ("proj_floor", "proj_mean", "proj_ceiling", "proj_hundo")
KEEP = (["season", "week", "draft_group_id", "player_dk_id", "player_name",
         "position", "team", "opponent", "competition_id", "salary",
         "p_play"] + list(VIEWS)
        + ["blend_factor", "own_double_up", "own_tournament", "own_hundo"])


def path(dgid: int):
    return DIR / f"dg{int(dgid)}.csv"


def blend_factor(cur: pd.DataFrame, matched: pd.DataFrame,
                 weight: float = 0.5) -> pd.Series:
    """Per player, what the blend multiplies every view by; 1.0 where DFF
    has no row. `matched` is dff.match(...) restricted to matched rows."""
    dff_proj = cur["player_dk_id"].map(
        matched.drop_duplicates("player_dk_id")
        .set_index("player_dk_id")["dff_proj"])
    mean = cur["proj_mean"].clip(lower=1e-6)
    f = (1 - weight) + weight * dff_proj / mean
    return f.where(dff_proj.notna() & np.isfinite(f), 1.0)


def take(cur: pd.DataFrame, factor: pd.Series | None = None) -> pd.DataFrame:
    snap = cur.copy()
    snap["blend_factor"] = 1.0 if factor is None else factor.to_numpy()
    for c in KEEP:
        if c not in snap.columns:
            snap[c] = np.nan
    snap = snap[KEEP]
    snap["snapshot_at"] = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    return snap


def save(dgid: int, snap: pd.DataFrame) -> None:
    DIR.mkdir(parents=True, exist_ok=True)
    snap.round(4).to_csv(path(dgid), index=False, lineterminator="\n")


def age_seconds(dgid: int) -> float | None:
    p = path(dgid)
    if not p.exists():
        return None
    return dt.datetime.now().timestamp() - p.stat().st_mtime


def load(dgid: int) -> pd.DataFrame | None:
    p = path(dgid)
    return pd.read_csv(p) if p.exists() else None


def with_method(snap: pd.DataFrame, method: str) -> pd.DataFrame:
    """The snapshot as the Model or the Blend control would show it."""
    if method == "model":
        return snap.copy()
    out = snap.copy()
    for c in VIEWS:
        out[c] = out[c] * out["blend_factor"]
    return out

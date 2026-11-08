"""Projected ownership: what share of a contest's field will roster a player.

Ownership is the crowd's aggregated pre-lock decision, and it is far more
predictable than points: the crowd reads the same free projections, chases
the same price-to-projection value, and in a double-up converges on a handful
of names. The only source of truth is the real ownership in a standings
export (data/ownership/<contest>.csv), so the model is trained on those and
can only ever be as good as the slates we have counted.

Built per (slate, contest type), because the same slate is owned very
differently by a double-up field (top 10 filling 5-6 of 9 slots) and a
tournament field. A Classic lineup has nine players, so a contest's
ownership sums to 9.0; predictions are rescaled to honour that, which makes
the model learn who is chalk RELATIVE to the slate rather than how chalky
slates are in general.

Everything here is pregame: salary, our projection, DFF's projection, Vegas.
Nothing reads DK's injury status from a cached payload (rule 8): p_play from
the model carries availability, as it does for points.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROSTER = 9.0
TYPES = ("double_up", "tournament", "hundo")
POSITIONS = ("QB", "RB", "WR", "TE", "DST")

FEATURES = (
    ["salary", "p_play", "proj", "value", "dff_proj", "dff_value",
     "implied_total", "spread"]
    + [f"pos_{p}" for p in POSITIONS]
    + ["salary_rank_pos", "proj_rank_pos", "value_rank_pos",
       "dff_value_rank_pos", "dff_proj_rank_pos", "value_rank_all",
       "dff_value_rank_all"]
    + [f"type_{t}" for t in TYPES]
)
# Without DFF: what our own numbers can explain alone.
MODEL_ONLY = [f for f in FEATURES if not f.startswith("dff_")
              and f != "dff_value_rank_all"]
SALARY_ONLY = (["salary", "salary_rank_pos"] + [f"pos_{p}" for p in POSITIONS]
               + [f"type_{t}" for t in TYPES])


def contest_type(paid_pct: float, entries: int) -> str:
    """Double-ups pay 43.5%; a Hundo is 100 entries paying 22; the rest are
    tournaments paying about a quarter of the field."""
    if entries <= 100:
        return "hundo"
    return "double_up" if paid_pct >= 0.40 else "tournament"


def slate_features(players: pd.DataFrame) -> pd.DataFrame:
    """Per-player features, computed WITHIN the slate.

    `players` needs salary, position, p_play, proj_mean, and optionally
    dff_proj, implied_total and spread. Ranks are 1 = best at the position.
    """
    f = players.copy()
    f["proj"] = f["proj_mean"]
    f["value"] = 1000 * f["proj"] / f["salary"]
    if "dff_proj" not in f.columns:
        f["dff_proj"] = np.nan
    f["dff_value"] = 1000 * f["dff_proj"] / f["salary"]
    for c in ("implied_total", "spread"):
        if c not in f.columns:
            f[c] = np.nan
    for p in POSITIONS:
        f[f"pos_{p}"] = (f["position"] == p).astype(int)
    by_pos = f.groupby("position")
    f["salary_rank_pos"] = by_pos["salary"].rank(ascending=False)
    f["proj_rank_pos"] = by_pos["proj"].rank(ascending=False)
    f["value_rank_pos"] = by_pos["value"].rank(ascending=False)
    f["dff_value_rank_pos"] = by_pos["dff_value"].rank(ascending=False)
    f["dff_proj_rank_pos"] = by_pos["dff_proj"].rank(ascending=False)
    f["value_rank_all"] = f["value"].rank(ascending=False)
    f["dff_value_rank_all"] = f["dff_value"].rank(ascending=False)
    return f


def with_type(f: pd.DataFrame, kind: str) -> pd.DataFrame:
    out = f.copy()
    for t in TYPES:
        out[f"type_{t}"] = int(t == kind)
    out["contest_type"] = kind
    return out


def fit(train: pd.DataFrame, cols: list[str] = FEATURES
        ) -> HistGradientBoostingRegressor:
    """Fit on sqrt(ownership): it is bounded below by zero and a few chalk
    players carry most of the mass, which squared error on the raw share
    would let dominate. Seeded, per rule 2."""
    m = HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
        min_samples_leaf=20, l2_regularization=1.0, random_state=0)
    m.fit(train[cols], np.sqrt(train["own"].clip(lower=0)))
    m.cols_ = list(cols)
    return m


def predict(model, f: pd.DataFrame, group: list[str] | None = None
            ) -> pd.Series:
    """Projected ownership, rescaled so each group sums to nine roster
    spots. `group` names the columns that identify one contest field
    (default: contest_type)."""
    raw = np.clip(model.predict(f[model.cols_]), 0, None) ** 2
    s = pd.Series(raw, index=f.index)
    keys = group or ["contest_type"]
    total = s.groupby([f[k] for k in keys]).transform("sum")
    return (ROSTER * s / total.replace(0, np.nan)).fillna(0.0)




def load_model():
    """The model fitted by scripts/ownership_model.py, or None."""
    import pickle

    from dfs.config import CURATED
    path = CURATED / "ownership_model.pkl"
    if not path.exists():
        return None
    return pickle.loads(path.read_bytes())


def project(model, players: pd.DataFrame) -> pd.DataFrame:
    """Projected ownership for a live Classic slate, one column per contest
    type: own_double_up, own_tournament, own_hundo. `players` is the
    dashboard's projected pool -- the MODEL's proj_mean, not a blended or
    overridden one, since the model was trained on unblended numbers -- with
    dff_proj where DFF matched."""
    base = slate_features(players)
    out = players[["player_dk_id"]].copy()
    for kind in TYPES:
        out[f"own_{kind}"] = predict(model, with_type(base, kind)).to_numpy()
    return out

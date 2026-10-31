"""Two-stage projection: P(plays) x E[points | plays], gradient boosting.

54% of players priced on a main slate never record a stat line. A single
regression has to learn "will he play" and "how much" through one head and
mostly learns the first, so participation gets its own classifier and the
points model trains only on players who played (docs/MODELING.md 1).

Distributional views come from quantile regressors on the same rows: the
floor (q30) for cash, the ceiling (q85) for tournaments, and a full grid for
the simulator. All views share ONE participation estimate.

The boosting is scikit-learn's HistGradientBoosting. It was LightGBM until
Windows Smart App Control began enforcing (2026-10-25) and blocked LightGBM's
unsigned native library; scikit-learn's histogram boosting is the same family
-- binned features, leaf-wise trees, native missing-value handling, quantile
loss -- and loads under that policy. It is seeded (rule 2): comparisons are
paired, and an unseeded model would make them measure noise.
"""
from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor)
from sklearn.isotonic import IsotonicRegression

from dfs.features import feature_columns
from dfs.simulate import ALPHAS, QCOLS

SEED = 0
MIN_ROWS = 100           # played rows needed to fit anything
FLOOR_Q, CEILING_Q = 0.30, 0.85
# Hundos (100-entry, top 22 paid): an upside view between the two. Over 93
# walk-forward weeks floor builds returned 0.38 less per $1 than q70/q85
# builds (p=0.04); see scripts/hundo_view.py and docs/MODELING.md 36.
HUNDO_Q = 0.70

BASE = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
            min_samples_leaf=30, l2_regularization=1.0, max_features=0.8,
            early_stopping=False, random_state=SEED)

warnings.filterwarnings("ignore", message=".*X does not have valid feature names.*")


def _cols(df: pd.DataFrame, exclude=()) -> list[str]:
    cols = feature_columns(df)
    if exclude:
        cols = [c for c in cols if not any(m in c for m in exclude)]
    return cols


def _X(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df[cols].astype(float)


def _enough(past: pd.DataFrame) -> bool:
    if past is None or past.empty or "played" not in past.columns:
        return False
    played = past["played"].astype(bool)
    return played.sum() >= MIN_ROWS and (~played).sum() >= 5


def _classifier(past, cols):
    clf = HistGradientBoostingClassifier(**BASE)
    clf.fit(_X(past, cols), past["played"].astype(int))
    return clf


def _regressor(past, cols, quantile: float | None = None):
    """The conditional points model; `quantile=a` fits that quantile."""
    rows = past[past["played"].astype(bool)]
    extra = ({"loss": "quantile", "quantile": quantile}
             if quantile is not None else {})
    reg = HistGradientBoostingRegressor(**{**BASE, **extra})
    reg.fit(_X(rows, cols), rows["dk_scored"].astype(float))
    return reg


def _p_play(clf, cur, cols) -> np.ndarray:
    return clf.predict_proba(_X(cur, cols))[:, 1]


def _quantiles(past, cur, cols, alphas) -> np.ndarray:
    out = np.column_stack([
        _regressor(past, cols, quantile=a).predict(_X(cur, cols))
        for a in alphas])
    return np.maximum.accumulate(np.clip(out, 0.0, None), axis=1)


# --- point projections (walk-forward signatures: fn(past, cur) -> Series) ----

def _empty(cur) -> pd.Series:
    return pd.Series(np.nan, index=cur.index)


def two_stage(past: pd.DataFrame, cur: pd.DataFrame, exclude=()) -> pd.Series:
    if not _enough(past):
        return _empty(cur)
    cols = _cols(past, exclude)
    p = _p_play(_classifier(past, cols), cur, cols)
    pts = np.clip(_regressor(past, cols).predict(_X(cur, cols)), 0, None)
    return pd.Series(p * pts, index=cur.index)


def _two_stage_q(past, cur, alpha, exclude=()) -> pd.Series:
    if not _enough(past):
        return _empty(cur)
    cols = _cols(past, exclude)
    p = _p_play(_classifier(past, cols), cur, cols)
    q = _quantiles(past, cur, cols, [alpha])[:, 0]
    return pd.Series(p * q, index=cur.index)


def two_stage_floor(past, cur, exclude=()) -> pd.Series:
    return _two_stage_q(past, cur, FLOOR_Q, exclude)


def two_stage_ceiling(past, cur, exclude=()) -> pd.Series:
    return _two_stage_q(past, cur, CEILING_Q, exclude)


def single_stage(past, cur, exclude=()) -> pd.Series:
    """One regression on every row, zeros included -- the thing the split
    exists to beat."""
    if not _enough(past):
        return _empty(cur)
    cols = _cols(past, exclude)
    reg = HistGradientBoostingRegressor(**BASE)
    reg.fit(_X(past, cols), past["dk_scored"].astype(float))
    return pd.Series(np.clip(reg.predict(_X(cur, cols)), 0, None),
                     index=cur.index)


def two_stage_calibrated(past, cur, exclude=(), holdout_weeks: int = 8
                         ) -> pd.Series:
    """two_stage passed through an isotonic predicted->actual map learned
    out-of-sample on the last weeks of `past`. In-sample predictions are
    optimistically accurate and would teach the wrong map (section 5)."""
    raw = two_stage(past, cur, exclude)
    key = past["season"] * 100 + past["week"]
    weeks = sorted(key.unique())
    if len(weeks) < holdout_weeks * 2:
        return raw
    cut = weeks[-holdout_weeks]
    fit_on, hold = past[key < cut], past[key >= cut]
    pred = two_stage(fit_on, hold, exclude)
    ok = pred.notna()
    if ok.sum() < 50:
        return raw
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(pred[ok].to_numpy(), hold.loc[ok, "dk_scored"].to_numpy())
    return pd.Series(iso.predict(raw.fillna(0).to_numpy()), index=cur.index)


# --- distributions -----------------------------------------------------------

_QPAT = re.compile(r"^q(\d\d)$")


def quantile_columns(df: pd.DataFrame) -> list[str]:
    """Quantile-grid columns present in `df`, in increasing order."""
    got = [c for c in df.columns if _QPAT.match(str(c))]
    return sorted(got, key=lambda c: int(c[1:]))


def two_stage_all(past: pd.DataFrame, cur: pd.DataFrame, exclude=()
                  ) -> pd.DataFrame:
    """Every view from one fit: p_play, the quantile grid, and floor / mean /
    ceiling both conditional (points_*) and unconditional (proj_*).

    Separately fitted quantiles cross for about one player in twenty, so the
    grid is made monotone and the views are pinned in order around the mean.
    """
    if not _enough(past):
        return pd.DataFrame()
    cols = _cols(past, exclude)
    p = _p_play(_classifier(past, cols), cur, cols)
    grid = _quantiles(past, cur, cols, ALPHAS)
    mean = np.clip(_regressor(past, cols).predict(_X(cur, cols)), 0, None)

    out = pd.DataFrame({"player_dk_id": cur["player_dk_id"].to_numpy(),
                        "p_play": np.clip(p, 0.0, 1.0)})
    for k, c in enumerate(QCOLS):
        out[c] = grid[:, k]
    floor = grid[:, ALPHAS.index(FLOOR_Q)]
    ceiling = grid[:, ALPHAS.index(CEILING_Q)]
    out["points_mean"] = mean
    out["points_floor"] = np.minimum(floor, mean)
    out["points_ceiling"] = np.maximum(ceiling, mean)
    out["points_hundo"] = np.clip(grid[:, ALPHAS.index(HUNDO_Q)],
                                  out["points_floor"], out["points_ceiling"])
    for view in ("floor", "mean", "ceiling", "hundo"):
        out[f"proj_{view}"] = out["p_play"] * out[f"points_{view}"]
    return out


def quantile_grid(past: pd.DataFrame, cur: pd.DataFrame) -> pd.DataFrame:
    """two_stage_all plus `projection` (the mean view), keyed on player_dk_id
    -- the walk-forward input to the simulator's calibration scripts."""
    out = two_stage_all(past, cur)
    if out.empty:
        return out
    out["projection"] = out["proj_mean"]
    return out


# --- explanation ---------------------------------------------------------------

def _contributions(reg, X: pd.DataFrame, reference: pd.Series
                   ) -> tuple[np.ndarray, np.ndarray]:
    """(per-feature contributions, baseline), exactly additive.

    Start every row from a reference player (the training medians) and swap
    in one real feature value at a time, crediting each feature with the
    change in prediction. The steps telescope, so contributions plus the
    baseline equal the prediction exactly. The credit depends on the order,
    so it is averaged over the forward and the reverse order -- still exact,
    because each order is.
    """
    n, k = X.shape
    ref = np.tile(reference.to_numpy(dtype=float), (n, 1))
    real = X.to_numpy(dtype=float)
    base = reg.predict(pd.DataFrame(ref, columns=X.columns))
    out = np.zeros((n, k))
    for order in (range(k), range(k - 1, -1, -1)):
        z = ref.copy()
        prev = base
        for j in order:
            z[:, j] = real[:, j]
            now = reg.predict(pd.DataFrame(z, columns=X.columns))
            out[:, j] += (now - prev) / 2.0
            prev = now
    return out, base


def explain(past: pd.DataFrame, cur: pd.DataFrame
            ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(summary, contributions). Contributions plus `_baseline` sum exactly to
    points_if_plays; an explanation that does not reconcile with its number
    invites trust it has not earned."""
    if not _enough(past):
        return pd.DataFrame(), pd.DataFrame()
    cols = _cols(past)
    p = _p_play(_classifier(past, cols), cur, cols)
    reg = _regressor(past, cols)
    X = _X(cur, cols)
    played = past[past["played"].astype(bool)]
    contrib, base = _contributions(reg, X, _X(played, cols).median())
    pts = reg.predict(X)
    summary = pd.DataFrame({
        "player_dk_id": cur["player_dk_id"].to_numpy(),
        "p_play": np.clip(p, 0, 1),
        "points_if_plays": pts,
    })
    summary["projection"] = summary["p_play"] * summary["points_if_plays"]
    c = pd.DataFrame(contrib, columns=cols)
    c["_baseline"] = base
    c.insert(0, "player_dk_id", cur["player_dk_id"].to_numpy())
    return summary, c


def top_contributions(contrib: pd.DataFrame, player_dk_id: int,
                      top: int = 10) -> pd.DataFrame:
    if contrib.empty:
        return pd.DataFrame(columns=["feature", "points"])
    row = contrib[contrib["player_dk_id"] == player_dk_id]
    if row.empty:
        return pd.DataFrame(columns=["feature", "points"])
    vals = row.drop(columns=["player_dk_id", "_baseline"]).iloc[0]
    out = pd.DataFrame({"feature": vals.index, "points": vals.to_numpy(float)})
    out = out.reindex(out["points"].abs().sort_values(ascending=False).index)
    return out.head(top).reset_index(drop=True)


def importance(df: pd.DataFrame, top: int = 20, sample: int = 3000
               ) -> pd.DataFrame:
    """How much each feature moves the conditional points model: the mean
    absolute contribution over a sample of rows. The column keeps the name
    `gain` because pages/1_Model.py reads each feature's share of it."""
    if not _enough(df):
        return pd.DataFrame(columns=["feature", "gain"])
    cols = _cols(df)
    reg = _regressor(df, cols)
    played = df[df["played"].astype(bool)]
    rows = played.sample(min(sample, len(played)), random_state=SEED)
    contrib, _ = _contributions(reg, _X(rows, cols), _X(played, cols).median())
    out = pd.DataFrame({"feature": cols, "gain": np.abs(contrib).mean(axis=0)})
    return out.sort_values("gain", ascending=False).head(top).reset_index(
        drop=True)

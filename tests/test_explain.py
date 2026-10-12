"""Projection explanations.

An explanation that does not reconcile with the number it explains is worse
than none: it invites trust it has not earned. LightGBM's per-feature
contributions plus the model baseline must sum exactly to the prediction, and
this pins that.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dfs import model
from dfs.features import feature_columns


@pytest.fixture
def history() -> pd.DataFrame:
    """Enough rows to clear the model's minimum training thresholds."""
    rng = np.random.default_rng(5)
    n = 900
    salary = rng.integers(25, 91, n) * 100.0
    played = rng.random(n) < 0.6
    df = pd.DataFrame({
        "season": 2023,
        "week": rng.integers(1, 15, n),
        "player_dk_id": np.arange(n),
        "player_name": [f"p{i}" for i in range(n)],
        "position": rng.choice(["QB", "RB", "WR", "TE", "DST"], n),
        "team": rng.choice(["BUF", "MIA", "KC", "DEN"], n),
        "salary": salary,
        "salary_pct": rng.random(n),
        "value": salary / 1000,
        "implied_total": rng.normal(23, 4, n),
        "dk_points_r3": rng.normal(9, 5, n),
        "played": played,
    })
    for pos in ("QB", "RB", "WR", "TE", "DST"):
        df[f"pos_{pos}"] = (df["position"] == pos).astype(float)
    df["dk_scored"] = np.where(
        played, salary / 500 + rng.normal(0, 4, n), 0.0
    ).clip(0)
    return df


def test_contributions_reconcile_with_the_prediction(history):
    """contributions + baseline == points-if-plays, exactly."""
    cur = history.head(40).copy()
    summary, contrib = model.explain(history, cur)
    assert not summary.empty

    cols = feature_columns(history)
    total = contrib[cols].sum(axis=1) + contrib["_baseline"]
    assert np.allclose(total.to_numpy(), summary["points_if_plays"].to_numpy())


def test_projection_is_the_product_of_the_two_halves(history):
    summary, _ = model.explain(history, history.head(40).copy())
    assert np.allclose(
        summary["projection"],
        summary["p_play"] * summary["points_if_plays"],
    )


def test_probability_of_playing_is_a_probability(history):
    summary, _ = model.explain(history, history.head(40).copy())
    assert summary["p_play"].between(0, 1).all()


def test_top_contributions_are_sorted_by_magnitude(history):
    cur = history.head(20).copy()
    _, contrib = model.explain(history, cur)
    pid = int(cur["player_dk_id"].iloc[0])
    top = model.top_contributions(contrib, pid, top=8)
    assert len(top) == 8
    mags = top["points"].abs().to_numpy()
    assert np.all(np.diff(mags) <= 1e-9), "not ordered by absolute impact"


def test_top_contributions_keeps_the_sign(history):
    """A feature that lowers the projection must show as negative."""
    cur = history.head(20).copy()
    _, contrib = model.explain(history, cur)
    pid = int(cur["player_dk_id"].iloc[0])
    top = model.top_contributions(contrib, pid, top=20)
    assert (top["points"] < 0).any(), "expected at least one negative driver"


def test_unknown_player_returns_empty(history):
    _, contrib = model.explain(history, history.head(10).copy())
    assert model.top_contributions(contrib, 10_000_000).empty


def test_too_little_history_returns_empty():
    tiny = pd.DataFrame({
        "season": [2023] * 10, "week": range(10),
        "player_dk_id": range(10), "position": ["RB"] * 10,
        "salary": [5000.0] * 10, "dk_scored": [10.0] * 10,
        "played": [True] * 10,
    })
    summary, contrib = model.explain(tiny, tiny)
    assert summary.empty and contrib.empty

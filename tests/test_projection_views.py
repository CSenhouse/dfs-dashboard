"""Floor / expected / ceiling views.

Each contest type reads a different point off the same predicted distribution.
They must be ordered, share one participation estimate, and each must actually
be the best lineup by its own measure -- otherwise the contest selector is
decorative.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dfs import model


@pytest.fixture
def history() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    n = 1200
    salary = rng.integers(25, 91, n) * 100.0
    played = rng.random(n) < 0.6
    df = pd.DataFrame({
        "season": 2023,
        "week": rng.integers(1, 15, n),
        "player_dk_id": np.arange(n),
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
        played, salary / 500 + rng.normal(0, 5, n), 0.0
    ).clip(0)
    return df


def test_views_are_strictly_ordered(history):
    """Floor <= expected <= ceiling for EVERY player, with no exceptions.

    The three quantiles are fitted independently, so they cross for about one
    player in twenty -- a known property of separate quantile fits. Left
    alone, a table would occasionally print a floor above the expected value,
    which reads as a bug. two_stage_all pins the ordering around the mean.
    """
    out = model.two_stage_all(history, history.head(200).copy())
    assert (out["points_floor"] <= out["points_mean"] + 1e-9).all()
    assert (out["points_mean"] <= out["points_ceiling"] + 1e-9).all()
    assert (out["proj_floor"] <= out["proj_mean"] + 1e-9).all()
    assert (out["proj_mean"] <= out["proj_ceiling"] + 1e-9).all()


def test_every_view_shares_one_participation_estimate(history):
    """The classifier is fit once; all three projections use the same p_play,
    so differences between them are about upside, not availability."""
    out = model.two_stage_all(history, history.head(120).copy())
    for view in ("floor", "mean", "ceiling"):
        assert np.allclose(
            out[f"proj_{view}"], out["p_play"] * out[f"points_{view}"]
        )


def test_probability_is_a_probability(history):
    out = model.two_stage_all(history, history.head(120).copy())
    assert out["p_play"].between(0.0, 1.0).all()


def test_projections_are_never_negative(history):
    out = model.two_stage_all(history, history.head(120).copy())
    for c in out.columns:
        if c.startswith(("proj_", "points_")):
            assert (out[c] >= 0).all(), c


def test_each_view_wins_on_its_own_measure(history):
    """A floor lineup must have the best floor, a ceiling lineup the best
    ceiling. If they did not, the contest selector would be cosmetic."""
    from dfs.optimizer import optimize

    cur = history.head(400).copy()
    out = model.two_stage_all(history, cur)
    cur = cur.reset_index(drop=True).merge(out, on="player_dk_id", how="left")
    cur["player_name"] = "p" + cur["player_dk_id"].astype(str)

    built = {}
    for view in ("floor", "mean", "ceiling"):
        lus = optimize(cur.assign(projection=cur[f"proj_{view}"]),
                       objective="projection", n_lineups=1)
        assert lus, f"no lineup for {view}"
        built[view] = lus[0]

    for view in ("floor", "mean", "ceiling"):
        own = built[view][f"proj_{view}"].sum()
        for other in ("floor", "mean", "ceiling"):
            assert own >= built[other][f"proj_{view}"].sum() - 1e-6, (
                f"{other} lineup beat the {view} lineup on {view}"
            )


def test_too_little_history_returns_empty():
    tiny = pd.DataFrame({
        "season": [2023] * 10, "week": range(10), "player_dk_id": range(10),
        "position": ["RB"] * 10, "salary": [5000.0] * 10,
        "dk_scored": [10.0] * 10, "played": [True] * 10,
    })
    assert model.two_stage_all(tiny, tiny).empty


# --- showdown pricing regime -------------------------------------------------

def test_salary_rank_can_be_scoped_to_the_slate():
    """Showdown prices a player against ~45 others, Classic against ~800.

    Ranking a Showdown salary against a Classic week would place a $10,000
    Showdown quarterback somewhere absurd, so the grouping has to follow the
    format.
    """
    from dfs import features as F

    df = pd.DataFrame({
        "season": [2024] * 6,
        "week": [1] * 6,
        "draft_group_id": [100, 100, 100, 200, 200, 200],
        "player_dk_id": range(6),
        "position": ["QB"] * 6,
        "team": ["BUF", "MIA", "KC", "DEN", "LV", "LAC"],
        # slate 100 is priced high, slate 200 low
        "salary": [11000.0, 10000.0, 9000.0, 5000.0, 4000.0, 3000.0],
        "gsis_id": [f"id{i}" for i in range(6)],
    })

    across = df.groupby(["season", "week", "position"])["salary"].rank(pct=True)
    within = df.groupby(["draft_group_id", "position"])["salary"].rank(pct=True)

    # Ranked across the week, slate 200's best quarterback looks like the
    # worst on the board. Ranked inside his own slate he is the best.
    assert across.iloc[3] < 0.6
    assert within.iloc[3] == pytest.approx(1.0)
    assert within.iloc[0] == pytest.approx(1.0)


def test_the_hundo_view_sits_between_floor_and_ceiling(history):
    """Hundos are scored on the 70th percentile: an upside view, but never
    above the ceiling or below the floor it is meant to sit between."""
    out = model.two_stage_all(history, history.head(200).copy())
    assert (out["points_floor"] <= out["points_hundo"] + 1e-9).all()
    assert (out["points_hundo"] <= out["points_ceiling"] + 1e-9).all()
    assert np.allclose(out["proj_hundo"], out["p_play"] * out["points_hundo"])

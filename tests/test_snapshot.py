"""Pre-lock snapshots.

A snapshot stores the blend as a factor rather than DFF's raw numbers, so the
one thing to pin is that applying the factor reproduces what the dashboard's
Blend control does through overrides.apply -- otherwise the tracker would be
scoring a blend nobody saw.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dfs import overrides, snapshot


@pytest.fixture
def cur():
    return pd.DataFrame({
        "player_dk_id": [1, 2, 3],
        "player_name": ["a", "b", "c"],
        "position": ["QB", "WR", "RB"],
        "team": ["KC", "KC", "SF"], "opponent": ["NYJ", "NYJ", "LV"],
        "salary": [6000.0, 5000.0, 7000.0], "p_play": [1.0, 0.9, 1.0],
        "points_floor": [15.0, 7.0, 10.0], "points_mean": [20.0, 11.0, 14.0],
        "points_ceiling": [26.0, 17.0, 20.0], "points_hundo": [23.0, 14.0, 17.0],
        "season": 2026, "week": 9, "draft_group_id": 1,
    })


def _views(c):
    c = c.copy()
    for v in ("floor", "mean", "ceiling", "hundo"):
        c[f"proj_{v}"] = c["p_play"] * c[f"points_{v}"]
    return c


def test_factor_reproduces_the_dashboard_blend(cur):
    c = _views(cur)
    matched = pd.DataFrame({"player_dk_id": [1, 2], "dff_proj": [24.0, 8.0]})
    snap = snapshot.take(c, snapshot.blend_factor(c, matched))
    blended = snapshot.with_method(snap, "blend")
    for col in ("proj_floor", "proj_ceiling", "proj_hundo"):
        base = c.set_index("player_dk_id")[col]
        mean = c.set_index("player_dk_id")["proj_mean"]
        rows = [{"player_dk_id": pid, "player_name": "x", "view": col,
                 "model_value": base[pid],
                 "user_value": 0.5 * base[pid] + 0.5 * d * base[pid] / mean[pid]}
                for pid, d in zip(matched["player_dk_id"], matched["dff_proj"])]
        via_overrides = overrides.apply(c, pd.DataFrame(rows), col)
        np.testing.assert_allclose(blended[col].to_numpy(),
                                   via_overrides[col].to_numpy(), rtol=1e-9)


def test_unmatched_players_are_left_alone(cur):
    c = _views(cur)
    f = snapshot.blend_factor(c, pd.DataFrame(columns=["player_dk_id",
                                                       "dff_proj"]))
    assert (f == 1.0).all()


def test_model_method_is_the_snapshot_itself(cur):
    snap = snapshot.take(_views(cur))
    pd.testing.assert_frame_equal(snapshot.with_method(snap, "model"), snap)


def test_round_trip(cur, tmp_path, monkeypatch):
    monkeypatch.setattr(snapshot, "DIR", tmp_path)
    snap = snapshot.take(_views(cur))
    snapshot.save(7, snap)
    back = snapshot.load(7)
    assert list(back.columns) == list(snap.columns)
    assert snapshot.age_seconds(7) < 60
    assert snapshot.load(8) is None

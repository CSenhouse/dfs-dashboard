"""Projected ownership.

The structural facts are what can be pinned: a Classic field rosters nine
players per entry, so a contest's ownership sums to 9.0 and so must the
prediction; ranks are computed inside the slate; and the contest type is read
from the payout, not the name.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dfs import ownership


def _slate(n_per_pos=8, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for pos, lo in (("QB", 5000), ("RB", 4000), ("WR", 3000), ("TE", 2500),
                    ("DST", 2000)):
        for i in range(n_per_pos):
            sal = lo + 500 * i
            proj = sal / 1000 * rng.uniform(1.5, 3.0)
            rows.append({"player_dk_id": len(rows), "position": pos,
                         "salary": float(sal), "p_play": 1.0,
                         "proj_mean": proj, "dff_proj": proj * 1.05,
                         "implied_total": 22.0, "spread": 0.0})
    return pd.DataFrame(rows)


def _train(slates=3):
    """Synthetic fields where ownership follows value within position."""
    frames = []
    for s in range(slates):
        f = ownership.slate_features(_slate(seed=s))
        for kind in ownership.TYPES:
            g = ownership.with_type(f, kind)
            w = np.exp(3 * g["value"]) * (2 if kind == "double_up" else 1)
            g["own"] = 9 * w / w.sum()
            g["draft_group"] = s
            frames.append(g)
    return pd.concat(frames, ignore_index=True)


def test_contest_type_comes_from_the_payout():
    assert ownership.contest_type(0.435, 3448) == "double_up"
    assert ownership.contest_type(0.2537, 5945) == "tournament"
    assert ownership.contest_type(0.22, 100) == "hundo"


def test_ranks_are_within_position_and_one_is_best():
    f = ownership.slate_features(_slate())
    best_rb = f[f["position"] == "RB"].sort_values("value").iloc[-1]
    assert best_rb["value_rank_pos"] == 1
    assert f.groupby("position")["salary_rank_pos"].max().eq(8).all()


def test_a_missing_dff_export_still_predicts():
    f = ownership.slate_features(_slate().drop(columns="dff_proj"))
    assert f["dff_proj"].isna().all() and f["dff_value_rank_pos"].isna().all()
    m = ownership.fit(_train())
    out = ownership.predict(m, ownership.with_type(f, "double_up"))
    assert out.notna().all()


@pytest.mark.parametrize("kind", ownership.TYPES)
def test_each_field_sums_to_nine_roster_spots(kind):
    m = ownership.fit(_train())
    f = ownership.with_type(ownership.slate_features(_slate(seed=9)), kind)
    assert ownership.predict(m, f).sum() == pytest.approx(9.0)


def test_live_projection_has_one_column_per_contest_type():
    m = ownership.fit(_train())
    out = ownership.project(m, _slate(seed=9))
    for kind in ownership.TYPES:
        assert out[f"own_{kind}"].sum() == pytest.approx(9.0)


def test_value_is_learned_as_chalk():
    m = ownership.fit(_train(slates=4))
    f = ownership.with_type(ownership.slate_features(_slate(seed=9)),
                            "double_up")
    pred = ownership.predict(m, f)
    assert np.corrcoef(pred.rank(), f["value"].rank())[0, 1] > 0.5

"""Availability filtering.

DraftKings prices players who cannot take a snap -- injured reserve, out,
suspended -- usually at the $300 minimum. To an optimizer that is a nearly
free roster spot, so an IR player will be rostered every time unless he is
removed from the pool. This happened on a live showdown slate: Isiah Pacheco,
on IR, at $300, projected 0.04, selected.
"""
from __future__ import annotations

import pandas as pd
import pytest

from dfs.sources.dk_api import playable, unavailable


def pool(*statuses) -> pd.DataFrame:
    return pd.DataFrame({
        "player_dk_id": range(len(statuses)),
        "player_name": [f"p{i}" for i in range(len(statuses))],
        "position": ["RB"] * len(statuses),
        "team": ["BUF"] * len(statuses),
        "salary": [300.0] * len(statuses),
        "status": list(statuses),
    })


@pytest.mark.parametrize("status", ["IR", "OUT", "O", "NA", "SUSP", "PUP",
                                    "DNP", "INACTIVE"])
def test_unavailable_statuses_are_removed(status):
    assert playable(pool(status)).empty
    assert len(unavailable(pool(status))) == 1


@pytest.mark.parametrize("status", ["Q", "D", "QUESTIONABLE", "DOUBTFUL"])
def test_questionable_players_stay_in_the_pool(status):
    """They frequently play; the projection handles them, not a hard filter."""
    assert len(playable(pool(status))) == 1
    assert unavailable(pool(status)).empty


def test_active_players_stay():
    assert len(playable(pool(None, "", "None"))) == 3


def test_status_matching_is_case_and_space_insensitive():
    assert playable(pool(" ir ", "Out")).empty


def test_mixed_pool_splits_correctly():
    df = pool("IR", None, "Q", "OUT")
    keep, drop = playable(df), unavailable(df)
    assert len(keep) == 2
    assert len(drop) == 2
    assert set(drop["status"]) == {"IR", "OUT"}
    assert not set(keep["player_dk_id"]) & set(drop["player_dk_id"])


def test_missing_status_column_is_a_no_op():
    df = pool("IR").drop(columns=["status"])
    assert len(playable(df)) == 1
    assert unavailable(df).empty


def test_empty_frame_is_handled():
    empty = pd.DataFrame(columns=["player_dk_id", "status"])
    assert playable(empty).empty
    assert unavailable(empty).empty


def test_an_ir_player_would_otherwise_be_rostered():
    """Shows why the filter is needed rather than relying on the projection:
    a $300 player with a near-zero projection still beats leaving salary
    unspent, so the optimizer takes him."""
    from dfs.optimizer import optimize_showdown

    # Ten priced-up starters, so the cap forces exactly one cheap sixth man:
    # a captain at 1.5x plus five at $8,000 would be $52,000.
    rows = [
        {"player_dk_id": i, "player_name": f"p{i}", "position": "WR",
         "team": "BUF" if i % 2 else "DET",
         "salary": 8000.0, "projection": 20.0, "status": None}
        for i in range(10)
    ]
    # The IR player and the cheapest healthy alternative are both $300, and
    # the IR player projects marginally higher -- which is the real situation.
    rows.append({
        "player_dk_id": 99, "player_name": "ir_guy", "position": "WR",
        "team": "DET", "salary": 300.0, "projection": 0.04, "status": "IR",
    })
    rows.append({
        "player_dk_id": 98, "player_name": "scrub", "position": "WR",
        "team": "BUF", "salary": 300.0, "projection": 0.01, "status": None,
    })
    df = pd.DataFrame(rows)

    with_ir = optimize_showdown(df, n_lineups=1)[0]
    assert 99 in set(with_ir["player_dk_id"]), "fixture should tempt the solver"

    without = optimize_showdown(playable(df), n_lineups=1)[0]
    assert 99 not in set(without["player_dk_id"])

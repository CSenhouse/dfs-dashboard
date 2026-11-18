"""Vacated opportunity.

Week 10 of 2026: Kenneth Walker III was ruled inactive on gameday, Emmett
Johnson inherited the Kansas City backfield and scored 38.1 at 87% owned,
and the model had him at 6.9. These pin who counts as absent, what he leaves,
and that it is lagged: only weeks BEFORE the one being built contribute.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dfs import features


def _hist():
    """Two KC backs over weeks 1-3: a starter and a backup."""
    rows = []
    for w in (1, 2, 3):
        rows.append(dict(gsis_id="walker", season=2026, week=w, team="KC",
                         position="RB", carries=18.0, targets=4.0,
                         attempts=np.nan, appeared=1.0))
        rows.append(dict(gsis_id="johnson", season=2026, week=w, team="KC",
                         position="RB", carries=4.0, targets=1.0,
                         attempts=np.nan, appeared=1.0))
    return pd.DataFrame(rows)


def _roster(walker_status: str):
    rows = []
    for w in (1, 2, 3, 4):
        for pid in ("walker", "johnson"):
            status = walker_status if (pid == "walker" and w == 4) else "ACT"
            rows.append(dict(gsis_id=pid, season=2026, week=w, team="KC",
                             position="RB", status=status))
    return pd.DataFrame(rows)


@pytest.fixture
def run(monkeypatch):
    def go(walker_status: str, inj: pd.DataFrame | None = None):
        monkeypatch.setattr(features, "_load",
                            lambda name, season=None: _roster(walker_status))
        f = pd.DataFrame({"gsis_id": ["walker", "johnson"], "season": 2026,
                          "week": 4})
        v = features._vacated(_hist(), f, [2026],
                              inj if inj is not None else pd.DataFrame())
        return v[v["week"] == 4].set_index("gsis_id")
    return go


def test_a_gameday_inactive_starter_hands_his_work_to_the_backup(run):
    v = run("INA")
    assert v.loc["johnson", "vac_usage"] == pytest.approx(22.0)
    assert v.loc["johnson", "vac_share"] == pytest.approx(1.0)
    assert v.loc["johnson", "vac_inherit"] == pytest.approx(22.0)


def test_the_absent_player_inherits_nothing(run):
    v = run("INA")
    assert v.loc["walker", "vac_share"] == 0.0
    assert v.loc["walker", "vac_inherit"] == 0.0


def test_nobody_absent_means_nothing_vacated(run):
    v = run("ACT")
    assert (v["vac_usage"] == 0).all() and (v["vac_inherit"] == 0).all()
    assert v.loc["johnson", "vac_share"] == pytest.approx(5 / 27)


def test_out_on_the_injury_report_counts_as_absent(run):
    inj = pd.DataFrame({"gsis_id": ["walker"], "season": [2026], "week": [4],
                        "injury_prob": [0.0]})
    v = run("ACT", inj)
    assert v.loc["johnson", "vac_inherit"] == pytest.approx(22.0)


def test_questionable_alone_is_not_absent(run):
    inj = pd.DataFrame({"gsis_id": ["walker"], "season": [2026], "week": [4],
                        "injury_prob": [0.75]})
    v = run("ACT", inj)
    assert v.loc["johnson", "vac_inherit"] == 0.0

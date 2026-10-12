"""Schedule validation.

DraftKings runs simulated video-game contests ("Madden Stream") that use real
team abbreviations, real player names and a real DST, with competition names
like "BUF @ LV". Nothing inside the slate payload distinguishes them from
football. The schedule does: those matchups are not played that week.

These tests stub the schedule so they run offline.
"""
from __future__ import annotations

import pandas as pd
import pytest

from dfs import slates


@pytest.fixture
def stub_schedule(monkeypatch):
    """A three-game week, plus one week whose games have no final score."""
    sched = pd.DataFrame({
        "season": [2024] * 3 + [2026] * 2,
        "week": [5, 5, 5, 1, 1],
        "game_type": ["REG"] * 5,
        "gameday": pd.to_datetime(
            ["2024-10-06"] * 3 + ["2026-09-13"] * 2
        ),
        "home_team": ["BUF", "KC", "DAL", "DET", "PIT"],
        "away_team": ["MIA", "DEN", "NYG", "NO", "ATL"],
        "result": [3.0, -7.0, 10.0, None, None],
    })
    monkeypatch.setattr(slates, "_games", lambda: sched)
    return sched


def dk_slate(pairs) -> pd.DataFrame:
    rows = []
    for cid, (home, away) in enumerate(pairs):
        for team in (home, away):
            rows.append({
                "competition_id": cid,
                "team": team,
                "player_dk_id": len(rows),
            })
    return pd.DataFrame(rows)


def test_real_matchups_pass(stub_schedule):
    dk = dk_slate([("BUF", "MIA"), ("KC", "DEN"), ("DAL", "NYG")])
    assert slates.matches_schedule(dk, 2024, 5)


def test_fabricated_matchups_are_rejected(stub_schedule):
    """Real teams, real week, but these games were never played."""
    dk = dk_slate([("BUF", "LV"), ("SEA", "PHI"), ("JAX", "MIA")])
    assert not slates.matches_schedule(dk, 2024, 5)


def test_home_away_order_does_not_matter(stub_schedule):
    dk = dk_slate([("MIA", "BUF"), ("DEN", "KC"), ("NYG", "DAL")])
    assert slates.matches_schedule(dk, 2024, 5)


def test_mostly_real_slate_still_passes(stub_schedule):
    """One odd competition should not sink an otherwise genuine slate."""
    dk = dk_slate([("BUF", "MIA"), ("KC", "DEN"), ("DAL", "NYG"),
                   ("BUF", "MIA")])
    assert slates.matches_schedule(dk, 2024, 5)


def test_half_fabricated_slate_fails(stub_schedule):
    dk = dk_slate([("BUF", "MIA"), ("KC", "DEN"),
                   ("SEA", "PHI"), ("JAX", "LV")])
    assert not slates.matches_schedule(dk, 2024, 5)


def test_unknown_week_is_rejected(stub_schedule):
    dk = dk_slate([("BUF", "MIA")])
    assert not slates.matches_schedule(dk, 2024, 99)


def test_empty_slate_is_rejected(stub_schedule):
    assert not slates.matches_schedule(pd.DataFrame(), 2024, 5)


# --- completed-week guard --------------------------------------------------

def test_finished_week_is_complete(stub_schedule):
    assert slates.week_is_complete(2024, 5)


def test_unplayed_week_is_not_complete(stub_schedule):
    """A live or future week has no actual scores. prepare() turns a missing
    score into a real 0, so training on it would teach the model that every
    player busted."""
    assert not slates.week_is_complete(2026, 1)


def test_missing_week_is_not_complete(stub_schedule):
    assert not slates.week_is_complete(2024, 99)

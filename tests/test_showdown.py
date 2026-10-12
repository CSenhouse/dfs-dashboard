"""Showdown (single-game) rules.

Showdown is a different game from Classic, not a variant: 1 Captain + 5 FLEX,
no position slots, both teams required, Captain at 1.5x points and 1.5x salary.

Running Classic rules on a Showdown slate produced a 9-player lineup priced
entirely at Captain rates and nothing errored, so these pin both the pricing
and the roster shape.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dfs.config import CAPTAIN_MULTIPLIER, SALARY_CAP
from dfs.optimizer import optimize_showdown


@pytest.fixture
def slate() -> pd.DataFrame:
    """One game, two teams, priced like a real showdown."""
    rng = np.random.default_rng(3)
    rows = []
    pid = 0
    for team in ("BUF", "DET"):
        for pos, n in (("QB", 2), ("RB", 5), ("WR", 8), ("TE", 4), ("DST", 1)):
            for _ in range(n):
                pid += 1
                salary = float(rng.integers(2, 121) * 100)
                rows.append({
                    "player_dk_id": pid,
                    "player_name": f"p{pid}",
                    "position": pos,
                    "team": team,
                    "salary": salary,
                    "captain_salary": salary * CAPTAIN_MULTIPLIER,
                    "projection": salary / 500 + rng.normal(0, 1),
                })
    return pd.DataFrame(rows)


def test_roster_is_one_captain_and_five_flex(slate):
    lu = optimize_showdown(slate, n_lineups=1)[0]
    assert len(lu) == 6
    assert (lu["slot"] == "CPT").sum() == 1
    assert (lu["slot"] == "FLEX").sum() == 5


def test_no_position_requirements(slate):
    """Unlike Classic there is no QB or DST requirement -- any six players."""
    lu = optimize_showdown(slate, n_lineups=1)[0]
    assert set(lu["position"]) <= {"QB", "RB", "WR", "TE", "DST"}
    # nothing forces a QB or a DST into the lineup
    assert len(lu) == 6


def test_captain_costs_and_scores_one_and_a_half(slate):
    lu = optimize_showdown(slate, n_lineups=1)[0]
    cpt = lu[lu["slot"] == "CPT"].iloc[0]
    assert cpt["lineup_salary"] == pytest.approx(
        cpt["salary"] * CAPTAIN_MULTIPLIER
    )
    assert cpt["lineup_points"] == pytest.approx(
        cpt["projection"] * CAPTAIN_MULTIPLIER
    )


def test_flex_players_cost_base_price(slate):
    lu = optimize_showdown(slate, n_lineups=1)[0]
    flex = lu[lu["slot"] == "FLEX"]
    assert np.allclose(flex["lineup_salary"], flex["salary"])
    assert np.allclose(flex["lineup_points"], flex["projection"])


def test_respects_the_cap_including_captain_surcharge(slate):
    for lu in optimize_showdown(slate, n_lineups=5):
        assert lu["lineup_salary"].sum() <= SALARY_CAP


def test_both_teams_are_represented(slate):
    """DK does not allow rostering a single side of the game."""
    for lu in optimize_showdown(slate, n_lineups=5):
        assert lu["team"].nunique() == 2


def test_no_player_is_both_captain_and_flex(slate):
    for lu in optimize_showdown(slate, n_lineups=5):
        assert not lu["player_dk_id"].duplicated().any()


def test_lineups_are_unique(slate):
    lus = optimize_showdown(slate, n_lineups=8, min_difference=1)
    sets = [frozenset(l["player_dk_id"]) for l in lus]
    assert len(set(sets)) == len(sets)


def test_surcharge_can_rule_out_the_top_projection_as_captain():
    """The 1.5x cost makes captaincy a real decision, not a label.

    Built so the best player is affordable as FLEX but not as Captain: at
    1.5x he consumes the cap and leaves nothing for five team-mates. A solver
    that ignored the surcharge would captain him anyway.
    """
    rows = [
        {"player_dk_id": 1, "player_name": "star", "position": "WR",
         "team": "BUF", "salary": 20000.0, "projection": 30.0},
    ]
    pid = 1
    for team in ("BUF", "DET"):
        for _ in range(6):
            pid += 1
            rows.append({
                "player_dk_id": pid, "player_name": f"p{pid}",
                "position": "WR", "team": team,
                "salary": 5900.0, "projection": 14.0,
            })
    pool = pd.DataFrame(rows)
    pool["captain_salary"] = pool["salary"] * CAPTAIN_MULTIPLIER

    lu = optimize_showdown(pool, n_lineups=1)[0]
    assert lu["lineup_salary"].sum() <= SALARY_CAP
    cpt = lu[lu["slot"] == "CPT"].iloc[0]
    # Captaining the star costs 30,000 and leaves 20,000 for five players at
    # 5,900 each (29,500) -- infeasible. So he must not be the captain.
    assert cpt["player_name"] != "star"


def test_falls_back_when_captain_salary_absent(slate):
    """A pool without the column should still price the captain at 1.5x."""
    lu = optimize_showdown(slate.drop(columns=["captain_salary"]), n_lineups=1)[0]
    cpt = lu[lu["slot"] == "CPT"].iloc[0]
    assert cpt["lineup_salary"] == pytest.approx(
        cpt["salary"] * CAPTAIN_MULTIPLIER
    )
    assert lu["lineup_salary"].sum() <= SALARY_CAP


def test_exposure_cap_is_enforced(slate):
    n = 10
    lus = optimize_showdown(slate, n_lineups=n, max_exposure=0.4,
                            min_difference=2)
    used = pd.concat(lus)["player_dk_id"].value_counts()
    assert used.max() <= int(0.4 * n)


def test_empty_pool_returns_nothing():
    empty = pd.DataFrame(
        columns=["player_dk_id", "player_name", "position", "team",
                 "salary", "projection"]
    )
    assert optimize_showdown(empty) == []


def test_a_captain_can_be_pinned_to_the_captain_slot(slate):
    """`locks` only requires a player somewhere on the roster. Comparing
    captain candidates means rebuilding the other five around each one, so the
    slot itself has to be forceable."""
    want = int(slate.sort_values("projection").iloc[0]["player_dk_id"])
    got = optimize_showdown(slate, captain=want)
    assert got, "a legal lineup exists with any captain"
    lu = got[0]
    cpt = lu[lu["slot"] == "CPT"].iloc[0]
    assert int(cpt["player_dk_id"]) == want
    assert (lu["slot"] == "FLEX").sum() == 5


def test_pinning_a_captain_costs_1_5x_salary(slate):
    want = int(slate.sort_values("salary").iloc[-1]["player_dk_id"])
    lu = optimize_showdown(slate, captain=want)[0]
    cpt = lu[lu["slot"] == "CPT"].iloc[0]
    assert cpt["lineup_salary"] == pytest.approx(
        cpt["salary"] * CAPTAIN_MULTIPLIER)
    assert lu["lineup_salary"].sum() <= SALARY_CAP


def test_an_unknown_captain_returns_nothing_rather_than_ignoring_it(slate):
    """Silently building someone else's lineup would be worse than failing."""
    assert optimize_showdown(slate, captain=-1) == []

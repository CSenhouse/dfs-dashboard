"""Field simulation.

The cash line estimate rests on this sampler, so its invariants matter: every
simulated entry must be a legal lineup, and the sampler must not quietly drop
the sharp end of the skill range by rejecting over-cap draws.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dfs import field
from dfs.config import SALARY_CAP


@pytest.fixture
def pool() -> pd.DataFrame:
    """A slate with enough depth at every position, priced realistically."""
    rng = np.random.default_rng(7)
    teams = ["BUF", "MIA", "NE", "NYJ", "KC", "DEN", "LV", "LAC",
             "CIN", "BAL", "PIT", "CLE", "DAL", "NYG", "PHI", "WAS"]
    rows = []
    pid = 0
    for pos, n in (("QB", 20), ("RB", 50), ("WR", 80), ("TE", 30), ("DST", 16)):
        for k in range(n):
            pid += 1
            salary = float(rng.integers(25, 91) * 100)
            rows.append({
                "player_dk_id": pid,
                "player_name": f"p{pid}",
                "position": pos,
                # spread across teams so the optimizer's two-game rule is
                # satisfiable
                "team": teams[k % len(teams)],
                "salary": salary,
                # projection tracks salary, as real pricing does
                "projection": salary / 500 + rng.normal(0, 1),
                "dk_scored": max(0.0, salary / 500 + rng.normal(0, 6)),
            })
    return pd.DataFrame(rows)


def test_sampler_fills_every_request(pool):
    """Rejecting over-cap draws instead of repairing them silently biases the
    sharp end of the field, where expensive players cluster."""
    for skill in (1.0, 3.0, 6.0):
        scores = field.sample_lineups(pool, n=500, field_skill=skill, seed=1)
        assert len(scores) == 500, f"skill {skill} only filled {len(scores)}"


def test_every_entry_respects_the_cap(pool, monkeypatch):
    """Capture salaries by scoring on salary instead of points."""
    priced = pool.assign(dk_scored=pool["salary"])
    totals = field.sample_lineups(priced, n=400, field_skill=3.0, seed=2)
    assert len(totals) == 400
    assert totals.max() <= SALARY_CAP


def test_entries_use_nine_players(pool):
    """Score each pick as 1 so the total is the roster size."""
    counted = pool.assign(dk_scored=1.0)
    totals = field.sample_lineups(counted, n=300, field_skill=3.0, seed=3)
    assert set(np.unique(totals)) == {9.0}


def test_higher_skill_scores_no_worse(pool):
    """Concentrating on better-projected players should not hurt."""
    low = np.median(field.sample_lineups(pool, n=1500, field_skill=1.0, seed=4))
    high = np.median(field.sample_lineups(pool, n=1500, field_skill=4.0, seed=4))
    assert high >= low


def test_cash_line_is_the_last_paid_position_not_the_median(pool):
    """A DraftKings double-up pays exactly 2.00x to the top 43.5%, read off
    the public contest API. The rake comes from paying fewer places at a full
    double, not from paying half the field less than double -- so the bar is
    the 56.5th percentile and is strictly harder than the median."""
    scores = field.sample_lineups(pool, n=1000, field_skill=3.0, seed=5)
    line = field.cash_line(pool, n=1000, field_skill=3.0, seed=5)
    assert line == pytest.approx(float(np.percentile(scores, 56.5)))
    assert line > float(np.median(scores))


def test_sensitivity_sweeps_skill(pool):
    out = field.cash_line_sensitivity(pool, skills=(1.0, 3.0), n=400, seed=6)
    assert list(out["field_skill"]) == [1.0, 3.0]
    assert (out["p75"] >= out["median"]).all()
    assert (out["median"] >= out["p25"]).all()


def test_is_reproducible(pool):
    a = field.sample_lineups(pool, n=200, field_skill=3.0, seed=11)
    b = field.sample_lineups(pool, n=200, field_skill=3.0, seed=11)
    assert np.array_equal(a, b)


def test_empty_pool_returns_nothing():
    empty = pd.DataFrame(
        columns=["player_dk_id", "position", "salary", "projection", "dk_scored"]
    )
    assert field.sample_lineups(empty, n=10).size == 0
    assert np.isnan(field.cash_line(empty, n=10))


def test_missing_position_returns_nothing(pool):
    """A slate with no DST cannot produce a legal lineup."""
    no_dst = pool[pool["position"] != "DST"]
    assert field.sample_lineups(no_dst, n=10).size == 0


# --- optimizing opponents ---------------------------------------------------

def test_mirror_opponents_at_zero_noise_reproduce_our_lineup(pool):
    """The boundary case that makes this experiment interpretable.

    With no noise and no independent projection, every opponent is solving the
    identical problem we are, so they must return the identical lineup. If
    this ever drifts, the comparison is measuring solver nondeterminism rather
    than skill.
    """
    from dfs import optimizer

    ours = optimizer.optimize(pool, objective="projection", n_lineups=1)
    assert ours, "fixture pool must admit a legal lineup"
    us = ours[0]["dk_scored"].sum()

    scores = field.optimizer_entries(pool, n=5, disagreement=0.0, seed=1)
    assert len(scores) == 5
    assert np.allclose(scores, us)


def test_disagreement_scatters_opponents(pool):
    tight = field.optimizer_entries(pool, n=8, disagreement=0.05, seed=2)
    wide = field.optimizer_entries(pool, n=8, disagreement=0.50, seed=2)
    assert wide.std() > tight.std()


def test_independent_opponents_need_not_match_us(pool):
    """Given a genuinely different projection, opponents can beat us -- which
    is the whole point of running the independent variant."""
    rng = np.random.default_rng(9)
    rival = pool.assign(rival=rng.permutation(pool["projection"].to_numpy()))
    scores = field.optimizer_entries(
        rival, n=6, disagreement=0.0, opponent_col="rival", seed=3
    )
    assert len(scores) == 6
    from dfs import optimizer
    us = optimizer.optimize(rival, objective="projection", n_lineups=1)[0]
    assert not np.allclose(scores, us["dk_scored"].sum())


def test_optimizer_entries_handles_empty_pool():
    empty = pd.DataFrame(
        columns=["player_dk_id", "position", "team", "salary",
                 "projection", "dk_scored"]
    )
    assert field.optimizer_entries(empty, n=3).size == 0


# --- rosters, not just scores -----------------------------------------------
# Simulation needs to re-score the SAME opponent in every simulated world, so
# the sampler has to hand back who was rostered rather than one total.

def test_indices_and_scores_describe_the_same_entries(pool):
    """The two views must agree, or a simulated field would be scoring
    different lineups than the cash-line estimate it is compared against."""
    scores = field.sample_lineups(pool, n=200, field_skill=3.0, seed=21)
    idx = field.sample_lineup_indices(pool, n=200, field_skill=3.0, seed=21)
    assert idx.shape == (200, 9)
    assert np.allclose(pool["dk_scored"].to_numpy()[idx].sum(axis=1), scores)


def test_indices_point_at_the_pool_that_was_passed_in(pool):
    """Rows get dropped for a missing projection, so the returned positions
    have to be mapped back or they silently address the wrong players -- and
    an off-by-a-few is invisible here, because every index is still some
    valid player.

    The holes are scattered rather than taken off the front: the fixture is
    ordered by position, so blanking a prefix would delete every quarterback
    and leave no legal lineup to sample at all.
    """
    holed = pool.copy()
    holed.loc[holed.index[::5], "projection"] = np.nan
    idx = field.sample_lineup_indices(holed, n=50, seed=22)
    assert idx.shape == (50, 9)
    assert idx.max() < len(holed)
    assert not np.isnan(holed["projection"].to_numpy()[idx.ravel()]).any()



def test_indices_work_without_any_actual_scores(pool):
    """A live slate has no result yet; that must not stop the field sampler."""
    live = pool.drop(columns=["dk_scored"])
    assert field.sample_lineup_indices(live, n=20, seed=23).shape == (20, 9)


# --- showdown ---------------------------------------------------------------

@pytest.fixture
def game() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    rows = []
    for pid in range(1, 41):
        salary = float(rng.integers(20, 111) * 100)
        rows.append({
            "player_dk_id": pid,
            "player_name": f"p{pid}",
            "position": ["QB", "RB", "WR", "TE", "K", "DST"][pid % 6],
            "team": "KC" if pid % 2 else "BUF",
            "salary": salary,
            "captain_salary": salary * 1.5,
            "projection": salary / 600 + rng.normal(0, 1),
        })
    return pd.DataFrame(rows)


def test_showdown_entries_are_legal_rosters(game):
    entries, weights = field.sample_showdown_indices(game, n=300, seed=1)
    assert entries.shape == (300, 6)
    salary = game["salary"].to_numpy()
    cpt = game["captain_salary"].to_numpy()
    teams = game["team"].to_numpy()
    for row in entries:
        assert len(set(row)) == 6, "a player was rostered twice"
        assert len(set(teams[row])) == 2, "DK requires both teams"
        spend = cpt[row[0]] + salary[row[1:]].sum()
        assert spend <= SALARY_CAP


def test_the_showdown_captain_carries_the_multiplier(game):
    _, weights = field.sample_showdown_indices(game, n=20, seed=2)
    assert (weights[:, 0] == 1.5).all()
    assert (weights[:, 1:] == 1.0).all()


def test_showdown_captains_are_not_all_the_same_player(game):
    """A field that captains one name every time is not a field. The captain
    costs 1.5x for 1.5x, so entrants spread the choice widely."""
    entries, _ = field.sample_showdown_indices(game, n=400, field_skill=3.0,
                                               seed=3)
    assert len(set(entries[:, 0])) > 5


def test_showdown_needs_two_teams(game):
    one_sided = game[game["team"] == "KC"]
    entries, _ = field.sample_showdown_indices(one_sided, n=10, seed=4)
    assert entries.size == 0

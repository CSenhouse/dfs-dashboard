"""Lineup simulation.

The simulator's output is a probability, which is exactly the kind of number
that looks fine while being wrong. These tests pin the properties that make it
mean something: that the quantile function round-trips, that the point mass at
zero is handled so the PIT stays uniform, and that correlation actually
reaches the variance of a stacked lineup -- which is the entire reason the
module exists.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from dfs import simulate
from dfs.config import CAPTAIN_MULTIPLIER

ALPHAS = simulate.ALPHAS


def grid_row(*values: float) -> np.ndarray:
    assert len(values) == len(ALPHAS)
    return np.array([list(values)], dtype=float)


# --- the quantile function --------------------------------------------------

def test_the_grid_is_reproduced_at_its_own_quantiles():
    g = grid_row(1, 3, 6, 9, 13, 18, 26)
    got = simulate.inverse_cdf(np.array([list(ALPHAS)]), g)
    assert got == pytest.approx(g)


def test_the_quantile_function_is_increasing():
    g = grid_row(0, 0, 0, 2, 5, 11, 20)
    u = np.linspace(1e-4, 1 - 1e-4, 500)[None, :]
    out = simulate.inverse_cdf(u, g)[0]
    assert np.all(np.diff(out) >= -1e-9)


def test_the_right_tail_goes_past_the_top_quantile():
    """Tournaments are decided above q95. A linear tail would cap the ceiling
    just above it and make every GPP probability too small."""
    g = grid_row(1, 3, 6, 9, 13, 18, 26)
    rare = simulate.inverse_cdf(np.array([[0.999]]), g)[0, 0]
    assert rare > 40


def test_a_flat_grid_has_no_tail_to_extrapolate():
    """A player the model gives no upside at all must not acquire one."""
    g = grid_row(0, 0, 0, 0, 0, 0, 0)
    out = simulate.inverse_cdf(np.array([[0.5, 0.9999]]), g)
    assert out == pytest.approx(0.0)


# --- the probability integral transform -------------------------------------

def test_pit_of_the_models_own_draws_is_uniform():
    """The round trip. Draw from a player's predicted distribution, then ask
    where those draws fell in it: the answer must be uniform, or every
    probability the simulator prints is distorted."""
    g = grid_row(1, 3, 6, 9, 13, 18, 26)
    rng = np.random.default_rng(0)
    u = rng.random((1, 20_000))
    draws = simulate.inverse_cdf(u, g)[0]
    back = simulate.pit(draws, np.repeat(g, draws.size, axis=0),
                        np.ones(draws.size), seed=1)
    assert stats.kstest(back, "uniform").pvalue > 0.01


def test_pit_stays_uniform_through_the_point_mass_at_zero():
    """Half of all priced players score exactly 0.00, so the mixture of "did
    not play" and "played badly" is not an edge case -- it is most of the
    data. Getting it wrong would pile the PIT up at one end."""
    p_play = 0.6
    g = grid_row(1, 3, 6, 9, 13, 18, 26)
    rng = np.random.default_rng(2)
    n = 20_000
    played = rng.random(n) < p_play
    draws = np.where(
        played, simulate.inverse_cdf(rng.random((1, n)), g)[0], 0.0
    )
    back = simulate.pit(draws, np.repeat(g, n, axis=0),
                        np.full(n, p_play), seed=3)
    assert stats.kstest(back, "uniform").pvalue > 0.01


def test_a_big_score_lands_high_and_a_zero_lands_low():
    g = np.repeat(grid_row(1, 3, 6, 9, 13, 18, 26), 2, axis=0)
    got = simulate.pit(np.array([0.0, 40.0]), g, np.array([0.9, 0.9]), seed=4)
    assert got[0] < 0.2
    assert got[1] > 0.95


# --- roles and correlation --------------------------------------------------

def test_roles_rank_by_projection_not_by_outcome():
    """Tiering on what a player went on to score would define the depth chart
    with hindsight and inflate every correlation measured with it."""
    df = pd.DataFrame({
        "team": ["KC"] * 3,
        "position": ["WR"] * 3,
        "projection": [4.0, 15.0, 9.0],
        "dk_scored": [30.0, 1.0, 2.0],
    })
    assert list(simulate.roles(df)) == ["WR3", "WR1", "WR2"]


def test_players_in_different_games_are_independent():
    pool = pd.DataFrame({
        "player_dk_id": [1, 2],
        "team": ["KC", "BUF"], "opponent": ["DEN", "MIA"],
        "position": ["QB", "WR"], "projection": [20.0, 15.0],
    })
    table = {("team", "QB", "WR1"): 0.5, ("opp", "QB", "WR1"): 0.5}
    corr = simulate.correlation_matrix(pool, table)
    assert corr[0, 1] == pytest.approx(0.0)


def test_a_quarterback_is_tied_to_his_own_top_receiver():
    pool = pd.DataFrame({
        "player_dk_id": [1, 2, 3],
        "team": ["KC", "KC", "KC"], "opponent": ["DEN"] * 3,
        "position": ["QB", "WR", "WR"], "projection": [20.0, 15.0, 6.0],
    })
    table = {("team", "QB", "WR1"): 0.45, ("team", "QB", "WR2"): 0.10}
    corr = simulate.correlation_matrix(pool, table)
    assert corr[0, 1] == pytest.approx(0.45)
    assert corr[0, 2] == pytest.approx(0.10)


def test_an_impossible_correlation_table_is_repaired_not_crashed():
    """Pairwise numbers need not describe a joint distribution. Cholesky fails
    outright on one that does not, so the block is projected first."""
    bad = np.array([[1.0, 0.9, -0.9], [0.9, 1.0, 0.9], [-0.9, 0.9, 1.0]])
    fixed = simulate._nearest_psd(bad)
    assert np.linalg.eigvalsh(fixed).min() >= -1e-9
    assert np.allclose(np.diag(fixed), 1.0)


# --- sampling ---------------------------------------------------------------

def sim_pool(n_wr: int = 2) -> pd.DataFrame:
    rows = [{"player_dk_id": 1, "player_name": "qb", "position": "QB",
             "team": "KC", "opponent": "DEN", "salary": 7000,
             "projection": 20.0, "p_play": 1.0}]
    for k in range(n_wr):
        rows.append({"player_dk_id": 2 + k, "player_name": f"wr{k}",
                     "position": "WR", "team": "KC" if k == 0 else "DEN",
                     "opponent": "DEN" if k == 0 else "KC",
                     "salary": 6000, "projection": 15.0 - k, "p_play": 1.0})
    df = pd.DataFrame(rows)
    for a, v in zip(ALPHAS, (1, 3, 6, 9, 13, 18, 26)):
        df[f"q{int(a * 100):02d}"] = float(v)
    return df


def test_sampling_needs_a_quantile_grid():
    pool = sim_pool().drop(columns=["q30"])
    with pytest.raises(KeyError, match="q30"):
        simulate.sample(pool, n_sims=10)


def test_a_stack_is_more_volatile_when_correlation_is_modelled():
    """The reason this module exists. A QB and his own WR1 move together, so
    rostering both widens the lineup's distribution. An independent simulator
    cannot see that, and would price a stack and a diversified pair the same."""
    pool = sim_pool()
    table = {("team", "QB", "WR1"): 0.60}
    stack = [1, 2]

    linked = simulate.sample(pool, n_sims=8000, seed=0, table=table)
    apart = simulate.sample(pool, n_sims=8000, seed=0, correlated=False)
    assert (linked.lineup_totals(stack).std()
            > 1.15 * apart.lineup_totals(stack).std())


def test_correlation_leaves_the_individual_players_alone():
    """It must move the joint distribution only. If a player's own mean
    shifted, every projection in the dashboard would quietly change."""
    pool = sim_pool()
    table = {("team", "QB", "WR1"): 0.60}
    linked = simulate.sample(pool, n_sims=20000, seed=0, table=table)
    apart = simulate.sample(pool, n_sims=20000, seed=0, correlated=False)
    a = linked.lineup_totals([1]).mean()
    b = apart.lineup_totals([1]).mean()
    assert a == pytest.approx(b, rel=0.05)


def test_a_player_who_cannot_play_scores_nothing():
    pool = sim_pool()
    pool.loc[pool["player_dk_id"] == 1, "p_play"] = 0.0
    sim = simulate.sample(pool, n_sims=200, seed=0)
    assert sim.lineup_totals([1]).max() == 0.0


def test_the_captain_is_worth_one_and_a_half():
    pool = sim_pool()
    sim = simulate.sample(pool, n_sims=500, seed=0)
    plain = sim.lineup_totals([1, 2])
    capped = sim.lineup_totals([1, 2], captain_id=1)
    assert capped == pytest.approx(
        plain + (CAPTAIN_MULTIPLIER - 1.0) * sim.lineup_totals([1]), rel=1e-5
    )


def test_scoring_many_entries_agrees_with_scoring_one():
    pool = sim_pool()
    sim = simulate.sample(pool, n_sims=300, seed=0)
    entries = np.array([sim.columns([1, 2]), sim.columns([1, 3])])
    bulk = sim.entry_totals(entries)
    assert bulk[:, 0] == pytest.approx(sim.lineup_totals([1, 2]), rel=1e-5)
    assert bulk[:, 1] == pytest.approx(sim.lineup_totals([1, 3]), rel=1e-5)


def test_simulation_is_reproducible():
    pool = sim_pool()
    a = simulate.sample(pool, n_sims=200, seed=5).scores
    b = simulate.sample(pool, n_sims=200, seed=5).scores
    assert np.array_equal(a, b)


# --- reading a decision off it ----------------------------------------------

def test_value_reports_the_spread_not_just_the_mean():
    totals = np.array([100.0, 120.0, 140.0, 160.0])
    got = simulate.value(totals)
    assert got["mean"] == pytest.approx(130.0)
    assert got["p10"] < got["p50"] < got["p90"]
    assert "p_cash" not in got


def test_cash_probability_is_measured_against_that_world_s_own_line():
    totals = np.array([100.0, 150.0, 200.0, 250.0])
    opp = np.array([[90.0, 110.0], [90.0, 110.0], [150.0, 170.0],
                    [300.0, 320.0]])
    got = simulate.value(totals, opp)
    # The bar is the 56.5th percentile of each world, not its median: 101.3,
    # 101.3, 161.3, 311.3. We clear two of the four.
    assert got["p_cash"] == pytest.approx(0.5)
    assert got["cash_roi"] == pytest.approx(0.5 * simulate.PAYOUT - 1.0)


def test_break_even_is_half_the_field_at_a_true_double():
    """Two corrections that pull opposite ways, both measured off the public
    contest API. The payout is 2.00x rather than the 1.8x assumed here, so
    break-even is 50% and not 55.6% -- but the bar is the top 43.5%, not the
    top half, so reaching it is harder than it was being scored."""
    assert simulate.PAYOUT == 2.0
    assert 1.0 / simulate.PAYOUT == pytest.approx(0.50)
    assert simulate.PAID_FRACTION < 0.5


def test_a_moving_line_gives_a_different_answer_than_a_fixed_one():
    """Why the line is simulated per world rather than estimated once.

    Here our score and the field's rise together, as they really do -- a week
    where the popular plays hit is a week everyone scores. Against a fixed
    median this lineup looks like a 37.5% play; against the line as it
    actually moves it is a 12.5% play. The fixed number is not a rounding
    difference, it is the wrong decision.
    """
    totals = np.arange(10.0, 90.0, 10.0)
    lines = totals + 5.0
    lines[0] = 5.0
    opp = np.column_stack([lines - 1.0, lines + 1.0])

    moving = simulate.value(totals, opp)["p_cash"]
    fixed = float((totals > np.median(lines)).mean())
    assert moving == pytest.approx(0.125)
    assert fixed == pytest.approx(0.375)


# --- the spread correction ---------------------------------------------------
# Pooled across all players the predicted distributions look uniform. That
# average is two opposite errors cancelling: punts too wide, and the top
# decile -- the only bucket a lineup is built from -- too narrow.

def ranked_pool(n: int = 60) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append({
            "player_dk_id": i + 1, "player_name": f"p{i}", "position": "WR",
            "team": f"T{i:02d}", "opponent": f"O{i:02d}", "salary": 5000,
            "projection": float(i), "p_play": 1.0,
        })
    df = pd.DataFrame(rows)
    for a, v in zip(ALPHAS, (1, 3, 6, 9, 13, 18, 26)):
        df[f"q{int(a * 100):02d}"] = float(v)
    return df


def test_the_widening_is_applied_by_projection_rank():
    from dfs.spread import SPREAD_HI, SPREAD_LO

    lo, hi = simulate.spread_factors(ranked_pool())
    if not SPREAD_LO:
        assert (lo == 1.0).all() and (hi == 1.0).all()
        return
    assert hi[-1] == pytest.approx(SPREAD_HI[max(SPREAD_HI)])
    assert lo[-1] == pytest.approx(SPREAD_LO[max(SPREAD_LO)])


def test_the_upper_end_is_stretched_harder_than_the_lower():
    """The narrowness is lopsided. For the top decile q30 runs 0.89 too high
    and q85 runs 2.09 too low, so one factor around the median cannot fix
    both -- and using one left the grid's implied mean 0.93 below the mean
    regressor sitting beside it, which is 8.4 points across a lineup."""
    from dfs.spread import SPREAD_HI, SPREAD_LO

    if not SPREAD_LO:
        pytest.skip("no calibration fitted")
    top = max(SPREAD_HI)
    assert SPREAD_HI[top] > SPREAD_LO[top]


def test_widening_the_top_decile_widens_its_simulated_range():
    """The whole point: a stud's distribution has to come out wider than the
    raw quantile fit says, or every lineup built from studs looks too safe."""
    from dfs.spread import SPREAD_HI

    if not SPREAD_HI or SPREAD_HI[max(SPREAD_HI)] <= 1.0:
        pytest.skip("no widening fitted for the top bucket")
    pool = ranked_pool()
    top = [int(pool["player_dk_id"].iloc[-1])]
    wide = simulate.sample(pool, n_sims=8000, seed=0, calibrated=True)
    raw = simulate.sample(pool, n_sims=8000, seed=0, calibrated=False)
    assert wide.lineup_totals(top).std() > raw.lineup_totals(top).std()



def test_the_widening_leaves_the_middle_alone():
    """It has to change the spread without moving the median, or it would
    silently re-rank players and change which lineup gets built."""
    pool = ranked_pool()
    top = [int(pool["player_dk_id"].iloc[-1])]
    wide = simulate.sample(pool, n_sims=20000, seed=0, calibrated=True)
    raw = simulate.sample(pool, n_sims=20000, seed=0, calibrated=False)
    a = np.median(wide.lineup_totals(top))
    b = np.median(raw.lineup_totals(top))
    assert a == pytest.approx(b, abs=0.5)


def test_widening_does_not_disturb_correlation():
    """Scaling each marginal on its own leaves a correlation matrix alone, so
    the spread fix and the copula must not interfere."""
    pool = sim_pool()
    pool["projection"] = [20.0, 15.0, 14.0]
    table = {("team", "QB", "WR1"): 0.60}
    linked = simulate.sample(pool, n_sims=20000, seed=0, table=table,
                             calibrated=True)
    a, b = linked.scores[:, 0], linked.scores[:, 1]
    assert np.corrcoef(a, b)[0, 1] > 0.4


# --- the cash line -----------------------------------------------------------
# The simulated field is too soft: it samples players and repairs over-cap
# lineups by swapping down to the cheapest option, so its median lands near 107
# on a main slate where a real double-up cashes nearer 140. Every probability
# read off it unshifted is too kind.

def _line_fixture():
    totals = np.array([100.0, 120.0, 140.0, 160.0, 180.0, 200.0])
    lines = np.array([90.0, 100.0, 110.0, 120.0, 130.0, 140.0])
    return totals, np.column_stack([lines - 5.0, lines + 5.0])


def test_a_target_line_moves_the_bar_to_where_it_was_asked_for():
    totals, opp = _line_fixture()
    got = simulate.value(totals, opp, target_line=140.0)
    assert got["cash_line"] == pytest.approx(140.0)
    # 56.5th percentile of each [line-5, line+5] pair, then the median of those
    assert got["simulated_line"] == pytest.approx(115.65)


def test_raising_the_line_can_only_lower_the_cash_probability():
    totals, opp = _line_fixture()
    soft = simulate.value(totals, opp)["p_cash"]
    hard = simulate.value(totals, opp, target_line=180.0)["p_cash"]
    assert hard < soft


def test_the_line_still_moves_week_to_week_after_shifting():
    """A flat threshold would be the easy fix and the wrong one. The bar has to
    keep moving with the slate, so a week where the chalk busts lowers it
    rather than stranding it above a low-scoring week."""
    totals, opp = _line_fixture()
    raw = np.median(opp, axis=1)
    shifted = simulate.value(totals, opp, target_line=140.0)
    span = shifted["cash_line_p90"] - shifted["cash_line_p10"]
    assert span == pytest.approx(np.percentile(raw, 90) - np.percentile(raw, 10))
    assert span > 0


def test_shifting_leaves_our_own_lineup_alone():
    """It moves the opponents, not us: the lineup's own range must not change
    because you typed a different number into the control."""
    totals, opp = _line_fixture()
    a = simulate.value(totals, opp)
    b = simulate.value(totals, opp, target_line=175.0)
    for k in ("mean", "sd", "p10", "p50", "p90"):
        assert a[k] == pytest.approx(b[k])

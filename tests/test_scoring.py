"""DK scoring engine.

The engine agrees with DraftKings' own published points on 99.574% of 15,946
player-weeks (scripts/validate_scoring.py). These tests pin the individual
rules so a refactor cannot quietly move a coefficient.
"""
from __future__ import annotations

import pandas as pd
import pytest

from dfs.scoring import points_allowed_score, score_dst, score_offense


def row(**kw) -> pd.DataFrame:
    return pd.DataFrame([kw])


def test_passing_yards_quarter_point_per_yard():
    assert score_offense(row(passing_yards=250)).iloc[0] == pytest.approx(10.0)


def test_passing_td_worth_four():
    assert score_offense(row(passing_tds=3)).iloc[0] == pytest.approx(12.0)


def test_interception_costs_one():
    assert score_offense(row(passing_interceptions=2)).iloc[0] == pytest.approx(-2.0)


def test_full_ppr_reception_worth_one():
    assert score_offense(row(receptions=7)).iloc[0] == pytest.approx(7.0)


def test_rushing_and_receiving_tds_worth_six():
    got = score_offense(row(rushing_tds=1, receiving_tds=1)).iloc[0]
    assert got == pytest.approx(12.0)


def test_three_hundred_yard_passing_bonus():
    under = score_offense(row(passing_yards=299)).iloc[0]
    over = score_offense(row(passing_yards=300)).iloc[0]
    # 1 extra yard is worth 0.04; the bonus adds 3 on top of it.
    assert over - under == pytest.approx(3.04)


def test_hundred_yard_rushing_bonus():
    under = score_offense(row(rushing_yards=99)).iloc[0]
    over = score_offense(row(rushing_yards=100)).iloc[0]
    assert over - under == pytest.approx(3.1)


def test_hundred_yard_receiving_bonus():
    under = score_offense(row(receiving_yards=99)).iloc[0]
    over = score_offense(row(receiving_yards=100)).iloc[0]
    assert over - under == pytest.approx(3.1)


def test_bonuses_stack_within_one_game():
    got = score_offense(row(rushing_yards=100, receiving_yards=100)).iloc[0]
    # 10 + 10 rushing/receiving points, plus two separate 3-point bonuses.
    assert got == pytest.approx(26.0)


def test_fumble_lost_costs_one():
    assert score_offense(row(fumbles_lost_total=2)).iloc[0] == pytest.approx(-2.0)


def test_fumbles_lost_falls_back_to_components():
    """Older seasons lack fumbles_lost_total; components must still count."""
    got = score_offense(
        row(rushing_fumbles_lost=1, receiving_fumbles_lost=1, sack_fumbles_lost=1)
    ).iloc[0]
    assert got == pytest.approx(-3.0)


def test_two_point_conversions_worth_two_each():
    got = score_offense(
        row(passing_2pt_conversions=1, rushing_2pt_conversions=1,
            receiving_2pt_conversions=1)
    ).iloc[0]
    assert got == pytest.approx(6.0)


def test_return_td_credited_to_the_player():
    """DK's rules credit a return TD to the individual, not only the DST."""
    assert score_offense(row(special_teams_tds=1)).iloc[0] == pytest.approx(6.0)


def test_missing_columns_are_zero_not_an_error():
    assert score_offense(row(receptions=1)).iloc[0] == pytest.approx(1.0)


def test_realistic_stat_line():
    """Josh Allen, 2024 week 14: 54.88 DK points."""
    got = score_offense(row(
        passing_yards=342, passing_tds=3, passing_interceptions=0,
        rushing_yards=82, rushing_tds=2, fumbles_lost_total=0,
    )).iloc[0]
    # 13.68 + 12 + 3 (300yd) + 8.2 + 12 = 48.88
    assert got == pytest.approx(48.88)


@pytest.mark.parametrize("allowed,expected", [
    (0, 10.0), (1, 7.0), (6, 7.0), (7, 4.0), (13, 4.0),
    (14, 1.0), (20, 1.0), (21, 0.0), (27, 0.0),
    (28, -1.0), (34, -1.0), (35, -4.0), (59, -4.0),
])
def test_dst_points_allowed_tiers(allowed, expected):
    got = points_allowed_score(pd.Series([allowed])).iloc[0]
    assert got == pytest.approx(expected)


def test_dst_scoring_components():
    got = score_dst(row(
        def_sacks=4, def_interceptions=2, fumble_recovery_opp=1,
        def_tds=1, def_safeties=0, points_allowed=10,
    )).iloc[0]
    # 4 sacks + 4 int + 2 fumble + 6 TD + 4 (7-13 tier)
    assert got == pytest.approx(20.0)


def test_dst_shutout_is_ten():
    got = score_dst(row(points_allowed=0)).iloc[0]
    assert got == pytest.approx(10.0)


# --- kicking ----------------------------------------------------------------
# Classic has no kicker slot, but Showdown prices them, so the engine has to
# score one. It scored every kicker 0.00 until this was added: Chris Boswell's
# 2024 week 1 (six field goals, three from 50+) came out as zero.

def test_field_goals_pay_by_distance():
    assert score_offense(row(fg_made_0_19=1)).iloc[0] == pytest.approx(3.0)
    assert score_offense(row(fg_made_20_29=1)).iloc[0] == pytest.approx(3.0)
    assert score_offense(row(fg_made_30_39=1)).iloc[0] == pytest.approx(3.0)
    assert score_offense(row(fg_made_40_49=1)).iloc[0] == pytest.approx(4.0)
    assert score_offense(row(fg_made_50_59=1)).iloc[0] == pytest.approx(5.0)
    assert score_offense(row(fg_made_60_=1)).iloc[0] == pytest.approx(5.0)


def test_extra_points_are_worth_one():
    assert score_offense(row(pat_made=4)).iloc[0] == pytest.approx(4.0)


def test_real_kicking_line():
    """Chris Boswell, 2024 week 1: 1 from 20-29, 2 from 40-49, 3 from 50-59."""
    got = score_offense(row(
        fg_made_20_29=1, fg_made_40_49=2, fg_made_50_59=3, pat_made=0,
    )).iloc[0]
    assert got == pytest.approx(26.0)


def test_missed_field_goals_are_not_penalised():
    """DK does not dock a kicker for a miss."""
    made = score_offense(row(fg_made_30_39=1)).iloc[0]
    with_miss = score_offense(row(fg_made_30_39=1, fg_missed=3)).iloc[0]
    assert made == pytest.approx(with_miss)


def test_kicking_does_not_affect_other_positions():
    """The kicking term is additive, so a receiver's score is untouched."""
    from dfs.scoring import score_kicking

    wr = row(receptions=6, receiving_yards=95, receiving_tds=1)
    assert score_kicking(wr).iloc[0] == pytest.approx(0.0)
    assert score_offense(wr).iloc[0] == pytest.approx(6 + 9.5 + 6)

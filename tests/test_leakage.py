"""Leakage guards.

The single failure mode that makes a DFS model look excellent and lose money
is a feature that knows the future. Every rolling feature must be shifted
before it is aggregated, so a week-W value is built only from weeks < W.

These tests use a synthetic series whose values make an off-by-one obvious:
week N scores exactly N points.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dfs.backtest import walk_forward
from dfs.features import _roll


@pytest.fixture
def ramp() -> pd.DataFrame:
    """One player, weeks 1..8, scoring exactly the week number."""
    return pd.DataFrame({
        "gsis_id": ["p1"] * 8,
        "season": [2022] * 8,
        "week": range(1, 9),
        "dk_points": [float(w) for w in range(1, 9)],
    })


def test_rolling_excludes_current_week(ramp):
    out = _roll(ramp, ["dk_points"], by="gsis_id", windows=(3,))
    out = out.set_index("week")["dk_points_r3"]
    # Week 5 must average weeks 2,3,4 -> 3.0. If it saw itself it would be 4.0.
    assert out.loc[5] == pytest.approx(3.0)
    assert out.loc[8] == pytest.approx(6.0)  # weeks 5,6,7


def test_first_week_has_no_history(ramp):
    out = _roll(ramp, ["dk_points"], by="gsis_id", windows=(3,))
    assert pd.isna(out.set_index("week")["dk_points_r3"].loc[1])


def test_rolling_never_equals_current_value(ramp):
    """A value equal to its own week is the signature of a leak."""
    out = _roll(ramp, ["dk_points"], by="gsis_id", windows=(3,)).set_index("week")
    for w in range(2, 9):
        assert out["dk_points_r3"].loc[w] != pytest.approx(float(w))


def test_players_do_not_leak_into_each_other():
    df = pd.DataFrame({
        "gsis_id": ["a", "a", "b", "b"],
        "season": [2022] * 4,
        "week": [1, 2, 1, 2],
        "dk_points": [10.0, 10.0, 99.0, 99.0],
    })
    out = _roll(df, ["dk_points"], by="gsis_id", windows=(3,))
    a2 = out[(out["gsis_id"] == "a") & (out["week"] == 2)]["dk_points_r3"].iloc[0]
    assert a2 == pytest.approx(10.0)  # not contaminated by b's 99


def test_walk_forward_never_shows_the_future():
    """The fit function must only ever receive weeks earlier than the target."""
    df = pd.DataFrame({
        "season": [2022] * 60,
        "week": np.repeat(np.arange(1, 7), 10),
        "player_dk_id": list(range(10)) * 6,
        "position": ["RB"] * 60,
        "team": ["BUF"] * 60,
        "salary": [5000.0] * 60,
        "dk_actual": [10.0] * 60,
    })

    seen: list[tuple[int, int]] = []

    def spy(past: pd.DataFrame, cur: pd.DataFrame) -> pd.Series:
        target_week = cur["week"].iloc[0]
        assert past["week"].max() < target_week, (
            f"walk_forward leaked week {past['week'].max()} "
            f"into a week {target_week} prediction"
        )
        seen.append((target_week, int(past["week"].max())))
        return pd.Series(0.0, index=cur.index)

    walk_forward(df, spy, min_train_weeks=2)
    assert seen, "walk_forward never called the model"
    for target, latest_train in seen:
        assert latest_train == target - 1


def test_prepare_treats_missing_score_as_zero_not_dropped():
    """A priced player who never played scored 0 -- dropping him is
    survivorship bias, and it is exactly the bust a real lineup absorbs."""
    from dfs.backtest import prepare

    df = pd.DataFrame({
        "season": [2022, 2022],
        "week": [1, 1],
        "player_dk_id": [1, 2],
        "position": ["RB", "WR"],
        "team": ["BUF", "BUF"],
        "salary": [5000.0, 4000.0],
        "dk_actual": [12.0, np.nan],
    })
    out = prepare(df)
    assert len(out) == 2
    assert out.loc[out["player_dk_id"] == 2, "dk_scored"].iloc[0] == 0.0
    assert not out.loc[out["player_dk_id"] == 2, "played"].iloc[0]
    assert out.loc[out["player_dk_id"] == 1, "played"].iloc[0]


# --- absences must count, not be skipped ------------------------------------

def test_rolling_window_counts_missed_weeks_as_zero():
    """A window must mean the last N WEEKS, not the last N APPEARANCES.

    nflverse emits no row for a player who recorded no stat, so a naive
    rolling window skips absences and reaches back to whenever he last
    played. Justin Fields, benched from week 7 of 2024, still showed
    dk_points_r3 = 25.2 and offense_pct_r3 = 0.99 in week 11 -- numbers from
    weeks 4-6 when he was the starter. The model projected him 22.1; he
    scored 1.7.
    """
    played = pd.DataFrame({
        "gsis_id": ["p1"] * 4,
        "season": [2024] * 4,
        "week": [1, 2, 3, 4],
        "dk_points": [20.0, 20.0, 20.0, 20.0],
        "appeared": [1.0] * 4,
    })
    # His team played weeks 1-8; he only appeared in 1-4.
    calendar = pd.DataFrame({
        "gsis_id": ["p1"] * 8,
        "season": [2024] * 8,
        "week": range(1, 9),
    })

    naive = _roll(played, ["dk_points", "appeared"], windows=(3,))
    aware = _roll(played, ["dk_points", "appeared"], windows=(3,),
                  calendar=calendar, zero_absences=True)

    # Week 8: he has not played since week 4.
    aware_w8 = aware.set_index("week").loc[8]
    assert aware_w8["dk_points_r3"] == pytest.approx(0.0)
    assert aware_w8["appeared_r3"] == pytest.approx(0.0)
    assert aware_w8["weeks_since_played"] == pytest.approx(4.0)

    # The naive version cannot even see week 8, and its last value is stale.
    assert 8 not in set(naive["week"])
    assert naive["dk_points_r3"].iloc[-1] == pytest.approx(20.0)


def test_partial_absence_is_averaged_not_ignored():
    played = pd.DataFrame({
        "gsis_id": ["p1"] * 3,
        "season": [2024] * 3,
        "week": [1, 2, 5],
        "dk_points": [30.0, 30.0, 30.0],
        "appeared": [1.0] * 3,
    })
    calendar = pd.DataFrame({
        "gsis_id": ["p1"] * 6, "season": [2024] * 6, "week": range(1, 7),
    })
    out = _roll(played, ["dk_points", "appeared"], windows=(3,),
                calendar=calendar, zero_absences=True).set_index("week")
    # Week 5 looks at weeks 2, 3, 4 -> 30, 0, 0.
    assert out.loc[5, "dk_points_r3"] == pytest.approx(10.0)
    # Week 6 looks at 3, 4, 5 -> 0, 0, 30.
    assert out.loc[6, "dk_points_r3"] == pytest.approx(10.0)
    assert out.loc[6, "weeks_since_played"] == pytest.approx(1.0)


def test_bye_weeks_do_not_count_against_a_player():
    """A bye is not a missed week. It is absent from the calendar, so it
    neither zeroes a player's form nor makes it look stale."""
    played = pd.DataFrame({
        "gsis_id": ["p1"] * 3,
        "season": [2024] * 3,
        "week": [1, 2, 4],          # week 3 is the team's bye
        "dk_points": [15.0, 15.0, 15.0],
        "appeared": [1.0] * 3,
    })
    calendar = pd.DataFrame({
        "gsis_id": ["p1"] * 3, "season": [2024] * 3, "week": [1, 2, 4],
    })
    out = _roll(played, ["dk_points", "appeared"], windows=(3,),
                calendar=calendar).set_index("week")
    assert out.loc[4, "dk_points_r3"] == pytest.approx(15.0)
    # He played the previous game week, so one game week has passed.
    assert out.loc[4, "weeks_since_played"] == pytest.approx(1.0)


def test_calendar_rolling_still_excludes_the_current_week():
    """The absence fix must not quietly reintroduce lookahead."""
    df = pd.DataFrame({
        "gsis_id": ["p1"] * 6, "season": [2024] * 6, "week": range(1, 7),
        "dk_points": [float(w) for w in range(1, 7)], "appeared": [1.0] * 6,
    })
    calendar = df[["gsis_id", "season", "week"]]
    out = _roll(df, ["dk_points", "appeared"], windows=(3,),
                calendar=calendar).set_index("week")
    assert out.loc[5, "dk_points_r3"] == pytest.approx(3.0)   # weeks 2,3,4
    for w in range(2, 7):
        assert out.loc[w, "dk_points_r3"] != pytest.approx(float(w))


def test_default_keeps_form_and_only_flags_staleness():
    """The default does NOT zero a missed week.

    Zeroing looks obviously right and measured worse: the model already has a
    separate P(play) stage, so it makes the regressor pay for absence twice.
    Paired over 82 weeks, keeping the form and flagging its age beat zeroing
    by 6.6 lineup points in 52 of 82 weeks (p = 0.017). The form survives; the
    staleness is reported alongside it.
    """
    played = pd.DataFrame({
        "gsis_id": ["p1"] * 4, "season": [2024] * 4, "week": [1, 2, 3, 4],
        "dk_points": [20.0] * 4, "appeared": [1.0] * 4,
    })
    calendar = pd.DataFrame({
        "gsis_id": ["p1"] * 8, "season": [2024] * 8, "week": range(1, 9),
    })
    out = _roll(played, ["dk_points", "appeared"], windows=(3,),
                calendar=calendar).set_index("week")

    # Week 7: the window still reaches a week he played, so his form when
    # playing survives rather than being diluted toward zero.
    assert out.loc[7, "dk_points_r3"] == pytest.approx(20.0)

    # Week 8: the window now covers only absences, so form is UNKNOWN -- NaN,
    # not 0. LightGBM reads a missing value as its own signal, which is a
    # different statement from "he scores nothing", and that distinction is
    # what the zeroed variant threw away.
    assert pd.isna(out.loc[8, "dk_points_r3"])

    # Participation and staleness carry the absence instead.
    assert out.loc[8, "appeared_r3"] == pytest.approx(0.0)
    assert out.loc[8, "weeks_since_played"] == pytest.approx(4.0)
    assert out.loc[6, "appeared_r3"] == pytest.approx(2 / 3)

    # And zeroing remains available, producing the opposite.
    zeroed = _roll(played, ["dk_points", "appeared"], windows=(3,),
                   calendar=calendar, zero_absences=True).set_index("week")
    assert zeroed.loc[8, "dk_points_r3"] == pytest.approx(0.0)

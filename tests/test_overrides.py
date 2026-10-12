"""Hand edits to a projection.

Two things have to hold. The edit must reach EVERY consumer of the projection,
because the optimizer reads proj_* and the simulator reads the quantile grid,
and a control that moves one but not the other produces a Lineup value tab
quietly disagreeing with the projection it was shown. And the edit must be
written down with the model's own number beside it, because the log is the
measurement -- it is the only unit of judgement here small enough to accumulate
a usable sample inside one season.
"""
from __future__ import annotations

import pandas as pd
import pytest

from dfs import overrides


@pytest.fixture(autouse=True)
def _tmp_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(overrides, "DIR", tmp_path / "overrides")


@pytest.fixture
def pool():
    """Two players with a full model output: p_play, grid and three views."""
    return pd.DataFrame({
        "player_dk_id": [1, 2],
        "player_name": ["Chuba Hubbard", "Tre Tucker"],
        "p_play": [0.95, 0.80],
        "q05": [1.0, 0.0], "q30": [6.0, 2.0], "q50": [10.0, 5.0],
        "q85": [18.0, 12.0], "q95": [26.0, 20.0],
        "points_floor": [6.0, 2.0], "points_mean": [10.0, 5.0],
        "points_ceiling": [18.0, 12.0],
        "proj_floor": [5.7, 1.6], "proj_mean": [9.5, 4.0],
        "proj_ceiling": [17.1, 9.6],
    })


def edit(pid, name, view, model, user):
    return pd.DataFrame([{"player_dk_id": pid, "player_name": name,
                          "view": view, "model_value": model,
                          "user_value": user}])


def test_an_override_moves_the_whole_distribution(pool):
    """The optimizer reads proj_*, the simulator reads q*. One factor has to
    move both or the two disagree about the same player."""
    e = edit(1, "Chuba Hubbard", "proj_mean", 9.5, 4.75)   # halve him
    got = overrides.apply(pool, e, "proj_mean")
    row = got[got["player_dk_id"] == 1].iloc[0]
    assert row["proj_mean"] == pytest.approx(4.75)
    for col in ("proj_floor", "proj_ceiling", "points_mean",
                "q05", "q30", "q50", "q85", "q95"):
        assert row[col] == pytest.approx(pool.iloc[0][col] / 2)


def test_the_chance_he_plays_is_not_touched(pool):
    """An override says what a player is worth, not whether he suits up.
    Moving p_play would make it a second, hidden availability control."""
    e = edit(1, "Chuba Hubbard", "proj_mean", 9.5, 19.0)
    got = overrides.apply(pool, e, "proj_mean")
    assert got[got["player_dk_id"] == 1].iloc[0]["p_play"] == pytest.approx(0.95)


def test_an_untouched_player_is_untouched(pool):
    e = edit(1, "Chuba Hubbard", "proj_mean", 9.5, 4.75)
    got = overrides.apply(pool, e, "proj_mean")
    before, after = pool.iloc[1], got[got["player_dk_id"] == 2].iloc[0]
    for col in ("proj_mean", "q85", "p_play"):
        assert after[col] == pytest.approx(before[col])


def test_the_factor_is_taken_against_the_view_on_screen(pool):
    """Editing the Ceiling view means the ceiling becomes that number. Taking
    the ratio against the mean instead would move the player somewhere nobody
    asked for."""
    e = edit(1, "Chuba Hubbard", "proj_ceiling", 17.1, 8.55)
    got = overrides.apply(pool, e, "proj_ceiling")
    row = got[got["player_dk_id"] == 1].iloc[0]
    assert row["proj_ceiling"] == pytest.approx(8.55)
    assert row["proj_mean"] == pytest.approx(9.5 / 2)


def test_an_edit_recorded_for_another_view_does_not_apply(pool):
    e = edit(1, "Chuba Hubbard", "proj_ceiling", 17.1, 8.55)
    got = overrides.apply(pool, e, "proj_mean")
    assert got[got["player_dk_id"] == 1].iloc[0]["proj_mean"] == pytest.approx(9.5)


def test_a_near_zero_projection_shifts_instead_of_scaling():
    """A grid of zeros cannot be scaled into anything, and a ratio against 0.2
    is noise multiplied. Below MIN_BASE the edit is added instead."""
    pool = pd.DataFrame({
        "player_dk_id": [3], "player_name": ["Deep Backup"],
        "p_play": [0.3], "q30": [0.0], "q85": [0.4],
        "points_mean": [0.2], "proj_mean": [0.06],
    })
    e = edit(3, "Deep Backup", "proj_mean", 0.06, 8.0)
    got = overrides.apply(pool, e, "proj_mean")
    row = got.iloc[0]
    assert row["proj_mean"] == pytest.approx(8.0)
    assert row["q30"] == pytest.approx(7.94), "shifted, not scaled"
    assert row["p_play"] == pytest.approx(0.3)


def test_nothing_goes_negative():
    pool = pd.DataFrame({
        "player_dk_id": [4], "player_name": ["X"], "p_play": [0.5],
        "q05": [0.5], "proj_mean": [4.0], "points_mean": [8.0],
    })
    e = edit(4, "X", "proj_mean", 4.0, 0.0)
    got = overrides.apply(pool, e, "proj_mean")
    assert (got[["q05", "proj_mean", "points_mean"]].to_numpy() >= 0).all()


def test_a_typo_cannot_produce_an_absurd_lineup(pool):
    """95 typed where 9.5 was meant is a factor of ten. Clamping turns a
    nonsense lineup into a visibly clamped number."""
    e = edit(1, "Chuba Hubbard", "proj_mean", 9.5, 9500.0)
    got = overrides.apply(pool, e, "proj_mean")
    row = got[got["player_dk_id"] == 1].iloc[0]
    assert row["proj_mean"] == pytest.approx(9.5 * overrides.MAX_RATIO)


def test_the_log_keeps_the_model_value_beside_the_entered_one():
    """This is the whole point of the file. Without the model's own number
    there is nothing to compare the edit against after the games."""
    overrides.record(154084, edit(1, "Chuba Hubbard", "proj_mean", 9.5, 4.75))
    got = overrides.load(154084)
    assert len(got) == 1
    row = got.iloc[0]
    assert row["model_value"] == pytest.approx(9.5)
    assert row["user_value"] == pytest.approx(4.75)
    assert row["player_name"] == "Chuba Hubbard"
    assert row["draft_group_id"] == 154084
    assert str(row["recorded_at"])[:2] == "20"


def test_an_edit_that_changes_nothing_is_not_logged():
    """Opening the editor and touching nothing must not fill the experiment
    log with rows where the model and the user agree by construction."""
    assert overrides.record(154084, edit(1, "X", "proj_mean", 9.5, 9.5)) == 0
    assert overrides.load(154084).empty


def test_the_latest_edit_wins_and_the_history_is_kept():
    """A slate is edited over several sittings as news lands. What is in force
    is the last number; what was entered before it is evidence about when the
    edit was made, so it stays on disk."""
    overrides.record(154084, edit(1, "X", "proj_mean", 9.5, 4.0))
    overrides.record(154084, edit(1, "X", "proj_mean", 9.5, 6.0))
    assert overrides.load(154084).iloc[0]["user_value"] == pytest.approx(6.0)
    raw = pd.read_csv(overrides.path(154084))
    assert len(raw) == 2, "history kept"


def test_slates_do_not_leak_into_each_other():
    overrides.record(154084, edit(1, "X", "proj_mean", 9.5, 4.0))
    assert overrides.load(153771).empty


def test_a_missing_file_is_not_a_crash():
    assert overrides.load(999999).empty


def test_no_edits_is_a_no_op(pool):
    pd.testing.assert_frame_equal(
        overrides.apply(pool, pd.DataFrame(), "proj_mean"), pool)
    pd.testing.assert_frame_equal(
        overrides.apply(pool, None, "proj_mean"), pool)


def test_clearing_removes_the_override_for_that_view_only():
    overrides.record(154084, edit(1, "X", "proj_mean", 9.5, 4.0))
    overrides.record(154084, edit(1, "X", "proj_ceiling", 17.1, 8.0))
    assert overrides.forget(154084, [1], "proj_mean") == 1
    left = overrides.load(154084)
    assert len(left) == 1
    assert left.iloc[0]["view"] == "proj_ceiling"


def test_clearing_an_unknown_player_is_harmless():
    overrides.record(154084, edit(1, "X", "proj_mean", 9.5, 4.0))
    assert overrides.forget(154084, [999], "proj_mean") == 0
    assert len(overrides.load(154084)) == 1
    assert overrides.forget(999999, [1], "proj_mean") == 0

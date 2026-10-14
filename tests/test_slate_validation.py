"""Slate and crosswalk validation.

Every guard here exists because something silently contaminated the dataset:
college football slates, snake-draft groups storing ranks in the salary field,
simulated "Madden Stream" contests, and a name join that paired a star receiver
with a same-named defensive back.
"""
from __future__ import annotations

import pandas as pd
import pytest

from dfs import crosswalk
from dfs.sources.dk_api import classify


def slate(positions, teams, salaries, comps, names=None) -> pd.DataFrame:
    n = len(positions)
    return pd.DataFrame({
        "player_dk_id": range(n),
        "player_name": names or [f"p{i}" for i in range(n)],
        "position": positions,
        "team": teams,
        "salary": salaries,
        "competition_id": comps,
        "competition_name": [f"c{c}" for c in comps],
        "start_time": pd.Timestamp("2024-12-08", tz="UTC"),
    })


def test_real_nfl_classic_slate_is_kept():
    df = slate(
        ["QB", "RB", "WR", "TE", "DST"] * 4,
        ["BUF", "MIA", "NE", "NYJ"] * 5,
        [7000, 6000, 5000, 4000, 3000] * 4,
        [1, 1, 2, 2] * 5,
    )
    info = classify(df)
    assert info["is_nfl"]
    assert info["kind"] == "classic"


def test_college_football_is_rejected():
    """CFB shares QB/RB/WR/TE with the NFL; the team names give it away."""
    df = slate(
        ["QB", "RB", "WR"] * 4,
        ["UMD", "MSU", "WISC", "FSU"] * 3,
        [9000, 8000, 7000] * 4,
        [1, 1, 2, 2] * 3,
    )
    assert not classify(df)["is_nfl"]


def test_slate_without_dst_is_not_nfl_classic():
    """DK always prices a DST on an NFL salary-cap slate."""
    df = slate(
        ["QB", "RB", "WR", "TE"] * 4,
        ["BUF", "MIA", "NE", "NYJ"] * 4,
        [7000, 6000, 5000, 4000] * 4,
        [1, 1, 2, 2] * 4,
    )
    assert not classify(df)["is_nfl"]


def test_rank_encoded_draft_group_is_not_a_salary_slate():
    """Snake drafts store a draft rank (1..N) in the salary field."""
    n = 20
    df = slate(
        ["QB", "RB", "WR", "TE"] * 5,
        ["BUF", "MIA", "NE", "NYJ"] * 5,
        list(range(1, n + 1)),
        [1, 1, 2, 2] * 5,
    )
    info = classify(df)
    assert info["kind"] == "draft"
    assert info["rank_like"]


def test_salaries_must_be_multiples_of_one_hundred():
    df = slate(
        ["QB", "RB", "WR", "DST"] * 4,
        ["BUF", "MIA", "NE", "NYJ"] * 4,
        [7013, 6027, 5041, 3099] * 4,
        [1, 1, 2, 2] * 4,
    )
    assert classify(df)["kind"] == "draft"


def test_single_game_is_showdown():
    df = slate(
        ["QB", "RB", "WR", "TE", "DST"] * 2,
        ["BUF", "MIA"] * 5,
        [11000, 9000, 7000, 5000, 300] * 2,
        [1] * 10,
    )
    info = classify(df)
    assert info["kind"] == "showdown"
    assert info["is_nfl"]


def test_showdown_low_salaries_are_not_mistaken_for_ranks():
    """Captain pricing reaches below a classic slate's floor; a fixed minimum
    salary test wrongly rejected real showdown slates."""
    df = slate(
        ["QB", "WR", "DST"] * 4,
        ["BUF", "MIA"] * 6,
        [300, 400, 18000] * 4,
        [1] * 12,
    )
    assert classify(df)["kind"] == "showdown"


def test_idp_slate_is_not_standard_nfl():
    """DK ran Showdown and Classic variants that price defensive players.
    Real teams, real DST, real schedule -- and a different game. An IDP
    Classic slate has more players than the real main slate, so the "most
    players" rule picked it as the week's main slate."""
    df = slate(
        ["QB", "RB", "WR", "TE", "DST", "LB", "CB"] * 2,
        ["BUF", "MIA"] * 7,
        [11000, 9000, 7000, 5000, 4000, 3000, 2000] * 2,
        [1] * 14,
    )
    assert not classify(df)["is_nfl"]


def test_a_classic_slate_with_kickers_is_a_variant():
    df = slate(
        ["QB", "RB", "WR", "TE", "DST", "K"] * 4,
        ["BUF", "MIA", "NE", "NYJ"] * 6,
        [7000, 6000, 5000, 4000, 3000, 4500] * 4,
        [1, 1, 2, 2] * 6,
    )
    assert classify(df)["kind"] == "variant"


def test_a_hockey_slate_is_not_nfl():
    """NHL Showdowns use C/LW/RW/G/D, and WSH/PHI normalise to NFL teams. A
    hockey defenseman ("D") was once mapped to DST, and Capitals at Flyers
    sat on a week the Commanders played the Eagles -- it passed everything."""
    df = slate(
        ["C", "LW", "RW", "G", "D"] * 2,
        ["WSH", "PHI"] * 5,
        [9000, 8000, 7000, 10000, 6000] * 2,
        [1] * 10,
    )
    assert not classify(df)["is_nfl"]


def test_empty_slate_is_handled():
    info = classify(pd.DataFrame())
    assert info["kind"] == "empty"
    assert not info["is_nfl"]


# --- crosswalk -------------------------------------------------------------

def test_name_normalisation_strips_suffixes_and_punctuation():
    got = crosswalk.normalize_name(
        pd.Series(["A.J. Brown", "Michael Woods II", "Amon-Ra St. Brown"])
    )
    assert list(got) == ["aj brown", "michael woods", "amonra st brown"]


def test_team_aliases_are_normalised():
    got = crosswalk.normalize_team(pd.Series(["OAK", "SD", "STL", "LAR", "WSH"]))
    assert list(got) == ["LV", "LAC", "LA", "LA", "WAS"]


def test_initial_last_collapses_first_name_variants():
    got = crosswalk.initial_last(pd.Series(["joshua palmer", "josh palmer"]))
    assert got.iloc[0] == got.iloc[1] == "j palmer"


def test_ambiguous_names_are_not_matched():
    """Two Michael Thomases on one team at one position must resolve to
    neither, rather than silently picking one."""
    roster = pd.DataFrame({
        "season": [2019, 2019],
        "week": [1, 1],
        "team": ["NO", "NO"],
        "position": ["WR", "WR"],
        "full_name": ["Michael Thomas", "Michael Thomas"],
        "gsis_id": ["00-0031381", "00-0029293"],
        "key_name": ["michael thomas", "michael thomas"],
        "key_init": ["m thomas", "m thomas"],
    })
    dk = pd.DataFrame({
        "season": [2019],
        "team": ["NO"],
        "position": ["WR"],
        "key_name": ["michael thomas"],
        "key_init": ["m thomas"],
        "player_dk_id": [1],
    })
    from dfs.crosswalk import _attempt

    hit = _attempt(dk, roster, ["season", "team", "position", "key_name"], "t")
    assert hit.empty, "an ambiguous name must not resolve to an arbitrary id"


def test_unambiguous_name_does_match():
    roster = pd.DataFrame({
        "season": [2019],
        "week": [1],
        "team": ["NO"],
        "position": ["WR"],
        "full_name": ["Michael Thomas"],
        "gsis_id": ["00-0031381"],
        "key_name": ["michael thomas"],
        "key_init": ["m thomas"],
    })
    dk = pd.DataFrame({
        "season": [2019], "team": ["NO"], "position": ["WR"],
        "key_name": ["michael thomas"], "key_init": ["m thomas"],
        "player_dk_id": [1],
    })
    from dfs.crosswalk import _attempt

    hit = _attempt(dk, roster, ["season", "team", "position", "key_name"], "t")
    assert len(hit) == 1
    assert hit["gsis_id"].iloc[0] == "00-0031381"

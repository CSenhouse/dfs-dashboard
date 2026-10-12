"""Observed cash lines.

`data/observed_cash_lines.csv` is the only ground truth about contest outcomes
anywhere in this project. Everything else -- the field sampler, the estimated
cash line, every cash probability -- models the field with nothing to anchor
it, and does so wrongly: the simulated field medians about 107 on a main slate
against a first observed line of 147.

So these tests pin the two things that make the file useful: that an
observation beats an estimate, and that a line is never applied to a slate it
says nothing about.
"""
from __future__ import annotations

import pandas as pd
import pytest

from dfs import cash_lines


@pytest.fixture
def recorded(tmp_path, monkeypatch):
    def write(rows):
        path = tmp_path / "observed.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        monkeypatch.setattr(cash_lines, "PATH", path)
    return write


def test_slate_size_decides_what_a_line_is_evidence_about():
    """A three-game afternoon slate says nothing about a Sunday main. Pooling
    them would drag the main-slate default down by dozens of points."""
    assert cash_lines.slate_class(15) == cash_lines.MAIN
    assert cash_lines.slate_class(6) == cash_lines.MEDIUM
    assert cash_lines.slate_class(3) == cash_lines.SMALL
    assert cash_lines.slate_class(1, showdown=True) == cash_lines.SHOWDOWN


def test_an_observation_beats_the_estimate(recorded):
    recorded([{"season": 2026, "week": 1, "games": 15, "kind": "classic",
               "line": 147.0, "note": ""}])
    line, source = cash_lines.default_line(15)
    assert line == 147
    assert "observed" in source


def test_a_main_slate_line_does_not_leak_into_a_small_slate(recorded):
    recorded([{"season": 2026, "week": 1, "games": 15, "kind": "classic",
               "line": 147.0, "note": ""}])
    line, source = cash_lines.default_line(3)
    assert line == cash_lines.ESTIMATES[cash_lines.SMALL]
    assert "estimate" in source


def test_several_weeks_use_the_median_not_the_latest(recorded):
    """One week is one draw from a distribution with real spread. Chasing the
    most recent number would move the default around more than the truth."""
    recorded([
        {"season": 2026, "week": 1, "games": 15, "kind": "classic",
         "line": 140.0, "note": ""},
        {"season": 2026, "week": 2, "games": 13, "kind": "classic",
         "line": 147.0, "note": ""},
        {"season": 2026, "week": 3, "games": 14, "kind": "classic",
         "line": 190.0, "note": "shootout week"},
    ])
    line, source = cash_lines.default_line(15)
    assert line == 147
    assert "3 observed weeks" in source


def test_nothing_recorded_falls_back_to_the_estimate(recorded):
    recorded([])
    for games, showdown, want in ((15, False, cash_lines.MAIN),
                                  (6, False, cash_lines.MEDIUM),
                                  (1, True, cash_lines.SHOWDOWN)):
        line, source = cash_lines.default_line(games, showdown)
        assert line == cash_lines.ESTIMATES[want]
        assert "estimate" in source


def test_a_missing_file_is_not_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr(cash_lines, "PATH", tmp_path / "absent.csv")
    assert cash_lines.load().empty
    assert cash_lines.default_line(15)[0] == cash_lines.ESTIMATES[cash_lines.MAIN]


def test_the_real_file_parses():
    """It is hand-edited after a contest, so a typo has to fail loudly here
    rather than silently drop the only ground truth in the project."""
    df = cash_lines.load()
    assert not df.empty, "observed_cash_lines.csv should not be empty"
    assert df["line"].between(40, 300).all()
    assert set(df["kind"].str.lower()) <= {"classic", "showdown"}


def test_a_comma_in_a_note_is_an_error_not_a_shrug(recorded, tmp_path, monkeypatch):
    """This exact mistake already happened. An unquoted comma in the note
    shifted every column one to the left, the line parsed as text, the loader
    returned nothing, and the dashboard quietly fell back to an estimate while
    saying so in the smallest caption on the page. The file is the only ground
    truth about contest outcomes in this project and it is edited by hand
    after a contest, so a mangled row has to be loud."""
    path = tmp_path / "broken.csv"
    path.write_text(
        "season,week,games,kind,line,note\n"
        "2026,1,14,classic,147.0,scoring 51st pct, top-40 98th pct\n",
        encoding="utf-8", newline="\n",
    )
    monkeypatch.setattr(cash_lines, "PATH", path)
    with pytest.raises(ValueError, match="quoted"):
        cash_lines.load()


def test_a_broken_file_still_gives_the_dashboard_a_number(recorded, tmp_path,
                                                          monkeypatch):
    """Loud, but not fatal: the page has to render, and the reason has to
    reach it rather than being swallowed."""
    path = tmp_path / "broken.csv"
    path.write_text(
        "season,week,games,kind,line,note\n"
        "2026,1,14,classic,147.0,scoring 51st pct, top-40 98th pct\n",
        encoding="utf-8", newline="\n",
    )
    monkeypatch.setattr(cash_lines, "PATH", path)
    line, source = cash_lines.default_line(14)
    assert line == cash_lines.ESTIMATES[cash_lines.MAIN]
    assert "quoted" in source


def test_a_quoted_note_with_commas_is_fine(recorded, tmp_path, monkeypatch):
    path = tmp_path / "ok.csv"
    path.write_text(
        "season,week,games,kind,line,note\n"
        '2026,1,14,classic,147.0,"scoring 51st pct, top-40 98th pct"\n',
        encoding="utf-8", newline="\n",
    )
    monkeypatch.setattr(cash_lines, "PATH", path)
    assert cash_lines.default_line(14) == (147, "median of 1 observed week")


def test_a_tournament_line_is_not_a_cash_line(tmp_path, monkeypatch):
    """The field is a property of the CONTEST, not the slate. On 2026 week 2,
    draft group 153428, the $5 double-up paying the top 43.5% cashed at 144.2
    while a $3 tournament paying the top 25.4% -- identical players, identical
    kickoff -- cashed at 134.6. Fewer places paid and a LOWER bar is impossible
    for one field and ordinary for two: a double-up draws grinders and a cheap
    tournament draws everyone. Averaging them would produce a threshold that
    describes no contest anyone can enter."""
    path = tmp_path / "mixed.csv"
    path.write_text(
        "season,week,games,kind,paid_pct,line,note\n"
        "2026,2,13,classic,0.4350,144.2,double-up\n"
        "2026,2,13,classic,0.2537,134.6,tournament\n",
        encoding="utf-8", newline="\n",
    )
    monkeypatch.setattr(cash_lines, "PATH", path)
    line, source = cash_lines.default_line(13)
    assert line == 144, "the tournament row must not drag the cash line down"
    assert "1 observed" in source


def test_a_row_without_a_payout_fraction_is_taken_as_a_double_up(tmp_path,
                                                                 monkeypatch):
    """Early rows predate the column and were all double-ups."""
    path = tmp_path / "old.csv"
    path.write_text(
        "season,week,games,kind,line,note\n"
        "2026,1,14,classic,147.0,recorded before paid_pct existed\n",
        encoding="utf-8", newline="\n",
    )
    monkeypatch.setattr(cash_lines, "PATH", path)
    assert cash_lines.default_line(14)[0] == 147


def test_two_contests_on_one_week_are_not_two_weeks(recorded):
    """The caption on the dashboard is the only place the strength of this
    evidence is stated, so it must not inflate it. Two Showdown double-ups on
    the SAME game, sharing 24 of 39 entrants and 22 identical lineups, are one
    observation of one field. Counting rows called that two weeks."""
    recorded([
        {"season": 2026, "week": 4, "games": 1, "kind": "showdown",
         "paid_pct": 0.4359, "line": 90.0, "note": ""},
        {"season": 2026, "week": 4, "games": 1, "kind": "showdown",
         "paid_pct": 0.4359, "line": 91.7, "note": ""},
    ])
    line, source = cash_lines.default_line(1, showdown=True)
    assert line == 91
    assert "1 observed week" in source
    assert "2 contests" in source
    assert "2 observed weeks" not in source


def test_distinct_weeks_are_still_counted_as_weeks(recorded):
    recorded([
        {"season": 2026, "week": 4, "games": 1, "kind": "showdown",
         "paid_pct": 0.4350, "line": 90.0, "note": ""},
        {"season": 2026, "week": 5, "games": 1, "kind": "showdown",
         "paid_pct": 0.4350, "line": 100.0, "note": ""},
    ])
    _, source = cash_lines.default_line(1, showdown=True)
    assert "2 observed weeks" in source
    assert "contests" not in source, "one contest per week needs no extra count"


def test_one_busy_week_does_not_outvote_the_others(recorded):
    """Week 4 of 2026 supplied three main-slate double-ups at 105.7, 106.1 and
    108.2 against single observations of 145.0 and 144.2. Taking the median
    over ROWS moved the default from 145 to 108 on one week's evidence out of
    three. Aggregate per week first."""
    recorded([
        {"season": 2026, "week": 1, "games": 12, "kind": "classic",
         "paid_pct": 0.435, "line": 145.0, "note": ""},
        {"season": 2026, "week": 2, "games": 13, "kind": "classic",
         "paid_pct": 0.435, "line": 144.2, "note": ""},
        {"season": 2026, "week": 4, "games": 12, "kind": "classic",
         "paid_pct": 0.435, "line": 108.2, "note": ""},
        {"season": 2026, "week": 4, "games": 12, "kind": "classic",
         "paid_pct": 0.431, "line": 106.1, "note": ""},
        {"season": 2026, "week": 4, "games": 12, "kind": "classic",
         "paid_pct": 0.435, "line": 105.7, "note": ""},
    ])
    line, source = cash_lines.default_line(12)
    assert line == 144, "the median of three WEEKS, not of five rows"
    assert "3 observed weeks" in source
    assert "5 contests" in source


def test_the_caption_carries_the_spread(recorded):
    """A 38-point swing between weeks is the dominant uncertainty. Showing one
    figure without it hides the thing that matters most."""
    recorded([
        {"season": 2026, "week": 1, "games": 12, "kind": "classic",
         "paid_pct": 0.435, "line": 145.0, "note": ""},
        {"season": 2026, "week": 4, "games": 12, "kind": "classic",
         "paid_pct": 0.435, "line": 106.1, "note": ""},
    ])
    _, source = cash_lines.default_line(12)
    assert "ranged 106 to 145" in source


def test_a_single_week_reports_no_range(recorded):
    recorded([{"season": 2026, "week": 1, "games": 12, "kind": "classic",
               "paid_pct": 0.435, "line": 145.0, "note": ""}])
    _, source = cash_lines.default_line(12)
    assert "ranged" not in source

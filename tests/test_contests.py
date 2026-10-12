"""Parsing real contest standings.

A standings export is two tables printed side by side in one CSV -- one row
per entry on the left, one row per player on the right, padded to the same
height and unrelated row by row. Reading it as ordinary records produces
nonsense that looks like data, so these tests pin the shape.

It is also the only source of ownership this project can ever have, and the
only measurement of the field that is not a model of itself.
"""
from __future__ import annotations

import zipfile

import pandas as pd
import pytest

from dfs import contests

EXPORT = (
    "Rank,EntryId,EntryName,TimeRemaining,Points,Lineup,,Player,"
    "Roster Position,%Drafted,FPTS\n"
    "1,111,alice,0,215.18,"
    "DST Panthers  FLEX Ashton Jeanty QB Bryce Young RB D'Andre Swift "
    "RB Aaron Jones Sr. TE Dalton Schultz WR Jaxon Smith-Njigba "
    "WR CeeDee Lamb WR Jalen Coker,,Bijan Robinson,RB,42.92%,11.1\n"
    "2,222,bob,0,209.06,"
    "DST Buccaneers  FLEX T.J. Hockenson QB Dak Prescott RB Chuba Hubbard "
    "RB Aaron Jones Sr. TE Dalton Schultz WR Jaxon Smith-Njigba "
    "WR CeeDee Lamb WR DeVonta Smith,,Bijan Robinson,FLEX,4.86%,11.1\n"
    "3,333,carol,0,100.00,"
    "DST Titans  FLEX Tre Tucker QB Bryce Young RB Chuba Hubbard "
    "RB Aaron Jones Sr. TE Dalton Schultz WR CeeDee Lamb "
    "WR Jalen Coker WR DeVonta Smith,,CeeDee Lamb,WR,19.20%,38.3\n"
)


@pytest.fixture
def export(tmp_path):
    p = tmp_path / "contest-standings-999.csv"
    p.write_text(EXPORT, encoding="utf-8", newline="\n")
    return p


def test_the_two_tables_are_separated(export):
    """They are printed side by side and padded to equal height. Treating a
    row as one record silently pairs an entry with an unrelated player."""
    entries, players = contests.load(export)
    assert len(entries) == 3
    assert len(players) == 3
    assert list(entries.columns) == ["rank", "entry_id", "entry_name",
                                     "points", "lineup"]
    assert entries["points"].tolist() == [215.18, 209.06, 100.00]


def test_a_lineup_splits_on_slot_labels_not_spaces(export):
    """Names carry spaces, apostrophes and suffixes, and nothing separates one
    player from the next slot label. Only the labels are unambiguous."""
    entries, _ = contests.load(export)
    got = contests.parse_lineup(entries["lineup"].iloc[0])
    assert got == [
        ("DST", "Panthers"), ("FLEX", "Ashton Jeanty"), ("QB", "Bryce Young"),
        ("RB", "D'Andre Swift"), ("RB", "Aaron Jones Sr."),
        ("TE", "Dalton Schultz"), ("WR", "Jaxon Smith-Njigba"),
        ("WR", "CeeDee Lamb"), ("WR", "Jalen Coker"),
    ]


def test_a_nine_man_roster_parses_as_nine(export):
    entries, _ = contests.load(export)
    for text in entries["lineup"]:
        assert len(contests.parse_lineup(text)) == 9


def test_ownership_sums_the_slots_a_player_can_fill(export):
    """DraftKings reports a receiver at WR and again at FLEX. The share of
    lineups containing him at all is the sum, and that is what leverage
    depends on."""
    _, players = contests.load(export)
    own = contests.ownership(players)
    bijan = own[own["player_name"] == "Bijan Robinson"].iloc[0]
    assert bijan["pct_drafted"] == pytest.approx(0.4292 + 0.0486)


def test_percentages_become_fractions(export):
    _, players = contests.load(export)
    assert players["pct_drafted"].max() <= 1.0


def test_the_cash_line_is_the_last_paid_score(export):
    entries, _ = contests.load(export)
    s = contests.field_summary(entries, paid_positions=2)
    assert s["cash_line"] == pytest.approx(209.06)
    assert s["entries"] == 3


def test_field_summary_describes_the_real_distribution(export):
    entries, _ = contests.load(export)
    s = contests.field_summary(entries)
    assert s["max"] == pytest.approx(215.18)
    assert s["min"] == pytest.approx(100.00)
    assert "cash_line" not in s, "no payout given, so no line to report"


def test_a_zipped_export_reads_the_same(export, tmp_path):
    """They arrive from the browser zipped."""
    z = tmp_path / "contest-standings-999.zip"
    with zipfile.ZipFile(z, "w") as fh:
        fh.write(export, arcname=export.name)
    a, _ = contests.load(export)
    b, _ = contests.load(z)
    pd.testing.assert_frame_equal(a, b)


def test_an_empty_lineup_string_is_not_a_crash():
    assert contests.parse_lineup("") == []
    assert contests.parse_lineup(None) == []


SHOWDOWN_EXPORT = (
    "Rank,EntryId,EntryName,TimeRemaining,Points,Lineup,,Player,"
    "Roster Position,%Drafted,FPTS\n"
    "1,111,alice,0,111.98,"
    "CPT Jaylen Warren FLEX Aaron Rodgers FLEX Deshaun Watson "
    "FLEX Quinshon Judkins FLEX Roman Wilson FLEX Darnell Washington,,"
    "Jaylen Warren,CPT,84.62%,23.4\n"
    "2,222,bob,0,89.68,"
    "CPT Jaylen Warren FLEX Aaron Rodgers FLEX Deshaun Watson "
    "FLEX KC Concepcion Jr. FLEX Chris Boswell FLEX Andre Szmyt,,"
    "Jaylen Warren,FLEX,15.38%,15.6\n"
    "3,333,carol,0,75.58,"
    "CPT Deshaun Watson FLEX Jaylen Warren FLEX Denzel Boston "
    "FLEX Harold Fannin Jr. FLEX Chris Boswell FLEX Browns,,"
    "Steelers ,FLEX,35.90%,6\n"
)


@pytest.fixture
def showdown(tmp_path):
    p = tmp_path / "contest-standings-888.csv"
    p.write_text(SHOWDOWN_EXPORT, encoding="utf-8", newline="\n")
    return p


def test_a_showdown_roster_is_six_with_a_captain(showdown):
    entries, _ = contests.load(showdown)
    got = contests.parse_lineup(entries["lineup"].iloc[0])
    assert len(got) == 6
    assert got[0] == ("CPT", "Jaylen Warren")
    assert [s for s, _ in got[1:]] == ["FLEX"] * 5


def test_the_captains_multiplier_is_divided_back_out(showdown):
    """DK reports the Captain's row at 1.5x, because that is what the Captain
    scored. Read as the player's points it inflates him by half -- Aaron
    Rodgers at 35.94 against a real 23.96 -- and every comparison against a box
    score silently fails."""
    _, players = contests.load(showdown)
    cpt = players[players["slot"] == "CPT"].iloc[0]
    assert cpt["dk_scored"] == pytest.approx(23.4), "the row is left as DK wrote it"
    assert cpt["base_points"] == pytest.approx(15.6), "the player scored 15.6"

    own = contests.ownership(players)
    warren = own[own["player_name"] == "Jaylen Warren"].iloc[0]
    assert warren["dk_scored"] == pytest.approx(15.6)
    assert warren["pct_drafted"] == pytest.approx(0.8462 + 0.1538)


def test_a_classic_slot_is_left_alone(export):
    """Every Classic slot pays 1x, so nothing should be divided."""
    _, players = contests.load(export)
    assert (players["base_points"] == players["dk_scored"]).all()

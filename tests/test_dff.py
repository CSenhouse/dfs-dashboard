"""Importing Daily Fantasy Fuel's projection export.

The first real export collected for this module was a trap: 53 rows, every
projection exactly 0.0, ownership entirely blank, taken six days before its
slate. Loaded without checking, that is not a worse set of projections -- it
is the claim that nobody on the slate will score, and an optimizer handed it
returns the nine cheapest names. Most of these tests are about refusing it.
"""
from __future__ import annotations

import pandas as pd
import pytest

from dfs.sources import dff

HEAD = ("first_name,last_name,position,injury_status,week,game_date,slate,"
        "team,opp,spread,over_under,implied_team_score,salary,L5_dvp_rank,"
        "L5_fppg_avg,L10_fppg_avg,szn_fppg_avg,ppg_projection,"
        "value_projection,ownership_projection\n")


def row(first, last, pos, team, opp, salary, proj, own="", status=""):
    return (f"{first},{last},{pos},{status},5,2026-10-11,,{team},{opp},"
            f"-2.5,45.5,24,{salary},12,9.0,9.5,9.2,{proj},"
            f"{proj / (salary / 1000) if salary else 0:.2f},{own}\n")


def write(tmp_path, rows, name="DFF_NFL_cheatsheet_2026-10-11.csv"):
    p = tmp_path / name
    p.write_text(HEAD + "".join(rows), encoding="utf-8", newline="\n")
    return p


@pytest.fixture
def ready(tmp_path):
    """A posted slate: 60 players carrying real projections."""
    rows = [row("First%d" % i, "Last%d" % i, "WR", "CIN", "JAX",
                3000 + 100 * i, 8.0 + 0.1 * i, own=12.5) for i in range(60)]
    return write(tmp_path, rows)


def test_the_real_unposted_export_is_refused(tmp_path):
    """53 rows, every projection 0.0 -- exactly the file that arrived."""
    rows = [row("Tyler", "Conklin", "TE", "DET", "ARI", 2500, 0.0,
                status="O")] * 53
    p = write(tmp_path, rows)
    with pytest.raises(ValueError, match="before DFF posted the slate"):
        dff.load(p)


def test_an_unposted_export_can_still_be_inspected(tmp_path):
    rows = [row("Tyler", "Conklin", "TE", "DET", "ARI", 2500, 0.0)] * 53
    got = dff.load(write(tmp_path, rows), force=True)
    assert len(got) == 53
    assert (got["dff_proj"] == 0.0).all()


def test_a_handful_of_projected_players_is_not_a_slate(tmp_path):
    """An export listing only the injury-flagged players has real numbers on a
    few of them. It is still not something to build a lineup from."""
    rows = [row("A%d" % i, "B%d" % i, "WR", "CIN", "JAX", 3000, 10.0)
            for i in range(12)]
    with pytest.raises(ValueError, match="incomplete export"):
        dff.load(write(tmp_path, rows))


def test_a_posted_slate_loads(ready):
    got = dff.load(ready)
    assert len(got) == 60
    assert got["dff_proj"].max() > 8.0
    assert set(["player_name", "position", "team", "salary",
                "dff_proj", "dff_own"]).issubset(got.columns)


def test_names_are_joined_from_the_two_halves(tmp_path):
    p = write(tmp_path, [row("Ja'Marr", "Chase", "WR", "CIN", "JAX", 8100,
                             22.4)] * 60)
    got = dff.read(p)
    assert got["player_name"].iloc[0] == "Ja'Marr Chase"


def test_team_codes_are_normalised(tmp_path):
    """DFF writes LAR and JAC where the slate says LA and JAX. Left alone,
    every Rams and Jaguars player fails to match."""
    p = write(tmp_path, [row("A", "B", "WR", "LAR", "JAC", 5000, 12.0)] * 60)
    got = dff.read(p)
    assert got["team"].iloc[0] == "LA"
    assert got["opponent"].iloc[0] == "JAX"


def test_ownership_is_reported_as_absent_when_it_is(tmp_path):
    """The column exists on the free tier and is empty there. Whether the one
    input we cannot derive is actually present has to be checkable."""
    blank = write(tmp_path, [row("A%d" % i, "B%d" % i, "WR", "CIN", "JAX",
                                 3000, 10.0) for i in range(60)])
    assert not dff.has_ownership(dff.load(blank))


def test_ownership_is_reported_as_present_when_it_is(ready):
    assert dff.has_ownership(dff.load(ready))


def test_the_join_uses_salary_and_team_not_the_name(tmp_path):
    """Rule 1. Both sides carry DraftKings' own salary, so the key can be DK's
    data on both halves rather than two sources spelling a person."""
    p = write(tmp_path, [row("J", "Chase", "WR", "CIN", "JAX", 8100, 22.4)]
              + [row("A%d" % i, "B%d" % i, "WR", "CIN", "JAX", 3000 + i * 100,
                     9.0) for i in range(60)])
    slate = pd.DataFrame({
        "player_dk_id": [11, 12],
        "player_name": ["Ja'Marr Chase", "Somebody Else"],   # spelled differently
        "position": ["WR", "WR"], "team": ["CIN", "CIN"],
        "salary": [8100, 3000],
    })
    got = dff.match(dff.load(p), slate)
    chase = got[got["salary"] == 8100].iloc[0]
    assert chase["matched"]
    assert chase["player_dk_id"] == 11
    assert chase["player_name_dk"] == "Ja'Marr Chase"


def test_an_ambiguous_salary_matches_nobody_rather_than_guessing(tmp_path):
    """Two players at the same team, position and salary cannot be told apart
    by this key, so neither is matched. Guessing is how rule 1 got written."""
    p = write(tmp_path, [row("A", "B", "WR", "CIN", "JAX", 5000, 10.0)]
              + [row("C%d" % i, "D%d" % i, "WR", "CIN", "JAX", 3000 + i * 100,
                     9.0) for i in range(60)])
    slate = pd.DataFrame({
        "player_dk_id": [21, 22],
        "player_name": ["One Guy", "Other Guy"],
        "position": ["WR", "WR"], "team": ["CIN", "CIN"],
        "salary": [5000, 5000],
    })
    got = dff.match(dff.load(p), slate)
    assert not got[got["salary"] == 5000]["matched"].any()


def test_a_missing_column_is_an_error_not_a_silent_nan(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("first_name,last_name\nJa'Marr,Chase\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        dff.read(p)


def test_generational_suffixes_do_not_break_the_tiebreaker():
    """DFF writes "Aaron Jones Sr." where DK may write "Aaron Jones". Inside a
    crowded salary cell the name is the only separator, so a suffix mismatch
    silently loses the player. This caught a regex whose word boundaries had
    been replaced with literal backspace characters, which matched nothing and
    stripped no suffix at all."""
    got = dff._key_name(pd.Series(
        ["Aaron Jones Sr.", "Aaron Jones", "Montorie Foster Jr.",
         "Marvin Harrison III", "Ja'Marr Chase"]))
    assert got.tolist() == ["aaronjones", "aaronjones", "montoriefoster",
                            "marvinharrison", "jamarrchase"]


def test_a_crowded_salary_cell_is_separated_by_name(tmp_path):
    """Only 41% of a 629-player Classic slate sits in a unique (team,
    position, salary) cell -- the minimum salary collects six Arizona
    receivers at once. Without a second pass most of the slate goes
    unmatched."""
    rows = [row("Devin", "Duvernay", "WR", "ARI", "DET", 3000, 6.0),
            row("Simi", "Fehoko", "WR", "ARI", "DET", 3000, 5.0)]
    rows += [row("A%d" % i, "B%d" % i, "WR", "CIN", "JAX", 4000 + i * 100,
                 9.0) for i in range(60)]
    slate = pd.DataFrame({
        "player_dk_id": [31, 32],
        "player_name": ["Devin Duvernay", "Simi Fehoko"],
        "position": ["WR", "WR"], "team": ["ARI", "ARI"],
        "salary": [3000, 3000],
    })
    got = dff.match(dff.load(write(tmp_path, rows)), slate)
    hit = got[got["matched"]]
    assert set(hit["player_dk_id"]) == {31, 32}
    assert (hit["matched_on"] == "name").all()
    d = got[got["player_name"] == "Devin Duvernay"].iloc[0]
    assert d["player_dk_id"] == 31


def test_two_identical_names_in_one_cell_still_match_nobody(tmp_path):
    """The tiebreaker is only safe because the cell already fixes team,
    position and price. If it cannot separate them either, resolve to neither
    -- that is what rule 1 is for."""
    rows = [row("Mike", "Williams", "WR", "ARI", "DET", 3000, 6.0)]
    rows += [row("A%d" % i, "B%d" % i, "WR", "CIN", "JAX", 4000 + i * 100,
                 9.0) for i in range(60)]
    slate = pd.DataFrame({
        "player_dk_id": [41, 42],
        "player_name": ["Mike Williams", "Mike Williams"],
        "position": ["WR", "WR"], "team": ["ARI", "ARI"],
        "salary": [3000, 3000],
    })
    got = dff.match(dff.load(write(tmp_path, rows)), slate)
    assert not got[got["player_name"] == "Mike Williams"]["matched"].any()


def test_one_export_covers_the_whole_week(tmp_path, monkeypatch):
    """The filename carries the week's FIRST game date; the file named
    2026-10-08 also holds Sunday and Monday. Keying the lookup on the filename
    misses every Sunday slate, which is most of them."""
    monkeypatch.setattr(dff, "DIR", tmp_path)
    rows = [row("A%d" % i, "B%d" % i, "WR", "CIN", "JAX", 3000 + i * 100, 9.0)
            for i in range(60)]
    rows = [r.replace(",2026-10-11,", ",2026-10-11,") for r in rows]
    thu = [row("T%d" % i, "U%d" % i, "WR", "PIT", "CLE", 3000 + i * 100, 9.0)
           .replace(",2026-10-11,", ",2026-10-08,") for i in range(20)]
    write(tmp_path, thu + rows, name="DFF_NFL_cheatsheet_2026-10-08.csv")

    sunday, why = dff.for_date("2026-10-11")
    assert why == "" and len(sunday) == 60

    thursday, why = dff.for_date("2026-10-08")
    assert why == "" and len(thursday) == 20, "a 2-team slate is still a slate"

    missing, why = dff.for_date("2026-10-04")
    assert missing.empty and "no rows for 2026-10-04" in why


def test_a_single_game_slate_is_not_an_incomplete_export(tmp_path):
    """A flat 50-projection floor rejected a perfectly good Thursday Showdown
    export, which prices about 45 players and projects fewer."""
    rows = [row("A%d" % i, "B%d" % i, "WR", "PIT", "CLE", 3000 + i * 100, 9.0)
            for i in range(20)]
    ok, why = dff.usable(dff.read(write(tmp_path, rows)))
    assert ok, why


def test_the_match_rate_ignores_games_this_slate_does_not_have(tmp_path):
    """An export covers every game that day; a DK slate is a subset. The raw
    rate read 82.9% on the 1pm main slate purely because the file also held
    the 9:30am London game."""
    rows = [row("On%d" % i, "Slate%d" % i, "WR", "CIN", "MIA", 3000 + i * 100,
                9.0) for i in range(60)]
    rows += [row("Off", "Slate", "WR", "PHI", "JAX", 7000, 15.0)]
    slate = pd.DataFrame({
        "player_dk_id": range(100, 160),
        "player_name": ["On%d Slate%d" % (i, i) for i in range(60)],
        "position": ["WR"] * 60, "team": ["CIN"] * 60,
        "salary": [3000 + i * 100 for i in range(60)],
    })
    got = dff.match(dff.load(write(tmp_path, rows)), slate)
    assert got["matched"].mean() < 1.0, "the PHI player cannot match"
    assert dff.match_rate(got) == pytest.approx(1.0)
    assert not got[got["team"] == "PHI"]["on_slate"].any()


def test_an_upload_is_validated_before_it_lands(tmp_path, monkeypatch):
    """A silently empty export is worse than none, so a file that cannot be
    parsed must never reach the folder the dashboard reads."""
    monkeypatch.setattr(dff, "DIR", tmp_path)
    with pytest.raises(ValueError, match="missing columns"):
        dff.save("bad.csv", b"a,b\n1,2\n")
    assert list(tmp_path.glob("*")) == [], "nothing written, no .part left over"

    with pytest.raises(ValueError, match="not a .csv"):
        dff.save("notes.txt", b"whatever")


def test_a_good_upload_is_stored_and_described(tmp_path, monkeypatch):
    monkeypatch.setattr(dff, "DIR", tmp_path)
    rows = [row("A%d" % i, "B%d" % i, "WR", "CIN", "JAX", 3000 + i * 100, 9.0)
            for i in range(60)]
    raw = (HEAD + "".join(rows)).encode("utf-8")
    saved, info = dff.save("DFF_NFL_cheatsheet_2026-10-11.csv", raw)
    assert saved.exists() and saved.name.endswith(".csv")
    assert info["rows"] == 60
    assert info["projected"] == 60
    assert info["dates"] == ["2026-10-11"]
    assert info["ownership"] is False
    assert dff.available() == [saved]


def test_an_unposted_upload_is_kept_rather_than_thrown_away(tmp_path, monkeypatch):
    """It parses, so it is a real file -- just not ready. Keeping it means the
    user does not have to re-download when DFF post the slate... and the
    refusal still happens at read time, where it belongs."""
    monkeypatch.setattr(dff, "DIR", tmp_path)
    rows = [row("A%d" % i, "B%d" % i, "WR", "CIN", "JAX", 3000, 0.0)
            for i in range(53)]
    saved, info = dff.save("DFF_NFL_cheatsheet_2026-10-11.csv",
                           (HEAD + "".join(rows)).encode("utf-8"))
    assert saved.exists()
    assert info["projected"] == 0
    got, why = dff.for_date("2026-10-11")
    assert got.empty and "before DFF posted the slate" in why


def test_a_slate_is_found_by_its_teams_not_its_date(tmp_path, monkeypatch):
    """Dates get this wrong twice. A slate's start_time is UTC while DFF write
    the Eastern game date, so a Thursday night slate starting 2026-10-09T00:15Z
    looked for 2026-10-09 while the export said 2026-10-08 -- that is the real
    bug the dashboard hit. And a Thu-Mon slate spans three dates, so matching
    the first would pull thirty Thursday players for a fifteen-game slate."""
    monkeypatch.setattr(dff, "DIR", tmp_path)
    # Both sides of each game, as a real export has.
    thu = [row("T%d" % i, "U%d" % i, "WR", t, o, 3000 + i * 100, 9.0)
           .replace(",2026-10-11,", ",2026-10-08,")
           for i in range(10) for t, o in (("PIT", "CLE"), ("CLE", "PIT"))]
    sun = [row("A%d" % i, "B%d" % i, "WR", t, o, 3000 + i * 100, 9.0)
           for i in range(30) for t, o in (("CIN", "MIA"), ("MIA", "CIN"))]
    write(tmp_path, thu + sun, name="DFF_NFL_cheatsheet_2026-10-08.csv")

    both, why = dff.for_teams(["PIT", "CLE", "CIN", "MIA"])
    assert why == "" and len(both) == 80, "a multi-day slate gets every day"

    one, why = dff.for_teams(["PIT", "CLE"])
    assert why == "" and len(one) == 20


def test_an_export_for_a_different_slate_is_refused(tmp_path, monkeypatch):
    """Covering a couple of a slate's teams is not covering the slate."""
    monkeypatch.setattr(dff, "DIR", tmp_path)
    rows = [row("A%d" % i, "B%d" % i, "WR", "CIN", "MIA", 3000 + i * 100, 9.0)
            for i in range(60)]
    write(tmp_path, rows)
    got, why = dff.for_teams(["CIN", "MIA", "DAL", "PHI", "GB", "CHI",
                              "SEA", "SF", "DEN", "KC"])
    assert got.empty and "different slate" in why


def test_team_aliases_apply_to_the_lookup_too(tmp_path, monkeypatch):
    """The slate says LA, the export says LAR. Normalising only one side
    leaves every Rams slate unmatched."""
    monkeypatch.setattr(dff, "DIR", tmp_path)
    rows = [row("A%d" % i, "B%d" % i, "WR", t, o, 3000 + i * 100, 9.0)
            for i in range(30) for t, o in (("LAR", "SF"), ("SF", "LAR"))]
    write(tmp_path, rows)
    got, why = dff.for_teams(["LA", "SF"])
    assert why == "" and len(got) == 60
    assert set(got["team"]) == {"LA", "SF"}, "LAR normalised on both sides"


def test_salary_cannot_be_the_key_across_formats(tmp_path):
    """DFF publish DraftKings' CLASSIC price. A Showdown slate prices the same
    player completely differently -- Dak Prescott is $6,700 on the main slate
    and $10,400 single-game -- so the salary key matched 2 of 30 on a real
    Showdown slate and the override silently did nothing. Fall back to the
    name within (team, position), which still cannot make rule 1's mistake:
    that was a receiver paired with a defensive back, and both fields agree
    here."""
    named = [("Dak", "Prescott", "QB", "DAL", "TB", 6700, 10400),
             ("CeeDee", "Lamb", "WR", "DAL", "TB", 8300, 11800),
             ("Baker", "Mayfield", "QB", "TB", "DAL", 5200, 9600)]
    # Distinct ALPHABETIC names: _key_name strips digits, so "A0 B0" and
    # "A1 B1" both reduce to "ab" and are refused as ambiguous -- correctly,
    # but it makes them useless as filler.
    _alpha = ["Alfa", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot", "Golf",
              "Hotel", "India", "Juliet", "Kilo", "Lima", "Mike", "November",
              "Oscar", "Papa", "Quebec"]
    filler = [(n, n + "son", "WR", "DAL" if i % 2 else "TB",
               "TB" if i % 2 else "DAL", 3000 + i * 100, 4500 + i * 150)
              for i, n in enumerate(_alpha)]
    both = named + filler
    rows = [row(f, l, pos, tm, op, classic, 8.0) for f, l, pos, tm, op,
            classic, _ in both]
    # Every one of them priced as a single game: nothing shares a salary with
    # the export.
    showdown = pd.DataFrame({
        "player_dk_id": list(range(1, len(both) + 1)),
        "player_name": [f + " " + l for f, l, *_ in both],
        "position": [pos for _, _, pos, *_ in both],
        "team": [tm for _, _, _, tm, *_ in both],
        "salary": [sd for *_, sd in both],
    })
    got = dff.match(dff.load(write(tmp_path, rows)), showdown)
    assert dff.match_rate(got) == pytest.approx(1.0), "every player, by name"
    assert (got.loc[got["matched"], "matched_on"] == "name across prices").all()
    dak = got[got["player_name"] == "Dak Prescott"].iloc[0]
    assert dak["player_dk_id"] == 1, "$6,700 in the export, $10,400 on the slate"


def test_the_cross_format_pass_still_refuses_an_ambiguous_name(tmp_path):
    """Two players sharing a team, a position and a name resolve to neither,
    whichever pass is doing the work."""
    rows = [row("Mike", "Williams", "WR", "DAL", "TB", 5000, 10.0)]
    rows += [row("A%d" % i, "B%d" % i, "WR", "DAL", "TB", 3000 + i * 100, 8.0)
             for i in range(60)]
    slate = pd.DataFrame({
        "player_dk_id": [7, 8],
        "player_name": ["Mike Williams", "Mike Williams"],
        "position": ["WR", "WR"], "team": ["DAL", "DAL"],
        "salary": [9000, 9400],          # different prices, same name
    })
    got = dff.match(dff.load(write(tmp_path, rows)), slate)
    assert not got[got["player_name"] == "Mike Williams"]["matched"].any()


def test_salary_still_wins_when_the_formats_agree(tmp_path):
    """The cross-format pass is a fallback, not a replacement: on a Classic
    slate the salary key should still do the primary work."""
    rows = [row("A%d" % i, "B%d" % i, "WR", "CIN", "MIA", 3000 + i * 100, 9.0)
            for i in range(60)]
    slate = pd.DataFrame({
        "player_dk_id": range(200, 260),
        "player_name": ["Different Spelling %d" % i for i in range(60)],
        "position": ["WR"] * 60, "team": ["CIN"] * 60,
        "salary": [3000 + i * 100 for i in range(60)],
    })
    got = dff.match(dff.load(write(tmp_path, rows)), slate)
    assert (got["matched_on"] == "salary").all(), "names disagree entirely"
    assert dff.match_rate(got) == pytest.approx(1.0)


def test_a_file_uploaded_for_a_slate_wins_over_the_week_file(tmp_path, monkeypatch):
    """DFF's per-slate download carries THAT slate's salaries. For a Showdown
    that is the difference between the price key working and matching two
    players in thirty (section 32), so it has to take precedence."""
    monkeypatch.setattr(dff, "DIR", tmp_path)
    week = [row("W%s" % n, n + "son", "WR",
                "DAL" if i % 2 else "TB", "TB" if i % 2 else "DAL",
                3000 + i * 100, 8.0)
            for i, n in enumerate(["Alfa", "Bravo", "Charlie", "Delta",
                                   "Echo", "Foxtrot", "Golf", "Hotel",
                                   "India", "Juliet", "Kilo", "Lima",
                                   "Mike", "November", "Oscar", "Papa"])]
    dff.save("DFF_NFL_cheatsheet_2026-10-11.csv",
             (HEAD + "".join(week)).encode("utf-8"))

    only = [row("S%s" % n, n + "ton", "WR",
                "DAL" if i % 2 else "TB", "TB" if i % 2 else "DAL",
                9000 + i * 200, 12.0)
            for i, n in enumerate(["Romeo", "Sierra", "Tango", "Uniform",
                                   "Victor", "Whiskey", "Xray", "Yankee",
                                   "Zulu", "Anchor", "Beacon", "Cobalt",
                                   "Dagger", "Ember", "Falcon", "Granite"])]
    saved, _ = dff.save("DFF_NFL_cheatsheet_showdown.csv",
                        (HEAD + "".join(only)).encode("utf-8"), dgid=154478)
    assert saved.name.startswith(dff.slate_prefix(154478))

    got, why, src = dff.for_slate(154478, ["DAL", "TB"])
    assert why == "" and src == saved.name
    assert got["dff_proj"].iloc[0] == pytest.approx(12.0), "the slate's file"

    other, why, src = dff.for_slate(999999, ["DAL", "TB"])
    assert why == "" and src == "a week file"
    assert other["dff_proj"].iloc[0] == pytest.approx(8.0)


def test_binding_a_file_twice_does_not_stack_the_prefix(tmp_path, monkeypatch):
    monkeypatch.setattr(dff, "DIR", tmp_path)
    rows = [row("N%d" % i, "Sur%d" % i, "WR",
                "DAL" if i % 2 else "TB", "TB" if i % 2 else "DAL",
                3000 + i * 100, 8.0) for i in range(16)]
    raw = (HEAD + "".join(rows)).encode("utf-8")
    a, _ = dff.save("sheet.csv", raw, dgid=154478)
    b, _ = dff.save(a.name, raw, dgid=154478)
    assert a.name == b.name == dff.slate_prefix(154478) + "sheet.csv"
    assert len(list(tmp_path.glob("*.csv"))) == 1

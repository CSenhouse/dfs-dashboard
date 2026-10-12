"""Pricing a lineup against a real contest.

The null test is the important one here, and it is the test that was missing
when this went wrong: a lineup with no edge by construction must price at the
rake and nothing better. Run against a frozen field with a top-heavy payout,
the first version of this returned a 160% edge for an average lineup, because
every simulated score past the field's maximum was handed first place.
"""
from __future__ import annotations

import numpy as np
import pytest

from dfs import contest_ev

# A $20K Pylon: 7,927 entries at $3, 2,011 paid, $2,000 for first.
PYLON = {(1, 1): 2000.0, (2, 2): 1000.0, (3, 3): 600.0, (4, 4): 400.0,
         (5, 5): 300.0, (6, 7): 250.0, (8, 10): 200.0, (11, 13): 150.0,
         (14, 16): 100.0, (17, 20): 75.0, (21, 25): 60.0, (26, 35): 50.0}


@pytest.fixture
def pay():
    return contest_ev.payout_table(PYLON, n_paid=2011, total=20000.0, floor=5.0)


@pytest.fixture
def field():
    """A field shaped like the real ones: mean ~130, sd ~25, a left tail of
    entries that never filled a roster."""
    rng = np.random.default_rng(17)
    f = rng.normal(130.0, 25.0, 7927)
    f[:120] = rng.uniform(0.0, 60.0, 120)
    return np.sort(f)


def test_the_table_sums_to_the_advertised_prize_pool(pay):
    """The only available check that the fitted tail is right."""
    assert pay.sum() == pytest.approx(20000.0, rel=1e-6)
    assert pay[1] == 2000.0
    assert pay[2011] == pytest.approx(5.0)


def test_the_listed_tiers_are_left_exactly_as_published(pay):
    for (lo, hi), v in PYLON.items():
        assert (pay[lo:hi + 1] == v).all()


def test_a_lineup_with_no_edge_prices_at_the_rake(pay, field):
    """THE null test. Every real entry, ranked against the field it came from,
    must average the prize pool divided by the entries -- no more. The version
    of this that shipped a 160% edge would have failed here."""
    got = contest_ev.expected_value(field, field, pay, 7927, 3.0)
    assert got["baseline"] == pytest.approx(20000.0 / 7927 / 3.0)
    assert got["ev"] == pytest.approx(got["baseline"], rel=0.12)
    assert got["cash_prob"] == pytest.approx(2011 / 7927, abs=0.01)


def test_a_fat_tail_beyond_the_field_is_flagged(pay, field):
    """A simulated lineup reaching past every score the field posted is
    awarded first place for free. The number stays computable; it stops being
    trustworthy, and must say so."""
    rng = np.random.default_rng(3)
    totals = rng.normal(130.0, 45.0, 20000)       # far wider than the field
    got = contest_ev.expected_value(totals, field, pay, 7927, 3.0)
    assert got["tail_share"] > contest_ev.TAIL_LIMIT
    assert not got["trustworthy"]
    assert got["ev"] > got["baseline"], "the artefact inflates it, as it did"


def test_a_well_behaved_lineup_is_trustworthy(pay, field):
    rng = np.random.default_rng(3)
    totals = rng.normal(135.0, 22.0, 20000)
    got = contest_ev.expected_value(totals, field, pay, 7927, 3.0)
    assert got["trustworthy"]


def test_ties_count_as_half_not_as_wins(pay):
    """A converged field puts hundreds of entries on one score. Counting them
    all as beaten flatters the top of the distribution, which is where the
    money is."""
    field = np.array([100.0] * 99 + [200.0])
    assert contest_ev.percentile(np.array([100.0]), field)[0] == pytest.approx(
        (1 + 0.5 * 99) / 100)
    assert contest_ev.percentile(np.array([300.0]), field)[0] == 0.0
    assert contest_ev.percentile(np.array([50.0]), field)[0] == 1.0


def test_cash_probability_does_not_depend_on_the_tail(pay, field):
    """It survived the bug that broke EV, which is why it is the number to
    quote. Stretching only the upper tail must not move it."""
    rng = np.random.default_rng(5)
    base = rng.normal(130.0, 22.0, 40000)
    stretched = base.copy()
    hi = stretched > np.quantile(stretched, 0.95)
    stretched[hi] = np.quantile(stretched, 0.95) + (
        stretched[hi] - np.quantile(stretched, 0.95)) * 4.0
    a = contest_ev.cash_probability(base, field, 2011 / 7927)
    b = contest_ev.cash_probability(stretched, field, 2011 / 7927)
    assert a == pytest.approx(b, abs=0.005)


def test_the_scoring_environment_moves_more_than_the_lineup(pay):
    """Observed: two weeks of the same contest had field means 116.7 and
    130.4. One lineup priced against both must come out very differently, or
    the quoted single number is hiding the real uncertainty."""
    rng = np.random.default_rng(11)
    soft = np.sort(rng.normal(116.7, 26.9, 7927))
    hard = np.sort(rng.normal(130.4, 25.4, 7927))
    totals = rng.normal(130.7, 22.0, 20000)
    a = contest_ev.cash_probability(totals, soft, 2011 / 7927)
    b = contest_ev.cash_probability(totals, hard, 2011 / 7927)
    assert a - b > 0.12, "a 14-point swing in the field has to show up"


def test_a_field_rebuilt_from_its_summary_matches_it():
    """The raw exports are not committed, so the dashboard reaches real fields
    only through the recorded percentiles. They have to come back."""
    row = {"min": 0.0, "max": 219.2, "p10": 98.9, "p25": 112.3, "p50": 129.1,
           "p75": 147.3, "p90": 164.1, "p99": 190.0}
    f = contest_ev.field_from_summary(row, n=20000)
    assert f.min() == pytest.approx(0.0, abs=1.0)
    assert f.max() == pytest.approx(219.2, abs=1.0)
    for p in (10, 25, 50, 75, 90):
        assert np.percentile(f, p) == pytest.approx(row[f"p{p}"], abs=1.5)


def test_a_rebuilt_field_is_never_decreasing():
    """contest_fields.csv is generated, but a hand-edited row must not produce
    a field whose quantiles run backwards."""
    row = {"min": 0.0, "max": 200.0, "p10": 120.0, "p25": 110.0, "p50": 130.0,
           "p75": 150.0, "p90": 160.0, "p99": 180.0}
    f = contest_ev.field_from_summary(row, n=2000)
    assert (np.diff(f) >= -1e-9).all()

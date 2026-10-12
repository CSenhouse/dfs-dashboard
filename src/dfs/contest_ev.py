"""Pricing a lineup against a real contest: payout tables, rank in a counted
field, and the guard against pricing a lineup off its own tail.

Rule 14. A standings export is a fixed list of scores whose maximum is one
week's realisation. A simulated lineup has a tail that runs past it, and the
ranking awards first place every time it does -- on a real $20K contest that
returned $2.18 per $1 for a lineup whose cash rate was BELOW the field's own,
62% of it from scores beyond the field maximum. The flaw is treating our score
and the field's as independent when they come from the same games.

So `expected_value` reports `tail_share` and `trustworthy`, and must be
checked. Percentile and cash probability do not depend on the tail and were
right in the same run that got EV wrong -- prefer them.
"""
from __future__ import annotations

import numpy as np

# EV is not trusted once this share of it comes from beating the field's max.
TAIL_LIMIT = 0.10

_SUMMARY_POINTS = ((0.0, "min"), (0.10, "p10"), (0.25, "p25"), (0.50, "p50"),
                   (0.75, "p75"), (0.90, "p90"), (0.99, "p99"), (1.0, "max"))


def payout_table(tiers: dict[tuple[int, int], float], n_paid: int,
                 total: float, floor: float) -> np.ndarray:
    """Prize by finishing rank, index 0 unused.

    Published tiers are kept exactly. DK lists only the top of the table, so
    the unlisted ranks are filled with a smooth decreasing curve from the last
    listed prize down to `floor` at `n_paid`, its shape solved so the whole
    table sums to the advertised prize pool -- the only check available.
    """
    pay = np.zeros(n_paid + 1)
    last_rank, last_value = 0, floor
    for (lo, hi), v in sorted(tiers.items()):
        pay[lo:hi + 1] = v
        if hi > last_rank:
            last_rank, last_value = hi, v
    if last_rank >= n_paid:
        return pay
    ranks = np.arange(last_rank + 1, n_paid + 1)
    rest = total - pay.sum()
    frac = (n_paid - ranks) / max(n_paid - last_rank, 1)

    def fill(k: float) -> np.ndarray:
        return floor + (last_value - floor) * frac ** k

    lo_k, hi_k = 1e-3, 1e3
    for _ in range(200):
        mid = np.sqrt(lo_k * hi_k)
        if fill(mid).sum() > rest:
            lo_k = mid
        else:
            hi_k = mid
    pay[ranks] = fill(np.sqrt(lo_k * hi_k))
    return pay


def percentile(totals: np.ndarray, field: np.ndarray) -> np.ndarray:
    """Share of the field finishing AHEAD of each total, ties counted half.
    0 is first place, 1 is last; lower is better."""
    f = np.sort(np.asarray(field, dtype=float))
    t = np.atleast_1d(np.asarray(totals, dtype=float))
    above = f.size - np.searchsorted(f, t, side="right")
    ties = np.searchsorted(f, t, side="right") - np.searchsorted(f, t, side="left")
    return (above + 0.5 * ties) / f.size


def cash_probability(totals: np.ndarray, field: np.ndarray,
                     paid_fraction: float) -> float:
    return float((percentile(totals, field) <= paid_fraction).mean())


def expected_value(totals: np.ndarray, field: np.ndarray, pay: np.ndarray,
                   n_entries: int, entry_fee: float) -> dict:
    """Return per $1 staked, with the share of it that came from the tail.

    Each total is ranked into the field; ties share the prizes across the
    places they occupy (midrank), which is what DK pays.
    """
    f = np.sort(np.asarray(field, dtype=float))
    t = np.asarray(totals, dtype=float)
    n_paid = len(pay) - 1
    padded = np.zeros(max(n_entries, f.size) + 2)
    padded[1:n_paid + 1] = pay[1:n_paid + 1]
    cum = np.cumsum(padded)

    right = np.searchsorted(f, t, side="right")
    left = np.searchsorted(f, t, side="left")
    above = f.size - right
    ties = np.maximum(right - left, 1)
    first = above + 1
    last = np.minimum(above + ties, len(padded) - 1)
    won = (cum[last] - cum[first - 1]) / (last - first + 1)

    total_won = won.sum()
    beyond = won[t > f.max()].sum() if f.size else 0.0
    tail_share = float(beyond / total_won) if total_won > 0 else 0.0
    baseline = float(pay.sum() / n_entries / entry_fee)
    return {
        "ev": float(won.mean() / entry_fee),
        "baseline": baseline,
        "cash_prob": float((won > 0).mean()),
        "tail_share": tail_share,
        "trustworthy": tail_share <= TAIL_LIMIT,
    }


def field_from_summary(row, n: int = 20000) -> np.ndarray:
    """A field rebuilt from its recorded percentiles (data/contest_fields.csv).

    The raw exports are not committed, so this is how a counted field stays
    reachable. The quantile function is linear between the recorded points
    and forced non-decreasing in case a row was edited by hand.
    """
    probs = np.array([p for p, _ in _SUMMARY_POINTS])
    vals = np.array([float(row[k]) for _, k in _SUMMARY_POINTS])
    vals = np.maximum.accumulate(vals)
    u = (np.arange(n) + 0.5) / n
    u = np.r_[0.0, u[1:-1], 1.0] if n > 2 else u
    return np.maximum.accumulate(np.interp(u, probs, vals))

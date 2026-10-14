"""Correlated slate simulation: a lineup's whole distribution, not a sum of
point estimates.

Each player's outcome is P(plays) x a draw from his predicted conditional
quantile grid. Players in the same game are coupled with a Gaussian copula
whose parameters were measured on walk-forward normal scores (correlations.py)
-- what is left after each player's own projection is divided out, so the
shared signal already in both projections is not counted twice.

What it is good for (docs/MODELING.md 11): an honest range for a lineup.
What it is not: a way to pick lineups -- ranking candidates by simulated
probability went 18-16 against the optimizer's own pick (p=0.75).

The cash line is the 56.5th percentile of each simulated world's field, not
its median: a DK double-up pays exactly 2.00x to the top 43.5%.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from dfs.config import CAPTAIN_MULTIPLIER
from dfs import field as field_mod

ALPHAS = (0.05, 0.15, 0.30, 0.50, 0.70, 0.85, 0.95)
QCOLS = [f"q{int(round(a * 100)):02d}" for a in ALPHAS]
PAYOUT = 2.0
PAID_FRACTION = 0.435
LINE_PCT = 100 * (1 - PAID_FRACTION)
PIT_EPS = 1e-6

_A = np.asarray(ALPHAS)


# --- the quantile function ----------------------------------------------------

def _tails(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Left anchor Q(0) and right exponential scale for each row."""
    q0, q1 = grid[:, 0], grid[:, 1]
    left = q0 - (q1 - q0) * _A[0] / (_A[1] - _A[0])
    left = np.maximum(left, np.minimum(q0, 0.0))
    scale = (grid[:, -1] - grid[:, -2]) / np.log((1 - _A[-2]) / (1 - _A[-1]))
    return left, np.maximum(scale, 0.0)


def inverse_cdf(u: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Quantile function: row r of `u` through row r of `grid`.

    Linear between the fitted quantiles; linear to an anchor below q05; and
    an exponential tail above q95, because tournaments are decided past it and
    a linear tail would cap every ceiling just above the top quantile.
    """
    u = np.asarray(u, dtype=float)
    g = np.asarray(grid, dtype=float)
    if u.ndim == 1:
        u = u[None, :]
    left, scale = _tails(g)
    k = len(_A)
    j = np.clip(np.searchsorted(_A, u, side="right") - 1, 0, k - 2)
    rows = np.arange(g.shape[0])[:, None]
    lo, hi = g[rows, j], g[rows, j + 1]
    frac = (u - _A[j]) / (_A[j + 1] - _A[j])
    out = lo + np.clip(frac, 0.0, 1.0) * (hi - lo)

    below = u < _A[0]
    if below.any():
        lft = np.broadcast_to(left[:, None], u.shape)
        q0 = np.broadcast_to(g[:, :1], u.shape)
        out = np.where(below, lft + (q0 - lft) * (u / _A[0]), out)
    above = u > _A[-1]
    if above.any():
        sc = np.broadcast_to(scale[:, None], u.shape)
        top = np.broadcast_to(g[:, -1:], u.shape)
        tail = np.log((1 - _A[-1]) / np.clip(1 - u, 1e-12, None))
        out = np.where(above, top + sc * tail, out)
    return out


def _cdf_pair(x: np.ndarray, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(G(x-), G(x)) of the conditional distribution, per row."""
    left, scale = _tails(grid)
    knots_q = np.column_stack([left, grid])
    knots_a = np.r_[0.0, _A]
    n, m = knots_q.shape
    rows = np.arange(n)
    x = np.asarray(x, dtype=float)

    def interior(j: np.ndarray) -> np.ndarray:
        jj = np.clip(j, 0, m - 2)
        q_lo, q_hi = knots_q[rows, jj], knots_q[rows, jj + 1]
        width = np.where(q_hi > q_lo, q_hi - q_lo, 1.0)
        a = knots_a[jj] + (x - q_lo) / width * (knots_a[jj + 1] - knots_a[jj])
        return np.clip(a, knots_a[jj], knots_a[jj + 1])

    def right_tail() -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            a = 1 - (1 - _A[-1]) * np.exp(-(x - grid[:, -1])
                                          / np.where(scale > 0, scale, 1.0))
        return np.where(scale > 0, a, 1.0)

    le = (knots_q <= x[:, None]).sum(axis=1) - 1
    lt = (knots_q < x[:, None]).sum(axis=1) - 1
    g_hi = np.where(le < 0, 0.0,
                    np.where(le >= m - 1, right_tail(), interior(le)))
    g_lo = np.where(lt < 0, 0.0,
                    np.where(lt >= m - 1, right_tail(), interior(lt)))
    return np.clip(g_lo, 0, 1), np.clip(g_hi, 0, 1)


def pit(scores: np.ndarray, grid: np.ndarray, p_play: np.ndarray,
        seed: int = 0) -> np.ndarray:
    """Randomized probability integral transform of observed scores.

    The predicted distribution is a mixture: 1 - p_play of a point mass at
    zero ("did not play"), p_play of the conditional grid. Half of priced
    players score exactly 0.00, so the point mass is most of the data, and a
    PIT that ignored it would pile up at one end.
    """
    g = np.sort(np.asarray(grid, dtype=float), axis=1)
    x = np.asarray(scores, dtype=float)
    p = np.clip(np.asarray(p_play, dtype=float), 0.0, 1.0)
    gm, gp = _cdf_pair(x, g)
    lower = p * gm + (1 - p) * (x > 0)
    upper = p * gp + (1 - p) * (x >= 0)
    r = np.random.default_rng(seed).random(x.size)
    # Strictly inside (0, 1). A score outside the predicted support -- a
    # negative day against a grid that starts at zero -- would otherwise land
    # on exactly 0 or 1, whose normal score is infinite, and one infinite row
    # turns a whole correlation cell into NaN.
    return np.clip(lower + r * (upper - lower), PIT_EPS, 1.0 - PIT_EPS)


# --- roles and correlation ---------------------------------------------------

SINGLE = {"QB", "DST", "K"}


def roles(df: pd.DataFrame) -> pd.Series:
    """Depth role within team and position, ranked by PROJECTION -- never by
    outcome, which would define the depth chart with hindsight."""
    key = next((c for c in ("projection", "proj_mean", "points_mean", "salary")
                if c in df.columns), None)
    rank = (df.groupby(["team", "position"])[key]
            .rank(method="first", ascending=False).fillna(1).astype(int)
            if key else pd.Series(1, index=df.index))
    pos = df["position"].astype(str)
    return pd.Series(
        [p if (p in SINGLE and r == 1) else f"{p}{r}"
         for p, r in zip(pos, rank)], index=df.index)


def _game_key(df: pd.DataFrame) -> pd.Series:
    t = df["team"].astype(str)
    if "opponent" in df.columns:
        o = df["opponent"].fillna("").astype(str)
        return pd.Series(np.where(o == "", t, np.where(t < o, t + "@" + o,
                                                       o + "@" + t)),
                         index=df.index)
    return t


def _default_table() -> dict:
    try:
        from dfs.correlations import CORRELATION
        return CORRELATION
    except ImportError:
        return {}


def _nearest_psd(a: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Project a symmetric matrix to a correlation matrix. Pairwise numbers
    need not describe a joint distribution, and Cholesky fails outright on
    one that does not."""
    a = (a + a.T) / 2
    w, v = np.linalg.eigh(a)
    fixed = (v * np.clip(w, eps, None)) @ v.T
    d = np.sqrt(np.diag(fixed))
    fixed = fixed / np.outer(d, d)
    np.fill_diagonal(fixed, 1.0)
    return fixed


def correlation_matrix(pool: pd.DataFrame, table: dict | None = None
                       ) -> np.ndarray:
    table = _default_table() if table is None else table
    n = len(pool)
    c = np.eye(n)
    if n == 0 or not table:
        return c
    p = pool.reset_index(drop=True)
    role = roles(p).to_numpy()
    team = p["team"].astype(str).to_numpy()
    game = _game_key(p).to_numpy()
    for g in np.unique(game):
        members = np.flatnonzero(game == g)
        if members.size < 2:
            continue
        for a_i, i in enumerate(members):
            for j in members[a_i + 1:]:
                rel = "team" if team[i] == team[j] else "opp"
                lo, hi = sorted((role[i], role[j]))
                r = table.get((rel, lo, hi))
                if r:
                    c[i, j] = c[j, i] = r
        block = np.ix_(members, members)
        if np.linalg.eigvalsh(c[block]).min() < 1e-9:
            c[block] = _nearest_psd(c[block])
    return c


# --- the spread correction ---------------------------------------------------

def spread_factors(pool: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(lower, upper) widening per player, by projection decile on the slate."""
    from dfs import spread
    n = len(pool)
    if not spread.SPREAD_LO or n == 0:
        return np.ones(n), np.ones(n)
    key = "projection" if "projection" in pool.columns else "proj_mean"
    r = pool[key].rank(method="first").to_numpy()
    b = np.minimum(((r - 1) * spread.BUCKETS / n).astype(int),
                   spread.BUCKETS - 1)
    lo = np.array([spread.SPREAD_LO.get(int(x), 1.0) for x in b])
    hi = np.array([spread.SPREAD_HI.get(int(x), 1.0) for x in b])
    return lo, hi


# --- sampling ------------------------------------------------------------------

class Simulation:
    """Sampled outcomes: scores[s, j] is player j's points in world s."""

    def __init__(self, scores: np.ndarray, ids: np.ndarray):
        self.scores = scores
        self.ids = np.asarray(ids)
        self._pos = {int(x): k for k, x in enumerate(self.ids)}

    def columns(self, ids) -> np.ndarray:
        return np.array([self._pos[int(i)] for i in ids], dtype=int)

    def lineup_totals(self, ids, captain_id: int | None = None) -> np.ndarray:
        cols = self.columns(ids)
        total = self.scores[:, cols].sum(axis=1)
        if captain_id is not None:
            total = total + (CAPTAIN_MULTIPLIER - 1.0) * \
                self.scores[:, self._pos[int(captain_id)]]
        return total

    def entry_totals(self, entries: np.ndarray,
                     weights: np.ndarray | None = None) -> np.ndarray:
        """(n_sims, n_entries) totals for many entries given as column
        indices, optionally weighted (a Showdown captain)."""
        entries = np.asarray(entries, dtype=int)
        if entries.size == 0:
            return np.zeros((self.scores.shape[0], 0))
        out = np.empty((self.scores.shape[0], entries.shape[0]))
        for k in range(entries.shape[0]):
            cols = entries[k]
            if weights is None:
                out[:, k] = self.scores[:, cols].sum(axis=1)
            else:
                out[:, k] = self.scores[:, cols] @ np.asarray(weights[k], float)
        return out


def sample(pool: pd.DataFrame, n_sims: int = 4000, seed: int = 0,
           table: dict | None = None, correlated: bool = True,
           calibrated: bool = True) -> Simulation:
    """Simulate every player in `pool` n_sims times."""
    for col in QCOLS:
        if col not in pool.columns:
            raise KeyError(f"quantile column {col} missing; sampling needs the "
                           f"full grid {QCOLS}")
    p = pool.reset_index(drop=True)
    n = len(p)
    rng = np.random.default_rng(seed)
    grid = np.sort(p[QCOLS].to_numpy(dtype=float), axis=1)
    play = (p["p_play"].to_numpy(dtype=float) if "p_play" in p.columns
            else np.ones(n))
    play = np.nan_to_num(play, nan=0.0)

    z = rng.standard_normal((n_sims, n))
    if correlated and n > 1:
        c = correlation_matrix(p, table)
        if not np.allclose(c, np.eye(n)):
            game = _game_key(p).to_numpy()
            for g in np.unique(game):
                m = np.flatnonzero(game == g)
                if m.size < 2:
                    continue
                block = c[np.ix_(m, m)]
                if np.allclose(block, np.eye(m.size)):
                    continue
                L = np.linalg.cholesky(block)
                z[:, m] = z[:, m] @ L.T
    if calibrated:
        lo, hi = spread_factors(p)
        z = np.where(z < 0, z * lo, z * hi)

    u = stats.norm.cdf(z)
    pts = inverse_cdf(u.T, grid).T
    plays = rng.random((n_sims, n)) < play
    scores = np.where(plays, pts, 0.0)
    return Simulation(scores, p["player_dk_id"].to_numpy())


def lineup_ids(lu: pd.DataFrame) -> tuple[list[int], int | None]:
    ids = [int(x) for x in lu["player_dk_id"]]
    captain = None
    if "slot" in lu.columns:
        cpt = lu[lu["slot"] == "CPT"]
        if len(cpt):
            captain = int(cpt["player_dk_id"].iloc[0])
    return ids, captain


def field_entries(pool: pd.DataFrame, sim: Simulation, n: int = 1000,
                  seed: int = 0, showdown: bool = False,
                  field_skill: float = 3.0) -> np.ndarray:
    """(n_sims, n) totals of a sampled field, re-scored in every simulated
    world so the line moves with the slate."""
    p = pool.reset_index(drop=True)
    if showdown:
        idx, w = field_mod.sample_showdown_indices(p, n=n,
                                                   field_skill=field_skill,
                                                   seed=seed)
    else:
        idx = field_mod.sample_lineup_indices(p, n=n, field_skill=field_skill,
                                              seed=seed)
        w = None
    if idx.size == 0:
        return np.zeros((sim.scores.shape[0], 0))
    cols = sim.columns(p["player_dk_id"].to_numpy())[idx]
    return sim.entry_totals(cols, w)


# --- reading a decision off it -------------------------------------------------

def value(totals: np.ndarray, opp: np.ndarray | None = None,
          target_line: float | None = None) -> dict:
    """A lineup's range and, against a simulated field, its cash chances.

    `target_line` moves the OPPONENTS so the median world's line sits where
    it was asked for, keeping its week-to-week movement; the lineup's own
    range is untouched.
    """
    t = np.asarray(totals, dtype=float)
    out = {"mean": float(t.mean()), "sd": float(t.std()),
           "p10": float(np.percentile(t, 10)), "p50": float(np.percentile(t, 50)),
           "p90": float(np.percentile(t, 90))}
    if opp is None:
        return out
    o = np.asarray(opp, dtype=float)
    if o.ndim != 2 or o.shape[1] == 0:
        return out
    lines = np.percentile(o, LINE_PCT, axis=1)
    simulated = float(np.median(lines))
    shift = 0.0 if target_line is None else float(target_line) - simulated
    lines = lines + shift
    top1 = np.percentile(o, 99, axis=1) + shift
    p_cash = float((t > lines).mean())
    out.update({
        "p_cash": p_cash,
        "cash_roi": p_cash * PAYOUT - 1.0,
        "p_top1": float((t > top1).mean()),
        "cash_line": float(np.median(lines)),
        "simulated_line": simulated,
        "cash_line_p10": float(np.percentile(lines, 10)),
        "cash_line_p90": float(np.percentile(lines, 90)),
    })
    return out

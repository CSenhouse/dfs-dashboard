"""A simulated field of opponents, and the cash line read off it.

Two kinds of opponent:

- `sample_lineup_indices`: entrants who pick players with probability
  proportional to projection ** field_skill and repair an over-cap roster by
  swapping down to the cheapest option. Every draw is repaired rather than
  rejected, because rejection silently removes the sharp end of the field,
  where expensive players cluster.
- `optimizer_entries`: opponents who also solve the MILP, on a noisy copy of
  our projection or on a projection of their own.

Known to be too soft (docs/MODELING.md 12, 17): the sampling field medians
about 107-111 on a main slate where real double-ups cash near 140-147, and is
1.26x too narrow against the first counted field. Cash lines in the dashboard
come from data/observed_cash_lines.csv, not from here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from dfs import optimizer
from dfs.config import SALARY_CAP, SHOWDOWN_SIZE

PAID_FRACTION = 0.435          # a DK double-up pays 2.00x to the top 43.5%
CLASSIC_SLOTS = ("QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX", "DST")
FLEX = ("RB", "WR", "TE")
EMPTY = np.zeros((0, len(CLASSIC_SLOTS)), dtype=int)


def _usable(pool: pd.DataFrame) -> np.ndarray:
    """Positions (into `pool`) of rows the sampler may draw."""
    if pool is None or pool.empty:
        return np.array([], dtype=int)
    proj = pd.to_numeric(pool.get("projection"), errors="coerce").to_numpy()
    sal = pd.to_numeric(pool.get("salary"), errors="coerce").to_numpy()
    ok = np.isfinite(proj) & np.isfinite(sal) & (sal > 0)
    return np.flatnonzero(ok)


def _weights(proj: np.ndarray, skill: float) -> np.ndarray:
    w = np.clip(proj, 0.01, None) ** skill
    return w / w.sum()


def sample_lineup_indices(pool: pd.DataFrame, n: int = 1000,
                          field_skill: float = 3.0, seed: int = 0
                          ) -> np.ndarray:
    """(n, 9) positions into `pool`, one legal Classic roster per row."""
    keep = _usable(pool)
    if keep.size == 0 or n <= 0:
        return EMPTY.copy()
    sub = pool.iloc[keep]
    pos = sub["position"].astype(str).to_numpy()
    sal = sub["salary"].to_numpy(dtype=float)
    proj = sub["projection"].to_numpy(dtype=float)

    groups = {p: np.flatnonzero(pos == p) for p in ("QB", "RB", "WR", "TE", "DST")}
    groups["FLEX"] = np.flatnonzero(np.isin(pos, FLEX))
    need = {"QB": 1, "RB": 3, "WR": 3, "TE": 1, "DST": 1}
    if any(groups[p].size < k for p, k in need.items()) \
            or groups["RB"].size + groups["WR"].size + groups["TE"].size < 7:
        return EMPTY.copy()
    w = {p: _weights(proj[g], field_skill) for p, g in groups.items()}
    cheap = {p: g[np.argsort(sal[g], kind="stable")] for p, g in groups.items()}

    rng = np.random.default_rng(seed)
    out = np.empty((n, len(CLASSIC_SLOTS)), dtype=int)
    for k in range(n):
        picked: list[int] = []
        for slot in CLASSIC_SLOTS:
            g, p = groups[slot], w[slot]
            mask = ~np.isin(g, picked)
            pp = p * mask
            pp = pp / pp.sum()
            picked.append(int(rng.choice(g, p=pp)))
        # Repair: swap the most expensive slot for the cheapest legal option
        # until the roster fits.
        total = sal[picked].sum()
        guard = 0
        while total > SALARY_CAP and guard < 40:
            order = np.argsort(-sal[picked], kind="stable")
            swapped = False
            for j in order:
                slot = CLASSIC_SLOTS[j]
                for c in cheap[slot]:
                    if c in picked or sal[c] >= sal[picked[j]]:
                        continue
                    picked[j] = int(c)
                    swapped = True
                    break
                if swapped:
                    break
            if not swapped:
                break
            total = sal[picked].sum()
            guard += 1
        out[k] = picked
    return keep[out]


def sample_lineups(pool: pd.DataFrame, n: int = 1000, field_skill: float = 3.0,
                   seed: int = 0) -> np.ndarray:
    """Actual totals (dk_scored) of n sampled entries."""
    idx = sample_lineup_indices(pool, n=n, field_skill=field_skill, seed=seed)
    if idx.size == 0:
        return np.array([], dtype=float)
    pts = pd.to_numeric(pool["dk_scored"], errors="coerce").fillna(0.0)
    return pts.to_numpy(dtype=float)[idx].sum(axis=1)


def cash_line(pool: pd.DataFrame, n: int = 1000, field_skill: float = 3.0,
              seed: int = 0) -> float:
    """The score at the last paid position of a double-up: the 56.5th
    percentile of the field, strictly harder than its median."""
    scores = sample_lineups(pool, n=n, field_skill=field_skill, seed=seed)
    if scores.size == 0:
        return float("nan")
    return float(np.percentile(scores, 100 * (1 - PAID_FRACTION)))


def cash_line_sensitivity(pool: pd.DataFrame, skills=(1.0, 2.0, 3.0, 4.0),
                          n: int = 1000, seed: int = 0) -> pd.DataFrame:
    rows = []
    for s in skills:
        sc = sample_lineups(pool, n=n, field_skill=s, seed=seed)
        if sc.size == 0:
            continue
        rows.append({"field_skill": s, "p25": float(np.percentile(sc, 25)),
                     "median": float(np.median(sc)),
                     "p75": float(np.percentile(sc, 75)),
                     "cash_line": float(np.percentile(sc,
                                                      100 * (1 - PAID_FRACTION)))})
    return pd.DataFrame(rows)


def optimizer_entries(pool: pd.DataFrame, n: int = 25,
                      disagreement: float = 0.15,
                      opponent_col: str = "projection", seed: int = 0
                      ) -> np.ndarray:
    """Actual totals of n opponents who optimize. Each solves on
    pool[opponent_col] times (1 + disagreement * noise)."""
    if pool is None or pool.empty or opponent_col not in pool.columns:
        return np.array([], dtype=float)
    rng = np.random.default_rng(seed)
    base = pd.to_numeric(pool[opponent_col], errors="coerce").to_numpy(float)
    scored = pd.to_numeric(pool.get("dk_scored"), errors="coerce")
    out = []
    for _ in range(n):
        noise = rng.normal(0.0, 1.0, size=base.size)
        mine = pool.assign(_opp=np.clip(base * (1.0 + disagreement * noise),
                                        0.0, None))
        lus = optimizer.optimize(mine, objective="_opp", n_lineups=1)
        if not lus:
            continue
        lu = lus[0]
        out.append(float(pd.to_numeric(lu["dk_scored"], errors="coerce")
                         .fillna(0).sum()) if scored is not None else np.nan)
    return np.array(out, dtype=float)


def sample_showdown_indices(pool: pd.DataFrame, n: int = 1000,
                            field_skill: float = 3.0, seed: int = 0
                            ) -> tuple[np.ndarray, np.ndarray]:
    """((n, 6) positions, (n, 6) multipliers). Column 0 is the Captain."""
    empty = (np.zeros((0, SHOWDOWN_SIZE), dtype=int),
             np.zeros((0, SHOWDOWN_SIZE)))
    keep = _usable(pool)
    if keep.size < SHOWDOWN_SIZE or n <= 0:
        return empty
    sub = pool.iloc[keep]
    team = sub["team"].astype(str).to_numpy()
    if len(set(team)) < 2:
        return empty
    sal = sub["salary"].to_numpy(dtype=float)
    if "captain_salary" in sub.columns:
        csal = pd.to_numeric(sub["captain_salary"], errors="coerce").fillna(
            sub["salary"] * 1.5).to_numpy(float)
    else:
        csal = sal * 1.5
    proj = sub["projection"].to_numpy(dtype=float)
    w = _weights(proj, field_skill)
    cheap = np.argsort(sal, kind="stable")
    rng = np.random.default_rng(seed)
    out = np.empty((n, SHOWDOWN_SIZE), dtype=int)
    m = len(sub)
    for k in range(n):
        picked = list(rng.choice(m, size=SHOWDOWN_SIZE, replace=False, p=w))
        # Captain cost: spread the choice rather than always taking the top.
        if len(set(team[picked])) < 2:
            other = [c for c in cheap if team[c] != team[picked[0]]
                     and c not in picked]
            if other:
                picked[-1] = int(other[0])

        def spend(p):
            return csal[p[0]] + sal[p[1:]].sum()
        guard = 0
        while spend(picked) > SALARY_CAP and guard < 40:
            order = np.argsort(-sal[picked[1:]], kind="stable") + 1
            swapped = False
            for j in list(order) + [0]:
                for c in cheap:
                    if c in picked or sal[c] >= sal[picked[j]]:
                        continue
                    trial = picked.copy()
                    trial[j] = int(c)
                    if len(set(team[trial])) < 2:
                        continue
                    picked = trial
                    swapped = True
                    break
                if swapped:
                    break
            if not swapped:
                break
            guard += 1
        out[k] = picked
    weights = np.ones((n, SHOWDOWN_SIZE))
    weights[:, 0] = 1.5
    return keep[out], weights

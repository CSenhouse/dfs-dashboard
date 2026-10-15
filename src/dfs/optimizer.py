"""Lineup optimization: a mixed-integer program per lineup.

Classic: QB, 2-3 RB, 3-4 WR, 1-2 TE (seven skill players), DST, $50,000,
players from at least two games.

Showdown is a different game, not a Classic variant (rule 10): 1 Captain at
1.5x points and 1.5x salary plus 5 FLEX, any positions, both teams.

The optimizer selects the highest projections on the slate, so error in the
top decile is concentrated rather than averaged -- the winner's curse
(docs/MODELING.md 5). Most of the edge measured against a sampling field was
the solver, not the projections (section 8); a better solver is worth very
little from here.
"""
from __future__ import annotations

import functools
import warnings

import numpy as np
import pandas as pd
import pulp

from dfs.config import CAPTAIN_MULTIPLIER, SALARY_CAP, SHOWDOWN_SIZE


def _quiet(fn):
    """PuLP 3.x warns on every variable about its 4.0 API -- thousands per
    lineup. Silenced at the call so it holds under pytest too."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return fn(*args, **kwargs)
    return wrapper

CLASSIC_SIZE = 9
LIMITS = {"QB": (1, 1), "RB": (2, 3), "WR": (3, 4), "TE": (1, 2), "DST": (1, 1)}
SKILL = ("RB", "WR", "TE")
STACK_WITH = ("WR", "TE")
BRING_BACK = ("RB", "WR", "TE")


def _solve(prob: pulp.LpProblem) -> bool:
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    return pulp.LpStatus[prob.status] == "Optimal"


def _game_keys(pool: pd.DataFrame) -> pd.Series:
    """A game identifier: the explicit one if present, else the team pair,
    else the team itself (which still forces two different teams)."""
    for col in ("game_key", "competition_id"):
        if col in pool.columns and pool[col].notna().all():
            return pool[col].astype(str)
    if "opponent" in pool.columns and pool["opponent"].notna().all():
        t, o = pool["team"].astype(str), pool["opponent"].astype(str)
        return pd.Series(np.where(t < o, t + "@" + o, o + "@" + t),
                         index=pool.index)
    return pool["team"].astype(str)


def _clean(pool: pd.DataFrame, objective: str) -> pd.DataFrame:
    if pool is None or pool.empty or objective not in pool.columns:
        return pd.DataFrame()
    p = pool.copy()
    p = p[pd.to_numeric(p[objective], errors="coerce").notna()]
    p = p[pd.to_numeric(p["salary"], errors="coerce").notna()]
    if "player_dk_id" in p.columns:
        p = p.drop_duplicates("player_dk_id")
    return p.reset_index(drop=True)


def _exposure_cap(n_lineups: int, max_exposure: float) -> int | None:
    if max_exposure >= 1.0 or n_lineups <= 1:
        return None
    return max(1, int(max_exposure * n_lineups))


@_quiet
def optimize(pool: pd.DataFrame, objective: str = "projection",
             n_lineups: int = 1, salary_cap: int = SALARY_CAP,
             stack_qb_with: int = 0, bring_back: int = 0,
             min_difference: int = 1, max_exposure: float = 1.0,
             locks: list[int] | None = None, bans: list[int] | None = None,
             min_salary: int = 0) -> list[pd.DataFrame]:
    """Up to `n_lineups` legal Classic lineups, best first.

    stack_qb_with  WR/TE from the QB's own team
    bring_back     RB/WR/TE from the team the QB is facing (needs `opponent`;
                   silently dropped when the column is absent)
    """
    p = _clean(pool, objective)
    if p.empty:
        return []
    p = p[p["position"].isin(LIMITS)].reset_index(drop=True)
    if p.empty:
        return []
    pos = p["position"].to_numpy()
    sal = p["salary"].to_numpy(dtype=float)
    val = p[objective].to_numpy(dtype=float)
    team = p["team"].astype(str).to_numpy()
    games = _game_keys(p).to_numpy()
    ids = p["player_dk_id"].to_numpy() if "player_dk_id" in p.columns \
        else np.arange(len(p))
    opp = (p["opponent"].astype(str).to_numpy()
           if "opponent" in p.columns and p["opponent"].notna().any() else None)
    locks, bans = set(locks or []), set(bans or [])
    cap = _exposure_cap(n_lineups, max_exposure)

    n = len(p)
    idx = range(n)
    out: list[pd.DataFrame] = []
    chosen: list[list[int]] = []
    used = np.zeros(n, dtype=int)

    for k in range(n_lineups):
        prob = pulp.LpProblem(f"classic_{k}", pulp.LpMaximize)
        x = [pulp.LpVariable(f"x{i}", cat="Binary") for i in idx]
        prob += pulp.lpSum(val[i] * x[i] for i in idx)
        prob += pulp.lpSum(x) == CLASSIC_SIZE
        prob += pulp.lpSum(sal[i] * x[i] for i in idx) <= salary_cap
        if min_salary:
            prob += pulp.lpSum(sal[i] * x[i] for i in idx) >= min_salary
        for ps, (lo, hi) in LIMITS.items():
            members = [x[i] for i in idx if pos[i] == ps]
            prob += pulp.lpSum(members) >= lo
            prob += pulp.lpSum(members) <= hi
        prob += pulp.lpSum(x[i] for i in idx if pos[i] in SKILL) == 7

        game_ids = sorted(set(games))
        y = {g: pulp.LpVariable(f"g{j}", cat="Binary")
             for j, g in enumerate(game_ids)}
        for g, yg in y.items():
            prob += yg <= pulp.lpSum(x[i] for i in idx if games[i] == g)
        prob += pulp.lpSum(y.values()) >= 2

        for i in idx:
            if ids[i] in bans:
                prob += x[i] == 0
            if ids[i] in locks:
                prob += x[i] == 1
            if cap is not None and used[i] >= cap and ids[i] not in locks:
                prob += x[i] == 0

        if stack_qb_with or bring_back:
            for q in idx:
                if pos[q] != "QB":
                    continue
                if stack_qb_with:
                    mates = [x[i] for i in idx
                             if team[i] == team[q] and pos[i] in STACK_WITH]
                    prob += pulp.lpSum(mates) >= stack_qb_with * x[q]
                if bring_back and opp is not None:
                    rivals = [x[i] for i in idx
                              if team[i] == opp[q] and pos[i] in BRING_BACK]
                    prob += pulp.lpSum(rivals) >= bring_back * x[q]

        for prev in chosen:
            prob += pulp.lpSum(x[i] for i in prev) <= CLASSIC_SIZE - min_difference

        if not _solve(prob):
            break
        pick = [i for i in idx if x[i].value() is not None and x[i].value() > 0.5]
        if len(pick) != CLASSIC_SIZE:
            break
        chosen.append(pick)
        used[pick] += 1
        lu = p.iloc[pick].copy()
        lu["lineup"] = k
        out.append(lu.reset_index(drop=True))
    return out


@_quiet
def optimize_showdown(pool: pd.DataFrame, objective: str = "projection",
                      n_lineups: int = 1, salary_cap: int = SALARY_CAP,
                      min_difference: int = 1, max_exposure: float = 1.0,
                      locks: list[int] | None = None,
                      bans: list[int] | None = None, min_salary: int = 0,
                      captain: int | None = None, min_qb: int = 0,
                      qb_captain: bool = False, qb_stack: int = 0
                      ) -> list[pd.DataFrame]:
    """Up to `n_lineups` legal Showdown lineups: 1 CPT + 5 FLEX.

    `captain` pins a player to the Captain slot; an unknown id returns
    nothing rather than quietly building someone else's lineup.

    Construction rules, all off by default and tested on 1,416 finished
    slates (docs/MODELING.md 34):

        min_qb      at least this many QBs (no-QB lineups cashed 35.7%
                    against 44.9%)
        qb_captain  the Captain is a QB (cash 45.9% against 41.6%); a
                    pinned `captain` overrides it
        qb_stack    every rostered QB has this many of his own WR/TE
                    (a lonely QB reached the top 1% at 0.86% against 1.04%)

    Each is skipped when the pool has no QB to satisfy it.
    """
    p = _clean(pool, objective)
    if p.empty:
        return []
    ids = p["player_dk_id"].to_numpy()
    if captain is not None and captain not in set(ids):
        return []
    sal = p["salary"].to_numpy(dtype=float)
    if "captain_salary" in p.columns:
        csal = pd.to_numeric(p["captain_salary"], errors="coerce")
        csal = csal.fillna(p["salary"] * CAPTAIN_MULTIPLIER).to_numpy(float)
    else:
        csal = sal * CAPTAIN_MULTIPLIER
    val = p[objective].to_numpy(dtype=float)
    team = p["team"].astype(str).to_numpy()
    pos = p["position"].astype(str).to_numpy()
    teams = sorted(set(team))
    locks, bans = set(locks or []), set(bans or [])
    cap = _exposure_cap(n_lineups, max_exposure)
    qbs = [i for i in range(len(p)) if pos[i] == "QB"]

    n = len(p)
    idx = range(n)
    out: list[pd.DataFrame] = []
    chosen: list[list[int]] = []
    used = np.zeros(n, dtype=int)

    for k in range(n_lineups):
        prob = pulp.LpProblem(f"showdown_{k}", pulp.LpMaximize)
        c = [pulp.LpVariable(f"c{i}", cat="Binary") for i in idx]
        f = [pulp.LpVariable(f"f{i}", cat="Binary") for i in idx]
        prob += pulp.lpSum(CAPTAIN_MULTIPLIER * val[i] * c[i] + val[i] * f[i]
                           for i in idx)
        prob += pulp.lpSum(c) == 1
        prob += pulp.lpSum(f) == SHOWDOWN_SIZE - 1
        for i in idx:
            prob += c[i] + f[i] <= 1
        spend = pulp.lpSum(csal[i] * c[i] + sal[i] * f[i] for i in idx)
        prob += spend <= salary_cap
        if min_salary:
            prob += spend >= min_salary
        if len(teams) >= 2:
            for t in teams:
                prob += pulp.lpSum(c[i] + f[i] for i in idx if team[i] == t) >= 1
        else:
            return []
        if qbs:
            if min_qb:
                prob += pulp.lpSum(c[i] + f[i] for i in qbs) >= min_qb
            if qb_captain and captain is None:
                prob += pulp.lpSum(c[i] for i in qbs) == 1
            if qb_stack:
                for q in qbs:
                    mates = [c[i] + f[i] for i in idx
                             if team[i] == team[q] and pos[i] in STACK_WITH]
                    prob += pulp.lpSum(mates) >= qb_stack * (c[q] + f[q])
        for i in idx:
            if ids[i] in bans:
                prob += c[i] + f[i] == 0
            if ids[i] in locks:
                prob += c[i] + f[i] == 1
            if cap is not None and used[i] >= cap and ids[i] not in locks \
                    and ids[i] != captain:
                prob += c[i] + f[i] == 0
            if captain is not None and ids[i] == captain:
                prob += c[i] == 1
        for prev in chosen:
            prob += pulp.lpSum(c[i] + f[i] for i in prev) \
                <= SHOWDOWN_SIZE - min_difference

        if not _solve(prob):
            break
        cpt = [i for i in idx if c[i].value() is not None and c[i].value() > 0.5]
        flex = [i for i in idx if f[i].value() is not None and f[i].value() > 0.5]
        if len(cpt) != 1 or len(flex) != SHOWDOWN_SIZE - 1:
            break
        pick = cpt + flex
        chosen.append(pick)
        used[pick] += 1
        lu = p.iloc[pick].copy()
        lu["slot"] = ["CPT"] + ["FLEX"] * len(flex)
        lu["lineup_salary"] = np.r_[csal[cpt], sal[flex]]
        lu["lineup_points"] = np.r_[CAPTAIN_MULTIPLIER * val[cpt], val[flex]]
        lu["lineup"] = k
        out.append(lu.reset_index(drop=True))
    return out


def summarize(lu: pd.DataFrame, col: str = "projection") -> str:
    """A printable lineup."""
    cols = [c for c in ("slot", "position", "player_name", "team", "salary",
                        col, "dk_scored") if c in lu.columns]
    body = lu[cols].to_string(index=False)
    spend = lu["lineup_salary"].sum() if "lineup_salary" in lu.columns \
        else lu["salary"].sum()
    pts = lu["lineup_points"].sum() if "lineup_points" in lu.columns \
        else lu[col].sum()
    return f"{body}\n  salary ${spend:,.0f}   {col} {pts:.2f}"

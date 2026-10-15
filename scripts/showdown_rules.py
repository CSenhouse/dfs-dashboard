"""Do common Showdown construction rules hold up on real outcomes?

For every finished Showdown slate, a field of realistic lineups is sampled
the way entrants draft (field.sample_showdown_indices: weighted toward
well-projected players, legal under the cap, both teams), each lineup is
scored on what actually happened (captain 1.5x), and lineups are tagged by
structure. Each structure is then compared with the lineups WITHOUT it on
the same slates:

    cash rate   beat 56.5% of the slate's field (a double-up line)
    top 1%      beat 99% of it (a tournament proxy)

Paired by slate, so a high-scoring game cannot flatter a structure that was
common in it. Projections are walk-forward (each week from earlier weeks
only), so the field drafts on information it could have had.

What this can and cannot say. The field is modelled, not counted -- the
real exports were lost -- and a structure is not randomly assigned: lineups
with two quarterbacks also differ in salary and in who else fits. These are
associations across ~1,400 slates, not controlled experiments.

    PYTHONPATH=src .venv/Scripts/python.exe scripts/showdown_rules.py
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

from dfs import backtest, field, model
from dfs.config import CURATED

N_FIELD = 3000
FIELD_SKILL = 3.0
LINE = 0.565          # a double-up pays the top 43.5%
TOP = 0.99
SKILL = ("WR", "TE")


def tags(pool: pd.DataFrame, idx: np.ndarray) -> dict[str, np.ndarray]:
    """Structure flags for every sampled lineup (rows of positions)."""
    pos = pool["position"].to_numpy()[idx]
    team = pool["team"].to_numpy()[idx]
    cpt_pos = pos[:, 0]
    qbs = pos == "QB"
    n_qb = qbs.sum(axis=1)
    n_te = (pos == "TE").sum(axis=1)

    qb_team = np.where(qbs, team, "")
    dst_team = np.where(pos == "DST", team, "")
    # QB with the DST of the OTHER team -- the defense playing against him.
    vs_dst = np.array([
        any(q and d and q != d for q in qt for d in dt)
        for qt, dt in zip(qb_team, dst_team)])
    # QB with no WR/TE of his own: a "lonely" quarterback.
    lonely = np.array([
        n == 1 and not any((t == qt[qt != ""][0]) and p in SKILL
                           for t, p in zip(tr, pr))
        for n, qt, tr, pr in zip(n_qb, qb_team, team, pos)])
    stacked = (n_qb == 1) & ~lonely & (n_qb > 0)
    return {
        "two QBs": n_qb == 2,
        "QB + opposing DST": vs_dst,
        "two TEs": n_te >= 2,
        "lonely QB (no own WR/TE)": lonely,
        "QB stacked with own WR/TE": stacked,
        "no QB at all": n_qb == 0,
        "QB captain": cpt_pos == "QB",
        "TE captain": cpt_pos == "TE",
        "kicker in lineup": (pos == "K").any(axis=1),
        "DST in lineup": (pos == "DST").any(axis=1),
    }


def main() -> int:
    f = pd.read_parquet(CURATED / "showdown_features.parquet")
    print(f"{len(f):,} rows, {f['draft_group_id'].nunique()} slates; "
          "walk-forward projections ...", flush=True)
    pred = backtest.walk_forward(f, model.two_stage, min_train_weeks=8)
    pred = pred.dropna(subset=["projection"])
    pred = pred[pred["salary"] > 0]

    per: dict[str, list[dict]] = {}
    t0 = time.time()
    slates = list(pred.groupby("draft_group_id"))
    for k, (dgid, pool) in enumerate(slates, 1):
        pool = pool.reset_index(drop=True)
        if len(pool) < 15 or pool["team"].nunique() != 2:
            continue
        idx, w = field.sample_showdown_indices(
            pool, n=N_FIELD, field_skill=FIELD_SKILL, seed=int(dgid) % 100000)
        if idx.size == 0:
            continue
        pts = pool["dk_scored"].to_numpy(float)[idx]
        score = (pts * w).sum(axis=1)
        rank = stats.rankdata(score) / len(score)      # 1.0 = best
        cash, top = rank > LINE, rank > TOP
        for name, flag in tags(pool, idx).items():
            if flag.sum() < 30 or (~flag).sum() < 30:
                continue
            per.setdefault(name, []).append({
                "share": float(flag.mean()),
                "cash_with": float(cash[flag].mean()),
                "cash_without": float(cash[~flag].mean()),
                "top_with": float(top[flag].mean()),
                "top_without": float(top[~flag].mean()),
            })
        if k % 200 == 0:
            print(f"  {k}/{len(slates)} slates  {time.time() - t0:.0f}s",
                  flush=True)

    print(f"\n{len(slates)} slates, {N_FIELD:,} sampled lineups each\n")
    print(f"{'structure':<28} {'slates':>6} {'share':>6}   "
          f"{'cash with/without':>18}  {'p':>7}   "
          f"{'top-1% with/without':>20}  {'p':>7}")
    for name, rows in per.items():
        d = pd.DataFrame(rows)
        pc = stats.wilcoxon(d["cash_with"], d["cash_without"]).pvalue
        pt = stats.wilcoxon(d["top_with"], d["top_without"]).pvalue \
            if (d["top_with"] != d["top_without"]).any() else float("nan")
        print(f"{name:<28} {len(d):>6} {d['share'].mean():>6.1%}   "
              f"{d['cash_with'].mean():>8.1%} / {d['cash_without'].mean():<7.1%}"
              f"  {pc:>7.1g}   "
              f"{d['top_with'].mean():>9.2%} / {d['top_without'].mean():<8.2%}"
              f"  {pt:>7.1g}")
    print("\ncash = beat 56.5% of the slate's sampled field; top 1% = beat 99%."
          "\nBase rates are 43.5% and 1.0% by construction.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

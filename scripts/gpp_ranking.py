"""Should a tournament lineup be chosen by its projection sum or its tail?

The question came from a live slate. The optimizer's top GPP lineup stacked
Kirk Cousins on a Las Vegas team implied for 20.25 points -- the lowest total
among the plausible quarterbacks -- because at $4,800 he was the best ceiling
per dollar on the board. That lineup had the HIGHEST sum of ceilings, 202.6,
and roughly half the simulated chance of reaching 200 of the builds around
Purdy or Goff.

Nothing is wrong with the projection. Cousins is correctly ranked below the
elite quarterbacks. The defect is the objective: a sum of nine 85th
percentiles is a CENTRAL statistic, and a tournament pays on a TAIL one.
Buying the cheapest acceptable ceiling and spending the savings elsewhere
maximises the first while quietly damaging the second.

So the candidates here are deliberately not near-copies. The earlier cash
experiment ranked twelve lineups that all came from one objective, they
differed by two players, and the ranking had nothing to work with. This builds
the best stacked lineup for EACH viable quarterback, which spans game
environments and reproduces exactly the Cousins-versus-Purdy choice.

Two rankings, same candidates, same weeks:

    sum of ceilings   what the dashboard does today
    simulated tail    P(the lineup reaches TARGET), from the copula simulator

Scored on what actually happened: points, and percentile within the field that
really played. Paired week by week -- the weekly lineup metric has a ~25 point
standard deviation and unpaired runs of it have given a wrong answer three
times in this project.

    PYTHONPATH=src .venv/Scripts/python.exe scripts/gpp_ranking.py
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

from dfs import field, optimizer, simulate
from dfs.config import CURATED

N_SIMS = 6000
N_QBS = 10          # candidates: the best stacked lineup for each of these
TARGET = 200.0      # a large-field winning score
FIELD_SKILL = 3.0


def candidates(pool: pd.DataFrame) -> list[pd.DataFrame]:
    """One stacked lineup per quarterback, so the set spans game environments.

    Locking the quarterback and re-solving is what makes these genuinely
    different lineups rather than permutations of one. Without it the ranking
    is choosing between near-identical rosters and cannot express a view.
    """
    qbs = (pool[pool["position"] == "QB"]
           .nlargest(N_QBS, "proj_ceiling")["player_dk_id"].tolist())
    out = []
    for pid in qbs:
        built = optimizer.optimize(pool, objective="proj_ceiling", n_lineups=1,
                                   stack_qb_with=1, locks=[int(pid)])
        if built:
            out.append(built[0])
    return out


def main(argv: list[str]) -> int:
    limit = int(argv[argv.index("--weeks") + 1]) if "--weeks" in argv else 0

    path = CURATED / "sim_predictions.parquet"
    if not path.exists():
        print(f"{path} missing -- run scripts/build_simulation_inputs.py",
              file=sys.stderr)
        return 1
    pred = pd.read_parquet(path)
    pred["proj_ceiling"] = pred["p_play"] * pred["q85"].clip(lower=0)
    pred["proj_mean"] = pred["p_play"] * pred["points_mean"].clip(lower=0)

    weeks = (pred[["season", "week"]].drop_duplicates()
             .sort_values(["season", "week"]).to_records(index=False))
    if limit:
        weeks = weeks[-limit:]
    print(f"{len(weeks)} walk-forward weeks, {N_QBS} candidates each\n")

    rows = []
    t0 = time.time()
    for season, week in weeks:
        pool = pred[(pred["season"] == season) & (pred["week"] == week)].copy()
        if len(pool) < 100:
            continue
        cands = candidates(pool)
        if len(cands) < 3:
            continue

        # Rank the pool on expected points for depth roles, not on the ceiling
        # view, so the copula sees the same depth chart it was measured with.
        sim = simulate.sample(pool.assign(projection=pool["proj_mean"]),
                              n_sims=N_SIMS, seed=1)
        realised = field.sample_lineups(pool, n=2000, field_skill=FIELD_SKILL,
                                        seed=5)
        if realised.size == 0:
            continue

        sums, tails, actuals = [], [], []
        for lu in cands:
            ids = [int(x) for x in lu["player_dk_id"]]
            totals = sim.lineup_totals(ids)
            sums.append(float(lu["proj_ceiling"].sum()))
            tails.append(float((totals >= TARGET).mean()))
            actuals.append(float(lu["dk_scored"].sum()))
        sums, tails, actuals = map(np.array, (sums, tails, actuals))

        by_sum, by_tail = int(np.argmax(sums)), int(np.argmax(tails))
        rows.append({
            "season": season, "week": week,
            "agree": by_sum == by_tail,
            "sum_actual": actuals[by_sum], "tail_actual": actuals[by_tail],
            "sum_pct": float((realised < actuals[by_sum]).mean()),
            "tail_pct": float((realised < actuals[by_tail]).mean()),
            "sum_qb_total": float(
                cands[by_sum].loc[cands[by_sum]["position"] == "QB",
                                  "salary"].iloc[0]),
            "tail_qb_total": float(
                cands[by_tail].loc[cands[by_tail]["position"] == "QB",
                                   "salary"].iloc[0]),
        })
        print(f"  {season} wk{week:>4}  sum={actuals[by_sum]:6.1f}  "
              f"tail={actuals[by_tail]:6.1f}"
              f"{'  (same lineup)' if by_sum == by_tail else ''}", flush=True)

    d = pd.DataFrame(rows)
    if d.empty:
        print("nothing built", file=sys.stderr)
        return 1
    d.to_parquet(CURATED / "gpp_ranking.parquet", index=False)

    diff = d["tail_actual"] - d["sum_actual"]
    changed = d[~d["agree"]]
    print(f"\n=== {len(d)} paired weeks, "
          f"{len(changed)} where the ranking disagreed ===")
    print(f"  by sum of ceilings: {d['sum_actual'].mean():6.2f} pts   "
          f"percentile {d['sum_pct'].mean():.1%}")
    print(f"  by simulated tail:  {d['tail_actual'].mean():6.2f} pts   "
          f"percentile {d['tail_pct'].mean():.1%}")
    print(f"  difference:         {diff.mean():+6.2f} pts   "
          f"{(d['tail_pct'] - d['sum_pct']).mean():+.1%}")

    if len(changed) > 1:
        cd = changed["tail_actual"] - changed["sum_actual"]
        p = stats.ttest_rel(changed["tail_actual"],
                            changed["sum_actual"]).pvalue
        print(f"\n  on the {len(changed)} disagreements: {cd.mean():+.2f} pts, "
              f"better in {int((cd > 0).sum())}, worse in {int((cd < 0).sum())}")
        print(f"  paired t p={p:.3f}")
        sign = stats.binomtest(int((cd > 0).sum()),
                               int((cd != 0).sum()), 0.5).pvalue
        print(f"  sign test p={sign:.3f}")

    for label, col in (("sum", "sum_actual"), ("tail", "tail_actual")):
        hits = [(d[col] >= t).mean() for t in (160, 180, 200)]
        print(f"  reached 160/180/200 by {label:<4}: "
              + "  ".join(f"{h:.1%}" for h in hits))
    print(f"\n  mean QB salary chosen: by sum ${d['sum_qb_total'].mean():,.0f}, "
          f"by tail ${d['tail_qb_total'].mean():,.0f}")
    print(f"\n{time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

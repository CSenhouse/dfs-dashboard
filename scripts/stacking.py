"""Does stacking actually win tournaments, or is it just folklore?

Everyone stacks. That is not evidence. Three feature efforts in this project
landed near zero after looking obviously right, so the rule here is that a
structure earns its place by beating its own absence on identical weeks.

The case FOR stacking is not that it raises expected points -- it lowers them,
because forcing a quarterback's receiver into the lineup means passing over a
better-projected player somewhere else. It is that a tournament does not pay
expected points. It pays the right tail, and the tail is where correlation
compounds:

    a lineup's total has spread S
    a winning score sits perhaps 3 spreads above its middle
    correlation lifts S by a few percent
    P(3 spreads out) moves far more than a few percent, because the normal
    tail falls away so steeply there

So the metric is NOT the mean. It is how often a lineup reaches a score that
would actually win something, and that is what this measures, on real results,
paired week by week.

Thresholds are absolute rather than percentiles of a simulated field. The
field sampler is known to be too soft -- it medians 111 against a first
observed cash line of 147 -- so a percentile against it would flatter every
structure equally and mean nothing. A score is a score.

    PYTHONPATH=src .venv/Scripts/python.exe scripts/stacking.py
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

from dfs import optimizer, simulate
from dfs.config import CURATED

# Structures to compare. All maximise the same ceiling projection, so the only
# thing that differs is the shape of the roster.
STRUCTURES = {
    "no stack": dict(stack_qb_with=0, bring_back=0),
    "QB + 1": dict(stack_qb_with=1, bring_back=0),
    "QB + 2": dict(stack_qb_with=2, bring_back=0),
    "QB + 1 + bring-back": dict(stack_qb_with=1, bring_back=1),
    "QB + 2 + bring-back": dict(stack_qb_with=2, bring_back=1),
}

# A double-up cashed at 147 last week, so these are the scores a tournament
# lineup has to reach to be doing anything. 180 is a good week, 200 wins a
# small field, 220 is a large-field winner.
THRESHOLDS = (160, 180, 200, 220)
N_SIMS = 3000


def main(argv: list[str]) -> int:
    path = CURATED / "sim_predictions.parquet"
    if not path.exists():
        print(f"{path} missing -- run scripts/build_simulation_inputs.py",
              file=sys.stderr)
        return 1

    pred = pd.read_parquet(path)
    pred["proj_ceiling"] = pred["p_play"] * pred["q85"].clip(lower=0)
    weeks = (pred[["season", "week"]].drop_duplicates()
             .sort_values(["season", "week"]).to_records(index=False))
    print(f"{len(weeks)} walk-forward weeks\n")

    rows, tails = [], []
    t0 = time.time()
    for season, week in weeks:
        pool = pred[(pred["season"] == season) & (pred["week"] == week)].copy()
        if len(pool) < 100:
            continue
        sim = simulate.sample(pool, n_sims=N_SIMS, seed=1)
        flat = simulate.sample(pool, n_sims=N_SIMS, seed=1, correlated=False)

        for name, kw in STRUCTURES.items():
            built = optimizer.optimize(pool, objective="proj_ceiling",
                                       n_lineups=1, **kw)
            if not built:
                continue
            lu = built[0]
            ids = [int(p) for p in lu["player_dk_id"]]
            rows.append({
                "season": season, "week": week, "structure": name,
                "actual": float(lu["dk_scored"].sum()),
                "projected": float(lu["proj_ceiling"].sum()),
                "sim_sd": float(sim.lineup_totals(ids).std()),
                "flat_sd": float(flat.lineup_totals(ids).std()),
            })
            # What the simulator says about the tail, with and without the
            # copula. A small change in spread is a large change out here.
            t = sim.lineup_totals(ids)
            f = flat.lineup_totals(ids)
            for thr in THRESHOLDS:
                tails.append({"structure": name, "threshold": thr,
                              "correlated": float((t >= thr).mean()),
                              "independent": float((f >= thr).mean())})
        print(f"  {season} wk{week:>4}", flush=True)

    d = pd.DataFrame(rows)
    if d.empty:
        print("nothing built", file=sys.stderr)
        return 1
    d.to_parquet(CURATED / "stacking.parquet", index=False)

    print(f"\n=== what each structure actually scored, {d['week'].size // len(STRUCTURES)} weeks ===")
    print("  structure              mean     sd    " +
          "  ".join(f">={t}" for t in THRESHOLDS))
    for name in STRUCTURES:
        g = d[d["structure"] == name]
        if g.empty:
            continue
        hits = "  ".join(f"{(g['actual'] >= t).mean():6.1%}"
                         for t in THRESHOLDS)
        print(f"  {name:<20} {g['actual'].mean():6.1f} {g['actual'].std():6.1f}"
              f"  {hits}")

    base = d[d["structure"] == "no stack"].set_index(["season", "week"])
    print("\n=== paired against no stack (same weeks) ===")
    for name in STRUCTURES:
        if name == "no stack":
            continue
        g = d[d["structure"] == name].set_index(["season", "week"])
        both = base.join(g, rsuffix="_s", how="inner").dropna(
            subset=["actual", "actual_s"])
        if len(both) < 5:
            continue
        diff = both["actual_s"] - both["actual"]
        p = stats.ttest_rel(both["actual_s"], both["actual"]).pvalue
        big = ((both["actual_s"] >= 200).mean()
               - (both["actual"] >= 200).mean())
        print(f"  {name:<20} mean {diff.mean():+6.2f} pts (p={p:.3f})   "
              f"reached 200: {big:+.1%}   "
              f"better in {int((diff > 0).sum())}/{len(both)}")

    t = pd.DataFrame(tails)
    print("\n=== what correlation does to the simulated tail ===")
    print("  (spread barely moves; the tail moves a lot, which is the point)")
    print("  structure            threshold   correlated  independent   lift")
    for name in STRUCTURES:
        g = d[d["structure"] == name]
        if g.empty:
            continue
        widen = (g["sim_sd"] / g["flat_sd"] - 1).mean()
        print(f"  {name:<20} spread {widen:+.1%}")
        for thr in THRESHOLDS:
            h = t[(t["structure"] == name) & (t["threshold"] == thr)]
            if h.empty or h["independent"].mean() <= 0:
                continue
            a, b = h["correlated"].mean(), h["independent"].mean()
            print(f"  {'':<20} >={thr:<8} {a:9.3%}   {b:9.3%}   "
                  f"{a / b - 1:+6.1%}")
    print(f"\n{time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

"""Is the simulated distribution real, or just confident?

A simulator always produces a probability. The question is whether that
probability means anything, and three separate things have to hold before it
does. Each is tested here against walk-forward predictions, never in-sample.

1. **Are the per-player distributions the right shape?**
   Map every observed score through its own predicted distribution. Under a
   correct model that lands uniform on [0, 1]. Too many results in the tails
   means the model is overconfident; too few means it hedges. RMSE cannot see
   either -- a model can have an excellent mean and a badly wrong spread.

2. **Is a LINEUP's distribution right?**
   This is where correlation shows up -- or fails to. Nine independent players
   have a narrower sum than nine correlated ones, so an independent simulator
   should put the actual result outside its own 80% interval too often. Run
   with and without the copula, on identical weeks and identical lineups.

   Two candidate styles, because they stress it differently. A cash lineup is
   built on the floor and barely stacks, so the copula hardly touches it. A
   GPP lineup deliberately pairs a QB with his own receiver, which is exactly
   where dependence should widen things. If it does not show up there, it is
   not doing anything anywhere.

3. **Does it pick better lineups?**
   The only test that pays. Take the SAME candidates each week, rank them the
   way the dashboard does today (largest sum of projections) and the way the
   simulator does (highest simulated probability), and see which finishes
   higher against the field that actually played.

   Scored on PERCENTILE within that field, not on a beat-the-line flag. The
   flag saturates: against a field that samples rather than optimizes, every
   candidate we build clears the median, so a binary test reads 100% for both
   methods and discriminates nothing. The percentile is continuous, never
   saturates, and is monotone in the same quantity.

   Paired week by week, because the weekly lineup metric has a ~25 point
   standard deviation and unpaired runs of it have given a wrong answer three
   times in this project.

    PYTHONPATH=src .venv/Scripts/python.exe scripts/validate_simulation.py
    PYTHONPATH=src .venv/Scripts/python.exe scripts/validate_simulation.py --weeks 20
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

from dfs import field, model, optimizer, simulate
from dfs.config import CURATED

N_SIMS = 3000
N_FIELD = 800
N_CANDIDATES = 12
FIELD_SKILL = 3.0

STYLES = {
    "cash": dict(objective="proj_floor", stack_qb_with=0),
    "gpp": dict(objective="proj_ceiling", stack_qb_with=1),
}

# Each variant drops one piece, so its contribution is measured rather than
# assumed. All three see identical weeks, identical lineups and the same seed.
VARIANTS = {
    "full": dict(correlated=True, calibrated=True),
    "no copula": dict(correlated=False, calibrated=True),
    "no spread fix": dict(correlated=True, calibrated=False),
}

# The spread correction was fitted on 2021-2023, so lineup-level results are
# reported on the seasons it has never seen. Measuring it on its own training
# years would report the fit, not the fix.
HELD_OUT = (2024, 2025)


def player_calibration(df: pd.DataFrame) -> pd.DataFrame:
    grid = df[model.quantile_columns(df)].to_numpy(dtype=float, copy=True)
    grid.sort(axis=1)
    u = simulate.pit(df["dk_scored"].to_numpy(float), grid,
                     df["p_play"].to_numpy(float), seed=3)
    out = df[["position"]].copy()
    out["u"] = u
    return out


def _uniformity(u: np.ndarray) -> tuple[float, float]:
    """KS distance from uniform, and the share landing in the middle 80%."""
    if u.size == 0:
        return float("nan"), float("nan")
    ks = float(stats.kstest(u, "uniform").statistic)
    inside = float(((u >= 0.10) & (u <= 0.90)).mean())
    return ks, inside


def lineup_pit(totals: np.ndarray, actual: float, rng) -> float:
    below = float((totals < actual).mean())
    tie = float((totals == actual).mean())
    return below + rng.random() * tie


def main(argv: list[str]) -> int:
    limit = 0
    if "--weeks" in argv:
        limit = int(argv[argv.index("--weeks") + 1])

    path = CURATED / "sim_predictions.parquet"
    if not path.exists():
        print(f"{path} missing -- run scripts/build_simulation_inputs.py",
              file=sys.stderr)
        return 1
    pred = pd.read_parquet(path)
    qcols = model.quantile_columns(pred)
    pred["proj_floor"] = pred["p_play"] * pred["q30"].clip(lower=0)
    pred["proj_ceiling"] = pred["p_play"] * pred["q85"].clip(lower=0)

    print(f"{len(pred):,} walk-forward player-weeks, {len(qcols)} quantiles\n")

    # --- 1. per-player shape -------------------------------------------------
    cal = player_calibration(pred)
    ks, inside = _uniformity(cal["u"].to_numpy())
    print("=== 1. player-level calibration (uniform is correct) ===")
    counts = np.histogram(cal["u"], bins=10, range=(0, 1))[0]
    print("  decile share: " + " ".join(f"{s:.3f}" for s in counts / counts.sum()))
    print(f"  KS={ks:.4f}   inside the middle 80%: {inside:.1%} (want 80.0%)")
    for pos, g in cal.groupby("position"):
        k, ins = _uniformity(g["u"].to_numpy())
        print(f"    {pos:<4} n={len(g):>6,}  KS={k:.4f}  inside={ins:.1%}")
    print()

    # --- 2 & 3. lineup level -------------------------------------------------
    lineup_rows = pred if "--all-seasons" in argv else pred[
        pred["season"].isin(HELD_OUT)]
    weeks = (lineup_rows[["season", "week"]].drop_duplicates()
             .sort_values(["season", "week"]).to_records(index=False))
    if limit:
        weeks = weeks[-limit:]
    print(f"lineup tests on {len(weeks)} weeks from "
          f"{sorted(set(int(w[0]) for w in weeks))}\n")

    rng = np.random.default_rng(11)
    cov, dec = [], []
    t0 = time.time()
    for season, week in weeks:
        pool = lineup_rows[(lineup_rows["season"] == season)
                           & (lineup_rows["week"] == week)].copy()
        if len(pool) < 100:
            continue

        # The field that actually played, scored on what actually happened.
        realised = field.sample_lineups(pool, n=2000, field_skill=FIELD_SKILL,
                                        seed=5)
        if realised.size == 0:
            continue

        sims = {name: simulate.sample(pool, n_sims=N_SIMS, seed=1, **kw)
                for name, kw in VARIANTS.items()}
        opp = simulate.field_entries(pool, sims["full"], n=N_FIELD,
                                     field_skill=FIELD_SKILL, seed=2)
        world_line = np.median(opp, axis=1)

        for style, kw in STYLES.items():
            cands = optimizer.optimize(pool, n_lineups=N_CANDIDATES,
                                       min_difference=2, **kw)
            if len(cands) < 2:
                continue
            actuals = np.array([float(lu["dk_scored"].sum()) for lu in cands])

            sim_metric = []
            for name, sim in sims.items():
                for i, lu in enumerate(cands):
                    ids = [int(x) for x in lu["player_dk_id"]]
                    totals = sim.lineup_totals(ids)
                    cov.append({
                        "season": season, "week": week, "style": style,
                        "sim": name, "cand": i,
                        "pit": lineup_pit(totals, actuals[i], rng),
                        "sd": float(totals.std()),
                    })
                    if name == "full":
                        if style == "cash":
                            m = float(np.mean(totals > world_line))
                        else:
                            pct = (totals[:, None] > opp).mean(axis=1)
                            m = float(np.mean(pct >= 0.99))
                        sim_metric.append(m)

            pick = int(np.argmax(sim_metric))
            dec.append({
                "season": season, "week": week, "style": style,
                "pick_index": pick,
                "status_quo": actuals[0], "sim_pick": actuals[pick],
                "sq_pct": float((realised < actuals[0]).mean()),
                "sim_pct": float((realised < actuals[pick]).mean()),
            })
        print(f"  {season} wk{week:>4}  field median={np.median(realised):6.1f}",
              flush=True)

    cov = pd.DataFrame(cov)
    dec = pd.DataFrame(dec)
    if cov.empty:
        print("no weeks evaluated", file=sys.stderr)
        return 1
    cov.to_parquet(CURATED / "simulation_coverage.parquet", index=False)
    dec.to_parquet(CURATED / "simulation_decisions.parquet", index=False)

    n_weeks = len(cov[["season", "week"]].drop_duplicates())
    print(f"\n=== 2. lineup-level calibration over {n_weeks} weeks ===")
    print("  correlation should widen a stacked GPP lineup; a cash lineup "
          "barely stacks, so it should barely move")
    for style in STYLES:
        for name in VARIANTS:
            g = cov[(cov["style"] == style) & (cov["sim"] == name)]
            if g.empty:
                continue
            _, ins = _uniformity(g["pit"].to_numpy())
            print(f"  {style:<5} {name:<12} n={len(g):>5}  "
                  f"inside the middle 80%: {ins:.1%} (want 80.0%)  "
                  f"mean lineup sd={g['sd'].mean():.2f}")
        a = cov[(cov["style"] == style) & (cov["sim"] == "full")]["sd"]
        for other, label in (("no copula", "correlation"),
                             ("no spread fix", "the spread fix")):
            b = cov[(cov["style"] == style) & (cov["sim"] == other)]["sd"]
            if len(a) == len(b) and len(a):
                widen = (a.to_numpy() / b.to_numpy() - 1).mean()
                print(f"  {style:<5} {label} widens the lineup by {widen:+.1%}")

    print("\n=== 3. does it pick better? ===")
    for style in STYLES:
        d = dec[dec["style"] == style]
        if d.empty:
            continue
        changed = d[d["pick_index"] != 0]
        print(f"\n  --- {style}, {len(d)} paired weeks "
              f"({len(changed)} where the simulator changed the pick) ---")
        print(f"    status quo: percentile {d['sq_pct'].mean():.1%}, "
              f"mean {d['status_quo'].mean():.1f} pts")
        print(f"    simulator:  percentile {d['sim_pct'].mean():.1%}, "
              f"mean {d['sim_pick'].mean():.1f} pts")
        if len(changed) > 1:
            cd = changed["sim_pct"] - changed["sq_pct"]
            print(f"    on the {len(changed)} changed weeks: {cd.mean():+.1%} "
                  f"percentile, better in {int((cd > 0).sum())}, worse in "
                  f"{int((cd < 0).sum())}")
            p = stats.ttest_rel(changed["sim_pct"], changed["sq_pct"]).pvalue
            print(f"    paired t p={p:.3f}")
        print(f"    points: {(d['sim_pick'] - d['status_quo']).mean():+.2f} "
              f"per week")
    print(f"\n{time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

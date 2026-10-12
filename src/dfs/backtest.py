"""Walk-forward evaluation: train strictly on earlier weeks, predict the next.

Rule 2: never compare two separate evaluation runs -- the weekly lineup
metric has a ~25 point sd and has flipped sign between runs. Use
scripts/compare.py or scripts/ablate.py, which pair on identical weeks.

Rule 3: select models on accuracy (spearman), not lineup scores.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy import stats

from dfs import optimizer


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Separate the two facts a missing score can mean (rule 5).

    dk_scored  what a lineup would have scored: a missing score is 0
    played     whether there was an observation at all -- the classifier's
               target, and the regressor trains only where it is true

    Dropping priced players who never played would be survivorship bias.
    """
    out = df.copy()
    if "dk_actual" not in out.columns:
        if "dk_scored" in out.columns and "played" in out.columns:
            return out
        out["dk_actual"] = np.nan
    actual = pd.to_numeric(out["dk_actual"], errors="coerce")
    out["dk_scored"] = actual.fillna(0.0)
    out["played"] = actual.notna()
    return out


def _week_key(df: pd.DataFrame) -> pd.Series:
    return df["season"].astype(int) * 100 + df["week"].astype(int)


def _weeks(df: pd.DataFrame) -> list[int]:
    return sorted(_week_key(df).unique())


def walk_forward(df: pd.DataFrame, fn, min_train_weeks: int = 8,
                 verbose: bool = False) -> pd.DataFrame:
    """Each week predicted by fn(past, cur) using only earlier weeks.
    Returns the predicted weeks with a `projection` column."""
    d = prepare(df)
    key = _week_key(d)
    weeks = _weeks(d)
    out = []
    t0 = time.time()
    for i, wk in enumerate(weeks):
        if i < min_train_weeks:
            continue
        past = d[key < wk]
        cur = d[key == wk]
        if past.empty or cur.empty:
            continue
        pred = fn(past, cur)
        if isinstance(pred, pd.Series):
            pred = pred.reindex(cur.index)
        else:
            pred = pd.Series(np.asarray(pred, dtype=float), index=cur.index)
        out.append(cur.assign(projection=pred.to_numpy(dtype=float)))
        if verbose:
            print(f"  {wk // 100} wk{wk % 100:>2}  {time.time() - t0:.0f}s",
                  flush=True)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def walk_forward_frame(df: pd.DataFrame, fn, min_train_weeks: int = 8,
                       verbose: bool = False) -> pd.DataFrame:
    """Like walk_forward, for an fn returning a frame keyed on player_dk_id."""
    d = prepare(df)
    key = _week_key(d)
    out = []
    t0 = time.time()
    for i, wk in enumerate(_weeks(d)):
        if i < min_train_weeks:
            continue
        past, cur = d[key < wk], d[key == wk]
        got = fn(past, cur)
        if got is None or got.empty:
            continue
        clash = [c for c in got.columns if c in cur.columns and c != "player_dk_id"]
        out.append(cur.drop(columns=clash).merge(got, on="player_dk_id",
                                                 how="left"))
        if verbose:
            print(f"  {wk // 100} wk{wk % 100:>2}  {time.time() - t0:.0f}s",
                  flush=True)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def accuracy(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (s, w), g in pred.groupby(["season", "week"]):
        g = g.dropna(subset=["projection"])
        if len(g) < 3:
            continue
        err = g["projection"] - g["dk_scored"]
        rows.append({
            "season": s, "week": w, "n": len(g),
            "spearman": float(stats.spearmanr(g["projection"],
                                              g["dk_scored"]).statistic),
            "rmse": float(np.sqrt((err ** 2).mean())),
            "mae": float(err.abs().mean()),
            "bias": float(err.mean()),
        })
    return pd.DataFrame(rows)


def lineup_results(pred: pd.DataFrame, n_lineups: int = 1,
                   objective: str = "projection") -> pd.DataFrame:
    """Each week's optimized lineups scored on what actually happened, next
    to the best lineup that was available in hindsight."""
    rows = []
    for (s, w), g in pred.groupby(["season", "week"]):
        pool = g.dropna(subset=[objective])
        pool = pool[pool["salary"].fillna(0) > 0]
        if len(pool) < 20:
            continue
        built = optimizer.optimize(pool, objective=objective,
                                   n_lineups=n_lineups)
        best = optimizer.optimize(pool.assign(_a=pool["dk_scored"]),
                                  objective="_a", n_lineups=1)
        if not built or not best:
            continue
        ceiling = float(best[0]["dk_scored"].sum())
        for lu in built:
            actual = float(lu["dk_scored"].sum())
            rows.append({
                "season": s, "week": w, "lineup": int(lu["lineup"].iloc[0]),
                "projected": float(lu[objective].sum()), "actual": actual,
                "ceiling": ceiling,
                "pct_of_ceiling": actual / ceiling if ceiling else np.nan,
                "played": int(lu["played"].sum()) if "played" in lu else np.nan,
            })
    return pd.DataFrame(rows)


def calibration(pred: pd.DataFrame, buckets: int = 10) -> pd.DataFrame:
    """Predicted against actual by projection decile WITHIN each week. The top
    bucket is the one the optimizer rosters from, so it is the one that
    matters (docs/MODELING.md 5)."""
    p = pred.dropna(subset=["projection"]).copy()
    if p.empty:
        return pd.DataFrame()
    p["decile"] = p.groupby(["season", "week"])["projection"].transform(
        lambda s: pd.qcut(s.rank(method="first"), buckets, labels=False))
    out = p.groupby("decile").agg(n=("projection", "size"),
                                  predicted=("projection", "mean"),
                                  actual=("dk_scored", "mean")).reset_index()
    out["gap"] = out["predicted"] - out["actual"]
    return out


def summarize(acc: pd.DataFrame, lineups: pd.DataFrame, name: str) -> str:
    sp = acc["spearman"].mean() if not acc.empty else float("nan")
    rmse = acc["rmse"].mean() if not acc.empty else float("nan")
    first = lineups[lineups["lineup"] == 0] if not lineups.empty else lineups
    lu = first["actual"].mean() if not first.empty else float("nan")
    pct = first["pct_of_ceiling"].mean() if not first.empty else float("nan")
    return (f"{name:<22} weeks={len(acc):>3}  spearman={sp:.4f}  "
            f"rmse={rmse:.3f}  lineup={lu:6.1f}  of best={pct:.1%}")

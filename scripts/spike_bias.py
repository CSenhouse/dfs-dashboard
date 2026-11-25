"""Does a projection over-react to one big game? (regression to the mean)

A player scores far above his usual output; next week his projection jumps.
If spikes are mostly noise, the next week regresses and the projection was
too high. The model sees form only through 3- and 6-game averages and the
prior season, and it learned their weights from history -- so it may
already discount spikes. This measures whether it does, and whether DFF's
half of the Blend does.

For every walk-forward prediction (each week from earlier weeks only), the
player's LAST game is compared with his baseline: the mean of up to six
appearances before it. spike = last - baseline. Residual = actual -
projection. If the model over-reacts, residuals are negative after big
spikes and positive after big duds, beyond what similar players without a
spike show. Restricted to projections >= 5 (the rostered range) and to
players with at least three prior appearances.

DFF is checked on the 2026 slates that have a DFF export bound to them.

    PYTHONPATH=src .venv/Scripts/python.exe scripts/spike_bias.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from dfs import backtest, model, slates
from dfs.config import CURATED, DATA
from dfs.sources import dff

BINS = [-np.inf, -10, -5, 5, 10, np.inf]
LABELS = ["dud <= -10", "-10 to -5", "normal", "+5 to +10", "spike >= +10"]


def history() -> pd.DataFrame:
    """(gsis_id, season, week) -> last game and the baseline before it."""
    pw = pd.read_parquet(CURATED / "player_week.parquet",
                         columns=["gsis_id", "season", "week", "dk_actual"])
    h = (pw.dropna(subset=["gsis_id", "dk_actual"])
           .drop_duplicates(["gsis_id", "season", "week"])
           .sort_values(["gsis_id", "season", "week"]))
    g = h.groupby("gsis_id")["dk_actual"]
    h["last"] = g.shift(1)
    h["baseline"] = g.transform(
        lambda s: s.shift(2).rolling(6, min_periods=3).mean())
    h["spike"] = h["last"] - h["baseline"]
    return h[["gsis_id", "season", "week", "last", "baseline", "spike"]]


def table(d: pd.DataFrame, proj: str, label: str) -> None:
    d = d.dropna(subset=["spike", proj, "dk_actual"]).copy()
    d = d[d[proj] >= 5]
    d["res"] = d["dk_actual"] - d[proj]
    d["bucket"] = pd.cut(d["spike"], BINS, labels=LABELS)
    print(f"\n{label}: {len(d):,} player-weeks, residual = actual - projection")
    print(f"  {'last game vs baseline':<22}{'n':>7}{'residual':>10}"
          f"{'vs normal':>11}{'p':>9}")
    normal = d.loc[d["bucket"] == "normal", "res"]
    for b in LABELS:
        x = d.loc[d["bucket"] == b, "res"]
        if len(x) < 20:
            continue
        p = stats.mannwhitneyu(x, normal).pvalue if b != "normal" else np.nan
        print(f"  {b:<22}{len(x):>7}{x.mean():>+10.2f}"
              f"{x.mean() - normal.mean():>+11.2f}{p:>9.4f}")
    sl = stats.linregress(d["spike"], d["res"])
    print(f"  slope: residual per point of spike {sl.slope:+.4f} "
          f"(p={sl.pvalue:.4f}); negative = over-reacting to the last game")


def dff_frame() -> pd.DataFrame:
    rows = []
    for f in sorted((DATA / "projections").glob("dg*__*.csv")):
        dgid = int(f.name[2:f.name.index("__")])
        try:
            up = slates.assemble(dgid)
        except Exception:
            continue
        if up.empty or up["dk_actual"].isna().all() or up["dk_actual"].nunique() < 20:
            continue
        if up["competition_id"].nunique() < 3:      # Classic slates only
            continue
        theirs = dff.load(f, force=True)
        m = dff.match(theirs, up[["player_dk_id", "player_name", "position",
                                  "team", "salary"]])
        m = m[m["matched"]][["player_dk_id", "dff_proj"]]
        x = up.merge(m, on="player_dk_id", how="inner")
        rows.append(x[["gsis_id", "season", "week", "dk_actual", "dff_proj"]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main() -> int:
    h = history()
    f = pd.read_parquet(CURATED / "features.parquet")
    print("walk-forward with the current model ...", flush=True)
    pred = backtest.walk_forward(f, model.two_stage, min_train_weeks=8)
    pred = pred.merge(h, on=["gsis_id", "season", "week"], how="left")
    table(pred, "projection", "MODEL (all walk-forward weeks)")
    recent = pred[pred["season"] == 2026]
    table(recent, "projection", "MODEL, 2026 only")

    d = dff_frame()
    if not d.empty:
        d = d.merge(h, on=["gsis_id", "season", "week"], how="left")
        table(d, "dff_proj", f"DFF ({d['week'].nunique()} slates of 2026)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

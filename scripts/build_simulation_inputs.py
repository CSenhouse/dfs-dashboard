"""Walk-forward distributional predictions: the input to every simulation.

Fits `model.quantile_grid` on each week's past and predicts the week, exactly
as `scripts/evaluate.py` does for a point projection. The output is the only
honest basis for measuring two things:

  * how player outcomes co-move once the model's own information is removed
    (scripts/correlations.py)
  * whether the simulated distribution is calibrated at all
    (scripts/validate_simulation.py)

Both need out-of-sample quantiles. Fitted in-sample, the residuals would be
too small and too independent, and the simulator would come out looking
confident and well-behaved for the same reason a random k-fold makes any time
series model look excellent.

    PYTHONPATH=src .venv/Scripts/python.exe scripts/build_simulation_inputs.py
    PYTHONPATH=src .venv/Scripts/python.exe scripts/build_simulation_inputs.py --showdown
"""
from __future__ import annotations

import sys
import time

import pandas as pd

from dfs import backtest, model
from dfs.config import CURATED

KEEP = [
    "season", "week", "draft_group_id", "player_dk_id", "player_name",
    "position", "team", "opponent", "salary", "dk_scored", "played",
    "p_play", "points_mean", "projection",
]


def main(argv: list[str]) -> int:
    showdown = "--showdown" in argv
    src = "showdown_features.parquet" if showdown else "features.parquet"
    dst = ("showdown_sim_predictions.parquet" if showdown
           else "sim_predictions.parquet")

    f = pd.read_parquet(CURATED / src)
    print(f"{src}: {len(f):,} rows", flush=True)

    t0 = time.time()
    pred = backtest.walk_forward_frame(f, model.quantile_grid, verbose=True)
    if pred.empty:
        print("no predictions produced", file=sys.stderr)
        return 1

    cols = KEEP + model.quantile_columns(pred)
    out = pred[[c for c in cols if c in pred.columns]].copy()
    out.to_parquet(CURATED / dst, index=False)

    weeks = out[["season", "week"]].drop_duplicates()
    print(f"\n{len(out):,} rows over {len(weeks)} weeks "
          f"in {time.time() - t0:.0f}s -> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

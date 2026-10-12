"""Baselines every model has to beat, on identical weeks.

expected         "informed": recent form, falling back to what the salary
                 implies when there is no form
expected_salary  "naive": DraftKings' own price, read as points by position
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _salary_fit(past: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Per-position least squares of points on salary, zeros included."""
    out = {}
    for pos, g in past.dropna(subset=["salary"]).groupby("position"):
        if len(g) < 10 or g["salary"].nunique() < 2:
            continue
        b, a = np.polyfit(g["salary"].to_numpy(float),
                          g["dk_scored"].to_numpy(float), 1)
        out[pos] = (a, b)
    return out


def expected_salary(past: pd.DataFrame, cur: pd.DataFrame) -> pd.Series:
    fit = _salary_fit(past)
    sal = cur["salary"].astype(float)
    a = cur["position"].map(lambda p: fit.get(p, (0.0, 0.0))[0])
    b = cur["position"].map(lambda p: fit.get(p, (0.0, 0.0))[1])
    return (a + b * sal).clip(lower=0).astype(float)


def expected(past: pd.DataFrame, cur: pd.DataFrame) -> pd.Series:
    by_price = expected_salary(past, cur)
    if "dk_points_r3" not in cur.columns:
        return by_price
    form = cur["dk_points_r3"]
    if "dk_points_r6" in cur.columns:
        form = form.fillna(cur["dk_points_r6"])
    return form.fillna(by_price).clip(lower=0).astype(float)

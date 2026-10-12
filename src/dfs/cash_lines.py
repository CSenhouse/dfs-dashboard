"""Observed cash lines: the only ground truth about contest outcomes here.

`data/observed_cash_lines.csv` is edited by hand after a contest. Everything
else -- the field sampler, every cash probability -- models the field with
nothing to anchor it, and does so too softly. So an observation beats an
estimate, a line is never applied to a slate it says nothing about, and a
mangled row fails loudly instead of silently falling back.

A tournament line is not a cash line: the field is a property of the contest,
not the slate. Only rows paying close to a double-up's 43.5% count, and rows
that predate the paid_pct column were all double-ups.
"""
from __future__ import annotations

import csv
import io

import numpy as np
import pandas as pd

from dfs.config import DATA

PATH = DATA / "observed_cash_lines.csv"

MAIN, MEDIUM, SMALL, SHOWDOWN = "main", "medium", "small", "showdown"

# Used only when nothing has been observed for that size of slate.
ESTIMATES = {MAIN: 140, MEDIUM: 135, SMALL: 125, SHOWDOWN: 90}

DOUBLE_UP = (0.40, 0.50)


def slate_class(n_games: int, showdown: bool = False) -> str:
    if showdown:
        return SHOWDOWN
    if n_games >= 9:
        return MAIN
    if n_games >= 5:
        return MEDIUM
    return SMALL


def load() -> pd.DataFrame:
    """The recorded lines. Raises ValueError on a malformed row."""
    if not PATH.exists():
        return pd.DataFrame()
    text = PATH.read_text(encoding="utf-8-sig")
    rows = list(csv.reader(io.StringIO(text)))
    rows = [r for r in rows if any(c.strip() for c in r)]
    if not rows:
        return pd.DataFrame()
    width = len(rows[0])
    for n, r in enumerate(rows[1:], start=2):
        if len(r) != width:
            raise ValueError(
                f"{PATH.name} line {n} has {len(r)} fields against {width} in "
                "the header -- a note containing a comma must be quoted")
    df = pd.read_csv(io.StringIO(text))
    if df.empty:
        return df
    line = pd.to_numeric(df["line"], errors="coerce")
    if line.isna().any():
        bad = df.loc[line.isna(), "line"].tolist()
        raise ValueError(f"{PATH.name}: non-numeric line values {bad}; a note "
                         "containing a comma must be quoted")
    df["line"] = line
    df["kind"] = df["kind"].astype(str).str.strip().str.lower()
    return df


def default_line(n_games: int, showdown: bool = False) -> tuple[int, str]:
    """(line, where it came from) for a slate of this size and format."""
    cls = slate_class(int(n_games), showdown)
    estimate = (ESTIMATES[cls], f"estimate for a {cls} slate, nothing observed")
    try:
        df = load()
    except ValueError as exc:
        return ESTIMATES[cls], f"estimate -- {exc}"
    if df.empty:
        return estimate

    is_sd = df["kind"].eq("showdown")
    df = df[[slate_class(int(g), bool(s)) == cls
             for g, s in zip(df["games"], is_sd)]]
    if "paid_pct" in df.columns:
        paid = pd.to_numeric(df["paid_pct"], errors="coerce")
        df = df[paid.isna() | paid.between(*DOUBLE_UP)]
    if df.empty:
        return estimate

    weekly = df.groupby(["season", "week"])["line"].median()
    w, c = len(weekly), len(df)
    line = int(round(float(np.median(weekly.to_numpy()))))
    source = f"median of {w} observed week{'s' if w != 1 else ''}"
    if c > w:
        source += f" ({c} contests)"
    if w > 1:
        source += (f", ranged {int(round(weekly.min()))} to "
                   f"{int(round(weekly.max()))}")
    return line, source

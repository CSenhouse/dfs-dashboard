"""Real contest standings, exported from the DraftKings gamecenter.

The only ground truth about fields in this project: every lineup, every score
and the exact ownership of every player. Export while logged in, THE SAME
WEEK -- they appear to expire after one to two weeks.

An export is two tables printed side by side in one CSV: one row per entry on
the left, one row per player on the right, padded to the same height and
unrelated row by row. They are separated here before anything reads them.
"""
from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from dfs.config import CAPTAIN_MULTIPLIER

SLOTS = ("CPT", "QB", "RB", "WR", "TE", "FLEX", "DST")
_LABEL = re.compile(r"(?:^|\s)(" + "|".join(SLOTS) + r")\s+")
MULTIPLIER = {"CPT": CAPTAIN_MULTIPLIER}


def _read(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
            raw = z.read(name)
    else:
        raw = path.read_bytes()
    return pd.read_csv(io.BytesIO(raw), dtype=str, keep_default_na=False,
                       encoding="utf-8-sig")


def load(path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(entries, players) from a standings export, .csv or .zip."""
    raw = _read(path)
    e = raw[raw["EntryId"].str.strip() != ""]
    entries = pd.DataFrame({
        "rank": pd.to_numeric(e["Rank"], errors="coerce").astype("Int64"),
        "entry_id": e["EntryId"].str.strip(),
        "entry_name": e["EntryName"].str.strip(),
        "points": pd.to_numeric(e["Points"], errors="coerce"),
        "lineup": e["Lineup"].str.strip(),
    }).reset_index(drop=True)

    p = raw[raw["Player"].str.strip() != ""]
    slot = p["Roster Position"].str.strip()
    pts = pd.to_numeric(p["FPTS"], errors="coerce")
    players = pd.DataFrame({
        "player_name": p["Player"].str.strip(),
        "slot": slot,
        "pct_drafted": pd.to_numeric(
            p["%Drafted"].str.replace("%", "", regex=False), errors="coerce"
        ) / 100.0,
        "dk_scored": pts,
    }).reset_index(drop=True)
    # A Captain's row carries 1.5x, because that is what the Captain scored.
    # Read as the player's own points it inflates every captain by half.
    mult = players["slot"].map(MULTIPLIER).fillna(1.0)
    players["base_points"] = players["dk_scored"] / mult
    return entries, players


def parse_lineup(text) -> list[tuple[str, str]]:
    """'QB Bryce Young RB ...' -> [('QB', 'Bryce Young'), ...].

    Names carry spaces, apostrophes and suffixes, and nothing separates one
    player from the next slot label, so the split is on the labels.
    """
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return []
    text = str(text).strip()
    if not text:
        return []
    marks = list(_LABEL.finditer(text))
    out = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        name = text[m.end():end].strip()
        if name:
            out.append((m.group(1), name))
    return out


def ownership(players: pd.DataFrame) -> pd.DataFrame:
    """Share of lineups holding each player at all: DK reports a receiver at
    WR and again at FLEX, and the share is the sum."""
    if players.empty:
        return pd.DataFrame(columns=["player_name", "pct_drafted", "dk_scored"])
    g = players.groupby("player_name", as_index=False).agg(
        pct_drafted=("pct_drafted", "sum"), dk_scored=("base_points", "max"))
    return g.sort_values("pct_drafted", ascending=False).reset_index(drop=True)


def field_summary(entries: pd.DataFrame, paid_positions: int | None = None
                  ) -> dict:
    """Distribution of a counted field, and its cash line if the payout is
    known (the score at the last paid position)."""
    pts = entries["points"].dropna().to_numpy(dtype=float)
    out = {
        "entries": int(len(pts)),
        "mean": float(pts.mean()) if len(pts) else float("nan"),
        "sd": float(pts.std(ddof=1)) if len(pts) > 1 else float("nan"),
        "min": float(pts.min()) if len(pts) else float("nan"),
        "max": float(pts.max()) if len(pts) else float("nan"),
    }
    for p in (10, 25, 50, 75, 90, 99):
        out[f"p{p}"] = float(np.percentile(pts, p)) if len(pts) else float("nan")
    if paid_positions:
        ranked = np.sort(pts)[::-1]
        out["cash_line"] = float(ranked[min(int(paid_positions), len(ranked)) - 1])
        out["paid_positions"] = int(paid_positions)
    return out

"""Paths and DraftKings constants shared across the package."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
CACHE = DATA / "cache"          # gzipped raw HTTP responses. THE ASSET.
CURATED = DATA / "curated"      # derived tables, rebuildable from the cache

for _d in (DATA, CACHE, CURATED):
    _d.mkdir(parents=True, exist_ok=True)

SALARY_CAP = 50_000
CAPTAIN_MULTIPLIER = 1.5

CLASSIC_ROSTER = {"QB": 1, "RB": 2, "WR": 3, "TE": 1, "FLEX": 1, "DST": 1}
CLASSIC_SIZE = 9
SHOWDOWN_SIZE = 6

"""Rebuild the historical DraftKings salary archive.

A blind sweep of every draft group id is ~70k requests. Ids increase
monotonically with time, so we instead sample sparsely to build an id->date
index, binary-search the NFL season windows, and densely sweep only those
bands. Everything is cached on disk, so this is resumable and only ever pays
for each id once -- including ids that turned out to be empty.

SLOW AND SEQUENTIAL on purpose. `api.draftkings.com` already returns 403 to
this project after an earlier sweep ran six workers at ~12 req/s; the legacy
endpoint used now is the last salary source left, and the live dashboard
needs it. One request at a time, `--interval` seconds apart, and the sweep
stops itself after a run of refusals rather than hammering on.

    python scripts/backfill_salaries.py --seasons 2025 2026 --id-min 128000
"""
from __future__ import annotations

import argparse
import bisect
import datetime as dt
import sys
import time

import pandas as pd

from dfs import http
from dfs.config import CURATED
from dfs.sources import dk_api

ID_MIN, ID_MAX = 10_000, 160_000
SAMPLE_STEP = 250
INTERVAL = 0.7          # seconds between requests: ~1.4 req/s
MAX_REFUSALS = 8        # consecutive 403/429s before giving up

# NFL runs early Sep through mid Jan; pad generously for preseason and playoffs.
SEASON_START = (8, 1)
SEASON_END = (1, 20)


class Refused(RuntimeError):
    pass


_refusals = [0]


def fetch(dgid: int, interval: float) -> pd.DataFrame:
    """Draftables for one id, pausing on 429 and stopping on repeated 403."""
    for attempt in range(4):
        df = dk_api.draftables(dgid, min_interval=interval, retry_miss=False)
        status = http.last_status
        if status in (403, 429) or (status is None and df.empty):
            _refusals[0] += 1
            if _refusals[0] >= MAX_REFUSALS:
                raise Refused(f"{_refusals[0]} consecutive refusals "
                              f"(last status {status}) at id {dgid}")
            wait = 60 * (attempt + 1)
            print(f"    status {status} at id {dgid}; waiting {wait}s",
                  flush=True)
            time.sleep(wait)
            continue
        _refusals[0] = 0
        return df
    return pd.DataFrame()


def _start(df: pd.DataFrame) -> dt.datetime | None:
    if df.empty or df["start_time"].isna().all():
        return None
    return df["start_time"].min().to_pydatetime()


def build_date_index(lo: int, hi: int, step: int, interval: float
                     ) -> list[tuple[int, dt.datetime]]:
    ids = list(range(lo, hi + 1, step))
    print(f"sampling {len(ids)} ids ({lo}..{hi} step {step}) to map id -> date",
          flush=True)
    out: list[tuple[int, dt.datetime]] = []
    for i, dgid in enumerate(ids, 1):
        when = _start(fetch(dgid, interval))
        if when is not None:
            out.append((dgid, when))
        if i % 25 == 0:
            print(f"  {i}/{len(ids)} sampled, {len(out)} dated", flush=True)
    out.sort()
    # Enforce monotonicity: a handful of ids are issued out of order.
    cleaned: list[tuple[int, dt.datetime]] = []
    for dgid, when in out:
        while cleaned and cleaned[-1][1] > when:
            cleaned.pop()
        cleaned.append((dgid, when))
    return cleaned


def band_for(index: list[tuple[int, dt.datetime]], start: dt.datetime,
             end: dt.datetime, pad: int) -> tuple[int, int]:
    dates = [d for _, d in index]
    lo_i = bisect.bisect_left(dates, start)
    hi_i = bisect.bisect_right(dates, end)
    lo = index[max(lo_i - 1, 0)][0] - pad
    hi = index[min(hi_i, len(index) - 1)][0] + pad
    return max(lo, ID_MIN), min(hi, ID_MAX)


def sweep(lo: int, hi: int, interval: float, season: int) -> list[dict]:
    ids = list(range(lo, hi + 1))
    print(f"  sweeping {len(ids)} ids [{lo}..{hi}]", flush=True)
    rows: list[dict] = []
    t0 = time.time()
    for i, dgid in enumerate(ids, 1):
        df = fetch(dgid, interval)
        if not df.empty:
            info = dk_api.classify(df)
            if info["is_nfl"] and info["kind"] in ("classic", "showdown"):
                rows.append({"draft_group_id": dgid, "kind": info["kind"],
                             "n_players": info["n"], "n_games": info["n_games"],
                             "start_time": info["start_time"]})
        if i % 250 == 0 or i == len(ids):
            rate = i / max(time.time() - t0, 1e-9)
            left = (len(ids) - i) / max(rate, 1e-9)
            print(f"    [{season}] {i}/{len(ids)}  NFL slates={len(rows)}  "
                  f"{rate:.2f}/s  ~{left / 60:.0f} min left", flush=True)
        if i % 1000 == 0 and rows:
            pd.DataFrame(rows).to_parquet(
                CURATED / f"dk_slate_catalog_{season}.partial.parquet",
                index=False)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="+", required=True)
    ap.add_argument("--step", type=int, default=SAMPLE_STEP)
    ap.add_argument("--pad", type=int, default=400)
    ap.add_argument("--interval", type=float, default=INTERVAL)
    ap.add_argument("--id-min", type=int, default=ID_MIN)
    ap.add_argument("--id-max", type=int, default=ID_MAX)
    ap.add_argument("--index-only", action="store_true")
    args = ap.parse_args()

    try:
        index = build_date_index(args.id_min, args.id_max, args.step,
                                 args.interval)
        if not index:
            print("no dated ids found -- API shape may have changed")
            return 1
        print(f"index: {len(index)} points, "
              f"{index[0][1].date()} (id {index[0][0]}) .. "
              f"{index[-1][1].date()} (id {index[-1][0]})", flush=True)
        pd.DataFrame(index, columns=["draft_group_id", "start_time"]).to_parquet(
            CURATED / "dk_id_date_index.parquet", index=False)
        if args.index_only:
            return 0

        all_rows: list[dict] = []
        for season in args.seasons:
            start = dt.datetime(season, *SEASON_START, tzinfo=dt.timezone.utc)
            end = dt.datetime(season + 1, *SEASON_END, tzinfo=dt.timezone.utc)
            lo, hi = band_for(index, start, end, args.pad)
            # Never probe past the newest id that exists: a not-yet-created
            # id would be cached as a permanent miss.
            lo, hi = max(lo, args.id_min), min(hi, args.id_max)
            print(f"\n[{season}] {start.date()}..{end.date()} -> ids {lo}..{hi}",
                  flush=True)
            rows = sweep(lo, hi, args.interval, season)
            all_rows.extend(rows)
            # Persist after each season: a long sweep must not throw away
            # completed seasons if it is interrupted.
            if rows:
                pd.DataFrame(rows).to_parquet(
                    CURATED / f"dk_slate_catalog_{season}.parquet", index=False)
                print(f"[{season}] wrote {len(rows)} slates", flush=True)
    except Refused as exc:
        print(f"\nSTOPPED: DraftKings is refusing requests ({exc}). "
              "Everything fetched so far is cached; re-run later to resume.",
              flush=True)
        return 2

    if not all_rows:
        print("no NFL slates found")
        return 1
    known = []
    for p in CURATED.glob("dk_slate_catalog_*.parquet"):
        if "partial" not in p.name:
            known.append(pd.read_parquet(p))
    cat = pd.concat(known, ignore_index=True).drop_duplicates("draft_group_id")
    cat = cat.sort_values("start_time")
    out = CURATED / "dk_slate_catalog.parquet"
    cat.to_parquet(out, index=False)
    print(f"\nwrote {len(cat)} NFL slates -> {out}")
    print(cat.groupby("kind").size().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())

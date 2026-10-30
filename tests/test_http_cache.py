"""HTTP cache semantics.

An empty cache file records a miss so a sweep does not re-request known holes.
That is right for an offline sweep and wrong for a live lookup: a draft group
probed before DraftKings created it is cached as a 404 and would stay
invisible forever. Live Showdown slates vanished from the dashboard exactly
this way -- the lobby listed 23 draft groups, all returning HTTP 200 with full
payloads, and only one reached the UI.
"""
from __future__ import annotations

import datetime as dt
import gzip
import os
import time

import pytest
import requests

from dfs import http


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(http, "CACHE", tmp_path)
    return tmp_path


class FakeResponse:
    def __init__(self, status: int, body: bytes):
        self.status_code = status
        self.content = body


def stub(monkeypatch, status: int, body: bytes = b"ok"):
    calls = []

    def fake_get(url, timeout=None):
        calls.append(url)
        return FakeResponse(status, body)

    monkeypatch.setattr(http._session, "get", fake_get)
    return calls


URL = "https://example.test/thing.json"


def test_a_miss_is_recorded_as_an_empty_file(cache, monkeypatch):
    stub(monkeypatch, 404)
    assert http.get(URL, suffix=".json", min_interval=0) is None
    marker = http.cache_path(URL, ".json")
    assert marker.with_name(marker.name + ".gz").read_bytes() == b""


def test_cache_only_honours_a_recorded_miss(cache, monkeypatch):
    stub(monkeypatch, 404)
    http.get(URL, suffix=".json", min_interval=0)
    calls = stub(monkeypatch, 200, b"now it exists")
    assert http.get(URL, suffix=".json", cache_only=True) is None
    assert calls == [], "cache_only must never touch the network"


def test_a_live_lookup_retries_a_recorded_miss(cache, monkeypatch):
    """The bug. A draft group that did not exist when it was swept must not
    stay invisible once DraftKings creates it."""
    stub(monkeypatch, 404)
    assert http.get(URL, suffix=".json", min_interval=0) is None

    calls = stub(monkeypatch, 200, b"now it exists")
    got = http.get(URL, suffix=".json", min_interval=0)
    assert got == b"now it exists"
    assert calls == [URL], "a recorded miss should be retried when online"


def test_a_successful_body_is_served_from_cache(cache, monkeypatch):
    stub(monkeypatch, 200, b"payload")
    assert http.get(URL, suffix=".json", min_interval=0) == b"payload"

    calls = stub(monkeypatch, 200, b"different")
    assert http.get(URL, suffix=".json", min_interval=0) == b"payload"
    assert calls == [], "a cached hit must not re-request"


def test_refresh_overrides_a_cached_body(cache, monkeypatch):
    stub(monkeypatch, 200, b"old")
    http.get(URL, suffix=".json", min_interval=0)
    stub(monkeypatch, 200, b"new")
    assert http.get(URL, suffix=".json", min_interval=0, refresh=True) == b"new"


def test_bodies_are_stored_gzipped(cache, monkeypatch):
    stub(monkeypatch, 200, b"x" * 5000)
    http.get(URL, suffix=".json", min_interval=0)
    p = http.cache_path(URL, ".json")
    gz = p.with_name(p.name + ".gz")
    assert gzip.decompress(gz.read_bytes()) == b"x" * 5000
    assert gz.stat().st_size < 5000


# --- content that changes at a fixed URL ------------------------------------
# Almost every response here is immutable once written, which is why the cache
# is the asset. nflverse is the exception: it rewrites the CURRENT season's box
# scores every week at the same address. Cached forever, a live slate gets
# projected on a snapshot taken mid-week-1 -- rolling form sees no games,
# weeks_since_played climbs for everyone, and participation drags every
# projection toward zero, worse each week. A week 3 slate came out at a third
# of its week 2 numbers before this was found.

def age(url: str, suffix: str, seconds: float) -> None:
    """Backdate a cache entry so it is older than its max age."""
    path = http.cache_path(url, suffix)
    gz = path.with_name(path.name + ".gz")
    stamp = time.time() - seconds
    os.utime(gz, (stamp, stamp))


def test_a_fresh_entry_is_served_without_a_request(cache, monkeypatch):
    stub(monkeypatch, 200, b"first")
    http.get(URL, suffix=".json", min_interval=0)
    calls = stub(monkeypatch, 200, b"second")
    assert http.get(URL, suffix=".json", max_age=3600) == b"first"
    assert calls == [], "a fresh entry should not hit the network"


def test_a_stale_entry_is_replaced(cache, monkeypatch):
    stub(monkeypatch, 200, b"week one only")
    http.get(URL, suffix=".json", min_interval=0)
    age(URL, ".json", 7200)

    stub(monkeypatch, 200, b"weeks one to three")
    assert http.get(URL, suffix=".json", min_interval=0,
                    max_age=3600) == b"weeks one to three"
    assert http.get(URL, suffix=".json", cache_only=True) == b"weeks one to three"


def test_without_a_max_age_an_entry_is_cached_forever(cache, monkeypatch):
    """A completed season is final and must never be re-fetched, whatever its
    age on disk. Re-downloading five seasons on every page load would be a
    different bug."""
    stub(monkeypatch, 200, b"final")
    http.get(URL, suffix=".json", min_interval=0)
    age(URL, ".json", 10 ** 7)

    calls = stub(monkeypatch, 200, b"should never be asked for")
    assert http.get(URL, suffix=".json") == b"final"
    assert calls == []


def test_a_failed_refresh_keeps_the_old_copy(cache, monkeypatch):
    """The cache is the asset. A stale entry is worth far more than nothing,
    so a non-OK refresh must not blank it."""
    stub(monkeypatch, 200, b"last good copy")
    http.get(URL, suffix=".json", min_interval=0)
    age(URL, ".json", 7200)

    stub(monkeypatch, 503, b"")
    assert http.get(URL, suffix=".json", min_interval=0,
                    max_age=3600) == b"last good copy"
    assert http.get(URL, suffix=".json", cache_only=True) == b"last good copy"


def test_a_network_error_during_refresh_keeps_the_old_copy(cache, monkeypatch):
    stub(monkeypatch, 200, b"last good copy")
    http.get(URL, suffix=".json", min_interval=0)
    age(URL, ".json", 7200)

    def explode(url, timeout=None):
        raise requests.ConnectionError("no network")

    monkeypatch.setattr(http._session, "get", explode)
    assert http.get(URL, suffix=".json", min_interval=0,
                    max_age=3600) == b"last good copy"


def test_offline_analysis_ignores_max_age(cache, monkeypatch):
    """cache_only must never reach the network, however old the entry is."""
    stub(monkeypatch, 200, b"old but all we have")
    http.get(URL, suffix=".json", min_interval=0)
    age(URL, ".json", 10 ** 7)

    calls = stub(monkeypatch, 200, b"fresher")
    assert http.get(URL, suffix=".json", max_age=1,
                    cache_only=True) == b"old but all we have"
    assert calls == []


def test_only_the_season_in_progress_expires():
    from dfs.sources import nflverse

    live = nflverse.current_season()
    assert nflverse._max_age(live) == nflverse.LIVE_MAX_AGE
    assert nflverse._max_age(live - 1) is None
    assert nflverse._max_age(None) is None


def test_a_season_is_named_for_the_year_it_started():
    """January belongs to the previous season, so a playoff-week refresh must
    not start asking for a season that has not begun."""
    from dfs.sources import nflverse

    assert nflverse.current_season(dt.date(2026, 9, 26)) == 2026
    assert nflverse.current_season(dt.date(2027, 1, 15)) == 2026
    assert nflverse.current_season(dt.date(2027, 3, 1)) == 2027


def test_a_draft_group_cached_before_it_was_priced_is_fetched_again(
        cache, monkeypatch):
    """DK creates featured main slates weeks ahead. A sweep that reached one
    early cached a body with no players, and cached forever the week 8 main
    slate vanished from the dashboard while DK served it in full."""
    import json
    from dfs.sources import dk_api

    empty = json.dumps({"playerList": [], "teamList": {}}).encode()
    stub(monkeypatch, 200, empty)
    assert dk_api.draftables(153728, min_interval=0).empty

    priced = json.dumps({
        "playerList": [{"pid": 1, "pdkid": 1, "fn": "A", "ln": "B", "pn": "QB",
                        "tid": 10, "htid": 10, "atid": 20, "htabbr": "DET",
                        "atabbr": "MIN", "s": 6000, "i": "", "tsid": 7,
                        "dgst": 1793556000000}],
        "teamList": {}}).encode()
    stub(monkeypatch, 200, priced)
    assert len(dk_api.draftables(153728, min_interval=0)) == 1
    # ...and offline analysis still never touches the network.
    calls = stub(monkeypatch, 200, priced)
    dk_api.draftables(999, cache_only=True)
    assert calls == []

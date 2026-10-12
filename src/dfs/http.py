"""Disk-cached HTTP.

Every response is stored gzipped under data/cache, keyed on the URL. The cache
is the asset: re-running a sweep is free, re-downloading it is not.

An empty cache file records a miss, so an offline pass (`cache_only=True`)
does not re-request known holes. A live lookup retries a recorded miss,
because a draft group probed before DraftKings created it would otherwise stay
invisible forever.

`max_age` exists for the one source that rewrites content at a fixed URL --
nflverse's in-progress season. Without it an entry is cached forever, which is
right for everything that is final. A failed refresh never destroys the copy
already held.
"""
from __future__ import annotations

import gzip
import hashlib
import threading
import time
from pathlib import Path

import requests

from dfs.config import CACHE

USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/140.0 Safari/537.36")

_session = requests.Session()
_session.headers.update({"User-Agent": USER_AGENT,
                         "Accept": "application/json, text/plain, */*"})

_lock = threading.Lock()
_last_request = [0.0]

# Status of the most recent network request ("cache" when none was made).
# A sweep reads it to notice being refused and stop, rather than recording
# thousands of 403s as missing data.
last_status: int | str | None = None

# Statuses that mean "this does not exist", worth remembering as a miss.
MISS_STATUSES = (400, 404, 410)


def cache_path(url: str, suffix: str = "") -> Path:
    """Where `url` is cached, before the .gz extension."""
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()
    return CACHE / f"{digest}{suffix}"


def _gz(url: str, suffix: str) -> Path:
    p = cache_path(url, suffix)
    return p.with_name(p.name + ".gz")


def _read(gz: Path) -> bytes:
    raw = gz.read_bytes()
    return gzip.decompress(raw) if raw else b""


def _write(gz: Path, body: bytes) -> None:
    gz.parent.mkdir(parents=True, exist_ok=True)
    tmp = gz.with_name(gz.name + ".part")
    tmp.write_bytes(gzip.compress(body, compresslevel=6) if body else b"")
    tmp.replace(gz)


def _throttle(min_interval: float) -> None:
    if min_interval <= 0:
        return
    with _lock:
        wait = _last_request[0] + min_interval - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last_request[0] = time.monotonic()


def get(url: str, suffix: str = "", *, min_interval: float = 1.0,
        cache_only: bool = False, refresh: bool = False,
        max_age: float | None = None, timeout: float = 60.0,
        retry_miss: bool = True) -> bytes | None:
    """Body of `url`, from the cache when possible. None means not available.

    cache_only  never touch the network, however old the entry
    refresh     re-request even a fresh hit (the old copy survives a failure)
    max_age     seconds after which a cached body is re-requested
    retry_miss  re-request a recorded miss; a resumable sweep passes False
    """
    global last_status
    last_status = "cache"
    gz = _gz(url, suffix)
    held: bytes | None = None
    if gz.exists():
        held = _read(gz)
        if cache_only or (not held and not retry_miss):
            return held or None
        if held and not refresh:
            age = time.time() - gz.stat().st_mtime
            if max_age is None or age <= max_age:
                return held
    elif cache_only:
        return None

    _throttle(min_interval)
    try:
        resp = _session.get(url, timeout=timeout)
    except requests.RequestException:
        last_status = None
        return held or None

    last_status = resp.status_code
    if resp.status_code == 200:
        body = resp.content
        _write(gz, body)
        return body
    if held:
        return held
    if resp.status_code in MISS_STATUSES:
        _write(gz, b"")
    return None

"""One-off: gzip cache entries written before compression was added."""
from __future__ import annotations

import gzip

from dfs.config import CACHE


def main() -> None:
    before = after = 0
    n = 0
    for p in sorted(CACHE.iterdir()):
        if p.suffix == ".gz" or not p.is_file():
            continue
        raw = p.read_bytes()
        gz = p.with_name(p.name + ".gz")
        gz.write_bytes(gzip.compress(raw, compresslevel=6) if raw else b"")
        before += len(raw)
        after += gz.stat().st_size
        p.unlink()
        n += 1
        if n % 500 == 0:
            print(f"  {n} migrated", flush=True)
    mb = 1024 * 1024
    print(f"migrated {n} files: {before/mb:,.0f}MB -> {after/mb:,.0f}MB "
          f"({after/before:.1%})" if before else "nothing to migrate")


if __name__ == "__main__":
    main()

"""Daily Fantasy Fuel's projection export: a second opinion, and (on their
premium tier) projected ownership.

IMPORTED, never scraped. Their projections are what they sell, and this
project already lost the DraftKings contest API by probing too hard. Download
"Projections as CSV" from their site and upload it in the dashboard sidebar,
or drop it in data/projections/.

Five traps, all found by real files:

1. An export taken before DFF post the slate has every projection at 0.0.
   That is not worse numbers but the claim nobody will score, which an
   optimizer answers with the nine cheapest names. `load` refuses it.
2. (team, position, salary) is unique for only 41% of a big slate -- the
   minimum salary collects whole receiving rooms -- so a second pass
   separates a crowded cell by name, where rule 1's failure cannot occur
   because team, position and price already agree.
3. DFF publish CLASSIC salaries, so on a Showdown slate the salary key breaks
   entirely. A third pass matches the name within (team, position).
4. One export covers the whole week, and a slate's start_time is UTC while
   DFF write the Eastern date, so a slate is found by its TEAMS.
5. The raw match rate counts players from games the slate does not have;
   `match_rate` restricts to on-slate teams.

Uploads can be BOUND to a slate (dg<id>__name.csv). DFF's per-slate download
carries that slate's own salaries, so `for_slate` prefers it.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from dfs.config import DATA
from dfs.crosswalk import normalize_team

DIR = DATA / "projections"

REQUIRED = ("first_name", "last_name", "position", "team", "salary",
            "ppg_projection")
OPTIONAL = ("opp", "injury_status", "game_date", "slate", "spread",
            "over_under", "implied_team_score", "ownership_projection",
            "value_projection")
SUFFIX = r"\b(jr|sr|ii|iii|iv|v)\b\.?"

# A slate is not usable if fewer players than this per game carry a
# projection: an export listing only the injury-flagged names is not a slate.
MIN_PROJECTED_PER_GAME = 15
# Share of a slate's teams an export must cover to be about that slate.
MIN_TEAM_COVERAGE = 0.8

TEAM_FIX = {"JAC": "JAX", "LAR": "LA", "WSH": "WAS", "LVR": "LV"}


def _team(s: pd.Series) -> pd.Series:
    return normalize_team(s.astype(str).str.strip().str.upper().replace(TEAM_FIX))


def _key_name(s: pd.Series) -> pd.Series:
    """'Aaron Jones Sr.' and 'Aaron Jones' -> 'aaronjones'."""
    return (
        s.fillna("").astype(str)
        .str.normalize("NFKD").str.encode("ascii", "ignore").str.decode("ascii")
        .str.lower()
        .str.replace(SUFFIX, " ", regex=True)
        .str.replace(r"[^a-z]", "", regex=True)
    )


def read(path) -> pd.DataFrame:
    """Parse an export into this project's column names. No judgement."""
    raw = pd.read_csv(path, dtype={"game_date": str})
    raw.columns = [c.strip() for c in raw.columns]
    missing = [c for c in REQUIRED if c not in raw.columns]
    if missing:
        raise ValueError(f"missing columns: {', '.join(missing)}")
    for c in OPTIONAL:
        if c not in raw.columns:
            raw[c] = pd.NA
    first = raw["first_name"].fillna("").astype(str).str.strip()
    last = raw["last_name"].fillna("").astype(str).str.strip()
    pos = raw["position"].astype(str).str.strip().str.upper().replace(
        {"D": "DST", "DEF": "DST", "PK": "K"})
    return pd.DataFrame({
        "player_name": (first + " " + last).str.strip(),
        "position": pos,
        "team": _team(raw["team"]),
        "opponent": _team(raw["opp"].fillna("")),
        "salary": pd.to_numeric(raw["salary"], errors="coerce"),
        "dff_proj": pd.to_numeric(raw["ppg_projection"], errors="coerce"),
        "dff_own": pd.to_numeric(raw["ownership_projection"], errors="coerce"),
        "injury_status": raw["injury_status"],
        "game_date": raw["game_date"].astype("string").str.strip(),
        "implied_team_score": pd.to_numeric(raw["implied_team_score"],
                                            errors="coerce"),
        "over_under": pd.to_numeric(raw["over_under"], errors="coerce"),
        "spread": pd.to_numeric(raw["spread"], errors="coerce"),
    })


def _n_games(df: pd.DataFrame) -> int:
    pairs = {frozenset((t, o)) if o else frozenset((t,))
             for t, o in zip(df["team"], df["opponent"].fillna(""))}
    return max(len(pairs), 1)


def usable(df: pd.DataFrame) -> tuple[bool, str]:
    """(ok, why not). Refuses an unposted or incomplete export."""
    if df.empty:
        return False, "the export is empty"
    projected = int((df["dff_proj"].fillna(0) > 0).sum())
    if projected == 0:
        return False, (f"every projection in this export is 0.0 -- it was "
                       f"taken before DFF posted the slate. Download it again "
                       f"closer to kickoff")
    games = _n_games(df)
    if projected < MIN_PROJECTED_PER_GAME * games:
        return False, (f"incomplete export: only {projected} projected "
                       f"players across {games} game(s)")
    return True, ""


def load(path, force: bool = False) -> pd.DataFrame:
    df = read(path)
    if not force:
        ok, why = usable(df)
        if not ok:
            raise ValueError(why)
    return df


def has_ownership(df: pd.DataFrame) -> bool:
    return bool("dff_own" in df.columns and df["dff_own"].notna().any())


# --- the join ------------------------------------------------------------------

def _unique_join(left: pd.DataFrame, right: pd.DataFrame, keys: list[str]
                 ) -> pd.DataFrame:
    """Pairs of (left index, right index) whose keys are unique on BOTH
    sides. A key shared by two players matches neither (rule 1)."""
    lk = left[keys].copy()
    lk["_li"] = left.index
    rk = right[keys].copy()
    rk["_ri"] = right.index
    lk = lk[~lk.duplicated(keys, keep=False)]
    rk = rk[~rk.duplicated(keys, keep=False)]
    lk = lk.dropna(subset=keys)
    rk = rk.dropna(subset=keys)
    return lk.merge(rk, on=keys, how="inner")[["_li", "_ri"]]


def _initial_key(name: pd.Series) -> pd.Series:
    """'Ja'Marr Chase' and 'J Chase' -> 'jchase'."""
    parts = name.fillna("").astype(str).str.split()
    first = parts.str[0].fillna("")
    last = parts.apply(lambda p: " ".join(p[1:]) if len(p) > 1 else "")
    return _key_name(first).str[:1] + _key_name(last)


def _salary_pass(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """The price key, which must be unique on the DraftKings side because that
    is the identity being assigned. Where DFF lists two players in one cell
    (one of them not priced on this slate), the name decides -- full, then
    initial and surname -- and only if exactly one of them fits."""
    keys = ["team", "position", "salary"]
    rk = right.dropna(subset=keys)
    rk = rk[~rk.duplicated(keys, keep=False)]
    pairs = []
    for li, row in left.dropna(subset=keys).iterrows():
        hit = rk[(rk["team"] == row["team"]) & (rk["position"] == row["position"])
                 & (rk["salary"] == row["salary"])]
        if len(hit) != 1:
            continue
        ri = hit.index[0]
        rivals = left[(left["team"] == row["team"])
                      & (left["position"] == row["position"])
                      & (left["salary"] == row["salary"])]
        if len(rivals) > 1:
            for col in ("_key", "_init"):
                fits = rivals[rivals[col] == right.at[ri, col]]
                if len(fits) == 1:
                    break
            else:
                continue
            if fits.index[0] != li:
                continue
        pairs.append((li, ri))
    return pd.DataFrame(pairs, columns=["_li", "_ri"])


def match(theirs: pd.DataFrame, slate: pd.DataFrame) -> pd.DataFrame:
    """One row per DFF player: matched, player_dk_id, player_name_dk,
    matched_on (salary / name / name across prices), on_slate."""
    d = theirs.reset_index(drop=True).copy()
    s = slate.reset_index(drop=True).copy()
    s["team"] = _team(s["team"])
    s["position"] = s["position"].astype(str).str.upper()
    s["salary"] = pd.to_numeric(s["salary"], errors="coerce")
    for frame in (d, s):
        frame["_key"] = _key_name(frame["player_name"])
        frame["_init"] = _initial_key(frame["player_name"])
    d = d.assign(matched=False, player_dk_id=pd.NA, player_name_dk=pd.NA,
                 matched_on=pd.NA)

    passes = (
        (None, "salary"),
        # Inside a cell the salary key could not separate.
        (["team", "position", "salary", "_key"], "name"),
        # DFF publish Classic prices; a Showdown slate shares none of them.
        (["team", "position", "_key"], "name across prices"),
    )
    taken: set = set()
    for keys, tag in passes:
        left = d[~d["matched"]]
        right = s[~s.index.isin(taken)]
        if left.empty or right.empty:
            break
        pairs = (_salary_pass(left, right) if keys is None
                 else _unique_join(left, right, keys))
        for li, ri in zip(pairs["_li"], pairs["_ri"]):
            d.at[li, "matched"] = True
            d.at[li, "player_dk_id"] = s.at[ri, "player_dk_id"]
            d.at[li, "player_name_dk"] = s.at[ri, "player_name"]
            d.at[li, "matched_on"] = tag
            taken.add(ri)

    d["on_slate"] = d["team"].isin(set(s["team"]))
    d["matched"] = d["matched"].astype(bool)
    return d.drop(columns=["_key", "_init"])


def match_rate(m: pd.DataFrame) -> float:
    """Share of on-slate DFF players that matched."""
    on = m[m["on_slate"]] if "on_slate" in m.columns else m
    return float(on["matched"].mean()) if len(on) else 0.0


# --- files ---------------------------------------------------------------------

def slate_prefix(dgid: int) -> str:
    return f"dg{int(dgid)}__"


def _strip_prefix(name: str) -> str:
    while name.startswith("dg") and "__" in name:
        head, rest = name.split("__", 1)
        if not head[2:].isdigit():
            break
        name = rest
    return name


def available() -> list[Path]:
    if not DIR.exists():
        return []
    return sorted(p for p in DIR.glob("*.csv") if p.is_file())


def _is_bound(p: Path) -> bool:
    return _strip_prefix(p.name) != p.name


def save(name: str, raw: bytes, dgid: int | None = None) -> tuple[Path, dict]:
    """Validate an upload and store it. Unparseable files never land; an
    unposted export does, because it is real and will be refused at read
    time, where that belongs."""
    base = Path(name).name
    if not base.lower().endswith(".csv"):
        raise ValueError(f"{base} is not a .csv file")
    final = _strip_prefix(base)
    if dgid is not None:
        final = slate_prefix(dgid) + final
    DIR.mkdir(parents=True, exist_ok=True)
    part = DIR / (final + ".part")
    part.write_bytes(raw)
    try:
        df = read(part)
    except Exception:
        part.unlink(missing_ok=True)
        raise
    dest = DIR / final
    part.replace(dest)
    dates = sorted({str(x) for x in df["game_date"].dropna() if str(x)})
    info = {
        "rows": int(len(df)),
        "projected": int((df["dff_proj"].fillna(0) > 0).sum()),
        "dates": dates,
        "teams": int(df["team"].nunique()),
        "ownership": has_ownership(df),
    }
    return dest, info


def for_date(date: str) -> tuple[pd.DataFrame, str]:
    """Rows for games on one (Eastern) date, from any week file."""
    files = [p for p in available() if not _is_bound(p)]
    if not files:
        return pd.DataFrame(), f"no DFF export in {DIR}"
    why = ""
    for p in sorted(files, key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            df = read(p)
        except ValueError as exc:
            why = str(exc)
            continue
        rows = df[df["game_date"] == str(date)]
        if rows.empty:
            continue
        ok, why = usable(rows)
        if ok:
            return rows.reset_index(drop=True), ""
        return pd.DataFrame(), why
    return pd.DataFrame(), why or f"no rows for {date} in {len(files)} export(s)"


def _for_teams_in(files: list[Path], teams: set[str],
                  min_coverage: float = MIN_TEAM_COVERAGE
                  ) -> tuple[pd.DataFrame, str]:
    why = ""
    for p in sorted(files, key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            df = read(p)
        except ValueError as exc:
            why = str(exc)
            continue
        have = set(df["team"]) & teams
        if not have:
            continue
        if len(have) / len(teams) < min_coverage:
            why = (f"{p.name} covers {len(have)} of this slate's {len(teams)} "
                   "teams -- it is for a different slate")
            continue
        rows = df[df["team"].isin(teams)]
        ok, reason = usable(rows)
        if ok:
            return rows.reset_index(drop=True), ""
        why = reason
    return pd.DataFrame(), why


def for_teams(teams) -> tuple[pd.DataFrame, str]:
    """Rows for a slate, found by its teams rather than a date."""
    want = set(_team(pd.Series(list(teams))))
    files = [p for p in available() if not _is_bound(p)]
    if not files:
        return pd.DataFrame(), f"no DFF export in {DIR}"
    df, why = _for_teams_in(files, want)
    if df.empty and not why:
        why = "no export covers this slate's teams"
    return df, why


def for_slate(dgid: int, teams) -> tuple[pd.DataFrame, str, str]:
    """(rows, why not, which file). A file bound to this slate wins; any week
    file covering its teams is the fallback."""
    want = set(_team(pd.Series(list(teams))))
    bound = [p for p in available() if p.name.startswith(slate_prefix(dgid))]
    bound_why = ""
    if bound:
        newest = max(bound, key=lambda x: x.stat().st_mtime)
        # Bound by the user to THIS slate, so no coverage test: the coverage
        # test exists to stop an unrelated week file being applied, and a
        # Sunday sheet bound to a Thu-Mon slate is partial on purpose. The
        # caption reports how many players it actually reached.
        df, bound_why = _for_teams_in([newest], want, min_coverage=0.0)
        if not df.empty:
            return df, "", newest.name
        bound_why = f"{newest.name}: {bound_why}" if bound_why else ""
    df, why = for_teams(teams)
    if df.empty and bound_why:
        why = bound_why
    return df, why, ("a week file" if not df.empty else "")

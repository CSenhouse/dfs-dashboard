# Data Sources — verified 2026-09-13

All findings below were probed live, not assumed. Re-verify before relying on any of it.

## Verdict: free data is sufficient to build the whole thing.

| Need | Source | Coverage | Status |
|---|---|---|---|
| DK salaries, 2022–present | DraftKings public API | 2023-02 confirmed, likely earlier | VERIFIED |
| DK salaries, 2011–2021 | RotoGuru `fyday.pl` | 2011–2021 only | VERIFIED |
| Box-score stats | nflverse `stats_player` | 1999–present | VERIFIED (releases listed) |
| Vegas lines (spread/total/ML) | nflverse `games.csv` | 1999–2026 incl. future schedule | VERIFIED |
| Weather, roof, surface, rest | nflverse `games.csv` | same file | VERIFIED |
| Ownership % | none free | — | GAP (paid only) |
| Contest results / payouts | none free | — | GAP (paid only) |

---

## 1. DraftKings public API — primary salary source

Unauthenticated. No key. Serves **historical** draft groups, not just live ones.

List live slates:
    https://www.draftkings.com/lobby/getcontests?sport=NFL
      -> .DraftGroups[] : DraftGroupId, StartDateEst, ContestTypeId

Pull a slate's players + salaries:
    https://api.draftkings.com/draftgroups/v1/draftgroups/{id}/draftables?format=json

Fields: draftableId, playerDkId, displayName, position, teamAbbreviation,
salary, status, newsStatus, rosterSlotId, competition{startTime}, draftStatAttributes

### Gotchas found by probing
- **Players repeat once per roster slot.** DG 153069 returned 1299 rows for
  ~650 players (RB appears as RB and as FLEX). **Dedupe on `playerDkId`.**
- **`salary=0` means a snake/best-ball draft group, not missing data.**
  Confirmed: DG 118000 (2024-12-01, NFL) and 138000 both salary-free drafts.
  Filter to salary-cap groups by requiring salary > 0.
- **IDs are shared across all sports.** ID alone tells you nothing; filter by
  position set (QB/DST present => NFL classic) and `competition.startTime`.
- **ContestTypeId 21 = classic, 96 = showdown/single-game.** Keep separate;
  showdown has different scoring (captain 1.5x pts and 1.5x salary).
- Some IDs 404 outright (110000, 126000). Expect holes; don't treat as failure.

### Observed ID landmarks (for backfill range planning)
    60000  -> 2015-ish NBA
    82000  -> 2023-02-11
    86000  -> 2023-04-21
    112000 -> 2024-08-25  (NFL, salaries present)
    132000 -> 2025-08-12  (NFL, salaries present)
    153069 -> 2026-09-13  (NFL main slate, current)

Backfill = sweep ~80000..153000, keep NFL salary-cap groups. ~70k requests.
Rate-limit politely, cache every response to disk, make it resumable.
Run it ONCE; the cache is the asset.

## 2. RotoGuru — 2011-2021 only

    http://rotoguru1.com/cgi-bin/fyday.pl?week={W}&year={Y}&game=dk&scsv=1

**HTTP only — no TLS.** Port 443 refuses connections; must use http://.

Returns semicolon-delimited inside <pre>:
    Week;Year;GID;Name;Pos;Team;h/a;Oppt;DK points;DK salary

Verified row counts: 2019 wk3 = 420, 2021 wk3 = 439.
**2022 onward returns 0 rows** (wk1/8/15 all empty). Site's own dropdown
confirms history ends at 2021. Do not plan around this for recent seasons.
No playoff-week data by the site's own note.

## 3. nflverse — stats and Vegas lines

Releases: https://github.com/nflverse/nflverse-data/releases
  stats_player, stats_team, weekly_rosters (to 2002), ftn_charting,
  espn_data, schedules, players, teams, trades
Contains **no** DFS/salary data — that's what the DK API is for.

Games/lines file (7550 rows, 1999 through full 2026 schedule):
    https://github.com/nflverse/nfldata/raw/master/data/games.csv

Vegas columns present: total, total_line, spread_line, away_moneyline,
home_moneyline, away_spread_odds, home_spread_odds, under_odds, over_odds
Context columns: roof, surface, temp, wind, away_rest, home_rest, div_game,
away_qb_id, home_qb_id, referee, stadium_id

=> **No paid odds subscription needed.** The single strongest feature set
   (implied team total, spread) is free and goes back to 1999.

## 4. Dead ends — checked, do not revisit
- `espechtsoftware/nfl-predictions` — a search result claimed it ships
  "dk_salaries_histo 2022-2025". It does not. Repo tree has only ingest code
  (`sql/raw/002_dk_salaries.sql`, `src/nfl_dfs/ingest/dk_client.py`); the data
  lives in the author's BigQuery, not in git. No committed dataset.
- dailyfantasyfuel.com — no CSV endpoint. `/api/projections/csv/...` and
  `/download/` both 404. HTML page renders but is current-slate only.

## 5. Scoring — recompute, never scrape
Derive DK points from nflverse box scores so any season can be backfilled and
so the salary source and the points source are independently verifiable.
Classic: 0.04/pass yd, 4 pass TD, -1 INT, 0.1 rush+rec yd, 6 TD, 1 rec (full
PPR), -1 fumble lost, +3 at 300 pass yd / 100 rush yd / 100 rec yd. DST tiered.
Validate the engine against RotoGuru's 2019-2021 "DK points" column — that
overlap is a free correctness test. Use it.

---

# ADDENDUM — verified 2026-09-13 (probe results)

## DK API history is far deeper than expected

Probed the id space directly. Earliest resolving draft group: **id 10000 =
2016-06-09**. Ids below ~9000 return nothing.

    10000 -> 2016-06-09      62000 -> 2022-01-12
    16000 -> 2017-10-18      72000 -> 2022-07-25
    22000 -> 2018-10-06      80000 -> 2023-01-01
    30000 -> 2019-09-30     118053 -> 2024-12-08 (NFL main slate, 649 players)
    40000 -> 2020-09-23     132000 -> 2025-08-12
    56000 -> 2021-09-12     153250 -> 2026-09-08

=> **~10 NFL seasons of DK salaries are free.** 2019-2021 overlaps RotoGuru,
giving a second independent cross-check (DK salary vs RotoGuru salary).

Holes exist (52000, 74000 return nothing). Expected; not a failure.

## Historical NFL *main* slates confirmed
DG 118053: 649 players, 13 games, 2024-12-08 — a real Week 14 Sun-Mon main
slate with full pricing. Historical backtesting is viable on real slate
composition, not a reconstruction.

Nearby 3-game mini-slates (117982/117986/117990/117994) show DK runs many
concurrent slates per week. The catalog must keep them distinct: a player's
salary differs between the main slate and a mini-slate.

## Scoring engine validation — PASSED
`scripts/validate_scoring.py` vs RotoGuru 2019-2021:

    matched 15,946 player-weeks, 99.574% exact agreement

Residual disagreements (68), all understood:
- ~45 are +/-0.10 or +/-0.04 => a 1-yard or 1-pass-yard difference. nflverse
  carries corrected official stats; DK scored live off the wire. Unfixable,
  harmless.
- 5 are +6.00 on return specialists (Jamal Agnew, Matthew Slater, Ashton
  Dulin, Olamide Zaccheaus, Brandon Zylstra). We credit the player a return
  TD; RotoGuru does not. DK's published rules DO credit return TDs to the
  individual player, so we believe our engine is right here. 0.03% of rows.
- Remainder are single stat-correction differences.

## Name joins are unsafe — use an ID crosswalk
A name-only join silently paired the 2019 Saints WR **Michael Thomas** with a
same-named defensive back, producing 0.00 points for a 44-point week. Same for
**Michael Carter** (Jets RB vs DB) in 2021. Adding position to the join key
fixed it (99.405% -> 99.574%), but that is a patch, not a solution.

**The pipeline must join DK `playerDkId` -> nflverse `player_id` through an
explicit crosswalk.** Never match on name alone.

---

# ADDENDUM -- verified 2026-10-12

## `api.draftkings.com` draftables now returns 403

Every variant (with and without `?format=json`, browser user agent, Origin
and Referer headers) returns 403, on live and historical draft groups alike.
The lobby (`www.draftkings.com/lobby/getcontests`) still answers, given a
browser user agent.

The older lobby endpoint works for both live and historical draft groups:

    https://www.draftkings.com/lineup/getavailableplayers?draftGroupId={id}
      -> playerList[] : pdkid (= playerDkId), pid, fn, ln, pn (position),
         s (salary), i (status: IR / O / Q / D / ""), tid, htid/atid,
         htabbr/atabbr, tsid (competition id), dgst (start, ms)
      -> teamList{tsid: {ht, at, tz}} : each game and its kickoff

Verified on 154488 (live MNF Showdown, 49 players), 153069 (711), 118053
(649) and 56000 (70). One row per player -- no per-slot duplicates -- and a
Showdown lists the FLEX price only. Captain is exactly 1.5x (rule 10), so it
is computed. `sources/dk_api.py` uses this endpoint first and falls back to a
cached `draftables` payload if one exists.

Be gentle with it. It is the last salary source this project has.

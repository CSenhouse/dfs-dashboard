# CLAUDE.md

NFL DraftKings DFS: projections, backtesting, lineup optimization.
Target mix: ~60% double-ups (cash), ~40% GPP.

## Run things

    .venv\Scripts\streamlit.exe run app.py          # dashboard
    PYTHONPATH=src .venv/Scripts/python.exe -m pytest
    PYTHONPATH=src .venv/Scripts/python.exe scripts/evaluate.py
    PYTHONPATH=src .venv/Scripts/python.exe scripts/compare.py MODEL_A MODEL_B
    PYTHONPATH=src .venv/Scripts/python.exe scripts/ablate.py --group pbp

    # distributional layer: quantile grid -> correlations -> simulation
    PYTHONPATH=src .venv/Scripts/python.exe scripts/build_simulation_inputs.py
    PYTHONPATH=src .venv/Scripts/python.exe scripts/correlations.py
    PYTHONPATH=src .venv/Scripts/python.exe scripts/validate_simulation.py

    # real contests: the only ground truth here. Export from the gamecenter
    # while logged in, THE SAME WEEK -- they go stale in one to two weeks.
    PYTHONPATH=src .venv/Scripts/python.exe scripts/ingest_contest.py \
        data/contests/contest-standings-196281323.csv --paid 17
    PYTHONPATH=src .venv/Scripts/python.exe scripts/vs_field.py 154084 \
        196281323 196281475    # our lineups dropped into a counted field
    PYTHONPATH=src .venv/Scripts/python.exe scripts/convergence.py
    PYTHONPATH=src .venv/Scripts/python.exe scripts/leverage.py
    PYTHONPATH=src .venv/Scripts/python.exe scripts/crowd_benchmark.py

    # hand edits to a projection: are they right more often than not?
    PYTHONPATH=src .venv/Scripts/python.exe scripts/override_value.py

    # ours vs a professional service. Download DFF's CSV, drop it in
    # data/projections/, run this on a FINISHED slate.
    PYTHONPATH=src .venv/Scripts/python.exe scripts/benchmark.py 154078         data/projections/DFF_NFL_cheatsheet_2026-10-04.csv --contest 196169740

`src` must be on PYTHONPATH; there is no installed package.

## Layout

    src/dfs/scoring.py     DK scoring, validated 99.57% vs DK's own points
    src/dfs/sources/       dk_api (salaries), nflverse (stats, lines, pbp), rotoguru
    src/dfs/sources/dff.py Daily Fantasy Fuel's CSV export: a second
                           opinion, and projected ownership. IMPORTED
    src/dfs/crosswalk.py   DK player -> nflverse gsis_id
    src/dfs/slates.py      assemble a slate: salaries + actual scores
    src/dfs/features.py    59 features, pregame/lagged separation enforced
    src/dfs/model.py       two-stage P(play) x points|play, LightGBM
    src/dfs/optimizer.py   MILP: optimize() classic, optimize_showdown()
                           single-game (1 CPT + 5 FLEX, captain 1.5x)
    src/dfs/backtest.py    walk-forward, accuracy, calibration, lineup results
    src/dfs/simulate.py    correlated slate simulation -> a lineup's whole
                           distribution, not a sum of point estimates
    src/dfs/correlations.py  GENERATED measured copula table
    src/dfs/field.py       simulated field -> estimated cash line
    src/dfs/contest_ev.py  payout tables, rank in a REAL field, and the
                           guard against pricing a lineup off its tail
    src/dfs/overrides.py   hand edits to a projection, logged as an experiment
    data/cache/            gzipped raw HTTP responses. THE ASSET. Never delete.

Read `docs/MODELING.md` before changing the model and `docs/DATA_SOURCES.md`
before touching a data source. Both record findings that are expensive to
rediscover.

## Rules that exist because something broke

1. **Never join on player names.** A name-only join paired the 2019 Saints WR
   Michael Thomas with a same-named defensive back and scored his 44-point
   week as 0.00. Use `crosswalk`, which resolves ambiguity to NEITHER rather
   than guessing.

2. **Never compare two separate evaluation runs.** The lineup metric has a ~25
   point weekly standard deviation. It has given a misleading answer three
   times, including flipping sign between a one-season and a two-season run.
   Use `scripts/compare.py` or `scripts/ablate.py`, which test pairwise on
   identical weeks. LightGBM is seeded for this reason -- do not remove the
   seeds.

3. **Select models on accuracy, not lineup scores.** Spearman has never given
   a wrong answer here; lineup means have. Validate on lineups only with many
   seasons.

4. **Shift before you aggregate.** Anything derived from a box score is
   post-game. `features._roll` shifts first. Vegas lines, injury reports and
   depth charts are genuinely pregame and may use the current week.

5. **A missing score is not always a zero.** nflverse emits no row for a
   player who recorded no stat. For lineup scoring that is 0; for training it
   may be a missing observation. `backtest.prepare` keeps both facts.

6. **Validate slates against the real schedule.** DraftKings runs simulated
   "Madden Stream" contests with real teams, real players and a real DST that
   are indistinguishable from football inside the payload. Also filter college
   football (no DK college DST) and snake drafts (they store a draft rank in
   the salary field; real salaries are multiples of $100).

7. **Never train on an unfinished week.** `slates.week_is_complete` gates it.
   A live week has no actual scores, and prepare() would turn every one into a
   real zero.

8. **Filter DK's own status before building a lineup.** DraftKings prices
   players on injured reserve, usually at the minimum salary, which makes
   them look like a free roster spot to an optimizer -- an IR player was
   rostered on a live showdown slate for exactly this reason. Use
   `dk_api.playable()`. IR/OUT/SUSP are removed; Q and D stay, since they
   often play and the projection handles them. LIVE SLATES ONLY: on a
   historical draft group the status field reflects whatever DK held when
   the payload was cached, not what was knowable before kickoff, so it must
   never become a backtest feature.

9. **Showdown has its own model and its own training set.** It is a separate
   pricing regime -- roughly 45 priced players instead of 800 -- so salary
   rank must be computed inside the slate (`features.build(...,
   salary_within_slate=True)`) and the model trains on
   `showdown_features.parquet`, not Classic rows. Built by
   `scripts/build_showdown_dataset.py`: 1,338 slates, 71,498 rows, spearman
   0.810 against a 0.712 baseline, better in 82/82 weeks.

   Two filters matter there. DK runs second-half and fourth-quarter contests
   on the same game with identical players AND identical salaries, so keep
   only the earliest draft group per game -- scoring a 2H slate with
   full-game results is badly wrong. The rest of the rejections are Madden
   Stream, preseason and playoff slates.

   Kickers are Showdown-only and are now projected properly (6.16 against an
   actual 6.49, rank correlation 0.53). The scoring engine pays field goals
   by distance (3 / 4 / 5 by range, 1 per PAT, no penalty for a miss); it
   scored every kicker 0.00 before that, including a 26-point Chris Boswell
   week.

10. **Showdown is a different game, not a Classic variant.** 1 Captain + 5
   FLEX, no position slots, both teams required. DK prices every player twice
   on those slates -- Captain at exactly 1.5x and FLEX at base -- so
   `draftables()` keeps the CHEAPEST row per player and exposes
   `captain_salary`. Keying on the lowest roster_slot_id instead picks
   Captain, which silently prices the whole slate at 1.5x.

11. **A lagged feature needs a row for the week it is predicting.** Every
   rolling team, defense and red-zone feature is keyed on (season, week, id)
   and built from games that have been PLAYED. In a backtest the week being
   predicted is already in the box scores, so the merge lands. On a live slate
   it never is, so the row does not exist and the feature merges as NaN for
   every player -- while training had it ~94% of the time. LightGBM reads
   missingness as signal, so a live slate looked like the few odd training
   rows with no history: P(plays) fell to 0.04 for the best receivers in
   football while the conditional points model beside it stayed healthy.
   Week 3 projected a third of week 2, and it reads exactly like early-season
   compression.

   `features._with_upcoming` appends a statistics-free placeholder row for the
   week being built, which `_roll` (shift then aggregate) fills from the weeks
   before it. Any new lagged builder must do the same. The pooled null rate
   hides this -- rz_carries was 100% missing live against 66% in training,
   a gap of 0.34 that passed a 0.5 filter, while for rostered receivers the
   real gap was 0.84. Measure missingness on the players you actually roster.

12. **nflverse rewrites the current season's files at a fixed URL.** Cached
   forever, a live slate is projected on a snapshot taken mid-week-1, and it
   degrades every week as more real games go unseen. `http.get(max_age=...)`
   expires the in-progress season only; completed seasons are final and stay
   cached permanently. A failed refresh serves the stale copy rather than
   destroying it.

13. **An override must move the whole distribution, not the column on
   screen.** `model.two_stage_all` returns p_play, a grid of conditional
   quantiles and three views read off it, and different parts of the dashboard
   consume different ones -- the optimizer takes proj_*, the simulator takes
   the grid. A control that scaled only the displayed column would leave the
   Lineup value tab quietly disagreeing with the projection it was shown,
   which is the same class of error as a train/serve skew and just as hard to
   see. `overrides.apply` moves every points-like column by one factor and
   leaves p_play alone: an override says what a player is worth, not whether
   he suits up. Use a lock for that.

14. **Never rank a simulated lineup against a frozen field and call it
   expected value.** A standings export is a fixed list of scores whose
   maximum is one week's realisation. A simulated lineup has a tail that runs
   past it, and the ranking then awards first place every time it does. On a
   real $20K contest (7,927 entries, $2,000 for first) this returned $2.18 per
   $1 against an $0.84 average -- a claimed 160% edge for a lineup whose cash
   rate was BELOW the field's own. 62% of that EV came from scores beyond the
   field maximum.

   The flaw is treating our score and the field's as independent when they
   come from the same twenty-two games: a lineup posts 220 because the slate
   went wild, and the field went with it. Top-heavy payouts turn that into
   most of the answer. `contest_ev.expected_value` returns `tail_share` and
   `trustworthy` and must be checked, never unwrapped blindly. Percentile and
   cash probability do not depend on the tail and were correct in the same
   run that got EV wrong -- prefer them. The null test that catches this is in
   `tests/test_contest_ev.py`: a lineup with no edge must price at the rake.

## Conventions

- Line endings are LF (`.gitattributes`). Normalize before string-patching a
  file, or anchors will not match.
- **Never put a backslash in a shell heredoc that patches a file.** The heredoc
  collapses `\\b` to `\b`, Python reads that as an escape, and a regex word
  boundary becomes a literal backspace (0x08) that matches nothing. It happened
  three times in one session and the last one was silent -- the suffix stripper
  in `dff._key_name` simply stopped stripping. Build such strings with
  `chr(92)`, or edit the file directly.
- Read/write files as UTF-8 explicitly; the Windows default is cp1252.
- Use the venv interpreter by path. Bare `python` is the Microsoft Store stub
  and does nothing.
- Streamlit hot-reloads `app.py` but NOT imported modules under `src/`. After
  changing one, restart the server or the running app keeps the old module and
  raises AttributeError on anything new.
- **`st.file_uploader` hands the same file back on every rerun.** Doing work
  whenever it is non-empty and then calling `st.rerun()` is an infinite loop:
  the screen flickers grey and never settles. Key the work on what was
  uploaded -- a name-and-size signature in `st.session_state` -- and do it
  once. The same applies to any widget whose value survives a rerun.
- Every HTTP response is cached gzipped. Re-running a sweep is cheap; deleting
  the cache is not.

## Rebuild, 2026-10-12 -- read this first

`src/dfs/`, `data/` (cache, contests, curated, ownership, projections) and the
git history were lost moving PCs. `src/dfs/` was rewritten from the tests,
scripts, app.py and these docs; all tests pass and
`scripts/validate_scoring.py` reproduces 99.582% agreement with the same 68
disagreements as the original. Retrained 2026-10-14 on 2021 through 2026
week 4, paired against the baseline on identical weeks:

                         rebuilt              lost model
    classic spearman     0.756 (92/92 wk)     0.798 (82/82)
      baseline           0.611                0.702
      lineup / of best   129.1 / 46.6%        128.4 / 46%
      top-decile gap     +2.8%                +3.9%
    showdown spearman    0.774 (106/106 wk)   0.810 (82/82)
      of best            67.2%                67.8%
      kickers            6.30 vs 6.51, 0.53   6.16 vs 6.49, 0.53

The spearman gap is NOT settled as a weaker model: the baseline is 0.09
lower too while lineup outcomes and calibration match or beat the original,
which points at the rebuilt baseline and spearman population differing from
the lost ones as much as at the features. Re-measure before concluding;
every other Status figure below is still from the lost model.

- **2022-23 IDP variants and two NHL Showdowns contaminated the first build.**
  `dk_api.classify` now requires standard NFL positions (and "D" is not
  DST); an IDP Classic slate has more players than the real main slate, so
  "most players" picked it. Pinned in tests/test_slate_validation.py.
- **`spread.py` and `correlations.py` are regenerated** (240 cells; QB/WR1
  +0.265, DST/opp QB -0.274). Top decile covers 77.0% -> 80.5% held out.
  `simulate.pit` must stay strictly inside (0, 1): one infinite normal
  score made the first regenerated table come out with two cells.
- **DraftKings salaries come from `lineup/getavailableplayers`.**
  `api.draftkings.com/.../draftables` returns 403 to this IP. The legacy
  endpoint serves historical draft groups too (verified back to id 56000),
  one row per player at FLEX price; captain salary is computed as 1.5x.
  `scripts/backfill_salaries.py` is sequential at ~1.4 req/s, stops after 8
  consecutive refusals and holds off idle sleep. The 2021-2026 sweep (10,518
  NFL slates) drew no refusals at that pace.
- **`features.py` is a re-implementation**: 59 features against the original
  87, same classes (lagged / pregame), same `_roll` semantics pinned by
  tests/test_leakage.py.
- Lost and unrecoverable: the twelve contest standings exports and the DFF
  files. `data/contest_fields.csv` and `observed_cash_lines.csv` survived.
- Python 3.12 is required (3.15 has no duckdb wheel). Remote:
  github.com/CSenhouse/dfs-dashboard (private).

## Status

Five seasons (2021-2025), 82 walk-forward weeks. Two-stage model at spearman
0.798, lineups averaging 128.4 points, +15.3 over baseline (p < 0.0001).

Against a simulated field that only samples players we win 79-94%. Against
opponents who also OPTIMIZE, that collapses: opponents holding our own
projection plus 15% noise -- strictly worse information -- still beat us 45%
of the time, against a 50% break-even. Most of the apparent edge was the
solver, not the projections. See docs/MODELING.md section 8.

Showdown has its own model: spearman 0.810 vs 0.712 baseline over 82 weeks
and 1,254 slates, lineups at 67.8% of the best available against Classic's 46%.

**That 67.8% is not evidence Showdown is where the model is strongest, and it
has been misread that way.** Percent-of-optimal measures how small the pool is.
A RANDOM legal lineup reaches 19.7% of optimal in Showdown and 12.8% in
Classic, so against that floor the ratios are 3.44x and 3.59x -- Classic is
marginally ahead. Edge over baseline agrees: Classic +0.095 spearman
(0.797 vs 0.702), Showdown +0.098. Indistinguishable.

**The Showdown field does converge harder, and that is now measured.** Two
double-up exports from one game (PIT at CLE, 2026 week 4, dg 154084, 39 entries
each, 17 paid) against the four Classic double-ups, on overlap -- the share of
your roster that a random other entry also holds, which is comparable across
formats where "top-10 ownership" is not:

    classic  tournament   0.141     relative spread 20.6%
    classic  double-up    0.380     relative spread 16.6%
    showdown double-up    0.630     relative spread  7.7%

Like for like against Classic double-ups, a Showdown field holds 63% of your
roster against 38% and is 2.2x tighter. Ownership marginals account for all of
it: overlap computed as sum(p^2)/S matches the real lineups to within 0.01 in
all nine contests, so a model of ownership would be enough and no model of the
field's lineups is needed.

The sharp consequence is that a third of the field sits on ONE score. The
most-submitted lineup was 33% and 38% of these two contests; in 196281323 it
finished 18th with 17 paid, so thirteen entries missed by 0.34 together. Being
in the chalk block turns a small Showdown double-up into one coin flip shared
by thirteen people.

Our builds in the counted field (`scripts/vs_field.py`): the cash build scored
93.86 and the neutral build 95.34, both clearing both lines; the ceiling build
scored 84.28 and missed both. Best possible in hindsight was 120.74. But salary
alone ranked players as well as our projection did (+0.61/+0.64 against
+0.60/+0.70) on 16 and 19 rostered players.

**Treat all of it as one observation.** 24 of 39 entrants appear in both
contests and 22 of those submitted the identical lineup in both, so the two
fields are near-copies on the same game. Nothing above is replicated. See
docs/MODELING.md 23.

`simulate.py` values a lineup as a distribution rather than a sum. Per-player
shapes are well calibrated (PIT KS 0.016); measured within-game correlation is
football-sensible (QB/own WR1 +0.23, DST/opposing QB -0.26) but widens a lineup
by only 1-2%. The real find was that pooled calibration hid an overconfident
top decile -- 74.3% coverage where the optimizer actually rosters -- which a
per-decile widening takes to 79.4% on held-out seasons.

It does NOT pick better lineups: ranking candidates by simulated probability
went 18-16 against the optimizer's own pick over 36 held-out weeks (p=0.75).
The value is measurement, not selection. See docs/MODELING.md section 11.

**The first observed cash line says this is not yet profitable.** A real
double-up cashed at 147 (data/observed_cash_lines.csv, the only ground truth
about contest outcomes in this repo). Against a line averaging 147 and moving
week to week, our cash lineup cleared it in 38.9% of 36 held-out weeks --
about -$0.22 per $1. Break-even needs the line to average about 133.

The payout structure itself was wrong here too, read off the public contest
API: a DK double-up pays exactly 2.00x to the top 43.5%, not 1.8x to the top
half. Break-even is therefore 50% of entries cashing, not 55.6%, and the bar
is the 56.5th percentile of the field rather than its median -- more generous
payout, harder threshold. The simulated field medians 111 against an observed
147, which is why every earlier cash-rate figure here looked fine. Record a
line after every contest played; one observation is not a measurement.

**Real contest standings are readable, and contest choice is worth 20 points.**
A gamecenter export (logged in, `scripts/ingest_contest.py`) carries every
lineup, every score and exact ownership. Five contests on ONE week 3 slate
(dg 153771, 688 players, same kickoff), all counted:

     entries        type    field mean   cash line
      17,835   tournament       127.2       143.0
       7,927   tournament       130.4       147.1
       4,597    double-up       147.3       158.9
       3,448    double-up       150.2       163.3
       2,298    double-up       152.8       163.3

**Week 4 reversed the sign, and the mechanism was being misread.** Five more
contests on one slate: the double-ups converged exactly as before (overlap
0.34-0.41 against 0.13, top-10 filling 5.2-5.8 of nine slots) and their field
means came in BELOW the tournaments -- a premium of -7.2 against +17.6 and
+21.3.

    week   double-up premium   chalk scored (ownership-weighted top 10)
       2              +17.62                                     18.13
       3              +21.32                                     20.04
       4               -7.20                                     11.72

A converged field scores whatever its chalk scores. Convergence is LEVERAGE,
not strength -- Parker Washington was 71% owned in those double-ups and scored
2.0. "Field strength rises as contests shrink" was two draws read as a trend.
What survives is the structural half: double-up fields are converged, in all
three weeks. A double-up is not reliably harder, it is reliably more correlated
with the chalk. See docs/MODELING.md 27.

Our week 3 GPP lineup scored 147.0: it CASHED the 17,835-entry tournament,
missed the 7,927 one by 0.1, and sat in the bottom third of every double-up.
Identical nine players. The cash build scored 133.4 and cleared nothing.

The 60/40 cash-heavy split predates any of this measurement, and every
measurement since says cash is the hardest field available. See
docs/MODELING.md 16-20.

Exports appear to expire after a week or two -- pull them the same week.
The DraftKings contest API now returns 403 to this project after it was
probed too hard; contests are identified by entry-cap fingerprint instead.

**First benchmark, and the first non-null in six attempts.** Ownership is the
crowd's aggregated pre-lock projection, so it ranks against outcomes like any
other. On the two slates with both our projections and real ownership:

                        week 2    week 3
    our projection      +0.635    +0.648
    crowd ownership     +0.600    +0.570
    salary alone        +0.507    +0.538

We rank players better than the field does -- the first evidence the
projections are not simply mediocre. But ownership keeps +0.29 and +0.18 of
rank correlation with actual points AFTER our projection is removed from both
sides, significant on both slates. That is the first measured signal the model
does not already have, following five straight nulls (red-zone usage,
calibration, stale form, game total, defensive matchup) -- now six, with the
extrapolation premium below. Using it needs
ownership BEFORE lock, which is a paid feed or a model of its own.

Fading chalk is not supported: ownership correlates +0.71 with points in
double-ups and +0.31 in tournaments, and top-1% entries were MORE chalky than
their field in five of six contests. That does not refute leverage -- within
one realised contest payout follows from score alone, so the claim is untestable
there -- but chalk is not a tax. See docs/MODELING.md 21.

**The bring-back works, measured on counted entries rather than simulated.**
76,877 real lineups across twelve exports, classified by what the QB is
connected to (`scripts/bring_back.py`, names matched 100%):

    construction   entries   mean z   top 1%
    none            16,890  +0.1264    0.52%
    stack           30,666  -0.0306    0.98%
    bring-back       4,126  -0.1113    0.56%
    both            25,195  -0.0147    1.46%

Unstacked entries score the HIGHEST on average and reach the top 1% least
often. Stacking costs mean and buys tail, seen in outcomes for the first time
here rather than asserted from a copula. A bring-back with no stack attached is
the worst build on the board -- it is an amplifier, not a thing.

Comparing only deliberate stackers, which holds the "stackers are just better
players" objection roughly constant: both vs stack is +0.016 z and a 1.51x
top-1% rate (p<0.0001). The falsification test sharpened with week 4 added --
in a tournament the bring-back gives 2.05x the top-1% rate and a higher cash
rate (28.4% vs 24.5%); in a double-up it gives 0.61x (p=0.003, now
significantly WORSE) and costs twelve points of cash rate. Better exactly
where the mechanism predicts, worse exactly where it predicts. Selection
cannot produce that pattern.

Per week, on tournaments: week 2 0.61x (p=0.054, n=7,921, the wrong way),
week 3 2.84x, week 4 2.23x (both p<0.0001). **Week 4 is the stress test** --
it was the chalk-bust week where the double-up premium went negative, and the
bring-back still worked, so the benefit is not just "the chalk did well".
Wired into the dashboard as a slider, GPP default 1. See docs/MODELING.md 25.

**The dashboard can be overruled now, and the overruling is measured.** The
Adjust projections tab edits any player's projection; `overrides.py` moves his
whole distribution by one factor and writes the edit to `data/overrides/` with
the model's own number beside it. That log exists because lineups are the wrong
unit to learn from -- a 25-point weekly sd needs ~70 weeks to settle a 5-point
edge -- while one edit against one actual score is a binomial that reaches
p<0.05 at about 60 edits. `scripts/override_value.py` asks whether the edits
moved toward what happened. Nothing recorded yet. See docs/MODELING.md 24.

**The extrapolation premium is a null.** Model projection minus recent
three-game average, used to argue a player was over-projected: across 242
rostered players with a projection, a recent average and an actual score,
corr(premium, model error) = +0.060 (p=0.35), no monotonic pattern by quartile,
and shading projections toward recent form makes ranking WORSE (0.555 against
0.602 for the model alone; recent form alone is 0.486). The lineup built on it
did win -- 148.02 against 119.12, top 11.6% of a 7,927-entry tournament --
because Tee Higgins beat his projection by +17.0 and Hockenson by +19.0, the
two biggest model errors on the slate, neither flagged by the column. A single
lineup cannot separate a good process from a lucky one. See docs/MODELING.md 28.

**A second opinion is importable, and it carries projected ownership.** Daily
Fantasy Fuel publish a "Download Projections as CSV"; drop it in
`data/projections/` and `sources/dff.py` reads it -- IMPORTED, never scraped,
because their projections are what they sell and this project already lost the
DraftKings contest API by probing too hard. The export's
`ownership_projection` column is the paid feed section 21 said was needed.

Upload it in the sidebar, or drop it in the folder. **`ownership_projection`
is empty in all 410 rows of a fully posted export -- it is premium-gated**, so
the signal section 21 wanted is still not available on the free tier.

**DFF publish CLASSIC salaries, so the salary key breaks on every Showdown
slate** -- Dak Prescott is $6,700 in the export and $10,400 single-game, and a
real Showdown slate matched 2 of 30 while the toggle appeared to do nothing.
The join has a third pass on the name within (team, position), which rule 1
still permits because both already agree. 28/28 after; Classic unchanged at
305/305. Better still, uploads are BOUND to the slate they were uploaded
against (`dg<dgid>__name.csv`) and `for_slate` prefers them, because DFF's
per-slate download carries that slate's own salaries -- one Classic upload
still serves the week. See docs/MODELING.md 32-33.

Five traps, all found by real files. An export taken before DFF post
the slate has every projection at 0.0 -- not worse numbers but the claim that
nobody will score, which an optimizer answers with the nine cheapest names --
so `dff.load` refuses it. (team, position, salary) is unique for only 41% of a
629-player slate because the minimum salary collects six Arizona receivers at
once, so the join falls back to the name INSIDE a cell, where rule 1's failure
mode cannot occur. ONE export covers the whole week, and a slate's start_time
is UTC while DFF write the Eastern date, so the lookup is keyed on the slate's
TEAMS -- `dff.for_teams` -- not its date or the filename. And the raw match
rate counts players from games the slate does not have: `match_rate` restricts
to on-slate teams, which turns 82.9% into 96.7%.

The Model / Blend / DFF control routes through `overrides.apply`, so it cannot
desync the optimizer from the simulator. Verified live: 410 of 771 matched,
100% of on-slate players, cash floor 89.3 -> 91.8. Agreement with our model is
spearman 0.944 across 295 players. See docs/MODELING.md 30-31.

Known gaps: ownership only from exports, participation uses a 4-week proxy
rather than the 90-minute inactive report, the cash line is simulated rather
than observed, stacked GPP lineups simulate too wide (87.3% coverage), and we
have no benchmark telling us whether our projections are good or mediocre
against professional ones. Showdown has two contests from one game and no
tournament at all, so the ceiling build is untested there. The training set
still lacks the `_with_upcoming` placeholder rows the live path now builds --
features.parquet needs rebuilding, paired per rule 2.

Windows note: `cd D:\dfs-dashboard` from another drive does nothing in
cmd.exe. Use `cd /d`.

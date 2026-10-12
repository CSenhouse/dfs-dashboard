# DFS Dashboard

NFL DraftKings projections, backtesting and lineup optimization.
Target mix: ~60% double-ups (cash), ~40% GPP.

## Status
- [x] Data source recon — see `docs/DATA_SOURCES.md`
- [x] DK scoring engine, validated to 99.57% vs RotoGuru 2019-2021
- [x] DraftKings API client (salaries, live + historical to 2016)
- [x] nflverse ingestion (box scores, Vegas lines)
- [x] Player ID crosswalk (99.85% coverage, fails safe)
- [x] Lineup optimizer (MILP, constraint-verified)
- [x] Walk-forward backtest harness + baselines
- [x] Feature layer (47 features, pregame/lagged separation enforced)
- [x] LightGBM two-stage projection model
- [~] Salary backfill (2021-2022 done, 2023-2026 sweeping)
- [ ] Ownership model (GPP)
- [ ] Correlated simulation (GPP)
- [ ] Dashboard

## Baseline to beat (2021, 14 walk-forward weeks)

| model | spearman | lineup actual | % of ceiling |
|---|---|---|---|
| salary only | 0.596 | 85.6 | 30.5% |
| recent 4wk mean | 0.687 | 124.8 | 44.7% |
| salary x P(play) | 0.671 | 108.4 | 38.9% |
| **recent x P(play)** | **0.692** | **128.4** | **46.0%** |

Two things these numbers establish:

1. **Participation dominates.** 54% of priced players on a main slate never
   record a stat line. Modelling P(play) moved the salary baseline from 85.6
   to 108.4 without touching the points model at all. The inactive report
   (90 minutes pre-kickoff) should replace the 4-week proxy.
2. **The optimizer over-projects by ~11 points.** Lineups project ~139 and
   score ~128. This is winner's curse: the optimizer selects precisely the
   players whose projections are too high, so error compounds instead of
   cancelling. Calibration matters more than accuracy here.

## Model results (2021-2025, 82 walk-forward weeks)

| model | spearman | RMSE | lineup actual |
|---|---|---|---|
| baseline: recent x P(play) | 0.702 | 5.26 | 113.1 |
| **LGBM two stage** | **0.798** | **4.35** | **128.4** |
| LGBM two stage, floor q30 | 0.793 | 4.94 | 128.6 |

Paired weekly tests vs the baseline:

    spearman  +0.0954  better in 82/82 weeks  p = 2e-38
    lineup    +15.34   better in 57/82 weeks  p < 0.0001

Lineup distribution: mean 128.4, sd 25.2, se 2.8.
44% of weeks clear 130 points; 21% clear 150.

### Calibration: the error is concentrated where it hurts

Predicted vs actual by projection decile. The model is well calibrated
overall (bias +0.05) and essentially exact through nine deciles:

    decile 0-8   gap within +/-0.05 points
    decile 9     predicted 16.83, actual 16.17, gap +0.66  (+3.9%)

The top decile is the only badly calibrated bucket, and it is the only one
the optimizer ever rosters from. Average bias is the wrong lens for a system
that deliberately selects the tail.

**Fixing it did not help lineups.** `two_stage_calibrated` fits an isotonic
predicted->actual map on held-out weeks. Paired over the same 82 weeks
(seeded, so the comparison measures only the change):

    top-decile gap  +4.0%  ->  +1.7%
    overall bias    +0.048 ->  -0.014
    spearman        +0.01297  better in 79/82 weeks  p < 0.0001
    lineup          -0.60     better in 41/82 weeks  p = 0.81  (n.s.)

The correction does exactly what it was designed to do and buys zero lineup
points. Isotonic is monotonic, and the optimizer selects on
projection-per-dollar ordering, so shrinking the top of the distribution
roughly uniformly builds almost the same lineup.

The lesson: *the projected total being wrong* and *the lineup being wrong*
are different failures. Calibration fixes the first. It makes the estimate of
how you will do honest -- which matters for contest selection and bankroll --
without making the lineup better.

Adopted anyway: it strictly improves ranking, costs nothing, and an honest
projected score has independent value.

### Feature ablation (paired, same 82 weeks)

Play-by-play usage features (red zone, goal line, team volume):

    spearman  +0.00638  better in 75/82 weeks  p < 0.0001
    lineup    -2.77     better in 39/82 weeks  p = 0.37 (n.s.)

Kept, but honestly: the ranking gain is robust yet small -- about 6% of the
model's total edge over baseline -- and there is no demonstrated lineup
benefit. Red-zone features specifically did not reach the top 20 by gain;
the useful play-by-play signal is `pbp_air_yards` (target depth) and
`team_rz_plays`.

A naive cross-run comparison had suggested these features cost 4.7 lineup
points. The paired test says -2.77 +/- 3.06, indistinguishable from zero.
That is the third time in this project that comparing separate runs gave a
misleading answer.

### The methodological point

A single weekly lineup score has a standard deviation of **31 points**, so a
12-week mean carries about +/-9 points of standard error.

We ran this comparison twice. On one season the lineup metric said the
baseline was **better by 7.5 points**. On two seasons it says the model is
**better by 2.3**. The sign flipped, and neither result was significant.
Meanwhile spearman favoured the model in 12/12 and then 28/28 weeks.

**Select models on accuracy metrics; validate on lineup outcomes only with
many seasons.** Choosing a model on a season of lineup scores is choosing on
noise. This is why the harness was built before the model.

### Participation is solved; projection is not

Optimizer lineup composition over 28 weeks (LGBM two-stage):

    players who actually played   9.00 / 9
    players scoring exactly 0     0.21 / 9
    weeks with no bust            23 / 28
    salary used                   $49,946 / 50,000

The two-stage split did its job -- the optimizer no longer rosters players who
never take a snap, which was the baseline's main failure. The remaining gap to
the 280-point hindsight ceiling is genuine projection difficulty, not roster
construction.

## Is it profitable? Probably not against opponents who optimize

Break-even in a double-up is a **55.6%** win rate, not 50% -- the payout is
about 1.8x.

Simulating a field and reading the cash line off its median
(`scripts/cash_line.py`, 82 weeks) looked excellent:

| simulated field | cash line | our win rate |
|---|---|---|
| naive (salary-picking) | 88-99 | 89-94% |
| informed (recent form) | 90-106 | 85-94% |
| sharp (our own model) | 95-112 | 79-93% |

**That result does not survive a harder field.** Those opponents only *sampled*
players, while our entry solved a MILP. Against opponents who also optimize
(`scripts/sharp_field.py`, 22 weeks, 25 opponents each):

| opponents | beat rate | ROI |
|---|---|---|
| our projection, no noise | 0% (100% ties) | -- |
| **our projection + 15% noise** | **54.9%** | **-1.2%** |
| our projection + 30% noise | 62.5% | +12.6% |
| a different model, no noise | 63.6% | +14.5% |
| a different model + 15% noise | 72.4% | +30.3% |

The second row is the one that matters. Opponents holding a **degraded copy of
our own projection** -- strictly worse information than we have -- still beat
us 45% of the time, which is below break-even once rake is taken.

**So most of the apparent edge was the optimizer, not the projections.** Our
numbers are imprecise enough that perturbing them 15% and re-optimizing
produces lineups that are about as good. Against anyone who also runs an
optimizer, the edge is roughly break-even.

We beat a *weaker* model (63.6%) comfortably. Whether real main-slate
opponents are weaker than us is the open question, and it is not answerable
from inside this repo -- it needs a benchmark against professional
projections. That is the strongest argument for a paid subscription: not the
data, the yardstick.

Next gains should come from the actual inactive report in place of a 4-week
participation proxy, and from ownership data for the GPP side.

### Top features (conditional points model)

    salary, salary_pct, dk_points_r6, value, dk_points_r3,
    offense_pct_r6, offense_pct_r3, temp, implied_total,
    receiving_yards_r3, wind, air_yards_share_r6, opp_implied_total

Salary dominates, as expected -- DraftKings prices with better information
than we have. Snap share and Vegas implied totals follow, which is the shape
the design predicted.

## Running the dashboard

    .venv\Scripts\streamlit.exe run app.py

Opens at http://localhost:8501. Pick a live slate, choose a contest type, and
it projects every priced player and builds lineups.

- **Cash / double-up** optimizes a floor (30th percentile).
- **GPP / tournament** optimizes upside and stacks the QB with a pass catcher.
- **Neutral** uses the calibrated mean -- the same lineups, but an honest
  projected total for deciding which contest to enter.

The header flags **Real NFL games: NO** when a slate's matchups are absent
from the schedule, which is how simulated "Madden Stream" contests are caught.

First load trains on five seasons and takes a minute; afterwards it is cached.
Re-run after inactives land, 90 minutes before the first game.

## Setup
    .venv\Scripts\python.exe -m pip install -r requirements.txt

Run anything with `src` on the path:

    PYTHONPATH=src .venv/Scripts/python.exe scripts/validate_scoring.py

## Principles
1. Recompute DK points from box scores; never trust a scraped points column.
2. Cache every HTTP response. Backfills are expensive and sources disappear
   (RotoGuru stopped publishing after 2021).
3. Join on ids, never on names.
4. Validate with walk-forward splits only. Never random k-fold on time series.
5. Beat the salary baseline before believing any model.

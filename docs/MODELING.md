# Modeling notes

Rationale and hard-won findings. Current numbers live in the README; this file
explains *why* things are shaped the way they are, so a future change does not
quietly undo a lesson.

## 1. The problem has two parts, not one

54% of players priced on a main slate never record a stat line. A single
regression on observed points has to learn "will he play" and "how much will
he score" through one head, and mostly learns the first, because participation
dominates the variance.

    E[points] = P(plays) x E[points | plays]

The classifier gets the pregame signals that determine participation (injury
report, depth chart, recent appearances). The regressor trains only on players
who played, so it is not dragged toward zero by scratches.

This split is worth more than any feature. Modelling participation alone lifted
the crudest baseline from 85.6 to 108.4 lineup points before the points model
changed at all.

**Result:** optimizer lineups now average 9.00 of 9 players who actually
played, against a baseline that routinely rostered backups who never took a
snap.

## 2. A missing score is not a zero, and not a missing row either

nflverse emits no row for a player who recorded no stat. There are two very
different reasons for that:

- He was a healthy scratch. That is a **missing observation**.
- He dressed, played, and did nothing. He genuinely **scored 0**.

For *lineup scoring* both are 0, because that is what your lineup would have
scored. For *training* they are not the same, and collapsing them teaches the
model that expensive players routinely bust. `slates.assemble` flags
`no_stat_line`; `backtest.prepare` sets the scoring target and a `played`
indicator separately.

Filtering the pool to players who turned out to play is survivorship bias and
inflates every backtest. Do not do it.

## 3. Leakage discipline

Every feature is classified by when it becomes knowable:

| class | examples | rule |
|---|---|---|
| LAGGED | rolling volume, snap share, red-zone usage, form, opponent defense | shift BEFORE aggregating; week W sees only weeks < W |
| PREGAME | Vegas lines, injury report, depth chart, rest, venue | current week is legal |

`features._roll` shifts before it rolls. `backtest.walk_forward` trains
strictly on earlier weeks. Both are pinned by tests in `tests/test_leakage.py`
using a ramp series where week N scores exactly N, so an off-by-one is visible.

The depth chart is a live hazard: nflverse changed it in 2025 from one row per
player-week to dated snapshots. A snapshot taken after kickoff knows who
started. `features._depth` maps each snapshot to the next week that had not yet
kicked off.

## 4. Select models on accuracy, not on lineup scores

A single weekly lineup has a standard deviation of about 31 points. We ran the
same comparison three times:

| training data | lineup diff vs baseline | significant? | spearman |
|---|---|---|---|
| 1 season (12 wk) | **-7.45** | no (p = 0.53) | better 12/12 |
| 2 seasons (28 wk) | **+2.26** | no (p = 0.77) | better 28/28 |
| 4 seasons (65 wk) | **+15.39** | yes (p = 0.001) | better 65/65 |
| 5 seasons (82 wk) | **+15.34** | yes (p < 0.0001) | better 82/82 |

The lineup metric changed sign. Spearman never did. Judging the model on one
season of lineup scores would have shipped the worse model with confidence.

Use accuracy metrics to choose. Use lineup outcomes only as a final check, and
only with many seasons.

### Corollary: never difference two separate runs

The play-by-play feature group looked like a 4.7 point lineup loss when the
run that had it was compared against an earlier run that did not. Ablated
properly -- same weeks, same data, feature group dropped by substring -- the
effect was **-2.77 +/- 3.06**, indistinguishable from zero, while ranking
improved in 75 of 82 weeks.

Two runs differ in more than the thing you changed. `scripts/ablate.py` and
`scripts/compare.py` exist so that no conclusion in this project rests on an
unpaired comparison.

## 5. The optimizer amplifies error rather than averaging it

The optimizer does not sample players at random. It deliberately selects the
highest projections on the slate, so error in the top decile is concentrated,
not cancelled. This is the winner's curse, and it means:

- **Calibration matters more than average accuracy.** `backtest.calibration`
  reports predicted vs actual by projection decile; the top bucket is the one
  the optimizer actually rosters from.

  Measured over 82 weeks, the model's overall bias is +0.05 points and nine of
  ten deciles are calibrated to within +/-0.05. The tenth over-projects by
  **+0.66 points (+3.9%)**. That single bucket supplies every rostered player,
  so across nine spots it compounds to roughly six points of phantom lineup
  score. An aggregate bias number would have called this model unbiased.

  `model.two_stage_calibrated` fits an isotonic predicted->actual map on held
  out weeks -- the correction has to be learned out-of-sample, because
  in-sample predictions are optimistically accurate and would teach the wrong
  relationship.

  **It works, and it buys no lineup points.** Paired over 82 seeded weeks the
  top-decile gap fell from +4.0% to +1.7% and spearman improved in 79/82
  weeks, while the lineup difference was -0.60 in 41/82 weeks (p = 0.81):
  exactly neutral.

  The reason is that isotonic is monotonic and the optimizer selects on
  projection-per-dollar ordering. Shrinking the top of the distribution
  roughly uniformly leaves that ordering nearly intact, so it rosters almost
  the same players.

  **"The projected total is wrong" and "the lineup is wrong" are different
  failures.** The winner's curse is real and measurable, but it inflates your
  estimate of the outcome rather than degrading the outcome. That still
  matters -- for contest selection and bankroll sizing you need an honest
  number -- but it is not a route to a better lineup. Beware of fixing a real
  defect and assuming the benefit lands where you expected.
- Cash play should optimize a **floor** (a low quantile), not a mean.
  Double-ups pay for clearing a threshold, not for upside.
- GPP should optimize a **ceiling** and force correlation (stacking), because
  tournaments pay for the right tail.

## 6. Position had to be an explicit feature

It was missing for a long time and nothing caught it, because in-sample the
model could infer position from usage history. It surfaced only on a live Week
1 slate, where every rolling feature is NaN and the model returned
byte-identical projections for a QB and a WR on the same team at the same
salary.

Cold starts -- Week 1, rookies, players back from long absence -- are exactly
when the model has least information and most needs structural priors. Hence
also `prior_season`, which covers 64% of rows against 37.8% for rolling
features.

## 7. Things that silently contaminated the dataset

Each of these passed every check in place at the time, and none raised an
error. They are all pinned by tests now.

| contamination | why it passed | real invariant that catches it |
|---|---|---|
| College football slates | CFB shares QB/RB/WR/TE | DK never prices a college DST |
| Snake-draft groups | store draft RANK (1..N) in the salary field, so `salary > 0` holds | real salaries are multiples of $100 and exceed the player count |
| "Madden Stream" simulations | real teams, real players, real DST, names like "BUF @ LV" | those matchups are not in the real schedule |
| In-progress week | every actual score NaN, which `prepare` turns into a real 0 | every game in the week must have a final score |
| Same-named players | name join picked one arbitrarily (Michael Thomas, WR and DB, 2019) | a key mapping to >1 player resolves to NEITHER |

The pattern: **structural checks catch structural problems; only a real-world
invariant catches real-world contamination.** String matching on "Madden" was
not a fix, because the label lives in the lobby, not the slate.

## 8. Most of the measured edge was the optimizer, not the projections

Simulating a field of entrants and reading the cash line off its median made
this system look strongly profitable -- 79% to 94% win rates against every
field tried. That field only *sampled* players, while our entry solved a MILP.

Rebuilt with opponents who also optimize, the picture changes:

| opponents (22 weeks, 25 each) | we beat | ROI at 1.8x |
|---|---|---|
| our projection, no noise | 0% (all ties) | -- |
| **our projection + 15% noise** | **54.9%** | **-1.2%** |
| our projection + 30% noise | 62.5% | +12.6% |
| a different model (recent form) | 63.6% | +14.5% |

Opponents holding a *degraded copy of our own projection* beat us 45% of the
time. Break-even is 55.6%, so that is a losing position against opponents with
strictly worse information than ours.

The reason is that our projections are not precise enough for the optimum to
be stable. Perturb them 15% and re-optimize and you get a lineup about as
good, because the ranking that survives the perturbation is most of what we
actually know.

Two consequences:

1. **Beware of measuring yourself against a field that cannot do what you
   do.** The 79-94% numbers were real arithmetic on a field that did not
   optimize. The comparison flattered the projections by crediting them with
   the solver's contribution.
2. **A better optimizer is worth very little from here; better projections are
   worth everything.** We already extract close to the maximum from the
   numbers we have.

Whether this is profitable in practice turns on whether real opponents project
better than we do, which cannot be answered from inside this repo. It needs a
benchmark against professional projections -- the value of a paid subscription
here is the yardstick, not the data.

## 9. Absence is already modelled once -- do not charge for it twice

The diagnostics page surfaced the worst week in five seasons: 2024 week 11,
61 points scored of 297 available. The lineup was led by Justin Fields at a
22.1 projection, who scored 1.7. He had been benched behind Russell Wilson
since week 7.

The data knew. `depth_team` was 2 and `is_starter` was 0. But `dk_points_r3`
read 25.2 and `offense_pct_r3` read 0.99 -- numbers from weeks 4 to 6, when he
was the starter.

**The rolling windows rolled over rows that exist, not over calendar weeks.**
nflverse emits no row for a player who recorded no stat, so a three-week window
silently skipped weeks 7, 8 and 10 and reached back a month. Every benched,
injured or returning player was inflated the same way.

The obvious fix is to fill missed weeks with zero. It measured worse:

| variant | spearman | lineup |
|---|---|---|
| old (rolls over appearances) | 0.7974 | 128.51 |
| zeroed | 0.7989 | **123.90** |
| **staleness flag only** | **0.7990** | **130.50** |

Paired, 82 weeks: **staleness beats zeroed by 6.60 lineup points in 52 of 82
weeks, p = 0.017.** Both give the identical (and tiny) ranking gain.

The reason is structural. The model already has a separate P(play) stage, so
zeroing a player's form makes the *regressor* pay for absence a second time --
and the regressor's question is "how good is he IF he plays", for which his
form when he actually played is the better input.

What works instead: keep the form, add `weeks_since_played`, and let the value
go **NaN** once the window contains only absences. NaN is a different statement
from zero -- "no recent evidence" rather than "produces nothing" -- and
LightGBM reads missingness as its own signal. Zeroing destroyed that
distinction.

Worth noting what the whole episode was worth: a correctly diagnosed, real
defect, fixed properly, bought **+2.0 lineup points and +0.0016 spearman**
against the original. That is the third feature-engineering effort in a row to
land near zero, after red-zone usage and calibration.

## 10. Known gaps

- **No measured cash line.** Contest results and payout thresholds are paid
  data. "Did this lineup cash" is currently inferred, not observed.
- **Participation uses a 4-week proxy.** The real inactive report lands 90
  minutes before kickoff and is far sharper.
- **No ownership model**, so no GPP leverage and no field simulation.
- **Salaries move during the week.** A projection built on Wednesday pricing
  is stale by Sunday; re-pull before lock.

## 11. A lineup is a distribution, and the model was overconfident where
it mattered

`src/dfs/simulate.py` draws the whole slate a few thousand times instead of
adding up nine point projections. Three things were tested against
walk-forward predictions, each against its own absence on identical weeks.

**Per-player distributions are well shaped.** Mapping every observed score
through its own predicted distribution gives PIT deciles between 0.092 and
0.106 and KS 0.016 over 63,906 player-weeks. That number turned out to be
worthless on its own -- see below.

**Correlation is real and smaller than expected.** Measured as the
correlation of NORMAL SCORES, not raw points: the shared signal that makes two
players on a 30-point team both project high is already inside both
projections, and correlating raw scores would count it twice. The table is
football-sensible -- QB with his own WR1 **+0.23**, DST against the opposing
QB **-0.26**, two opposing QBs **+0.11** -- but its effect on a LINEUP is
small: it widens a cash lineup by 0.9% and a stacked GPP lineup by 2.4%. With
36 pairs in a nine-man roster and only one or two of them coupled, that is
just the arithmetic. Cross-game correlation is +0.005, negligible.

**The pooled calibration number hid a serious defect.** Split by projection
decile, the punt end covers 83% of results inside its own 80% interval and the
top decile covers 74.3%. The top decile is the only bucket an optimizer ever
rosters from, so the error that survives into a lineup is the one the average
concealed -- and it runs the dangerous way. Nine too-tight distributions made
a lineup look far safer than it was: 74.3% lineup-level coverage against a
target of 80%.

This is docs section 5's winner's curse one step further along. There the
optimizer concentrated error in the MEAN; here it concentrates it in the
SPREAD.

Widening by projection decile (`scripts/calibrate_spread.py`) takes cash
lineups from 74.3% to **79.4%** coverage on seasons the correction never saw.
Two honest caveats. The defect's SIZE is not stable -- the top decile covered
72.6% in 2021-23 and 76.7% in 2024-25, so a full-strength correction
overshoots to 83.4% and the fix is applied at half strength. And it costs
pooled KS, 0.0206 to 0.0242, because it deliberately mis-calibrates punt plays
that never enter a lineup in order to calibrate the ones that do.

**It does not pick better lineups.** Ranking twelve near-optimal candidates by
simulated probability against taking the optimizer's own top lineup, paired
over 36 held-out weeks: 18 better, 16 worse, p=0.75, and 1.3 percentage points
LOWER in field percentile on average. GPP is the same coin flip. The
candidates are near-identical by construction and the differences in their
simulated probabilities are below what the simulation can resolve.

So the value delivered is measurement, not selection: an honest range for a
lineup, useful for contest choice and bankroll sizing, with no evidence it
improves which lineup you enter. The dashboard says so on the tab.

**Known limitation.** Stacked GPP lineups come out too WIDE -- 87.3% coverage
against the 80% target, made worse by the spread fix (83.6% before it). The
correction is fitted per player on the marginal population and cannot tell
which lineup a player will end up in.


## 12. The first observed cash line

Everything about contest outcomes in this project was modelled until someone
reported a real one: **147 to cash a double-up, 2026 week 1**. It is a single
observation and it is worth more than the field simulator, because the
simulator had nothing to anchor it and is wrong in a measurable direction.

    simulated field median, main slate      111
    optimizing opponents, median            114
    first observed cash line                147

147 sits at the **92nd percentile** of our simulated field. The real median
entrant plays like our model's 92nd percentile, which is not a tuning error --
the sampler picks players by projection weight and repairs over-cap lineups by
swapping down to the cheapest option, and that drafts worse than a human with
free projections and a cap tool.

The consequence is the important part. Against a line that averages 147 and
moves week to week, our cash lineup cleared it in **41.7%** of 36 held-out
weeks. Break-even on a 1.8x double-up is 55.6%.

    assumed average line   cash rate   EV per $1
        125                  66.7%       +0.20
        130                  55.6%        0.00
        140                  44.4%       -0.20
        147                  41.7%       -0.25
        150                  30.6%       -0.45

Break-even needs the line to average about **130**. Both the observation (147)
and the outside estimate (~140) sit above it.

Three things could move this and none of them are yet measured:

- **One observation is not a measurement.** 147 may have been a high-scoring
  week. The file exists so this stops being a guess.
- **The backtest lineup is handicapped.** It uses a four-week participation
  proxy, not the inactive report that lands 90 minutes before kickoff. Live
  play knows who is out; the backtest does not, and that gap is exactly the
  kind that puts zeros in a lineup.
- **Only the level was corrected.** The week-to-week shape of the line still
  comes from the simulated field.

This supersedes the framing in section 8. That section concluded most of the
apparent edge was the solver rather than the projections; this says the bar
itself was in the wrong place, and the honest answer to "is this profitable"
is currently no.

## 13. Stacking: the mechanism is real, the outcome test has no power

`scripts/stacking.py` builds the same ceiling-optimised lineup five ways over
86 walk-forward weeks and scores each on what actually happened.

**The mechanism behaves exactly as the theory says.** Spread rises with the
strength of the stack, and the tail lift compounds far beyond it:

    structure              spread    P(>=180)   P(>=200)
    no stack                +0.8%      +1.6%      +6.0%
    QB + 1                  +2.3%      +6.7%      +6.9%
    QB + 2                  +3.9%     +16.5%     +29.4%
    QB + 1 + bring-back     +2.8%      +8.5%     +14.8%
    QB + 2 + bring-back     +4.6%     +21.8%     +28.8%

(each figure is correlated versus independent sampling of the same lineup.)
A few percent more spread is worth twenty to thirty percent more tail,
because the tail falls away so steeply out there. This is why the section 11
conclusion -- "correlation only widens a lineup 2.4%, it barely matters" --
is right for cash and wrong for tournaments. Coverage of the 80% interval is
a central measure; a tournament is decided four spreads out.

**The outcome test cannot confirm any of it, and never could.** No structure
beat no-stack on real points (best was QB+1 at +2.50, p=0.123), and NOTHING
reached 200 in 86 weeks for any structure. That is not evidence against
stacking. It is arithmetic:

    P(a 0.43% event happening at least once in 86 weeks)   31%
    expected hits in 86 weeks                              0.37
    weeks per arm to separate 0.33% from 0.43% at 80% power  63,000
                                                       = 3,706 seasons

Observing zero is the expected result. Any claim that this experiment
disproves stacking would be reading noise, and any claim it proves stacking
would be reading the simulator's own assumptions back out.

**What the data does say, and it is not comfortable.** Our GPP lineup averages
128.3 actual points with a 22.3 spread. A large-field winner needs about 200,
which is 3.2 spreads out. On a single entry we are not in the race, and that
is consistent with the cash finding in section 12 rather than separate from
it.

Two things follow.

- **The portfolio is the unit, not the lineup.** At 0.43% per entry, one entry
  reaches 200 with probability 0.4%; twenty genuinely different entries reach
  it 8.2% of the time, and 150 reach it 47%. Every single-lineup number in
  this project understates tournament play, and `max_exposure` plus
  `min_difference` are crude stand-ins for real portfolio construction.
- **The tail numbers are not yet trustworthy.** Section 11 measured stacked
  GPP lineups simulating too WIDE -- 87.3% coverage against an 80% target --
  so these tail probabilities are, if anything, overstated. That has to be
  fixed before the simulator is allowed to choose a tournament lineup.

Cost of stacking, for the record: about one expected point, not significant
either way. It is close to free in the mean and moves probability into the
tail. That is a reasonable basis to stack, and it rests on the simulator's
tail being right rather than on any measured outcome.

## 14. Correlation does not vary with the game total

A live week 3 slate put Kirk Cousins at the top of the GPP build on a Las
Vegas team implied for 20.25 points, the lowest total among the plausible
quarterbacks. He was there on price: $4,800 made him the best ceiling per
dollar on the board, and the optimizer banked the saving on a $8,800 running
back. That lineup had the highest sum of ceilings, 202.6, and roughly half the
simulated chance of reaching 200 that the Purdy and Goff builds had.

The obvious explanation was that the simulator understates it, because the
copula table is keyed on (relationship, role, role) and nothing else -- a
QB and his WR1 get +0.23 whether the game totals 20 or 30. Stacking is
supposed to be worth more in a shootout, so conditioning on the game total
looked like a clear improvement.

It is not there. Splitting every within-game pair at the median game total of
44.5 and comparing:

    pair          low total     high total    diff    p
    QB-WR1         +0.253        +0.275      +0.022  0.55
    QB-WR2         +0.229        +0.284      +0.055  0.13
    QB-TE1         +0.193        +0.244      +0.051  0.17
    DST-opp QB     -0.304        -0.299      +0.005  0.89
    QB-opp QB      +0.125        +0.148      +0.023  0.67

Every difference runs the expected way and not one is significant, with about
1,300 pairs per cell. Five tests, so 0.010 would be needed after correction
and the best on offer is 0.13. Adding the dimension would halve the data per
cell to buy nothing measurable, so it was not adopted.

That is the fourth feature-shaped idea in this project to land at zero, after
red-zone usage, calibration and the stale-form fix.

It also relocates the defect. The simulator was not wrong about the Cousins
stack; the OBJECTIVE is. A sum of nine 85th percentiles is a central
statistic and a tournament pays on a tail one, so buying the cheapest
acceptable ceiling and spending the saving elsewhere maximises the first while
damaging the second. Whether ranking on the simulated tail instead actually
scores better is a separate question, measured in scripts/gpp_ranking.py.


## 15. Ranking tournament lineups on the simulated tail does not help

Section 14 relocated the Cousins defect from the simulator to the objective:
a sum of nine 85th percentiles is a central statistic and a tournament pays on
a tail one. The obvious repair is to rank candidates on P(reaching a winning
score) instead. Measured in `scripts/gpp_ranking.py` over 86 walk-forward
weeks, it does not work.

The candidates were built deliberately to give the ranking something to say.
The earlier cash experiment in section 11 ranked twelve lineups that all came
out of one objective and differed by two players, which is close to no choice
at all. This builds the best stacked lineup for EACH of the ten
best-projected quarterbacks, so the set spans game environments and
reproduces the actual Cousins-versus-Purdy decision.

    by sum of ceilings   131.44 pts   73.6th percentile of the real field
    by simulated tail    129.43 pts   72.2nd
    difference            -2.00 pts   -1.4%

    66 of 86 weeks disagreed: better in 30, worse in 36
    paired t p=0.329, sign test p=0.539

One pattern runs the way the theory predicts -- the tail ranking reached 180
in 2 weeks against 1, while reaching 160 in 5 against 11 -- which is exactly
the trade a tail objective should make, buying extreme outcomes with middling
ones. At two events against one it is not evidence of anything.

Three separate nulls now sit on this question. Correlation does not vary with
the game total (section 14). Ranking on the tail does not improve outcomes.
And the model has no unexploited signal in the game environment at all:
regressing walk-forward residuals on implied team total gives correlations
between -0.012 and +0.016, every p above 0.25, with quartile means inside
±0.36 points. Vegas information is already priced.

So the honest answer to "should the optimizer avoid a cheap quarterback in a
20-point game" is that no measurement available here says yes. The cap
arbitrage is real and the explanation is sound; the correction is not
demonstrable.

What remains unfixed, and is the likeliest reason the tail ranking fails, is
that the GPP tail is miscalibrated -- stacked lineups cover 87.3% of outcomes
inside their own 80% band (section 11). Ranking on a distorted number cannot
be expected to work, and that calibration should land before this question is
asked again.


## 16. Does the field belong to the contest? One week says yes, one says no

Four thresholds, each confirmed against DraftKings' public contest API rather
than recalled, and arranged so both weeks have two contests on ONE draft group:

    wk1  dg 151307   $5 Double Up   top 43.50%   145.0
    wk1  dg 151307   $40K Pylon     top 25.21%   160.7
    wk2  dg 153428   $5 Double Up   top 43.50%   144.2
    wk2  dg 153428   $20K Pylon     top 25.37%   134.6

Paying fewer places must ask for more, so on one field the tournament line has
to sit above the double-up line. Week 1 obeys that and is consistent with a
single field of mean 139.9 and sd 31.1. Week 2 does not: the tournament paying
the top quarter cashed 9.6 points BELOW the double-up paying the top 43.5% on
identical players, which no single distribution can produce -- solving for it
returns a negative standard deviation.

Read as two fields at a common sd of 27.2, the double-up premium is

    week 1    -2.0 points   (no premium; the tournament field looks equal)
    week 2   +23.2 points

One week with a large effect and one with none. The mechanism is plausible --
a double-up draws grinders and a $3 tournament draws everyone -- and week 2 is
genuinely impossible for a single field, so something real is going on there.
But the SIZE is not established and is not stable, and a prior version of this
section reported the 23 points as a finding on the strength of week 2 alone.
It is one observation.

Worth keeping in mind that the inversion could also be a reading artefact: a
gamecenter shows several numbers and the last paying position is not the most
prominent of them. Two more weeks would separate "the fields differ" from "one
number was misread", and that is cheap to collect.

What survives regardless: every cash probability in this project is computed
against a single simulated field per slate, and `field.sample_lineups` has no
idea which contest is being entered. Whether or not the premium is 23 points,
the model has no way to express a difference that week 2 shows can exist.

One process note. Four attributions in this exchange were wrong before the API
settled them -- a slate size, a week, a contest, and a line recalled as 147
that was 145. The numbers were never the problem; remembering which contest
produced them was. Rows in `data/observed_cash_lines.csv` now carry
`contest_id` and every one of them is verified.


## 17. The first real field, and what it settles

A standings export arrived for contest 195661381 -- the $20K Pylon on draft
group 153428, 7,927 entries. It carries every submitted lineup, every score
and the exact ownership of every player, which ends three separate guesses.

**The two-field reading of week 2 was right.** Section 16 inferred, from the
double-up and the tournament cashing at 144.2 and 134.6 on the same slate,
that the tournament field must be far weaker -- around mean 116.6 with sd 27.2
under a normal assumption. Measured:

    predicted   mean 116.6   sd 27.2
    actual      mean 116.7   sd 26.9

Section 16 then walked that finding back because week 1 showed no such gap.
The walk-back was too cautious: week 2's premium is now measured on one side
and is real. Week 1 remains unresolved and needs its own export. The double-up
field on that slate implies a mean near 139.8, so the two populations sat
about 23 points apart on identical players.

**The simulated field is wrong in both directions at once.**

    real        mean 116.7   sd 26.9
    simulated   mean  99.9   sd 21.3

Too low by 16.8 points and 1.26x too narrow. Earlier estimates of that spread
error -- 1.86x from mismatched slates, then 1.04x from a single-field fit --
were both artefacts of not having this file. Every cash probability this
project has printed rests on that sampler.

**Our lineups finished where the score said they would.** 90.2 and 93.1 in a
field averaging 116.7: ranks 6,592 and 6,320 of 7,927. Week 2 was not a near
miss.

**Ownership exists now.** Bijan Robinson 47.8%, Derrick Henry 37.9%, Dalton
Schultz and Aaron Jones 25.9%, CeeDee Lamb 22.1%. This is the input that
cannot be derived from any amount of projection accuracy, and the reason it
matters is visible in the same table: Bijan was on 47.8% of rosters and scored
11.1, while Jaxon Smith-Njigba scored 45.5 on a fraction of them. The winning
lineup paired Bryce Young with Jalen Coker and the Panthers defence.

What this does NOT settle is whether leverage helps us. That needs the same
export for contests across several weeks, and then a paired test of whether
ownership-aware lineup selection would have finished higher. The data now
exists to ask the question, which it did not before.

## 18. Two fields on one slate, both measured

Standings for a double-up and a tournament on the SAME draft group (153428,
2026 week 2, identical player pool and kickoff) settle the question section 16
raised and section 17 half-answered.

                          double-up     tournament     gap
    entries                   4,597          7,927
    entry fee                    $5             $3
    pays top                  43.5%          25.4%
    field mean                134.3          116.7    +17.6
    field median              137.7          115.1    +22.6
    p25                       114.5           96.9    +17.5
    p75                       154.8          135.0    +19.8

The double-up field is stronger at every percentile. This is not a modelling
artefact and needs no assumption: both distributions are counted, not fitted.
Section 16's inference of a ~23 point premium was directionally right and
somewhat too large; the measured figure is 17.6 on the mean.

The mechanism is visible in the ownership tables. Counting how many of the
nine roster slots the ten most-popular players occupy:

    double-up   4.38 slots
    tournament  2.61 slots

Grinders converge. Bijan Robinson was on 75.2% of double-up rosters and 47.8%
of tournament rosters; Dalton Schultz 64.2% against 25.9%. A double-up field
is a few hundred near-identical lineups plus noise, which is exactly why its
median sits 22.6 points higher on the same players.

**What it costs to break even.** Combining each contest's real field with its
real payout curve, the score needed for zero EV:

    GIANT $5 Double Up     142.4     binary: below is -100%, above is +100%
    $20K Pylon ($3)        134.6     graduated: 135 -> +67%, 165 -> +167%,
                                     180 -> +567%

The tournament asks 7.8 fewer points AND pays for exceeding the bar rather
than only clearing it. On this slate, playing the tournament instead of the
double-up with an identical lineup is straightforwardly better.

Two cautions before acting on it. This is one slate: field strength and
scoring environment both move week to week, and week 2 was an 18th-percentile
scoring week league-wide. And the tournament's variance is far higher -- most
entries return nothing, so the same edge arrives in a much lumpier form, which
matters for a bankroll even when the EV is identical.

Our own lineups scored 90.2 and 93.1 that week and cleared neither bar.

What this does change immediately: the 60/40 cash-heavy allocation was chosen
from general DFS advice, and the only measurement this project has ever made
of the two fields says the cash side is 17.6 points harder. Three more weeks
of exports would turn that from one observation into a policy.


## 19. The two-sided widening helped the players and not the lineups

Section 11's spread correction was symmetric, and section 15's diagnosis found
the lineup defect was LOCATION rather than width -- simulated lineup totals
came in about 9 points below the model's own projection for the same nine
players. The repair was to widen each side of the distribution separately,
since the top decile runs 0.89 too high at q30 and 2.09 too low at q85 and one
multiplier cannot fix both.

At player level it worked. On held-out seasons the top decile moved from 76.7%
to 80.0% coverage against an 80% target, and pooled KS came to 0.0224 against
the symmetric version's 0.0242 -- so unlike the symmetric fix it did not buy
the rostered slice at the expense of everything else.

At lineup level it did not.

                      before    after   target
    cash coverage      79.4%    79.4%    80.0%
    gpp coverage       87.3%    87.3%    80.0%
    cash median PIT    0.575    0.549    0.500
    gpp median PIT     0.569    0.544    0.500

The location bias closed by about a third and the coverage did not move at
all. That is what widening does to a shifted distribution: it stretches the
interval as fast as it fills the tails, so the fraction inside stays put.
Diagnosing the defect as location and then treating it with a width correction
was the wrong instrument, chosen because it was the one already built.

The GPP over-width therefore remains open, and with it the ranking question in
section 15: a tail-based objective was tested on a simulator whose stacked
lineups still cover 87.3% against an 80% target, so that null is not clean
either.

Section 17 changes what to do about it. The field is now measurable from
standings exports rather than modelled, and a simulator that can be checked
against a counted distribution is a different problem from one that can only
be checked against itself.

## 20. Five contests, one player pool: contest choice is worth 20 points

Week 3 standings for every contest played on draft group 153771. Same 688
players, same kickoff, five different fields, all counted rather than modelled.

     entries        type    field mean   sd    cash line
      17,835   tournament       127.2   24.4      143.0
       7,927   tournament       130.4   25.4      147.1
       4,597    double-up       147.3   24.2      158.9
       3,448    double-up       150.2   22.7      163.3
       2,298    double-up       152.8   21.4      163.3

Field strength rises monotonically as contests get smaller and sharper, and
the spread narrows as it rises -- a tight field of near-identical lineups. The
bar moves 20.3 points from the softest contest to the hardest, on an identical
player pool.

**The premium replicates.** Section 18 measured the double-up field 17.6
points above the tournament field on one slate and called it an observation
rather than a policy. Matched by entry cap on an independent slate:

    week 2   7,927 tournament 116.7   4,597 double-up 134.3   +17.6
    week 3   7,927 tournament 130.4   4,597 double-up 147.3   +16.9

Two slates, two weeks, same contests, same answer.

**What it did to a real lineup.** Our week 3 GPP build scored 147.0:

    tournament 17,835   line 143.0   rank  3,720 (20.9%)   CASHED
    tournament  7,927   line 147.1   rank  2,023 (25.5%)   missed by 0.1
    double-up   4,597   line 158.9   rank  2,670 (58.1%)   missed
    double-up   3,448   line 163.3   rank  2,196 (63.7%)   missed
    double-up   2,298   line 163.3   rank  1,590 (69.2%)   missed

Nine identical players, paid in one contest and in the bottom third of
another. The cash build scored 133.4 and cleared nothing.

Three weeks of real results now exist for the Classic cash lineup: 160.4
(cashed), 90.2, 133.4. One for three, and the two misses were not close.

**The implication for the 60/40 split.** That allocation came from general DFS
advice, before this project could measure anything. Every measurement it has
since made says the cash side is the hardest field on the board, by roughly
the margin that separates our lineups from the bar. Nothing here says we would
WIN in the softer contests -- one cash in three weeks is not an edge -- but it
does say the double-up pool is the worst available place to find out.

Caveats that still hold. Three weeks. Tournament payouts are top-heavy, so a
softer bar arrives with much lumpier returns. And the largest field here is
only inferred to pay 25.37%, matched from a sibling contest, because the
DraftKings contest API has been returning 403 since this project probed it too
hard -- the entry-cap fingerprint identified the contests instead.

## 21. Chalk is good, and the crowd is a free benchmark

Five standings exports carry exact ownership, which makes two questions
answerable that previously were not.

**Does fading the chalk pay?** `scripts/leverage.py` sums each entry's nine
players against the ownership table from the same contest -- arithmetic on
counted entries, no model anywhere -- and correlates that with what the entry
scored.

    double-ups    corr +0.71   top 1% ownership 3.64 against a field 3.42
    tournaments   corr +0.31   top 1% ownership 1.44 against a field 1.27

Ownership correlates POSITIVELY with points everywhere, and the entries that
finished top 1% were MORE chalky than the field they came from in five of the
six contests. Naive contrarianism would have hurt. The field identifies good
players correctly, which is why they became chalk.

That does not refute leverage, and the experiment cannot. Within one realised
contest, payout is a deterministic function of score -- rank follows from
points and nothing else -- so ownership cannot affect what an entry earned
given what it scored. The leverage claim is about the ex-ante spread of rank:
a contrarian lineup's score is less correlated with the field's, so it reaches
the top more often for the same mean. Testing that needs either many weeks of
the same strategy or a field simulator that works, and ours does not yet.
What is established is narrower and still useful: chalk is not a tax.

**Is our model better than the crowd?** Ownership is thousands of entrants'
aggregated pre-lock opinion, so it can be ranked against outcomes exactly as a
projection can. This is the professional benchmark the project has never had,
and it is free.

                        week 2    week 3
    our projection      +0.635    +0.648
    crowd ownership     +0.600    +0.570
    salary alone        +0.507    +0.538

Our model ranks players better than the field does on both slates. That is the
first evidence in this project that the projections are not simply mediocre,
and it arrives after section 8 suggested the opposite by showing opponents
with degraded copies of our own numbers still beating us 45% of the time.
Those are consistent: being better at ranking players is not the same as
winning contests.

**The crowd still knows something we do not.** Removing our projection from
both sides on ranks, ownership retains +0.29 and +0.18 of correlation with
actual points, significant on both slates.

That is the first non-null in this project after five consecutive ones --
red-zone usage, calibration, stale form, game total, defensive matchup. All
five tested ideas the model already knew. Ownership is the first measured
signal it does not have, which is unsurprising: it is the only input that
cannot be computed from a box score.

Using it is a separate problem. Ownership is known after lock, and a lineup
has to be built before. Capturing it needs either a paid projected-ownership
feed or a model of ownership itself, and whether either pays is untested. Two
slates is also two slates.


## 22. Percent-of-optimal measures the pool, not the model

The Showdown model reaches 67.8% of the best available lineup and the Classic
model 46%, and that pair of numbers was used in this project -- by me, in
advice -- to claim Showdown is where the projections are strongest. It does
not say that. It says a Showdown pool is small.

Measured with lineups that have no edge by construction:

    a RANDOM legal lineup reaches
        Showdown   19.7% of optimal   (40 slates)
        Classic    12.8% of optimal   (20 slates)

Against that floor the model multiplies by 3.44x in Showdown and 3.59x in
Classic. Normalised, Classic is marginally ahead, not behind. Six players
chosen from roughly 45 simply cannot land far from the optimum, and neither
can the field's.

Edge over baseline says the same thing more directly. Classic runs 0.7974
against a 0.7022 baseline, Showdown 0.810 against 0.712 -- +0.095 and +0.098.
There is no meaningful difference between the formats.

What remains true about Showdown is narrower: it has a model that beats its
own baseline in 82 of 82 weeks, and it is the only format that prices kickers.
What is NOT true is that we have more edge there.

And no Showdown contest has ever been observed in this project. Every cash
line and every ownership file is Classic. The format is unmeasured, which is
a different claim from better, and the structural guess runs the other way: a
45-player pool should concentrate the field harder than Classic's, where
double-up ownership already puts 5.2 to 6.0 of nine roster slots in ten
players.

The general lesson is the one this project keeps relearning in new costumes. A
ratio whose denominator moves between the things being compared is not a
comparison. Percent-of-optimal joins pooled KS, pooled null rate and pooled
residual mean on that list.

## 23. The first Showdown contests: the field converges, as predicted

Section 22 ended with a structural guess and said so: a 45-player pool should
concentrate the field harder than Classic's. Two exports from one game now
measure it. Both are double-ups on PIT at CLE, 2026 week 4 (draft group
154084, 51 priced players), 39 entries each, 17 paid.

The metric has to work across formats, so it cannot be "ownership of the top
ten", because top ten of 45 and top ten of 800 are not the same quantity. What
is comparable is overlap: of your S roster slots, how many does another
randomly chosen entry also hold, divided by S. It runs from S/pool to 1.0
whatever S is. `scripts/convergence.py`:

    format    structure     n   overlap   roster sd   relative spread
    classic   tournament    3     0.141       0.041            20.6%
    classic   double-up     4     0.380       0.141            16.6%
    showdown  double-up     2     0.630       0.110             7.7%

Payout structure, not entry count, is the split that matters -- paid/entries
near 0.435 is a double-up and near 0.25 a tournament -- because every Showdown
export here is a double-up and Classic double-ups already converge hard. On
that like-for-like comparison a Showdown field holds 63% of your roster
against 38%, and its scores are 2.2x tighter in relative spread. The reasoning
in section 22 was right.

**Ownership alone accounts for all of it.** Computing overlap two ways -- from
ownership marginals as sum(p^2)/S, and from the real submitted lineups -- the
two agree to within 0.01 in every one of the nine contests and both formats.
So the thirteen identical lineups in a 39-entry Showdown are not clustering on
top of high ownership; they are what those marginals predict. Ownership is a
sufficient statistic for overlap, which means a model of ownership would be
enough and no lineup-level model of the field is needed.

### A third of the field sits on one score

The consequence is sharper than the convergence itself:

    contest     line     tied ON the line    tied just under it
    196281323   90.02                   1    13 entries, 0.34 under   (33%)
    196281475   91.68                   5    15 entries, 2.00 under   (38%)

The most-submitted lineup -- the same six players in both contests -- was a
third of the field, and its score is a single point mass landing within two
points of the pay line. In 196281323 it finished 18th with 17 paid: thirteen
entries missed by 0.34 together. Classic double-ups do this too (408 and 478
entries tied at 163.30 in two of them), but there the block lands ON the line
and splits it rather than a point below it.

The actionable form is not "be contrarian for upside", which section 21 found
unsupported. It is that in a small Showdown double-up the chalk block is a
third of the field and its outcome is all-or-nothing, so being in it converts
the contest into one coin flip shared by thirteen people.

### What our own lineups would have done

`scripts/vs_field.py 154084 196281323 196281475` drops our builds into the
counted field. Scored from nflverse box scores and cross-checked against the
export's own FPTS column, which now agrees to 0.00 points across all 35
players after the captain bug below was fixed:

    build (view optimized)    scored    196281323       196281475
    cash     (proj_floor)      93.86   9th, CASHED    12th, CASHED
    neutral  (proj_mean)       95.34   9th, CASHED     8th, CASHED
    gpp      (proj_ceiling)    84.28   35th, missed   38th, missed
    best possible, hindsight  120.74

The cash and neutral builds cleared both lines; the ceiling build missed both
by 5.7 and 7.4. That is the expected direction -- a ceiling build is for a
tournament -- but there is no Showdown tournament here to test it against, so
it is consistent with the Classic finding rather than independent of it.

Ranking players on the same two contests:

                        196281323   196281475
    our projection         +0.598      +0.698
    crowd ownership        +0.239      +0.567
    salary alone           +0.609      +0.637

**Salary ranks players about as well as our model does here**, beating it on
one contest and losing narrowly on the other. On Classic slates with 273 to
281 rostered players our projection beat salary by +0.13 and +0.11 (section
21). This is 16 and 19 players, so the comparison is nearly powerless -- but it
is the only Showdown evidence that exists, and it does not show the model
adding much over DK's own pricing.

### These are one observation, not two

24 of the 39 entrants appear in both contests, and 22 of those 24 submitted the
identical lineup in both. The two fields are near-copies of each other on the
same game, so every figure above is n=1 for practical purposes. Nothing here
should be treated as replicated.

One number did move the other way and is worth recording as not-a-finding:
roster ownership correlates +0.10 with points in these Showdown contests
against +0.71 in Classic double-ups. The obvious explanation is range
restriction, and it does not hold -- per roster slot the Showdown fields spread
MORE than the Classic tournaments, which still manage +0.31. So the near-zero
figure is not an artefact, and with one effective observation it is not
evidence either.

### The captain bug this turned up

`contests.ownership` aggregated each player's points with `max` across the
slots he filled. On a Classic export every slot pays 1x and that is harmless.
On a Showdown export the Captain row carries 1.5x the player's points, because
that is what the Captain scored, so `max` returned the inflated figure as the
player's own: Aaron Rodgers read 35.94 against a real 23.96, and every captain
was 50% too high. `load()` now divides the multiplier back out into
`base_points` and `ownership` aggregates that. Two tests pin it.

It surfaced as a cross-check against DK's own FPTS disagreeing by up to 11.98
points, which is exactly 0.5x a captain's score -- the kind of error that is
invisible without an independent number to compare against.

A second, quieter bug: `scripts/leverage.py` globbed
`contest-standings-*.zip`, so a contest saved as bare CSV was skipped in
silence. Both Showdown exports arrived that way.

## 24. Hand edits, and why they are a better experiment than lineups

The dashboard had two player-level controls, a hard lock and a hard ban, which
is a sledgehammer when what you mean is "move him down four points". Section 21
is the reason that gap matters: the crowd holds +0.29 and +0.18 of rank
correlation with outcomes AFTER our projection is removed, so there is signal
the model does not have, and the person using the dashboard has sometimes seen
it -- that a receiver is in a bad spot, that a back is chalk this week. Until
now there was no way to say so in a number.

`overrides.py` is that control, and it is logged because of a power problem.

**Lineups are the wrong unit.** Whether the dashboard's lineups beat a
hand-built one cannot be settled by playing. Weekly lineup scores carry about
a 25-point standard deviation (rule 2); paired on one slate call it 15. To
detect a true 5-point edge at 80% power is roughly 70 weeks. A season is 18.
Every version of "play more and see" runs into this, which is why the honest
answer to "should I use the dashboard lineups yet" is that playing will not
answer it.

**An override is a much smaller unit.** One player, one number, against one
actual score. The primary test is direction -- of the edits made, how many
moved toward what happened? -- which costs one bit per edit, assumes no
distribution and is a plain binomial against 50%. If the true hit rate is 65%,
that reaches p<0.05 at about 60 edits. Twenty a week is three weeks.

`scripts/override_value.py` runs three tests in increasing order of what they
ask of the data: direction, then whether the edited number ranks the
overridden players better than the model's own did, then the partial rank
correlation that asks whether the edit carries signal after the model's number
is removed from both sides -- the same test that found ownership's +0.29.
Verified on constructed data where 75% of edits were right by construction:
direction recovered 30/40 at p=0.002.

Overridden players are a SELECTED sample by construction -- they are the ones
that looked wrong enough to touch -- so none of it says anything about players
left alone. The claim is narrow: when you decide to overrule the model, are you
right more often than not?

### An override moves the distribution, not a column

This is now rule 13 because the failure mode is silent. `two_stage_all` returns
p_play, a grid of conditional quantiles and three views read off it, and the
optimizer reads proj_* while the simulator reads the grid. Scaling only the
column on screen would have left the Lineup value tab disagreeing with the
projection it was shown.

So one factor moves every points-like column and p_play is left alone -- an
override says what a player is worth, not whether he suits up, which is what a
lock and `dk_api.playable` are for. Shape is preserved, so the floor-to-ceiling
spread survives. Measured on the live week 4 slate: Josh Allen's floor 17.38 ->
5.00 as entered, his q85 30.35 -> 8.73, the same 0.288 factor, p_play untouched
at 1.000.

Below a projection of 1.0 the factor becomes additive instead, because a ratio
against nothing is meaningless and a grid of zeros cannot be scaled into
anything. A player the model rates at nothing is better handled with a lock.
Ratios are clamped at 20x so that 95 typed where 9.5 was meant produces a
visibly clamped number rather than a nonsense lineup.

### Two things found by driving the real dashboard

Streamlit 1.63 renders a NaN in a `NumberColumn` as the word "None" in every
unedited cell, so the blank-means-no-override design read as a grid full of
"None". The column is pre-filled with the number in force instead, which is
also clearer: it always shows what the optimizer is using, and setting it back
to the model's value drops the override.

The first end-to-end test looked like a failure and was not. Josh Allen's floor
was cut from 17.38 to 5.00 and the lineup did not change -- because Allen was
never in it; Mahomes was the QB at $6,500. Checked properly, overriding Mahomes
down swapped the QB to Jared Goff and boosting Allen to 26.0 put him in. A
control that appears to do nothing is worth confirming against the actual
roster before assuming either way.

## 25. The bring-back works, and counted entries can say so

Section 13 measured stacking inside the simulator and was explicit that this
proves nothing: a simulator fed a correlation table reports what that table
implies. The outcome test available then -- our own lineup over 86 walk-forward
weeks -- needed something like 3,700 seasons to separate the tail rates that
matter, so it was not a test either.

Seven standings exports hold 48,579 lineups that real people submitted, each
with the score it posted. `scripts/bring_back.py` classifies every one by what
its quarterback is connected to and compares what they actually scored. No
projection, no correlation table and no simulation enters it. Names matched
100% in all seven contests.

    construction   entries   share   mean z   top 1%   top 10%
    none            12,452   25.6%   +0.1231   0.47%    12.32%
    stack           20,009   41.2%   -0.0271   1.05%     9.06%
    bring-back       2,743    5.6%   -0.2495   0.15%     4.37%
    both            13,375   27.5%   -0.0037   1.65%    11.94%

("stack" is a pass catcher from the QB's own team, "bring-back" a skill player
from the opposing team, "both" the full game stack. z is standardised within
each contest, because a week 3 double-up and a week 2 tournament are different
scales.)

**The mean/tail trade-off is exactly as advertised, and this is the first time
it has been seen in outcomes here.** Entries that did not stack at all scored
the HIGHEST on average and reached the top 1% least often -- 0.47% against
1.65% for the full game stack, a factor of 3.6. Stacking costs mean and buys
tail, measured on counted entries rather than asserted from a copula.

**A bring-back with no stack attached is the worst construction on the board.**
Lowest mean of the four and a 0.15% top-1% rate, a third of the unstacked rate.
That is sensible: without the QB-side correlation it is not a game stack, just
two players from one game. The opponent piece is an amplifier, not a thing.

### Controlling for the obvious objection

Stackers are not randomly assigned -- someone who stacks has read about
stacking, and that travels with everything else they know. But "stack" and
"both" are both groups of deliberate stackers, so comparing them holds most of
that constant and what remains is the bring-back itself:

    both vs stack   mean +0.0234 z (p=0.043)   top-1% rate 1.58x (p<0.0001)

### The falsification test, which it passes

If the bring-back works by buying tail, it must help where the tail is paid and
not where it is not. A double-up pays a flat amount for clearing a bar; who
else cleared it is irrelevant, so there the extra variance should be pure cost.

    structure     n        mean       top 1%          cash rate
    tournament   33,646   +0.0689 z   1.97x (p<0.0001)   28.70% vs 24.55%
    double-up    14,933   -0.3219 z   0.92x (p=0.83)     24.96% vs 36.17%

In a tournament the bring-back nearly doubles the top-1% rate AND raises the
cash rate. In a double-up it does nothing to the tail -- 0.92x at p=0.83, a
flat null -- and costs eleven percentage points of cash rate.

**This is why the result survives the selection objection.** If bring-back
entrants were simply better players, they would be better in both structures.
They are better in exactly the structure where the mechanism predicts a benefit
and worse in exactly the structure where it predicts a cost. That is a
mechanism signature, not a selection signature.

The residual confound worth naming: the same person does not necessarily enter
both structures, so it is possible that bringing back is a sophisticated move
in a tournament and an unsophisticated one in a double-up, with different
populations doing it. That requires the population to flip sophistication by
contest type, which is a stretch next to the simpler explanation.

### What this changes

`optimizer.optimize(bring_back=N)` has existed, tested, since the stacking work
and was never wired into the dashboard. It should be, for tournament builds
only. And the 60/40 cash-heavy target now has a third measurement against it:
the construction that pays in a tournament is actively harmful in a double-up,
so the two formats need different lineups, not one lineup entered in both.

Magnitudes line up with section 13's simulation better than they had any right
to. Simulated, QB+1+bring-back lifted P(>=180) by 8.5% against 6.7% for QB+1
alone. Counted, the top-1% rate goes 1.05% to 1.65%. Same direction, and the
real effect is larger.

## 26. Pricing a lineup against a real contest, and the tail that broke it

Two contests needed pricing: a $10K Daily Dollar (11,890 entries, 3,080 paid,
$1) and a $20K Pylon (7,927 entries, 2,011 paid, $3). The Pylon is the exact
contest two exports already describe, so for once there was a real field rather
than a modelled one, and a real payout table rather than a guess.

The answer came back at $2.18 per $1 staked against $0.84 for an average entry.
A 160% edge, from a lineup whose cash rate was 24.5% against the field's own
25.4%. An entry that cashes less often than average cannot return two and a
half times average.

**Where it came from.** The field is a frozen list of 7,927 numbers with a
maximum of 219.2. A simulated lineup has a tail reaching past that, and the
ranking hands it first place and $2,000 every time. That was 38% of the EV with
the width correction applied and 62% without it. The rest came from the same
defect lower down: the simulated lineup is wider than the entire field, which
cannot be true, because the field's spread already contains both outcome
variance and the variation between 7,927 different rosters.

The error is not the frozen field by itself. It is treating our score and the
field's as independent when they are drawn from the same twenty-two games. A
lineup posts 220 because the slate went wild, and the field went with it.
Against a top-heavy payout that one assumption becomes most of the answer.

**What survived.** Percentile and cash probability do not touch the tail, and
they were right in the same run: 24.5% against a 25.4% baseline, correctly
reporting an average lineup. They are what `contest_ev` exposes and what the
dashboard shows.

    lineup                 field assumed      pct-ile   P(cash)   P(top 1%)
    A (bring-back)         wk2, mean 116.7      29.6%     42.9%       2.69%
    A (bring-back)         wk3, mean 130.4      47.5%     24.5%       1.31%
    B (no bring-back)      wk2, mean 116.7      28.2%     45.4%       2.94%
    B (no bring-back)      wk3, mean 130.4      45.7%     26.4%       1.39%

**The scoring environment moves more than the lineup does.** The only two
observations of this contest had field means 14 points apart, and that swings
the cash probability from 43% to 25% -- far more than any difference between
two sensible lineups. A single quoted probability hides the dominant
uncertainty; quote the range across observed fields.

### The null test, which was missing

A lineup with no edge by construction must price at the rake. Running every
real entry against the field it came from returns $2.35 against an exact $2.52,
and 25.37% cash against an exact 25.37% -- the 7% shortfall is midrank
flattening ties at the top, which is conservative. That test now lives in
`tests/test_contest_ev.py` and would have caught this immediately.

Worth recording that the same method applied to the earlier 237-entry contest
was fine: null-tested it returns $0.86 against a true $0.844, 2% high. The
failure is specific to contests where one place pays 10-20% of the pool. The
EV figures quoted for that contest stand; the ones for these two do not.

### What went into the dashboard

- `bring_back` reached the UI at last. It has been in the optimizer, tested,
  since the stacking work, and was unreachable from the app the whole time --
  while being the best-evidenced construction rule in the project (section 25).
  GPP defaults to 1, cash to 0.
- Stacking a CASH lineup now raises a warning, because section 25 measured
  unstacked double-up entries cashing 60.9% against 21-40% for stacked ones.
- The Lineup value tab ranks the lineup against every counted field, rebuilt
  from the percentiles in `data/contest_fields.csv` -- the raw exports are not
  committed, so the summary is how a real field stays reachable. No EV column,
  and the caption says why.
- `contest_fields.csv` now records `format` and `slots`. Without it the two
  39-entry Showdown contests appeared as reference fields for a nine-man
  Classic lineup, which is how the first version of that table shipped.
- A Streamlit note, because it cost a round trip: `st.column_config` formats
  are literal printf, so a fraction passed to `"%.1f%%"` renders 0.254 as
  "0.3%". Multiply by 100 first.

## 27. The double-up premium is leveraged chalk, not a stronger field

Sections 16-20 built to a headline: contest choice is worth 20 points, because
double-up fields score 17-21 points above tournament fields on the same slate.
It replicated on two independent slates. Week 4 reversed the sign.

Five contests on one slate (dg 154078, 2026 week 4), all counted:

    entries        type   overlap   top-10 of 9   field mean
     11,890  tournament     0.129          2.38       116.57
      7,927  tournament     0.134          2.42       117.73
      3,448   double-up     0.344          5.17       110.46
      2,758   double-up     0.395          5.62       109.76
      2,298   double-up     0.414          5.79       109.65

The structural fingerprint is exactly as before -- the double-ups converge,
overlap 0.34-0.41 against 0.13, the ten most-owned filling 5.2-5.8 of nine
slots against 2.4. These are the same kind of contest. The premium is simply
**negative**: -7.2 instead of +17.6 and +21.3.

The mechanism was there all along and was being read as something else:

    week   double-up premium   chalk scored (ownership-weighted top 10)
       2              +17.62                                     18.13
       3              +21.32                                     20.04
       4               -7.20                                     11.72

**A converged field scores whatever its chalk scores.** Convergence is
leverage, not strength: it amplifies the chalk in both directions. Weeks 2 and
3 were chalk-hits, so the double-ups looked like a wall. Week 4 was a
chalk-bust -- Parker Washington was 71% owned in those double-ups and scored
2.0 -- and the same wall fell below the tournaments.

Three observations, so the correlation is not a measurement. The mechanism is
the claim, and it is falsifiable: the premium should track chalk performance
week to week, and `scripts/leverage.py` already computes everything needed.

What survives from section 20 is the structural half -- double-up fields ARE
converged, and that is stable across all three weeks. What does not survive is
"field strength rises as contests shrink", which was two draws from a
distribution read as a trend. The practical reading changes with it: a
double-up is not reliably harder, it is reliably more correlated with the
chalk, and whether that is good or bad is a bet on the chalk.

## 28. The extrapolation premium does not predict anything

While looking at a lineup the user distrusted, I showed a "premium" column --
model projection minus recent three-game average -- and used it to argue that
the player with the largest premium was the one to drop. He was right about the
player: Jordan Addison, premium +6.2, scored 9.1 against a projection of 14.2.

The diagnostic itself is a null. Across 242 rostered players on that slate with
a projection, a recent average and an actual score:

    corr(premium, model error) = +0.0604  (p = 0.3496)

and no monotonic pattern by quartile (mean error -0.78, -0.59, +0.66, -1.13).
Shading the projection toward recent form makes ranking WORSE, not better --
spearman 0.5550 against 0.6015 for the model alone, while recent form alone
manages 0.4864. The extrapolation the column flags is, on average, the model
being right.

That is six measured nulls now, not five: red-zone usage, calibration, stale
form, game total, defensive matchup, and this.

### Why the lineup built on it still won

The lineup that replaced Addison with Tee Higgins scored 148.02 against 119.12
for the original, and would have finished top 11.6% of a 7,927-entry tournament
against top 45.8%. That is not the diagnostic working. Higgins beat his
projection by +17.0 and Hockenson, also in it, by +19.0 -- the two largest
model errors on the slate, both in that lineup's favour, neither flagged by the
premium column (Higgins -2.1, Hockenson +1.3). The three biggest misses in the
other direction were Rashee Rice -17.3 (he scored 0.00), Ja'Marr Chase -16.7
and Parker Washington -11.8, and the Higgins lineup held two of them.

A single lineup's result cannot separate a good process from a lucky one, which
is exactly what section 24 says about using lineups as the unit of learning.
The Addison call was right and worth making. The reason given for it does not
generalise, and 242 players say so on the same slate that produced the win.

## 29. The bring-back replicates, including on the week the chalk busted

Week 4 added five Classic exports, taking section 25 from 48,579 counted
entries to 76,877 across twelve contests. The finding holds and the
falsification test sharpened.

    construction   entries   mean z   top 1%
    none            16,890  +0.1264    0.52%
    stack           30,666  -0.0306    0.98%
    bring-back       4,126  -0.1113    0.56%
    both            25,195  -0.0147    1.46%

    structure     n        both vs stack
    tournament   53,441   top-1% 2.05x (p<0.0001), cash 28.35% vs 24.50%
    double-up    23,436   top-1% 0.61x (p=0.0030), cash 27.29% vs 39.04%

In a double-up the bring-back was a flat null before; with week 4 it is
significantly HARMFUL -- 0.61x the top-1% rate and twelve points of cash rate.
That is a stronger result than the null, and in the direction the mechanism
predicts.

Per week, tournaments only:

    week 2   n= 7,921   0.61x (p=0.0538)   mean -0.0038 z
    week 3   n=25,725   2.84x (p<0.0001)   mean +0.0969 z
    week 4   n=19,795   2.23x (p<0.0001)   mean +0.1165 z

**Week 4 matters more than the other two.** It was the chalk-bust week -- the
one where the double-up premium went negative because the ten most-owned
players scored 11.7 against 18-20 in weeks 2 and 3 (section 27). If the
bring-back only paid when the chalk paid, it should have failed there. It
returned 2.23x.

Week 2 runs the other way at p=0.054 on a single 7,921-entry tournament. One
contest is one draw; the honest reading is two strong replications, one
contrary observation, and a mechanism that survived the week designed to break
it.

### The cash line is not a number, it is a range

Week 4 also supplied three main-slate double-up lines: 105.7, 106.1 and 108.2,
against 145.0 in week 1 and 144.2 in week 2. A 38-point swing in the scoring
environment.

That exposed a weighting bug in `cash_lines.observed`, which took the median
over ROWS. Three contests from one week outvoted two weeks with one each and
moved the dashboard default from 145 to 108 on one week's evidence out of
three. It now aggregates per week, then across weeks, and returns the observed
range so the caption can carry it: "median of 3 observed weeks (5 contests),
which ranged 106 to 145".

The range is the more honest figure. A 38-point swing between weeks is far
larger than any difference between two lineups, and a single number hides it.

## 30. A second opinion, imported rather than scraped

`docs/MODELING.md` has carried one gap since the beginning: everything here
compares the model to a baseline, to the crowd, or to itself, and none of that
says whether a professional service does better. Daily Fantasy Fuel publish a
"Download Projections as CSV" link, which answers it in one slate.

**Imported, not scraped, and the distinction is not pedantry.** Their
projections are the product they sell subscriptions for, so pulling them out of
the page is a terms problem where clicking their own button is not -- and this
project has already lost the DraftKings contest API permanently by probing it
too hard (section 17). The file arrives the way standings exports do, which is
the most productive data pipeline this repo has.

    first_name last_name position injury_status week game_date slate
    team opp spread over_under implied_team_score salary
    L5_dvp_rank L5_fppg_avg L10_fppg_avg szn_fppg_avg
    ppg_projection value_projection ownership_projection

The last column is the prize. Ownership is the only input measured to carry
signal the model does not have -- +0.29 and +0.18 of rank correlation with
outcomes after our own projection is removed (section 21) -- and the only one
that cannot be derived from a box score. Section 21 ends "using it needs
ownership BEFORE lock, which is a paid feed or a model of its own". That column
is the paid feed.

### The export is usually not ready, and a blank one is dangerous

The first file collected, six days before its slate, had 53 rows -- only the
injury-flagged players -- every `ppg_projection` exactly 0.0 and
`ownership_projection` entirely empty.

Loaded without checking, that is not a worse set of projections. It is the
claim that nobody on the slate will score, and an optimizer handed it returns
the nine cheapest names on the board. So `dff.usable()` rejects three shapes --
all projections zero, too few projected to be a slate, no rows at all -- and
`dff.load()` refuses by default, with a message saying to download again
closer to kickoff.

### The join: salary first, name only inside a cell

Rule 1 is never join on player names, and DFF publish DraftKings' own salary,
so (team, position, salary) is DK's data on both halves of the join rather than
two sources' spellings of a person.

That key alone is not enough. On the 629-player week 4 Classic slate only
**41%** of players sit in a unique (team, position, salary) cell, because the
minimum salary collects them -- six Arizona receivers at $3,000, seven
Jacksonville ones:

    players sharing a cell with:  1    2    3    4    5    6   7
    number of cells             229   26   24   23   13    5   2

So a second pass separates players INSIDE a cell by name. That is safe in a way
a global name join is not: everyone in the cell already shares a team, a
position and a price, so the failure that wrote rule 1 -- a 2019 Saints
receiver paired with a same-named defensive back -- cannot occur. Two identical
names in one cell still resolve to nobody. `matched_on` records which pass
caught each row; on that slate it was 229 by salary and 325 by name, 100% total.

### Applied through the override path

The dashboard's "Projections" control (Model / Blend 50/50 / DFF) does not
build a parallel projection path. It converts DFF's number into a per-player
factor and runs it through `overrides.apply`, which moves every points-like
column together and leaves `p_play` alone (rule 13). The optimizer and the
simulator therefore cannot end up disagreeing about a player, and the blend is
just a partial factor. Verified on the week 4 slate: Trey McBride's proj_mean
17.92 -> 24.70 under DFF with q85 26.62 -> 36.69, the same ratio, p_play
untouched.

`scripts/benchmark.py` is written and waits on a posted export: ours, theirs,
salary alone and the two averaged against actual points; the partial
correlations each way; and their projected ownership against what a contest
field really rostered.

### A bug worth recording for its shape

The suffix-stripping regex in the name tiebreaker was written with its word
boundaries replaced by literal backspace characters (0x08), because the shell
heredoc used to patch the file collapsed `\b` to `\b` and Python read that as
an escape. The pattern then matched nothing, "Aaron Jones Sr." and "Aaron
Jones" stayed distinct, and the only symptom would have been a slightly lower
match rate on real data -- invisible against synthetic names that already
agree. The convention note about normalising line endings before string-patching
a file now has a companion: build any string containing a backslash with
`chr(92)`, or use an editor rather than a heredoc.


## 31. The first real DFF export, and the four things it broke

Section 30 built the importer against a skeleton file. The first populated
export -- week 5, 410 rows, 393 projected, DSTs included -- broke four things,
none of which a synthetic fixture could have shown.

**ownership_projection is empty in all 410 rows of a fully posted export.** The
blank in the skeleton was not a timing artefact: the column is premium-gated.
The one input measured to carry signal the model lacks stays behind the
paywall. `has_ownership()` reports it and the reader handles the column, so a
premium export would work the day it arrives, but it is not available on the
free tier.

**One export covers the whole week.** `DFF_NFL_cheatsheet_2026-10-08.csv` holds
game_date 2026-10-08, 2026-10-11 AND 2026-10-12. The filename carries the
week's first game date and identifies nothing.

**A single-game slate is still a slate.** `usable()` required 50 projected
players, which rejected the Thursday Showdown export -- 30 rows, 2 teams -- as
incomplete. The floor is now 15 when the export covers two teams or fewer.

**The match rate was measuring the wrong thing.** Against the 1pm main slate
the join read 82.9%, under the 95% guard. All 51 worst offenders were ATL,
BAL, JAX and PHI: the 9:30am London game and other windows. An export covers
every game that day; a DK slate is a subset. `match()` now marks `on_slate`
and `match_rate()` counts only players on a team the slate contains. 96.7%,
and 100% on the Thu-Mon slate.

### Keyed on teams, because dates are wrong twice

The lookup was written on the slate's date and failed immediately in the live
dashboard, with its own error message giving the diagnosis:

    no rows for 2026-10-09 in 2 export(s); they cover 2026-10-08,
    2026-10-11, 2026-10-12

A slate's `start_time` is UTC. The Thursday night slate starts
2026-10-09T00:15Z, which is 2026-10-08 at 8:15pm Eastern, and DFF write the
Eastern date. Converting the timezone would have fixed that one and left the
second: a Thu-Mon slate spans three dates, so matching the first would have
pulled thirty Thursday players for a fifteen-game slate.

The teams in a slate identify its games exactly and no team plays twice in a
week, so `for_teams()` keys on those. Team aliases are applied to the query as
well as the file, or every Rams slate misses (the slate says LA, the export
says LAR). An export covering under 90% of a slate's teams is refused as
belonging to a different slate.

### Uploading

`st.file_uploader` in the sidebar, validated by `dff.save()` before the file
lands: a file that cannot be parsed never reaches the folder the dashboard
reads, because the whole argument of section 30 is that a silently empty
export is worse than none. It is written to a `.part` and renamed only after
it parses. An export that parses but is not ready is kept -- it may be the
right file for another day in the same week -- and refused at read time, where
the refusal belongs.

### Verified in the running dashboard

Blend 50/50 on the Thu-Mon slate: 410 of 771 players matched, 100% of those on
a team in the slate, and the cash lineup moved from a floor of 89.3 to 91.8.
Agreement with our model across 295 matched players on the Sunday slate is
spearman 0.944, means 7.04 against 6.70 -- two readings of the same thing
rather than rival worldviews, which is what makes the disagreements worth
reading.

Two operational notes, both of which cost a round trip.

Streamlit hot-reloads `app.py` but NOT imported modules under `src/`. After
changing `dfs/sources/dff.py` the running server still held the old module and
raised `AttributeError: module 'dfs.sources.dff' has no attribute 'for_teams'`.
Restart the server after changing anything outside `app.py`.

And `st.file_uploader` hands the SAME file back on every rerun. The first
version saved whenever the widget was non-empty and then called `st.rerun()`
to pick the new file up, which is an infinite loop -- save, rerun, see the
file again, save again. The reported symptom was the screen flickering grey
and never settling, which is exactly what an app rerunning forever looks like.
The work is now keyed on a name-and-size signature held in `st.session_state`
and happens once; measured after the fix, one rerun then twenty consecutive
idle samples. Any widget whose value survives a rerun has this shape.


## 32. DFF publish Classic salaries, which breaks every Showdown slate

The user reported three times that the projection toggle changed nothing, and
twice I answered with stale browser tabs and restarts. They were right and I
was wrong: on the slate they were actually looking at -- a TB at DAL Showdown
-- the toggle genuinely did nothing.

DFF's export carries DraftKings' CLASSIC salary. A single-game slate prices
the same player completely differently:

    player            DFF (classic)   showdown
    CeeDee Lamb               8,300     11,800
    Javonte Williams          7,300     10,800
    Dak Prescott              6,700     10,400
    Baker Mayfield            5,200      9,600

Keyed on (team, position, salary) that slate matched **2 of 30**. The override
loop is guarded by `if _rows:`, so with nothing to apply it applied nothing,
and the lineup was identical under Model and under DFF. Two screenshots side
by side, byte-identical.

The join now has a third pass: the name within (team, position), for rows the
price cannot reach. That still cannot make rule 1's mistake -- that was a 2019
Saints receiver paired with a same-named defensive back, and here both the
team and the position must already agree. Ambiguity resolves to nobody, as
before. Measured: the Showdown slate goes 2/30 to 28/28, and the Classic slate
is unchanged at 305/305 with salary still doing the primary work (204 by
salary, 101 by name inside a crowded cell).

### It failed silently, which is the worse half

Nothing in the interface said the override had matched nobody. The caption
reporting the match rate was rendered AFTER the upload expander, far enough
down the sidebar to be off-screen, and said nothing special at zero. A control
that quietly does nothing is indistinguishable from a control that is broken,
which is why three rounds went into diagnosing browser state.

Now: the caption sits directly under the radio, and a match of zero is an
error box rather than a caption -- "DFF selected but NOTHING matched, so the
projections below are still the model's."

The general lesson is the one rule 14 states in a different costume. A
pipeline that can produce "no result" needs to say so where the result would
have been. `if _rows:` silently skipping is the same shape as an EV number
quietly built from its own tail.

A fixture note worth keeping: the first version of the regression test used
filler names "A0 B0", "A1 B1" and so on, every one of which `_key_name`
reduces to "ab" once digits are stripped. They were all refused as ambiguous
-- correct behaviour, useless as filler. Test names for a name-matching test
have to differ in letters.


## 33. Per-slate exports, because DFF price them per slate

Section 32 fixed the Showdown match by falling back to names. That works, but
it is a workaround for using the wrong file: DFF offer a per-slate download,
and a Showdown one carries the SINGLE-GAME salaries, which makes the price key
valid again instead of unusable.

Uploads are therefore bound to the slate they were uploaded against, stored as
`dg<draft_group>__<original name>.csv`. `for_slate()` prefers a file bound to
that draft group and falls back to any week file covering its teams, so one
Classic upload still serves every Classic slate of that week while a Showdown
gets its own. The caption now names which file is in use -- "From a week file"
or the filename -- because three rounds of this conversation went into not
knowing what the dashboard was actually reading.

The widget keys carry the draft group too (`proj_src_<dgid>`,
`dff_upload_<dgid>`). Without that, switching slates would have carried the
previous slate's radio selection and the previous slate's pending upload into
a slate they do not belong to.

Measured on the live TB at DAL Showdown with only the week file present: 30
rows, 30 matched, 28 of them through the cross-format name pass and 2 through
salary by coincidence. With a per-slate file the salary pass should do all of
it, which is both faster to verify and harder to get wrong.

"""DFS dashboard.

    .venv\\Scripts\\streamlit.exe run app.py

Picks a live DraftKings slate, projects every priced player, and builds
lineups. Cash mode optimizes a floor because double-ups pay for clearing a
threshold; GPP mode optimizes upside and stacks, because tournaments pay for
the right tail.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from dfs import (backtest, cash_lines, contest_ev, features,  # noqa: E402
                 model, optimizer, overrides, simulate, slates)
from dfs.config import CURATED, SALARY_CAP  # noqa: E402
from dfs.optimizer import SHOWDOWN_SIZE  # noqa: E402
from dfs.sources import dff, dk_api  # noqa: E402

st.set_page_config(page_title="DFS Dashboard", page_icon="🏈", layout="wide")

# Each contest optimises a different point on the same predicted distribution.
# The floor is what you roster on; the mean is what you expect.
CONTESTS = {
    "Cash / double-up": dict(
        col="proj_floor", view="Floor", stack=0, back=0, min_diff=1,
        blurb="Optimizes the **floor** -- the 30th percentile, a "
              "bad-but-not-disastrous day. Double-ups pay for clearing a "
              "threshold, so you want the lineup least likely to fall apart. "
              "**Do not stack here:** across 14,933 counted double-up entries, "
              "lineups with no stack at all cashed 60.9% of the time against "
              "21-40% for stacked ones.",
    ),
    "GPP / tournament": dict(
        col="proj_ceiling", view="Ceiling", stack=1, back=1, min_diff=3,
        blurb="Optimizes the **ceiling** -- the 85th percentile, a day he "
              "reaches about one time in seven -- then stacks the QB with a "
              "pass catcher and brings one back from the opposing side. "
              "Across 33,646 counted tournament entries the bring-back gave "
              "1.97x the top-1% rate (p<0.0001). Tournaments pay the tail.",
    ),
    "Neutral (expected)": dict(
        col="proj_mean", view="Expected", stack=0, back=0, min_diff=1,
        blurb="Optimizes the **mean** -- what he averages over many runs. "
              "The honest number for judging a slate, though it does not "
              "build better lineups than the floor.",
    ),
}


# --- data -------------------------------------------------------------------

@st.cache_data(ttl=300, show_spinner=False)
def live_slates() -> pd.DataFrame:
    """Live NFL slates in the DraftKings lobby.

    The lobby already reports a contest type, so we only download the player
    payload for salary-cap formats. Fetching every draft group meant ~40
    round trips of roughly 300KB each before the page could render anything.
    """
    live = dk_api.live_draftgroups("NFL")
    if live.empty:
        return pd.DataFrame()

    keep = live[live["contest_type_id"].isin(
        (dk_api.CONTEST_TYPE_CLASSIC, dk_api.CONTEST_TYPE_SHOWDOWN)
    )]
    if keep.empty:          # unknown type ids: fall back to checking them all
        keep = live

    rows = []
    bar = st.sidebar.progress(0.0, text="Checking DraftKings for live slates...")
    for i, r in enumerate(keep.itertuples(), 1):
        info = dk_api.peek(r.draft_group_id, cache_only=False)
        bar.progress(i / len(keep),
                     text=f"Checking slates... {i} of {len(keep)}")
        if not info or not info["is_nfl"] or info["kind"] == "draft":
            continue
        rows.append({**info, "suffix": r.suffix})
    bar.empty()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df.sort_values(["kind", "n_games"], ascending=[True, False])


@st.cache_data(show_spinner="Loading five seasons of history...")
def history(showdown: bool = False) -> pd.DataFrame:
    """Training rows for the format being played.

    Showdown is a separate pricing regime -- roughly 45 priced players instead
    of 800, and the only format that prices kickers -- so it has its own
    training set. Borrowing the Classic model gave kickers no projection at
    all and mispriced everyone else.
    """
    name = "showdown_features" if showdown else "features"
    path = CURATED / f"{name}.parquet"
    if not path.exists():
        return pd.DataFrame()
    return backtest.prepare(pd.read_parquet(path))


@st.cache_data(show_spinner=False)
def metrics() -> dict:
    path = CURATED / "metrics.json"
    return json.loads(path.read_text()) if path.exists() else {}


@st.cache_data(show_spinner="Training on history and projecting the slate...")
def project(dgid: int, showdown: bool = False) -> pd.DataFrame:
    """Floor, mean and ceiling for every priced player.

    Cached on the slate and its format, not the contest type: all three views
    come out of one fit, so switching between cash and GPP is instant instead
    of retraining.
    """
    upcoming = slates.assemble(dgid)
    if upcoming.empty:
        return pd.DataFrame()
    season = int(upcoming["season"].iloc[0])
    week = int(upcoming["week"].iloc[0])

    past = history(showdown)
    past = past[(past["season"] < season)
                | ((past["season"] == season) & (past["week"] < week))]
    if past.empty:
        return pd.DataFrame()

    cur = features.build(upcoming, salary_within_slate=showdown)
    cur = cur[cur["salary"].fillna(0) > 0].reset_index(drop=True)
    views = model.two_stage_all(past, cur)
    if views.empty:
        return pd.DataFrame()
    return cur.merge(views, on="player_dk_id", how="left")


def _training_split(upcoming: pd.DataFrame, showdown: bool = False
                    ) -> tuple[pd.DataFrame, int, int]:
    season = int(upcoming["season"].iloc[0])
    week = int(upcoming["week"].iloc[0])
    past = history(showdown)
    past = past[(past["season"] < season)
                | ((past["season"] == season) & (past["week"] < week))]
    return past, season, week


@st.cache_data(show_spinner="Working out what drives each projection...")
def explanation(dgid: int, showdown: bool = False
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-player projection split and feature attribution."""
    upcoming = slates.assemble(dgid)
    if upcoming.empty:
        return pd.DataFrame(), pd.DataFrame()
    past, _, _ = _training_split(upcoming, showdown)
    if past.empty:
        return pd.DataFrame(), pd.DataFrame()
    cur = features.build(upcoming, salary_within_slate=showdown)
    cur = cur[cur["salary"].fillna(0) > 0].reset_index(drop=True)
    return model.explain(past, cur)


# Showdown construction rules, tested on 1,416 finished slates
# (docs/MODELING.md 34). Only rules with a clear measured effect are here:
# two-QB builds and QB + opposing DST were NOT harmful, so neither is banned.
# QB Captain was a cash default until the first real field contradicted it:
# on SEA @ DEN (contest 196688500) 92% of a 62-entry double-up captained JSN
# and the chalk block WAS the cash line, so a QB Captain was a bet against
# the field that a double-up does not pay for. The outcome test behind it
# used a modelled field that converges far less than real ones. Still
# available as optimizer.optimize_showdown(qb_captain=True).
SHOWDOWN_RULES = {
    "proj_floor": dict(min_qb=1),                       # cash
    "proj_ceiling": dict(min_qb=1, qb_stack=1),         # tournament
    "proj_mean": dict(min_qb=1),
}

N_SIMS = 4000
N_FIELD = 1000


def build_lineups(pool: pd.DataFrame, showdown: bool, n: int, min_diff: int,
                  stack_n: int, min_salary: int, max_exposure: float,
                  locks: list[int], bans: list[int], back_n: int = 0,
                  captain: int | None = None,
                  rules: dict | None = None) -> list[pd.DataFrame]:
    """One entry point for both formats, so the lineups shown and the lineups
    simulated cannot drift apart.

    `back_n` is the bring-back, and it reached the dashboard long after it
    reached the optimizer. It is the best-evidenced construction rule here --
    48,579 counted entries, docs/MODELING.md 25 -- and it was unreachable from
    the app the whole time.
    """
    if showdown:
        # One game, so every player is already in it; there is nothing to
        # bring back and no QB slot to stack from.
        return optimizer.optimize_showdown(
            pool, objective="projection", n_lineups=n, min_difference=min_diff,
            max_exposure=max_exposure, locks=locks, bans=bans,
            min_salary=min_salary, captain=captain, **(rules or {}),
        )
    return optimizer.optimize(
        pool, objective="projection", n_lineups=n, stack_qb_with=stack_n,
        bring_back=back_n, min_difference=min_diff, max_exposure=max_exposure,
        locks=locks, bans=bans, min_salary=min_salary,
    )


# The cash line comes from data/observed_cash_lines.csv when anything has been
# recorded for this size of slate, and from an estimate otherwise. The
# simulated field cannot be used for it: it medians about 107 on a main slate
# against a first observed line of 147, because it samples players and repairs
# over-cap lineups by swapping down to the cheapest option.
def default_cash_line(n_games: int, showdown: bool) -> tuple[int, str]:
    return cash_lines.default_line(int(n_games), showdown)


@st.cache_data(show_spinner="Simulating the slate a few thousand times...")
def simulate_slate(pool: pd.DataFrame, showdown: bool, n_sims: int = N_SIMS,
                   n_field: int = N_FIELD, seed: int = 0):
    """Sampled outcomes for every player, plus a field to measure against.

    Cached on the pool, so changing which lineups you look at is free -- and
    every lineup is compared in the SAME simulated worlds, which makes the
    comparison paired rather than a set of independently noisy estimates.
    """
    sim = simulate.sample(pool, n_sims=n_sims, seed=seed)
    opp = simulate.field_entries(pool, sim, n=n_field, seed=seed + 1,
                                 showdown=showdown)
    return sim, opp


@st.cache_data(show_spinner=False)
def dff_export(dgid: int, teams: tuple[str, ...]
               ) -> tuple[pd.DataFrame, str, str]:
    """(DFF's projections for this slate, why not). Imported, never scraped.

    DFF publish a "Download Projections as CSV" link; drop the file in
    data/projections/ and it is picked up by its date. An export taken before
    they post the slate carries every projection as 0.0, which an optimizer
    reads as "nobody will score" and answers with the nine cheapest names --
    so `dff.load` refuses it and the reason is shown rather than swallowed.
    """
    # Matched on the slate's TEAMS. One export covers the whole week, so the
    # filename identifies nothing; and the date gets it wrong twice -- a
    # slate's start_time is UTC while DFF write the Eastern game date, and a
    # Thu-Mon slate spans three of them.
    try:
        return dff.for_slate(dgid, teams)
    except (ValueError, OSError) as exc:
        return pd.DataFrame(), str(exc), ""


@st.cache_data(show_spinner=False)
def real_fields(showdown: bool = False) -> pd.DataFrame:
    """Contests of this format whose standings have been ingested.

    These are the only measurements of a field this project has that are not a
    model of themselves. Filtered by format, because a six-man Showdown field
    scoring around 90 says nothing about a nine-man Classic lineup -- and a row
    with no format recorded is skipped rather than guessed at.
    """
    path = Path("data/contest_fields.csv")
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    need = {"entries", "paid_positions", "mean", "min", "max", "format"}
    if df.empty or not need.issubset(df.columns):
        return pd.DataFrame()
    want = "showdown" if showdown else "classic"
    df = df[df["format"].astype(str) == want]
    return df.dropna(subset=["paid_positions"]).sort_values(
        "entries", ascending=False).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def ruled_out(dgid: int) -> pd.DataFrame:
    """Players DraftKings flags as unable to play (IR, OUT, suspended)."""
    return dk_api.unavailable(dk_api.draftables(dgid))


# --- sidebar ----------------------------------------------------------------

st.sidebar.title("🏈 DFS Dashboard")

# Something on screen before the slow work starts. A cold load checks the
# DraftKings lobby and then trains on five seasons, which is a minute or two
# of nothing happening -- and a blank page is indistinguishable from a hang.
_boot = st.empty()
with _boot.container():
    st.title("🏈 DFS Dashboard")
    st.write("Starting up...")
    st.caption(
        "First load reads the DraftKings lobby, then trains on five seasons "
        "of history. That takes a minute or two. Afterwards it is cached, and "
        "switching slate or contest type is immediate."
    )

live = live_slates()
if live.empty:
    st.sidebar.warning("No live NFL slates in the DraftKings lobby.")
    st.title("No live slates")
    st.write(
        "DraftKings has no NFL slates open right now. The lobby fills up "
        "during the week ahead of each slate."
    )
    st.stop()

live["label"] = live.apply(
    lambda r: f"{r['kind']} - {r['n_games']}g - {r['suffix'] or r['draft_group_id']}",
    axis=1,
)
choice = st.sidebar.selectbox("Slate", live["label"], index=0)
row = live[live["label"] == choice].iloc[0]
dgid = int(row["draft_group_id"])

contest = st.sidebar.radio("Contest type", list(CONTESTS), index=0)
st.sidebar.caption(CONTESTS[contest]["blurb"])

n_lineups = st.sidebar.slider("Lineups", 1, 20, 1)
max_exposure = st.sidebar.slider(
    "Max exposure per player", 0.1, 1.0, 1.0, 0.05,
    help="Cap on the share of your lineups any one player can appear in.",
) if n_lineups > 1 else 1.0

st.sidebar.divider()
m = metrics()
if m:
    st.sidebar.caption("**Model (backtested)**")
    st.sidebar.caption(
        f"{m.get('weeks', '?')} weeks, {m.get('seasons', '?')}  \n"
        f"Rank correlation **{m.get('spearman', 0):.3f}**  \n"
        f"Lineup avg **{m.get('lineup_mean', 0):.0f}** pts "
        f"(+{m.get('lineup_edge', 0):.0f} vs baseline)"
    )

# --- main -------------------------------------------------------------------

_boot.empty()   # real content is ready; clear the startup placeholder
st.title(choice)

# The format decides which model and which training set to use, so it has to
# be known before projecting rather than only at lineup time.
is_showdown = row["kind"] == "showdown"
cur = project(dgid, is_showdown)
if cur.empty:
    st.error("Could not assemble this slate. It may not map to a scheduled week.")
    st.stop()

# The contest type selects which view of the distribution to optimise. All
# three were computed together, so switching is instant.
_view_col = CONTESTS[contest]["col"]

# Hand edits are applied here, before anything reads a projection, and they
# move the whole distribution rather than one column -- the optimizer reads
# proj_*, the simulator reads the quantile grid, and a control that moved only
# the number on screen would leave the Lineup value tab disagreeing with the
# projection it was shown. The log on disk is the source of truth rather than
# widget state, so edits survive a restart and can be judged after the games.
# A second opinion, when one has been downloaded for this slate. It is
# applied through the SAME path as a hand edit -- one factor moving every
# points-like column, p_play untouched -- so the optimizer and the simulator
# cannot end up disagreeing about a player (rule 13).
_slate_teams = tuple(sorted(cur["team"].dropna().astype(str).unique()))
_slate_date = str(pd.to_datetime(row["start_time"]).date())
_dff, _dff_why, _dff_src = dff_export(dgid, _slate_teams)
_src = "Model"
if not _dff.empty:
    # An explicit key pins the widget's identity. Without one Streamlit
    # derives it from the label, the options AND the position in the widget
    # tree, so a sidebar that changes shape around it can silently reset the
    # selection to the default -- which looks exactly like the control doing
    # nothing.
    _src = st.sidebar.radio(
        "Projections", ["Model", "Blend 50/50", "DFF"], index=0,
        key=f"proj_src_{dgid}",
        help="DFF's own numbers, imported from their CSV export. Blend is the "
             "average -- two independent projections usually rank better than "
             "either, though that is untested here.",
    )
elif _dff_why:
    st.sidebar.caption(f"DFF projections unavailable: {_dff_why}")

if _src != "Model" and not _dff.empty:
    _m_all = dff.match(_dff, cur[["player_dk_id", "player_name", "position",
                                  "team", "salary"]])
    _m = _m_all[_m_all["matched"]]
    _w = 0.5 if _src.startswith("Blend") else 1.0
    _base = cur.set_index("player_dk_id")[_view_col]
    _rows = []
    for r in _m.itertuples():
        pid = int(r.player_dk_id)
        if pid not in _base.index or not np.isfinite(_base[pid]):
            continue
        # DFF give one number; scale their mean onto whichever view is being
        # optimised so the factor is computed like for like.
        scale = _base[pid] / max(cur.loc[cur["player_dk_id"] == pid,
                                         "proj_mean"].iloc[0], 1e-6)
        want = (1 - _w) * _base[pid] + _w * r.dff_proj * scale
        _rows.append({"player_dk_id": pid, "player_name": r.player_name,
                      "view": _view_col, "model_value": float(_base[pid]),
                      "user_value": float(want)})
    if _rows:
        cur = overrides.apply(cur, pd.DataFrame(_rows), _view_col)
    _rate = dff.match_rate(_m_all)
    if not _rows:
        st.sidebar.error(
            f"**{_src} selected but NOTHING matched**, so the projections "
            "below are still the model's. The export and this slate do not "
            "line up.", icon=":material/error:")
    else:
        st.sidebar.caption(
            f"**{_src}** applied to {len(_rows)} of {len(cur)} players "
            f"({_rate:.0%} of those on a team in this slate"
            + (f", via {_m_all.loc[_m_all['matched'], 'matched_on'].mode().iat[0]}"
               if _m_all["matched"].any() else "")
            + ")."
            + f"  From {_dff_src}."
            + ("  Projected ownership included."
               if dff.has_ownership(_dff) else
               "  No projected ownership in this export.")
        )

with st.sidebar.expander(f"Upload DFF projections for {choice}",
                         expanded=True):
    st.caption(
        "Their **Download Projections as CSV**. A file uploaded here is "
        "bound to **this slate** and used for it in preference to anything "
        "else -- which matters for Showdown, where DFF's per-slate download "
        "carries the single-game salaries and the week file carries Classic "
        "ones. A week file still covers any slate it has the teams for, so "
        "one upload is enough for Classic. Download close to kickoff: taken "
        "early, every projection is 0.0 and it will be refused."
    )
    _up = st.file_uploader("DFF cheatsheet (.csv)", type="csv",
                           key=f"dff_upload_{dgid}",
                           label_visibility="collapsed")
    # A file_uploader hands the SAME file back on every rerun, so saving
    # whenever it is non-empty and then calling st.rerun() is an infinite
    # loop -- the screen flickers grey and never settles. Do the work once,
    # keyed on what was actually uploaded.
    if _up is not None:
        _sig = f"{dgid}:{_up.name}:{len(_up.getvalue())}"
        if st.session_state.get("dff_done") != _sig:
            try:
                _saved, _info = dff.save(_up.name, _up.getvalue(), dgid=dgid)
            except (ValueError, OSError) as exc:
                st.session_state["dff_done"] = _sig
                st.session_state["dff_msg"] = ("error", f"Not stored: {exc}")
            else:
                st.session_state["dff_done"] = _sig
                st.session_state["dff_msg"] = ("ok", (
                    f"Stored {_saved.name}: {_info['rows']} rows, "
                    f"{_info['projected']} projected, {_info['teams']} teams, "
                    f"covering {', '.join(_info['dates'])}."
                    + ("  Projected ownership included."
                       if _info["ownership"] else
                       "  No projected ownership (that column is premium).")))
                dff_export.clear()      # the lookup is cached on the folder
                st.rerun()              # once, to pick the new file up
    _kind, _msg = st.session_state.get("dff_msg", (None, None))
    if _msg:
        (st.success if _kind == "ok" else st.error)(_msg)
    _mine = [f.name for f in dff.available()
             if f.name.startswith(dff.slate_prefix(dgid))]
    _other = [f.name for f in dff.available()
              if not f.name.startswith(dff.slate_prefix(dgid))]
    if _mine:
        st.caption("Bound to this slate: " + ", ".join(_mine))
    if _other:
        st.caption("Other files: " + ", ".join(_other))

_edits = overrides.load(dgid)
cur = overrides.apply(cur, _edits, _view_col)
_n_edits = int((_edits["view"] == _view_col).sum()) if not _edits.empty else 0

cur["projection"] = cur[_view_col]
cur["pts_per_1k"] = cur["projection"] / (cur["salary"] / 1000)

season, week = int(cur["season"].iloc[0]), int(cur["week"].iloc[0])
valid = slates.matches_schedule(dk_api.draftables(dgid), season, week)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Season / week", f"{season} W{week}")
c2.metric("Priced players", len(cur))
c3.metric("Games", int(row["n_games"]))
c4.metric("Ruled out", len(ruled_out(dgid)))
c5.metric("Real NFL games", "yes" if valid else "NO")

_out = ruled_out(dgid)
if not _out.empty:
    with st.expander(f"{len(_out)} players excluded as unable to play"):
        st.caption(
            "DraftKings still prices players on injured reserve, usually at "
            "the $300 minimum. That makes them look like a free roster spot "
            "to an optimizer, so they are removed from the pool entirely."
        )
        st.dataframe(
            _out[["player_name", "position", "team", "salary", "status"]]
            .rename(columns={"player_name": "Player", "position": "Pos",
                             "team": "Team", "salary": "Salary",
                             "status": "Status"})
            .sort_values("Status"),
            hide_index=True, width="stretch",
        )

# There was an "early season -- projections are compressed" notice here,
# telling you a healthy starter might show a 70% chance of playing in week 3
# and to treat the point totals as low until week 5. It was written to explain
# a bug and it outlived it. Two defects made every LIVE slate project about a
# third of what it should -- a stale nflverse cache, and lagged features with
# no row for the unplayed week -- and the symptom looked exactly like
# early-season compression, so the page ended up reassuring you that a broken
# projection was expected behaviour.
#
# There is no compression to warn about. Over five walk-forward seasons, for
# the top 40 players by salary:
#
#     week 1   p_play 0.961   projected 17.52   actual 16.48
#     week 3   p_play 0.886   projected 16.44   actual 15.40
#     week 8+  p_play 0.902   projected 16.48   actual 16.00
#
# Early weeks project slightly HIGHER than late ones and track actuals just as
# well. Do not re-add this without measuring it first.


if not valid:
    st.error(
        "**These matchups are not in the NFL schedule.** DraftKings runs "
        "simulated video-game contests that look identical to real slates. "
        "Do not trust these projections."
    )

# Drop anyone DK has flagged as unable to play. They are usually priced at the
# $300 minimum, which makes an IR player look like a free roster spot to an
# optimizer -- Isiah Pacheco was rostered on a live showdown slate this way.
out_players = ruled_out(dgid)
if not out_players.empty:
    cur = cur[~cur["player_dk_id"].isin(out_players["player_dk_id"])]
    cur = cur.reset_index(drop=True)

# Kickers are priced on Showdown only, and are now trained on: the Showdown
# model projects them at 6.16 against an actual 6.49, rank correlation 0.526.
kickers = cur[cur["position"] == "K"].copy()

# --- controls ---------------------------------------------------------------
# Rendered here rather than with the rest of the sidebar because they need the
# player pool, which is not known until the slate has been projected.
st.sidebar.divider()
st.sidebar.subheader("Controls")

_label = cur.assign(
    _l=cur["player_name"] + " (" + cur["position"] + ", " + cur["team"] + ")"
).set_index("_l")["player_dk_id"]
_by_proj = cur.sort_values("projection", ascending=False)
_ordered = (_by_proj["player_name"] + " (" + _by_proj["position"] + ", "
            + _by_proj["team"] + ")").tolist()

lock_names = st.sidebar.multiselect(
    "Lock in", _ordered, help="Force these players into every lineup."
)
ban_names = st.sidebar.multiselect(
    "Rule out", _ordered, help="Never roster these players."
)
locks = [int(_label[n]) for n in lock_names if n in _label.index]
bans = [int(_label[n]) for n in ban_names if n in _label.index]

captain_id = None
use_rules = False

with st.sidebar.expander("More constraints"):
    min_proj = st.slider(
        "Minimum projection", 0.0, 12.0, 0.0, 0.5,
        help="Drop near-zero punt plays. Useful for cash, where a player who "
             "scores nothing sinks the lineup. The best few at each position "
             "are always kept, so this cannot make the roster impossible.",
    )
    min_salary = st.slider(
        "Minimum salary spend", 0, SALARY_CAP, 0, 500,
        help="Force the lineup to use at least this much of the cap.",
    )
    if not is_showdown:
        stack_n = st.slider(
            "Pass catchers with your QB", 0, 3,
            CONTESTS[contest]["stack"],
            help="Require this many WR/TE from the QB's own team. Measured "
                 "correlation: QB/own WR1 +0.23, QB/own TE1 +0.19, but "
                 "QB/own RB1 only +0.07 -- a running back on the QB's team "
                 "is not a stack.",
        )
        back_n = st.slider(
            "Bring back from the other side", 0, 2,
            CONTESTS[contest]["back"],
            help="Require this many RB/WR/TE from the team your QB is playing "
                 "AGAINST. On 33,646 counted tournament entries, adding this "
                 "to a stack gave 1.97x the top-1% rate (p<0.0001). On 14,933 "
                 "counted double-up entries it did nothing for the tail "
                 "(0.92x, p=0.83) and cost 11 points of cash rate. It buys "
                 "tail, so use it only where the tail is paid.",
        )
    else:
        stack_n = back_n = 0
        use_rules = st.toggle(
            "Apply tested Showdown rules", value=True,
            help="On: at least one QB in every lineup; in tournaments, "
                 "every QB comes with one of his own WR/TE. Measured on 1,416 "
                 "finished slates: no-QB lineups cashed 35.7% vs 44.9%, and a "
                 "QB with no receiver of his own reached the top 1% less "
                 "often. Two-QB builds and QB + opposing DST are NOT banned "
                 "-- two QBs was the best structure tested. The Captain is "
                 "left to the optimizer: in a converged double-up, check who "
                 "the field's obvious Captain is before choosing another.",
        )
        # The Captain is the only Showdown control that matters, and it is not
        # a value decision: DK charges exactly 1.5x salary for 1.5x points, so
        # points per dollar is IDENTICAL there. What it buys is concentration
        # -- 1.5x of one player's outcome -- paid for in cap space the other
        # five then do not have. Comparing candidates means rebuilding the
        # roster around each, which a lock cannot do.
        _cap_opts = ["(optimizer chooses)"] + list(_ordered)
        _cap_pick = st.selectbox(
            "Captain", _cap_opts, index=0,
            help="Pin the 1.5x slot. Points per dollar is the same at Captain "
                 "and FLEX, so this is about which player you want one and a "
                 "half times of -- not about value.",
        )
        if _cap_pick != "(optimizer chooses)" and _cap_pick in _label.index:
            captain_id = int(_label[_cap_pick])

    # Stacking buys tail and costs mean. A double-up pays a flat amount for
    # clearing a bar, so there the tail is worthless and the cost is not.
    if CONTESTS[contest]["col"] == "proj_floor" and (stack_n or back_n):
        st.warning(
            "Stacking a cash lineup is measured as harmful. Of 14,933 counted "
            "double-up entries, unstacked lineups cashed **60.9%** of the "
            "time; stacked ones 21-40%. Stacking buys tail, and a double-up "
            "does not pay the tail.",
            icon=":material/warning:",
        )
    games = sorted(cur["competition_name"].dropna().unique()) \
        if "competition_name" in cur.columns else []
    drop_games = st.multiselect(
        "Exclude games", games,
        help="Remove whole games -- weather, a blowout you want no part of.",
    )

def apply_floor(df: pd.DataFrame, floor: float, showdown: bool) -> pd.DataFrame:
    """Drop punt plays without starving a required roster slot.

    A flat floor across all positions is unusable: defences and tight ends
    project far below running backs, so any floor high enough to cut punts
    also empties DST and makes the roster infeasible. We therefore always keep
    the best few at each position regardless of the floor -- the control
    removes weak options, it never removes your only options.
    """
    if floor <= 0:
        return df
    keep = df["projection"] >= floor
    if not showdown:
        # Comfortably more than each slot needs, so the solver still chooses.
        headroom = {"QB": 3, "RB": 6, "WR": 8, "TE": 4, "DST": 3}
        for pos, n in headroom.items():
            best = df[df["position"] == pos].nlargest(n, "projection").index
            keep.loc[best] = True
    return df[keep]


sd_rules = (SHOWDOWN_RULES.get(CONTESTS[contest]["col"], {})
            if is_showdown and use_rules else None)

pool = apply_floor(cur, min_proj, is_showdown)
# No projection, no automatic selection. A locked kicker is re-added below.
pool = pool[pool["projection"].notna()]
if drop_games:
    pool = pool[~pool["competition_name"].isin(drop_games)]
# A locked player is an explicit instruction, so it outranks every filter --
# including the kicker hold-out. Give him a nominal projection so the solver
# has something to optimise; the lock is what puts him in the lineup.
if locks:
    forced = cur[cur["player_dk_id"].isin(locks)].copy()
    forced["projection"] = forced["projection"].fillna(0.0)
    pool = pd.concat([pool[~pool["player_dk_id"].isin(locks)], forced])
pool = pool.reset_index(drop=True)

_dropped = len(cur) - len(pool)
if _dropped > 0:
    st.sidebar.caption(f"Filters removed {_dropped} of {len(cur)} players.")

# Showdown is a different game, not a Classic variant: 1 Captain + 5 FLEX, no
# position slots, captain at 1.5x points and 1.5x salary.
built = build_lineups(pool, is_showdown, n_lineups,
                      CONTESTS[contest]["min_diff"], stack_n, min_salary,
                      max_exposure, locks, bans, back_n, captain_id, sd_rules)
rules_dropped = False
if not built and sd_rules:
    built = build_lineups(pool, is_showdown, n_lineups,
                          CONTESTS[contest]["min_diff"], stack_n, min_salary,
                          max_exposure, locks, bans, back_n, captain_id)
    rules_dropped = bool(built)

tab_lu, tab_sim, tab_players, tab_adjust, tab_why, tab_slate, tab_chart = \
    st.tabs(["Lineups", "Lineup value", "All players", "Adjust projections",
             "Why this projection", "Slate context", "Value chart"])

with tab_lu:
    if not built:
        # Naming the likely culprit matters: with several filters active,
        # "infeasible" alone leaves you guessing which one to relax.
        reasons = []
        if locks:
            locked = cur[cur["player_dk_id"].isin(locks)]
            spent = locked["salary"].sum()
            if len(locks) > (SHOWDOWN_SIZE if is_showdown else 9):
                reasons.append(f"you locked {len(locks)} players into a "
                               f"{SHOWDOWN_SIZE if is_showdown else 9}-man roster")
            elif spent > SALARY_CAP:
                reasons.append(f"your locked players alone cost ${spent:,.0f}")
            else:
                reasons.append(f"{len(locks)} locked players may not fit "
                               "alongside a legal roster")
        if min_proj > 0 and not is_showdown:
            by_pos = pool["position"].value_counts().to_dict()
            thin = [p for p, need in (("QB", 1), ("RB", 2), ("WR", 3),
                                      ("TE", 1), ("DST", 1))
                    if by_pos.get(p, 0) < need]
            if thin:
                reasons.append(
                    f"a minimum projection of {min_proj:g} leaves too few "
                    f"players at {', '.join(thin)}"
                )
            else:
                # Cheapest legal roster: the least expensive qualifier at each
                # slot. If even that busts the cap, the floor priced you out.
                cheapest = 0.0
                for p, need in (("QB", 1), ("RB", 2), ("WR", 3),
                                ("TE", 1), ("DST", 1)):
                    cheapest += pool[pool["position"] == p].nsmallest(
                        need, "salary")["salary"].sum()
                cheapest += pool[pool["position"].isin(("RB", "WR", "TE"))][
                    "salary"].nsmallest(4).iloc[-1] if len(pool) else 0
                if cheapest > SALARY_CAP:
                    reasons.append(
                        f"a minimum projection of {min_proj:g} leaves only "
                        f"expensive players -- the cheapest legal roster would "
                        f"cost about ${cheapest:,.0f}, over the ${SALARY_CAP:,} cap"
                    )
        if min_salary > SALARY_CAP * 0.99:
            reasons.append(f"the minimum spend of ${min_salary:,} is close to "
                           "the cap")
        if drop_games:
            reasons.append(f"{len(drop_games)} excluded game(s) shrank the pool")

        if reasons:
            st.warning("No legal lineup with these constraints: "
                       + "; ".join(reasons) + ".")
        else:
            st.warning(
                "No legal lineup under the salary cap. Try relaxing the "
                "constraints in the sidebar."
            )
    if rules_dropped:
        st.warning(
            "Your locks or filters cannot be met together with the tested "
            "Showdown rules, so these lineups were built WITHOUT them. Turn "
            "the rules off in the sidebar to silence this, or relax a lock.",
            icon=":material/warning:",
        )
    elif sd_rules:
        st.caption(
            "Tested Showdown rules applied: "
            + ", ".join({"min_qb": "at least one QB",
                         "qb_captain": "QB Captain",
                         "qb_stack": "every QB with his own WR/TE"}[k]
                        for k in sd_rules)
            + ". Switch them off in the sidebar."
        )
    if is_showdown:
        st.caption(
            "**Showdown rules:** 1 Captain + 5 FLEX, any positions, both "
            "teams required. The Captain scores 1.5x and costs 1.5x, so the "
            "highest projection is often not worth the surcharge."
        )
    if not kickers.empty:
        st.caption(
            "Kickers are eligible here and the model projects them properly: "
            "over five seasons of Showdown slates it called them at 6.16 "
            "against an actual 6.49, rank correlation 0.53. On this slate — "
            + ", ".join(
                f"{r.player_name} ({r.team}, ${r.salary:,.0f})"
                for r in kickers.itertuples()
            ) + "."
        )
    order = {"QB": 0, "RB": 1, "WR": 2, "TE": 3, "DST": 4}
    for lu in built:
        if is_showdown:
            lu = lu.sort_values("slot")  # CPT before FLEX
            cols = ["slot", "player_name", "position", "team",
                    "lineup_salary", "lineup_points"]
            names = {"slot": "Slot", "player_name": "Player",
                     "position": "Pos", "team": "Team",
                     "lineup_salary": "Salary", "lineup_points": "Proj"}
            spend, points = lu["lineup_salary"].sum(), lu["lineup_points"].sum()
        else:
            lu = lu.sort_values("position", key=lambda s: s.map(order))
            # Name the three views rather than reprinting the optimised one
            # under a second heading called "Proj". That column was always a
            # copy of Floor, Expected or Ceiling depending on the contest --
            # invisible in cash mode, and in GPP it printed every ceiling
            # twice.
            cols = ["position", "player_name", "team", "salary",
                    "proj_floor", "proj_mean", "proj_ceiling"]
            names = {"position": "Pos", "player_name": "Player",
                     "team": "Team", "salary": "Salary",
                     "proj_floor": "Floor", "proj_mean": "Expected",
                     "proj_ceiling": "Ceiling"}
            spend, points = lu["salary"].sum(), lu["projection"].sum()

        st.subheader(f"Lineup {int(lu['lineup'].iloc[0]) + 1}")
        a, b = st.columns([3, 1])
        a.dataframe(
            lu[cols].rename(columns=names),
            hide_index=True, width="stretch",
            column_config={
                "Salary": st.column_config.NumberColumn(format="$%d"),
                "Floor": st.column_config.NumberColumn(format="%.1f"),
                "Expected": st.column_config.NumberColumn(format="%.1f"),
                "Ceiling": st.column_config.NumberColumn(format="%.1f"),
            },
        )
        _view = CONTESTS[contest]["view"]
        if not is_showdown:
            a.caption(
                f"**{_view}** is the column this contest optimises. The other "
                "two are shown so you can see the shape of each player's "
                "range, not just the number driving the lineup."
            )
        b.metric(f"{_view} total" if not is_showdown else "Projected",
                 f"{points:.1f}")
        if not is_showdown and _view != "Expected" and "proj_mean" in lu.columns:
            b.metric("Expected total", f"{lu['proj_mean'].sum():.1f}",
                     help="What this lineup averages over many runs.")
        b.metric("Salary", f"${spend:,.0f}",
                 delta=f"${spend - SALARY_CAP:,.0f}")
        if contest == "GPP / tournament":
            b.caption(
                "This total adds nine individual 85th-percentile outcomes, "
                "which is **not** the lineup's 85th percentile -- all nine "
                "would have to have a top-seven-games day at once. Over five "
                "seasons a lineup built this way averaged 128 actual points "
                "against a sum like this one. Use it to rank lineups, not to "
                "predict a score."
            )
        if contest == "Cash / double-up":
            b.caption(
                "This total adds nine individual 30th-percentile outcomes, "
                "which is **not** the lineup's 30th percentile -- nine "
                "players rarely all hit their floor at once, so the real "
                "lineup floor is higher. The **Lineup value** tab computes "
                "the lineup's own range instead of adding up the players'."
            )
    if n_lineups > 1 and built:
        st.subheader("Exposure")
        exp = (pd.concat(built)["player_name"].value_counts() / len(built))
        st.dataframe(
            exp.head(20).rename("share of lineups").to_frame().style.format("{:.0%}"),
            width='stretch',
        )

with tab_sim:
    st.caption(
        "The **Proj** number on the Lineups tab adds up nine separate "
        "projections. That is the middle of the range and says nothing about "
        "the range itself -- and a double-up is won or lost entirely on the "
        "range. This tab plays the slate out a few thousand times instead, "
        "with the players who move together moving together, and reads the "
        "answer off what happened."
    )

    _qcols = model.quantile_columns(cur)
    sim_pool = cur.dropna(subset=_qcols + ["p_play", "salary"]).copy()
    # The field drafts from the whole slate on expected points, not on the
    # floor or ceiling view this contest happens to be optimising.
    sim_pool["projection"] = sim_pool["proj_mean"]
    sim_pool = sim_pool[sim_pool["salary"] > 0].reset_index(drop=True)

    ctl1, ctl2 = st.columns(2)
    n_candidates = ctl1.slider(
        "Alternative lineups to compare", 3, 24, 10,
        help="Builds this many near-optimal lineups and values every one of "
             "them in the same simulated seasons.",
    )
    _default, _source = default_cash_line(int(row["n_games"]), is_showdown)
    cash_line = ctl2.number_input(
        "Cash line to beat", min_value=60, max_value=220,
        value=_default, step=1,
        help="The score the median double-up entry posts. Taken from lines "
             "you have actually seen where any are recorded, and from an "
             "estimate otherwise. The simulated field is NOT used for it -- "
             "it medians near 107 on a main slate against a first observed "
             "line of 147.",
    )
    ctl2.caption(
        f"Default {_default}: {_source}. Record what a contest really cashed "
        "for in `data/observed_cash_lines.csv` and this moves with it."
    )

    if len(_qcols) < 3 or len(sim_pool) < 20:
        st.info("Not enough of the slate is projected to simulate it.")
    else:
        sim, opp = simulate_slate(sim_pool, is_showdown)
        cands = build_lineups(pool, is_showdown, n_candidates, 2, stack_n,
                              min_salary, max_exposure, locks, bans,
                              back_n, captain_id,
                              None if rules_dropped else sd_rules)
        if not cands:
            st.warning("No legal lineup to simulate under these constraints.")
        else:
            rows = []
            totals = {}
            for i, lu in enumerate(cands):
                ids, captain = simulate.lineup_ids(lu)
                t = sim.lineup_totals(ids, captain)
                totals[i] = t
                v = simulate.value(t, opp, target_line=cash_line)
                rows.append({
                    "Lineup": i + 1,
                    "Proj": float(lu.get("lineup_points", lu["projection"]).sum()),
                    "Typical": v["p50"], "Bad day": v["p10"],
                    "Great day": v["p90"],
                    "Beats cash line": v["p_cash"],
                    "Top 1%": v["p_top1"],
                    "Players": ", ".join(lu["player_name"].head(3)) + " ...",
                })
            table = pd.DataFrame(rows)

            metric = ("Beats cash line" if contest == "Cash / double-up"
                      else "Top 1%" if contest == "GPP / tournament"
                      else "Typical")
            best = int(table[metric].idxmax())
            ref = simulate.value(totals[0], opp, target_line=cash_line)
            line, sim_line = ref["cash_line"], ref["simulated_line"]

            a, b = st.columns(2)
            a.metric("Cash line in use", f"{line:.0f}",
                     delta=f"{line - sim_line:+.0f} vs simulated {sim_line:.0f}",
                     delta_color="off",
                     help="Probabilities below are measured against this. The "
                          "simulated field's own median is shown for "
                          "comparison and is known to be too low.")
            b.metric("Spread across these candidates",
                     f"{table[metric].min():.0%} - {table[metric].max():.0%}"
                     if metric != "Typical"
                     else f"{table[metric].min():.0f} - "
                          f"{table[metric].max():.0f}")

            # Two errors used to partly cancel: a field that drafts too
            # softly, and -- early in the season -- a simulation whose whole
            # scoring level is compressed because the participation model is
            # unsure who plays. Fixing only the first exposes the second.
            SIM_GAP = 25
            if line - sim_line > SIM_GAP:
                st.warning(
                    f"**These percentages are not meaningful in absolute "
                    f"terms on this slate.** The simulated field medians "
                    f"{sim_line:.0f} against the {line:.0f} you have set, a "
                    f"gap of {line - sim_line:.0f} points. Part of that is "
                    "the field model being too soft, which the control fixes. "
                    "The rest is this slate's own scoring level: early in the "
                    "season the model is unsure who plays, so every "
                    "projection is pulled toward the average and the whole "
                    "simulation sits low. Holding a realistic line against a "
                    "compressed simulation makes every lineup look hopeless.\n\n"
                    "Compare the candidates against **each other** here, or "
                    "drop the line toward the simulated figure to read "
                    "percentages that mean something on this slate."
                )

            st.info(
                "**Read this for the range, not for which lineup to play.** "
                "Ranking these candidates by simulated probability was tested "
                "against the optimizer's own pick over 36 held-out weeks: it "
                "won 18 and lost 16, a coin flip (p = 0.75), and finished "
                "1.3 percentage points LOWER against the real field on "
                "average. The candidates are near-identical by construction, "
                "and the differences between their simulated probabilities "
                "are smaller than the simulation can resolve.\n\n"
                "What the simulation is good for is the number the Lineups "
                "tab cannot give you: how wide this lineup's range actually "
                "is. That part is calibrated -- over the same 36 weeks the "
                "real result landed inside the 10th-to-90th band 79.4% of "
                "the time, against a target of 80%.\n\n"
                "That holds for cash lineups. Stacked tournament lineups come "
                "out too WIDE -- 87.3% against the same 80% target -- so a "
                "GPP range here is conservative and its Top 1% number is if "
                "anything overstated."
            )

            # x100 for display only: st.column_config formats are literal
            # printf, so a fraction passed to "%.0f%%" prints 0.45 as "0%"
            # (docs/MODELING.md 26). The metrics above still read fractions.
            shown = table.drop(columns=["Players"]).copy()
            shown[["Beats cash line", "Top 1%"]] *= 100.0
            st.dataframe(
                shown,
                hide_index=True, width="stretch",
                column_config={
                    "Proj": st.column_config.NumberColumn(
                        format="%.1f",
                        help="Sum of the nine individual projections the "
                             "optimizer maximised. Not a predicted score -- "
                             "the simulated Typical column is that."),
                    "Typical": st.column_config.NumberColumn(format="%.1f"),
                    "Bad day": st.column_config.NumberColumn(
                        format="%.1f",
                        help="One week in ten goes worse than this."),
                    "Great day": st.column_config.NumberColumn(
                        format="%.1f",
                        help="One week in ten goes better than this."),
                    "Beats cash line": st.column_config.NumberColumn(
                        format="%.0f%%",
                        help="Share of simulated weeks this lineup clears "
                             "the cash line set above. Break-even on a 1.8x "
                             "double-up is 55.6%."),
                    "Top 1%": st.column_config.NumberColumn(format="%.1f%%"),
                },
            )
            st.caption(
                "**Bad day** and **Great day** are the lineup's own 10th and "
                "90th percentiles, not the sum of nine players' floors and "
                "ceilings. Nine players rarely bust at once, so the real "
                "floor is far higher than the added-up one -- which is why "
                "the Lineups tab warns you not to read its Proj total as a "
                "score."
            )

            # Against the fields that were actually counted, rather than the
            # modelled one. Raw exports are not committed, so these are rebuilt
            # from the percentiles in data/contest_fields.csv.
            _recorded = real_fields(is_showdown)
            if not _recorded.empty:
                st.markdown("##### Against contests that really happened")
                st.caption(
                    "The percentages above come from a simulated field, which "
                    "was measured 16.8 points too low and 1.26x too narrow "
                    "against the first real contest exported. These rows rank "
                    "Lineup 1 against fields that were counted. **Note how "
                    "much the scoring environment moves them** -- that swing "
                    "is larger than the difference between any two lineups "
                    "here, and it is the real uncertainty."
                )
                _rows = []
                for _, fr in _recorded.iterrows():
                    _f = contest_ev.field_from_summary(fr)
                    _paid = float(fr["paid_positions"]) / float(fr["entries"])
                    _pct = contest_ev.percentile(totals[0], _f)
                    # x100: st.column_config formats are literal printf, so a
                    # fraction passed to "%.1f%%" prints 0.254 as "0.3%".
                    _rows.append({
                        "Contest": str(fr["contest_id"]),
                        "Entries": int(fr["entries"]),
                        "Pays": 100.0 * _paid,
                        "Field mean": float(fr["mean"]),
                        "Our median finish": 100.0 * float(np.median(_pct)),
                        "Cash": 100.0 * float((_pct <= _paid).mean()),
                        "Top 1%": 100.0 * float((_pct <= 0.01).mean()),
                    })
                st.dataframe(
                    pd.DataFrame(_rows), hide_index=True, width="stretch",
                    column_config={
                        "Pays": st.column_config.NumberColumn(format="%.1f%%"),
                        "Field mean": st.column_config.NumberColumn(format="%.1f"),
                        "Our median finish": st.column_config.NumberColumn(
                            format="%.1f%%",
                            help="Percentile of the field. 50% is an average "
                                 "entry; lower is better."),
                        "Cash": st.column_config.NumberColumn(format="%.1f%%"),
                        "Top 1%": st.column_config.NumberColumn(format="%.2f%%"),
                    },
                )
                st.caption(
                    "No expected-value column, deliberately. Ranking a "
                    "simulated lineup against a frozen field hands it first "
                    "place every time its tail runs past the highest score "
                    "that field posted -- on a real $20K contest that was 62% "
                    "of the computed EV, turning an average lineup into a "
                    "claimed 160% edge. `contest_ev.expected_value` reports "
                    "that share and refuses to call itself trustworthy. "
                    "Percentile and cash probability do not depend on the "
                    "tail, so they are what is shown."
                )

            show = st.selectbox(
                "Show the distribution for", [f"Lineup {i + 1}" for i in
                                              range(len(cands))],
                index=0,
            )
            pick = int(show.split()[-1]) - 1
            dist = pd.DataFrame({"score": totals[pick]})
            chart = (
                alt.Chart(dist)
                .mark_bar(opacity=0.8)
                .encode(
                    alt.X("score:Q", bin=alt.Bin(maxbins=60),
                          title="lineup score"),
                    alt.Y("count()", title="simulated weeks"),
                )
            )
            rule = (
                alt.Chart(pd.DataFrame({"line": [line]}))
                .mark_rule(color="red", strokeDash=[6, 4], size=2)
                .encode(x="line:Q")
            )
            st.altair_chart(chart + rule, use_container_width=True)
            st.caption(
                "Red is the estimated cash line. Everything to the right of "
                "it is a week you double up."
            )
            st.info(
                "**What is assumed here.** Player distributions come from the "
                "same model as the projections, so they inherit its blind "
                "spots. Correlations are measured from five seasons of "
                "walk-forward residuals.\n\n"
                "**The cash line is the weakest link.** No contest result has "
                "ever been observed here -- they are paid data -- so the "
                "simulated field is all the project can build, and it is too "
                "soft: it samples players and repairs over-cap lineups by "
                "swapping down to the cheapest option, which drafts worse "
                "than real entrants do. Its median sits near 107 on a main "
                "slate where a real double-up cashes closer to 140. The "
                "control above sets the level; the simulation still moves the "
                "bar week to week around it, so a slate where the chalk busts "
                "lowers the line instead of leaving it stranded."
            )


with tab_players:
    pos = st.multiselect(
        "Position", ["QB", "RB", "WR", "TE", "DST"],
        default=["QB", "RB", "WR", "TE", "DST"],
    )
    st.caption(
        f"Sorted by **{CONTESTS[contest]['view']}**, the view this contest "
        f"optimizes (*{contest}*). All three are shown so you can see the "
        "shape of a player's range rather than one number from it. "
        "**Pts/$1k** is value against that same view."
    )
    view = cur[cur["position"].isin(pos)].nlargest(300, "projection")
    st.dataframe(
        view[["player_name", "position", "team", "salary",
              "proj_floor", "proj_mean", "proj_ceiling", "p_play",
              "pts_per_1k"]].rename(columns={
                  "player_name": "Player", "position": "Pos", "team": "Team",
                  "salary": "Salary",
                  "proj_floor": "Floor", "proj_mean": "Expected",
                  "proj_ceiling": "Ceiling", "p_play": "Plays",
                  "pts_per_1k": "Pts/$1k"}),
        hide_index=True, width="stretch", height=600,
        column_config={
            "Salary": st.column_config.NumberColumn(format="$%d"),
            "Floor": st.column_config.NumberColumn(format="%.1f"),
            "Expected": st.column_config.NumberColumn(format="%.1f"),
            "Ceiling": st.column_config.NumberColumn(format="%.1f"),
            "Plays": st.column_config.NumberColumn(format="%.0f%%"),
            "Pts/$1k": st.column_config.NumberColumn(format="%.2f"),
        },
    )

with tab_adjust:
    st.caption(
        f"Type over the **{CONTESTS[contest]['view']}** projection for any "
        "player. The whole distribution moves with it -- floor, ceiling and "
        "the quantile grid the simulator reads -- so the Lineups and Lineup "
        "value tabs agree about the player. The chance he plays is left alone; "
        "to force in someone the model does not rate, lock him instead."
    )
    st.caption(
        "Every edit is written to `data/overrides/` with the model's own "
        "number beside it. That log is the point: whether these lineups beat "
        "a hand-built one needs dozens of weeks to settle, but one player "
        "against one actual score is a small enough unit that a few weeks of "
        "edits can say whether they carry signal the model lacks."
    )

    _adj_pos = st.multiselect(
        "Position", ["QB", "RB", "WR", "TE", "DST", "K"],
        default=["QB", "RB", "WR", "TE", "DST", "K"], key="adjust_pos",
    )
    _in_force = _edits[_edits["view"] == _view_col] if not _edits.empty         else _edits
    _forced_ids = set(_in_force["player_dk_id"].astype(int)) if len(_in_force)         else set()

    _base = cur[cur["position"].isin(_adj_pos)].copy()
    # Show what the model said, not the already-overridden number, or a second
    # edit would compound against the first and the ratio would drift.
    _model_now = dict(zip(_in_force.get("player_dk_id", pd.Series(dtype=int)
                                       ).astype(int),
                          _in_force.get("model_value", pd.Series(dtype=float))))
    _base["_model"] = [
        _model_now.get(int(i), v)
        for i, v in zip(_base["player_dk_id"], _base[_view_col])
    ]
    _base = _base.nlargest(200, "_model")
    # Pre-filled with the number in force rather than left blank. Streamlit
    # renders a NaN in a number column as the word "None" in every unedited
    # row, and the semantics are clearer this way too: the column always shows
    # what the optimizer is using, and setting it back to Model removes the
    # override.
    _grid = pd.DataFrame({
        "Player": _base["player_name"].to_numpy(),
        "Pos": _base["position"].to_numpy(),
        "Team": _base["team"].to_numpy(),
        "Salary": _base["salary"].to_numpy(),
        "Model": _base["_model"].to_numpy(dtype=float).round(2),
        "Yours": _base[_view_col].to_numpy(dtype=float).round(2),
    })

    _done = st.data_editor(
        _grid, hide_index=True, width="stretch", height=520,
        key=f"adjust_{dgid}_{_view_col}",
        disabled=["Player", "Pos", "Team", "Salary", "Model"],
        column_config={
            "Salary": st.column_config.NumberColumn(format="$%d"),
            "Model": st.column_config.NumberColumn(
                format="%.2f", help="What the model projects, unedited."),
            "Yours": st.column_config.NumberColumn(
                format="%.2f", min_value=0.0, max_value=80.0,
                help="What the optimizer uses. Set it back to Model to drop "
                     "the override."),
        },
    )

    _c1, _c2, _c3 = st.columns([1, 1, 3])
    if _c1.button("Apply edits", type="primary"):
        _ids = dict(zip(_base["player_name"], _base["player_dk_id"]))
        _d = _done.dropna(subset=["Yours", "Model"]).copy()
        _d["pid"] = [int(_ids[n]) for n in _d["Player"]]
        _moved = (_d["Yours"] - _d["Model"]).abs() > 0.005

        overrides.record(dgid, pd.DataFrame({
            "player_dk_id": _d.loc[_moved, "pid"].to_numpy(),
            "player_name": _d.loc[_moved, "Player"].to_numpy(),
            "view": _view_col,
            "model_value": _d.loc[_moved, "Model"].to_numpy(dtype=float),
            "user_value": _d.loc[_moved, "Yours"].to_numpy(dtype=float),
        }))
        # Set back to the model's own number means "drop it". The row stays in
        # the file's history; only the override in force is removed.
        _gone = sorted(_forced_ids & set(_d.loc[~_moved, "pid"]))
        if _gone:
            overrides.forget(dgid, _gone, _view_col)
        st.rerun()

    if _c2.button("Clear all") and _forced_ids:
        overrides.forget(dgid, sorted(_forced_ids), _view_col)
        st.rerun()

    if _n_edits:
        _c3.caption(f"**{_n_edits}** override(s) in force on this view. "
                    "Switching contest type switches which view you are "
                    "editing, because a floor and a ceiling are different "
                    "claims about the same player.")
    else:
        _c3.caption("No overrides on this view yet.")

    if not _in_force.empty:
        _show = _in_force[["player_name", "model_value", "user_value",
                           "recorded_at"]].copy()
        _show["change"] = (_show["user_value"] - _show["model_value"]).round(2)
        st.dataframe(
            _show.rename(columns={"player_name": "Player",
                                  "model_value": "Model",
                                  "user_value": "Yours",
                                  "recorded_at": "Entered",
                                  "change": "Change"}),
            hide_index=True, width="stretch",
        )

with tab_why:
    # This tab explains OUR model and cannot explain anyone else's: the
    # attribution is over our features, and DFF publish a number without
    # them. Applying the override here would put their projection next to
    # our arithmetic and imply the second produced the first.
    if _src != "Model":
        st.info(
            f"**Projections are set to {_src}, but this tab always explains "
            "the Model.** The breakdown below is our own feature "
            "attribution; DFF publish a number and not the reasoning behind "
            "it, so there is nothing of theirs to take apart. Where the two "
            "differ, both are shown.",
            icon=":material/info:",
        )
    summary, contrib = explanation(dgid, is_showdown)
    if summary.empty:
        st.info("Not enough history before this slate to explain projections.")
    else:
        merged = cur[["player_dk_id", "player_name", "position", "team",
                      "salary"]].merge(summary, on="player_dk_id", how="inner")
        merged = merged.sort_values("projection", ascending=False)
        labels = merged.apply(
            lambda r: f"{r['player_name']} ({r['position']}, {r['team']}) "
                      f"- {r['projection']:.1f}", axis=1
        )
        pick = st.selectbox("Player", labels, index=0)
        chosen = merged.iloc[list(labels).index(pick)]
        pid = int(chosen["player_dk_id"])

        _live = cur.loc[cur["player_dk_id"] == pid, "proj_mean"]
        _live = float(_live.iloc[0]) if len(_live) else float("nan")
        _delta = None
        if _src != "Model" and np.isfinite(_live):
            _delta = f"{_live - chosen['projection']:+.2f} in use ({_src})"

        a, b, c, d = st.columns(4)
        a.metric("Model projection", f"{chosen['projection']:.2f}",
                 delta=_delta, delta_color="off")
        b.metric("Chance he plays", f"{chosen['p_play']:.0%}")
        c.metric("Points if he plays", f"{chosen['points_if_plays']:.2f}")
        d.metric("Salary", f"${chosen['salary']:,.0f}")
        st.caption(
            "The projection is **chance he plays x points if he plays**. A 4.0 "
            "means something very different at 25% than at 95%, and the "
            "product alone hides which one you are looking at."
        )

        top_c = model.top_contributions(contrib, pid, top=12)
        if not top_c.empty:
            base = float(contrib.loc[contrib["player_dk_id"] == pid,
                                     "_baseline"].iloc[0])
            st.subheader("What moved the points-if-he-plays number")
            st.caption(
                f"Starting from the model's baseline of {base:.2f} points, "
                "these features added or removed the most. This is the actual "
                "arithmetic behind this one player, not a general ranking."
            )
            bar = (
                alt.Chart(top_c)
                .mark_bar()
                .encode(
                    x=alt.X("points:Q", title="Points added / removed"),
                    y=alt.Y("feature:N", sort="-x", title=None),
                    color=alt.condition(
                        alt.datum.points > 0,
                        alt.value("#2b8a3e"), alt.value("#c92a2a"),
                    ),
                    tooltip=["feature", alt.Tooltip("points:Q", format="+.2f")],
                )
                .properties(height=340)
            )
            st.altair_chart(bar, width="stretch")

        raw = cur[cur["player_dk_id"] == pid]
        if not raw.empty:
            st.subheader("His underlying numbers")
            groups = {
                "Recent form": ["dk_points_r3", "dk_points_r6", "prior_ppg",
                                "prior_best"],
                "Volume": ["offense_pct_r3", "targets_r3", "carries_r3",
                           "target_share_r3", "pbp_air_yards_r3"],
                "Red zone": ["rz_targets_r3", "rz_carries_r3", "gl_carries_r3"],
                "Matchup & game": ["implied_total", "opp_implied_total",
                                   "spread", "game_total", "def_allowed_r3",
                                   "is_home", "temp", "wind"],
                "Availability": ["injury_prob", "depth_team", "is_starter"],
            }
            cols_ = st.columns(len(groups))
            for (title, keys), col in zip(groups.items(), cols_):
                have = [k for k in keys if k in raw.columns]
                if not have:
                    continue
                vals = raw[have].iloc[0]
                col.caption(f"**{title}**")
                col.dataframe(
                    pd.DataFrame({"": vals.index, " ": vals.to_numpy()}),
                    hide_index=True, width="stretch",
                    column_config={" ": st.column_config.NumberColumn(
                        format="%.2f")},
                )
            st.caption(
                "`_r3` and `_r6` are 3- and 6-week rolling averages, and they "
                "exclude this week by construction. Blank means no history -- "
                "a rookie, a Week 1 slate, or someone back from a long absence."
            )

with tab_slate:
    need = {"team", "opponent", "implied_total", "game_total", "spread"}
    if not need <= set(cur.columns):
        st.info("No Vegas data for this slate yet. Lines usually appear a few "
                "days out.")
    else:
        st.caption(
            "Implied team total is the single strongest input the model has. "
            "It is the market's estimate of how many points a team will "
            "score, derived from the game total and the spread -- so a team "
            "favoured by 7 in a 48-point game is expected to score 27.5."
        )
        games = (
            cur.groupby("team")
            .agg(opponent=("opponent", "first"),
                 implied=("implied_total", "first"),
                 opp_implied=("opp_implied_total", "first"),
                 total=("game_total", "first"),
                 spread=("spread", "first"),
                 is_home=("is_home", "first"),
                 temp=("temp", "first"),
                 wind=("wind", "first"),
                 dome=("is_dome", "first"),
                 priced=("player_dk_id", "size"),
                 best=("projection", "max"))
            .reset_index()
            .dropna(subset=["implied"])
        )
        if built:
            used = pd.concat(built)["team"].value_counts()
            games["in_lineups"] = games["team"].map(used).fillna(0).astype(int)
        games = games.sort_values("implied", ascending=False)
        games["book_spread"] = -games["spread"]

        st.subheader("Teams by implied total")
        bar = (
            alt.Chart(games)
            .mark_bar()
            .encode(
                x=alt.X("implied:Q", title="Implied team total",
                        scale=alt.Scale(zero=False)),
                y=alt.Y("team:N", sort="-x", title=None),
                color=alt.Color("implied:Q", legend=None,
                                scale=alt.Scale(scheme="blues")),
                tooltip=["team", "opponent",
                         alt.Tooltip("implied:Q", format=".1f"),
                         alt.Tooltip("total:Q", format=".1f"),
                         alt.Tooltip("book_spread:Q", title="spread",
                                     format="+.1f")],
            )
            .properties(height=max(260, 16 * len(games)))
        )
        st.altair_chart(bar, width="stretch")

        show = games.copy()
        # Internally `spread` is positive when a team is favoured, which is
        # the opposite of how a sportsbook prints it. Negate for display only:
        # flipping the feature itself would be a monotonic transform the trees
        # are indifferent to, but it would invalidate the stored model.
        show["spread"] = -show["spread"]
        show["weather"] = show.apply(
            lambda r: "indoors" if r["dome"] == 1.0
            else (f"{r['temp']:.0f}F, {r['wind']:.0f}mph wind"
                  if pd.notna(r["temp"]) else "-"),
            axis=1,
        )
        cols = ["team", "opponent", "implied", "opp_implied", "total",
                "spread", "weather", "priced", "best"]
        if "in_lineups" in show.columns:
            cols.append("in_lineups")
        st.dataframe(
            show[cols].rename(columns={
                "team": "Team", "opponent": "Opp", "implied": "Implied",
                "opp_implied": "Opp implied", "total": "Game total",
                "spread": "Spread", "weather": "Weather",
                "priced": "Priced", "best": "Best proj",
                "in_lineups": "In your lineups"}),
            hide_index=True, width="stretch", height=560,
            column_config={
                "Implied": st.column_config.NumberColumn(format="%.1f"),
                "Opp implied": st.column_config.NumberColumn(format="%.1f"),
                "Game total": st.column_config.NumberColumn(format="%.1f"),
                "Spread": st.column_config.NumberColumn(format="%+.1f"),
                "Best proj": st.column_config.NumberColumn(format="%.1f"),
            },
        )
        st.caption(
            "Spread is shown the way a sportsbook prints it: **negative means "
            "favoured**, so SF -12.5 is a twelve-and-a-half point favourite. "
            "Heavy favourites tend to run the ball late, which helps their "
            "backs and hurts their receivers -- the sort of game-script "
            "effect the model cannot read out of a box score, and worth "
            "checking before you trust a projection."
        )

with tab_chart:
    st.caption(
        "Players above the trend line are priced below what the model expects "
        "them to score."
    )
    plot = cur[cur["projection"] > 0]
    base = alt.Chart(plot).encode(
        x=alt.X("salary:Q", title="Salary", scale=alt.Scale(zero=False)),
        y=alt.Y("projection:Q", title="Projected DK points"),
    )
    points = base.mark_circle(size=70, opacity=0.65).encode(
        color=alt.Color("position:N", title="Position"),
        tooltip=["player_name", "position", "team", "salary",
                 alt.Tooltip("projection:Q", format=".1f"),
                 alt.Tooltip("pts_per_1k:Q", format=".2f")],
    )
    # Built from `base`, not from `points`: a regression layered on a chart
    # that already carries the colour encoding inherits that scale with no
    # value of its own and renders a phantom "undefined" legend entry.
    trend = base.transform_regression("salary", "projection").mark_line(
        color="grey", strokeDash=[4, 4]
    )
    st.altair_chart((points + trend).interactive(), width="stretch")

st.divider()
st.caption(
    "Projections are conditional on playing and weighted by the chance a "
    "player suits up. Salaries and injury news move up to kickoff -- re-run "
    "after inactives land, 90 minutes before the first game."
)

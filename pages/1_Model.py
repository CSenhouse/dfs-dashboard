"""Model diagnostics: how well it does, and where it goes wrong.

Everything here is precomputed by scripts/diagnostics.py -- a walk-forward run
takes about twenty minutes and cannot happen on demand.

The section that earns its place is "Weeks it got wrong". Aggregate metrics
tell you how good the model is on average; that section shows the lineup it
built next to the lineup that was actually available, which is something a
football person can look at and diagnose.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dfs.config import CURATED  # noqa: E402

st.set_page_config(page_title="Model diagnostics", page_icon="📊",
                   layout="wide")
st.title("📊 Model diagnostics")


@st.cache_data(show_spinner=False)
def load(name: str) -> pd.DataFrame:
    path = CURATED / f"{name}.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def metrics() -> dict:
    path = CURATED / "metrics.json"
    return json.loads(path.read_text()) if path.exists() else {}


weekly = load("weekly")
if weekly.empty:
    st.warning(
        "No diagnostics yet. Generate them with:\n\n"
        "`PYTHONPATH=src .venv/Scripts/python.exe scripts/diagnostics.py`\n\n"
        "It takes roughly twenty minutes -- three walk-forward runs over five "
        "seasons."
    )
    st.stop()

m = metrics()
two = weekly[weekly["model"] == "two_stage"]
base = weekly[weekly["model"] == "baseline"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Weeks tested", int(two["week"].count()))
c2.metric("Rank correlation", f"{two['spearman'].mean():.3f}",
          delta=f"{two['spearman'].mean() - base['spearman'].mean():+.3f} vs baseline")
c3.metric("Lineup average", f"{two['actual'].mean():.1f}",
          delta=f"{two['actual'].mean() - base['actual'].mean():+.1f} vs baseline")
c4.metric("Share of the best possible", f"{two['pct_of_ceiling'].mean():.0%}")

st.caption(
    "Walk-forward: every week is predicted using only earlier weeks. "
    "'Best possible' is the highest-scoring legal lineup that week in "
    "hindsight -- nobody reaches it, it is there for scale."
)

tab_week, tab_cal, tab_wrong, tab_miss, tab_feat = st.tabs(
    ["Week by week", "Calibration", "Weeks it got wrong",
     "Biggest misses", "What it uses"]
)

with tab_week:
    st.subheader("Lineup score, week by week")
    plot = weekly.dropna(subset=["actual"]).copy()
    plot["label"] = (plot["season"].astype(int).astype(str) + " W"
                     + plot["week"].astype(int).astype(str))
    plot["order"] = plot["season"] * 100 + plot["week"]
    chart = (
        alt.Chart(plot)
        .mark_line(point=True, opacity=0.85)
        .encode(
            x=alt.X("order:Q", title="Season / week", axis=None),
            y=alt.Y("actual:Q", title="Lineup points",
                    scale=alt.Scale(zero=False)),
            color=alt.Color("model:N", title="Model"),
            tooltip=["label", "model", alt.Tooltip("actual:Q", format=".1f"),
                     alt.Tooltip("ceiling:Q", format=".1f")],
        )
        .properties(height=340)
        .interactive()
    )
    st.altair_chart(chart, width="stretch")
    st.caption(
        "The spread here is the point: a single week swings about 25 points "
        "either way, which is why a model is never judged on a handful of "
        "weeks."
    )

    st.subheader("Rank correlation, week by week")
    sp = (
        alt.Chart(weekly.dropna(subset=["spearman"]).assign(
            order=lambda d: d["season"] * 100 + d["week"]))
        .mark_line(point=False)
        .encode(
            x=alt.X("order:Q", title="Season / week", axis=None),
            y=alt.Y("spearman:Q", title="Spearman",
                    scale=alt.Scale(zero=False)),
            color=alt.Color("model:N", title="Model"),
        )
        .properties(height=240)
    )
    st.altair_chart(sp, width="stretch")
    st.caption(
        "Far steadier than lineup score, which is why model choices are made "
        "on this and not on points."
    )

with tab_cal:
    cal = load("calibration")
    if cal.empty:
        st.info("No calibration data.")
    else:
        st.subheader("Predicted against actual, by projection decile")
        melted = cal.melt(
            id_vars=["bucket", "n"], value_vars=["predicted", "actual"],
            var_name="series", value_name="points",
        )
        ch = (
            alt.Chart(melted)
            .mark_line(point=True)
            .encode(
                x=alt.X("bucket:O", title="Projection decile (9 = highest)"),
                y=alt.Y("points:Q", title="Mean DK points"),
                color=alt.Color("series:N", title=None),
                tooltip=["bucket", "series",
                         alt.Tooltip("points:Q", format=".2f")],
            )
            .properties(height=320)
        )
        st.altair_chart(ch, width="stretch")

        top = cal.iloc[-1]
        gap_pct = top["gap"] / top["predicted"] if top["predicted"] else 0
        st.metric("Top-decile over-projection", f"{top['gap']:+.2f} pts",
                  delta=f"{gap_pct:+.1%}", delta_color="inverse")
        st.caption(
            "Nine of ten deciles land almost exactly on the line. The tenth "
            "runs high -- and it is the only bucket the optimizer ever "
            "rosters from, so average bias is the wrong thing to look at. "
            "Correcting it improves the honesty of the projected total but "
            "does not build better lineups (docs/MODELING.md section 5)."
        )
        st.dataframe(cal.round(3), hide_index=True, width="stretch")

with tab_wrong:
    worst = load("worst_weeks")
    if worst.empty:
        st.info("No worst-week detail. Re-run scripts/diagnostics.py.")
    else:
        st.subheader("The weeks it did worst")
        st.caption(
            "Each shows the lineup the model built next to the best lineup "
            "that was available. This is the most useful page here: if you "
            "can see why the right players were missed, that is a feature "
            "the model does not have."
        )
        wk = worst[["season", "week", "lineup_actual", "ceiling"]].drop_duplicates()
        wk["label"] = (wk["season"].astype(int).astype(str) + " Week "
                       + wk["week"].astype(int).astype(str)
                       + f"  —  scored " + wk["lineup_actual"].round(0).astype(int).astype(str)
                       + " of " + wk["ceiling"].round(0).astype(int).astype(str))
        pick = st.selectbox("Week", wk["label"])
        sel = wk[wk["label"] == pick].iloc[0]
        rows = worst[(worst["season"] == sel["season"])
                     & (worst["week"] == sel["week"])]

        a, b = st.columns(2)
        for col, tag in ((a, "picked"), (b, "best available")):
            side = rows[rows["which"] == tag]
            col.caption(f"**{tag.title()}** — {side['dk_scored'].sum():.1f} pts")
            col.dataframe(
                side[["position", "player_name", "team", "salary",
                      "projection", "dk_scored"]]
                .rename(columns={"position": "Pos", "player_name": "Player",
                                 "team": "Team", "salary": "Salary",
                                 "projection": "Proj", "dk_scored": "Actual"}),
                hide_index=True, width="stretch",
                column_config={
                    "Salary": st.column_config.NumberColumn(format="$%d"),
                    "Proj": st.column_config.NumberColumn(format="%.1f"),
                    "Actual": st.column_config.NumberColumn(format="%.1f"),
                },
            )

with tab_miss:
    misses = load("misses")
    if misses.empty:
        st.info("No miss data.")
    else:
        kind = st.radio(
            "Show", ["under-projected", "over-projected"], horizontal=True,
            help="Under-projected: scored far more than expected. "
                 "Over-projected: the busts.",
        )
        sub = misses[misses["kind"] == kind].copy()
        sub = sub.reindex(sub["error"].abs().sort_values(ascending=False).index)
        st.dataframe(
            sub.head(100)[["season", "week", "player_name", "position", "team",
                           "salary", "projection", "dk_scored", "error"]]
            .rename(columns={"season": "Season", "week": "Wk",
                             "player_name": "Player", "position": "Pos",
                             "team": "Team", "salary": "Salary",
                             "projection": "Proj", "dk_scored": "Actual",
                             "error": "Miss"}),
            hide_index=True, width="stretch", height=560,
            column_config={
                "Salary": st.column_config.NumberColumn(format="$%d"),
                "Proj": st.column_config.NumberColumn(format="%.1f"),
                "Actual": st.column_config.NumberColumn(format="%.1f"),
                "Miss": st.column_config.NumberColumn(format="%+.1f"),
            },
        )
        st.caption(
            "Touchdowns are most of what shows up here. They are worth six "
            "points, largely unpredictable week to week, and no projection "
            "will ever call them reliably -- which is why the model targets "
            "opportunity rather than scoring."
        )

with tab_feat:
    imp = load("importance")
    if imp.empty:
        st.info("No feature importance data.")
    else:
        st.subheader("What the model leans on")
        top = imp.head(25).copy()
        top["share"] = top["gain"] / imp["gain"].sum()
        ch = (
            alt.Chart(top)
            .mark_bar()
            .encode(
                x=alt.X("share:Q", title="Share of total gain",
                        axis=alt.Axis(format="%")),
                y=alt.Y("feature:N", sort="-x", title=None),
                tooltip=["feature", alt.Tooltip("share:Q", format=".1%")],
            )
            .properties(height=560)
        )
        st.altair_chart(ch, width="stretch")
        st.caption(
            "Salary dominates, as it should -- DraftKings prices players with "
            "better information than we have, so their number is the single "
            "most informative input available. Recent form, snap share and "
            "Vegas implied totals follow."
        )

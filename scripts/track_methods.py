"""Which projection source builds better lineups? A running, paired record.

For each finished slate, rebuild the lineup each method WOULD have produced
-- the model alone and the 50/50 blend with Daily Fantasy Fuel, in both the
cash (floor) and tournament (ceiling, stack + bring-back) settings, with the
dashboard's own rules -- score every lineup on what actually happened, and
check it against the real cash lines of the contests played that week. The
lineups actually entered are recorded beside them.

Every method sees the same slate, the same rules and the same field, so the
weekly comparison is paired (rule 2). One week is one draw with a ~25-point
spread; read the cumulative table, not a single row.

Reconstruction caveat: features are rebuilt after the fact, so pregame inputs
published late (that week's injury report) are visible here and were not to
the live dashboard. Both methods share that, so it does not favour either.

    PYTHONPATH=src .venv/Scripts/python.exe scripts/track_methods.py 154941
    PYTHONPATH=src .venv/Scripts/python.exe scripts/track_methods.py --report

Writes data/tracking/lineups.csv (one row per player per lineup) and
data/tracking/results.csv (one row per lineup).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from dfs import backtest, contest_ev, contests, features, model, optimizer, \
    overrides, slates
from dfs.config import CURATED, DATA
from dfs.sources import dff, dk_api

OUT = DATA / "tracking"
ME = "senhouse"

VIEWS = {
    "cash": dict(col="proj_floor", stack=0, back=0),
    "gpp": dict(col="proj_ceiling", stack=1, back=1),
}
SHOWDOWN_RULES = {"cash": dict(min_qb=1), "gpp": dict(min_qb=1, qb_stack=1)}
METHODS = {"model": 0.0, "blend": 0.5}


def project(dgid: int) -> tuple[pd.DataFrame, bool, int, int]:
    up = slates.assemble(dgid)
    if up.empty:
        raise SystemExit(f"draft group {dgid} did not assemble")
    season, week = int(up["season"].iloc[0]), int(up["week"].iloc[0])
    showdown = dk_api.classify(dk_api.draftables(dgid, cache_only=True))[
        "kind"] == "showdown"
    name = "showdown_features" if showdown else "features"
    past = backtest.prepare(pd.read_parquet(CURATED / f"{name}.parquet"))
    past = past[(past["season"] < season)
                | ((past["season"] == season) & (past["week"] < week))]
    cur = features.build(up, salary_within_slate=showdown)
    cur = cur[cur["salary"].fillna(0) > 0].reset_index(drop=True)
    cur = cur.merge(model.two_stage_all(past, cur), on="player_dk_id",
                    how="left")
    out = dk_api.unavailable(dk_api.draftables(dgid, cache_only=True))
    cur = cur[~cur["player_dk_id"].isin(out["player_dk_id"])]
    return cur.reset_index(drop=True), showdown, season, week


def blend(cur: pd.DataFrame, dgid: int, col: str, weight: float
          ) -> pd.DataFrame:
    """Exactly the dashboard's Model / Blend control, through overrides."""
    if weight == 0:
        return cur
    teams = tuple(sorted(cur["team"].dropna().astype(str).unique()))
    theirs, why, _ = dff.for_slate(dgid, teams)
    if theirs.empty:
        raise SystemExit(f"no DFF export for {dgid}: {why}")
    m = dff.match(theirs, cur[["player_dk_id", "player_name", "position",
                               "team", "salary"]])
    m = m[m["matched"]]
    base = cur.set_index("player_dk_id")[col]
    mean = cur.set_index("player_dk_id")["proj_mean"]
    rows = []
    for r in m.itertuples():
        pid = int(r.player_dk_id)
        if pid not in base.index or not np.isfinite(base[pid]):
            continue
        scale = base[pid] / max(mean[pid], 1e-6)
        rows.append({"player_dk_id": pid, "player_name": r.player_name,
                     "view": col, "model_value": float(base[pid]),
                     "user_value": float((1 - weight) * base[pid]
                                         + weight * r.dff_proj * scale)})
    return overrides.apply(cur, pd.DataFrame(rows), col)


def build(pool: pd.DataFrame, showdown: bool, kind: str) -> pd.DataFrame:
    v = VIEWS[kind]
    pool = pool.assign(projection=pool[v["col"]])
    pool = pool[pool["projection"].notna()]
    if showdown:
        lus = optimizer.optimize_showdown(pool, objective="projection",
                                          n_lineups=1, **SHOWDOWN_RULES[kind])
    else:
        lus = optimizer.optimize(pool, objective="projection", n_lineups=1,
                                 stack_qb_with=v["stack"],
                                 bring_back=v["back"])
    return lus[0] if lus else pd.DataFrame()


def contests_for(dgid: int) -> pd.DataFrame:
    """The recorded contests on this draft group, with their lines."""
    obs = pd.read_csv(DATA / "observed_cash_lines.csv")
    obs = obs[obs["draft_group"].astype("Int64") == dgid].copy()
    obs["kind"] = np.where(pd.to_numeric(obs["paid_pct"]) > 0.40, "cash", "gpp")
    return obs


def actual_points(dgid: int, obs: pd.DataFrame) -> pd.Series:
    """player_dk_id -> DK points. nflverse once the week is final; until
    then DraftKings' own FPTS from the standings exports of this slate,
    matched on DK's name to DK's player -- one source, so rule 1's
    cross-source failure does not arise; an ambiguous name is dropped."""
    up = slates.assemble(dgid)
    if up["dk_actual"].notna().any():
        return up.set_index("player_dk_id")["dk_actual"].fillna(0.0)
    pts = {}
    for cid in obs["contest_id"].astype(str):
        path = next(iter((DATA / "contests").glob(
            f"contest-standings-{cid}.*")), None)
        if path is None:
            continue
        _, players = contests.load(path)
        for name, v in contests.ownership(players).set_index(
                "player_name")["dk_scored"].items():
            pts[name] = v
    dk = dk_api.draftables(dgid, cache_only=True)
    names = dk["player_name"].where(dk["position"] != "DST", dk["team"])
    # Exports name a DST by nickname ("Panthers"); DK's draftables by player.
    dk = dk.assign(key=dk["player_name"])
    dupes = dk["key"].duplicated(keep=False)
    dk = dk[~dupes]
    return pd.Series({int(r.player_dk_id): pts.get(r.key, np.nan)
                      for r in dk.itertuples()}, dtype=float)


def entered(obs: pd.DataFrame) -> list[tuple[str, str, list[str], float]]:
    """(contest_id, kind, player names, score) for each of our entries."""
    out = []
    for r in obs.itertuples():
        path = next(iter((DATA / "contests").glob(
            f"contest-standings-{r.contest_id}.*")), None)
        if path is None:
            continue
        e, _ = contests.load(path)
        me = e[e["entry_name"].str.lower() == ME]
        if me.empty:
            continue
        names = [n for _, n in contests.parse_lineup(me["lineup"].iloc[0])]
        out.append((str(r.contest_id), r.kind, names,
                    float(me["points"].iloc[0])))
    return out


def judge(score: float, kind: str, obs: pd.DataFrame) -> dict:
    """Against every recorded contest of that kind on the slate."""
    rel = obs[obs["kind"] == kind]
    if rel.empty:
        return {"contests": 0, "cashed": 0, "margin": np.nan}
    lines = pd.to_numeric(rel["line"]).to_numpy()
    return {"contests": len(lines), "cashed": int((score >= lines).sum()),
            "margin": float(score - np.median(lines))}


def record(dgid: int) -> pd.DataFrame:
    cur, showdown, season, week = project(dgid)
    obs = contests_for(dgid)
    if obs.empty:
        print(f"note: no recorded contests on {dgid}; lineups saved, "
              "cash lines unknown")
    pts = actual_points(dgid, obs)
    rows, players = [], []
    for method, weight in METHODS.items():
        for kind, v in VIEWS.items():
            pool = blend(cur, dgid, v["col"], weight)
            lu = build(pool, showdown, kind)
            if lu.empty:
                continue
            mult = np.where(lu.get("slot", pd.Series("", index=lu.index))
                            .eq("CPT"), 1.5, 1.0)
            got = lu["player_dk_id"].map(pts)
            score = float((got.fillna(0) * mult).sum())
            rows.append({"season": season, "week": week,
                         "draft_group_id": dgid, "method": method,
                         "kind": kind, "projected": float(
                             (lu[v["col"]] * mult).sum()),
                         "score": score,
                         "unscored": int(got.isna().sum()),
                         **judge(score, kind, obs)})
            players.append(lu.assign(method=method, kind=kind,
                                     actual=got.to_numpy())[
                ["player_dk_id", "player_name", "position", "team",
                 "salary", v["col"], "actual", "method", "kind"]]
                .rename(columns={v["col"]: "projection"}))
    for cid, kind, names, score in entered(obs):
        rows.append({"season": season, "week": week, "draft_group_id": dgid,
                     "method": f"entered ({cid})", "kind": kind,
                     "projected": np.nan, "score": score, "unscored": 0,
                     **judge(score, kind, obs)})
    res = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, df in (("results", res), ("lineups", pd.concat(players).assign(
            season=season, week=week, draft_group_id=dgid))):
        path = OUT / f"{name}.csv"
        if path.exists():
            old = pd.read_csv(path)
            old = old[old["draft_group_id"] != dgid]
            df = pd.concat([old, df], ignore_index=True)
        df.to_csv(path, index=False)
    return res


def report() -> None:
    path = OUT / "results.csv"
    if not path.exists():
        print("nothing tracked yet")
        return
    r = pd.read_csv(path)
    built = r[r["method"].isin(METHODS)]
    print("=== per slate ===")
    print(r[["season", "week", "draft_group_id", "method", "kind", "score",
             "cashed", "contests", "margin"]].round(2).to_string(index=False))
    print("\n=== running totals, rebuilt lineups ===")
    t = built.groupby(["kind", "method"]).agg(
        slates=("score", "size"), mean_score=("score", "mean"),
        cashed=("cashed", "sum"), contests=("contests", "sum"),
        mean_margin=("margin", "mean"))
    t["cash_rate"] = t["cashed"] / t["contests"].replace(0, np.nan)
    print(t.round(2).to_string())
    for kind in VIEWS:
        p = built[built["kind"] == kind].pivot_table(
            index="draft_group_id", columns="method", values="score")
        if {"model", "blend"} <= set(p.columns) and len(p):
            d = (p["blend"] - p["model"]).dropna()
            print(f"\n{kind}: blend minus model {d.mean():+.2f} per slate, "
                  f"blend better on {int((d > 0).sum())} of {len(d)} slates")


def main(argv: list[str]) -> int:
    if "--report" in argv or not argv:
        report()
        return 0
    for a in argv:
        res = record(int(a))
        print(res[["method", "kind", "projected", "score", "unscored",
                   "cashed", "contests", "margin"]].round(2).to_string(
                       index=False))
    print()
    report()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

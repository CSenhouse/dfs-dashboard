"""Train and test the projected-ownership model on real contest ownership.

For every Classic slate that has counted ownership (data/ownership/, mapped
to its slate through data/observed_cash_lines.csv), the slate is projected
the way the dashboard projects it -- trained only on weeks before it -- and
DFF's projection is joined from the export bound to it. Real ownership is
averaged per contest type, weighted by entries, with an unlisted player at
zero.

Testing is leave-one-slate-out: each slate is predicted by a model that
never saw it. Three feature sets are compared on the same held-out players:
salary and position only, our model's numbers, and those plus DFF's.

Then the question docs/MODELING.md 21 left open: real ownership carried
rank correlation with actual points AFTER our projection was removed. Does
PREDICTED ownership carry any of it? If it does not, the crowd's extra
signal is information we do not have, and no model of ownership built from
our inputs will recover it.

    PYTHONPATH=src .venv/Scripts/python.exe scripts/ownership_model.py
    PYTHONPATH=src .venv/Scripts/python.exe scripts/ownership_model.py --rebuild
"""
from __future__ import annotations

import argparse
import pickle
import sys

import numpy as np
import pandas as pd
from scipy import stats

from dfs import backtest, features, model, ownership, slates
from dfs.config import CURATED, DATA
from dfs.sources import dff

TRAIN = CURATED / "ownership_train.parquet"
MODEL = CURATED / "ownership_model.pkl"


def project(dgid: int, history: pd.DataFrame) -> pd.DataFrame:
    """The slate as the dashboard saw it before lock."""
    up = slates.assemble(dgid)
    s, w = int(up["season"].iloc[0]), int(up["week"].iloc[0])
    past = history[(history["season"] < s)
                   | ((history["season"] == s) & (history["week"] < w))]
    cur = features.build(up)
    cur = cur[cur["salary"].fillna(0) > 0].reset_index(drop=True)
    cur = cur.merge(model.two_stage_all(past, cur), on="player_dk_id",
                    how="left")
    teams = tuple(sorted(cur["team"].dropna().astype(str).unique()))
    theirs, why, _ = dff.for_slate(dgid, teams)
    if not theirs.empty:
        m = dff.match(theirs, cur[["player_dk_id", "player_name", "position",
                                   "team", "salary"]])
        m = m[m["matched"]][["player_dk_id", "dff_proj"]]
        cur = cur.merge(m.drop_duplicates("player_dk_id"),
                        on="player_dk_id", how="left")
    else:
        print(f"  dg {dgid}: no DFF ({why})")
    cur["season"], cur["week"] = s, w
    return cur


def build() -> pd.DataFrame:
    lines = pd.read_csv(DATA / "observed_cash_lines.csv")
    fields = pd.read_csv(DATA / "contest_fields.csv")
    lines = lines.merge(fields[["contest_id", "format"]], on="contest_id",
                        how="left")
    have = {p.stem for p in (DATA / "ownership").glob("*.csv")}
    lines = lines[lines["contest_id"].astype(str).isin(have)
                  & (lines["format"] == "classic")]
    lines["contest_type"] = [ownership.contest_type(p, n) for p, n in
                             zip(lines["paid_pct"], lines["entries"])]
    history = backtest.prepare(pd.read_parquet(CURATED / "features.parquet"))

    frames = []
    for dgid, contests in lines.groupby("draft_group"):
        print(f"dg {dgid}: {len(contests)} contests", flush=True)
        cur = project(int(dgid), history)
        cur = cur[cur["proj_mean"].notna()]
        # Ownership exports carry names only. Join on the name INSIDE this
        # slate, and drop any name the slate prices twice (rule 1: resolve
        # ambiguity to neither).
        dup = cur["player_name"].duplicated(keep=False)
        names = cur.loc[~dup, ["player_name", "player_dk_id"]]
        base = ownership.slate_features(cur)
        for kind, group in contests.groupby("contest_type"):
            owned, weight = [], []
            for r in group.itertuples():
                o = pd.read_csv(DATA / "ownership" / f"{r.contest_id}.csv")
                hit = o.merge(names, on="player_name", how="left")
                lost = hit.loc[hit["player_dk_id"].isna(), "pct_drafted"].sum()
                print(f"  {r.contest_id} {kind:<10} {r.entries:>6} entries, "
                      f"unmatched ownership {lost:.2f} of {o['pct_drafted'].sum():.2f}")
                owned.append(hit.dropna(subset=["player_dk_id"])
                             .set_index("player_dk_id")["pct_drafted"])
                weight.append(r.entries)
            own = (pd.concat(owned, axis=1).fillna(0.0)
                   .mul(weight, axis=1).sum(axis=1) / sum(weight))
            f = ownership.with_type(base, kind)
            f["own"] = f["player_dk_id"].map(own).fillna(0.0)
            f["draft_group"] = int(dgid)
            f["n_contests"] = len(group)
            frames.append(f)
    out = pd.concat(frames, ignore_index=True)
    keep = (["draft_group", "season", "week", "player_dk_id", "player_name",
             "team", "dk_actual", "n_contests", "own", "contest_type"]
            + list(ownership.FEATURES))
    out = out[[c for c in dict.fromkeys(keep) if c in out.columns]]
    out.to_parquet(TRAIN, index=False)
    return out


def evaluate(d: pd.DataFrame) -> None:
    sets = {"salary only": ownership.SALARY_ONLY,
            "our model": ownership.MODEL_ONLY,
            "model + DFF": list(ownership.FEATURES)}
    slates_ = sorted(d["draft_group"].unique())
    rows, held = [], []
    for g in slates_:
        tr, te = d[d["draft_group"] != g], d[d["draft_group"] == g].copy()
        for name, cols in sets.items():
            m = ownership.fit(tr, cols)
            te[name] = ownership.predict(m, te)
        held.append(te)
        for kind, t in te.groupby("contest_type"):
            top = t.nlargest(10, "own")
            for name in sets:
                rows.append({
                    "slate": g, "week": int(t["week"].iloc[0]),
                    "type": kind, "features": name,
                    "spearman": stats.spearmanr(t[name], t["own"])[0],
                    "top10 MAE": (top[name] - top["own"]).abs().mean(),
                    "top10 hit": len(set(t.nlargest(10, name)["player_dk_id"])
                                     & set(top["player_dk_id"])),
                })
    r = pd.DataFrame(rows)
    print("\nleave-one-slate-out, every slate predicted by a model that "
          "never saw it\n")
    print(r.pivot_table(index=["week", "type"], columns="features",
                        values="spearman").round(3).to_string())
    print("\nmean over held-out slates and types:")
    print(r.groupby("features")[["spearman", "top10 MAE", "top10 hit"]]
          .mean().round(3).to_string())

    h = pd.concat(held, ignore_index=True)
    h.to_parquet(CURATED / "ownership_heldout.parquet", index=False)
    print("\ndoes ownership know something our projection does not? "
          "(partial rank correlation with actual points, our projection "
          "removed; rostered-range players, projection >= 5)")
    for kind, t in h[(h["proj"] >= 5) & h["dk_actual"].notna()].groupby(
            "contest_type"):
        for name, col in (("real ownership", "own"),
                          ("predicted (model + DFF)", "model + DFF")):
            pr = _partial(t[col], t["dk_actual"], t["proj"])
            print(f"  {kind:<10} {name:<24} {pr[0]:+.3f}  p={pr[1]:.3f}  "
                  f"n={len(t)}")


def _partial(x, y, z) -> tuple[float, float]:
    """Spearman of x and y after removing z from both (on ranks)."""
    rx, ry, rz = (pd.Series(v).rank().to_numpy() for v in (x, y, z))
    ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
    ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    return stats.pearsonr(ex, ey)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()
    d = build() if args.rebuild or not TRAIN.exists() else pd.read_parquet(TRAIN)
    print(f"\n{d['draft_group'].nunique()} slates, "
          f"{d.groupby(['draft_group', 'contest_type']).ngroups} slate-types, "
          f"{len(d):,} player rows")
    evaluate(d)
    final = ownership.fit(d, list(ownership.FEATURES))
    MODEL.write_bytes(pickle.dumps(final))
    print(f"\nfinal model on all slates -> {MODEL.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

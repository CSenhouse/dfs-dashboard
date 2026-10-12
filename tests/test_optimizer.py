"""Optimizer constraints.

An illegal lineup is rejected at upload, and a subtly wrong one (a duplicated
player, a single-game roster) is worse -- it looks fine and is not what the
model chose. These pin the DK classic rules.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dfs import optimizer
from dfs.optimizer import optimize


@pytest.fixture
def pool() -> pd.DataFrame:
    """A synthetic slate: 6 games, enough depth at every position."""
    rng = np.random.default_rng(0)
    rows = []
    pid = 0
    teams = ["BUF", "MIA", "NE", "NYJ", "KC", "DEN", "LV", "LAC", "CIN",
             "BAL", "PIT", "CLE"]
    for gi in range(6):
        for team in (teams[gi * 2], teams[gi * 2 + 1]):
            for pos, n in (("QB", 2), ("RB", 4), ("WR", 6), ("TE", 3), ("DST", 1)):
                for _ in range(n):
                    pid += 1
                    rows.append({
                        "player_dk_id": pid,
                        "player_name": f"p{pid}",
                        "position": pos,
                        "team": team,
                        "game_key": f"g{gi}",
                        "salary": float(rng.integers(2500, 9000)),
                        "projection": float(rng.uniform(2, 25)),
                    })
    return pd.DataFrame(rows)


def counts(lu: pd.DataFrame) -> dict:
    return lu["position"].value_counts().to_dict()


def test_roster_is_legal(pool):
    lu = optimize(pool, n_lineups=1)[0]
    c = counts(lu)
    assert len(lu) == 9
    assert c["QB"] == 1
    assert c["DST"] == 1
    assert 2 <= c.get("RB", 0) <= 3
    assert 3 <= c.get("WR", 0) <= 4
    assert 1 <= c.get("TE", 0) <= 2
    assert c.get("RB", 0) + c.get("WR", 0) + c.get("TE", 0) == 7


def test_respects_salary_cap(pool):
    lu = optimize(pool, n_lineups=1)[0]
    assert lu["salary"].sum() <= 50_000


def test_lower_cap_is_binding(pool):
    lu = optimize(pool, n_lineups=1, salary_cap=40_000)[0]
    assert lu["salary"].sum() <= 40_000


def test_no_duplicate_players(pool):
    for lu in optimize(pool, n_lineups=5):
        assert not lu["player_dk_id"].duplicated().any()


def test_uses_at_least_two_games(pool):
    """DK requires players from two different games."""
    for lu in optimize(pool, n_lineups=5):
        assert lu["game_key"].nunique() >= 2


def test_lineups_are_unique(pool):
    lus = optimize(pool, n_lineups=10, min_difference=1)
    sets = [frozenset(l["player_dk_id"]) for l in lus]
    assert len(set(sets)) == len(sets)


def test_min_difference_is_enforced(pool):
    lus = optimize(pool, n_lineups=8, min_difference=3)
    sets = [set(l["player_dk_id"]) for l in lus]
    for i, a in enumerate(sets):
        for b in sets[i + 1:]:
            assert len(a & b) <= 9 - 3


def test_exposure_cap_is_enforced(pool):
    n = 10
    lus = optimize(pool, n_lineups=n, max_exposure=0.3, min_difference=2)
    used = pd.concat(lus)["player_dk_id"].value_counts()
    assert used.max() <= int(0.3 * n)


def test_lock_forces_a_player_in(pool):
    target = int(pool[pool["position"] == "WR"]["player_dk_id"].iloc[0])
    lu = optimize(pool, n_lineups=1, locks=[target])[0]
    assert target in set(lu["player_dk_id"])


def test_ban_keeps_a_player_out(pool):
    """Ban the highest projected player; he must not appear."""
    target = int(pool.nlargest(1, "projection")["player_dk_id"].iloc[0])
    lu = optimize(pool, n_lineups=1, bans=[target])[0]
    assert target not in set(lu["player_dk_id"])


def test_stacking_pairs_qb_with_own_receivers(pool):
    lu = optimize(pool, n_lineups=1, stack_qb_with=2)[0]
    qb = lu[lu["position"] == "QB"].iloc[0]
    mates = lu[(lu["team"] == qb["team"]) & (lu["position"].isin(["WR", "TE"]))]
    assert len(mates) >= 2


def test_first_lineup_is_optimal(pool):
    """Nothing should beat lineup 1 on the objective."""
    lus = optimize(pool, n_lineups=5)
    totals = [l["projection"].sum() for l in lus]
    assert totals[0] == max(totals)


def test_empty_pool_returns_nothing():
    empty = pd.DataFrame(
        columns=["player_dk_id", "player_name", "position", "team",
                 "salary", "projection"]
    )
    assert optimize(empty) == []


# --- bring-back -------------------------------------------------------------
# A bring-back is a skill player from the team the stacked QB is facing. It
# buys the shootout: the measured residual correlation between two opposing
# quarterbacks in this project's own data is +0.11, POSITIVE, because the team
# that falls behind throws. A lineup with no bring-back is quietly betting on
# a blowout, which is the lower-scoring outcome.

def two_game_pool() -> pd.DataFrame:
    rng = np.random.default_rng(4)
    pairs = [("BUF", "MIA"), ("KC", "DEN"), ("NE", "NYJ"), ("LV", "LAC")]
    opp = {a: b for a, b in pairs}
    opp.update({b: a for a, b in pairs})
    rows, pid = [], 0
    for pos, n in (("QB", 8), ("RB", 32), ("WR", 48), ("TE", 24), ("DST", 8)):
        for k in range(n):
            pid += 1
            team = list(opp)[k % len(opp)]
            salary = float(rng.integers(30, 90) * 100)
            rows.append({
                "player_dk_id": pid, "player_name": f"p{pid}",
                "position": pos, "team": team, "opponent": opp[team],
                "salary": salary,
                "projection": salary / 500 + rng.normal(0, 1),
            })
    return pd.DataFrame(rows)


def test_bring_back_puts_an_opponent_in_the_lineup():
    pool = two_game_pool()
    lu = optimizer.optimize(pool, n_lineups=1, stack_qb_with=1, bring_back=1)[0]
    qb = lu[lu["position"] == "QB"].iloc[0]
    rivals = lu[(lu["team"] == qb["opponent"])
                & (lu["position"].isin(["WR", "TE", "RB"]))]
    assert len(rivals) >= 1


def test_bring_back_and_the_stack_apply_together():
    pool = two_game_pool()
    lu = optimizer.optimize(pool, n_lineups=1, stack_qb_with=2, bring_back=1)[0]
    qb = lu[lu["position"] == "QB"].iloc[0]
    mates = lu[(lu["team"] == qb["team"]) & (lu["position"].isin(["WR", "TE"]))]
    rivals = lu[(lu["team"] == qb["opponent"])
                & (lu["position"].isin(["WR", "TE", "RB"]))]
    assert len(mates) >= 2
    assert len(rivals) >= 1


def test_stacking_costs_projected_points():
    """It has to, and that is the trade. Forcing the QB's receiver in means
    passing over a better-projected player, which is a bad deal in a cash game
    and the entire point in a tournament."""
    pool = two_game_pool()
    free = optimizer.optimize(pool, n_lineups=1)[0]["projection"].sum()
    stacked = optimizer.optimize(
        pool, n_lineups=1, stack_qb_with=2, bring_back=1
    )[0]["projection"].sum()
    assert stacked <= free + 1e-6


def test_bring_back_is_skipped_when_there_is_no_opponent_column():
    """Older curated frames predate the column; the constraint should be
    dropped rather than crash or silently match everybody."""
    pool = two_game_pool().drop(columns=["opponent"])
    assert optimizer.optimize(pool, n_lineups=1, bring_back=2)

"""
Step 4 - Simulate the IPL 2027 season 10,000 times.

How it works
  1. Fit the small "core 3" logistic model (Elo, batting strength, wicket-taking strength) on all matches.
  2. Rate every team for 2027 from its squad (squads_2027.csv) and its carried-over Elo.
  3. Turn those ratings into a 10x10 table of single-match win probabilities.
  4. Play 70 league matches + the IPL playoff format 10,000 times -> title / top-4 probabilities.

squads_2027.csv
  First run: auto-built from each team's 2026 players (a "nothing changes" baseline).
  After the 16 Dec 2026 auction: edit it (delete released players, add bought players, set xi=1 for the
  expected XI; new players get bat_val / wk_val by hand - see the printed percentiles), then re-run.
  To rebuild it from scratch, delete the file.

Usage:  python src/04_season_sim.py
"""
import importlib
from pathlib import Path

import numpy as np
import pandas as pd

from common import CORE, fit_lr, make_X, predict_lr

f2 = importlib.import_module("02_features")  # reuse its constants so training and simulation stay consistent
DATA, RES = Path("data"), Path("results")
SQUADS = Path("live/squads_2027.csv")
N_SIMS = 10_000
SEED = 2027


def make_squads(active):
    pr = pd.read_csv(DATA / "player_ratings_latest.csv")
    pr = pr[(pr["last_season"] == pr["last_season"].max()) & pr["last_team"].isin(active)].copy()
    d = f2.PLAYER_DECAY  # same season roll-over as in 02_features.py
    m = pr["matches_weighted"] * d
    pr["bat_val"] = (pr["runs_weighted"] * d + f2.BAT_PRIOR * f2.BAT_W) / (m + f2.BAT_W)
    pr["wk_val"] = (pr["wk_weighted"] * d + f2.WK_PRIOR * f2.WK_W) / (m + f2.WK_W)
    pr = pr.sort_values(["last_team", "matches_weighted"], ascending=[True, False])
    pr["xi"] = (pr.groupby("last_team").cumcount() < 11).astype(int)  # regulars = most recent games
    out = pr.rename(columns={"last_team": "team"})[
        ["team", "pkey", "player", "xi", "bat_val", "wk_val", "matches_weighted"]]
    out.to_csv(SQUADS, index=False)
    print(f"Built {SQUADS} from 2026 squads ({len(out)} players). Edit it after the auction and re-run.\n")
    return out


def team_ratings(squads, elo_now):
    rows = []
    for team, g in squads.groupby("team"):
        xi = g[g["xi"] == 1]
        bats = sorted(xi["bat_val"], reverse=True)[:7]
        wks = sorted(xi["wk_val"], reverse=True)[:5]
        bats += [f2.BAT_PRIOR] * (7 - len(bats))      # pad if fewer than 7 flagged
        wks += [f2.WK_PRIOR] * (5 - len(wks))
        elo = f2.ELO_START + f2.ELO_CARRY * (elo_now[team] - f2.ELO_START)  # new-season regression
        rows.append({"team": team, "elo": elo, "bat": sum(bats), "wk": sum(wks), "xi_players": len(xi)})
    return pd.DataFrame(rows).sort_values("team").reset_index(drop=True)


def win_matrix(model, r):
    n = len(r)
    P = np.full((n, n), 0.5)
    for i in range(n):
        for j in range(n):
            if i != j:
                x = pd.DataFrame([{"elo_diff": (r.elo[i] - r.elo[j]) / 100.0,
                                   "bat_diff": r.bat[i] - r.bat[j],
                                   "wicket_diff": r.wk[i] - r.wk[j]}])
                P[i, j] = predict_lr(model, x, cols=CORE)[0]
    return P


def schedule():
    """70 league games, 14 per team: 5 opponents twice, 4 once (the real IPL shape; fixtures aren't out yet)."""
    A, B = [], []
    for d, reps in ((1, 2), (3, 2), (5, 2), (2, 1), (4, 1)):
        for i in range(10):
            if d == 5 and i >= 5:
                continue
            for _ in range(reps):
                A.append(i)
                B.append((i + d) % 10)
    assert len(A) == 70
    return np.array(A), np.array(B)


def simulate(P, n_sims, rng):
    T = P.shape[0]
    A, B = schedule()
    title, final, top4, wins = (np.zeros(T) for _ in range(4))
    play = lambda x, y: x if rng.random() < P[x, y] else y
    for _ in range(n_sims):
        perm = rng.permutation(T)                       # random team -> fixture-slot mapping each season
        ta, tb = perm[A], perm[B]
        winners = np.where(rng.random(len(A)) < P[ta, tb], ta, tb)
        w = np.bincount(winners, minlength=T)
        order = np.argsort(-(2 * w + rng.random(T) * 0.9))  # random tiebreak stands in for net run rate
        t1, t2, t3, t4 = order[:4]
        w1 = play(t1, t2); l1 = t2 if w1 == t1 else t1      # Qualifier 1
        we = play(t3, t4)                                   # Eliminator
        w2 = play(l1, we)                                   # Qualifier 2
        champ = play(w1, w2)                                # Final
        wins += w
        top4[order[:4]] += 1
        final[[w1, w2]] += 1
        title[champ] += 1
    return pd.DataFrame({"p_title": title / n_sims, "p_final": final / n_sims,
                         "p_top4": top4 / n_sims, "avg_wins": wins / n_sims})


def main():
    RES.mkdir(exist_ok=True)
    elo_now = pd.read_csv(DATA / "team_ratings_latest.csv").set_index("team")["elo"].to_dict()
    active = sorted(elo_now)
    if len(active) != 10:
        raise SystemExit(f"Expected 10 active teams, found {len(active)}: {active}")

    squads = pd.read_csv(SQUADS) if SQUADS.exists() else make_squads(active)
    print("bat_val / wk_val percentiles in squads (use these to set values for new players):")
    print(squads[["bat_val", "wk_val"]].quantile([.1, .5, .9]).round(2).to_string(), "\n")

    feats = pd.read_csv(DATA / "features.csv")
    model = fit_lr(make_X(feats), feats["a_won"].to_numpy(), cols=CORE)

    r = team_ratings(squads, elo_now)
    P = win_matrix(model, r)
    off = P[~np.eye(10, dtype=bool)]
    print(r.round(2).to_string(index=False))
    print(f"\nSingle-match win probability range between any two teams: {off.min():.3f} - {off.max():.3f}")

    res = simulate(P, N_SIMS, np.random.default_rng(SEED))
    res.insert(0, "team", r["team"])
    res = res.sort_values("p_title", ascending=False)
    res.to_csv(RES / "title_probabilities.csv", index=False)
    r.to_csv(RES / "team_strength_2027.csv", index=False)

    print(f"\n=== IPL 2027 outcome probabilities ({N_SIMS:,} simulated seasons) ===")
    print(res.assign(p_title=lambda d: (d.p_title * 100).round(1), p_final=lambda d: (d.p_final * 100).round(1),
                     p_top4=lambda d: (d.p_top4 * 100).round(1), avg_wins=lambda d: d.avg_wins.round(1))
          .to_string(index=False))
    print(f"\nSaved -> {RES/'title_probabilities.csv'}")


if __name__ == "__main__":
    main()

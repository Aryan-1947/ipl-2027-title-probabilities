"""
Step 10 - IPL 2027 title probabilities with the v3 skill model.

  * Team strength comes from the expected XI in squads_2027.csv (xi = 1), rated on each player's ENTIRE men's
    T20 record up to the start of 2027 (step 9). Players with no T20 history count as average.
  * Home advantage: in the 70-match league a pair that meets twice hosts one game each; a pair that meets once
    has a random host. Playoffs are neutral.
  * Uncertainty: the model is re-fitted on 30 bootstrap resamples of the match history, 1,000 seasons each;
    the range is the 5th-95th percentile across re-fits (player-rating noise is not included).

Update squads_2027.csv after the auction (columns team, pkey, xi). To add a player with T20 history,
copy his player id (pkey) from data/t20_player_matches.csv (column player_id) - the names are in
results/player_impact.csv.

Outputs: results/title_probabilities.csv, results/team_rating_v3.csv
Usage:   python 10_season_sim_v3.py
"""
import datetime
import importlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

m9 = importlib.import_module("09_t20_model")
m4 = importlib.import_module("04_season_sim")
m6 = m9.m6
RES, MOD = Path("results"), Path("models")
N_BOOT, SIMS_PER_BOOT, SEED = 30, 1000, 2027
START_2027 = datetime.date(2027, 3, 25).toordinal()


def load_teams():
    """Returns bundle, team list, per-team feature vectors and the neutral-venue logit matrix Z."""
    bundle = joblib.load(MOD / "v3.pkl")
    if not m4.SQUADS.exists():
        raise SystemExit("squads_2027.csv not found - run 04_season_sim.py once first.")
    squads = pd.read_csv(m4.SQUADS)
    active = sorted(pd.read_csv("data/team_ratings_latest.csv")["team"])
    if len(active) != 10:
        raise SystemExit(f"Expected 10 active teams, found {len(active)}")
    hl, el = bundle["params"]["hl"], bundle["elo"]
    elo = np.array([1500.0 + el["carry"] * (el["final"].get(t, 1500.0) - 1500.0) for t in active])
    F = np.array([m9.team_vec(squads.loc[(squads["team"] == t) & (squads["xi"] == 1), "pkey"].tolist(),
                              START_2027, bundle["states"], hl) for t in active])
    return bundle, active, elo, F


def logit_matrix(beta, elo, F):
    d_elo = (elo[:, None] - elo[None, :]) / 100.0
    d_skill = (F[:, None, :] - F[None, :, :]) @ beta[1:5]
    return beta[0] * d_elo + d_skill


def schedule_h():
    """70 league games: 5 opponents twice (one home each), 4 once (random host). H: +1/-1 fixed, 0 = random."""
    A, B, H = [], [], []
    for d, reps in ((1, 2), (3, 2), (5, 2), (2, 1), (4, 1)):
        for i in range(10):
            if d == 5 and i >= 5:
                continue
            j = (i + d) % 10
            if reps == 2:
                A += [i, i]; B += [j, j]; H += [1, -1]
            else:
                A.append(i); B.append(j); H.append(0)
    assert len(A) == 70
    return np.array(A), np.array(B), np.array(H)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def simulate(Z, beta_home, n_sims, rng):
    T = Z.shape[0]
    A, B, H = schedule_h()
    neutral = sigmoid(Z)
    play = lambda x, y: x if rng.random() < neutral[x, y] else y
    title, final, top4, wins = (np.zeros(T) for _ in range(4))
    for _ in range(n_sims):
        perm = rng.permutation(T)
        ta, tb = perm[A], perm[B]
        h = np.where(H == 0, rng.choice([-1, 1], size=len(A)), H)
        winners = np.where(rng.random(len(A)) < sigmoid(Z[ta, tb] + beta_home * h), ta, tb)
        w = np.bincount(winners, minlength=T)
        order = np.argsort(-(2 * w + rng.random(T) * 0.9))     # random tiebreak stands in for net run rate
        t1, t2, t3, t4 = order[:4]
        w1 = play(t1, t2); l1 = t2 if w1 == t1 else t1          # Qualifier 1
        we = play(t3, t4)                                       # Eliminator
        w2 = play(l1, we)                                       # Qualifier 2
        champ = play(w1, w2)                                    # Final
        wins += w
        top4[order[:4]] += 1
        final[[w1, w2]] += 1
        title[champ] += 1
    return pd.DataFrame({"p_title": title / n_sims, "p_final": final / n_sims,
                         "p_top4": top4 / n_sims, "avg_wins": wins / n_sims})


def main():
    RES.mkdir(exist_ok=True)
    bundle, active, elo, F = load_teams()
    beta, X, y = bundle["beta"], bundle["X"], bundle["y"]
    Z0 = logit_matrix(beta, elo, F)
    strength = beta[0] * (elo - elo.mean()) / 100.0 + (F - F.mean(0)) @ beta[1:5]
    rating = pd.DataFrame({"team": active, "elo_2027": elo.round(1), "bat_runs": F[:, 0].round(2),
                           "bat_outs": F[:, 1].round(3), "bowl_saved": F[:, 2].round(2),
                           "bowl_wk": F[:, 3].round(3), "strength_logit": strength.round(3)})
    rating.to_csv(RES / "team_rating_v3.csv", index=False)
    print(rating.sort_values("strength_logit", ascending=False).to_string(index=False))
    off = sigmoid(Z0[~np.eye(10, dtype=bool)])
    print(f"\nNeutral-venue single-match win probability range: {off.min():.3f} - {off.max():.3f}"
          f"   |   home advantage = {sigmoid(beta[5]) * 100:.1f}% for an evenly matched home side")

    rng = np.random.default_rng(SEED)
    runs = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(y), len(y))
        bb = m9.raw_beta(m9.fit_lr(X[idx], y[idx], bundle["params"]["C"]))
        runs.append(simulate(logit_matrix(bb, elo, F), bb[5], SIMS_PER_BOOT, rng))
    stack = lambda col: np.vstack([r[col].to_numpy() for r in runs])
    pt = stack("p_title")
    res = pd.DataFrame({
        "team": active, "p_title": pt.mean(0),
        "p_title_lo": np.percentile(pt, 5, axis=0), "p_title_hi": np.percentile(pt, 95, axis=0),
        "p_final": stack("p_final").mean(0), "p_top4": stack("p_top4").mean(0),
        "avg_wins": stack("avg_wins").mean(0),
    }).sort_values("p_title", ascending=False)
    res.to_csv(RES / "title_probabilities.csv", index=False)

    show = pd.DataFrame({
        "team": res["team"], "title_%": (res["p_title"] * 100).round(1),
        "title_range_%": [f"{a*100:.1f}-{b*100:.1f}" for a, b in zip(res["p_title_lo"], res["p_title_hi"])],
        "final_%": (res["p_final"] * 100).round(1), "top4_%": (res["p_top4"] * 100).round(1),
        "avg_wins": res["avg_wins"].round(1)})
    print(f"\n=== IPL 2027 outcome probabilities ({N_BOOT} model re-fits x {SIMS_PER_BOOT:,} seasons) ===")
    print(show.to_string(index=False))
    print(f"\nSaved -> {RES/'title_probabilities.csv'}")


if __name__ == "__main__":
    main()

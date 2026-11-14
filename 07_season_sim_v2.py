"""
Step 7 - IPL 2027 title probabilities using the improved (v2) model, WITH uncertainty ranges.

For each of 30 bootstrap re-fits of the model (resampled history), the season is simulated 1,000 times.
The spread across those re-fits shows how much each team's title chance depends on model uncertainty:
the reported range is the 5th-95th percentile across re-fits.

Needs: squads_2027.csv (created by 04_season_sim.py; edit it after the auction - only the
`team`, `pkey` and `xi` columns matter here) and the outputs of steps 2 and 6.
To add a player who has IPL history, copy his `pkey` from results/player_impact.csv.
A player with no IPL history (pkey not in that file) is treated as an average player (impact 0).

Outputs: results/title_probabilities.csv (replaces the v1 table), results/team_rating_v2.csv

Usage:  python 07_season_sim_v2.py
"""
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import fit_lr, predict_lr

m6 = importlib.import_module("06_player_model")
m4 = importlib.import_module("04_season_sim")
RES, MOD = Path("results"), Path("models")
N_BOOT, SIMS_PER_BOOT, SEED = 30, 1000, 2027


def fit_components(diffs, y, M, season, params, now, idx=None):
    """Fit Elo logistic + player ridge on all history (or on a bootstrap sample `idx`)."""
    idx = np.arange(len(y)) if idx is None else idx
    X = pd.DataFrame({"elo_diff": diffs[idx] / 100.0})
    elo_lr = fit_lr(X, y[idx], cols=["elo_diff"])
    w = m6.decay(season[idx], now, params["half_life"])
    coef = m6.fit_ridge(M[idx], y[idx], w, params["C"])
    return elo_lr, coef


def win_matrix(elo_lr, coef, elo, impact, params):
    n = len(elo)
    ii, jj = np.where(~np.eye(n, dtype=bool))
    p_e = predict_lr(elo_lr, pd.DataFrame({"elo_diff": (elo[ii] - elo[jj]) / 100.0}), cols=["elo_diff"])
    p_r = m6.sigmoid(impact[ii] - impact[jj])
    p = m6.blend_prob(p_e, p_r, params["blend_w"], params["shrink"])
    P = np.full((n, n), 0.5)
    P[ii, jj] = p
    return (P + (1 - P.T)) / 2.0            # enforce P[i,j] = 1 - P[j,i]


def main():
    params = json.load(open(MOD / "v2_params.json"))
    feats, pm, xi, pidx, M = m6.load_all()
    y, season = feats["a_won"].to_numpy(), feats["season"].to_numpy()
    now = season.max() + 1

    diffs, elo_final = m6.elo_run(feats, params["elo_k"], params["elo_carry"])
    active = sorted(pd.read_csv("data/team_ratings_latest.csv")["team"])
    if len(active) != 10:
        raise SystemExit(f"Expected 10 active teams, found {len(active)}")
    elo = np.array([1500.0 + params["elo_carry"] * (elo_final.get(t, 1500.0) - 1500.0) for t in active])

    if not m4.SQUADS.exists():
        raise SystemExit("squads_2027.csv not found - run 04_season_sim.py once first.")
    squads = pd.read_csv(m4.SQUADS)
    xi_sq = squads[squads["xi"] == 1]

    def squad_impact(coef):
        out = []
        for t in active:
            keys = xi_sq.loc[xi_sq["team"] == t, "pkey"]
            out.append(sum(coef[pidx[k]] for k in keys if k in pidx))
        return np.array(out)

    # point estimate
    elo_lr, coef = fit_components(diffs, y, M, season, params, now)
    imp0 = squad_impact(coef)
    P0 = win_matrix(elo_lr, coef, elo, imp0, params)
    rating = pd.DataFrame({"team": active, "elo_2027": elo.round(1), "squad_impact": imp0.round(3)})
    rating.to_csv(RES / "team_rating_v2.csv", index=False)
    print(rating.sort_values("squad_impact", ascending=False).to_string(index=False))
    off = P0[~np.eye(10, dtype=bool)]
    print(f"\nSingle-match win probability range between any two teams: {off.min():.3f} - {off.max():.3f}")

    # bootstrap the model, simulate each re-fit
    rng = np.random.default_rng(SEED)
    runs = []
    for b in range(N_BOOT):
        idx = rng.integers(0, len(y), len(y))
        e_lr, c = fit_components(diffs, y, M, season, params, now, idx)
        P = win_matrix(e_lr, c, elo, squad_impact(c), params)
        runs.append(m4.simulate(P, SIMS_PER_BOOT, rng))
    stack = lambda col: np.vstack([r[col].to_numpy() for r in runs])
    pt = stack("p_title")
    res = pd.DataFrame({
        "team": active,
        "p_title": pt.mean(0),
        "p_title_lo": np.percentile(pt, 5, axis=0),
        "p_title_hi": np.percentile(pt, 95, axis=0),
        "p_final": stack("p_final").mean(0),
        "p_top4": stack("p_top4").mean(0),
        "avg_wins": stack("avg_wins").mean(0),
    }).sort_values("p_title", ascending=False)
    res.to_csv(RES / "title_probabilities.csv", index=False)

    print(f"\n=== IPL 2027 outcome probabilities ({N_BOOT} model re-fits x {SIMS_PER_BOOT:,} seasons) ===")
    show = res.copy()
    show["title_%"] = (show["p_title"] * 100).round(1)
    show["title_range_%"] = [f"{a*100:.1f}-{b*100:.1f}" for a, b in zip(show["p_title_lo"], show["p_title_hi"])]
    show["final_%"] = (show["p_final"] * 100).round(1)
    show["top4_%"] = (show["p_top4"] * 100).round(1)
    show["avg_wins"] = show["avg_wins"].round(1)
    print(show[["team", "title_%", "title_range_%", "final_%", "top4_%", "avg_wins"]].to_string(index=False))
    print(f"\nSaved -> {RES/'title_probabilities.csv'}")


if __name__ == "__main__":
    main()

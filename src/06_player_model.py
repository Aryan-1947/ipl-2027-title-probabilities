"""
Step 6 - Improved match model: tuned Elo + player-impact (ridge) model.

Design (this is the "analyst" part - worth explaining in your README)
  * Tune on the 6 seasons BEFORE the test window (2016-2021), then test ONCE on the last 5 seasons
    (2022-2026). Nothing is tuned on the test window.
  * Player-impact model: each player gets one rating. A match is "sum of Team A XI ratings minus sum of
    Team B XI ratings", fitted with L2-regularised logistic regression (adjusted plus-minus, as used in
    basketball / football analytics). Recent seasons count more (exponential time decay).
  * Blend with a tuned Elo model, and shrink towards 50% if that helps (guards against over-confidence).
  * Significance: bootstrap confidence interval of the log-loss gain over a coin flip.

Outputs
    results/model_comparison.csv   (v2 rows are added to the table from step 3)
    results/player_impact.csv      adjusted impact rating for every player
    models/v2_params.json          tuned settings used by 07_season_sim_v2.py

Usage:  python src/06_player_model.py
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from common import fit_lr, predict_lr

DATA, RES, MOD = Path("data"), Path("results"), Path("models")
ELO_K_GRID = (6, 10, 20)
ELO_CARRY_GRID = (0.25, 0.5, 0.75)
HALF_LIFE_GRID = (1.5, 3.0, 6.0)      # seasons
C_GRID = (0.01, 0.03, 0.1)            # ridge strength (smaller = more shrinkage)
BLEND_W_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
SHRINK_GRID = (0.5, 0.75, 1.0)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# ------------------------------------------------------------------ data -----------------
def build_matrix(f, xi, pidx):
    rows, cols, vals = [], [], []
    for i, (mid, a, b) in enumerate(zip(f["match_id"], f["team_a"], f["team_b"])):
        for p in xi.get((mid, a), []):
            rows.append(i); cols.append(pidx[p]); vals.append(1.0)
        for p in xi.get((mid, b), []):
            rows.append(i); cols.append(pidx[p]); vals.append(-1.0)
    return sparse.csr_matrix((vals, (rows, cols)), shape=(len(f), len(pidx)))


def load_all():
    feats = pd.read_csv(DATA / "features.csv")
    pm = pd.read_csv(DATA / "player_matches.csv")
    pm["pkey"] = pm["player_id"].fillna(pm["player"])
    xi = {k: g["pkey"].tolist() for k, g in pm.groupby(["match_id", "team"])}
    pidx = {p: i for i, p in enumerate(sorted(pm["pkey"].unique()))}
    return feats, pm, xi, pidx, build_matrix(feats, xi, pidx)


# ------------------------------------------------------------------ models ---------------
def elo_run(f, k, carry):
    """Walk-forward Elo. Returns pre-match rating differences and the final ratings."""
    elo = defaultdict(lambda: 1500.0)
    diffs = np.empty(len(f))
    prev = None
    for i, (s, a, b, w) in enumerate(zip(f["season"], f["team_a"], f["team_b"], f["a_won"])):
        if prev is not None and s != prev:
            for t in list(elo):
                elo[t] = 1500.0 + carry * (elo[t] - 1500.0)
        prev = s
        diffs[i] = elo[a] - elo[b]
        e = 1.0 / (1.0 + 10 ** ((elo[b] - elo[a]) / 400.0))
        elo[a] += k * (w - e)
        elo[b] -= k * (w - e)
    return diffs, dict(elo)


def decay(seasons, now, half_life):
    return 0.5 ** ((now - np.asarray(seasons, dtype=float)) / half_life)


def fit_ridge(M, y, weights, C):
    m = LogisticRegression(C=C, fit_intercept=False, max_iter=3000)
    m.fit(M, y, sample_weight=weights / weights.mean())
    return m.coef_[0]


def elo_walk(diffs, y, season, eval_s):
    X = pd.DataFrame({"elo_diff": diffs / 100.0})
    p = np.full(len(y), np.nan)
    for s in eval_s:
        tr, te = season < s, season == s
        if tr.sum() < 50:
            continue
        m = fit_lr(X[tr], y[tr], cols=["elo_diff"])
        p[te] = predict_lr(m, X[te], cols=["elo_diff"])
    return p


def ridge_walk(M, y, season, eval_s, hl, C):
    p = np.full(len(y), np.nan)
    for s in eval_s:
        tr, te = season < s, season == s
        if tr.sum() < 50:
            continue
        coef = fit_ridge(M[tr], y[tr], decay(season[tr], s, hl), C)
        p[te] = sigmoid(M[te] @ coef)
    return p


def blend_prob(p_e, p_r, w, shrink):
    p = w * p_r + (1 - w) * p_e
    return 0.5 + shrink * (p - 0.5)


# ------------------------------------------------------------------ evaluation -----------
def metrics(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    acc = accuracy_score(y, p > 0.5)
    return acc, 1.96 * np.sqrt(acc * (1 - acc) / len(y)), log_loss(y, p), brier_score_loss(y, p)


def gain_vs_coinflip(y, p, rng, n=2000):
    """Mean log-loss gain over a coin flip (positive = better) with a bootstrap 95% interval."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    d = np.log(2) - (-(y * np.log(p) + (1 - y) * np.log(1 - p)))
    boots = d[rng.integers(0, len(d), (n, len(d)))].mean(axis=1)
    return d.mean(), np.percentile(boots, [2.5, 97.5])


def main():
    RES.mkdir(exist_ok=True)
    MOD.mkdir(exist_ok=True)
    feats, pm, xi, pidx, M = load_all()
    y = feats["a_won"].to_numpy()
    season = feats["season"].to_numpy()
    seasons = sorted(set(season))
    test_s, val_s = seasons[-5:], seasons[-11:-5]
    rng = np.random.default_rng(7)
    print(f"Tune on seasons {val_s[0]}-{val_s[-1]}, test once on {test_s[0]}-{test_s[-1]}\n")

    # 1) tune Elo ------------------------------------------------------------------------
    best_elo = None
    for k in ELO_K_GRID:
        for carry in ELO_CARRY_GRID:
            d, _ = elo_run(feats, k, carry)
            p = elo_walk(d, y, season, val_s)
            m = ~np.isnan(p)
            ll = log_loss(y[m], p[m])
            if best_elo is None or ll < best_elo[0]:
                best_elo = (ll, k, carry)
    _, elo_k, elo_carry = best_elo
    print(f"Elo tuned: K={elo_k}, season carry-over={elo_carry}  (validation log loss {best_elo[0]:.4f})")

    # 2) tune player ridge ----------------------------------------------------------------
    best_r = None
    for hl in HALF_LIFE_GRID:
        for C in C_GRID:
            p = ridge_walk(M, y, season, val_s, hl, C)
            m = ~np.isnan(p)
            ll = log_loss(y[m], p[m])
            if best_r is None or ll < best_r[0]:
                best_r = (ll, hl, C)
    _, half_life, C = best_r
    print(f"Player model tuned: half-life={half_life} seasons, C={C}  (validation log loss {best_r[0]:.4f})")

    # 3) tune blend + shrinkage -----------------------------------------------------------
    diffs, elo_final = elo_run(feats, elo_k, elo_carry)
    pe_val = elo_walk(diffs, y, season, val_s)
    pr_val = ridge_walk(M, y, season, val_s, half_life, C)
    m = ~np.isnan(pe_val) & ~np.isnan(pr_val)
    best_b = None
    for w in BLEND_W_GRID:
        for sh in SHRINK_GRID:
            ll = log_loss(y[m], blend_prob(pe_val[m], pr_val[m], w, sh))
            if best_b is None or ll < best_b[0]:
                best_b = (ll, w, sh)
    _, blend_w, shrink = best_b
    print(f"Blend tuned: weight on player model={blend_w}, shrink factor={shrink}  "
          f"(validation log loss {best_b[0]:.4f}; coin flip = 0.6931)\n")

    # 4) one-shot test --------------------------------------------------------------------
    pe = elo_walk(diffs, y, season, test_s)
    pr = ridge_walk(M, y, season, test_s, half_life, C)
    t = ~np.isnan(pe) & ~np.isnan(pr)
    yt = y[t]
    preds = {"v2 Elo (tuned)": pe[t], "v2 Player model": pr[t],
             "v2 Blend (final)": blend_prob(pe[t], pr[t], blend_w, shrink)}
    rows = []
    print(f"TEST on {t.sum()} unseen matches ({test_s[0]}-{test_s[-1]}):")
    print(f"{'model':<20}{'acc':>7}{'logloss':>9}{'brier':>8}   gain vs coin flip (95% CI)")
    for name, p in preds.items():
        acc, marg, ll, br = metrics(yt, p)
        g, (lo, hi) = gain_vs_coinflip(yt, p, rng)
        verdict = "significantly better" if lo > 0 else ("significantly WORSE" if hi < 0 else "not significant")
        print(f"{name:<20}{acc:>7.3f}{ll:>9.4f}{br:>8.4f}   {g:+.4f} [{lo:+.4f}, {hi:+.4f}]  {verdict}")
        rows.append({"model": name, "accuracy": acc, "acc_95%_margin": marg, "log_loss": ll, "brier": br})

    path = RES / "model_comparison.csv"
    old = pd.read_csv(path) if path.exists() else pd.DataFrame(
        [{"model": "Coin flip", "accuracy": 0.5, "acc_95%_margin": 1.96 * 0.5 / np.sqrt(len(yt)),
          "log_loss": np.log(2), "brier": 0.25}])
    old = old[~old["model"].isin(preds)]
    pd.concat([old, pd.DataFrame(rows)], ignore_index=True).to_csv(path, index=False)

    # 5) final fit on everything + save ---------------------------------------------------
    now = max(seasons) + 1
    coef = fit_ridge(M, y, decay(season, now, half_life), C)
    json.dump({"elo_k": elo_k, "elo_carry": elo_carry, "half_life": half_life, "C": C,
               "blend_w": blend_w, "shrink": shrink}, open(MOD / "v2_params.json", "w"), indent=2)

    info = pd.read_csv(DATA / "player_ratings_latest.csv")[["pkey", "player", "last_team", "last_season"]]
    n_matches = pm.groupby("pkey").size().rename("matches")
    imp = pd.DataFrame({"pkey": list(pidx), "impact": coef}).merge(info, on="pkey", how="left")
    imp = imp.merge(n_matches, left_on="pkey", right_index=True, how="left")
    imp = imp.sort_values("impact", ascending=False)
    imp.to_csv(RES / "player_impact.csv", index=False)

    cur = imp[imp["last_season"] == imp["last_season"].max()]
    cols = ["player", "last_team", "matches", "impact"]
    print("\nTop 10 current players by adjusted impact (credit is shared with team-mates, so read with care):")
    print(cur.head(10)[cols].round(3).to_string(index=False))
    print(f"\nSaved -> {path}, {RES/'player_impact.csv'}, {MOD/'v2_params.json'}")


if __name__ == "__main__":
    main()

"""
Step 9 - Skill-based IPL match model (v3).

Each player is rated from ALL his men's T20 cricket (step 8), walking forward in time so a match only ever
uses games played before it:
    batting  : runs above average per ball, and dismissals above average per ball
    bowling  : runs saved per ball vs average, and wickets above average per ball
All rates are measured against the average of the competition and date they came from, shrunk towards zero
for small samples (empirical Bayes), and old games fade (exponential half-life).
A team's strength is its XI's expected contribution per match (rate x expected balls faced / bowled).
Features for a match = Team A minus Team B for those four numbers, plus Elo and home advantage.

Tuned on 2016-2021 (half-life, weight of non-IPL games, regularisation), tested once on 2022-2026.
Ablations show what the skill ratings add over Elo + home advantage alone.

Outputs: results/model_comparison.csv (rows added), models/v3.pkl (used by steps 10 and 11)
Usage:   python 09_t20_model.py        (several minutes: the whole T20 history is replayed 6 times)
"""
import datetime
import importlib
import json
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

m6 = importlib.import_module("06_player_model")
DATA, RES, MOD = Path("data"), Path("results"), Path("models")

HL_GRID = (500, 1100)                 # half-life of old games, in days
W_OTH_GRID = (0.25, 0.6, 1.0)         # weight of non-IPL games relative to IPL games
C_GRID = (0.03, 0.1, 0.3, 1.0)        # logistic regularisation
K_BAT, K_OUT, K_BOWL, K_WK = 250, 400, 450, 400   # shrinkage (pseudo-balls)
PRIOR_BF, PRIOR_BB, M0 = 11.0, 11.0, 3.0           # default balls faced / bowled per match, prior weight
PRIORS = {"bat_rpb": 1.30, "out_pb": 0.055, "bowl_rpb": 1.40, "wk_pb": 0.045}
P_AVG = 2000.0                                     # prior weight (balls) for competition averages
FEATS = ["elo", "bat_runs", "bat_outs", "bowl_saved", "bowl_wk", "home"]
NEUTRAL_SEASONS = {2009, 2014, 2020, 2021, 2022}   # played abroad / in bubbles: no real home venues


# ------------------------------------------------------------------ ratings --------------
def comp_rates(ca):
    """Competition averages so far (with a small prior): batter runs/ball, outs/ball, conceded/ball, wkts/ball."""
    return ((ca[0] + PRIORS["bat_rpb"] * P_AVG) / (ca[1] + P_AVG),
            (ca[2] + PRIORS["out_pb"] * P_AVG) / (ca[1] + P_AVG),
            (ca[3] + PRIORS["bowl_rpb"] * P_AVG) / (ca[4] + P_AVG),
            (ca[5] + PRIORS["wk_pb"] * P_AVG) / (ca[4] + P_AVG))


def get_state(states, p, d, hl):
    st = states.get(p)
    if st is None:
        return None
    if d != st[0]:
        f = 0.5 ** ((d - st[0]) / hl)
        for k in range(1, 8):
            st[k] *= f
        st[0] = d
    return st


def team_vec(pids, d, states, hl):
    """Expected (runs above avg, outs above avg, runs saved, wickets above avg) per match for an XI."""
    br = bo = bs = bw = 0.0
    for p in pids:
        st = get_state(states, p, d, hl)
        if st is None:
            continue
        _, m, bf, er, eo, bb, sv, ew = st
        ebf = (bf + PRIOR_BF * M0) / (m + M0)
        ebb = (bb + PRIOR_BB * M0) / (m + M0)
        br += ebf * er / (bf + K_BAT)
        bo += ebf * eo / (bf + K_OUT)
        bs += ebb * sv / (bb + K_BOWL)
        bw += ebb * ew / (bb + K_WK)
    return np.array([br, bo, bs, bw])


def prep_pool():
    pool = pd.read_csv(DATA / "t20_player_matches.csv")
    days = {s: datetime.date.fromisoformat(s).toordinal() for s in pool["date"].unique()}
    pool["day"] = pool["date"].map(days)
    pool = pool.sort_values(["day", "key"], kind="stable").reset_index(drop=True)
    keys = pool["key"].to_numpy()
    starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
    ends = np.r_[starts[1:], len(keys)]
    cols = {c: pool[c].tolist() for c in ["key", "comp", "day", "team", "player_id",
                                          "balls", "runs", "out", "bb", "rc", "wk"]}
    return cols, starts, ends


def run_pass(pool, feat_keys, hl, w_oth):
    """Replay all T20 history once. Returns (n_ipl_matches x 4 feature diffs, final player states)."""
    P, starts, ends = pool
    V = np.zeros((len(feat_keys), 4))
    states, tot = {}, defaultdict(lambda: [0.0] * 6)
    for s, e in zip(starts, ends):
        comp, d, key = P["comp"][s], P["day"][s], P["key"][s]
        wt = 1.0 if comp == "ipl" else w_oth
        info = feat_keys.get(key)
        if info is not None:
            i, ta, tb = info
            xa = [P["player_id"][r] for r in range(s, e) if P["team"][r] == ta]
            xb = [P["player_id"][r] for r in range(s, e) if P["team"][r] == tb]
            V[i] = team_vec(xa, d, states, hl) - team_vec(xb, d, states, hl)
        t = tot[comp]
        rb, ob, rbw, wkb = comp_rates(t)
        sums = [0.0] * 6
        for r in range(s, e):
            p = P["player_id"][r]
            st = states.get(p)
            if st is None:
                st = states[p] = [d, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            else:
                get_state(states, p, d, hl)
            balls, runs, out, bb, rc, wk = (P[c][r] for c in ("balls", "runs", "out", "bb", "rc", "wk"))
            st[1] += wt
            st[2] += wt * balls
            st[3] += wt * (runs - rb * balls)
            st[4] += wt * (out - ob * balls)
            st[5] += wt * bb
            st[6] += wt * (rbw * bb - rc)
            st[7] += wt * (wk - wkb * bb)
            sums[0] += runs; sums[1] += balls; sums[2] += out; sums[3] += rc; sums[4] += bb; sums[5] += wk
        for k in range(6):
            t[k] += sums[k]
    return V, states


# ------------------------------------------------------------------ features / model -----
def home_flags(f):
    """+1 if Team A is playing at its home ground, -1 if Team B is, else 0 (fixtures are known in advance)."""
    cnt = defaultdict(int)
    for s, v, a, b in zip(f["season"], f["venue"], f["team_a"], f["team_b"]):
        cnt[(s, a, v)] += 1
        cnt[(s, b, v)] += 1
    best = {}
    for (s, t, v), c in cnt.items():
        if c > best.get((s, t), ("", 0))[1]:
            best[(s, t)] = (v, c)
    claim = defaultdict(list)
    for (s, t), (v, c) in best.items():
        if c >= 5 and s not in NEUTRAL_SEASONS:
            claim[(s, v)].append(t)
    home = {(s, t): v for (s, v), ts in claim.items() if len(ts) == 1 for t in ts}
    h = np.zeros(len(f))
    for i, (s, v, a, b) in enumerate(zip(f["season"], f["venue"], f["team_a"], f["team_b"])):
        if home.get((s, a)) == v:
            h[i] = 1.0
        elif home.get((s, b)) == v:
            h[i] = -1.0
    return h


def fit_lr(X, y, C):
    m = make_pipeline(StandardScaler(with_mean=False),
                      LogisticRegression(C=C, fit_intercept=False, max_iter=3000))
    return m.fit(X, y)


def raw_beta(m):
    return m[-1].coef_[0] / m[0].scale_


def walk(X, y, season, eval_s, C):
    p = np.full(len(y), np.nan)
    for s in eval_s:
        tr, te = season < s, season == s
        if tr.sum() < 50:
            continue
        p[te] = fit_lr(X[tr], y[tr], C).predict_proba(X[te])[:, 1]
    return p


def main():
    RES.mkdir(exist_ok=True)
    MOD.mkdir(exist_ok=True)
    feats = pd.read_csv(DATA / "features.csv")
    y, season = feats["a_won"].to_numpy(), feats["season"].to_numpy()
    seasons = sorted(set(season))
    test_s, val_s = seasons[-5:], seasons[-11:-5]
    rng = np.random.default_rng(11)

    pj = MOD / "v2_params.json"
    ep = json.load(open(pj)) if pj.exists() else {"elo_k": 10, "elo_carry": 0.5}
    elo_d, elo_final = m6.elo_run(feats, ep["elo_k"], ep["elo_carry"])
    home = home_flags(feats)
    print(f"Home ground identified in {(home != 0).mean():.0%} of matches; "
          f"home team wins {np.mean(np.where(home[home != 0] > 0, y[home != 0], 1 - y[home != 0])):.3f} of those.")

    pool = prep_pool()
    feat_keys = {f"ipl_{mid}": (i, a, b) for i, (mid, a, b) in
                 enumerate(zip(feats["match_id"], feats["team_a"], feats["team_b"]))}
    keyset = set(pool[0]["key"])
    if not any(k in keyset for k in feat_keys):
        raise SystemExit("IPL matches not found in data/t20_player_matches.csv - run 08_t20_data.py first.")

    print(f"\nTuning on {val_s[0]}-{val_s[-1]} (each setting replays the whole T20 history) ...")
    cache, best = {}, None
    for hl in HL_GRID:
        for w in W_OTH_GRID:
            V, states = run_pass(pool, feat_keys, hl, w)
            X = np.column_stack([elo_d / 100.0, V, home])
            cache[(hl, w)] = (X, states)
            for C in C_GRID:
                p = walk(X, y, season, val_s, C)
                m = ~np.isnan(p)
                ll = m6.log_loss(y[m], np.clip(p[m], 1e-6, 1 - 1e-6))
                print(f"  half-life {hl:>4}d  non-IPL weight {w:<4}  C {C:<5} -> validation log loss {ll:.4f}")
                if best is None or ll < best[0]:
                    best = (ll, hl, w, C)
    vll, hl, w, C = best
    print(f"\nBest on validation: half-life {hl}d, non-IPL weight {w}, C {C}  (log loss {vll:.4f}; coin flip 0.6931)")

    X, states = cache[(hl, w)]
    t = np.isin(season, test_s)
    yt = y[t]
    variants = {
        "v3 skill model (all T20)": list(range(6)),
        "v3 ablation: Elo + home only": [0, 5],
        "v3 ablation: player skills only": [1, 2, 3, 4],
    }
    rows = []
    print(f"\nTEST on {t.sum()} unseen matches ({test_s[0]}-{test_s[-1]}):")
    print(f"{'model':<34}{'acc':>7}{'logloss':>9}{'brier':>8}   gain vs coin flip (95% CI)")
    for name, cols in variants.items():
        p = walk(X[:, cols], y, season, test_s, C)[t]
        acc, marg, ll, br = m6.metrics(yt, p)
        g, (lo, hi) = m6.gain_vs_coinflip(yt, p, rng)
        verdict = "significantly better" if lo > 0 else ("significantly WORSE" if hi < 0 else "not significant")
        print(f"{name:<34}{acc:>7.3f}{ll:>9.4f}{br:>8.4f}   {g:+.4f} [{lo:+.4f}, {hi:+.4f}]  {verdict}")
        rows.append({"model": name, "accuracy": acc, "acc_95%_margin": marg, "log_loss": ll, "brier": br})
    path = RES / "model_comparison.csv"
    old = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["model", "accuracy", "acc_95%_margin", "log_loss", "brier"])
    old = old[~old["model"].isin(variants)]
    pd.concat([old, pd.DataFrame(rows)], ignore_index=True).to_csv(path, index=False)

    final = fit_lr(X, y, C)
    beta = raw_beta(final)
    coefs = pd.DataFrame({"feature": FEATS, "std_coef": final[-1].coef_[0], "raw_coef": beta})
    print("\nFinal model coefficients (std_coef: positive = helps Team A; expected signs: + - + + +):")
    print(coefs.round(3).to_string(index=False))

    joblib.dump({"params": {"hl": hl, "w_oth": w, "C": C}, "beta": beta, "feats": FEATS, "states": states,
                 "X": X, "y": y, "season": season,
                 "elo": {"k": ep["elo_k"], "carry": ep["elo_carry"], "final": elo_final}},
                MOD / "v3.pkl")
    print(f"\nSaved -> {path}, {MOD/'v3.pkl'}")


if __name__ == "__main__":
    main()

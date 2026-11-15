"""
Step 13 - Final attempt, with the rules fixed BEFORE looking at any result.

Why simpler: models with 6+ fitted parameters on ~900 matches lose about 0.003 log loss to estimation noise,
probably more than the real signal is worth. So every candidate here has 1-3 parameters.

The four candidates (no others will be tried):
    C1  home advantage only
    C2  Elo with extra shrinkage at mega-auction seasons (2011, 2014, 2018, 2022, 2025)
    C3  one "expected run margin" number from the all-T20 skill ratings
        (batting runs above average - 12 x dismissals above average + bowling runs saved + 12 x wickets above average)
    C4  all three together

Test: walk-forward over EVERY season from 2012 (train only on earlier seasons, predict the next).
Success rule: the gain in log loss over a coin flip must be positive even with a 98.75% bootstrap interval
(95% corrected for testing 4 candidates). Anything else is reported as "no edge shown".

Needs: models/v3.pkl (from 09_t20_model.py).   Usage: python src/13_final_attempt.py
"""
import importlib
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

m9 = importlib.import_module("09_t20_model")
m6 = m9.m6
DATA, RES, MOD = Path("data"), Path("results"), Path("models")
MEGA = {2011, 2014, 2018, 2022, 2025}
WICKET_RUNS = 12.0
FIRST_EVAL = 2012
N_CANDIDATES = 4
C_REG = 1.0


def elo_reset(f, k=10, carry=0.5, mega_carry=0.25):
    elo = defaultdict(lambda: 1500.0)
    diffs = np.empty(len(f))
    prev = None
    for i, (s, a, b, w) in enumerate(zip(f["season"], f["team_a"], f["team_b"], f["a_won"])):
        if prev is not None and s != prev:
            c = mega_carry if s in MEGA else carry
            for t in list(elo):
                elo[t] = 1500.0 + c * (elo[t] - 1500.0)
        prev = s
        diffs[i] = elo[a] - elo[b]
        e = 1.0 / (1.0 + 10 ** ((elo[b] - elo[a]) / 400.0))
        elo[a] += k * (w - e)
        elo[b] -= k * (w - e)
    return diffs


def gain_ci(y, p, rng, alpha, n=4000):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    d = np.log(2) - (-(y * np.log(p) + (1 - y) * np.log(1 - p)))
    boots = d[rng.integers(0, len(d), (n, len(d)))].mean(axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return d.mean(), lo, hi


def main():
    RES.mkdir(exist_ok=True)
    feats = pd.read_csv(DATA / "features.csv")
    y, season = feats["a_won"].to_numpy(), feats["season"].to_numpy()
    seasons = sorted(set(season))
    eval_s = [s for s in seasons if s >= FIRST_EVAL]
    last5 = seasons[-5:]
    X3 = joblib.load(MOD / "v3.pkl")["X"]          # columns: elo, bat_runs, bat_outs, bowl_saved, bowl_wk, home
    margin = X3[:, 1] - WICKET_RUNS * X3[:, 2] + X3[:, 3] + WICKET_RUNS * X3[:, 4]
    elo_r = elo_reset(feats) / 100.0
    home = X3[:, 5]
    cands = {
        "C1 home advantage only": home[:, None],
        "C2 Elo with mega-auction resets": elo_r[:, None],
        "C3 skill margin (run-equivalent)": (margin / 10.0)[:, None],
        "C4 all three": np.column_stack([home, elo_r, margin / 10.0]),
    }
    rng = np.random.default_rng(13)
    alpha_b = 0.05 / N_CANDIDATES
    rows, winners = [], []
    print(f"Walk-forward over seasons {eval_s[0]}-{eval_s[-1]}; 4 candidates; success needs the "
          f"{100*(1-alpha_b):.2f}% interval above zero.\n")
    print(f"{'candidate':<34}{'matches':>8}{'acc':>7}{'logloss':>9}{'gain':>9}  {'95% CI':<20}{'98.75% CI':<20}{'last 5 seasons'}")
    for name, Xc in cands.items():
        p = m9.walk(Xc, y, season, eval_s, C_REG)
        m = ~np.isnan(p)
        g, lo95, hi95 = gain_ci(y[m], p[m], rng, 0.05)
        _, lo_b, hi_b = gain_ci(y[m], p[m], rng, alpha_b)
        pl = p[np.isin(season, last5)]
        yl = y[np.isin(season, last5)]
        g5 = gain_ci(yl, pl, rng, 0.05)[0]
        acc = float(np.mean((p[m] > 0.5) == (y[m] == 1)))
        ll = float(np.log(2) - g)
        print(f"{name:<34}{m.sum():>8}{acc:>7.3f}{ll:>9.4f}{g:>+9.4f}  "
              f"[{lo95:+.4f},{hi95:+.4f}]  [{lo_b:+.4f},{hi_b:+.4f}]  {g5:+.4f}")
        rows.append({"candidate": name, "matches": int(m.sum()), "accuracy": acc, "log_loss": ll,
                     "gain_vs_coinflip": g, "ci95_lo": lo95, "ci95_hi": hi95,
                     "ci9875_lo": lo_b, "ci9875_hi": hi_b, "gain_last5": g5})
        if lo_b > 0:
            winners.append(name)
    pd.DataFrame(rows).to_csv(RES / "final_attempt.csv", index=False)

    print("\nVERDICT (rule fixed in advance):")
    if winners:
        print("  SUCCESS - beats a coin flip after correcting for 4 candidates:", ", ".join(winners))
    else:
        nominal = [r["candidate"] for r in rows if r["ci95_lo"] > 0]
        print("  No candidate beats a coin flip once the 4 tests are corrected for.")
        if nominal:
            print("  (Nominally significant at 95% but not after correction - treat as a hint only:",
                  ", ".join(nominal) + ")")
    for name in ("C1 home advantage only", "C4 all three"):
        Xc = cands[name]
        mdl = m9.fit_lr(Xc, y, C_REG)
        print(f"  fitted coefficients, {name}: {np.round(m9.raw_beta(mdl), 3)}")
    print(f"\nSaved -> {RES/'final_attempt.csv'}")


if __name__ == "__main__":
    main()

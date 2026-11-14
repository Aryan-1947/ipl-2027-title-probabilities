"""
Step 12 - Two quick diagnostics after the v3 result.

  A) Face-validity check of the player ratings: do the best bowlers/batters by rating look like the best
     players? If yes, the rating code is fine and the wrong-signed coefficients are noise, not a bug.
  B) The simplest possible model: "the home team is slightly favoured" and nothing else, tested with the
     same walk-forward protocol (train on earlier seasons, predict the next).

Usage:  python 12_diagnostics.py
"""
import importlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

m9 = importlib.import_module("09_t20_model")
m6 = m9.m6
DATA, RES, MOD = Path("data"), Path("results"), Path("models")


def main():
    # ---- A) ratings face validity ------------------------------------------------------------
    bundle = joblib.load(MOD / "v3.pkl")
    names = pd.read_csv(RES / "player_impact.csv").set_index("pkey")["player"].to_dict()
    rows = [(names[p], st[5], st[6] / (st[5] + m9.K_BOWL), st[7] / (st[5] + m9.K_WK),
             st[3] / (st[2] + m9.K_BAT), st[4] / (st[2] + m9.K_OUT))
            for p, st in bundle["states"].items() if p in names and st[5] > 150]
    df = pd.DataFrame(rows, columns=["player", "balls_bowled_w", "runs_saved_per_ball",
                                     "wkts_above_avg_per_ball", "bat_runs_above_per_ball", "bat_outs_above_per_ball"])
    print("BOWLERS - best 10 by runs saved per ball (players with >150 weighted balls bowled):")
    print(df.sort_values("runs_saved_per_ball", ascending=False).head(10)[
        ["player", "runs_saved_per_ball", "wkts_above_avg_per_ball"]].round(3).to_string(index=False))
    print("\nBOWLERS - worst 5:")
    print(df.sort_values("runs_saved_per_ball").head(5)[
        ["player", "runs_saved_per_ball", "wkts_above_avg_per_ball"]].round(3).to_string(index=False))
    bat = df.dropna()
    print("\nBATTERS - best 10 by runs above average per ball:")
    print(bat.sort_values("bat_runs_above_per_ball", ascending=False).head(10)[
        ["player", "bat_runs_above_per_ball", "bat_outs_above_per_ball"]].round(3).to_string(index=False))

    # ---- B) home advantage only --------------------------------------------------------------
    feats = pd.read_csv(DATA / "features.csv")
    y, season = feats["a_won"].to_numpy(), feats["season"].to_numpy()
    seasons = sorted(set(season))
    test_s = seasons[-5:]
    home = m9.home_flags(feats)
    X = home.reshape(-1, 1)
    t = np.isin(season, test_s)
    p = m9.walk(X, y, season, test_s, 1.0)[t]
    acc, marg, ll, br = m6.metrics(y[t], p)
    g, (lo, hi) = m6.gain_vs_coinflip(y[t], p, np.random.default_rng(3))
    verdict = "significantly better" if lo > 0 else ("significantly WORSE" if hi < 0 else "not significant")
    played_home = home[t] != 0
    hw = np.mean(np.where(home[t][played_home] > 0, y[t][played_home], 1 - y[t][played_home]))
    print(f"\nHOME-ONLY model on {t.sum()} test matches ({test_s[0]}-{test_s[-1]}): "
          f"accuracy {acc:.3f}, log loss {ll:.4f}, gain vs coin flip {g:+.4f} [{lo:+.4f}, {hi:+.4f}] {verdict}")
    print(f"Home side won {hw:.3f} of the {played_home.sum()} test matches with a clear home ground.")


if __name__ == "__main__":
    main()

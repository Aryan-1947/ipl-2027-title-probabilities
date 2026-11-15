"""
Step 11 - Forward test: the only fully honest test of a predictor.

How it fits together
  15_fetch_schedule.py  writes fixtures_2027.csv as soon as the official schedule exists
  14_update_squads.py   keeps squads_2027.csv current (trades, auction buys)
  11_forward_test.py    freezes predictions BEFORE the season, then scores them against real results

1. BEFORE the first ball (needs fixtures_2027.csv and the current squads):
       python src/11_forward_test.py predict
   Writes predictions_2027.csv (win probability for team_a in every fixture, home edge applied where the venue
   is the host's home ground) and predictions_2027_squads.csv (the exact squads used). COMMIT AND PUSH BOTH TO
   GITHUB before the season starts - the commit timestamp proves they were made in advance. An existing
   predictions file is never overwritten unless you pass --force.

2. DURING / AFTER the season: refresh results (delete data/raw, re-run 01_data_cleaning.py), then
       python src/11_forward_test.py score
   Prints accuracy, log loss and the gain over a coin flip with a bootstrap 95% interval.
   Use --year to rehearse on a past season (needs fixtures_<year>.csv).

Usage:  python src/11_forward_test.py predict|score [--year 2027] [--force]
"""
import argparse
import datetime
import hashlib
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

import engine as E

DATA, RES = Path("data"), Path("results")


def pair_key(date, a, b):
    return f"{date}|" + "|".join(sorted([a, b]))


def predict(year, force):
    fix_path, pred_path = Path(f"live/fixtures_{year}.csv"), Path(f"live/predictions_{year}.csv")
    if not fix_path.exists():
        raise SystemExit(f"{fix_path} not found. Run 15_fetch_schedule.py (or create fixtures_{year}_manual.csv) first.")
    if pred_path.exists() and not force:
        raise SystemExit(f"{pred_path} already exists. Frozen predictions are never overwritten (use --force).")
    fx = pd.read_csv(fix_path)
    sq = E.load_squads()
    model = E.Model(n_boot=0)
    stage = fx["stage"].fillna("") if "stage" in fx else pd.Series("", index=fx.index)
    out = fx[["date", "team_a", "team_b"]].copy()
    out["venue"] = fx["venue"] if "venue" in fx else ""
    out["home"] = fx["home"].fillna("") if "home" in fx else ""
    out["stage"] = stage
    out["p_team_a_win"] = np.round(model.fixture_probs(sq, fx), 4)
    out = out.dropna(subset=["p_team_a_win"])
    out["model"] = "v2 Elo + player-impact blend, home edge 0.15 logit; squads as of creation"
    out["created_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    out["squads_sha1"] = hashlib.sha1(Path(E.SQUADS).read_bytes()).hexdigest()[:12]
    out.to_csv(pred_path, index=False)
    shutil.copy(E.SQUADS, f"live/predictions_{year}_squads.csv")
    print(f"Saved {len(out)} frozen predictions -> {pred_path} (squads fingerprint {out['squads_sha1'].iloc[0]})")
    print("Now commit and push predictions and squads files to GitHub BEFORE the season starts.")


def score(year):
    pred_path = Path(f"live/predictions_{year}.csv")
    if not pred_path.exists():
        raise SystemExit(f"{pred_path} not found - run 'predict' first.")
    pr = pd.read_csv(pred_path)
    m = pd.read_csv(DATA / "matches.csv")
    m = m[(m["season"] == year) & m["winner"].notna()].copy()
    m["key"] = [pair_key(d, a, b) for d, a, b in zip(m["date"], m["team1"], m["team2"])]
    pr["key"] = [pair_key(d, a, b) for d, a, b in zip(pr["date"], pr["team_a"], pr["team_b"])]
    j = pr.merge(m[["key", "winner"]], on="key", how="inner")
    if j.empty:
        raise SystemExit("No played matches found yet for these predictions (refresh the results first).")
    y = (j["winner"] == j["team_a"]).astype(int).to_numpy()
    p = np.clip(j["p_team_a_win"].to_numpy(), 1e-6, 1 - 1e-6)
    ll = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    j["correct"] = ((p > 0.5) == (y == 1)).astype(int)
    j["log_loss"] = ll
    RES.mkdir(exist_ok=True)
    j.drop(columns=["key"]).to_csv(RES / f"forward_test_{year}.csv", index=False)
    g, (lo, hi) = E.m6.gain_vs_coinflip(y, p, np.random.default_rng(1))
    verdict = "significantly better" if lo > 0 else ("significantly WORSE" if hi < 0 else "not significant yet")
    print(f"Forward test {year}: {len(j)} of {len(pr)} predicted matches played")
    print(f"  accuracy          {j['correct'].mean():.3f}   (coin flip 0.500)")
    print(f"  log loss          {ll.mean():.4f}   (coin flip 0.6931)")
    print(f"  gain vs coin flip {g:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  {verdict}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["predict", "score"])
    ap.add_argument("--year", type=int, default=2027)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    predict(a.year, a.force) if a.mode == "predict" else score(a.year)

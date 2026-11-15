"""
Step 3 - Predict match winners and compare models honestly.

Validation is walk-forward by season: for each of the last 5 seasons, train ONLY on earlier
seasons and predict that season. No random splits, no peeking.

Models compared
    Coin flip      always 50%
    Elo only       logistic regression on Elo difference
    Logistic       logistic regression on all pre-toss features
    XGBoost        small, heavily regularised gradient-boosted trees

Outputs
    results/model_comparison.csv   accuracy / log loss / Brier for every model
    results/lr_coefficients.csv    what drives the logistic model
    results/xgb_importance.csv     what drives XGBoost
    models/match_model.joblib      final models trained on ALL matches (used in step 4)

Usage:
    python src/03_match_model.py
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from common import CORE, FEATURES, fit_lr, fit_xgb, make_X, predict_lr, predict_xgb

DATA, RES, MOD = Path("data"), Path("results"), Path("models")
N_TEST_SEASONS = 5


def main() -> None:
    RES.mkdir(exist_ok=True)
    MOD.mkdir(exist_ok=True)

    feats = pd.read_csv(DATA / "features.csv")
    X = make_X(feats)
    y = feats["a_won"].to_numpy()
    seasons = sorted(feats["season"].unique())
    test_seasons = [s for s in seasons[-N_TEST_SEASONS:] if s > seasons[0]]

    names = ["Coin flip", "Elo only", "Logistic (all features)", "Logistic (core 3)", "XGBoost"]
    pred = {n: np.full(len(feats), np.nan) for n in names}

    for s in test_seasons:
        tr = (feats["season"] < s).to_numpy()
        te = (feats["season"] == s).to_numpy()
        pred["Coin flip"][te] = 0.5
        elo_m = fit_lr(X[tr], y[tr], cols=["elo_diff"])
        pred["Elo only"][te] = predict_lr(elo_m, X[te], cols=["elo_diff"])
        pred["Logistic (all features)"][te] = predict_lr(fit_lr(X[tr], y[tr]), X[te])
        pred["Logistic (core 3)"][te] = predict_lr(fit_lr(X[tr], y[tr], cols=CORE), X[te], cols=CORE)
        pred["XGBoost"][te] = predict_xgb(fit_xgb(X[tr], y[tr]), X[te])

    mask = feats["season"].isin(test_seasons).to_numpy()
    yt = y[mask]
    rows = []
    for n in names:
        p = np.clip(pred[n][mask], 1e-6, 1 - 1e-6)
        acc = accuracy_score(yt, p > 0.5) if n != "Coin flip" else 0.5
        se = np.sqrt(acc * (1 - acc) / len(yt))
        rows.append({"model": n, "accuracy": acc, "acc_95%_margin": 1.96 * se,
                     "log_loss": log_loss(yt, p), "brier": brier_score_loss(yt, p)})
    table = pd.DataFrame(rows)
    table.to_csv(RES / "model_comparison.csv", index=False)

    print(f"Walk-forward test: seasons {test_seasons[0]}-{test_seasons[-1]}  ({mask.sum()} matches)")
    print("Lower log loss / Brier = better. Coin flip log loss is 0.693.\n")
    print(table.round(3).to_string(index=False))

    print("\nPer-season accuracy:")
    per = pd.DataFrame({n: pd.Series(
        (pred[n][mask] > 0.5) == yt).groupby(feats.loc[mask, "season"].to_numpy()).mean()
        for n in names[1:]})
    print(per.round(3).to_string())

    # ---- final models on everything + interpretation ----
    lr_final, xgb_final = fit_lr(X, y), fit_xgb(X, y)
    joblib.dump({"lr": lr_final, "xgb": xgb_final, "features": FEATURES}, MOD / "match_model.joblib")

    coefs = pd.DataFrame({"feature": FEATURES, "coef": lr_final[-1].coef_[0]})
    coefs = coefs.reindex(coefs["coef"].abs().sort_values(ascending=False).index)
    coefs.to_csv(RES / "lr_coefficients.csv", index=False)
    imp = pd.DataFrame({"feature": FEATURES, "importance": xgb_final.feature_importances_})
    imp.sort_values("importance", ascending=False).to_csv(RES / "xgb_importance.csv", index=False)

    print("\nLogistic coefficients (standardised; positive = helps Team A):")
    print(coefs.round(3).to_string(index=False))
    print("\nXGBoost feature importance:")
    print(imp.sort_values("importance", ascending=False).round(3).to_string(index=False))
    print(f"\nSaved -> {RES}/ and {MOD/'match_model.joblib'}")


if __name__ == "__main__":
    main()

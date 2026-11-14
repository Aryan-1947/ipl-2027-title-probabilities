"""Shared feature definitions and model helpers (used by 03_match_model.py and 04_season_sim.py)."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Pre-toss features only: a season simulation has to predict matches before any toss happens.
# Every feature is "Team A minus Team B", so swapping the teams just flips the sign.
CORE = ["elo_diff", "bat_diff", "wicket_diff"]
FEATURES = ["elo_diff", "form_diff", "h2h_edge", "venue_wr_diff", "bat_diff", "wicket_diff", "econ_diff"]


def make_X(f: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame({
        "elo_diff": f["elo_diff"] / 100.0,
        "form_diff": f["form_a"] - f["form_b"],
        "h2h_edge": f["h2h_a"] - 0.5,
        "venue_wr_diff": f["venue_wr_a"] - f["venue_wr_b"],
        "bat_diff": f["bat_diff"],
        "wicket_diff": f["wicket_diff"],
        "econ_diff": f["econ_diff"],
    }, index=f.index)
    return X[FEATURES]


def _augment(X, y):
    """Add the mirror image of every match (A<->B) so the model can't learn an A/B bias."""
    return pd.concat([X, -X]), np.concatenate([np.asarray(y), 1 - np.asarray(y)])


def fit_lr(X, y, cols=None):
    cols = cols or FEATURES
    Xa, ya = _augment(X[cols], y)
    m = make_pipeline(StandardScaler(with_mean=False),
                      LogisticRegression(C=0.3, fit_intercept=False, max_iter=1000))
    m.fit(Xa, ya)
    return m


def predict_lr(m, X, cols=None):
    return m.predict_proba(X[cols or FEATURES])[:, 1]


def fit_xgb(X, y):
    Xa, ya = _augment(X[FEATURES], y)
    m = XGBClassifier(n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8,
                      colsample_bytree=0.8, min_child_weight=5, reg_lambda=5.0,
                      eval_metric="logloss", random_state=42, n_jobs=1)
    m.fit(Xa, ya)
    return m


def predict_xgb(m, X):
    X = X[FEATURES]
    return 0.5 * (m.predict_proba(X)[:, 1] + 1 - m.predict_proba(-X)[:, 1])  # symmetrised

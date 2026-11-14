"""
engine.py - the forecasting engine behind the Streamlit app, the CLI scripts and the forward test.

Model: the published v2 model (tuned Elo + player-impact ridge, blended; see steps 6-7) plus a small fixed
home-advantage term. IMPORTANT: no model in this project has beaten a coin flip on unseen seasons, so every
number this engine produces is a model-implied lean with uncertainty, not a validated forecast.

Two season formats are simulated:
  * generic  - 70 league games (5 opponents twice, 4 once) drawn at random, used until the real schedule is out
  * fixtures - the real league fixtures from fixtures_2027.csv (written by 15_fetch_schedule.py)
"""
import importlib
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

m1 = importlib.import_module("01_data_cleaning")
m6 = importlib.import_module("06_player_model")
m7 = importlib.import_module("07_season_sim_v2")

DATA, RES, MOD = Path("data"), Path("results"), Path("models")
SQUADS = Path("squads_2027.csv")
BASELINE = DATA / "squads_baseline_2026.csv"
FIXTURES = Path("fixtures_2027.csv")
HOME_LOGIT = 0.15          # ~53.7% for an evenly matched home side (historical average; not confirmed in 2022-26)
PRECISION = {"Fast": (8, 400), "Standard": (20, 600), "Full": (30, 1000)}   # (model re-fits, seasons per re-fit)


# ------------------------------------------------------------------ venues / schedule --
def _squash(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def venue_match(a, b):
    a, b = _squash(m1.norm_venue(str(a))), _squash(m1.norm_venue(str(b)))
    return bool(a and b and (a == b or (min(len(a), len(b)) >= 8 and (a in b or b in a))))


def home_venues(feats, last_n=2, min_games=5):
    """Each team's home grounds = venues where it played >= min_games in the last seasons."""
    seasons = sorted(feats["season"].unique())[-last_n:]
    f = feats[feats["season"].isin(seasons)]
    cnt = defaultdict(int)
    for s, v, a, b in zip(f["season"], f["venue"], f["team_a"], f["team_b"]):
        cnt[(s, a, v)] += 1
        cnt[(s, b, v)] += 1
    out = defaultdict(set)
    for (s, t, v), c in cnt.items():
        if c >= min_games:
            out[t].add(v)
    return out


def generic_schedule():
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
    return {"A": np.array(A), "B": np.array(B), "H": np.array(H), "mode": "generic", "n": 70}


def schedule_from_fixtures(fx, teams, hv):
    """League fixtures -> index arrays. A team is 'at home' only if the venue is one of its known home grounds."""
    idx = {t: i for i, t in enumerate(teams)}
    stage = fx["stage"].fillna("") if "stage" in fx else pd.Series([""] * len(fx), index=fx.index)
    league = fx[~stage.str.contains("qualifier|eliminator|final", case=False, regex=True)]
    A, B, H = [], [], []
    for r in league.itertuples():
        if r.team_a not in idx or r.team_b not in idx:
            continue
        h = 0
        host = getattr(r, "home", None)
        venue = getattr(r, "venue", None)
        if isinstance(host, str) and host in (r.team_a, r.team_b) and isinstance(venue, str):
            if any(venue_match(venue, v) for v in hv.get(host, ())):
                h = 1 if host == r.team_a else -1
        A.append(idx[r.team_a]); B.append(idx[r.team_b]); H.append(h)
    return {"A": np.array(A), "B": np.array(B), "H": np.array(H), "mode": "fixtures", "n": len(A)}


def fixtures_complete(fx, teams):
    """Every team needs >= 14 league games for the schedule to be used in the simulation."""
    if fx is None or len(fx) == 0:
        return False
    stage = fx["stage"].fillna("") if "stage" in fx else pd.Series([""] * len(fx), index=fx.index)
    lg = fx[~stage.str.contains("qualifier|eliminator|final", case=False, regex=True)]
    counts = pd.concat([lg["team_a"], lg["team_b"]]).value_counts()
    return all(counts.get(t, 0) >= 14 for t in teams)


# ------------------------------------------------------------------ simulation -----------
def simulate(P, sched, n_sims, rng):
    T = P.shape[0]
    A, B, H = sched["A"], sched["B"], sched["H"]
    Pc = np.clip(P, 1e-4, 1 - 1e-4)
    Z = np.log(Pc / (1 - Pc))
    generic = sched["mode"] == "generic"
    title, final, top4, wins = (np.zeros(T) for _ in range(4))
    play = lambda x, y: x if rng.random() < P[x, y] else y
    for _ in range(n_sims):
        if generic:
            perm = rng.permutation(T)
            ta, tb = perm[A], perm[B]
            h = np.where(H == 0, rng.choice([-1, 1], size=len(A)), H)
        else:
            ta, tb, h = A, B, H
        p = 1.0 / (1.0 + np.exp(-(Z[ta, tb] + HOME_LOGIT * h)))
        winners = np.where(rng.random(len(A)) < p, ta, tb)
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


# ------------------------------------------------------------------ model ----------------
class Model:
    """Fits the v2 model once (plus bootstrap re-fits for uncertainty). Reuse it for any squads / fixtures."""

    def __init__(self, n_boot=20, seed=2027):
        import json
        self.params = json.load(open(MOD / "v2_params.json"))
        feats, _pm, _xi, self.pidx, self.M = m6.load_all()
        self.feats = feats
        self.y = feats["a_won"].to_numpy()
        self.season = feats["season"].to_numpy()
        self.now = int(self.season.max()) + 1
        self.diffs, elo_final = m6.elo_run(feats, self.params["elo_k"], self.params["elo_carry"])
        self.teams = sorted(pd.read_csv(DATA / "team_ratings_latest.csv")["team"])
        if len(self.teams) != 10:
            raise RuntimeError(f"Expected 10 active teams, found {len(self.teams)}")
        carry = self.params["elo_carry"]
        self.elo = np.array([1500.0 + carry * (elo_final.get(t, 1500.0) - 1500.0) for t in self.teams])
        self.hv = home_venues(feats)
        rng = np.random.default_rng(seed)
        self.fits = [m7.fit_components(self.diffs, self.y, self.M, self.season, self.params, self.now)]
        for _ in range(n_boot):
            idx = rng.integers(0, len(self.y), len(self.y))
            self.fits.append(m7.fit_components(self.diffs, self.y, self.M, self.season, self.params, self.now, idx))

    # -- helpers -------------------------------------------------------------------------
    def _impact(self, squads, coef):
        xi = squads[squads["xi"] == 1]
        out = []
        for t in self.teams:
            keys = xi.loc[xi["team"] == t, "pkey"]
            out.append(sum(coef[self.pidx[k]] for k in keys if k in self.pidx))
        return np.array(out)

    def _P(self, squads, fit):
        elo_lr, coef = fit
        return m7.win_matrix(elo_lr, coef, self.elo, self._impact(squads, coef), self.params)

    def ratings(self, squads):
        imp = self._impact(squads, self.fits[0][1])
        return pd.DataFrame({"team": self.teams, "elo_2027": self.elo.round(1), "squad_impact": imp.round(3)})

    def fixture_probs(self, squads, fx):
        """Point-estimate win probability of team_a for every fixture (home edge applied where known)."""
        P = self._P(squads, self.fits[0])
        idx = {t: i for i, t in enumerate(self.teams)}
        Pc = np.clip(P, 1e-4, 1 - 1e-4)
        out = []
        for r in fx.itertuples():
            if r.team_a not in idx or r.team_b not in idx:
                out.append(np.nan)
                continue
            i, j = idx[r.team_a], idx[r.team_b]
            h = 0
            host, venue = getattr(r, "home", None), getattr(r, "venue", None)
            if isinstance(host, str) and host in (r.team_a, r.team_b) and isinstance(venue, str):
                if any(venue_match(venue, v) for v in self.hv.get(host, ())):
                    h = 1 if host == r.team_a else -1
            z = np.log(Pc[i, j] / (1 - Pc[i, j])) + HOME_LOGIT * h
            out.append(1 / (1 + np.exp(-z)))
        return np.array(out)

    # -- forecast ------------------------------------------------------------------------
    def forecast(self, squads, fixtures=None, sims_per_fit=500, seed=7):
        use_fx = fixtures is not None and fixtures_complete(fixtures, self.teams)
        sched = schedule_from_fixtures(fixtures, self.teams, self.hv) if use_fx else generic_schedule()
        rng = np.random.default_rng(seed)
        fits = self.fits[1:] if len(self.fits) > 1 else self.fits
        runs = [simulate(self._P(squads, f), sched, sims_per_fit, rng) for f in fits]
        stack = lambda c: np.vstack([r[c].to_numpy() for r in runs])
        pt = stack("p_title")
        res = pd.DataFrame({
            "team": self.teams, "p_title": pt.mean(0),
            "p_title_lo": np.percentile(pt, 5, axis=0), "p_title_hi": np.percentile(pt, 95, axis=0),
            "p_final": stack("p_final").mean(0), "p_top4": stack("p_top4").mean(0),
            "avg_wins": stack("avg_wins").mean(0)}).sort_values("p_title", ascending=False).reset_index(drop=True)
        return {"results": res, "ratings": self.ratings(squads), "schedule_mode": sched["mode"],
                "n_league_matches": sched["n"], "n_fits": len(fits), "sims": len(fits) * sims_per_fit}


def load_squads(path=SQUADS):
    sq = pd.read_csv(path)
    if "xi" not in sq:
        raise RuntimeError(f"{path} has no 'xi' column")
    return sq


def load_fixtures(path=FIXTURES):
    return pd.read_csv(path) if Path(path).exists() else None

"""
Step 2 - Build a leak-free, pre-match feature table.

Every feature for a match is computed ONLY from matches played before it (walk-forward),
so a model trained on this table can't "see the future".

Outputs
    data/features.csv               one row per decided match, from a randomly-oriented "Team A" view
    data/team_ratings_latest.csv    Elo of each team after the last match in the data
    data/player_ratings_latest.csv  batting / bowling value of every player after the last match
                                    (used in step 4 to rate the 2027 squads)

Usage:
    python src/02_features.py
"""
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("data")

# ---- tunable knobs (worth mentioning in your README) -------------------------------------
ELO_START = 1500
ELO_K = 20
ELO_CARRY = 0.75       # at each new season, keep 75% of a team's Elo deviation (squads change)
PLAYER_DECAY = 0.80    # at each new season, old player stats count 80% as much (form ages)
FORM_N = 5             # last N matches for the form feature
BAT_PRIOR, BAT_W = 15.0, 10      # league-avg runs per player per match, prior weight (matches)
WK_PRIOR, WK_W = 0.5, 10         # league-avg wickets per player per match
ECON_PRIOR_BALLS, ECON_PRIOR = 60, 8.5
SEED = 42


def expected(ea: float, eb: float) -> float:
    return 1.0 / (1.0 + 10 ** ((eb - ea) / 400.0))


def shrink(wins: float, n: float, prior: float = 0.5, w: float = 4.0) -> float:
    """Win rate pulled towards `prior` when there is little data."""
    return (wins + prior * w) / (n + w)


def bat_val(p: dict) -> float:
    return (p["runs"] + BAT_PRIOR * BAT_W) / (p["m"] + BAT_W)


def wk_val(p: dict) -> float:
    return (p["wk"] + WK_PRIOR * WK_W) / (p["m"] + WK_W)


def new_player() -> dict:
    return {"m": 0.0, "runs": 0.0, "wk": 0.0, "balls": 0.0, "rc": 0.0}


def xi_strength(pids, players) -> tuple:
    """Return (batting strength, wicket-taking strength, pooled economy) for an XI."""
    bats = sorted((bat_val(players[p]) for p in pids), reverse=True)[:7]
    wks = sorted((wk_val(players[p]) for p in pids), reverse=True)[:5]
    rc = sum(players[p]["rc"] for p in pids)
    balls = sum(players[p]["balls"] for p in pids)
    econ = 6 * (rc + ECON_PRIOR * ECON_PRIOR_BALLS / 6) / (balls + ECON_PRIOR_BALLS)
    return sum(bats), sum(wks), econ


def main() -> None:
    matches = pd.read_csv(DATA / "matches.csv")
    pm = pd.read_csv(DATA / "player_matches.csv")
    pm["pkey"] = pm["player_id"].fillna(pm["player"])
    matches = matches.sort_values(["date", "match_id"]).reset_index(drop=True)

    xi = {k: g["pkey"].tolist() for k, g in pm.groupby(["match_id", "team"])}
    lines = {k: g for k, g in pm.groupby("match_id")}
    names = pm.drop_duplicates("pkey", keep="last").set_index("pkey")["player"].to_dict()

    rng = np.random.default_rng(SEED)
    elo = defaultdict(lambda: float(ELO_START))
    form = defaultdict(lambda: deque(maxlen=FORM_N))
    h2h = defaultdict(lambda: deque(maxlen=10))
    venue_team = defaultdict(lambda: [0, 0])      # (venue, team) -> [wins, played]
    venue_bf = defaultdict(lambda: [0, 0])        # venue -> [bat-first wins, played]
    players = defaultdict(new_player)
    last_team = {}
    prev_season = None
    rows = []

    for m in matches.itertuples(index=False):
        if prev_season is not None and m.season != prev_season:
            for t in list(elo):
                elo[t] = ELO_START + ELO_CARRY * (elo[t] - ELO_START)
            for p in players.values():
                for k in p:
                    p[k] *= PLAYER_DECAY
        prev_season = m.season

        a, b = (m.team1, m.team2) if rng.random() < 0.5 else (m.team2, m.team1)
        decided = isinstance(m.winner, str) and m.winner in (a, b)

        if decided:
            xa, xb = xi.get((m.match_id, a), []), xi.get((m.match_id, b), [])
            bat_a, wk_a, econ_a = xi_strength(xa, players)
            bat_b, wk_b, econ_b = xi_strength(xb, players)

            hist = [w for w in h2h[frozenset((a, b))]]
            h2h_a = shrink(sum(w == a for w in hist), len(hist))
            fa, fb = form[a], form[b]
            va, vb = venue_team[(m.venue, a)], venue_team[(m.venue, b)]
            vbf = venue_bf[m.venue]
            vbf_rate = shrink(vbf[0], vbf[1])

            toss_known = isinstance(m.toss_winner, str) and isinstance(m.toss_decision, str)
            if toss_known:
                bat_first = m.toss_winner if m.toss_decision == "bat" else (b if m.toss_winner == a else a)
            else:
                bat_first = None
            a_bats_first = 1 if bat_first == a else 0

            rows.append({
                "match_id": m.match_id, "date": m.date, "season": m.season, "venue": m.venue,
                "team_a": a, "team_b": b, "is_playoff": int(isinstance(m.stage, str)),
                "elo_a": elo[a], "elo_b": elo[b], "elo_diff": elo[a] - elo[b],
                "form_a": shrink(sum(fa), len(fa), w=2), "form_b": shrink(sum(fb), len(fb), w=2),
                "h2h_a": h2h_a,
                "venue_wr_a": shrink(va[0], va[1]), "venue_wr_b": shrink(vb[0], vb[1]),
                "toss_won_by_a": int(m.toss_winner == a) if toss_known else 0,
                "toss_bat": int(m.toss_decision == "bat") if toss_known else 0,
                "a_bats_first": a_bats_first if toss_known else 0,
                "bat_first_venue_edge_a": (vbf_rate - 0.5) * (1 if a_bats_first else -1) if toss_known else 0.0,
                "bat_diff": bat_a - bat_b,
                "wicket_diff": wk_a - wk_b,
                "econ_diff": econ_b - econ_a,   # positive = A's XI has historically bowled more cheaply
                "a_won": int(m.winner == a),
            })

            # ---- update team-level state AFTER recording the features ----
            ea = expected(elo[a], elo[b])
            res = 1.0 if m.winner == a else 0.0
            elo[a] += ELO_K * (res - ea)
            elo[b] -= ELO_K * (res - ea)
            form[a].append(int(m.winner == a))
            form[b].append(int(m.winner == b))
            h2h[frozenset((a, b))].append(m.winner)
            va[0] += int(m.winner == a); va[1] += 1
            vb[0] += int(m.winner == b); vb[1] += 1
            if bat_first is not None:
                vbf[0] += int(m.winner == bat_first); vbf[1] += 1

        # player stats update for every match with a scorecard (even no-results)
        g = lines.get(m.match_id)
        if g is not None:
            for r in g.itertuples(index=False):
                p = players[r.pkey]
                p["m"] += 1
                p["runs"] += r.runs
                p["wk"] += r.wickets
                p["balls"] += r.balls_bowled
                p["rc"] += r.runs_conceded
                last_team[r.pkey] = (r.team, m.season)

    feats = pd.DataFrame(rows)
    feats.to_csv(DATA / "features.csv", index=False)

    # Only franchises that played in the latest season (drops Kochi, Pune Warriors, Gujarat Lions, ...)
    latest = matches[matches["season"] == matches["season"].max()]
    active = set(latest["team1"]) | set(latest["team2"])
    pd.DataFrame(
        sorted(((t, e) for t, e in elo.items() if t in active), key=lambda kv: -kv[1]),
        columns=["team", "elo"],
    ).to_csv(DATA / "team_ratings_latest.csv", index=False)

    pr = pd.DataFrame([
        {"pkey": k, "player": names.get(k, k), "last_team": last_team[k][0], "last_season": last_team[k][1],
         "matches_weighted": v["m"], "bat_val": bat_val(v), "wk_val": wk_val(v),
         "balls_bowled": v["balls"], "runs_conceded": v["rc"],
         "runs_weighted": v["runs"], "wk_weighted": v["wk"]}
        for k, v in players.items() if k in last_team
    ])
    pr.to_csv(DATA / "player_ratings_latest.csv", index=False)

    # ---- sanity checks: this is what you'll quote in the README ----
    print(f"features: {feats.shape[0]} matches x {feats.shape[1]} columns "
          f"({feats['season'].min()}-{feats['season'].max()})")
    later = feats[feats["season"] >= feats["season"].min() + 2]
    print(f"\nBaseline 'higher Elo wins' accuracy (excluding first 2 seasons): "
          f"{((later['elo_diff'] > 0).astype(int) == later['a_won']).mean():.3f}")
    recent = feats[feats["season"] >= feats["season"].max() - 1]
    print(f"Same baseline on the last 2 seasons: "
          f"{((recent['elo_diff'] > 0).astype(int) == recent['a_won']).mean():.3f}")
    tk = feats[feats["toss_won_by_a"].notna()]
    toss_win = np.where(tk["toss_won_by_a"] == 1, tk["a_won"], 1 - tk["a_won"]).mean()
    print(f"Toss winner wins the match: {toss_win:.3f}   (close to 0.50 = toss barely matters)")
    print("\nCurrent Elo ranking:")
    print(pd.read_csv(DATA / "team_ratings_latest.csv").round(0).to_string(index=False))
    print(f"\nSaved -> {DATA/'features.csv'}, team_ratings_latest.csv, player_ratings_latest.csv")


if __name__ == "__main__":
    main()

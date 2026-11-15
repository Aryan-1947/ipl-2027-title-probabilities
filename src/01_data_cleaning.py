"""
Step 1 - Download Cricsheet IPL ball-by-ball data and flatten it into two tidy CSVs.

    data/matches.csv          one row per match
    data/player_matches.csv   one row per player per match (batting + bowling line)

Usage:
    python src/01_data_cleaning.py

Data source: https://cricsheet.org/downloads/ (free, ball-by-ball, ODbL licence - credit them in your README).
"""
import io
import json
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

URL = "https://cricsheet.org/downloads/ipl_json.zip"
DATA = Path("data")
RAW = DATA / "raw"

# Franchises that were simply renamed. (Deccan Chargers, Gujarat Lions, Pune Warriors, Kochi etc.
# are different/defunct franchises, so they are deliberately NOT merged into anyone.)
TEAM_ALIASES = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Rising Pune Supergiants": "Rising Pune Supergiant",
}

# Stadiums appear under several spellings. We strip the ", City" suffix first, then apply this.
# Run the script once and eyeball the printed venue list - add anything that still looks duplicated.
VENUE_ALIASES = {
    "M.Chinnaswamy Stadium": "M Chinnaswamy Stadium",
    "Feroz Shah Kotla": "Arun Jaitley Stadium",
    "Punjab Cricket Association Stadium": "IS Bindra Stadium",
    "Punjab Cricket Association IS Bindra Stadium": "IS Bindra Stadium",
    "Sardar Patel Stadium": "Narendra Modi Stadium",
    "Rajiv Gandhi International Stadium": "Rajiv Gandhi Intl Stadium",
    "Rajiv Gandhi International Cricket Stadium": "Rajiv Gandhi Intl Stadium",
    "Dr DY Patil Sports Academy": "DY Patil Stadium",
    "Subrata Roy Sahara Stadium": "Maharashtra Cricket Association Stadium",
    "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium": "ACA-VDCA Stadium",
}

# Wicket types that are not credited to the bowler.
NOT_BOWLER_WICKET = {"run out", "retired hurt", "retired out", "obstructing the field"}


def norm_team(name: str) -> str:
    return TEAM_ALIASES.get(name, name)


def norm_venue(name: str) -> str:
    base = name.split(",")[0].strip()
    return VENUE_ALIASES.get(base, base)


def download() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    if any(RAW.glob("*.json")):
        print(f"Using cached files in {RAW}/ (delete the folder to re-download)")
        return
    print("Downloading Cricsheet IPL data ...")
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        payload = r.read()
    zipfile.ZipFile(io.BytesIO(payload)).extractall(RAW)
    print(f"Extracted {len(list(RAW.glob('*.json')))} match files")


def parse_match(path: Path):
    with open(path, encoding="utf-8") as f:
        m = json.load(f)
    info = m["info"]
    match_id = path.stem
    teams = [norm_team(t) for t in info["teams"]]
    outcome = info.get("outcome", {})
    by = outcome.get("by", {})
    toss = info.get("toss", {})
    event = info.get("event", {})
    registry = info.get("registry", {}).get("people", {})
    date = info["dates"][0]

    match_row = {
        "match_id": match_id,
        "date": date,
        # Season = calendar year of first match date (Cricsheet writes things like "2007/08" for 2008).
        "season": int(date[:4]),
        "venue": norm_venue(info.get("venue", "Unknown")),
        "city": info.get("city"),
        "team1": teams[0],
        "team2": teams[1],
        "toss_winner": norm_team(toss["winner"]) if "winner" in toss else None,
        "toss_decision": toss.get("decision"),
        "winner": norm_team(outcome["winner"]) if "winner" in outcome else None,
        "result": outcome.get("result", "normal"),  # normal / tie / no result
        "win_by_runs": by.get("runs"),
        "win_by_wickets": by.get("wickets"),
        "dls": int(outcome.get("method") == "D/L"),
        "stage": event.get("stage"),  # Final / Qualifier 1 / Eliminator ... (NaN in the league phase)
        "player_of_match": (info.get("player_of_match") or [None])[0],
    }

    # ---- per-player lines -------------------------------------------------------------
    stats = {}

    def get(name, team):
        if name not in stats:
            stats[name] = {
                "match_id": match_id, "date": date, "season": match_row["season"],
                "team": team, "player": name, "player_id": registry.get(name),
                "runs": 0, "balls": 0, "dismissed": 0,
                "balls_bowled": 0, "runs_conceded": 0, "wickets": 0,
            }
        return stats[name]

    for team_raw, names in info.get("players", {}).items():
        for n in names:
            get(n, norm_team(team_raw))

    for inn in m.get("innings", []):
        if inn.get("super_over"):
            continue
        bat_team = norm_team(inn["team"])
        bowl_team = teams[1] if teams[0] == bat_team else teams[0]
        for over in inn.get("overs", []):
            for d in over["deliveries"]:
                extras = d.get("extras", {})
                runs = d["runs"]
                b = get(d["batter"], bat_team)
                b["runs"] += runs["batter"]
                if "wides" not in extras:
                    b["balls"] += 1
                bw = get(d["bowler"], bowl_team)
                bw["runs_conceded"] += runs["total"] - extras.get("byes", 0) - extras.get("legbyes", 0)
                if "wides" not in extras and "noballs" not in extras:
                    bw["balls_bowled"] += 1
                for w in d.get("wickets", []):
                    if w["kind"] != "retired hurt":
                        get(w["player_out"], bat_team)["dismissed"] = 1
                    if w["kind"] not in NOT_BOWLER_WICKET:
                        bw["wickets"] += 1

    return match_row, list(stats.values())


def main() -> None:
    download()
    files = sorted(RAW.glob("*.json"))
    if not files:
        raise SystemExit("No JSON files found in data/raw/")

    matches, players = [], []
    for p in files:
        try:
            mr, pr = parse_match(p)
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            print(f"  skipped {p.name}: {e!r}")
            continue
        matches.append(mr)
        players.extend(pr)

    matches = pd.DataFrame(matches).sort_values(["date", "match_id"]).reset_index(drop=True)
    players = pd.DataFrame(players)
    players["batted"] = ((players["balls"] > 0) | (players["dismissed"] > 0)).astype(int)

    DATA.mkdir(exist_ok=True)
    matches.to_csv(DATA / "matches.csv", index=False)
    players.to_csv(DATA / "player_matches.csv", index=False)

    print("\n=== Summary ===")
    print(f"matches: {len(matches)}  |  player-match rows: {len(players)}")
    print(f"seasons: {matches['season'].min()} - {matches['season'].max()}")
    print("\nmatches per season:")
    print(matches.groupby("season").size().to_string())
    print("\nno-result / tie rows:", (matches["winner"].isna()).sum())
    print("\nTop venues (check for duplicate spellings and add them to VENUE_ALIASES):")
    print(matches["venue"].value_counts().head(40).to_string())
    print(f"\nSaved -> {DATA/'matches.csv'}, {DATA/'player_matches.csv'}")


if __name__ == "__main__":
    main()

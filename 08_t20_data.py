"""
Step 8 - Pool ball-by-ball player lines from ALL men's T20 cricket on Cricsheet
(IPL, T20 internationals, BBL, PSL, CPL, SA20, ILT20, MLC, Blast, ...).

Cricsheet gives every player the same id in every competition, so a player's form anywhere in the world
can count towards his rating - including new auction signings who have little or no IPL history.

Output: data/t20_player_matches.csv   one row per player per match
Usage:  python 08_t20_data.py          (first run downloads several hundred MB and takes a few minutes;
                                        competitions that fail to download are skipped and reported)
"""
import importlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

m1 = importlib.import_module("01_data_cleaning")   # reuse its team-name fixes and IPL download folder
DATA = Path("data")
RAW_T20 = DATA / "raw_t20"
URL = "https://cricsheet.org/downloads/{}_json.zip"

# our label -> Cricsheet file name. If a name is wrong it is simply skipped - you can add more.
COMPS = {
    "ipl": "ipl", "t20i": "t20s", "bbl": "bbl", "psl": "psl", "cpl": "cpl", "blast": "ntb",
    "sa20": "sat", "ilt20": "ilt", "mlc": "mlc", "lpl": "lpl", "bpl": "bpl", "smash": "ssm",
    "smat": "smat",
}


def folder(comp: str) -> Path:
    return m1.RAW if comp == "ipl" else RAW_T20 / comp


def fetch(comp: str, name: str) -> bool:
    d = folder(comp)
    if any(d.glob("*.json")):
        return True
    d.mkdir(parents=True, exist_ok=True)
    try:
        req = urllib.request.Request(URL.format(name), headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=600) as r:
            payload = r.read()
        zipfile.ZipFile(io.BytesIO(payload)).extractall(d)
        return True
    except Exception as e:  # noqa: BLE001 - any failure just skips this competition
        print(f"  {comp}: could not download ({e!r}) - skipped")
        return False


def parse(path: Path, comp: str) -> list:
    try:
        with open(path, encoding="utf-8") as f:
            m = json.load(f)
    except (OSError, json.JSONDecodeError):
        return []
    info = m.get("info", {})
    if info.get("gender") != "male" or info.get("match_type") not in ("T20", "IT20") or not m.get("innings"):
        return []
    teams = [m1.norm_team(t) for t in info.get("teams", [])]
    if len(teams) != 2:
        return []
    reg = info.get("registry", {}).get("people", {})
    date = info["dates"][0]
    key = f"{comp}_{path.stem}"
    stats = {}

    def get(name, team):
        if name not in stats:
            stats[name] = {"key": key, "comp": comp, "date": date, "team": team, "player_id": reg.get(name, name),
                           "balls": 0, "runs": 0, "out": 0, "bb": 0, "rc": 0, "wk": 0}
        return stats[name]

    for team_raw, names in info.get("players", {}).items():
        for n in names:
            get(n, m1.norm_team(team_raw))

    for inn in m["innings"]:
        if inn.get("super_over"):
            continue
        bat = m1.norm_team(inn["team"])
        bowl = teams[1] if teams[0] == bat else teams[0]
        for over in inn.get("overs", []):
            for d in over["deliveries"]:
                ex, runs = d.get("extras", {}), d["runs"]
                b = get(d["batter"], bat)
                b["runs"] += runs["batter"]
                if "wides" not in ex:
                    b["balls"] += 1
                bw = get(d["bowler"], bowl)
                bw["rc"] += runs["total"] - ex.get("byes", 0) - ex.get("legbyes", 0)
                if "wides" not in ex and "noballs" not in ex:
                    bw["bb"] += 1
                for w in d.get("wickets", []):
                    if w["kind"] != "retired hurt":
                        get(w["player_out"], bat)["out"] = 1
                    if w["kind"] not in m1.NOT_BOWLER_WICKET:
                        bw["wk"] += 1
    return list(stats.values())


def main() -> None:
    rows = []
    for comp, name in COMPS.items():
        if not fetch(comp, name):
            continue
        files = sorted(folder(comp).glob("*.json"))
        n_before = len(rows)
        for p in files:
            rows.extend(parse(p, comp))
        n_matches = len({r["key"] for r in rows[n_before:]})
        print(f"  {comp:<6} {n_matches:>5} men's T20 matches")
    if not rows:
        raise SystemExit("No data found.")
    df = pd.DataFrame(rows).sort_values(["date", "key"]).reset_index(drop=True)
    df.to_csv(DATA / "t20_player_matches.csv", index=False)
    print(f"\nTotal: {df['key'].nunique()} matches, {df['player_id'].nunique()} players, "
          f"{df['date'].min()} to {df['date'].max()}")
    print(f"Saved -> {DATA/'t20_player_matches.csv'}")


if __name__ == "__main__":
    main()

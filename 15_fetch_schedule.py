"""
Step 15 - Check whether the IPL 2027 schedule has been released and, if so, save it as fixtures_2027.csv.

Source 1  the official IPL match-schedule feed (https://scores.iplt20.com/ipl/feeds/<competition id>-matchschedule.js).
          The competition id of the 2027 season is not known in advance, so the script probes ids and keeps the
          one whose matches are dated 2027 (cached in data/ipl_comp_id.json). Pass --comp-id N to skip probing.
Source 2  a file you create by hand: fixtures_2027_manual.csv with columns
              date,team_a,team_b,venue[,home,stage]      (team_a = home team if known; stage blank for league games)
Not released yet -> nothing is written and the app keeps using its generic fixture model.
A partial release (a few weeks of matches) is saved but NOT used for the forecast until every team has 14 games.

Outputs: fixtures_2027.csv, data/schedule_status.json
Usage:   python 15_fetch_schedule.py [--force] [--comp-id N]
"""
import argparse
import datetime
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

DATA = Path("data")
FIXTURES, MANUAL = Path("fixtures_2027.csv"), Path("fixtures_2027_manual.csv")
STATUS, COMP_CACHE = DATA / "schedule_status.json", DATA / "ipl_comp_id.json"
HOSTS = ["https://scores.iplt20.com/ipl/feeds/{}-matchschedule.js",
         "https://ipl-stats-sports-mechanic.s3.ap-south-1.amazonaws.com/ipl/feeds/{}-matchschedule.js"]
PROBE_RANGE = range(150, 451)
RECHECK_HOURS = 12
YEAR = 2027
HEADERS = {"User-Agent": "Mozilla/5.0 (ipl-2027-predictions student project)"}

TEAMS = {
    "Chennai Super Kings": ["csk"], "Delhi Capitals": ["dc", "delhi daredevils"],
    "Gujarat Titans": ["gt"], "Kolkata Knight Riders": ["kkr"], "Lucknow Super Giants": ["lsg"],
    "Mumbai Indians": ["mi"], "Punjab Kings": ["pbks", "kxip", "kings xi punjab"],
    "Rajasthan Royals": ["rr"], "Royal Challengers Bengaluru": ["rcb", "royal challengers bangalore"],
    "Sunrisers Hyderabad": ["srh"],
}
ALIAS = {}
for full, others in TEAMS.items():
    for k in [full.lower()] + others:
        ALIAS[k] = full


def canon(name):
    return ALIAS.get(re.sub(r"\s+", " ", str(name)).strip().lower())


# ------------------------------------------------------------------ feed parsing ---------
def parse_feed(text):
    """JSONP -> list of match dicts (first list of dicts found in the payload)."""
    m = re.search(r"\((\{.*\}|\[.*\])\)\s*;?\s*$", text.strip(), re.S)
    payload = json.loads(m.group(1) if m else text)
    if isinstance(payload, dict):
        for v in payload.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return v
        return []
    return payload if isinstance(payload, list) else []


def _first(d, keys):
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return None


def to_fixtures(rows):
    out = []
    for d in rows:
        raw_date = _first(d, ["MATCH_COMMENCE_START_DATE", "MatchDateNew", "MatchDate", "match_date", "date"])
        dt = pd.to_datetime(re.sub(r"^[A-Za-z]+,\s*", "", str(raw_date)), errors="coerce", dayfirst=False) if raw_date else pd.NaT
        home = canon(_first(d, ["HomeTeamName", "HomeTeam", "home_team"]) or _first(d, ["HomeTeamCode"]) or "")
        away = canon(_first(d, ["AwayTeamName", "AwayTeam", "away_team"]) or _first(d, ["AwayTeamCode"]) or "")
        if not (home and away):
            home = canon(_first(d, ["FirstBattingTeamName", "team_a"]) or "")
            away = canon(_first(d, ["SecondBattingTeamName", "team_b"]) or "")
        order = str(_first(d, ["MatchOrder", "MatchName", "stage"]) or "")
        stage = order if re.search(r"qualifier|eliminator|final", order, re.I) else ""
        out.append({"date": dt.strftime("%Y-%m-%d") if pd.notna(dt) else None, "team_a": home, "team_b": away,
                    "venue": _first(d, ["GroundName", "Venue", "venue"]) or "", "home": home or "", "stage": stage,
                    "match_no": order})
    return pd.DataFrame(out)


def valid_fixtures(df, year=YEAR):
    df = df.dropna(subset=["date", "team_a", "team_b"])
    df = df[(df["team_a"] != "") & (df["team_b"] != "") & (df["date"].str[:4] == str(year))]
    return df.sort_values("date").reset_index(drop=True)


# ------------------------------------------------------------------ fetching -------------
def _fetch(comp_id):
    import requests
    for host in HOSTS:
        try:
            r = requests.get(host.format(comp_id), headers=HEADERS, timeout=15)
            if r.status_code == 200 and len(r.text) > 50:
                return comp_id, r.text
        except Exception:  # noqa: BLE001
            continue
    return comp_id, None


def find_feed(comp_id=None):
    if comp_id is None and COMP_CACHE.exists():
        comp_id = json.load(open(COMP_CACHE)).get("id")
    if comp_id is not None:
        _, text = _fetch(comp_id)
        if text:
            fx = valid_fixtures(to_fixtures(parse_feed(text)))
            if len(fx):
                return comp_id, fx
    with ThreadPoolExecutor(max_workers=8) as ex:
        for cid, text in ex.map(_fetch, PROBE_RANGE):
            if not text:
                continue
            try:
                fx = valid_fixtures(to_fixtures(parse_feed(text)))
            except Exception:  # noqa: BLE001
                continue
            if len(fx):
                return cid, fx
    return None, None


def write_status(status, **kw):
    DATA.mkdir(exist_ok=True)
    kw.update({"status": status, "checked_at": datetime.datetime.now().isoformat(timespec="seconds")})
    json.dump(kw, open(STATUS, "w"), indent=2)
    return kw


def fetch_schedule(force=False, comp_id=None):
    if not force and STATUS.exists():
        last = json.load(open(STATUS))
        age = (datetime.datetime.now() - datetime.datetime.fromisoformat(last["checked_at"])).total_seconds() / 3600
        if last.get("status") == "not_released" and age < RECHECK_HOURS and not MANUAL.exists():
            last["message"] = f"Checked {age:.1f}h ago: still not released (re-check after {RECHECK_HOURS}h or use --force)"
            return last
    source, fx = None, None
    if MANUAL.exists():
        fx = valid_fixtures(_manual())
        source = "manual file" if len(fx) else None
    if fx is None or not len(fx):
        try:
            cid, fx = find_feed(comp_id)
        except ImportError:
            return write_status("error", message="Install requests: pip install requests")
        source = f"official feed (competition id {cid})" if cid is not None else None
        if cid is not None:
            json.dump({"id": cid, "year": YEAR}, open(COMP_CACHE, "w"))
    if fx is None or not len(fx):
        return write_status("not_released", matches=0,
                            message=f"No {YEAR} schedule found yet - the app uses its generic fixture model.")
    fx.to_csv(FIXTURES, index=False)
    stage = fx["stage"].fillna("")
    league = fx[~stage.str.contains("qualifier|eliminator|final", case=False, regex=True)]
    counts = pd.concat([league["team_a"], league["team_b"]]).value_counts()
    complete = len(counts) == 10 and counts.min() >= 14
    return write_status("complete" if complete else "partial", matches=int(len(fx)), league_matches=int(len(league)),
                        source=source, first_date=fx["date"].min(), last_date=fx["date"].max(),
                        message=("Full league schedule loaded - the forecast now uses the real fixtures." if complete else
                                 "Partial schedule saved; the forecast keeps the generic model until every team has 14 games."))


def _manual():
    m = pd.read_csv(MANUAL)
    m["team_a"], m["team_b"] = m["team_a"].map(canon), m["team_b"].map(canon)
    if "home" in m:
        m["home"] = m["home"].map(lambda x: canon(x) or "")
    else:
        m["home"] = m["team_a"]
    for c in ("venue", "stage"):
        if c not in m:
            m[c] = ""
    m["match_no"] = ""
    m["date"] = pd.to_datetime(m["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return m[["date", "team_a", "team_b", "venue", "home", "stage", "match_no"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="ignore the 12-hour re-check limit")
    ap.add_argument("--comp-id", type=int, default=None)
    a = ap.parse_args()
    res = fetch_schedule(a.force, a.comp_id)
    print(f"[{res['status']}] {res.get('message', '')}")
    if res.get("matches"):
        print(f"{res['matches']} matches saved to {FIXTURES}  ({res.get('first_date')} -> {res.get('last_date')})")


if __name__ == "__main__":
    main()

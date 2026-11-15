import importlib
import json
import re

"""
Step 14 - Update the 2027 squads (trades, releases, auction buys) and coaches.

Sources, in this order of priority
  1. LIVE      Wikipedia team pages (current squad + coaching staff). Used for a team only if the table parses
               and looks sane (>= 14 players, >= 60% recognised from IPL history).
  2. BASELINE  the 2026 squads (data/squads_baseline_2026.csv, created from your step-4 file the first time).
  3. LEDGER    squad_moves.csv - confirmed trades / releases / signings, applied on top of 1 or 2.
               Idempotent: a move that is already reflected in the squad does nothing, so a fresh Wikipedia
               page and the ledger never double-count.  Rows with status 'proposed' are ignored unless you pass
               --include-proposed.   Coaches: coaches.csv wins for the roles it lists; live data fills the rest.

Everything fetched is best-effort and logged. If the internet or a page layout fails, you simply keep the
baseline + ledger squads. Names that cannot be matched to an IPL player are listed in data/squad_unmatched.csv
(fix them in name_aliases.csv: columns alias,player where player is the Cricsheet name, e.g. "CV Varun").
Unknown players (e.g. new overseas signings) are rated as average and start on the bench unless you set xi=1
in the ledger.

Outputs: squads_2027.csv, data/coaches.csv, data/snapshots/, results/squad_changes_log.csv
Usage:   python src/14_update_squads.py [--no-live] [--include-proposed] [--dry-run]
"""
import argparse
import datetime
import importlib
import re
import shutil
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

DATA, RES = Path("data"), Path("results")
SQUADS = Path("live/squads_2027.csv")
BASELINE = DATA / "squads_baseline_2026.csv"
MOVES, COACHES, ALIASES = Path("inputs/squad_moves.csv"), Path("inputs/coaches.csv"), Path("inputs/name_aliases.csv")
TEAMS = ["Chennai Super Kings", "Delhi Capitals", "Gujarat Titans", "Kolkata Knight Riders",
         "Lucknow Super Giants", "Mumbai Indians", "Punjab Kings", "Rajasthan Royals",
         "Royal Challengers Bengaluru", "Sunrisers Hyderabad"]
WIKI_API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "ipl-2027-predictions/1.0 (student analytics project; github.com/Aryan-1947)"}
COLS = ["team", "pkey", "player", "xi", "score", "nat", "source", "xi_force", "wiki"]
NOT_PLAYER = {n.lower() for n in TEAMS} | {"india", "australia", "england", "south africa", "new zealand", "west indies",
              "afghanistan", "sri lanka", "bangladesh", "pakistan", "zimbabwe", "ireland", "netherlands", "nepal",
              "delhi daredevils", "kings xi punjab", "royal challengers bangalore", "deccan chargers", "gujarat lions"}


# ------------------------------------------------------------------ names ----------------
def _plain(s):
    s = unicodedata.normalize("NFKD", str(s))
    return "".join(c for c in s if not unicodedata.combining(c))


def clean_name(s):
    s = re.sub(r"\[.*?\]|\(.*?\)|[†‡*§]", "", _plain(s))
    return re.sub(r"\s+", " ", s.replace(".", " ")).strip()


def name_key(name):
    """(surname, first initial): 'Rishabh Pant' and Cricsheet's 'RR Pant' both -> ('pant', 'r')."""
    toks = clean_name(name).split(" ")
    toks = [t for t in toks if t]
    return (toks[-1].lower(), toks[0][0].lower()) if toks else ("", "")


class Names:
    """Index of every IPL player Cricsheet knows (name -> id), with a recency score for XI selection."""

    def __init__(self):
        imp = pd.read_csv(RES / "player_impact.csv")[["pkey", "player"]]
        rat = pd.read_csv(DATA / "player_ratings_latest.csv")[["pkey", "matches_weighted"]]
        df = imp.merge(rat, on="pkey", how="left").fillna({"matches_weighted": 0.0})
        self.by_key, self.exact, self.score, self.display = defaultdict(list), {}, {}, {}
        for r in df.itertuples():
            self.by_key[name_key(r.player)].append(r.pkey)
            self.exact[clean_name(r.player).lower()] = r.pkey
            self.score[r.pkey] = float(r.matches_weighted)
            self.display[r.pkey] = r.player
        self.alias = {}
        if ALIASES.exists():
            for a in pd.read_csv(ALIASES).itertuples():
                self.alias[clean_name(a.alias).lower()] = clean_name(a.player).lower()

    def match(self, name, prefer=()):
        n = clean_name(name).lower()
        n = self.alias.get(n, n)
        if n in self.exact:
            return self.exact[n]
        cands = self.by_key.get(name_key(n), [])
        if len(cands) == 1:
            return cands[0]
        if len(cands) > 1:
            pref = [c for c in cands if c in set(prefer)]
            pool = pref or cands
            return max(pool, key=lambda c: self.score.get(c, 0.0))
        return None


def new_row(team, name, names, nat="", source="", prefer=()):
    pk = names.match(name, prefer)
    if pk is None:
        pk = "NEW:" + re.sub(r"[^a-z0-9]+", "-", clean_name(name).lower()).strip("-")
    return {"team": team, "pkey": pk, "player": names.display.get(pk, clean_name(name)), "xi": 0,
            "score": names.score.get(pk, 0.0), "nat": nat, "source": source, "xi_force": np.nan, "wiki": clean_name(name)}


# ------------------------------------------------------------------ baseline / moves -----
def load_baseline(names):
    if not BASELINE.exists():
        DATA.mkdir(exist_ok=True)
        if not SQUADS.exists():
            m4 = importlib.import_module("04_season_sim")
            m4.make_squads(TEAMS)
        shutil.copy(SQUADS, BASELINE)
    b = pd.read_csv(BASELINE)
    if "score" not in b:
        b["score"] = b["matches_weighted"] if "matches_weighted" in b else b["pkey"].map(names.score).fillna(0.0)
    for c, default in (("nat", ""), ("source", "baseline 2026"), ("xi_force", np.nan), ("wiki", "")):
        if c not in b:
            b[c] = default
    return b[COLS].copy()


def _find(sq, name, team=None):
    k = name_key(name)
    m = sq[sq["player"].map(name_key) == k]
    if team is not None:
        m = m[m["team"] == team]
    return m.index[0] if len(m) else None


def apply_moves(sq, moves, names, include_proposed=False):
    sq, log = sq.copy().reset_index(drop=True), []
    for m in moves.itertuples():
        status = str(getattr(m, "status", "confirmed")).lower()
        if status != "confirmed" and not include_proposed:
            continue
        action, player = str(m.action).lower(), m.player
        frm = m.from_team if isinstance(m.from_team, str) else None
        to = m.to_team if isinstance(m.to_team, str) else None
        if action == "move":
            if _find(sq, player, to) is not None:
                j = _find(sq, player, frm) if frm else None
                if j is not None and frm != to:
                    sq = sq.drop(index=j).reset_index(drop=True)
                    log.append(f"= {player} already at {to}; removed stale copy from {frm}")
                else:
                    log.append(f"= {player} already at {to}")
                continue
            i = _find(sq, player, frm) if frm else _find(sq, player)
            if i is not None:
                sq.loc[i, "team"], sq.loc[i, "source"] = to, f"ledger {m.trade_id}"
                log.append(f"> {sq.loc[i, 'player']}: {frm} -> {to}")
            else:
                prefer = sq.loc[sq["team"] == frm, "pkey"] if frm else ()
                sq = pd.concat([sq, pd.DataFrame([new_row(to, player, names, source=f"ledger {m.trade_id}", prefer=prefer)])],
                               ignore_index=True)
                log.append(f"+ {player} added to {to} (not found at {frm})")
        elif action == "release":
            i = _find(sq, player, frm)
            if i is not None:
                sq = sq.drop(index=i).reset_index(drop=True)
                log.append(f"- {player} released by {frm}")
        elif action == "add":
            if _find(sq, player, to) is None:
                sq = pd.concat([sq, pd.DataFrame([new_row(to, player, names, source=f"ledger {m.trade_id}")])],
                               ignore_index=True)
                log.append(f"+ {player} added to {to}")
        xi = getattr(m, "xi", np.nan)
        if pd.notna(xi) and to:
            i = _find(sq, player, to)
            if i is not None:
                sq.loc[i, "xi_force"] = float(xi)
    return sq, log


def dedupe_players(sq, log, base_team):
    """One real player, one squad. If two squads claim the same id, keep him where he was in 2026: the other claim is
    usually a DIFFERENT player with the same surname + first initial, so that row becomes an unknown, unrated new player."""
    sq = sq.copy()
    sq["_fresh"] = (~sq["source"].astype(str).str.startswith("baseline")).astype(int)
    real = ~sq["pkey"].astype(str).str.startswith("NEW:")
    drop = []
    for pk, g in sq[real].groupby("pkey"):
        if len(g) < 2:
            continue
        home = g.index[g["team"] == base_team.get(pk)]
        keep = home[0] if len(home) else g.sort_values("_fresh", ascending=False, kind="stable").index[0]
        for i in g.index:
            if i == keep:
                continue
            wiki = str(sq.loc[i, "wiki"]) if "wiki" in sq and pd.notna(sq.loc[i, "wiki"]) else ""
            if wiki and sq.loc[i, "_fresh"] == 1:
                slug = re.sub(r"[^a-z0-9]+", "-", wiki.lower()).strip("-")
                sq.loc[i, ["pkey", "player", "score"]] = ["NEW:" + slug, wiki, 0.0]
                log.append(f"! {wiki} ({g.loc[i, 'team']}) shares an id with {g.loc[keep, 'player']} ({g.loc[keep, 'team']}): "
                           f"treated as a different, unrated player")
            else:
                drop.append(i)
                log.append(f"! duplicate {g.loc[i, 'player']}: removed from {g.loc[i, 'team']} (kept at {g.loc[keep, 'team']})")
    return sq.drop(index=drop).drop(columns="_fresh").reset_index(drop=True)


def choose_xi(sq, overseas_cap=4):
    """Expected XI: the 11 most recently active players (<= 4 overseas where nationality is known)."""
    sq = sq.copy()
    sq["xi"] = 0
    is_os = lambda n: isinstance(n, str) and n.strip().lower() not in ("", "india", "ind", "nan")
    for team, g in sq.groupby("team"):
        picked = list(g.index[g["xi_force"] == 1])
        blocked = set(g.index[g["xi_force"] == 0])
        n_os = sum(is_os(sq.loc[i, "nat"]) for i in picked)
        for i in g.sort_values("score", ascending=False).index:
            if len(picked) >= 11:
                break
            if i in picked or i in blocked:
                continue
            if is_os(sq.loc[i, "nat"]) and n_os >= overseas_cap:
                continue
            picked.append(i)
            n_os += is_os(sq.loc[i, "nat"])
        sq.loc[picked, "xi"] = 1
    return sq


# ------------------------------------------------------------------ live (Wikipedia) -----
def _get_html(page, follow=True):
    """Wikipedia page HTML. follow=False: a redirect (e.g. a season page that does not exist yet) counts as missing."""
    import requests
    params = {"action": "parse", "page": page, "prop": "text", "format": "json", "formatversion": 2}
    if follow:
        params["redirects"] = 1
    r = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=15)
    r.raise_for_status()
    data = r.json()
    if "parse" not in data:
        raise RuntimeError(data.get("error", {}).get("info", "page not found"))
    html = data["parse"]["text"]
    if "redirectMsg" in html and not follow:
        raise RuntimeError("redirect (page does not exist yet)")
    return html


def _table_grid(rows):
    """Lay the cells of each <tr> out on the table's real column grid (rowspan/colspan expanded).
    Team pages mix plain rows with rows that inherit cells from a rowspan above, so a cell's position in
    find_all() is not its column: without this, a nationality read as column 2 picks up a birth date."""
    grid, carried = [], {}
    for r in rows:
        cells, row, col = iter(r.find_all(["th", "td"])), {}, 0
        while True:
            if col in carried:                      # cell inherited from a rowspan above
                cell, left = carried[col]
                row[col] = cell
                if left <= 1:
                    del carried[col]
                else:
                    carried[col] = (cell, left - 1)
                col += 1
                continue
            cell = next(cells, None)
            if cell is None:
                break
            row[col] = cell
            rs = max(1, int(cell.get("rowspan") or 1))
            cs = max(1, int(cell.get("colspan") or 1))
            for k in range(1, cs):
                row[col + k] = cell
            if rs > 1:
                for k in range(cs):
                    carried[col + k] = (cell, rs - 1)
            col += cs
        grid.append(row)
    return grid


def nationality(cell):
    """Country from a nationality cell: the flag image's alt/title if it has one, else the cell text.
    Flag icons are often decorative (empty alt), and a mis-aligned cell holds a birth date - both -> text check."""
    img = cell.find("img")
    if img is not None:
        alt = (img.get("alt") or img.get("title") or "").strip()
        if alt:
            return alt
    txt = cell.get_text(" ", strip=True)
    return "" if not txt or re.search(r"\d", txt) or len(txt) > 40 else txt


def parse_squad(html):
    """Largest wikitable with a Name column plus Nat./Batting/Bowling columns -> [(name, nationality)].
    The header row may sit in any of the first 4 rows (some pages put a title row on top)."""
    from bs4 import BeautifulSoup
    soup, best = BeautifulSoup(html, "lxml"), []
    for tbl in soup.select("table.wikitable"):
        rows = tbl.find_all("tr")
        if len(rows) < 10:
            continue
        grid = _table_grid(rows)
        hi = ni = nat_i = None
        for k, row in enumerate(grid[:4]):
            head = {c: cell.get_text(" ", strip=True).lower() for c, cell in row.items()}
            n = next((c for c, h in sorted(head.items()) if h.startswith("name") or h == "player"), None)
            nat_c = next((c for c, h in sorted(head.items()) if h.startswith("nat")), None)
            if n is not None and (nat_c is not None or any("bat" in h or "bowl" in h for h in head.values())):
                hi, ni = k, n
                nat_i = nat_c
                break
        if hi is None:
            continue
        players = []
        for row in grid[hi + 1:]:
            cell = row.get(ni)
            if cell is None or int(cell.get("colspan") or 1) > 1:   # section row ("Batters", "Bowlers", ...)
                continue
            nm = clean_name(cell.get_text(" ", strip=True))
            if not nm or re.search(r"\d", nm) or nm.lower() in NOT_PLAYER:
                continue
            players.append((nm, nationality(row[nat_i]) if nat_i is not None and nat_i in row else ""))
        if len(players) > len(best):
            best = players
    return best


def parse_coaches(html):
    from bs4 import BeautifulSoup
    soup, out = BeautifulSoup(html, "lxml"), []
    for tbl in soup.select("table"):
        rows = tbl.find_all("tr")
        head = [c.get_text(" ", strip=True).lower() for c in rows[0].find_all(["th", "td"])] if rows else []
        if not (any(h in ("position", "role") for h in head) and any(h.startswith("name") for h in head)):
            continue
        pi = next(i for i, h in enumerate(head) if h in ("position", "role"))
        ni = next(i for i, h in enumerate(head) if h.startswith("name"))
        for r in rows[1:]:
            cells = r.find_all(["th", "td"])
            if len(cells) > max(pi, ni):
                role, nm = clean_name(cells[pi].get_text(" ", strip=True)), clean_name(cells[ni].get_text(" ", strip=True))
                if nm and re.search(r"coach|mentor|director|analyst", role, re.I):
                    out.append((role, nm))
    return out


def describe_tables(html, limit=4):
    """For diagnostics: header row of the biggest tables on a page that did not parse."""
    from bs4 import BeautifulSoup
    out = []
    for tbl in BeautifulSoup(html, "lxml").select("table"):
        rows = tbl.find_all("tr")
        if len(rows) >= 8:
            head = [c.get_text(" ", strip=True)[:18] for c in rows[0].find_all(["th", "td"])][:7]
            out.append(f"{len(rows)} rows: {head}")
    return out[:limit]


LAST_SEASON = 2026          # the season the baseline squads come from; its pages still carry nationalities
CACHE_DIR = DATA / "wiki_cache"
CACHE_FRESH_MIN, CACHE_MAX_DAYS = 30, 14     # reuse data < 30 min old; fall back to data up to 14 days old if Wikipedia fails


def _cache_path(team):
    return CACHE_DIR / (re.sub(r"[^A-Za-z0-9]+", "_", team) + ".json")


def _cache_read(team):
    p = _cache_path(team)
    if not p.exists():
        return None, None
    d = json.loads(p.read_text(encoding="utf-8"))
    return {"squad": [tuple(x) for x in d["squad"]], "coaches": [tuple(x) for x in d["coaches"]]}, (time.time() - d["ts"]) / 60


def _cache_write(team, got):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _cache_path(team).write_text(json.dumps({"ts": time.time(), "squad": got["squad"], "coaches": got["coaches"]}), encoding="utf-8")


def _fetch_with_retry(page, follow):
    for attempt in range(2):
        try:
            return _get_html(page, follow)
        except Exception as e:  # noqa: BLE001
            transient = any(k in type(e).__name__ for k in ("Connection", "Timeout")) or "429" in str(e) or "503" in str(e)
            if attempt == 1 or not transient:
                raise
            time.sleep(3 * (attempt + 1))


def _nationality_reference(team, log):
    """Last season's squad list, used ONLY to fill in missing nationalities - never to replace a squad
    (that page is a year old and still names players who have since been traded or released)."""
    try:
        html = _fetch_with_retry(f"{LAST_SEASON} {team} season".replace(" ", "_"), True)
        ref = parse_squad(html)
        if len(ref) >= 14:
            log.append(f"live {team}: no squad table to use - taking nationalities from the {LAST_SEASON} season page")
            return ref
    except Exception as e:  # noqa: BLE001
        log.append(f"live {team}: no usable squad table and no {LAST_SEASON} season page ({str(e)[:60]})")
    return []


def fetch_live(log):
    """Per team: recent cache -> '2027 <team> season' page (strict) -> main team page -> older cache if the web fails."""
    live, no_2027, blocked = {}, 0, False
    for team in TEAMS:
        cached, age = _cache_read(team)
        if cached and age < CACHE_FRESH_MIN:
            live[team] = cached
            log.append(f"live {team}: using data fetched {age:.0f} min ago ({len(cached['squad'])} players)")
            continue
        got, staff_only = None, []
        if not blocked:
            for page, follow in ((f"2027 {team} season", False), (team, True)):
                try:
                    html = _fetch_with_retry(page.replace(" ", "_"), follow)
                except Exception as e:  # noqa: BLE001
                    if "429" in str(e):
                        blocked = True
                        log.append("Wikipedia is rate-limiting this computer - pausing live fetches (cached data is used instead)")
                        break
                    if page.startswith("2027"):
                        no_2027 += 1
                    else:
                        log.append(f"live {team}: failed ({type(e).__name__}: {str(e)[:70]})")
                    continue
                sq, co = parse_squad(html), parse_coaches(html)
                if len(sq) >= 14:
                    got = {"squad": sq, "coaches": co}
                    _cache_write(team, got)
                    log.append(f"live {team}: {len(sq)} players, {len(co)} staff from '{page}'")
                    break
                staff_only = staff_only or co
                log.append(f"live {team}: '{page}' gave {len(sq)} players (tables: {describe_tables(html, 3)})")
            time.sleep(1.0)
        if got is None and cached and age < CACHE_MAX_DAYS * 1440:
            got = cached
            log.append(f"live {team}: Wikipedia unavailable - using cached data from {age / 60:.1f} h ago")
        if got is None and not blocked:
            ref = _nationality_reference(team, log)
            if ref:
                got = {"squad": [], "coaches": staff_only, "nat_ref": ref}
        live[team] = got or ({"squad": [], "coaches": staff_only} if staff_only else None)
    if no_2027:
        log.append(f"({no_2027} teams have no '2027 ... season' page yet - normal before the auction; main pages used)")
    return live


# ------------------------------------------------------------------ main logic -----------
def build_current_squads(use_live=True, include_proposed=False, names=None, log=None):
    """Returns (squads, coaches, log, unmatched). Pure function of files + (optionally) the internet."""
    log = [] if log is None else log
    names = names or Names()
    sq = load_baseline(names)
    base_team = dict(zip(sq["pkey"], sq["team"]))   # where each player was in 2026
    live = fetch_live(log) if use_live else {}
    unmatched = []
    for team, data in live.items():
        if not data or len(data["squad"]) < 14:
            continue
        prefer = sq.loc[sq["team"] == team, "pkey"].tolist()
        rows = [new_row(team, n, names, nat, "wikipedia", prefer) for n, nat in data["squad"]]
        known = sum(not r["pkey"].startswith("NEW:") for r in rows)
        base_keys = set(sq.loc[sq["team"] == team, "pkey"])
        overlap = sum(r["pkey"] in base_keys for r in rows)
        if known / len(rows) < 0.6 or overlap / len(rows) < 0.35:
            log.append(f"live {team}: only {known}/{len(rows)} recognised and {overlap} match the known squad "
                       f"- looks wrong, ignored (keeping baseline + ledger)")
            continue
        added = 0
        updated = 0
        for n, nat in data["squad"]:
            pk = names.match(n, prefer=tuple(sq.loc[sq["team"] == team, "pkey"]))
            if pk is None:
                pk = "NEW:" + re.sub(r"[^a-z0-9]+", "-", clean_name(n).lower()).strip("-")
            mask = (sq["team"] == team) & (sq["pkey"] == pk)
            if mask.any():
                idx = sq.index[mask][0]
                if nat:
                    sq.loc[idx, "nat"] = nat
                sq.loc[idx, "wiki"] = clean_name(n)
                if str(sq.loc[idx, "source"]) != "wikipedia":
                    sq.loc[idx, "source"] = "wikipedia"
                    updated += 1
            else:
                new = new_row(team, n, names, nat, "wikipedia", tuple(sq.loc[sq["team"] == team, "pkey"]))
                sq = pd.concat([sq, pd.DataFrame([new])], ignore_index=True)
                added += 1
                if new["pkey"].startswith("NEW:"):
                    unmatched.append((team, new["player"]))
        log.append(f"live {team}: merged {added} new, {updated} updated from Wikipedia ({len(data['squad'])} live players)")
    moves = pd.read_csv(MOVES) if MOVES.exists() else pd.DataFrame(
        columns=["trade_id", "date", "player", "from_team", "to_team", "action", "status"])
    sq, mlog = apply_moves(sq, moves, names, include_proposed)
    log += mlog
    # Baseline rows carry no nationality, and a team whose page has no squad table (CSK today) would otherwise keep
    # a blank column: fill from every table we parsed, so the overseas cap in choose_xi actually applies.
    lookup = {}
    for data in live.values():
        for n, nat in (data or {}).get("squad", []) + (data or {}).get("nat_ref", []):
            if isinstance(nat, str) and nat.strip():
                lookup.setdefault(name_key(n), nat.strip())
    blank = sq["nat"].isna() | sq["nat"].astype(str).str.strip().isin(["", "nan"])
    filled = 0
    for i in sq.index[blank]:
        nat = lookup.get(name_key(sq.loc[i, "player"])) or lookup.get(name_key(sq.loc[i, "wiki"]))
        if nat:
            sq.loc[i, "nat"], filled = nat, filled + 1
    if filled:
        log.append(f"nationality filled for {filled} player(s) from the parsed Wikipedia tables")
    sq = dedupe_players(sq, log, base_team)
    sq = choose_xi(sq)[COLS].sort_values(["team", "xi", "score"], ascending=[True, False, False]).reset_index(drop=True)

    co = pd.read_csv(COACHES) if COACHES.exists() else pd.DataFrame(columns=["team", "role", "name", "source", "date"])
    have = set(zip(co["team"], co["role"].str.lower()))
    extra = []
    for team, data in live.items():
        for role, nm in (data or {}).get("coaches", []):
            if (team, role.lower()) not in have:
                extra.append({"team": team, "role": role, "name": nm, "source": "wikipedia", "date": ""})
                have.add((team, role.lower()))
    co = pd.concat([co, pd.DataFrame(extra)], ignore_index=True) if extra else co
    return sq, co, log, unmatched


def update_squads(use_live=True, include_proposed=False, write=True):
    names = Names()
    old = pd.read_csv(SQUADS) if SQUADS.exists() else None
    sq, co, log, unmatched = build_current_squads(use_live, include_proposed, names)
    if write:
        DATA.mkdir(exist_ok=True)
        RES.mkdir(exist_ok=True)
        if old is not None:
            (DATA / "snapshots").mkdir(exist_ok=True)
            old.to_csv(DATA / "snapshots" / f"squads_{datetime.datetime.now():%Y%m%d_%H%M%S}.csv", index=False)
            before = dict(zip(old["pkey"], old["team"]))
            changes = [{"time": datetime.datetime.now().isoformat(timespec="seconds"), "player": r.player,
                        "from": before.get(r.pkey, "(not in squads)"), "to": r.team}
                       for r in sq.itertuples() if before.get(r.pkey) != r.team]
            changes += [{"time": datetime.datetime.now().isoformat(timespec="seconds"), "player": r.player,
                         "from": r.team, "to": "(left squad)"} for r in old.itertuples()
                        if r.pkey not in set(sq["pkey"])]
            if changes:
                p = RES / "squad_changes_log.csv"
                pd.concat([pd.read_csv(p) if p.exists() else pd.DataFrame(), pd.DataFrame(changes)]).to_csv(p, index=False)
        sq.to_csv(SQUADS, index=False)
        co.to_csv(DATA / "coaches.csv", index=False)
        pd.DataFrame(unmatched, columns=["team", "player"]).to_csv(DATA / "squad_unmatched.csv", index=False)
    return {"squads": sq, "coaches": co, "log": log, "unmatched": unmatched}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-live", action="store_true", help="skip the internet; use baseline + ledger only")
    ap.add_argument("--include-proposed", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    try:
        out = update_squads(not a.no_live, a.include_proposed, write=not a.dry_run)
    except FileNotFoundError as e:
        raise SystemExit(f"Missing input file: {e.filename}\nRun steps 2, 4 and 6 first.")
    print("\n".join(out["log"]))
    sq = out["squads"]
    print(f"\n{len(sq)} players in {sq['team'].nunique()} squads; "
          f"{int(sq['pkey'].str.startswith('NEW:').sum())} without IPL history (rated as average).")

    print("\nPer-team summary (players + coaching staff):")
    print(f"{'Team':<30} {'Players':>8} {'Staff':>6} {'Source':<14}")
    print("-" * 62)
    staff_counts = out["coaches"].groupby("team").size() if not out["coaches"].empty else pd.Series(dtype=int)
    for team in TEAMS:
        n_players = int((sq["team"] == team).sum())
        n_staff = int(staff_counts.get(team, 0))
        src = "live" if ((sq["team"] == team) & (sq["source"].astype(str) == "wikipedia")).any() else "baseline+ledger"
        print(f"{team:<30} {n_players:>8} {n_staff:>6} {src:<14}")

    if out["unmatched"]:
        print(f"{len(out['unmatched'])} names not recognised - see data/squad_unmatched.csv")
    print("Saved live/squads_2027.csv" if not a.dry_run else "(dry run - nothing written)")


if __name__ == "__main__":
    main()

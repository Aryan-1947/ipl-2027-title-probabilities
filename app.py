"""
IPL 2027 Title Predictor - Streamlit app.      Run:  streamlit run app.py

The backend does the work on demand:
  * "Update squads & trades"  -> 14_update_squads.py   (internet + trade ledger)
  * "Check for schedule"      -> 15_fetch_schedule.py  (official IPL feed; if the schedule is not out, nothing changes)
  * the forecast (engine.py) uses the current squads and, only when a complete schedule exists, the real fixtures.
Unofficial fan analytics project - not affiliated with the BCCI / IPL / any franchise. Not betting advice.
"""
import datetime
import importlib
import json
from pathlib import Path

import pandas as pd
import streamlit as st

import engine as E
import ui_art as art

m14 = importlib.import_module("14_update_squads")
m15 = importlib.import_module("15_fetch_schedule")

st.set_page_config(page_title="IPL 2027 Title Predictor", page_icon="🏏", layout="wide", initial_sidebar_state="expanded")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Poppins:wght@400;500;600;700;800&display=swap');
:root{--orange:#FF8A00;--gold:#F9C74F;--card:rgba(255,255,255,.06);--line:rgba(255,255,255,.12);--text:#EAF0FF;--mut:#9FB0D6}
html,body,.stApp,[class*="css"]{font-family:'Poppins',sans-serif;color:var(--text)}
.stApp{background:radial-gradient(1100px 480px at 50% -8%,rgba(255,138,0,.20),transparent 60%),linear-gradient(180deg,#050B1F 0%,#0B1F4B 55%,#0a1236 100%)}
header[data-testid="stHeader"]{background:transparent}#MainMenu,footer{visibility:hidden}
.block-container{padding-top:3.4rem;max-width:1180px}[data-testid="stToolbar"],[data-testid="stAppDeployButton"],.stDeployButton{display:none!important}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0B1F4B,#060D26);border-right:1px solid var(--line)}
h1,h2,h3{font-family:'Bebas Neue',sans-serif!important;letter-spacing:.05em;font-weight:400!important}
.stButton>button{background:linear-gradient(90deg,#FF8A00,#FF5E00);color:#fff;font-weight:700;border:0;border-radius:999px;padding:.55rem 1.1rem;box-shadow:0 6px 18px rgba(255,106,0,.35);transition:.15s}
.stButton>button:hover{filter:brightness(1.12);transform:translateY(-1px);color:#fff}
.stTabs [data-baseweb="tab-list"]{gap:.35rem;border-bottom:1px solid var(--line)}
.stTabs [data-baseweb="tab"]{background:var(--card);border-radius:12px 12px 0 0;padding:.6rem 1.1rem;font-weight:600;color:var(--mut)}
.stTabs [aria-selected="true"]{background:linear-gradient(180deg,rgba(255,138,0,.30),rgba(255,138,0,.05));color:#fff!important;border-bottom:3px solid var(--orange)}
.hero{position:relative;border-radius:22px;overflow:hidden;border:1px solid var(--line);box-shadow:0 20px 60px rgba(0,0,0,.5);margin-bottom:.8rem}
.hero svg{display:block;width:100%;height:300px}
.hero .txt{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;pointer-events:none;padding:0 6%}
.hero .eyebrow{font-weight:700;letter-spacing:.4em;font-size:.8rem;color:var(--gold);text-transform:uppercase}
.hero h1{font-size:clamp(2.6rem,7.2vw,5.4rem);margin:.1rem 0 0;padding:0;line-height:.95;background:linear-gradient(180deg,#fff 10%,#F9C74F 60%,#FF8A00);-webkit-background-clip:text;background-clip:text;color:transparent;filter:drop-shadow(0 0 22px rgba(255,138,0,.45))}
.hero .sub{font-size:.95rem;color:#d6e0ff;margin-top:.4rem;text-shadow:0 2px 8px #000}
.tw{animation:tw 3s ease-in-out infinite}.cone{animation:flick 5s ease-in-out infinite}.trophy{animation:float 4s ease-in-out infinite}
@keyframes tw{0%,100%{opacity:.25}50%{opacity:1}}@keyframes flick{0%,100%{opacity:1}50%{opacity:.65}}@keyframes float{0%,100%{transform:translateY(0)}50%{transform:translateY(-7px)}}
@keyframes rise{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:none}}@keyframes marq{from{transform:translateX(0)}to{transform:translateX(-50%)}}
.chips{display:flex;flex-wrap:wrap;gap:.5rem;margin:.2rem 0 .8rem}
.chip{display:inline-flex;gap:.45rem;align-items:center;padding:.38rem .85rem;border-radius:999px;background:rgba(255,255,255,.08);border:1px solid var(--line);font-size:.8rem}
.dot{width:9px;height:9px;border-radius:50%;display:inline-block}.ok{background:#34D399;box-shadow:0 0 8px #34D399}.warn{background:#FBBF24;box-shadow:0 0 8px #FBBF24}.info{background:#60A5FA;box-shadow:0 0 8px #60A5FA}
.ticker{display:flex;align-items:center;background:linear-gradient(90deg,#FF5E00,#FF8A00);border-radius:10px;overflow:hidden;margin-bottom:1rem;box-shadow:0 8px 24px rgba(255,106,0,.25)}
.tk-label{background:#0a1236;color:var(--gold);font-weight:800;letter-spacing:.12em;font-size:.72rem;padding:.55rem .9rem;white-space:nowrap}
.tk-win{overflow:hidden;flex:1}.tk-track{display:inline-block;white-space:nowrap;animation:marq 40s linear infinite;font-weight:600;font-size:.84rem;color:#fff;padding:.5rem 0}
.tk-track span{padding:0 2.4rem}
.card{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:1rem 1.2rem;backdrop-filter:blur(6px);animation:rise .5s both}
.podium{display:flex;gap:1rem;align-items:flex-end;justify-content:center;margin:.4rem 0 1.2rem;flex-wrap:wrap}
.pc{flex:1 1 220px;max-width:340px;text-align:center;border-radius:20px;padding:1.1rem .8rem 1.2rem;border:1px solid var(--line);animation:rise .6s both;position:relative;overflow:hidden}
.pc.p1{padding-top:1.6rem;padding-bottom:1.7rem;border-color:rgba(249,199,79,.7);box-shadow:0 0 40px rgba(249,199,79,.22)}
.pc .medal{font-size:1.5rem}.pc .nm{font-weight:700;font-size:1rem;margin-top:.2rem}
.pc .big{font-family:'Bebas Neue',sans-serif;font-size:3.1rem;line-height:1;margin-top:.2rem;background:linear-gradient(180deg,#fff,#F9C74F);-webkit-background-clip:text;background-clip:text;color:transparent}
.pc.p1 .big{font-size:4rem}.pc .rng{font-size:.78rem;color:var(--mut)}
.lb{display:flex;flex-direction:column;gap:.55rem}
.lb-row{display:flex;align-items:center;gap:.9rem;background:var(--card);border:1px solid var(--line);border-radius:16px;padding:.55rem 1rem;animation:rise .5s both}
.lb-rank{font-family:'Bebas Neue',sans-serif;font-size:1.7rem;width:1.7rem;text-align:center;color:var(--gold)}
.lb-main{flex:1;min-width:0}.lb-name{font-weight:700;font-size:.95rem;display:flex;justify-content:space-between;gap:.6rem;flex-wrap:wrap}
.lb-name small{font-weight:500;color:var(--mut);font-size:.74rem}
.lb-bar{position:relative;height:14px;border-radius:99px;background:rgba(255,255,255,.08);margin-top:.35rem}
.lb-fill{position:absolute;left:0;top:0;bottom:0;border-radius:99px;box-shadow:0 0 14px rgba(255,255,255,.18)}
.lb-whisk{position:absolute;top:50%;height:2px;background:rgba(255,255,255,.9);transform:translateY(-50%)}
.lb-whisk:before,.lb-whisk:after{content:"";position:absolute;top:-4px;width:2px;height:10px;background:rgba(255,255,255,.9)}.lb-whisk:before{left:0}.lb-whisk:after{right:0}
.lb-stat{text-align:center;min-width:64px;font-weight:700;font-size:1rem}.lb-stat span{display:block;font-size:.62rem;color:var(--mut);font-weight:500;letter-spacing:.08em;text-transform:uppercase}
.lb-stat.big{font-family:'Bebas Neue',sans-serif;font-size:1.9rem;color:var(--gold);line-height:1}
.trade{display:flex;gap:1rem;align-items:center;flex-wrap:wrap}.trade .tp{display:flex;align-items:center;gap:.7rem;flex:1 1 260px;background:rgba(255,255,255,.05);border-radius:14px;padding:.5rem .8rem}
.arrow{font-size:1.2rem;color:var(--orange)}.pill{font-size:.68rem;font-weight:700;letter-spacing:.08em;padding:.15rem .55rem;border-radius:99px;background:rgba(52,211,153,.18);color:#6EE7B7;border:1px solid rgba(52,211,153,.4)}
.pill.prop{background:rgba(251,191,36,.15);color:#FCD34D;border-color:rgba(251,191,36,.4)}
.dv{display:flex;align-items:center;gap:.7rem;margin:.3rem 0}.dv .lab{width:210px;font-size:.85rem;display:flex;align-items:center;gap:.5rem}
.dv .axis{position:relative;flex:1;height:20px}.dv .axis:before{content:"";position:absolute;left:50%;top:-2px;bottom:-2px;width:1px;background:rgba(255,255,255,.35)}
.dv .b{position:absolute;top:3px;height:14px;border-radius:4px}.dv .val{width:72px;text-align:right;font-weight:700;font-size:.85rem}
.tcard{border-radius:18px;overflow:hidden;border:1px solid var(--line);background:var(--card);animation:rise .5s both;margin-bottom:1rem}
.tcard .th{display:flex;gap:.9rem;align-items:center;padding:.8rem 1rem}.tcard .th b{font-size:1.05rem}.tcard .th small{color:rgba(255,255,255,.8)}
.tcard .tb{padding:.7rem 1rem 1rem}.xi{display:flex;flex-wrap:wrap;gap:.35rem}.xi span{background:rgba(255,255,255,.09);border-radius:8px;padding:.18rem .55rem;font-size:.76rem}
.xi.bench span{opacity:.6;background:rgba(255,255,255,.04)}.sec{font-size:.68rem;letter-spacing:.14em;color:var(--mut);text-transform:uppercase;margin:.6rem 0 .3rem;font-weight:700}
.empty{text-align:center;padding:1.6rem 1rem}.note{font-size:.8rem;color:var(--mut)}
.fx{display:flex;align-items:center;gap:.7rem;background:var(--card);border:1px solid var(--line);border-radius:14px;padding:.5rem .9rem;margin-bottom:.45rem}
.fx .d{width:84px;font-size:.78rem;color:var(--mut)}.fx .t{flex:1;display:flex;align-items:center;gap:.5rem;font-weight:600;font-size:.88rem}
.pbar{display:flex;height:10px;border-radius:99px;overflow:hidden;width:150px}.footer{color:var(--mut);font-size:.72rem;text-align:center;margin:2rem 0 .5rem}
table.mt{width:100%;border-collapse:collapse;font-size:.85rem}table.mt th{color:var(--mut);text-align:left;font-weight:600;border-bottom:1px solid var(--line);padding:.4rem}table.mt td{padding:.4rem;border-bottom:1px solid rgba(255,255,255,.06)}
table.mt tr.hl td{color:#FCD34D;font-weight:700}
</style>
"""


def H(html):
    """Render HTML without Markdown turning indented lines into code blocks."""
    st.markdown("\n".join(l.strip() for l in html.splitlines() if l.strip()), unsafe_allow_html=True)


def mtime(p):
    p = Path(p)
    return p.stat().st_mtime if p.exists() else 0.0


def pct(x, d=1):
    return f"{x * 100:.{d}f}%"


# ------------------------------------------------------------------ backend --------------
@st.cache_resource(show_spinner="Fitting the model (one-off, a few seconds)...")
def get_model(n_boot):
    return E.Model(n_boot=n_boot)


@st.cache_data(show_spinner=False)
def forecast_current(_model, n_boot, sims, sq_sig, fx_sig, use_fx):
    return _model.forecast(E.load_squads(), E.load_fixtures() if use_fx else None, sims)


@st.cache_data(show_spinner=False)
def forecast_baseline(_model, n_boot, sims, fx_sig, use_fx):
    return _model.forecast(E.load_squads(E.BASELINE), E.load_fixtures() if use_fx else None, sims)


def schedule_status():
    if m15.STATUS.exists():
        return json.load(open(m15.STATUS))
    return {"status": "unknown", "message": "Not checked yet."}


def load_coaches():
    for p in (E.DATA / "coaches.csv", Path("coaches.csv")):
        if p.exists():
            df = pd.read_csv(p)
            df["role"] = df["role"].fillna("")
            return df
    return pd.DataFrame(columns=["team", "role", "name"])


# ------------------------------------------------------------------ UI blocks ------------
def hero(chips):
    H(f'<div class="hero">{art.hero_svg()}{art.hero_title_svg()}</div>')
    H('<div class="chips">' + "".join(f'<span class="chip"><span class="dot {c}"></span>{t}</span>' for c, t in chips) + "</div>")


def ticker(res):
    top = res.iloc[0]
    txt = (f"MODEL-IMPLIED LEAN, NOT A GUARANTEE — held-out tests show IPL matches are close to coin flips · "
           f"current favourite in the simulation: {top.team} ({pct(top.p_title)}) · equal chance would be 10% each · "
           f"ranges show model uncertainty · ")
    H(f'<div class="ticker"><div class="tk-label">READ THIS</div><div class="tk-win"><div class="tk-track"><span>{txt}</span><span>{txt}</span></div></div></div>')


def podium(res):
    S = res.p_title_hi.max()
    order = [1, 0, 2]
    cards = []
    for k in order:
        r = res.iloc[k]
        t = art.team(r.team)
        medal = ["🥇", "🥈", "🥉"][k]
        cards.append(f'<div class="pc p{k + 1}" style="background:linear-gradient(160deg,{t["c1"]}55,{t["c2"]}33 55%,rgba(5,11,31,.5));animation-delay:{k * 90}ms">'
                     f'<div class="medal">{medal}</div>{art.badge(r.team, 104 if k == 0 else 80)}'
                     f'<div class="nm">{r.team}</div><div class="big">{pct(r.p_title)}</div>'
                     f'<div class="rng">to win the title · range {pct(r.p_title_lo)}–{pct(r.p_title_hi)}</div></div>')
    H('<div class="podium">' + "".join(cards) + "</div>")


def leaderboard(res, coaches):
    S = max(res.p_title_hi.max() * 1.05, 0.05)
    hc = coaches[coaches["role"].str.lower() == "head coach"].set_index("team")["name"].to_dict() if len(coaches) else {}
    rows = []
    for i, r in res.iterrows():
        t = art.team(r.team)
        rows.append(
            f'<div class="lb-row" style="animation-delay:{i * 55}ms"><div class="lb-rank">{i + 1}</div>{art.badge(r.team, 40)}'
            f'<div class="lb-main"><div class="lb-name"><span>{r.team}</span><small>Coach: {hc.get(r.team, "—")}</small></div>'
            f'<div class="lb-bar"><div class="lb-fill" style="width:{r.p_title / S * 100:.1f}%;background:linear-gradient(90deg,{t["c1"]},{t["c2"]})"></div>'
            f'<div class="lb-whisk" style="left:{r.p_title_lo / S * 100:.1f}%;width:{max((r.p_title_hi - r.p_title_lo) / S * 100, 0.5):.1f}%"></div></div></div>'
            f'<div class="lb-stat big">{pct(r.p_title)}<span>title</span></div><div class="lb-stat">{pct(r.p_top4, 0)}<span>top 4</span></div>'
            f'<div class="lb-stat">{pct(r.p_final, 0)}<span>final</span></div><div class="lb-stat">{r.avg_wins:.1f}<span>avg wins</span></div></div>')
    H('<div class="lb">' + "".join(rows) + "</div>")


def diverging(df, col, fmt):
    """df: team, <col> (delta, in fraction). Bars to the right = better."""
    m = max(df[col].abs().max(), 1e-9)
    rows = []
    for r in df.itertuples():
        v = getattr(r, col)
        w = abs(v) / m * 48
        colr = "#34D399" if v >= 0 else "#F87171"
        pos = f"left:50%;width:{w:.1f}%" if v >= 0 else f"left:{50 - w:.1f}%;width:{w:.1f}%"
        rows.append(f'<div class="dv"><div class="lab">{art.badge(r.team, 24)}<span>{r.team}</span></div>'
                    f'<div class="axis"><div class="b" style="{pos};background:{colr}"></div></div><div class="val" style="color:{colr}">{fmt(v)}</div></div>')
    H("".join(rows))


def trade_wire():
    if not Path(m14.MOVES).exists():
        st.info("No trade ledger found (squad_moves.csv).")
        return
    mv = pd.read_csv(m14.MOVES)
    for tid, g in mv.groupby("trade_id", sort=False):
        status = str(g.status.iloc[0]).lower()
        pill = '<span class="pill">CONFIRMED</span>' if status == "confirmed" else '<span class="pill prop">PROPOSED</span>'
        parts = "".join(
            f'<div class="tp">{art.badge(r.from_team, 34)}<span class="arrow">➜</span>{art.badge(r.to_team, 34)}'
            f'<div><b>{r.player}</b><br><span class="note">{art.team(r.from_team)["code"]} → {art.team(r.to_team)["code"]}</span></div></div>'
            for r in g.itertuples())
        note = g.note.iloc[0] if isinstance(g.note.iloc[0], str) else ""
        H(f'<div class="card" style="margin-bottom:.7rem"><div style="display:flex;justify-content:space-between;margin-bottom:.5rem">'
          f'<span class="note">Trade {tid} · {g.date.iloc[0]} · {note}</span>{pill}</div><div class="trade">{parts}</div></div>')


def team_cards(sq, coaches, ratings):
    hc = coaches[coaches["role"].str.lower() == "head coach"].set_index("team")["name"].to_dict() if len(coaches) else {}
    rt = ratings.set_index("team")
    cols = st.columns(2)
    for i, team in enumerate(sorted(sq.team.unique())):
        t = art.team(team)
        g = sq[sq.team == team].sort_values(["xi", "score"], ascending=False)
        xi = "".join(f"<span>{n}</span>" for n in g[g.xi == 1].player)
        bench = "".join(f"<span>{n}</span>" for n in g[g.xi == 0].player)
        staff = coaches[(coaches.team == team) & (coaches.role.str.lower() != "head coach")] if len(coaches) else []
        extra = " · ".join(f"{r.role}: {r.name}" for r in staff.itertuples()) if len(staff) else ""
        rr = rt.loc[team] if team in rt.index else None
        with cols[i % 2]:
            H(f'<div class="tcard"><div class="th" style="background:linear-gradient(100deg,{t["c1"]},{t["c2"]})">{art.badge(team, 52)}'
              f'<div><b style="color:{t["ink"]}">{team}</b><br><small style="color:{t["ink"]}">Head coach: {hc.get(team, "—")}</small></div></div>'
              f'<div class="tb"><div class="sec">Expected XI</div><div class="xi">{xi}</div><div class="sec">Squad depth</div><div class="xi bench">{bench}</div>'
              + (f'<div class="sec">Support staff</div><div class="note">{extra}</div>' if extra else "")
              + (f'<div class="sec">Model ratings</div><div class="note">Elo {rr.elo_2027:.0f} · squad impact {rr.squad_impact:+.2f}</div>' if rr is not None else "")
              + "</div></div>")


# ------------------------------------------------------------------ page -----------------
H(CSS)
for need in (E.MOD / "v2_params.json", E.DATA / "team_ratings_latest.csv", E.SQUADS):
    if not Path(need).exists():
        st.error(f"Missing {need}. Run the pipeline first (steps 1-7, then `python 14_update_squads.py`).")
        st.stop()

with st.sidebar:
    H(f'<div style="text-align:center;margin:.3rem 0 .8rem">{art.small_icons()}<div style="font-family:Bebas Neue;font-size:1.6rem;letter-spacing:.08em;color:#F9C74F">CONTROL ROOM</div></div>')
    precision = st.select_slider("Simulation precision", ["Fast", "Standard", "Full"], value="Fast",
                                 help="More model re-fits and simulated seasons = steadier numbers, slower.")
    use_fx = st.toggle("Use official schedule when available", value=True)
    st.markdown("---")
    if st.button("⟳  Update squads & trades", width="stretch"):
        with st.spinner("Fetching squads, trades and coaches..."):
            try:
                out = m14.update_squads(use_live=True)
                st.session_state["log"] = out["log"] + [f"Done: {len(out['squads'])} players."]
            except Exception as e:  # noqa: BLE001
                st.session_state["log"] = [f"Squad update failed: {e}"]
    if st.button("⟳  Check for schedule", width="stretch"):
        with st.spinner("Asking the official IPL feed..."):
            try:
                r = m15.fetch_schedule(force=True)
                st.session_state["log"] = [f"Schedule: [{r['status']}] {r.get('message', '')}"]
            except Exception as e:  # noqa: BLE001
                st.session_state["log"] = [f"Schedule check failed: {e}"]
    if st.button("⟳  Refresh everything", width="stretch"):
        with st.spinner("Updating squads, checking schedule..."):
            try:
                out = m14.update_squads(use_live=True)
                r = m15.fetch_schedule(force=True)
                st.session_state["log"] = out["log"] + [f"Schedule: [{r['status']}] {r.get('message', '')}"]
            except Exception as e:  # noqa: BLE001
                st.session_state["log"] = [f"Refresh failed: {e}"]
    if st.session_state.get("log"):
        with st.expander("Backend log", expanded=False):
            st.code("\n".join(st.session_state["log"]), language=None)
    st.markdown("---")
    st.caption("Unofficial fan project. Not affiliated with the BCCI, the IPL or any franchise. Not betting advice.")

n_boot, sims = E.PRECISION[precision]
model = get_model(n_boot)
sq_sig, fx_sig = mtime(E.SQUADS), mtime(E.FIXTURES)
with st.spinner("Simulating the season..."):
    fc = forecast_current(model, n_boot, sims, sq_sig, fx_sig, use_fx)
res, ratings = fc["results"], fc["ratings"]
sq = E.load_squads()
coaches = load_coaches()
status = schedule_status()
mv = pd.read_csv(m14.MOVES) if Path(m14.MOVES).exists() else pd.DataFrame(columns=["trade_id", "status"])
n_trades = mv[mv.status.astype(str).str.lower() == "confirmed"].trade_id.nunique() if len(mv) else 0
upd = datetime.datetime.fromtimestamp(sq_sig).strftime("%d %b %Y, %H:%M")
fx_on = fc["schedule_mode"] == "fixtures"

hero([("ok", f"Squads updated {upd} · {n_trades} confirmed trades applied"),
      ("ok" if fx_on else "warn", "Official schedule loaded — real fixtures in use" if fx_on else
       "Schedule not released yet — generic fixture model (it switches automatically)"),
      ("info", f"{fc['sims']:,} simulated seasons · {fc['n_fits']} model re-fits")])
ticker(res)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏆  Title Race", "🔁  Trades & Squads", "📅  Schedule", "🧢  Teams", "🧪  How it works"])

with tab1:
    st.markdown("### Who lifts the trophy?")
    podium(res)
    leaderboard(res, coaches)
    st.caption("Whiskers = 5th–95th percentile across model re-fits. Equal chance would be 10% each. "
               "No model in this project has beaten a coin flip on unseen seasons, so read the spread, not the order.")

with tab2:
    st.markdown("### Trade wire")
    trade_wire()
    st.markdown("### What did the trades change? (model-implied)")
    if Path(E.BASELINE).exists():
        base = forecast_baseline(model, n_boot, sims, fx_sig, use_fx)["results"].set_index("team")["p_title"]
        d = (res.set_index("team")["p_title"] - base).sort_values(ascending=False).reset_index()
        d.columns = ["team", "delta"]
        diverging(d, "delta", lambda v: f"{v * 100:+.1f} pts")
        st.caption("Change in title probability: current squads vs the 2026 squads, same simulation settings. "
                   "Differences of about a point or less are within simulation noise.")
    else:
        st.info("Run `python 14_update_squads.py` once to create the 2026 baseline for comparison.")
    st.markdown("### Trade lab — try your own trade")
    c1, c2, c3 = st.columns([2, 3, 2])
    team_from = c1.selectbox("From", sorted(sq.team.unique()), key="tl_from")
    pool = sq[sq.team == team_from].sort_values(["xi", "score"], ascending=False)
    label = {r.Index: f"{r.player}{'  ★ XI' if r.xi == 1 else ''}" for r in pool.itertuples()}
    pick = c2.selectbox("Player", list(label), format_func=lambda i: label[i], key="tl_player")
    team_to = c3.selectbox("To", [t for t in sorted(sq.team.unique()) if t != team_from], key="tl_to")
    if st.button("▶  Run what-if"):
        sq2 = sq.copy()
        sq2.loc[pick, "team"] = team_to
        sq2 = m14.choose_xi(sq2)
        with st.spinner("Re-simulating the season..."):
            alt = model.forecast(sq2, E.load_fixtures() if use_fx else None, sims)["results"].set_index("team")["p_title"]
        dd = (alt - res.set_index("team")["p_title"]).sort_values(ascending=False).reset_index()
        dd.columns = ["team", "delta"]
        st.markdown(f"**{sq.loc[pick, 'player']}: {team_from} → {team_to}**")
        diverging(dd, "delta", lambda v: f"{v * 100:+.1f} pts")
        st.caption("What-ifs reuse the same random numbers, so differences are meaningful but small effects are still noise.")
    lp = E.RES / "squad_changes_log.csv"
    if lp.exists():
        with st.expander("Squad change log"):
            st.dataframe(pd.read_csv(lp).tail(40), width="stretch", hide_index=True)

with tab3:
    st.markdown("### Schedule")
    fx = E.load_fixtures()
    if fx_on and fx is not None:
        probs = model.fixture_probs(sq, fx)
        fx = fx.assign(p_a=probs)
        st.success(f"{status.get('message', 'Fixtures loaded.')}  ({len(fx)} matches, {status.get('first_date', '')} → {status.get('last_date', '')})")
        st.markdown("#### Next matches — model win probabilities")
        for r in fx.head(12).itertuples():
            pa = r.p_a if r.p_a == r.p_a else 0.5
            ta, tb = art.team(r.team_a), art.team(r.team_b)
            H(f'<div class="fx"><div class="d">{r.date}</div><div class="t">{art.badge(r.team_a, 26)}{ta["code"]}<span class="note">vs</span>{art.badge(r.team_b, 26)}{tb["code"]}</div>'
              f'<div class="note" style="flex:1">{r.venue if isinstance(r.venue, str) else ""}</div>'
              f'<div class="pbar"><div style="width:{pa * 100:.0f}%;background:{ta["c1"]}"></div><div style="width:{(1 - pa) * 100:.0f}%;background:{tb["c1"]}"></div></div>'
              f'<div class="lb-stat" style="min-width:90px;font-size:.8rem">{pa * 100:.0f}% · {(1 - pa) * 100:.0f}%</div></div>')
        st.markdown("#### Schedule difficulty — expected wins from the fixture list")
        stg = fx["stage"].fillna("") if "stage" in fx else pd.Series("", index=fx.index)
        lg = fx[~stg.str.contains("qualifier|eliminator|final", case=False, regex=True)]
        long = pd.concat([pd.DataFrame({"team": lg.team_a, "p": lg.p_a}), pd.DataFrame({"team": lg.team_b, "p": 1 - lg.p_a})])
        dd = (long.groupby("team").p.mean() - 0.5).sort_values(ascending=False).reset_index()
        dd.columns = ["team", "delta"]
        diverging(dd, "delta", lambda v: f"{(v + .5) * 100:.1f}%")
        st.caption("Average model win probability across each team's scheduled games (higher = easier draw / stronger team).")
        with st.expander("Full fixture list"):
            st.dataframe(fx.drop(columns=["p_a"]), width="stretch", hide_index=True)
    else:
        H(f'<div class="card empty"><svg viewBox="0 0 260 60" width="220" height="52">{art.DEFS}{art.stumps(130, 58, .7)}{art.ball(170, 46, 8)}</svg>'
          f'<h3 style="margin:.4rem 0">Schedule not released yet</h3>'
          f'<div class="note">{status.get("message", "")}<br>The IPL has indicated a tentative mid-March to end-May window. '
          f'Until the full league fixture list is out, the forecast uses a generic 70-match format.<br>'
          f'Press <b>Check for schedule</b> in the sidebar — the moment fixtures exist, the app loads them and re-runs the simulation with real home/away games.</div></div>')
        if status.get("status") == "partial":
            st.warning(status.get("message", ""))

with tab4:
    st.markdown("### Squads & staff")
    st.caption("Expected XI = the 11 most recently active players (max 4 overseas where nationality is known). "
               "Coaches are shown for context — the model does not use them.")
    team_cards(sq, coaches, ratings)

with tab5:
    st.markdown("### How it works — and how far to trust it")
    H('<div class="card"><b>The honest summary.</b> I tested Elo, logistic regression, XGBoost, a player-impact model and skill ratings built from '
      'every men\'s T20 match, all with strict walk-forward validation. <b>None beat a coin flip on unseen seasons</b>, and a final test with rules fixed in advance also failed. '
      'So this app shows <b>probabilities with uncertainty</b>, not a pick. Squad changes and the schedule move the numbers a little; they do not turn the IPL into a predictable league.</div>')
    cmp = E.RES / "model_comparison.csv"
    if cmp.exists():
        mc = pd.read_csv(cmp)
        rows = "".join(f'<tr class="{"hl" if r.model == "Coin flip" else ""}"><td>{r.model}</td><td>{r.accuracy * 100:.1f}%</td><td>{r.log_loss:.4f}</td></tr>' for r in mc.itertuples())
        st.markdown("#### Held-out test (IPL 2022–2026) — lower log loss is better; a coin flip scores 0.693")
        H(f'<div class="card"><table class="mt"><tr><th>Model</th><th>Accuracy</th><th>Log loss</th></tr>{rows}</table></div>')
    st.markdown("""
**Pipeline.** Cricsheet ball-by-ball data → leak-free features → models tested on seasons they never saw → Monte Carlo season simulator (league + playoffs).
**Live parts.** *Update squads & trades* reads current squads and coaches and applies a ledger of confirmed trades; *Check for schedule* probes the official IPL feed.
If the schedule is not out, it plays no role. **Forward test:** once fixtures exist, `11_forward_test.py` freezes predictions before the season so IPL 2027 itself can judge the model.
""")

H('<div class="footer">Unofficial fan analytics project · not affiliated with the BCCI / IPL / franchises · team badges are original drawings, not official logos · '
  'ball-by-ball data from Cricsheet (ODbL) · statistics only, not betting advice</div>')

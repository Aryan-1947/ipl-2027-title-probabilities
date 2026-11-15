"""
Step 5 - Make the three README charts from the results of steps 3 and 4.

    charts/title_probability.png   who wins IPL 2027, in the simulation
    charts/model_vs_coinflip.png   did any model beat a coin flip? (the honest chart)
    charts/team_strength_2027.png  batting vs wicket-taking strength of each squad

Usage:  python src/05_charts.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RES, OUT = Path("results"), Path("charts")
SHORT = {
    "Chennai Super Kings": "CSK", "Delhi Capitals": "DC", "Gujarat Titans": "GT",
    "Kolkata Knight Riders": "KKR", "Lucknow Super Giants": "LSG", "Mumbai Indians": "MI",
    "Punjab Kings": "PBKS", "Rajasthan Royals": "RR", "Royal Challengers Bengaluru": "RCB",
    "Sunrisers Hyderabad": "SRH",
}
BLUE, GREY, RED = "#2F5D8C", "#9AA5B1", "#C0392B"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


def title_probability():
    d = pd.read_csv(RES / "title_probabilities.csv").sort_values("p_title")
    labels = d["team"].map(SHORT).fillna(d["team"])
    fig, ax = plt.subplots(figsize=(8, 5.5))
    has_ci = {"p_title_lo", "p_title_hi"} <= set(d.columns)
    err = [(d["p_title"] - d["p_title_lo"]) * 100, (d["p_title_hi"] - d["p_title"]) * 100] if has_ci else None
    ax.barh(labels, d["p_title"] * 100, color=BLUE, xerr=err, ecolor="#333", capsize=3)
    ax.axvline(10, color=RED, ls="--", lw=1.2)
    ax.text(10.15, -0.65, "equal chance (10%)", color=RED, fontsize=9, va="center")
    right = (d["p_title_hi"] if has_ci else d["p_title"]) * 100
    for y, (v, r) in enumerate(zip(d["p_title"] * 100, right)):
        ax.text(r + 0.3, y, f"{v:.1f}%", va="center", fontsize=10)
    ax.set_xlabel("Probability of winning IPL 2027 (%)")
    ax.set_title("IPL 2027 title probability (30,000 simulated seasons)", loc="left", fontweight="bold")
    ax.set_xlim(0, max(right) + 5)
    fig.text(0.01, 0.01, "Bars = average over 30 model re-fits; whiskers = 5th-95th percentile. "
             "Squads = 2026 players; re-run after the Dec 2026 auction.", fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "title_probability.png", dpi=160)
    plt.close(fig)


def model_vs_coinflip():
    d = pd.read_csv(RES / "model_comparison.csv")
    d = d[d["model"] != "Coin flip"].iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.scatter(d["log_loss"], d["model"], s=90, color=BLUE, zorder=3)
    ax.axvline(0.693, color=RED, ls="--", lw=1.2, label="coin flip (0.693)")
    ax.legend(frameon=False, loc="lower right")
    for x, y in zip(d["log_loss"], d["model"]):
        ax.text(x, y, f"  {x:.3f}", va="center", fontsize=10)
    lo, hi = min(d["log_loss"].min(), 0.693), max(d["log_loss"].max(), 0.693)
    ax.set_xlim(lo - 0.006, hi + 0.012)
    ax.set_xlabel("Log loss on unseen seasons (lower is better)")
    ax.set_title("Single-match models vs a coin flip (lower log loss is better)", loc="left", fontweight="bold")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "model_vs_coinflip.png", dpi=160)
    plt.close(fig)


def team_strength():
    d = pd.read_csv(RES / "team_strength_2027.csv")
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.scatter(d["bat"], d["wk"], s=(d["elo"] - d["elo"].min() + 20) * 4, color=BLUE, alpha=0.75)
    for _, r in d.iterrows():
        ax.annotate(SHORT.get(r["team"], r["team"]), (r["bat"], r["wk"]), textcoords="offset points",
                    xytext=(8, 6), fontsize=10)
    ax.axvline(d["bat"].median(), color=GREY, ls=":", lw=1)
    ax.axhline(d["wk"].median(), color=GREY, ls=":", lw=1)
    ax.set_xlabel("Batting strength (top-7 expected runs per match)")
    ax.set_ylabel("Wicket-taking strength (top-5 expected wickets per match)")
    ax.set_title("Squad strength going into 2027 (bubble size = Elo)", loc="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "team_strength_2027.png", dpi=160)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    title_probability()
    model_vs_coinflip()
    team_strength()
    print(f"Saved 3 charts to {OUT}/")


if __name__ == "__main__":
    main()

"""Static figures for README and slides (from results/)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R, F = ROOT / "results", ROOT / "docs" / "figures"
F.mkdir(parents=True, exist_ok=True)
INK, MUTED, GRID, NEUTRAL = "#14202b", "#5a6b79", "#dbe2e8", "#a7b1ba"
SERIES = {"dosetwin_hybrid": "#2a78d6", "cgm_trend_30": "#eb6834", "lightgbm": "#1baf7a", "twin_40": "#eda100"}
NAMES = {"dosetwin_hybrid": "DoseTwin hybrid", "cgm_trend_30": "CGM trend (30 min)", "lightgbm": "LightGBM", "twin_40": "Twin alone"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": .6, "text.color": INK})


def alert_curves():
    al = pd.read_csv(R / "alert_curves.csv")
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for m, c in SERIES.items():
        g = al[al.method == m].groupby("threshold")[["false_alerts_per_day", "sensitivity"]].mean().sort_values("false_alerts_per_day")
        g = g[g.false_alerts_per_day <= 4]
        ax.plot(g.false_alerts_per_day, g.sensitivity * 100, color=c, lw=2, label=NAMES[m])
    ax.axvline(2, color=MUTED, lw=.8, ls=":")
    ax.set_xlim(0, 4); ax.set_ylim(0, 100)
    ax.set_xlabel("False alerts per patient per day"); ax.set_ylabel("Hypoglycaemia events caught (%)")
    ax.set_title("Lows caught vs alarm burden (held-out, 90 patient-weeks)", loc="left", fontsize=11)
    ax.legend(frameon=False, loc="lower right", fontsize=8.5)
    fig.tight_layout(); fig.savefig(F / "alerts.png", dpi=200); plt.close(fig)


def arms():
    a = pd.read_csv(R / "arms_by_patient.csv").groupby("arm").mean(numeric_only=True)
    order = [("A_standard", "Threshold-alarm CGM\n+ calculator"), ("B_standard+trend_alert", "Predictive-alert CGM\n+ calculator"),
             ("D_dosetwin", "DoseTwin")]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.4))
    for ax, (col, title) in zip(axes, [("tbr_lt70", "Time below 70 mg/dL (%)"), ("night_tbr_lt70", "Night-time below 70 (%)"),
                                      ("tir_70_180", "Time in range 70–180 (%)")]):
        vals = [a.loc[k, col] for k, _ in order]
        colors = [NEUTRAL, NEUTRAL, SERIES["dosetwin_hybrid"]]
        bars = ax.barh([n for _, n in order], vals, color=colors, height=.6)
        for b, v in zip(bars, vals):
            ax.text(v + max(vals) * .02, b.get_y() + b.get_height() / 2, f"{v:.1f}" if v > 10 else f"{v:.2f}", va="center", fontsize=9)
        ax.set_title(title, loc="left", fontsize=10); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
        ax.set_xlim(0, max(vals) * 1.25)
        if col == "tir_70_180":
            ax.set_xlim(0, 100)
        if ax is not axes[0]:
            ax.set_yticklabels([])
    fig.suptitle("Outcomes on identical days, held-out cohorts (30 patients × 3)", x=.01, ha="left", fontsize=11)
    fig.tight_layout(); fig.savefig(F / "arms.png", dpi=200); plt.close(fig)


def forecasts():
    fc = pd.read_csv(R / "forecast_by_patient.csv").groupby(["method", "horizon"])[["rmse", "rmse_low"]].mean().reset_index()
    cols = {"personal_twin": ("Personal twin", SERIES["dosetwin_hybrid"]), "lightgbm": ("LightGBM", SERIES["lightgbm"]),
            "persistence": ("CGM stays flat", SERIES["cgm_trend_30"])}
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, (col, title) in zip(axes, [("rmse", "All glucose levels"), ("rmse_low", "When true glucose < 90 mg/dL")]):
        for m, (n, c) in cols.items():
            g = fc[fc.method == m]
            ax.plot(g.horizon, g[col], color=c, lw=2, marker="o", ms=5, label=n)
            ax.annotate(n, (g.horizon.iloc[-1], g[col].iloc[-1]), xytext=(5, 0), textcoords="offset points", va="center", fontsize=8.5)
        ax.set_title(title, loc="left", fontsize=10); ax.set_xticks([30, 60, 120]); ax.set_xlim(20, 165)
        ax.set_xlabel("Forecast horizon (min)")
    axes[0].set_ylabel("RMSE vs true glucose (mg/dL)"); axes[0].legend(frameon=False, fontsize=8.5)
    fig.tight_layout(); fig.savefig(F / "forecast.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    alert_curves(); arms(); forecasts()
    print("figures written to", F)

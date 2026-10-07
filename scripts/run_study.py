"""DoseTwin in-silico study: reproduces every number in the README / deck.

    python scripts/run_study.py            # ~15 min on a laptop CPU

Design
------
* 30 hidden virtual patients (UVA/Padova parameter set: 10 children,
  10 adolescents, 10 adults), 14 days of MDI life each (see truth/cohort.py).
* Development cohort (seed 7): used ONLY to choose the frozen settings
  (fit horizon, prior strength, advisor margin, alert threshold) and to train
  the LightGBM baseline.
* Held-out cohorts (seeds 2026, 2027, 2028): new meals, behaviour, walks and
  sensor noise. Nothing is tuned on them.
* Days 1-7: standard care (run-in); twins are fitted on these days only.
  Days 8-14: evaluation; arms share identical meals/errors/noise (paired).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dosetwin.alerts import HybridAlert, TrendAlert, TwinAlert  # noqa: E402
from dosetwin.baselines import GBMForecaster  # noqa: E402
from dosetwin.ehr import build_records, ehr_table  # noqa: E402
from dosetwin.evaluation import alert_event_metrics  # noqa: E402
from dosetwin.metrics import glycemic_metrics  # noqa: E402
from dosetwin.truth.cohort import StandardCalculator, World  # noqa: E402
from dosetwin.truth.uva_padova import load_params, true_settings  # noqa: E402
from dosetwin.twin import identify as I  # noqa: E402
from dosetwin.twin import model as M  # noqa: E402
from dosetwin.twin.advisor import TwinAdvisor  # noqa: E402
from dosetwin.twin.tracker import TwinTracker  # noqa: E402

D = 1440
DEV_SEED = 7
TEST_SEEDS = [2026, 2027, 2028]
FIT_H, PRIOR_STRENGTH = 300, 150.0          # frozen on dev
TREND_ALERT_THR = 50.0                      # trend30 at <=2 false alerts/day on dev
HYBRID_ALERT_THR = 65.0                     # hybrid at <=2 false alerts/day on dev
HORIZONS = (30, 60, 120)
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
NAMES = list(load_params()["Name"])


def log_msg(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def fit_twins(log, ehr):
    thetas, priors, secs = {}, {}, []
    for i, n in enumerate(NAMES):
        p = I.from_log(log, ehr, i).slice(0, 7 * D)
        t0 = time.time()
        fr = I.fit(p, H=FIT_H, prior_strength=PRIOR_STRENGTH)
        secs.append(time.time() - t0)
        thetas[n], priors[n] = fr.theta, fr.prior
    return thetas, priors, secs


def forecast_eval(log, ehr, thetas, priors, gbm, seed):
    rows = []
    starts = np.arange(7 * D, 14 * D - max(HORIZONS) - 5, 5, dtype=np.int64)
    for i, n in enumerate(NAMES):
        p = I.from_log(log, ehr, i)
        y_now = log["cgm"][i, starts]
        preds = {}
        for tag, th in (("personal_twin", thetas[n]), ("population_twin", priors[n])):
            z0 = M.steady_state(th, p.basal, 120.0)
            Z = M.sync_run(th, p.u, p.basal, p.c, p.steps, p.W, p.cgm, 0, I.DEFAULT_GAINS, z0)
            F = M.window_forecasts(th, Z, p.u, p.basal, p.c, p.steps, p.W, 0, starts, max(HORIZONS), False)
            preds[tag] = {h: F[:, h] for h in HORIZONS}
        g = gbm.predict(log, ehr, i, starts)
        preds["lightgbm"] = {h: g[h] for h in HORIZONS}
        preds["persistence"] = {h: y_now for h in HORIZONS}
        preds["cgm_trend"] = {h: np.array([TrendAlert(h).score(i, int(s), log) for s in starts]) for h in HORIZONS}
        for m, ph in preds.items():
            for h in HORIZONS:
                y = log["bg"][i, starts + h]
                e = ph[h] - y
                low = y < 90
                rows.append({"seed": seed, "patient": n, "method": m, "horizon": h,
                             "rmse": float(np.sqrt(np.mean(e ** 2))),
                             "mae": float(np.mean(np.abs(e))),
                             "rmse_low": float(np.sqrt(np.mean(e[low] ** 2))) if low.any() else np.nan,
                             "n_low": int(low.sum())})
    return pd.DataFrame(rows)


def alert_scores(log, ehr, thetas, gbm):
    times = np.arange(7 * D, 14 * D, 5)
    tr = TwinTracker(thetas, ehr, NAMES)
    S = {"cgm_trend_30": [], "twin_40": [], "dosetwin_hybrid": [], "lightgbm": []}
    tw, t30 = TwinAlert(tr, 40), TrendAlert(30)
    for i in range(len(NAMES)):
        a = np.array([tw.score(i, int(t), log) for t in times])
        b = np.array([t30.score(i, int(t), log) for t in times])
        S["twin_40"].append(a)
        S["cgm_trend_30"].append(b)
        S["dosetwin_hybrid"].append(0.5 * (a + b))
        S["lightgbm"].append(-gbm.predict(log, ehr, i, times)["p_hypo60"])
    return times, {k: np.array(v) for k, v in S.items()}


def alert_curves(times, S, bg, seed):
    rows = []
    grids = {"cgm_trend_30": np.arange(20, 121, 2.5), "twin_40": np.arange(40, 131, 2.5),
             "dosetwin_hybrid": np.arange(30, 126, 2.5), "lightgbm": -np.linspace(0.995, 0.01, 60)}
    for m, s in S.items():
        for th in grids[m]:
            r = alert_event_metrics(s, th, bg, times)
            r.update({"seed": seed, "method": m, "threshold": float(th)})
            rows.append(r)
    return pd.DataFrame(rows)


def run_arms(w, ehr, thetas, priors):
    std = StandardCalculator(ehr)

    def after7(alerter):
        return (lambda i, t, log, c: t >= 7 * D and alerter(i, t, log, c)) if alerter else None

    trp, trq = TwinTracker(thetas, ehr, NAMES), TwinTracker(priors, ehr, NAMES)
    arms = {
        "A_standard": (std, None),
        "B_standard+trend_alert": (std, TrendAlert(30, TREND_ALERT_THR)),
        "C_twin_dose+trend_alert": (TwinAdvisor(thetas, ehr, NAMES, tracker=TwinTracker(thetas, ehr, NAMES)),
                                    TrendAlert(30, TREND_ALERT_THR)),
        "D_dosetwin": (TwinAdvisor(thetas, ehr, NAMES, tracker=trp), HybridAlert(trp, HYBRID_ALERT_THR)),
        "E_dosetwin_population_prior": (TwinAdvisor(priors, ehr, NAMES, tracker=trq), HybridAlert(trq, HYBRID_ALERT_THR)),
    }
    rows, decisions = [], []
    for k, (pol, al) in arms.items():
        t0 = time.time()
        lg = w.run(lambda d, pol=pol: std if d < 7 else pol, alerter=after7(al))
        for i, n in enumerate(NAMES):
            m = glycemic_metrics(lg["bg"][i, 7 * D:])
            night = np.concatenate([lg["bg"][i, d * D:d * D + 7 * 60] for d in range(7, 14)])
            mn = glycemic_metrics(night)
            ev = [e for e in lg["events"][i] if e[0] >= 7 * D]
            m.update({"night_tbr_lt70": mn["tbr_lt70"], "night_tbr_lt54": mn["tbr_lt54"],
                      "alerts_per_day": sum(e[1] == "predicted_low_alert" for e in ev) / 7,
                      "rescue_carbs_per_day": sum(e[3] for e in ev if e[1] in ("hypo_rescue", "preemptive_carbs")) / 7,
                      "bolus_u_per_day": float(lg["bolus"][i, 7 * D:].sum() / 7),
                      "arm": k, "patient": n, "seed": w.seed})
            rows.append(m)
        if hasattr(pol, "decisions"):
            decisions += [(w.seed, k) + d for d in pol.decisions if d[1] >= 7 * D]
        log_msg(f"  arm {k} done in {time.time() - t0:.0f}s")
    return pd.DataFrame(rows), decisions


def paired(df, a, b, metric):
    """Paired difference b - a, averaged per patient across seeds (n=30),
    with a patient-level bootstrap 95% CI and Wilcoxon signed-rank p."""
    pa = df[df.arm == a].groupby("patient")[metric].mean()
    pb = df[df.arm == b].groupby("patient")[metric].mean()
    d = (pb - pa).to_numpy()
    rng = np.random.RandomState(0)
    boots = [rng.choice(d, len(d)).mean() for _ in range(5000)]
    try:
        p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    except ValueError:
        p = 1.0
    return {"comparison": f"{b} vs {a}", "metric": metric, "mean_a": float(pa.mean()), "mean_b": float(pb.mean()),
            "diff": float(d.mean()), "ci_low": float(np.percentile(boots, 2.5)),
            "ci_high": float(np.percentile(boots, 97.5)), "wilcoxon_p": p,
            "patients_improved": int(np.sum(d > 0)), "n": int(len(d))}


def main():
    t_all = time.time()
    # ---------------- development cohort: train the ML baseline ------------
    log_msg("dev cohort (seed 7): simulate + train LightGBM baseline")
    wd = World(NAMES, days=14, seed=DEV_SEED)
    ehr_d = wd.ehr()
    log_d = wd.run(lambda d: StandardCalculator(ehr_d))
    ehr_d = ehr_table(wd, log_d, DEV_SEED)
    gbm = GBMForecaster(HORIZONS).fit(log_d, ehr_d, np.arange(120, 14 * D - 130, 5))

    truth = true_settings(NAMES).set_index("patient")
    fc, al, arms_all, settings, decisions, fit_secs = [], [], [], [], [], []
    for seed in TEST_SEEDS:
        log_msg(f"held-out cohort seed {seed}")
        w = World(NAMES, days=14, seed=seed)
        ehr = w.ehr()
        log = w.run(lambda d: StandardCalculator(ehr))
        ehr = ehr_table(w, log, seed)  # adds the HbA1c lab from the synthetic EHR
        (OUT / f"ehr_records_seed{seed}.json").write_text(json.dumps(build_records(w, log, seed), indent=1))
        thetas, priors, secs = fit_twins(log, ehr)
        fit_secs += secs
        log_msg(f"  twins fitted: median {np.median(secs):.2f}s/patient")
        for i, n in enumerate(NAMES):
            p = I.from_log(log, ehr, i).slice(0, 7 * D)
            s = I.implied_settings(thetas[n], p)
            settings.append({"seed": seed, "patient": n, "cr_prescribed": p.CR, "cr_twin": s["cr"],
                             "cr_true": truth.loc[n, "true_cr"], "cf_prescribed": p.CF, "cf_twin": s["cf"],
                             "cf_true": truth.loc[n, "true_cf"],
                             **{f"theta_{k}": float(v) for k, v in zip(M.P_NAMES, thetas[n])}})
        fc.append(forecast_eval(log, ehr, thetas, priors, gbm, seed))
        log_msg("  forecasts evaluated")
        times, S = alert_scores(log, ehr, thetas, gbm)
        al.append(alert_curves(times, S, log["bg"], seed))
        log_msg("  alerts evaluated")
        a, dec = run_arms(w, ehr, thetas, priors)
        arms_all.append(a)
        decisions += dec

    fc = pd.concat(fc); al = pd.concat(al); arms = pd.concat(arms_all)
    settings = pd.DataFrame(settings)
    fc.to_csv(OUT / "forecast_by_patient.csv", index=False)
    al.to_csv(OUT / "alert_curves.csv", index=False)
    arms.to_csv(OUT / "arms_by_patient.csv", index=False)
    settings.to_csv(OUT / "settings_recovery.csv", index=False)
    pd.DataFrame(decisions, columns=["seed", "arm", "patient", "minute", "carbs_logged", "dose", "standard_dose"]) \
        .to_csv(OUT / "dose_decisions.csv", index=False)

    # ----------------------------- summary ----------------------------------
    summary = {"design": {"patients": len(NAMES), "test_seeds": TEST_SEEDS, "dev_seed": DEV_SEED,
                          "days_runin": 7, "days_eval": 7},
               "fit_seconds_median": float(np.median(fit_secs))}
    summary["forecast_rmse"] = (fc.groupby(["method", "horizon"])[["rmse", "rmse_low"]].mean()
                                .round(2).reset_index().to_dict(orient="records"))
    summary["glycemic_by_arm"] = (arms.groupby("arm")[["tir_70_180", "tbr_lt70", "tbr_lt54", "tar_gt180",
                                                       "tar_gt250", "night_tbr_lt70", "night_tbr_lt54",
                                                       "lbgi", "hbgi", "mean", "cv", "alerts_per_day",
                                                       "rescue_carbs_per_day", "bolus_u_per_day"]]
                                  .mean().round(3).reset_index().to_dict(orient="records"))
    comps = []
    for a, b in [("B_standard+trend_alert", "D_dosetwin"), ("A_standard", "D_dosetwin"),
                 ("E_dosetwin_population_prior", "D_dosetwin"), ("B_standard+trend_alert", "C_twin_dose+trend_alert")]:
        for mtr in ["tir_70_180", "tbr_lt70", "tbr_lt54", "tar_gt250", "night_tbr_lt70"]:
            comps.append(paired(arms, a, b, mtr))
    summary["paired_comparisons"] = comps

    def err(col):
        return np.abs(np.log(settings[col] / settings[col.split("_")[0] + "_true"]))
    summary["settings_recovery"] = {
        "cr_median_abs_pct_error_prescribed": float(np.expm1(err("cr_prescribed").median()) * 100),
        "cr_median_abs_pct_error_twin": float(np.expm1(err("cr_twin").median()) * 100),
        "cf_median_abs_pct_error_prescribed": float(np.expm1(err("cf_prescribed").median()) * 100),
        "cf_median_abs_pct_error_twin": float(np.expm1(err("cf_twin").median()) * 100),
        "cr_twin_closer_fraction": float((err("cr_twin") < err("cr_prescribed")).mean()),
        "cf_twin_closer_fraction": float((err("cf_twin") < err("cf_prescribed")).mean()),
    }
    ops = []
    for budget in (1.0, 2.0, 3.0):
        for m, g in al.groupby("method"):
            agg = g.groupby("threshold")[["sensitivity", "false_alerts_per_day", "median_lead_min", "precision"]].mean()
            ok = agg[agg.false_alerts_per_day <= budget]
            if len(ok):
                best = ok.sort_values("sensitivity").iloc[-1]
                ops.append({"budget_false_per_day": budget, "method": m, **best.round(3).to_dict()})
    summary["alerts_at_matched_false_alarm"] = ops
    summary["runtime_minutes"] = round((time.time() - t_all) / 60, 1)
    (OUT / "study.json").write_text(json.dumps(summary, indent=2))
    log_msg("done", summary["runtime_minutes"], "min")


if __name__ == "__main__":
    main()

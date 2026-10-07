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
* Frozen on the development cohort: the eGFR coefficient of the EHR prior,
  the bedtime-check threshold (80% specificity) and both LightGBM baselines.
* Ablations (refitted, not just switched off): twin without the EHR labs and
  diagnosis (HbA1c, eGFR, history of severe hypoglycaemia) and twin without
  the extra wearable streams (heart rate and sleep).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dosetwin import bedtime as BT  # noqa: E402
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
LAB_COLS = ["hba1c_pct", "egfr_ml_min", "severe_hypo_history"]
EVAL_NIGHTS = (7, 12)                       # nights starting on days 8..13 (all inside days 8-14)
OUT = ROOT / "results"
NAMES = list(load_params()["Name"])
if os.environ.get("DOSETWIN_QUICK"):  # smoke test: 6 patients, one held-out cohort
    NAMES, TEST_SEEDS, OUT = NAMES[::5], TEST_SEEDS[:1], Path(os.environ["DOSETWIN_QUICK"])
OUT.mkdir(exist_ok=True, parents=True)


def log_msg(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def patient(log, ehr, i, variant="full"):
    """variant: full | no_labs (EHR labs/diagnosis dropped) | no_wearables (steps only, no HR or sleep)."""
    p = I.from_log(log, ehr, i, use_ehr_labs=(variant != "no_labs"))
    if variant == "no_wearables":
        p.act = log["steps"][i].astype(float).copy()
        p.sd = np.zeros_like(p.sd)
    return p


def wearable_log(log, variant):
    """Device log as seen by a twin variant."""
    if variant != "no_wearables":
        return log
    v = dict(log)
    v["act"] = log["steps"].astype(float)
    v["sleep_debt"] = np.zeros_like(log["sleep_debt"])
    return v


def fit_twins(log, ehr, variant="full"):
    thetas, priors, secs = {}, {}, []
    for i, n in enumerate(NAMES):
        p = patient(log, ehr, i, variant).slice(0, 7 * D)
        prior = I.ehr_prior(p)
        if variant == "no_wearables":
            prior[M.I_BS] = 0.0
        t0 = time.time()
        fr = I.fit(p, prior=prior, H=FIT_H, prior_strength=PRIOR_STRENGTH)
        secs.append(time.time() - t0)
        thetas[n], priors[n] = fr.theta, fr.prior
    return thetas, priors, secs


def estimate_egfr_coef(log, ehr):
    """Dev cohort only: how much more potent is insulin per unit of kidney impairment?
    Slope of (fitted - prior) log insulin sensitivity on clip((90 - eGFR)/60, 0, 1)."""
    I.EGFR_COEF = 0.0
    thetas, priors, _ = fit_twins(log, ehr)
    x = np.array([I.kidney_impairment(ehr.iloc[i]["egfr_ml_min"]) for i in range(len(NAMES))])
    y = np.array([thetas[n][M.I_LSI] - priors[n][M.I_LSI] for n in NAMES])
    slope = float(np.polyfit(x, y, 1)[0])
    return max(slope, 0.0), thetas


def forecast_eval(log, ehr, twins, gbm, seed):
    """twins: {method_name: (thetas, variant)}"""
    rows = []
    starts = np.arange(7 * D, 14 * D - max(HORIZONS) - 5, 5, dtype=np.int64)
    for i, n in enumerate(NAMES):
        y_now = log["cgm"][i, starts]
        preds = {}
        for tag, (ths, variant) in twins.items():
            p = patient(log, ehr, i, variant)
            th = ths[n]
            z0 = M.steady_state(th, p.basal, 120.0)
            Z = M.sync_run(th, p.u, p.basal, p.c, p.act, p.sd, p.W, p.cgm, 0, I.DEFAULT_GAINS, z0)
            F = M.window_forecasts(th, Z, p.u, p.basal, p.c, p.act, p.sd, p.W, 0, starts, max(HORIZONS), False)
            preds[tag] = {h: F[:, h] for h in HORIZONS}
        g = gbm.predict(log, ehr, i, starts)
        preds["lightgbm"] = {h: g[h] for h in HORIZONS}
        preds["persistence"] = {h: y_now for h in HORIZONS}
        preds["cgm_trend"] = {h: np.array([TrendAlert(h).score(i, int(s), log) for s in starts]) for h in HORIZONS}
        # moments after exercise that the step counter cannot see (cycling, gym, yoga)
        hr_only = np.array([(log["act"][i, max(0, s - 120):s] > log["steps"][i, max(0, s - 120):s] + 20).any()
                            for s in starts])
        short_night = log["sleep_debt"][i, starts] >= 2.0
        for m, ph in preds.items():
            for h in HORIZONS:
                y = log["bg"][i, starts + h]
                e = ph[h] - y
                low = y < 90
                rows.append({"seed": seed, "patient": n, "method": m, "horizon": h,
                             "rmse": float(np.sqrt(np.mean(e ** 2))),
                             "mae": float(np.mean(np.abs(e))),
                             "rmse_low": float(np.sqrt(np.mean(e[low] ** 2))) if low.any() else np.nan,
                             "rmse_after_hr_only_exercise": float(np.sqrt(np.mean(e[hr_only] ** 2))) if hr_only.any() else np.nan,
                             "rmse_after_short_night": float(np.sqrt(np.mean(e[short_night] ** 2))) if short_night.any() else np.nan,
                             "n_low": int(low.sum())})
    return pd.DataFrame(rows)


def night_table(log, ehr, trackers, first_day, last_day, seed):
    """One row per patient-night: label, lead time, features and twin scores."""
    rows = []
    for i, n in enumerate(NAMES):
        for t in BT.night_starts(first_day, last_day):
            cg = log["cgm"][i, t - (t % 5)]
            if not np.isfinite(cg) or cg < 70:
                continue
            ev, lead = BT.night_label(log["bg"][i], t)
            r = {"seed": seed, "patient": n, "night_start": t, "event": int(ev), "lead_min": lead,
                 **dict(zip(BT.FEATURE_NAMES, BT.night_features(log, ehr, i, t)))}
            for tag, (tr, lg) in trackers.items():
                sc = BT.twin_score(tr, i, t, lg)
                r[f"{tag}_min_pes"] = sc["twin_min_pessimistic"]
                r[f"{tag}_min_nom"] = sc["twin_min_nominal"]
            rows.append(r)
    return pd.DataFrame(rows)


TWIN_NIGHT_FEATURES = ["twin_min_pes", "twin_min_nom"]


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


def run_arms(w, ehr, thetas, priors, thetas_nolab, bed_thr):
    std = StandardCalculator(ehr)
    ehr_nolab = ehr.drop(columns=LAB_COLS)

    def after7(alerter):
        return (lambda i, t, log, c: t >= 7 * D and alerter(i, t, log, c)) if alerter else None

    def bed_after7(chk):
        return lambda i, t, log: chk(i, t, log) if t >= 7 * D else 0.0

    trp, trq = TwinTracker(thetas, ehr, NAMES), TwinTracker(priors, ehr, NAMES)
    trn = TwinTracker(thetas_nolab, ehr_nolab, NAMES)
    trg = TwinTracker(thetas, ehr, NAMES)
    arms = {
        "A_standard": (std, None, None),
        "B_standard+trend_alert": (std, TrendAlert(30, TREND_ALERT_THR), None),
        "C_twin_dose+trend_alert": (TwinAdvisor(thetas, ehr, NAMES, tracker=TwinTracker(thetas, ehr, NAMES)),
                                    TrendAlert(30, TREND_ALERT_THR), None),
        "D_dosetwin": (TwinAdvisor(thetas, ehr, NAMES, tracker=trp), HybridAlert(trp, HYBRID_ALERT_THR, ehr), None),
        "E_dosetwin_population_prior": (TwinAdvisor(priors, ehr, NAMES, tracker=trq),
                                        HybridAlert(trq, HYBRID_ALERT_THR, ehr), None),
        "F_dosetwin_without_ehr_labs": (TwinAdvisor(thetas_nolab, ehr_nolab, NAMES, tracker=trn),
                                        HybridAlert(trn, HYBRID_ALERT_THR, ehr_nolab), None),
        "G_dosetwin+bedtime_check": (TwinAdvisor(thetas, ehr, NAMES, tracker=trg), HybridAlert(trg, HYBRID_ALERT_THR, ehr),
                                     BT.BedtimeCheck(trg, bed_thr, ehr)),
    }
    rows, decisions = [], []
    for k, (pol, al, bed) in arms.items():
        t0 = time.time()
        lg = w.run(lambda d, pol=pol: std if d < 7 else pol, alerter=after7(al),
                   bedtime=bed_after7(bed) if bed else None)
        for i, n in enumerate(NAMES):
            m = glycemic_metrics(lg["bg"][i, 7 * D:])
            night = np.concatenate([lg["bg"][i, d * D:d * D + 7 * 60] for d in range(8, 14)])
            mn = glycemic_metrics(night)
            ev = [e for e in lg["events"][i] if e[0] >= 7 * D]
            b = w.behaviours[i]
            m.update({"night_tbr_lt70": mn["tbr_lt70"], "night_tbr_lt54": mn["tbr_lt54"],
                      "night_events": sum(BT.night_label(lg["bg"][i], t)[0] for t in BT.night_starts(*EVAL_NIGHTS)),
                      "alerts_per_day": sum(e[1] == "predicted_low_alert" for e in ev) / 7,
                      "bedtime_snacks_per_week": sum(e[1] == "bedtime_snack" for e in ev),
                      "rescue_carbs_per_day": sum(e[3] for e in ev if e[1] in ("hypo_rescue", "preemptive_carbs",
                                                                                "bedtime_snack")) / 7,
                      "bolus_u_per_day": float(lg["bolus"][i, 7 * D:].sum() / 7),
                      "impaired_awareness": int(b.impaired_awareness), "egfr_true": b.egfr,
                      "severe_hypo_history": int(ehr.iloc[i]["severe_hypo_history"]),
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


def night_scores(nt, gbm_n, gbm_nt):
    X = nt[BT.FEATURE_NAMES].to_numpy(float)
    Xt = np.column_stack([X, nt[["full_min_pes", "full_min_nom"]].to_numpy(float)])
    return {
        # history of severe hypoglycaemia moves the alarm up by 7 mg/dL (same as BedtimeCheck)
        "dosetwin_bedtime": -nt["full_min_pes"].to_numpy() + 7.0 * nt["severe_hypo_history"].fillna(0).to_numpy(),
        "dosetwin_bedtime_without_ehr_labs": -nt["no_labs_min_pes"].to_numpy(),
        "dosetwin_bedtime_without_hr_sleep": -nt["no_wearables_min_pes"].to_numpy(),
        "bedtime_glucose_rule": -nt["cgm_bed"].to_numpy(),
        "lightgbm_night": gbm_n.predict(X),
        "lightgbm_night+twin": gbm_nt.predict(Xt),
    }


def oof_predict(X, y, groups, k=5):
    """Out-of-fold LightGBM probabilities, folds split by patient."""
    pats = np.unique(groups)
    rng = np.random.RandomState(0)
    rng.shuffle(pats)
    out = np.zeros(len(y))
    for f in range(k):
        test = np.isin(groups, pats[f::k])
        out[test] = BT.NightGBM().fit(X[~test], y[~test]).predict(X[test])
    return out


def patient_bootstrap_auc(nt, s, n_boot=2000):
    from dosetwin.bedtime import auroc
    pats = nt["patient"].unique()
    rng = np.random.RandomState(0)
    idx = {p: np.where(nt["patient"].to_numpy() == p)[0] for p in pats}
    y = nt["event"].to_numpy()
    out = []
    for _ in range(n_boot):
        k = np.concatenate([idx[p] for p in rng.choice(pats, len(pats))])
        out.append(auroc(s[k], y[k]))
    return float(np.nanpercentile(out, 2.5)), float(np.nanpercentile(out, 97.5))


def main():
    t_all = time.time()
    # ---------------- development cohort ----------------------------------
    log_msg("dev cohort (seed 7): simulate, train LightGBM, estimate eGFR coefficient, bedtime threshold")
    wd = World(NAMES, days=14, seed=DEV_SEED)
    ehr_d = wd.ehr()
    log_d = wd.run(lambda d: StandardCalculator(ehr_d))
    ehr_d = ehr_table(wd, log_d, DEV_SEED)
    gbm = GBMForecaster(HORIZONS).fit(log_d, ehr_d, np.arange(120, 14 * D - 130, 5))
    I.EGFR_COEF, _ = estimate_egfr_coef(log_d, ehr_d)
    log_msg(f"  eGFR coefficient (frozen): {I.EGFR_COEF:.3f}")
    th_d, _, _ = fit_twins(log_d, ehr_d)
    tr_d = TwinTracker(th_d, ehr_d, NAMES)
    nt_dev_all = night_table(log_d, ehr_d, {}, 1, 12, DEV_SEED)
    nt_dev = night_table(log_d, ehr_d, {"full": (tr_d, log_d)}, *EVAL_NIGHTS, DEV_SEED)
    gbm_n = BT.NightGBM().fit(nt_dev_all[BT.FEATURE_NAMES].to_numpy(float), nt_dev_all["event"].to_numpy())
    gbm_nt = BT.NightGBM(with_twin=True).fit(
        np.column_stack([nt_dev[BT.FEATURE_NAMES].to_numpy(float), nt_dev[["full_min_pes", "full_min_nom"]].to_numpy(float)]),
        nt_dev["event"].to_numpy())
    neg = nt_dev.loc[nt_dev.event == 0, "full_min_pes"].to_numpy()
    bed_thr = float(np.percentile(neg, 20))  # alert if pessimistic night minimum below this: 80% specificity on dev
    log_msg(f"  bedtime threshold (frozen): {bed_thr:.1f} mg/dL; dev night-low prevalence {nt_dev.event.mean():.2f}")

    truth = true_settings(NAMES).set_index("patient")
    fc, al, arms_all, settings, decisions, fit_secs, nights = [], [], [], [], [], [], []
    for seed in TEST_SEEDS:
        log_msg(f"held-out cohort seed {seed}")
        w = World(NAMES, days=14, seed=seed)
        ehr = w.ehr()
        log = w.run(lambda d: StandardCalculator(ehr))
        ehr = ehr_table(w, log, seed)  # adds HbA1c, eGFR and hypo-history from the synthetic EHR
        (OUT / f"ehr_records_seed{seed}.json").write_text(json.dumps(build_records(w, log, seed), indent=1))
        thetas, priors, secs = fit_twins(log, ehr)
        th_nl, pr_nl, _ = fit_twins(log, ehr, "no_labs")
        th_nw, _, _ = fit_twins(log, ehr, "no_wearables")
        fit_secs += secs
        log_msg(f"  twins fitted: median {np.median(secs):.2f}s/patient (+2 ablation fits each)")
        for i, n in enumerate(NAMES):
            p = I.from_log(log, ehr, i).slice(0, 7 * D)
            s = I.implied_settings(thetas[n], p)
            settings.append({"seed": seed, "patient": n, "cr_prescribed": p.CR, "cr_twin": s["cr"],
                             "cr_true": truth.loc[n, "true_cr"], "cf_prescribed": p.CF, "cf_twin": s["cf"],
                             "cf_true": truth.loc[n, "true_cf"], "sleep_sens_true": w.behaviours[i].sleep_sens,
                             "egfr_lab": p.egfr, "clearance_true": w.behaviours[i].clearance,
                             **{f"theta_{k}": float(v) for k, v in zip(M.P_NAMES, thetas[n])}})
        fc.append(forecast_eval(log, ehr, {"personal_twin": (thetas, "full"), "population_twin": (priors, "full"),
                                           "personal_twin_without_hr_sleep": (th_nw, "no_wearables"),
                                           "population_twin_without_ehr_labs": (pr_nl, "no_labs")}, gbm, seed))
        log_msg("  forecasts evaluated")
        ehr_nl = ehr.drop(columns=LAB_COLS)
        trackers = {"full": (TwinTracker(thetas, ehr, NAMES), log),
                    "no_labs": (TwinTracker(th_nl, ehr_nl, NAMES), log),
                    "no_wearables": (TwinTracker(th_nw, ehr, NAMES), wearable_log(log, "no_wearables"))}
        nights.append(night_table(log, ehr, trackers, *EVAL_NIGHTS, seed))
        log_msg("  bedtime predictions evaluated")
        times, S = alert_scores(log, ehr, thetas, gbm)
        al.append(alert_curves(times, S, log["bg"], seed))
        log_msg("  alerts evaluated")
        a, dec = run_arms(w, ehr, thetas, priors, th_nl, bed_thr)
        arms_all.append(a)
        decisions += dec

    fc = pd.concat(fc); al = pd.concat(al); arms = pd.concat(arms_all); nt = pd.concat(nights, ignore_index=True)
    settings = pd.DataFrame(settings)
    fc.to_csv(OUT / "forecast_by_patient.csv", index=False)
    al.to_csv(OUT / "alert_curves.csv", index=False)
    arms.to_csv(OUT / "arms_by_patient.csv", index=False)
    settings.to_csv(OUT / "settings_recovery.csv", index=False)
    pd.DataFrame(decisions, columns=["seed", "arm", "patient", "minute", "carbs_logged", "dose", "standard_dose"]) \
        .to_csv(OUT / "dose_decisions.csv", index=False)

    # ----------------------------- bedtime ----------------------------------
    S = night_scores(nt, gbm_n, gbm_nt)
    for k, v in S.items():
        nt[f"score_{k}"] = v
    nt.to_csv(OUT / "bedtime_nights.csv", index=False)
    y = nt["event"].to_numpy().astype(bool)
    # frozen operating points: 80% specificity on the development cohort
    # (LightGBM thresholds from out-of-fold, patient-grouped predictions on dev)
    neg_d = nt_dev.event.to_numpy() == 0
    dev_scores = {"bedtime_glucose_rule": -nt_dev["cgm_bed"].to_numpy(),
                  "lightgbm_night": oof_predict(nt_dev_all[BT.FEATURE_NAMES].to_numpy(float), nt_dev_all["event"].to_numpy(),
                                                nt_dev_all["patient"].to_numpy())[nt_dev_all.night_start.to_numpy() >= EVAL_NIGHTS[0] * D],
                  "lightgbm_night+twin": oof_predict(np.column_stack([nt_dev[BT.FEATURE_NAMES].to_numpy(float),
                                                                      nt_dev[["full_min_pes", "full_min_nom"]].to_numpy(float)]),
                                                     nt_dev["event"].to_numpy(), nt_dev["patient"].to_numpy())}
    neg_all = nt_dev_all.loc[nt_dev_all.night_start >= EVAL_NIGHTS[0] * D, "event"].to_numpy() == 0
    bed = []
    for k, s in S.items():
        if k.startswith("dosetwin_bedtime"):
            thr = -bed_thr
        elif k == "lightgbm_night":
            thr = float(np.percentile(dev_scores[k][neg_all], 80))
        else:
            thr = float(np.percentile(dev_scores[k][neg_d], 80))
        flag = s > thr if thr is not None else None
        lo, hi = patient_bootstrap_auc(nt, s)
        r = {"method": k, "auroc": BT.auroc(s, y), "auroc_ci_low": lo, "auroc_ci_high": hi,
             "nights": int(len(y)), "nights_with_low": int(y.sum())}
        if flag is not None:
            r.update({"sensitivity": float(flag[y].mean()), "specificity": float((~flag[~y]).mean()),
                      "flags_per_week": float(flag.mean() * 7),
                      "median_warning_h": float(np.nanmedian(nt.loc[flag & y, "lead_min"]) / 60),
                      "iqr_warning_h": [float(np.nanpercentile(nt.loc[flag & y, "lead_min"], q) / 60) for q in (25, 75)]})
        bed.append(r)

    # ----------------------------- summary ----------------------------------
    summary = {"design": {"patients": len(NAMES), "test_seeds": TEST_SEEDS, "dev_seed": DEV_SEED,
                          "days_runin": 7, "days_eval": 7},
               "frozen_on_dev": {"egfr_coef": I.EGFR_COEF, "bedtime_threshold_mgdl": bed_thr},
               "fit_seconds_median": float(np.median(fit_secs))}
    summary["forecast_rmse"] = (fc.groupby(["method", "horizon"])[["rmse", "rmse_low", "rmse_after_hr_only_exercise",
                                                                  "rmse_after_short_night"]].mean()
                                .round(2).reset_index().to_dict(orient="records"))
    cols = ["tir_70_180", "tbr_lt70", "tbr_lt54", "tar_gt180", "tar_gt250", "night_tbr_lt70", "night_tbr_lt54",
            "night_events", "lbgi", "hbgi", "mean", "cv", "alerts_per_day", "bedtime_snacks_per_week",
            "rescue_carbs_per_day", "bolus_u_per_day"]
    summary["glycemic_by_arm"] = arms.groupby("arm")[cols].mean().round(3).reset_index().to_dict(orient="records")
    summary["glycemic_by_arm_subgroup"] = {
        "severe_hypo_history": arms[arms.severe_hypo_history == 1].groupby("arm")[cols].mean().round(3).reset_index().to_dict(orient="records"),
        "egfr_below_60": arms[arms.egfr_true < 60].groupby("arm")[cols].mean().round(3).reset_index().to_dict(orient="records"),
        "n_patient_weeks": {"severe_hypo_history": int((arms[arms.arm == "A_standard"].severe_hypo_history == 1).sum()),
                            "egfr_below_60": int((arms[arms.arm == "A_standard"].egfr_true < 60).sum())}}
    comps = []
    for a, b in [("B_standard+trend_alert", "D_dosetwin"), ("A_standard", "D_dosetwin"),
                 ("E_dosetwin_population_prior", "D_dosetwin"), ("B_standard+trend_alert", "C_twin_dose+trend_alert"),
                 ("F_dosetwin_without_ehr_labs", "D_dosetwin"), ("D_dosetwin", "G_dosetwin+bedtime_check"),
                 ("B_standard+trend_alert", "G_dosetwin+bedtime_check"), ("A_standard", "G_dosetwin+bedtime_check")]:
        for mtr in ["tir_70_180", "tbr_lt70", "tbr_lt54", "tar_gt250", "night_tbr_lt70", "night_events"]:
            comps.append(paired(arms, a, b, mtr))
    summary["paired_comparisons"] = comps
    summary["bedtime_prediction"] = bed

    def err(col):
        return np.abs(np.log(settings[col] / settings[col.split("_")[0] + "_true"]))
    summary["settings_recovery"] = {
        "cr_median_abs_pct_error_prescribed": float(np.expm1(err("cr_prescribed").median()) * 100),
        "cr_median_abs_pct_error_twin": float(np.expm1(err("cr_twin").median()) * 100),
        "cf_median_abs_pct_error_prescribed": float(np.expm1(err("cf_prescribed").median()) * 100),
        "cf_median_abs_pct_error_twin": float(np.expm1(err("cf_twin").median()) * 100),
        "cr_twin_closer_fraction": float((err("cr_twin") < err("cr_prescribed")).mean()),
        "cf_twin_closer_fraction": float((err("cf_twin") < err("cf_prescribed")).mean()),
        "beta_sleep_vs_true_spearman": float(settings[["theta_beta_sleep", "sleep_sens_true"]].corr("spearman").iloc[0, 1]),
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
    (OUT / "study.json").write_text(json.dumps(summary, indent=2, default=float))
    log_msg("done", summary["runtime_minutes"], "min")


if __name__ == "__main__":
    main()

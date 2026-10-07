"""Export data for the web demo (web/demo_data.json).

Uses held-out cohort seed 2026. For three featured patients (one child, one
adolescent, one adult) it exports the clinic record, the fitted twin, what the
twin learned in plain numbers, one evaluation day under standard care + trend
alerts (arm B) and under DoseTwin (arm D), and one meal decision with the twin
state so the browser can re-simulate any dose live.

Featured meal selection rule (stated in the UI): among evaluation-week meals
where the two arms dosed differently, the one with the largest difference in
true time-below-70 over the next 5 h. It is an illustrative example; the
cohort-level results are shown alongside it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dosetwin.alerts import HybridAlert, TrendAlert  # noqa: E402
from dosetwin.ehr import ehr_table  # noqa: E402
from dosetwin.truth.cohort import StandardCalculator, World  # noqa: E402
from dosetwin.truth.uva_padova import load_params, true_settings  # noqa: E402
from dosetwin.twin import identify as I  # noqa: E402
from dosetwin.twin import model as M  # noqa: E402
from dosetwin.twin.advisor import TwinAdvisor  # noqa: E402
from dosetwin.twin.tracker import TwinTracker  # noqa: E402

D = 1440
SEED = 2026
FEATURED = ["child#004", "adolescent#001", "adult#006"]


def r(x, n=1):
    return [None if (v is None or np.isnan(v)) else round(float(v), n) for v in x]


def main():
    I.EGFR_COEF = json.loads((ROOT / "results" / "study.json").read_text())["frozen_on_dev"]["egfr_coef"]
    names = list(load_params()["Name"])
    w = World(names, days=14, seed=SEED)
    ehr = w.ehr()
    std = StandardCalculator(ehr)
    log0 = w.run(lambda d: std)
    ehr = ehr_table(w, log0, SEED)
    thetas = {}
    for i, n in enumerate(names):
        thetas[n] = I.fit(I.from_log(log0, ehr, i).slice(0, 7 * D), H=300, prior_strength=150).theta
    after7 = lambda al: (lambda i, t, log, c: t >= 7 * D and al(i, t, log, c))
    logB = w.run(lambda d: std, alerter=after7(TrendAlert(30, 50.0)))
    tr = TwinTracker(thetas, ehr, names)
    adv = TwinAdvisor(thetas, ehr, names, tracker=tr)
    logD = w.run(lambda d: std if d < 7 else adv, alerter=after7(HybridAlert(tr, 65.0, ehr)))
    truth = true_settings(FEATURED).set_index("patient")

    out = {"seed": SEED, "patients": []}
    for n in FEATURED:
        i = names.index(n)
        e = ehr.iloc[i]
        p = I.from_log(log0, ehr, i).slice(0, 7 * D)
        th = thetas[n]
        prior = I.ehr_prior(p)
        s = I.implied_settings(th, p)
        # activity effect: 45-min brisk walk at 18:00 from 140 mg/dL, no meal
        z0 = M.steady_state(th, p.basal, 140.0)
        th0 = th.copy(); th0[M.I_GB] = 140.0; th0[M.I_DAWN] = 0.0
        stp = np.zeros(240); stp[0:45] = 110
        walk = M.forecast(th0, z0, np.zeros(240), p.basal, np.zeros(240), stp, 0.0, p.W, 1080, 240)
        rest = M.forecast(th0, z0, np.zeros(240), p.basal, np.zeros(240), np.zeros(240), 0.0, p.W, 1080, 240)
        # meal decisions in evaluation week where arms differ
        best = None
        for (name, t, carbs, dose, sdose) in adv.decisions:
            if name != n or t < 7 * D or t > 13 * D or abs(dose - sdose) < 0.5:
                continue
            bB = logB["bg"][i, t:t + 300]; bD = logD["bg"][i, t:t + 300]
            gap = np.mean(bB < 70) - np.mean(bD < 70)
            score = gap if gap > 0 else -1 + (np.mean((bD >= 70) & (bD <= 180)) - np.mean((bB >= 70) & (bB <= 180)))
            if best is None or score > best[0]:
                best = (score, t, carbs, dose, sdose)
        _, tm, carbs, dose, sdose = best
        # twin state at that meal, reconstructed from arm D's logged data
        trk = TwinTracker(thetas, ehr, names)
        z = trk.state(i, tm, logD)
        day = tm // D
        sl = slice(day * D, (day + 1) * D, 5)

        def events(lg):
            return [{"m": int(ev[0] - day * D), "type": ev[1], "units": round(float(ev[2]), 1),
                     "carbs": round(float(ev[3]), 0)} for ev in lg["events"][i] if day * D <= ev[0] < (day + 1) * D]

        meal_h = slice(tm, tm + 301, 5)
        out["patients"].append({
            "id": n, "label": {"child": "Child", "adolescent": "Teen", "adult": "Adult"}[n.split("#")[0]],
            "ehr": {"age": int(e.age), "weight_kg": float(e.weight_kg), "cr": float(e.carb_ratio_g_per_u),
                    "cf": float(e.correction_factor_mgdl_per_u), "basal_u_per_day": float(e.basal_u_per_day)},
            "twin": {"theta": [float(v) for v in th], "basal_u_per_min": p.basal, "W": p.W,
                     "cr": round(s["cr"], 1), "cf": round(s["cf"], 1),
                     "true_cr": round(float(truth.loc[n, "true_cr"]), 1), "true_cf": round(float(truth.loc[n, "true_cf"]), 1),
                     "carb_effect_vs_record": round(float(np.exp(th[M.I_LFC] - prior[M.I_LFC])), 2),
                     "walk_drop_mgdl": round(float(rest.min() - walk.min()), 0),
                     "dawn_rise_mgdl_per_h": round(float(th[M.I_DAWN] * 60), 0),
                     "carb_absorption_min": round(float(np.exp(th[M.I_LTM])), 0),
                     "insulin_absorption_min": round(float(np.exp(th[M.I_LTI])), 0),
                     "sleep_effect_pct_after_4h_night": round(float((1 - np.exp(-th[M.I_BS] * 3.0)) * 100), 0)},
            "day": {"index": int(day + 1),
                    "cgm_B": r(logB["cgm"][i, sl]), "cgm_D": r(logD["cgm"][i, sl]),
                    "steps": r(np.add.reduceat(logD["act"][i, day * D:(day + 1) * D], np.arange(0, D, 5)), 0),
                    "events_B": events(logB), "events_D": events(logD)},
            "meal": {"minute_of_day": int(tm - day * D), "minute_abs": int(tm), "carbs_logged": float(carbs),
                     "sleep_debt": round(float(logD["sleep_debt"][i, tm]), 2),
                     "standard_dose": float(sdose), "twin_dose": float(dose), "z": [float(v) for v in z],
                     "true_after_B": r(logB["bg"][i, meal_h]), "true_after_D": r(logD["bg"][i, meal_h])},
        })
        print(n, "featured meal day", day + 1, "min", tm - day * D, "carbs", carbs, "std", sdose, "twin", dose, flush=True)
    (ROOT / "web" / "demo_data.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main()

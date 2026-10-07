"""Export data for the clinician dashboard (web/clinic_data.json).

Held-out cohort seed 2026, run-in week (days 1-7, standard care): what a clinic
would have before a review visit. Everything shown to the "doctor" is device
data, the synthetic EHR and the fitted twin; the hidden truth is not exported.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dosetwin.ehr import build_records, ehr_table  # noqa: E402
from dosetwin.metrics import glycemic_metrics, hypo_events  # noqa: E402
from dosetwin.truth.cohort import StandardCalculator, World  # noqa: E402
from dosetwin.truth.uva_padova import load_params  # noqa: E402
from dosetwin.twin import identify as I  # noqa: E402
from dosetwin.twin.tracker import TwinTracker  # noqa: E402

D = 1440
SEED = 2026
DAY = 6  # representative day replayed in the dashboard (day 7 of the run-in)


def main():
    names = list(load_params()["Name"])
    w = World(names, days=14, seed=SEED)
    log = w.run(lambda d: StandardCalculator(w.ehr()))
    ehr = ehr_table(w, log, SEED)
    recs = build_records(w, log, SEED)
    thetas = {n: I.fit(I.from_log(log, ehr, i).slice(0, 7 * D), H=300, prior_strength=150).theta
              for i, n in enumerate(names)}
    tr = TwinTracker(thetas, ehr, names)
    out = []
    for i, n in enumerate(names):
        p = I.from_log(log, ehr, i).slice(0, 7 * D)
        cg = log["cgm"][i, :7 * D]
        cg5 = cg[::5]
        m = glycemic_metrics(cg5)
        night = np.concatenate([cg[d * D:d * D + 7 * 60:5] for d in range(7)])
        mn = glycemic_metrics(night)
        lows = len(hypo_events(np.repeat(cg5, 5)))
        # AGP: percentiles by 30-min bin of the day
        bins = cg5.reshape(7, 288).T.reshape(48, 6 * 7)
        agp = {q: np.round(np.nanpercentile(bins, q, axis=1), 0).tolist() for q in (10, 25, 50, 75, 90)}
        s = I.implied_settings(thetas[n], p)
        t0 = DAY * D
        z0 = tr.state(i, t0, log)
        rescue = {}
        for ev in log["events"][i]:
            if ev[1] in ("hypo_rescue", "preemptive_carbs") and t0 <= ev[0] < t0 + D:
                rescue[ev[0]] = rescue.get(ev[0], 0.0) + ev[3]
        meals = [{"m": int(t - t0), "carbs": float(log["carbs_logged"][i, t] - rescue.get(t, 0.0))}
                 for t in range(t0, t0 + D) if log["carbs_logged"][i, t] - rescue.get(t, 0.0) > 0]
        boluses = [{"m": int(t - t0), "u": float(log["bolus"][i, t])}
                   for t in range(t0, t0 + D) if log["bolus"][i, t] > 0]
        out.append({
            "id": n, "ehr": recs[i],
            "week": {"tir": round(m["tir_70_180"], 1), "tbr": round(m["tbr_lt70"], 2), "tbr54": round(m["tbr_lt54"], 2),
                     "tar": round(m["tar_gt180"], 1), "mean": round(m["mean"]), "cv": round(m["cv"], 1),
                     "gmi": round(m["gmi"], 1), "night_tbr": round(mn["tbr_lt70"], 2), "low_events": lows},
            "agp": agp,
            "twin": {"theta": [float(v) for v in thetas[n]], "basal_u_per_min": p.basal, "W": p.W,
                     "cr": round(s["cr"], 1), "cf": round(s["cf"], 1),
                     "basal_u_per_day": round(s["basal_u_per_day"], 1)},
            "day": {"index": DAY + 1, "z0": [float(v) for v in z0], "meals": meals, "boluses": boluses,
                    "steps": np.add.reduceat(log["steps"][i, t0:t0 + D], np.arange(0, D, 5)).round().tolist(),
                    "cgm": [None if np.isnan(v) else round(float(v)) for v in log["cgm"][i, t0:t0 + D:5]]},
        })
        print(n, out[-1]["week"]["tir"], out[-1]["week"]["tbr"], "twin", s["cr"], s["cf"], round(s["basal_u_per_day"], 1),
              "rx", p.CR, p.CF, round(p.basal * 1440, 1), flush=True)
    (ROOT / "web" / "clinic_data.json").write_text(json.dumps({"seed": SEED, "patients": out}))


if __name__ == "__main__":
    main()

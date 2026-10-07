"""Twin-guided dosing: test candidate doses on the twin before injecting.

At each meal the advisor
  1. brings the patient's twin up to date with everything logged so far
     (CGM via the observer, pen doses, meal log, steps);
  2. simulates the next 5 h for every candidate pen dose (half-unit steps);
  3. scores each trajectory with the Kovatchev glycaemic risk index and rejects
     doses whose pessimistic trajectory (forecast minus a safety margin growing
     12 mg/dL per hour, capped at 25 mg/dL) dips below 70 mg/dL;
  4. returns the lowest-risk safe dose (hard cap: 2x the standard dose, +3 U).

Margin settings were frozen on the development cohort (seed 7) before the
held-out evaluation. The advisor never sees the hidden physiology.
"""
from __future__ import annotations

import numpy as np

from .tracker import TwinTracker

PEN_STEP = 0.5
H = 300  # min


def risk(g: np.ndarray) -> np.ndarray:
    """Kovatchev symmetrised glucose risk (0 at ~112 mg/dL)."""
    f = 1.509 * (np.log(np.clip(g, 20, 600)) ** 1.084 - 5.381)
    return 10 * f ** 2


class TwinAdvisor:
    """Meal-bolus policy driven by per-patient twins."""

    def __init__(self, thetas: dict, ehr, names: list, name="twin", margin_per_hour=12.0,
                 margin_cap=25.0, hypo_floor=70.0, cap_factor=2.0, tracker: TwinTracker | None = None):
        self.name = name
        self.names = names
        self.tracker = tracker or TwinTracker(thetas, ehr, names)
        e = ehr.set_index("patient").loc[names]
        self.CF = e["correction_factor_mgdl_per_u"].to_numpy(float)
        self.CR = e["carb_ratio_g_per_u"].to_numpy(float)
        self.margin_per_hour = margin_per_hour
        self.margin_cap = margin_cap
        self.hypo_floor = hypo_floor
        self.cap_factor = cap_factor
        self.decisions = []

    def standard_dose(self, i, est_carbs, cgm, iob):
        d = est_carbs / self.CR[i] + max(0.0, (cgm - 120.0) / self.CF[i]) - iob
        return max(0.0, round(d / PEN_STEP) * PEN_STEP)

    def evaluate_doses(self, i: int, t: int, est_carbs: float, log: dict, doses=None):
        """(doses, trajectories, mean risk, feasible) - also used by the dashboard export."""
        tr = self.tracker
        z = tr.state(i, t, log)
        std = self.standard_dose(i, est_carbs, z[8], 0.0)
        if doses is None:
            dmax = max(self.cap_factor * max(std, 1.0), std + 3.0)
            doses = np.arange(0.0, dmax + 1e-9, PEN_STEP)
        traj = np.stack([tr.forecast(i, t, log, H, bolus=d, carbs=est_carbs) for d in doses])
        margin = np.minimum(self.margin_per_hour * np.arange(H + 1) / 60.0, self.margin_cap)
        feasible = (traj - margin).min(axis=1) >= self.hypo_floor
        return doses, traj, risk(traj).mean(axis=1), feasible

    def meal_bolus(self, i: int, t: int, est_carbs: float, cgm: float, iob: float, log) -> float:
        doses, traj, r, feasible = self.evaluate_doses(i, t, est_carbs, log)
        k = int(np.argmin(np.where(feasible, r, np.inf))) if feasible.any() else 0
        dose = float(doses[k])
        self.decisions.append((self.names[i], t, est_carbs, dose, self.standard_dose(i, est_carbs, cgm, iob)))
        return dose


class SettingsPolicy:
    """Standard calculator, but with twin-recommended CR/CF (clinician-approved)."""

    def __init__(self, cr: np.ndarray, cf: np.ndarray, name="twin_settings"):
        self.CR, self.CF, self.name = cr, cf, name

    def meal_bolus(self, i, t, est_carbs, cgm, iob, log):
        d = est_carbs / self.CR[i] + max(0.0, (cgm - 120.0) / self.CF[i]) - iob
        return max(0.0, round(d / PEN_STEP) * PEN_STEP)

"""Personalisation: EHR-informed prior  ->  data-driven posterior (MAP).

1. Prior. The clinic record (weight, prescribed carb ratio CR, correction
   factor CF, basal dose, HbA1c, eGFR) is converted into twin parameters by
   matching the twin's simulated response to 1 U of insulin (=CF) and to a meal
   covered by CR, then raising insulin potency for reduced kidney function
   (EGFR_COEF, estimated on the development cohort). This is the "population
   twin": personalised only by the EHR.
2. Posterior. Parameters are re-estimated from the patient's own device data
   (CGM + pen log + meal log + steps/heart rate + sleep) by minimising
   multi-step open-loop prediction error (5 min .. 5 h) plus a prior penalty
   (MAP, robust loss).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from . import model as M

DEFAULT_GAINS = np.array([0.6, 0.5, 0.004])
# prior std-devs in parameter space (log-params ~ fractional)
PRIOR_SD = np.array([0.5, 25.0, 0.4, 0.35, 0.3, 2.5, 0.15, 0.6, 0.04])
LOWER = np.array([np.log(0.02), 70.0, np.log(0.2), np.log(15.0), np.log(25.0), 0.0, 0.0, np.log(1e-3), 0.0])
UPPER = np.array([np.log(50.0), 260.0, np.log(2.5), np.log(120.0), np.log(120.0), 12.0, 0.8, np.log(0.05), 0.25])
# log insulin-potency shift per unit of kidney impairment x = clip((90 - eGFR)/60, 0, 1).
# Estimated on the development cohort by scripts/run_study.py (estimate_egfr_coef)
# and frozen before the held-out cohorts are touched.
EGFR_COEF = 0.0


def kidney_impairment(egfr) -> float:
    if egfr is None or not np.isfinite(egfr):
        return 0.0
    return float(np.clip((90.0 - egfr) / 60.0, 0.0, 1.0))


@dataclass
class PatientData:
    """Device-visible minute-resolution data for one patient."""

    name: str
    u: np.ndarray          # bolus U at minute
    c: np.ndarray          # logged carbs g at minute
    act: np.ndarray        # activity, step-equivalents per minute (steps + heart rate)
    sd: np.ndarray         # sleep debt in force (h), wearable estimate
    cgm: np.ndarray        # mg/dL, NaN when no sample
    basal: float           # U/min (prescribed long-acting, flat)
    W: float               # kg
    CR: float              # prescribed g/U
    CF: float              # prescribed mg/dL/U
    t_start: int = 0       # minute-of-study of index 0
    hba1c: float | None = None  # % (EHR lab), optional
    egfr: float | None = None   # mL/min/1.73m2 (EHR lab), optional
    hypo_history: int = 0       # EHR diagnosis E16.0

    def slice(self, a: int, b: int) -> "PatientData":
        return PatientData(self.name, self.u[a:b], self.c[a:b], self.act[a:b], self.sd[a:b], self.cgm[a:b],
                           self.basal, self.W, self.CR, self.CF, self.t_start + a, self.hba1c,
                           self.egfr, self.hypo_history)


def from_log(log: dict, ehr, i: int, use_ehr_labs: bool = True) -> PatientData:
    """use_ehr_labs=False drops HbA1c, eGFR and the hypo-history diagnosis (ablation)."""
    r = ehr.iloc[i]
    T = log["bolus"].shape[1]
    act = log["act"][i] if "act" in log else log.get("steps", np.zeros((i + 1, T)))[i]
    sd = log["sleep_debt"][i] if "sleep_debt" in log else np.zeros(T)
    lab = (lambda k: float(r[k]) if use_ehr_labs and k in r.index else None)
    return PatientData(
        name=r["patient"], u=log["bolus"][i].copy(), c=log["carbs_logged"][i].copy(),
        act=np.asarray(act, float).copy(), sd=np.asarray(sd, float).copy(), cgm=log["cgm"][i].copy(),
        basal=float(r["basal_u_per_day"]) / 1440.0, W=float(r["weight_kg"]),
        CR=float(r["carb_ratio_g_per_u"]), CF=float(r["correction_factor_mgdl_per_u"]),
        hba1c=lab("hba1c_pct"), egfr=lab("egfr_ml_min"),
        hypo_history=int(lab("severe_hypo_history") or 0))


# --------------------------------------------------------------- responses
def insulin_drop(th, b, W, G0=180.0, H=300) -> float:
    """mg/dL fall at H min after 1 U from steady state (the twin's own CF)."""
    z0 = M.steady_state(th, b, G0)
    u = np.zeros(H); u[0] = 1.0
    zeros = np.zeros(H)
    th2 = th.copy(); th2[M.I_GB] = G0; th2[M.I_DAWN] = 0.0  # isolate insulin effect
    g = M.forecast(th2, z0, u, b, zeros, zeros, 0.0, W, 720, H)
    return float(G0 - g[-1])


def meal_rise(th, b, W, grams=50.0, G0=120.0, H=300, units=0.0) -> float:
    """mg/dL change at H min after a meal (optionally with a bolus)."""
    z0 = M.steady_state(th, b, G0)
    u = np.zeros(H); u[0] = units
    c = np.zeros(H); c[0] = grams
    zeros = np.zeros(H)
    th2 = th.copy(); th2[M.I_GB] = G0; th2[M.I_DAWN] = 0.0
    g = M.forecast(th2, z0, u, b, c, zeros, 0.0, W, 720, H)
    return float(g[-1] - G0)


def _bisect(f, lo, hi, it=40):
    flo = f(lo)
    for _ in range(it):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if (fm > 0) == (flo > 0):
            lo, flo = mid, fm
        else:
            hi = mid
    return 0.5 * (lo + hi)


def ehr_prior(p: PatientData, fasting_glucose: float | None = None, egfr_coef: float | None = None) -> np.ndarray:
    """Population twin personalised only by the clinic record."""
    th = M.default_theta()
    if fasting_glucose is None and p.hba1c is not None:
        # HbA1c lab -> ADAG estimated average glucose; fasting prior sits a little below it
        fasting_glucose = float(np.clip(28.7 * p.hba1c - 46.7 - 10.0, 80.0, 220.0))
    if fasting_glucose is not None:
        th[M.I_GB] = fasting_glucose
    th[M.I_LSI] = _bisect(lambda ls: insulin_drop(_with(th, M.I_LSI, ls), p.basal, p.W) - p.CF,
                          np.log(0.02), np.log(50.0))
    g = 50.0
    th[M.I_LFC] = _bisect(lambda lf: meal_rise(_with(th, M.I_LFC, lf), p.basal, p.W, g, units=g / p.CR),
                          np.log(0.2), np.log(2.5))
    # reduced kidney function -> slower insulin clearance -> each unit acts more strongly
    coef = EGFR_COEF if egfr_coef is None else egfr_coef
    th[M.I_LSI] += coef * kidney_impairment(p.egfr)
    return np.clip(th, LOWER + 1e-6, UPPER - 1e-6)


def _with(th, k, v):
    t = th.copy(); t[k] = v
    return t


# -------------------------------------------------------------------- fit
@dataclass
class FitResult:
    theta: np.ndarray
    prior: np.ndarray
    cost: float
    nfev: int
    rmse_by_h: dict = field(default_factory=dict)


def _windows(p: PatientData, every=30, H=180, warmup=360):
    T = len(p.u)
    return np.arange(warmup, T - H, every, dtype=np.int64)


def residuals(th, p: PatientData, starts, H, gains):
    z0 = M.steady_state(th, p.basal, _first_cgm(p))
    Z = M.sync_run(th, p.u, p.basal, p.c, p.act, p.sd, p.W, p.cgm, p.t_start, gains, z0)
    F = M.window_forecasts(th, Z, p.u, p.basal, p.c, p.act, p.sd, p.W, p.t_start, starts, H, True)
    hs = np.arange(5, H + 1, 5)
    idx = starts[:, None] + hs[None, :]
    y = p.cgm[idx]
    r = F[:, hs] - y
    r = np.where(np.isnan(r), 0.0, r)
    return r.ravel()


def _first_cgm(p):
    v = p.cgm[~np.isnan(p.cgm)]
    return float(v[0]) if len(v) else 120.0


def fit(p: PatientData, prior: np.ndarray | None = None, H=180, gains=DEFAULT_GAINS,
        every=30, max_nfev=300, prior_strength=40.0) -> FitResult:
    prior = ehr_prior(p) if prior is None else prior
    starts = _windows(p, every=every, H=H)
    n_res = len(starts) * (H // 5)
    scale = 1.0 / np.sqrt(max(n_res, 1))
    # weight the prior like ~40 extra windows of evidence (MAP with robust loss)
    prior_w = np.sqrt(prior_strength * (H // 5)) * scale * 15.0

    def fun(th):
        r = residuals(th, p, starts, H, gains) * scale
        pr = (th - prior) / PRIOR_SD * prior_w
        return np.concatenate([r, pr])

    x0 = np.clip(prior, LOWER + 1e-6, UPPER - 1e-6)
    sol = least_squares(fun, x0, bounds=(LOWER, UPPER), loss="soft_l1", f_scale=20.0 * scale,
                        x_scale=PRIOR_SD, max_nfev=max_nfev, diff_step=1e-3)
    return FitResult(theta=sol.x, prior=prior, cost=float(sol.cost), nfev=int(sol.nfev))


def implied_settings(th, p: PatientData, target=110.0) -> dict:
    """Clinically readable settings implied by the twin (for clinician review)."""
    cf = insulin_drop(th, p.basal, p.W)
    g = 50.0
    units = _bisect(lambda x: meal_rise(th, p.basal, p.W, g, units=x), 0.0, 30.0)
    cr = g / max(units, 1e-3)
    SI = np.exp(th[M.I_LSI]); SG = np.exp(th[M.I_LSG]); Gb = th[M.I_GB]
    # basal giving fasting steady state = target (closed form for the twin)
    x_ss = SG * (Gb / target - 1.0)
    basal_new = p.basal + x_ss * M.KE * p.W / SI
    return {"cf": cf, "cr": cr, "basal_u_per_day": max(basal_new, 0.0) * 1440.0,
            "fasting_glucose_at_current_basal": Gb}

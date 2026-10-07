"""Synthetic-but-principled patient lives for the in-silico study.

Each virtual patient (hidden UVA/Padova physiology) lives 14 days of
multiple-daily-injection (MDI) therapy with a flash/real-time CGM, a phone step
counter, and an insulin pen with dose logging. Behaviour is modelled on what is
reported for people with T1D in India and elsewhere:

* four eating occasions, rice/roti-heavy lunch and late dinner (~21:00);
* carbohydrate counting error: a persistent personal bias plus ~30% random
  error per meal (carb-counting studies report 20-50% error);
* missed (10%) and late (12%) meal boluses; unbolused evening snacks;
* "rule of 15" hypo rescue (at night only when the <55 mg/dL alarm wakes
  the patient) and ad-hoc correction boluses for highs;
* evening walks (wearable steps) that transiently raise insulin sensitivity;
* day-to-day insulin-sensitivity variation and a dawn effect;
* a mis-titrated long-acting basal dose (approximated as a flat infusion).

Every random draw is seeded per patient so that two dosing policies can be
compared on *exactly* the same meals, errors, walks and sensor noise.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .uva_padova import CGMNoise, PatientBatch, load_params

MIN_PER_DAY = 1440
CGM_PERIOD = 5  # min
PEN_STEP = 0.5  # U, half-unit pen
TARGET = 120.0  # mg/dL bolus-calculator target


@dataclass
class Behaviour:
    """Per-patient hidden behaviour/physiology modifiers."""

    carb_bias: float          # persistent multiplicative carb-count bias
    basal_factor: float       # prescribed basal / physiological basal
    exercise_gain: float      # insulin-sensitivity gain while active
    dawn: float               # fractional EGP rise 04:00-08:00
    meal_scale: float         # portion size relative to adult
    day_sens: np.ndarray      # day-to-day sensitivity multipliers
    cr_factor: float = 1.0    # prescribed carb ratio / ideal (clinic mis-titration)
    cf_factor: float = 1.0    # prescribed correction factor / ideal


@dataclass
class Scenario:
    """Minute-resolution exogenous inputs for one patient."""

    meals: list               # (minute, true_g, est_g, bolus_mode) bolus_mode: 0 normal,1 missed,2 late
    walks: np.ndarray         # activity intensity 0..1 per minute
    steps: np.ndarray         # observed steps per minute (wearable)


def _age_scale(name: str) -> float:
    return 0.55 if name.startswith("child") else (0.85 if name.startswith("adolescent") else 1.0)


def make_behaviour(name: str, rng: np.random.RandomState, days: int) -> Behaviour:
    return Behaviour(
        carb_bias=rng.uniform(0.65, 1.10),
        basal_factor=rng.uniform(0.80, 1.20),
        exercise_gain=rng.uniform(0.6, 1.6),
        dawn=rng.uniform(0.0, 0.35),
        meal_scale=_age_scale(name),
        day_sens=np.exp(rng.normal(0, 0.18, days)),
        cr_factor=rng.uniform(0.75, 1.30),
        cf_factor=rng.uniform(0.75, 1.30),
    )


def make_scenario(beh: Behaviour, rng: np.random.RandomState, days: int) -> Scenario:
    T = days * MIN_PER_DAY
    meals = []
    for d in range(days):
        base = d * MIN_PER_DAY
        plan = [  # (mean minute of day, sd, carb lo, carb hi, prob, bolus prob)
            (8 * 60, 30, 40, 80, 1.0, 1.0),
            (13 * 60 + 30, 40, 70, 120, 1.0, 1.0),
            (17 * 60 + 30, 30, 20, 40, 0.7, 0.4),
            (21 * 60, 40, 70, 130, 1.0, 1.0),
        ]
        for mu, sd, lo, hi, p, pb in plan:
            if rng.rand() > p:
                continue
            t = int(base + np.clip(rng.normal(mu, sd), mu - 2.5 * sd, mu + 2.5 * sd))
            g = rng.uniform(lo, hi) * beh.meal_scale
            est = g * beh.carb_bias * np.exp(rng.normal(0, 0.3))
            r = rng.rand()
            mode = 1 if (r < 0.10 or rng.rand() > pb) else (2 if r < 0.22 else 0)
            meals.append((t, round(g, 1), round(5 * round(est / 5), 1), mode))
    walks = np.zeros(T)
    for d in range(days):
        if rng.rand() < 0.5:
            s = int(d * MIN_PER_DAY + rng.normal(18.5 * 60, 45))
            dur = int(rng.uniform(30, 60))
            walks[s:s + dur] = rng.uniform(0.6, 1.0)
    # wearable steps: walking ~ 110 steps/min * intensity + daytime background
    minute_of_day = np.arange(T) % MIN_PER_DAY
    awake = (minute_of_day > 7 * 60) & (minute_of_day < 23 * 60)
    background = awake * rng.poisson(6, T)
    steps = (walks * 110 + rng.normal(0, 8, T) * (walks > 0)).clip(0) + background
    return Scenario(meals=meals, walks=walks, steps=steps.round())


class StandardCalculator:
    """Standard-of-care bolus calculator (what pen users are taught).

    bolus = carbs/CR + (CGM - target)/CF - IOB, rounded to half units.
    """

    name = "standard"

    def __init__(self, ehr: pd.DataFrame):
        # uses the *prescribed* settings from the patient's record
        self.CR = ehr["carb_ratio_g_per_u"].to_numpy(float)
        self.CF = ehr["correction_factor_mgdl_per_u"].to_numpy(float)

    def meal_bolus(self, i: int, t: int, est_carbs: float, cgm: float, iob: float, log) -> float:
        dose = est_carbs / self.CR[i] + max(0.0, (cgm - TARGET) / self.CF[i]) - iob
        return max(0.0, round(dose / PEN_STEP) * PEN_STEP)


def insulin_on_board(bolus_log: list, t: int, dia: int = 240) -> float:
    """Linear IOB over a 4 h duration of insulin action (pen-user rule of thumb)."""
    return sum(u * max(0.0, 1 - (t - tb) / dia) for tb, u in bolus_log if t - tb < dia)


@dataclass
class World:
    names: list
    days: int = 14
    seed: int = 7
    sensor: str = "Dexcom"
    params: pd.DataFrame = field(init=False)

    def __post_init__(self):
        self.params = load_params(self.names)
        self.N = len(self.names)
        self.behaviours, self.scenarios = [], []
        for i, n in enumerate(self.names):
            rng = np.random.RandomState(self.seed * 1000 + i)
            b = make_behaviour(n, rng, self.days)
            self.behaviours.append(b)
            self.scenarios.append(make_scenario(b, rng, self.days))
        pb = PatientBatch(self.params)
        self.basal = pb.basal_u_per_min * np.array([b.basal_factor for b in self.behaviours])

    def ehr(self, basal_u_per_min=None) -> pd.DataFrame:
        """What the clinic record holds: demographics + *prescribed* settings."""
        q = self.params
        basal = self.basal if basal_u_per_min is None else basal_u_per_min
        return pd.DataFrame({
            "patient": self.names,
            "age": q["Age"].to_numpy(),
            "weight_kg": q["BW"].round(1).to_numpy(),
            "total_daily_insulin_u": q["TDI"].round(1).to_numpy(),
            "carb_ratio_g_per_u": (q["CR"] * [b.cr_factor for b in self.behaviours]).round(1).to_numpy(),
            "correction_factor_mgdl_per_u": (q["CF"] * [b.cf_factor for b in self.behaviours]).round(1).to_numpy(),
            "basal_u_per_day": (basal * MIN_PER_DAY).round(1),
        })

    def run(self, policy_for_day, rescue=True, alerter=None, alert_response=0.8) -> dict:
        """Simulate all patients. `policy_for_day(day)` returns the meal-bolus policy.

        Returns minute-resolution logs (what the devices saw + hidden truth).
        """
        N, T = self.N, self.days * MIN_PER_DAY
        pb = PatientBatch(self.params)
        noise = CGMNoise(N, self.sensor, seed=self.seed, minutes=T)
        basal = self.basal
        tau_on, tau_off = 20.0, 90.0
        E = np.zeros(N)
        exg = np.array([b.exercise_gain for b in self.behaviours])
        dawn = np.array([b.dawn for b in self.behaviours])
        log = {k: np.zeros((N, T)) for k in ["bg", "cgm", "carbs_true", "carbs_logged", "bolus", "steps"]}
        log["cgm"][:] = np.nan
        boluses = [[] for _ in range(N)]
        events = [[] for _ in range(N)]
        meal_at = [dict() for _ in range(N)]
        late = [dict() for _ in range(N)]
        for i, sc in enumerate(self.scenarios):
            for (tm, g, est, mode) in sc.meals:
                meal_at[i][tm] = (g, est, mode)
            log["steps"][i] = sc.steps
        alert_rng = [np.random.RandomState(self.seed * 7919 + i) for i in range(N)]
        last_rescue = np.full(N, -999)
        last_corr = np.full(N, -999)
        cgm_now = pb.bg.copy()
        mod = np.arange(T) % MIN_PER_DAY
        dawn_shape = np.exp(-0.5 * ((mod - 6 * 60) / 60.0) ** 2)
        for t in range(T):
            day = t // MIN_PER_DAY
            policy = policy_for_day(day)
            if t % CGM_PERIOD == 0:
                cgm_now = noise.measure(pb.bg, t)
                log["cgm"][:, t] = cgm_now
            cho = np.zeros(N)
            bol = np.zeros(N)
            for i in range(N):
                m = meal_at[i].get(t)
                if m is not None:
                    g, est, mode = m
                    cho[i] += g
                    log["carbs_logged"][i, t] += est
                    if mode == 0:
                        iob = insulin_on_board(boluses[i], t)
                        u = policy.meal_bolus(i, t, est, cgm_now[i], iob, log)
                        bol[i] += u
                        events[i].append((t, "meal_bolus", u, est, policy.name))
                    elif mode == 2:
                        late[i][t + 30] = est
                    else:
                        events[i].append((t, "missed_bolus", 0.0, est, policy.name))
                lt = late[i].pop(t, None)
                if lt is not None:
                    iob = insulin_on_board(boluses[i], t)
                    # late bolus: patient still uses the carb estimate, CGM-adjusted
                    u = policy.meal_bolus(i, t, lt, cgm_now[i], iob, log)
                    bol[i] += u
                    events[i].append((t, "late_bolus", u, lt, policy.name))
                if rescue and t % CGM_PERIOD == 0:
                    awake = 7 * 60 <= mod[t] <= 23 * 60
                    thr = 70.0 if awake else 55.0  # asleep: only the low alarm wakes them
                    if cgm_now[i] < thr and t - last_rescue[i] >= 20:
                        cho[i] += 15.0
                        log["carbs_logged"][i, t] += 15.0
                        last_rescue[i] = t
                        events[i].append((t, "hypo_rescue", 0.0, 15.0, policy.name))
                    elif (alerter is not None and t - last_rescue[i] >= 30
                          and alerter(i, t, log, cgm_now[i])):
                        events[i].append((t, "predicted_low_alert", 0.0, 0.0, policy.name))
                        if alert_rng[i].rand() < alert_response:
                            cho[i] += 15.0
                            log["carbs_logged"][i, t] += 15.0
                            last_rescue[i] = t
                            events[i].append((t, "preemptive_carbs", 0.0, 15.0, policy.name))
                    if (awake and cgm_now[i] > 250 and t - last_corr[i] >= 180
                            and insulin_on_board(boluses[i], t) < 0.5):
                        u = max(0.0, round(((cgm_now[i] - TARGET) / policy_cf(policy, i)) / PEN_STEP) * PEN_STEP)
                        bol[i] += u
                        last_corr[i] = t
                        events[i].append((t, "correction", u, 0.0, policy.name))
                if bol[i] > 0:
                    boluses[i].append((t, bol[i]))
            # exercise state (wearable-driven in reality; hidden here)
            a = np.array([sc.walks[t] for sc in self.scenarios])
            tau = np.where(a > E, tau_on, tau_off)
            E = E + (a - E) / tau
            sens = np.array([b.day_sens[day] for b in self.behaviours])
            vm = sens * (1 + exg * E)
            egp = 1 + dawn * dawn_shape[t]
            log["bg"][:, t] = pb.bg
            log["carbs_true"][:, t] = cho
            log["bolus"][:, t] = bol
            pb.step(cho, basal + bol, vm_mult=vm, egp_mult=egp)
        log["basal_u_per_min"] = basal
        log["events"] = events
        return log


def policy_cf(policy, i):
    return policy.CF[i] if hasattr(policy, "CF") else 40.0


def to_dataframes(world: World, log: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Device-visible data at 5-min resolution + events + EHR summary."""
    T = log["bg"].shape[1]
    rows = []
    for i, n in enumerate(world.names):
        t5 = np.arange(0, T, CGM_PERIOD)
        df = pd.DataFrame({
            "patient": n,
            "minute": t5,
            "cgm": log["cgm"][i, t5],
            "bg_true_hidden": log["bg"][i, t5],
            "carbs_logged": np.add.reduceat(log["carbs_logged"][i], t5),
            "bolus_u": np.add.reduceat(log["bolus"][i], t5),
            "steps": np.add.reduceat(log["steps"][i], t5),
        })
        rows.append(df)
    data = pd.concat(rows, ignore_index=True)
    ev = pd.DataFrame(
        [(world.names[i],) + e for i in range(world.N) for e in log["events"][i]],
        columns=["patient", "minute", "event", "units", "carbs_logged", "policy"],
    )
    ehr = world.ehr(log["basal_u_per_min"])
    return data, ev, ehr


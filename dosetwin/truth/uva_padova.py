"""Ground-truth virtual patients: UVA/Padova T1D model (vectorised).

This re-implements the patient equations of the open-source `simglucose`
package (MIT, (c) Jinyu Xie), which implements the UVA/Padova T1DM simulator
(Dalla Man et al., 2007; Kovatchev et al., 2009).  We vectorise it across many
patients and integrate with fixed-step RK4 so that a 14-day, 30-patient cohort
runs in seconds instead of minutes.  `tests/test_truth_parity.py` checks that
this implementation matches the original `simglucose.T1DPatient` trajectory.

Extensions beyond simglucose (all off by default, documented in docs/METHODS.md):
  * `vm_mult`  - multiplier on insulin-dependent utilisation (Vm0, Vmx); used to
                 model day-to-day insulin-sensitivity variation and exercise.
  * `egp_mult` - multiplier on endogenous glucose production kp1 (dawn effect).

IMPORTANT: the digital twin never sees these equations or parameters. They are
the hidden "real patient" that the twin must learn from sensor data alone.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PARAM_DIR = Path(__file__).parent / "params"
EAT_RATE = 5.0  # g/min, as in simglucose

_COLS = [
    "BW", "kmax", "kmin", "b", "d", "kabs", "f", "kp1", "kp2", "kp3", "Fsnc",
    "ke1", "ke2", "k1", "k2", "Vm0", "Vmx", "Km0", "m1", "m2", "m4", "m30",
    "Vi", "p2u", "Ib", "ki", "ka1", "ka2", "kd", "ksc", "Vg", "u2ss",
]


def load_params(names: list[str] | None = None) -> pd.DataFrame:
    p = pd.read_csv(PARAM_DIR / "vpatient_params.csv")
    q = pd.read_csv(PARAM_DIR / "Quest.csv")
    p = p.merge(q, on="Name")
    if names is not None:
        p = p.set_index("Name").loc[names].reset_index()
    return p


@dataclass
class PatientBatch:
    """N patients simulated in lock-step at 1-minute resolution."""

    params: pd.DataFrame
    substeps: int = 2

    def __post_init__(self):
        self.N = len(self.params)
        self.P = {c: self.params[c].to_numpy(dtype=float) for c in _COLS}
        self.names = list(self.params["Name"])
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self):
        x0 = self.params[[f"x0_{i:2d}" for i in range(1, 14)]].to_numpy(float)
        self.x = x0.copy()
        self.t = 0
        self.last_Qsto = self.x[:, 0] + self.x[:, 1]
        self.last_foodtaken = np.zeros(self.N)
        self.is_eating = np.zeros(self.N, bool)
        self.planned = np.zeros(self.N)
        self.last_cho = np.zeros(self.N)

    @property
    def basal_u_per_min(self) -> np.ndarray:
        return self.P["u2ss"] * self.P["BW"] / 6000.0

    @property
    def bg(self) -> np.ndarray:
        """Subcutaneous glucose (mg/dL) - what an ideal CGM would read."""
        return self.x[:, 12] / self.P["Vg"]

    @property
    def plasma_bg(self) -> np.ndarray:
        return self.x[:, 3] / self.P["Vg"]

    # ------------------------------------------------------------- dynamics
    def _f(self, x, cho_mg, ins, vm_mult, egp_mult, Dbar):
        P = self.P
        dx = np.zeros_like(x)
        qsto = x[:, 0] + x[:, 1]
        dx[:, 0] = -P["kmax"] * x[:, 0] + cho_mg
        with np.errstate(divide="ignore", invalid="ignore"):
            aa = 5.0 / (2.0 * Dbar * (1.0 - P["b"]))
            cc = 5.0 / (2.0 * Dbar * P["d"])
            kgut = P["kmin"] + (P["kmax"] - P["kmin"]) / 2.0 * (
                np.tanh(aa * (qsto - P["b"] * Dbar))
                - np.tanh(cc * (qsto - P["d"] * Dbar)) + 2.0)
        kgut = np.where(Dbar > 0, kgut, P["kmax"])
        dx[:, 1] = P["kmax"] * x[:, 0] - x[:, 1] * kgut
        dx[:, 2] = kgut * x[:, 1] - P["kabs"] * x[:, 2]
        Rat = P["f"] * P["kabs"] * x[:, 2] / P["BW"]
        EGPt = P["kp1"] * egp_mult - P["kp2"] * x[:, 3] - P["kp3"] * x[:, 8]
        Et = np.where(x[:, 3] > P["ke2"], P["ke1"] * (x[:, 3] - P["ke2"]), 0.0)
        d3 = np.maximum(EGPt, 0) + Rat - P["Fsnc"] - Et - P["k1"] * x[:, 3] + P["k2"] * x[:, 4]
        dx[:, 3] = (x[:, 3] >= 0) * d3
        Vmt = (P["Vm0"] + P["Vmx"] * x[:, 6]) * vm_mult
        Uidt = Vmt * x[:, 4] / (P["Km0"] + x[:, 4])
        d4 = -Uidt + P["k1"] * x[:, 3] - P["k2"] * x[:, 4]
        dx[:, 4] = (x[:, 4] >= 0) * d4
        d5 = -(P["m2"] + P["m4"]) * x[:, 5] + P["m1"] * x[:, 9] + P["ka1"] * x[:, 10] + P["ka2"] * x[:, 11]
        It = x[:, 5] / P["Vi"]
        dx[:, 5] = (x[:, 5] >= 0) * d5
        dx[:, 6] = -P["p2u"] * x[:, 6] + P["p2u"] * (It - P["Ib"])
        dx[:, 7] = -P["ki"] * (x[:, 7] - It)
        dx[:, 8] = -P["ki"] * (x[:, 8] - x[:, 7])
        d9 = -(P["m1"] + P["m30"]) * x[:, 9] + P["m2"] * x[:, 5]
        dx[:, 9] = (x[:, 9] >= 0) * d9
        d10 = ins - (P["ka1"] + P["kd"]) * x[:, 10]
        dx[:, 10] = (x[:, 10] >= 0) * d10
        d11 = P["kd"] * x[:, 10] - P["ka2"] * x[:, 11]
        dx[:, 11] = (x[:, 11] >= 0) * d11
        d12 = -P["ksc"] * x[:, 12] + P["ksc"] * x[:, 3]
        dx[:, 12] = (x[:, 12] >= 0) * d12
        return dx

    def step(self, cho_announce, insulin_u_per_min, vm_mult=None, egp_mult=None):
        """Advance one minute.

        cho_announce: g of carbohydrate *started* this minute (eaten at 5 g/min)
        insulin_u_per_min: insulin delivered during this minute (U/min)
        """
        N = self.N
        vm_mult = np.ones(N) if vm_mult is None else vm_mult
        egp_mult = np.ones(N) if egp_mult is None else egp_mult
        # meal queue (simglucose semantics)
        self.planned += cho_announce
        to_eat = np.where(self.planned > 0, np.minimum(EAT_RATE, self.planned), 0.0)
        self.planned = np.maximum(self.planned - to_eat, 0.0)
        start = (to_eat > 0) & (self.last_cho <= 0)
        self.last_Qsto = np.where(start, self.x[:, 0] + self.x[:, 1], self.last_Qsto)
        self.last_foodtaken = np.where(start, 0.0, self.last_foodtaken)
        self.is_eating |= start
        self.last_foodtaken = self.last_foodtaken + np.where(self.is_eating, to_eat, 0.0)
        ended = (to_eat <= 0) & (self.last_cho > 0)
        self.is_eating &= ~ended
        self.last_cho = to_eat

        cho_mg = to_eat * 1000.0
        ins = insulin_u_per_min * 6000.0 / self.P["BW"]
        Dbar = self.last_Qsto + self.last_foodtaken * 1000.0
        h = 1.0 / self.substeps
        x = self.x
        for _ in range(self.substeps):
            k1 = self._f(x, cho_mg, ins, vm_mult, egp_mult, Dbar)
            k2 = self._f(x + h / 2 * k1, cho_mg, ins, vm_mult, egp_mult, Dbar)
            k3 = self._f(x + h / 2 * k2, cho_mg, ins, vm_mult, egp_mult, Dbar)
            k4 = self._f(x + h * k3, cho_mg, ins, vm_mult, egp_mult, Dbar)
            x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        self.x = np.maximum(x, 0.0)
        self.t += 1


class CGMNoise:
    """simglucose's CGM error model (Johnson-SU transformed AR(1) at 15 min,
    cubic-interpolated to the sensor sample time). Vectorised across patients."""

    def __init__(self, n_patients: int, sensor: str = "Dexcom", seed: int = 0, minutes: int = 0):
        from scipy.interpolate import CubicSpline

        sp = pd.read_csv(PARAM_DIR / "sensor_params.csv").set_index("Name").loc[sensor]
        self.sample_time = int(sp.sample_time)
        self.lo, self.hi = sp["min"], sp["max"]
        rng = np.random.RandomState(seed)
        n15 = minutes // 15 + 3
        e = np.zeros((n_patients, n15))
        e[:, 0] = rng.randn(n_patients)
        for k in range(1, n15):
            e[:, k] = sp.PACF * (e[:, k - 1] + rng.randn(n_patients))
        eps = sp.xi + sp["lambda"] * np.sinh((e - sp.gamma) / sp.delta)
        t15 = np.arange(n15) * 15
        self.noise = CubicSpline(t15, eps, axis=1)(np.arange(minutes + 1))

    def measure(self, bg: np.ndarray, t: int) -> np.ndarray:
        return np.clip(bg + self.noise[:, t], self.lo, self.hi)


def true_settings(names: list[str], H: int = 300) -> pd.DataFrame:
    """Reference CF/CR measured on the hidden physiology with the SAME
    experiments the twin uses (1 U from fasting steady state; 50 g meal and the
    bolus that returns glucose to baseline at 5 h). Used only for evaluation."""
    out = []
    params = load_params(names)
    for n in names:
        p1 = params[params.Name == n].reset_index(drop=True)

        def run(units, grams):
            pb = PatientBatch(p1)
            b = pb.basal_u_per_min
            g0 = pb.bg[0]
            for t in range(H):
                pb.step(np.array([grams if t == 0 else 0.0]), b + (units if t == 0 else 0.0))
            return g0, pb.bg[0]

        g0, g1 = run(1.0, 0.0)
        cf = g0 - g1
        lo, hi = 0.0, 40.0
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            _, g = run(mid, 50.0)
            if g > g0:
                lo = mid
            else:
                hi = mid
        out.append({"patient": n, "true_cf": cf, "true_cr": 50.0 / (0.5 * (lo + hi)), "fasting_bg": g0})
    return pd.DataFrame(out)

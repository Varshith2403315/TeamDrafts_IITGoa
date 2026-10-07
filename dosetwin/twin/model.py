"""DoseTwin: a personal glucose-insulin digital twin.

Model family (deliberately different from the hidden UVA/Padova "truth"):
an extended Bergman minimal model with two-compartment subcutaneous insulin
and carbohydrate absorption, a wearable-driven activity term, a dawn term and
an observer that keeps the twin synchronised to the CGM.

State  z = [S1, S2, I, X, D1, D2, E, G, Gs, d]
  S1,S2  subcutaneous insulin depots (U)          ti   insulin absorption time
  I      plasma insulin (U-equivalent)             ke   elimination (fixed)
  X      remote insulin action (1/min)             SI   insulin sensitivity
  D1,D2  gut carbohydrate (g)                      tm   carb absorption time
  E      activity state from step counter (0-1)    fc   effective carb factor
  G      plasma glucose (mg/dL)                    Gb   glucose at prescribed basal
  Gs     interstitial / sensor glucose (mg/dL)     SG   glucose effectiveness
  d      unexplained glucose rate (mg/dL/min)      alpha activity effect, dawn
                                                   bs   sensitivity loss per hour
                                                        of sleep debt

Inputs per minute: bolus u (U), basal rate b (U/min), logged carbs c (g),
activity in step-equivalents per minute (steps fused with heart rate, see
dosetwin/fusion.py) and sleep debt sd (h, from the wearable's estimate of the
previous night). Insulin sensitivity in force is SI * exp(-bs * sd).
Time step: 1 minute (explicit Euler; stable for these
time-constants). All heavy loops are numba-compiled.
"""
from __future__ import annotations

import numpy as np
from numba import njit

# parameter vector layout -------------------------------------------------
P_NAMES = ["log_SI", "Gb", "log_fc", "log_tm", "log_ti", "alpha", "dawn", "log_SG", "beta_sleep"]
I_LSI, I_GB, I_LFC, I_LTM, I_LTI, I_ALPHA, I_DAWN, I_LSG, I_BS = range(9)
NP = 9
NZ = 10
KE = 0.138    # 1/min plasma insulin elimination
P2 = 0.02     # 1/min remote insulin action rate
TS = 8.0      # min plasma->interstitial lag
TE = 60.0     # min activity state time-constant
TD = 30.0     # min decay of unexplained rate in open-loop forecasts
VG = 1.6      # dL/kg glucose distribution volume


def default_theta() -> np.ndarray:
    th = np.zeros(NP)
    th[I_LSI] = np.log(1.0)
    th[I_GB] = 140.0
    th[I_LFC] = np.log(0.9)
    th[I_LTM] = np.log(40.0)
    th[I_LTI] = np.log(55.0)
    th[I_ALPHA] = 3.0
    th[I_DAWN] = 0.1
    th[I_LSG] = np.log(0.01)
    th[I_BS] = 0.05
    return th


@njit(cache=True)
def _activity(steps):
    a = (steps - 40.0) / 80.0
    if a < 0.0:
        return 0.0
    if a > 1.0:
        return 1.0
    return a


@njit(cache=True)
def _dawn_shape(minute):
    m = minute % 1440
    return np.exp(-0.5 * ((m - 360.0) / 60.0) ** 2)


@njit(cache=True)
def step(z, th, u, b, c, steps, sd, W, minute, decay_d):
    """Advance the twin state one minute (in place)."""
    SI = np.exp(th[0] - th[8] * sd); Gb = th[1]; fc = np.exp(th[2]); tm = np.exp(th[3])
    ti = np.exp(th[4]); alpha = th[5]; dawn = th[6]; SG = np.exp(th[7])
    S1, S2, I, X, D1, D2, E, G, Gs, d = z[0], z[1], z[2], z[3], z[4], z[5], z[6], z[7], z[8], z[9]
    Ib = b / KE
    Ri = S2 / ti
    Ra = fc * D2 / tm * 1000.0 / (VG * W)
    a = _activity(steps)
    dG = (-SG * (G - Gb) - X * G - alpha * E * G * 1e-3 + Ra
          + dawn * _dawn_shape(minute) + d)
    z[0] = S1 + u + b - S1 / ti
    z[1] = S2 + (S1 - S2) / ti
    z[2] = I + Ri - KE * I
    z[3] = X + (-P2 * X + P2 * SI * (I - Ib) / W)
    z[4] = D1 + c - D1 / tm
    z[5] = D2 + (D1 - D2) / tm
    z[6] = E + (a - E) / TE
    z[7] = max(G + dG, 10.0)
    z[8] = Gs + (G - Gs) / TS
    if decay_d:
        z[9] = d - d / TD


@njit(cache=True)
def steady_state(th, b, G0):
    """State at prescribed basal with no meals, glucose G0."""
    ti = np.exp(th[4])
    z = np.zeros(NZ)
    z[0] = b * ti
    z[1] = b * ti
    z[2] = b / KE
    z[7] = G0
    z[8] = G0
    return z


@njit(cache=True)
def sync_run(th, u, b, c, steps, sd, W, cgm, t_start, gains, z0):
    """Run the twin along recorded inputs, correcting with CGM (observer).

    Returns the full state history Z[T, NZ] (state at the *start* of minute t).
    cgm: NaN where no sample.
    """
    T = u.shape[0]
    Z = np.empty((T, NZ))
    z = z0.copy()
    k1, k2, k3 = gains[0], gains[1], gains[2]
    for t in range(T):
        y = cgm[t]
        if not np.isnan(y):
            e = y - z[8]
            z[8] += k1 * e
            z[7] += k2 * e
            z[9] += k3 * e
            if z[9] > 3.0:
                z[9] = 3.0
            elif z[9] < -3.0:
                z[9] = -3.0
        for j in range(NZ):
            Z[t, j] = z[j]
        step(z, th, u[t], b, c[t], steps[t], sd[t], W, t_start + t, False)
    return Z


@njit(cache=True)
def forecast(th, z0, u, b, c, steps, sd, W, minute0, H):
    """Open-loop forecast of sensor glucose for H minutes from state z0.

    u, c, steps: future inputs of length >= H (zeros if unknown).
    sd: sleep debt in force (h), held constant over the horizon.
    Returns Gs path of length H+1 (index k = k minutes ahead).
    """
    z = z0.copy()
    out = np.empty(H + 1)
    out[0] = z[8]
    for k in range(H):
        step(z, th, u[k], b, c[k], steps[k], sd, W, minute0 + k, True)
        out[k + 1] = z[8]
    return out


@njit(cache=True)
def window_forecasts(th, Z, u, b, c, steps, sd, W, t_start, starts, H, known_future):
    """Open-loop forecasts from many snapshot times.

    known_future=True  -> future boluses, carbs and steps are fed in. Used ONLY
                          for system identification (fitting), never for evaluation.
    known_future=False -> honest forecasting: nothing after the snapshot is known.
    """
    n = starts.shape[0]
    out = np.empty((n, H + 1))
    for i in range(n):
        s = starts[i]
        z = Z[s].copy()
        out[i, 0] = z[8]
        for k in range(H):
            t = s + k
            if known_future:
                step(z, th, u[t], b, c[t], steps[t], sd[t], W, t_start + t, True)
            else:
                step(z, th, 0.0, b, 0.0, 0.0, sd[s], W, t_start + t, True)
            out[i, k + 1] = z[8]
    return out

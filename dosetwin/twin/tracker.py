"""Keeps each patient's twin synchronised with streaming device data."""
from __future__ import annotations

import numpy as np

from . import model as M
from .identify import DEFAULT_GAINS


class TwinTracker:
    def __init__(self, thetas: dict, ehr, names: list, gains=DEFAULT_GAINS):
        self.thetas = thetas
        self.names = names
        e = ehr.set_index("patient").loc[names]
        self.basal = e["basal_u_per_day"].to_numpy(float) / 1440.0
        self.W = e["weight_kg"].to_numpy(float)
        self.gains = gains
        self._z, self._t = {}, {}

    def theta(self, i):
        return self.thetas[self.names[i]]

    def state(self, i: int, t: int, log: dict) -> np.ndarray:
        """Twin state at the start of minute t, using all data before t (+CGM at t)."""
        th = self.theta(i)
        t0 = self._t.get(i)
        if t0 is None or t0 > t:
            t0 = max(0, t - 24 * 60)
            cg = log["cgm"][i, t0:t + 1]
            first = cg[~np.isnan(cg)]
            z = M.steady_state(th, self.basal[i], float(first[0]) if len(first) else 120.0)
        else:
            z = self._z[i].copy()
        if t > t0:
            Z = M.sync_run(th, log["bolus"][i, t0:t], self.basal[i], log["carbs_logged"][i, t0:t],
                           log["steps"][i, t0:t], self.W[i], log["cgm"][i, t0:t], t0, self.gains, z)
            z = Z[-1].copy()
            M.step(z, th, log["bolus"][i, t - 1], self.basal[i], log["carbs_logged"][i, t - 1],
                   log["steps"][i, t - 1], self.W[i], t - 1, False)
        self._z[i], self._t[i] = z.copy(), t
        # apply observer correction for a CGM sample at t (not stored, idempotent)
        y = log["cgm"][i, t]
        if not np.isnan(y):
            e = y - z[8]
            z[8] += self.gains[0] * e
            z[7] += self.gains[1] * e
            z[9] = np.clip(z[9] + self.gains[2] * e, -3.0, 3.0)
        return z

    def forecast(self, i: int, t: int, log: dict, H: int, bolus=0.0, carbs=0.0) -> np.ndarray:
        z = self.state(i, t, log)
        u = np.zeros(H); u[0] = bolus
        c = np.zeros(H); c[0] = carbs
        return M.forecast(self.theta(i), z, u, self.basal[i], c, np.zeros(H), self.W[i], t, H)

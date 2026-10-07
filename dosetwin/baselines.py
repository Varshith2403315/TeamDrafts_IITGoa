"""Data-driven baseline: LightGBM forecaster / hypo classifier.

This is the approach most CGM-prediction projects take. It is trained on the
development cohort (seed 7, all 30 patients) and evaluated on held-out cohorts,
so it gets *more* training data than any single patient's twin.
"""
from __future__ import annotations

import numpy as np

LAGS = np.arange(0, 65, 5)  # minutes back


def _iob(bolus_row, t, dia=240):
    s = max(0, t - dia)
    seg = bolus_row[s:t]
    if seg.size == 0:
        return 0.0
    age = t - np.arange(s, t)
    return float(np.sum(seg * np.clip(1 - age / dia, 0, 1)))


def _cob(carb_row, t, absorb=180):
    s = max(0, t - absorb)
    seg = carb_row[s:t]
    if seg.size == 0:
        return 0.0
    age = t - np.arange(s, t)
    return float(np.sum(seg * np.clip(1 - age / absorb, 0, 1)))


def features(log: dict, ehr, i: int, times: np.ndarray) -> np.ndarray:
    e = ehr.iloc[i]
    cgm, bol, carbs, steps = log["cgm"][i], log["bolus"][i], log["carbs_logged"][i], log["steps"][i]
    rows = []
    for t in times:
        lagv = []
        for L in LAGS:
            k = t - L
            k5 = k - (k % 5)
            lagv.append(cgm[k5] if k5 >= 0 else np.nan)
        lagv = np.array(lagv)
        d = lagv[0] - lagv[1:4]
        mod = t % 1440
        rows.append(np.concatenate([
            lagv, d,
            [_iob(bol, t), _cob(carbs, t), steps[max(0, t - 30):t].sum(), steps[max(0, t - 60):t].sum(),
             np.sin(2 * np.pi * mod / 1440), np.cos(2 * np.pi * mod / 1440),
             e["carb_ratio_g_per_u"], e["correction_factor_mgdl_per_u"], e["weight_kg"], e["basal_u_per_day"],
             e["hba1c_pct"] if "hba1c_pct" in e.index else np.nan],
        ]))
    return np.array(rows)


class GBMForecaster:
    def __init__(self, horizons=(30, 60, 120)):
        self.horizons = horizons
        self.models = {}
        self.clf = None

    def fit(self, log, ehr, times):
        import lightgbm as lgb
        X, Y, L = [], {h: [] for h in self.horizons}, []
        from .metrics import hypo_events
        for i in range(len(ehr)):
            Xi = features(log, ehr, i, times)
            X.append(Xi)
            for h in self.horizons:
                Y[h].append(log["bg"][i, times + h] - Xi[:, 0])
            onsets = np.array(hypo_events(log["bg"][i]))
            lab = np.array([np.any((onsets > t) & (onsets <= t + 60)) if len(onsets) else False for t in times])
            L.append(lab)
        X = np.vstack(X)
        params = dict(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=40,
                      subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=0)
        for h in self.horizons:
            self.models[h] = lgb.LGBMRegressor(**params).fit(X, np.concatenate(Y[h]))
        self.clf = lgb.LGBMClassifier(**params).fit(X, np.concatenate(L).astype(int))
        return self

    def predict(self, log, ehr, i, times):
        X = features(log, ehr, i, times)
        out = {h: X[:, 0] + m.predict(X) for h, m in self.models.items()}
        out["p_hypo60"] = self.clf.predict_proba(X)[:, 1]
        return out

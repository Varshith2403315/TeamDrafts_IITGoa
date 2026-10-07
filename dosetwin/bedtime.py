"""Bedtime check: will this patient go low tonight?

At 23:00 the twin is brought up to date with everything logged so far
(CGM, pen doses, meals, activity, last night's sleep) and simulates the night
to 07:00 with no further food or bolus. Two forecasts are run:

  * nominal     - the patient's fitted twin;
  * pessimistic - the same twin with insulin sensitivity raised by 25%
                  (about one standard deviation of day-to-day variation).

The bedtime risk score is the lowest glucose reached by the pessimistic path.
If it falls below the alert threshold, the twin also sizes a bedtime snack:
the smallest of 10/15/20/25/30 g whose pessimistic night stays above 80 mg/dL.

Evaluation event = true glucose < 70 mg/dL for >= 15 min between 23:00 and
07:00 (international consensus definition), on nights where CGM at 23:00 is
not already below 70. Baselines get the same information:

  * bedtime-glucose rule (the clinical rule of thumb: risk if CGM < 120 at bed);
  * LightGBM night classifier on bedtime CGM and trend, insulin and carbs on
    board, today's activity, sleep debt, the patient's own history of night
    lows and night glucose so far, and the EHR fields.
"""
from __future__ import annotations

import numpy as np

from .baselines import _cob, _iob
from .metrics import hypo_events
from .twin import model as M

D = 1440
BED = 23 * 60
NIGHT = 8 * 60
PESSIMISTIC_SI = np.log(1.25)
SNACKS = (10.0, 15.0, 20.0, 25.0, 30.0)


def night_starts(first_day: int, last_day: int) -> list:
    """Bedtimes (minute index) for nights starting on days first_day..last_day."""
    return [d * D + BED for d in range(first_day, last_day + 1)]


def night_label(bg_row: np.ndarray, t_bed: int) -> tuple[bool, float]:
    """(event tonight?, minutes from bedtime to onset)."""
    seg = bg_row[t_bed:t_bed + NIGHT]
    on = hypo_events(seg)
    return (True, float(on[0])) if on else (False, np.nan)


def twin_night(tracker, i: int, t: int, log: dict, carbs: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    th = tracker.theta(i)
    thp = th.copy()
    thp[M.I_LSI] += PESSIMISTIC_SI
    nom = tracker.forecast(i, t, log, NIGHT, carbs=carbs)
    pes = tracker.forecast(i, t, log, NIGHT, carbs=carbs, theta=thp)
    return nom, pes


def twin_score(tracker, i, t, log) -> dict:
    nom, pes = twin_night(tracker, i, t, log)
    return {"twin_min_nominal": float(nom.min()), "twin_min_pessimistic": float(pes.min()),
            "twin_time_below80": float(np.mean(pes < 80))}


def snack_size(tracker, i, t, log, floor=80.0) -> float:
    for g in SNACKS:
        _, pes = twin_night(tracker, i, t, log, carbs=g)
        if pes.min() >= floor:
            return g
    return SNACKS[-1]


class BedtimeCheck:
    """Callback for World.run: grams of bedtime snack to suggest (0 = none)."""

    def __init__(self, tracker, threshold: float, ehr=None, history_bump=7.0):
        self.tracker, self.threshold = tracker, threshold
        n = len(tracker.names)
        hist = np.zeros(n)
        if ehr is not None and "severe_hypo_history" in ehr.columns:
            hist = ehr.set_index("patient").loc[tracker.names, "severe_hypo_history"].to_numpy(float)
        self.thresholds = threshold + history_bump * hist
        self.decisions = []

    def __call__(self, i, t, log) -> float:
        s = twin_score(self.tracker, i, t, log)["twin_min_pessimistic"]
        if s >= self.thresholds[i]:
            return 0.0
        g = snack_size(self.tracker, i, t, log)
        self.decisions.append((self.tracker.names[i], t, s, g))
        return g


# ----------------------------------------------------------------- features
def night_features(log: dict, ehr, i: int, t: int) -> list:
    e = ehr.iloc[i]
    col = (lambda k: float(e[k]) if k in e.index else np.nan)
    cgm = log["cgm"][i]
    k = t - (t % 5)
    now = cgm[k]
    prev = cgm[k - 15] if k >= 15 else np.nan
    prev30 = cgm[k - 30] if k >= 30 else np.nan
    day0 = (t // D) * D
    act = log["act"][i] if "act" in log else log["steps"][i]
    seen = _cgm_filled(cgm)  # device-visible: past nights judged from CGM, not hidden truth
    past_nights = [night_label(seen, b)[0] for b in range(BED, t - NIGHT + 1, D)]
    nights_cgm = [np.nanmean(cgm[b:b + NIGHT]) for b in range(BED, t - NIGHT + 1, D)]
    return [now, now - prev, now - prev30, _iob(log["bolus"][i], t), _cob(log["carbs_logged"][i], t),
            act[day0:t].sum(), log["sleep_debt"][i, t] if "sleep_debt" in log else 0.0,
            np.mean(past_nights) if past_nights else np.nan,
            np.mean(nights_cgm) if nights_cgm else np.nan,
            col("carb_ratio_g_per_u"), col("correction_factor_mgdl_per_u"), col("weight_kg"),
            col("basal_u_per_day"), col("hba1c_pct"), col("egfr_ml_min"), col("severe_hypo_history")]


FEATURE_NAMES = ["cgm_bed", "trend15", "trend30", "iob", "cob", "activity_today", "sleep_debt",
                 "past_night_low_rate", "past_night_mean_cgm", "cr", "cf", "weight", "basal",
                 "hba1c", "egfr", "severe_hypo_history"]


def _cgm_filled(cgm):
    """Device-visible night history: CGM forward-filled to minutes (no hidden truth)."""
    idx = np.where(~np.isnan(cgm), np.arange(len(cgm)), 0)
    np.maximum.accumulate(idx, out=idx)
    return cgm[idx]


class NightGBM:
    """LightGBM bedtime classifier (optionally with twin forecasts as features)."""

    def __init__(self, with_twin: bool = False):
        self.with_twin = with_twin
        self.clf = None

    def fit(self, X, y):
        import lightgbm as lgb
        self.clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
                                      min_child_samples=15, subsample=0.8, subsample_freq=1,
                                      colsample_bytree=0.8, verbose=-1, random_state=0).fit(X, y)
        return self

    def predict(self, X):
        return self.clf.predict_proba(X)[:, 1]


def auroc(score_high_is_risk: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import rankdata
    y = np.asarray(y, bool)
    if y.all() or (~y).all():
        return np.nan
    r = rankdata(score_high_is_risk)
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

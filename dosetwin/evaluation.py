"""Evaluation helpers: forecast accuracy and event-based alert performance."""
from __future__ import annotations

import numpy as np

from .metrics import hypo_events


def sample_times(t_from: int, t_to: int, every: int = 5) -> np.ndarray:
    return np.arange(t_from, t_to, every)


def scores_for(score_fn, n_patients: int, times: np.ndarray, log: dict) -> np.ndarray:
    return np.array([[score_fn(i, int(t), log) for t in times] for i in range(n_patients)])


def alert_event_metrics(scores: np.ndarray, threshold: float, bg: np.ndarray, times: np.ndarray,
                        window: int = 60, refractory: int = 30) -> dict:
    """Event-level metrics for 'alert if score < threshold'.

    Event  = true glucose < 70 mg/dL for >= 15 min (international consensus).
    Hit    = an alert in the `window` minutes before onset (lead time >= 0).
    False  = alert episode with no event onset within `window` min after it
             (alerts are de-duplicated with a `refractory` period).
    """
    n_events = hits = false = n_alerts = 0
    leads = []
    days = (times[-1] - times[0] + (times[1] - times[0])) / 1440.0
    for i in range(scores.shape[0]):
        onsets = [o for o in hypo_events(bg[i]) if times[0] + window <= o < times[-1]]
        raw = times[scores[i] < threshold]
        # alerts only count if glucose is not already low (otherwise it's a low alarm, not a prediction)
        raw = np.array([t for t in raw if bg[i, t] >= 70.0])
        alerts, last = [], -10 ** 9
        for t in raw:
            if t - last >= refractory:
                alerts.append(t)
                last = t
        alerts = np.array(alerts)
        n_alerts += len(alerts)
        for o in onsets:
            n_events += 1
            prior = alerts[(alerts <= o) & (alerts >= o - window)] if len(alerts) else []
            if len(prior):
                hits += 1
                leads.append(o - prior.min())
        for a in alerts:
            if not any(a <= o <= a + window for o in onsets):
                false += 1
    n_p = scores.shape[0]
    return {
        "events": n_events,
        "sensitivity": hits / max(n_events, 1),
        "median_lead_min": float(np.median(leads)) if leads else 0.0,
        "false_alerts_per_day": false / (n_p * days),
        "alerts_per_day": n_alerts / (n_p * days),
        "precision": (n_alerts - false) / max(n_alerts, 1),
    }


def threshold_for_false_rate(scores, bg, times, target_per_day: float, grid) -> float:
    """Largest threshold whose false-alert rate stays <= target (dev cohort only)."""
    best = grid[0]
    for th in grid:
        if alert_event_metrics(scores, th, bg, times)["false_alerts_per_day"] <= target_per_day:
            best = th
    return float(best)

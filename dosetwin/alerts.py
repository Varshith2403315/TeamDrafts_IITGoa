"""Predictive low-glucose alerts.

* TrendAlert  - what CGM apps do today: least-squares slope of the last 15 min,
                linearly projected `horizon` minutes ahead; alert if < threshold.
* TwinAlert   - DoseTwin: open-loop twin forecast (knows insulin on board,
                carbs on board, recent activity, dawn effect); alert if the
                forecast minimum over the next `horizon` min is < threshold.
Both are evaluated on identical held-out data; thresholds are frozen on the
development cohort.
"""
from __future__ import annotations

import numpy as np

from .twin.tracker import TwinTracker


def trend_projection(cgm_hist: np.ndarray, horizon: int) -> float:
    """cgm_hist: last 15 min of 5-min samples (oldest first, may contain NaN)."""
    y = cgm_hist[~np.isnan(cgm_hist)]
    if len(y) < 2:
        return float(y[-1]) if len(y) else np.nan
    x = np.arange(len(y)) * 5.0
    slope = np.polyfit(x, y, 1)[0]
    return float(y[-1] + slope * horizon)


class TrendAlert:
    name = "cgm_trend"

    def __init__(self, horizon=20, threshold=70.0):
        self.horizon, self.threshold = horizon, threshold

    def score(self, i, t, log) -> float:
        return trend_projection(log["cgm"][i, max(0, t - 15):t + 1], self.horizon)

    def __call__(self, i, t, log, cgm_now):
        return self.score(i, t, log) < self.threshold


class TwinAlert:
    name = "twin"

    def __init__(self, tracker: TwinTracker, horizon=40, threshold=75.0):
        self.tracker, self.horizon, self.threshold = tracker, horizon, threshold

    def score(self, i, t, log) -> float:
        return float(self.tracker.forecast(i, t, log, self.horizon).min())

    def __call__(self, i, t, log, cgm_now):
        return self.score(i, t, log) < self.threshold


class HybridAlert:
    """DoseTwin alert: average of the twin's 40-min forecast minimum and the
    30-min CGM trend projection. The twin contributes physiology (insulin and
    carbs on board, activity); the trend contributes fast-drop sensitivity.
    Threshold frozen on the development cohort at <=2 false alerts/day."""

    name = "dosetwin_hybrid"

    def __init__(self, tracker: TwinTracker, threshold=65.0):
        self.twin = TwinAlert(tracker, horizon=40)
        self.trend = TrendAlert(horizon=30)
        self.threshold = threshold

    def score(self, i, t, log) -> float:
        return 0.5 * (self.twin.score(i, t, log) + self.trend.score(i, t, log))

    def __call__(self, i, t, log, cgm_now):
        return self.score(i, t, log) < self.threshold

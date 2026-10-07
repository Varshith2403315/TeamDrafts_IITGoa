"""Wearable signal fusion: steps + heart rate -> activity; sleep -> sleep debt.

The twin's activity input is expressed in step-equivalents per minute so that
walking (seen by the step counter) and exercise with few steps (cycling, gym,
yoga, badminton; seen only by heart rate) drive the same activity state.

    a_hr   = clip((HR - HR_rest - 25) / 50, 0, 1)
    act    = max(steps, 40 + 80 * a_hr)   if a_hr > 0 else steps

HR_rest is the 10th percentile of the previous 24 h of heart rate, so the
signal is causal (day 0 uses its own data).

Sleep debt (hours) for the day after a night is
    debt = max(0, 7 - hours_slept) + 0.5 * max(0, awakenings - 2)
and is held from wake-up until the next wake-up.
"""
from __future__ import annotations

import numpy as np

D = 1440


def sleep_debt(hours: float, awakenings: float) -> float:
    return max(0.0, 7.0 - hours) + 0.5 * max(0.0, awakenings - 2.0)


def resting_hr(hr: np.ndarray) -> np.ndarray:
    """Per-minute causal resting-HR estimate (10th pct of the previous day)."""
    T = hr.shape[-1]
    out = np.empty_like(hr, dtype=float)
    days = int(np.ceil(T / D))
    for d in range(days):
        src = hr[..., max(0, d - 1) * D:max(1, d) * D]
        out[..., d * D:(d + 1) * D] = np.percentile(src, 10, axis=-1)[..., None]
    return out


def activity_equiv(steps: np.ndarray, hr: np.ndarray) -> np.ndarray:
    a_hr = np.clip((hr - resting_hr(hr) - 25.0) / 50.0, 0.0, 1.0)
    return np.where(a_hr > 0, np.maximum(steps, 40.0 + 80.0 * a_hr), steps)

"""Consensus CGM metrics (Battelino et al., Diabetes Care 2019) and risk indices."""
import numpy as np


def glycemic_metrics(bg: np.ndarray) -> dict:
    bg = np.asarray(bg, float)
    bg = bg[~np.isnan(bg)]
    f = 1.509 * (np.log(np.clip(bg, 1, None)) ** 1.084 - 5.381)
    rl = np.where(f < 0, 10 * f ** 2, 0)
    rh = np.where(f > 0, 10 * f ** 2, 0)
    return {
        "mean": float(bg.mean()),
        "tir_70_180": float(np.mean((bg >= 70) & (bg <= 180)) * 100),
        "tbr_lt70": float(np.mean(bg < 70) * 100),
        "tbr_lt54": float(np.mean(bg < 54) * 100),
        "tar_gt180": float(np.mean(bg > 180) * 100),
        "tar_gt250": float(np.mean(bg > 250) * 100),
        "cv": float(bg.std() / bg.mean() * 100),
        "lbgi": float(rl.mean()),
        "hbgi": float(rh.mean()),
        "gmi": float(3.31 + 0.02392 * bg.mean()),
    }


def hypo_events(bg: np.ndarray, thr=70.0, min_len=15, step=1) -> list:
    """Start indices of episodes with bg<thr lasting >= min_len minutes (consensus definition)."""
    below = bg < thr
    starts, i, n = [], 0, len(bg)
    need = max(1, min_len // step)
    while i < n:
        if below[i]:
            j = i
            while j < n and below[j]:
                j += 1
            if j - i >= need:
                starts.append(i)
            i = j
        else:
            i += 1
    return starts

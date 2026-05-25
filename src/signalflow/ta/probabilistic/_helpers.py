"""Causal helpers for probabilistic features.

Reuses the same convention as ``ta/stat/_causal_helpers.py``: every helper
guarantees length-preserving, causal output (NaN during warmup, then
warmup-invariant values for the rest of the series).
"""
from __future__ import annotations

import numpy as np


def log_returns(close: np.ndarray) -> np.ndarray:
    """First-difference of log close with safe first-bar handling.

    Returns array of len(close); ``r[0] = 0`` (no return on first bar).
    """
    c = close.astype(np.float64)
    c_safe = np.maximum(c, 1e-12)
    log_c = np.log(c_safe)
    return np.diff(log_c, prepend=log_c[0])


def causal_rolling_logvol(returns: np.ndarray, smoother: int) -> np.ndarray:
    """log of rolling realised vol (sqrt of mean(r²) over `smoother` bars).

    First ``smoother-1`` bars are NaN.
    """
    n = len(returns)
    out = np.full(n, np.nan)
    if n < smoother:
        return out
    sq = returns * returns
    csum = np.concatenate([[0.0], np.cumsum(sq)])
    for i in range(smoother - 1, n):
        rv = np.sqrt((csum[i + 1] - csum[i + 1 - smoother]) / smoother)
        out[i] = np.log(max(rv, 1e-12))
    return out

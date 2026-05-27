"""Causal / warmup-invariant helpers shared across cross-disciplinary stat features.

All functions guarantee:
  • Output depends only on past data (no look-ahead).
  • Output for bar T is invariant to where the input series begins, AS LONG AS
    the input contains at least ``window`` bars before T (warmup complete).
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as _swv


def log_returns(close: np.ndarray) -> np.ndarray:
    """Log returns with correct first-bar handling (no extreme outlier).

    Returns array of len(close) with ``r[0] = 0`` (no return on first bar).
    """
    c = close.astype(np.float64)
    c_safe = np.maximum(c, 1e-12)
    log_c = np.log(c_safe)
    return np.diff(log_c, prepend=log_c[0])  # prepend SAME log → first diff = 0


def truncated_ema(arr: np.ndarray, tau: int, window_factor: int = 5) -> np.ndarray:
    """Window-bounded exponential moving average — warmup-invariant.

    Standard ewm_mean is recursive (`s_t = α·x_t + (1−α)·s_{t−1}`) and depends
    on the entire prior history, so the same bar T gives different values
    when the input series starts at different points.

    This implementation uses a fixed-length kernel ``exp(-i/tau)`` over the
    last ``window_factor·tau`` bars (covers ≈ 1 − e^(−5) ≈ 99.3% of total
    EMA weight). After ``window_factor·tau`` bars of warmup, output is
    bit-identical regardless of input start.

    Returns NaN for first ``window-1`` bars (no warmup yet).
    """
    window = int(tau * window_factor)
    n = len(arr)
    if n < window or window < 2:
        return np.full(n, np.nan, dtype=np.float64)
    decay = np.exp(-np.arange(window) / tau)
    kernel = (decay / decay.sum())[::-1]  # most-recent bar gets highest weight (last in window)
    wins = _swv(arr.astype(np.float64), window)
    out_v = wins @ kernel
    return np.concatenate([np.full(window - 1, np.nan, dtype=np.float64), out_v])


def rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """Simple rolling mean — already warmup-invariant via sliding_window_view."""
    n = len(arr)
    if n < window:
        return np.full(n, np.nan, dtype=np.float64)
    wins = _swv(arr.astype(np.float64), window)
    out_v = wins.mean(axis=1)
    return np.concatenate([np.full(window - 1, np.nan, dtype=np.float64), out_v])


def rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    n = len(arr)
    if n < window:
        return np.full(n, np.nan, dtype=np.float64)
    wins = _swv(arr.astype(np.float64), window)
    out_v = wins.std(axis=1)
    return np.concatenate([np.full(window - 1, np.nan, dtype=np.float64), out_v])


def rolling_sum(arr: np.ndarray, window: int) -> np.ndarray:
    n = len(arr)
    if n < window:
        return np.full(n, np.nan, dtype=np.float64)
    wins = _swv(arr.astype(np.float64), window)
    out_v = wins.sum(axis=1)
    return np.concatenate([np.full(window - 1, np.nan, dtype=np.float64), out_v])

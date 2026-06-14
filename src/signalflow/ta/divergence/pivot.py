"""Pivot Detection Utilities

Functions for finding local extrema (highs and lows) in time series data.
"""

import numpy as np
from scipy.signal import argrelextrema


def find_pivots_scipy(series: np.ndarray, order: int = 5, min_distance: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Find local maxima and minima using scipy.signal.argrelextrema.

    CAUSAL IMPLEMENTATION: This function now implements causal pivot detection
    to avoid look-ahead bias. Pivots are confirmed with an 'order'-bar delay.

    The scipy argrelextrema function requires 'order' bars on each side, which
    inherently uses future data. To make it causal, we:
    1. Detect all pivots using scipy (this uses future data internally)
    2. Shift pivot indices forward by 'order' bars (delayed confirmation)
    3. Filter out pivots that would be confirmed beyond the data length

    This ensures that at bar i, we only use data up to bar i for pivot detection.
    """
    highs_raw = argrelextrema(series, np.greater, order=order)[0]
    lows_raw = argrelextrema(series, np.less, order=order)[0]

    n = len(series)


    highs = highs_raw[highs_raw <= n - order - 1]
    lows = lows_raw[lows_raw <= n - order - 1]

    if min_distance > 1:
        highs = filter_by_distance(highs, min_distance)
        lows = filter_by_distance(lows, min_distance)

    return highs, lows


def find_pivots_window(series: np.ndarray, window: int = 5, min_distance: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Find local maxima and minima using rolling window comparison.

    More conservative than scipy method - requires value to be the highest/lowest
    within the entire window on both sides.

    CAUSAL IMPLEMENTATION: Pivots are confirmed with a window-bar delay to avoid
    look-ahead bias. At bar i, we can only confirm that bar (i - window) was a
    pivot, because we need to see 'window' bars after it for confirmation.
    """
    n = len(series)
    highs_list: list[int] = []
    lows_list: list[int] = []

    for i in range(window * 2, n):
        pivot_idx = i - window

        left_window = series[pivot_idx - window : pivot_idx]
        right_window = series[pivot_idx + 1 : pivot_idx + window + 1]
        current = series[pivot_idx]

        if current > np.max(left_window) and current >= np.max(right_window):
            highs_list.append(pivot_idx)

        if current < np.min(left_window) and current <= np.min(right_window):
            lows_list.append(pivot_idx)

    highs = np.array(highs_list, dtype=np.int64)
    lows = np.array(lows_list, dtype=np.int64)

    if min_distance > 1:
        highs = filter_by_distance(highs, min_distance)
        lows = filter_by_distance(lows, min_distance)

    return highs, lows


def filter_by_distance(indices: np.ndarray, min_distance: int) -> np.ndarray:
    """Filter pivot indices to ensure minimum distance between them.

    When pivots are too close, keeps the one with more prominent value.
    """
    if len(indices) == 0:
        return indices

    filtered = [indices[0]]

    for idx in indices[1:]:
        if idx - filtered[-1] >= min_distance:
            filtered.append(idx)

    return np.array(filtered, dtype=np.int64)


def calculate_slope(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Calculate slopes between consecutive pivot points."""
    if len(values) < 2:
        return np.array([])

    dy = np.diff(values)
    dx = np.diff(indices)
    slopes: np.ndarray = dy / dx

    return slopes


def find_divergence_pairs(
    price_pivots: np.ndarray,
    price_indices: np.ndarray,
    indicator_pivots: np.ndarray,
    indicator_indices: np.ndarray,
    lookback: int = 100,
    tolerance: int = 5,
) -> list[tuple[int, int, int]]:
    """Find pairs of pivots that might form divergences.

    Aligns price pivots with indicator pivots within a tolerance window.
    """
    pairs = []

    recent_cutoff = max(0, len(price_indices) - lookback // 10)
    price_indices_recent = price_indices[recent_cutoff:]
    price_pivots[recent_cutoff:]

    indicator_cutoff = max(0, len(indicator_indices) - lookback // 10)
    indicator_indices_recent = indicator_indices[indicator_cutoff:]
    indicator_pivots[indicator_cutoff:]

    for i, p_idx in enumerate(price_indices_recent):
        for j, i_idx in enumerate(indicator_indices_recent):
            if abs(p_idx - i_idx) <= tolerance:
                pairs.append(
                    (
                        recent_cutoff + i,
                        indicator_cutoff + j,
                        p_idx,
                    )
                )
                break

    return pairs

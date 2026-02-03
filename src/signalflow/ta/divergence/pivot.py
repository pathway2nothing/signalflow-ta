"""
Pivot Detection Utilities

Functions for finding local extrema (highs and lows) in time series data.
"""

import numpy as np
from typing import Tuple
from scipy.signal import argrelextrema


def find_pivots_scipy(
    series: np.ndarray,
    order: int = 5,
    min_distance: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find local maxima and minima using scipy.signal.argrelextrema.

    Parameters
    ----------
    series : np.ndarray
        Input time series data
    order : int, default 5
        How many points on each side to use for comparison
    min_distance : int, default 1
        Minimum distance between consecutive pivots

    Returns
    -------
    highs_idx : np.ndarray
        Indices of local maxima
    lows_idx : np.ndarray
        Indices of local minima
    """
    # Find local maxima
    highs = argrelextrema(series, np.greater, order=order)[0]

    # Find local minima
    lows = argrelextrema(series, np.less, order=order)[0]

    # Filter by minimum distance
    if min_distance > 1:
        highs = filter_by_distance(highs, min_distance)
        lows = filter_by_distance(lows, min_distance)

    return highs, lows


def find_pivots_window(
    series: np.ndarray,
    window: int = 5,
    min_distance: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find local maxima and minima using rolling window comparison.

    More conservative than scipy method - requires value to be the highest/lowest
    within the entire window on both sides.

    Parameters
    ----------
    series : np.ndarray
        Input time series data
    window : int, default 5
        Window size for comparison (bars on each side)
    min_distance : int, default 10
        Minimum distance between consecutive pivots

    Returns
    -------
    highs_idx : np.ndarray
        Indices of local maxima
    lows_idx : np.ndarray
        Indices of local minima
    """
    n = len(series)
    highs = []
    lows = []

    for i in range(window, n - window):
        # Get window around current point
        left_window = series[i - window:i]
        right_window = series[i + 1:i + window + 1]
        current = series[i]

        # Check if local maximum
        if current > np.max(left_window) and current > np.max(right_window):
            highs.append(i)

        # Check if local minimum
        if current < np.min(left_window) and current < np.min(right_window):
            lows.append(i)

    highs = np.array(highs, dtype=np.int64)
    lows = np.array(lows, dtype=np.int64)

    # Filter by minimum distance
    if min_distance > 1:
        highs = filter_by_distance(highs, min_distance)
        lows = filter_by_distance(lows, min_distance)

    return highs, lows


def filter_by_distance(indices: np.ndarray, min_distance: int) -> np.ndarray:
    """
    Filter pivot indices to ensure minimum distance between them.

    When pivots are too close, keeps the one with more prominent value.

    Parameters
    ----------
    indices : np.ndarray
        Array of pivot indices
    min_distance : int
        Minimum distance required between pivots

    Returns
    -------
    filtered : np.ndarray
        Filtered pivot indices
    """
    if len(indices) == 0:
        return indices

    filtered = [indices[0]]

    for idx in indices[1:]:
        if idx - filtered[-1] >= min_distance:
            filtered.append(idx)

    return np.array(filtered, dtype=np.int64)


def calculate_slope(
    values: np.ndarray,
    indices: np.ndarray
) -> np.ndarray:
    """
    Calculate slopes between consecutive pivot points.

    Parameters
    ----------
    values : np.ndarray
        Y-values at pivot points
    indices : np.ndarray
        X-indices of pivot points

    Returns
    -------
    slopes : np.ndarray
        Array of slopes between consecutive points
    """
    if len(values) < 2:
        return np.array([])

    dy = np.diff(values)
    dx = np.diff(indices)
    slopes = dy / dx

    return slopes


def find_divergence_pairs(
    price_pivots: np.ndarray,
    price_indices: np.ndarray,
    indicator_pivots: np.ndarray,
    indicator_indices: np.ndarray,
    lookback: int = 100,
    tolerance: int = 5
) -> list:
    """
    Find pairs of pivots that might form divergences.

    Aligns price pivots with indicator pivots within a tolerance window.

    Parameters
    ----------
    price_pivots : np.ndarray
        Price values at pivot points
    price_indices : np.ndarray
        Indices of price pivots
    indicator_pivots : np.ndarray
        Indicator values at pivot points
    indicator_indices : np.ndarray
        Indices of indicator pivots
    lookback : int, default 100
        Maximum lookback period for finding pairs
    tolerance : int, default 5
        Maximum index difference to consider pivots as aligned

    Returns
    -------
    pairs : list of tuple
        List of (price_idx, indicator_idx, index) tuples for aligned pivots
    """
    pairs = []

    # Only look at recent pivots
    recent_cutoff = max(0, len(price_indices) - lookback // 10)
    price_indices_recent = price_indices[recent_cutoff:]
    price_pivots_recent = price_pivots[recent_cutoff:]

    indicator_cutoff = max(0, len(indicator_indices) - lookback // 10)
    indicator_indices_recent = indicator_indices[indicator_cutoff:]
    indicator_pivots_recent = indicator_pivots[indicator_cutoff:]

    # Find aligned pivots
    for i, p_idx in enumerate(price_indices_recent):
        for j, i_idx in enumerate(indicator_indices_recent):
            if abs(p_idx - i_idx) <= tolerance:
                pairs.append((
                    recent_cutoff + i,  # Price pivot index in full array
                    indicator_cutoff + j,  # Indicator pivot index in full array
                    p_idx  # Actual bar index
                ))
                break

    return pairs

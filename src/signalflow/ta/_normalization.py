"""Normalization utilities for technical indicators.

This module provides functions for normalizing technical indicators:
- Bounded indicators: linear scaling to standard ranges
- Unbounded indicators: rolling z-score normalization
"""
import numpy as np


def normalize_bounded(
    values: np.ndarray,
    original_range: tuple[float, float],
    target_range: tuple[float, float] = (-1, 1)
) -> np.ndarray:
    """
    Linearly scale bounded values to target range.

    Args:
        values: Input array
        original_range: (min, max) of original values
        target_range: (min, max) of target values

    Returns:
        Normalized array

    Examples:
        >>> normalize_bounded(rsi, (0, 100), (0, 1))  # RSI to [0,1]
        >>> normalize_bounded(willr, (-100, 0), (0, 1))  # Williams %R to [0,1]
        >>> normalize_bounded(cmo, (-100, 100), (-1, 1))  # CMO to [-1,1]
    """
    orig_min, orig_max = original_range
    target_min, target_max = target_range

    # Linear scaling: (x - orig_min) / (orig_max - orig_min) * (target_max - target_min) + target_min
    normalized = (values - orig_min) / (orig_max - orig_min)
    normalized = normalized * (target_max - target_min) + target_min

    return normalized


def normalize_zscore(
    values: np.ndarray,
    window: int,
    robust: bool = False
) -> np.ndarray:
    """
    Apply rolling z-score normalization to unbounded values.

    Args:
        values: Input array
        window: Rolling window size for statistics
        robust: If True, use median and MAD instead of mean and std

    Returns:
        Z-score normalized array (typically in range ±3)

    Examples:
        >>> normalize_zscore(sma_values, window=60)  # Standard z-score
        >>> normalize_zscore(macd_values, window=90, robust=True)  # Robust z-score

    Notes:
        - Z-scores are unbounded but typically in ±3 range (99.7% of data)
        - Handles NaN values gracefully
        - Returns NaN for insufficient data points
    """
    n = len(values)
    result = np.full(n, np.nan)

    if robust:
        # Robust z-score: (x - median) / (1.4826 * MAD)
        # 1.4826 is the scaling factor to make MAD comparable to std dev
        scale = 1.4826
        for i in range(window - 1, n):
            window_vals = values[i - window + 1:i + 1]
            valid = window_vals[~np.isnan(window_vals)]
            if len(valid) > 1:
                median = np.median(valid)
                mad = np.median(np.abs(valid - median))
                if mad > 1e-10:
                    result[i] = (values[i] - median) / (scale * mad)
    else:
        # Standard z-score: (x - mean) / std
        for i in range(window - 1, n):
            window_vals = values[i - window + 1:i + 1]
            valid = window_vals[~np.isnan(window_vals)]
            if len(valid) > 1:
                mean = np.mean(valid)
                std = np.std(valid, ddof=1)
                if std > 1e-10:
                    result[i] = (values[i] - mean) / std

    return result


def get_norm_window(period: int, multiplier: float = 3.0, minimum: int = 60) -> int:
    """
    Calculate appropriate normalization window based on indicator period.

    Args:
        period: Base indicator period
        multiplier: Multiplier for period (default: 3.0)
        minimum: Minimum window size (default: 60)

    Returns:
        Recommended normalization window

    Examples:
        >>> get_norm_window(14)  # 60 (minimum)
        >>> get_norm_window(20)  # 60 (minimum)
        >>> get_norm_window(50)  # 150 (50 * 3)
        >>> get_norm_window(200)  # 600 (200 * 3)

    Notes:
        - 3x period ensures statistical stability (Central Limit Theorem)
        - Minimum of 60 bars (~1 hour of minute data) for reliable statistics
    """
    return max(int(period * multiplier), minimum)


def normalize_ma_pct(source: np.ndarray, ma: np.ndarray) -> np.ndarray:
    """
    Normalize moving average as percentage difference from source.

    normalized = clip((source - ma) / source, -1, 1)

    Args:
        source: Source price array
        ma: Moving average array

    Returns:
        Normalized array as percentage difference, clipped to [-1, 1]

    Examples:
        >>> source = np.array([100, 100, 100])
        >>> ma = np.array([98, 100, 102])
        >>> normalize_ma_pct(source, ma)
        array([0.02, 0.0, -0.02])  # 2% below, equal, 2% above

    Notes:
        - Positive values: MA below source (bullish)
        - Negative values: MA above source (bearish)
        - Clipped to [-1, 1] range for extreme cases
        - Adds epsilon (1e-10) to avoid division by zero
    """
    result = (source - ma) / (source + 1e-10)
    return np.clip(result, -1, 1)

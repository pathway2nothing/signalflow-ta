"""Normalization utilities for technical indicators.

This module provides functions for normalizing technical indicators:
- Bounded indicators: linear scaling to standard ranges
- Unbounded indicators: rolling z-score normalization
"""

import numpy as np

from signalflow.ta._numba_kernels import normalize_zscore_nb, normalize_zscore_robust_nb


def normalize_bounded(
    values: np.ndarray,
    original_range: tuple[float, float],
    target_range: tuple[float, float] = (-1, 1),
) -> np.ndarray:
    """Linearly scale bounded values to target range."""
    orig_min, orig_max = original_range
    target_min, target_max = target_range

    normalized = (values - orig_min) / (orig_max - orig_min)
    normalized = normalized * (target_max - target_min) + target_min

    return normalized


def normalize_zscore(values: np.ndarray, window: int, robust: bool = False) -> np.ndarray:
    """Apply rolling z-score normalization to unbounded values."""
    if robust:
        result: np.ndarray = normalize_zscore_robust_nb(values.astype(np.float64), window)
        return result
    else:
        result_std: np.ndarray = normalize_zscore_nb(values.astype(np.float64), window)
        return result_std


def get_norm_window(period: int, multiplier: float = 3.0, minimum: int = 60) -> int:
    """Calculate appropriate normalization window based on indicator period."""
    return max(int(period * multiplier), minimum)


def normalize_ma_pct(source: np.ndarray, ma: np.ndarray) -> np.ndarray:
    """Normalize moving average as percentage difference from source.

    normalized = clip((source - ma) / source, -1, 1)
    """
    result = (source - ma) / (source + 1e-10)
    result_clipped: np.ndarray = np.clip(result, -1, 1)
    return result_clipped

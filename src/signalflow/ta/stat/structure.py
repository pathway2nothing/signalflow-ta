"""Market structure statistics - direction changes, reversals, timing."""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import polars as pl
from numba import njit

from signalflow import sf_component
from signalflow.feature.base import Feature


@njit
def _calculate_reverse_points(close: np.ndarray, window_size: int) -> np.ndarray:
    """Calculate number of reverse points (direction changes) in sliding window.

    Args:
        close: Price array.
        window_size: Size of sliding window.

    Returns:
        Array with count of reverse points in each window.
    """
    n = len(close)
    reverse_counts = np.zeros(n, dtype=np.int32)
    tail_idx = 0
    reverse_count = 0

    # First window
    for i in range(2, min(window_size, n)):
        if close[i] > close[i - 1] and close[i - 1] < close[i - 2]:
            reverse_count += 1
        elif close[i] < close[i - 1] and close[i - 1] > close[i - 2]:
            reverse_count += 1
        reverse_counts[i] = reverse_count

    tail_idx = 1

    # Rolling window
    for i in range(window_size, n):
        if close[i] > close[i - 1] and close[i - 1] < close[i - 2]:
            reverse_count += 1
        elif close[i] < close[i - 1] and close[i - 1] > close[i - 2]:
            reverse_count += 1

        if (
            close[tail_idx] > close[tail_idx - 1]
            and close[tail_idx] > close[tail_idx + 1]
        ):
            reverse_count -= 1
        elif (
            close[tail_idx] < close[tail_idx - 1]
            and close[tail_idx] < close[tail_idx + 1]
        ):
            reverse_count -= 1

        tail_idx += 1
        reverse_counts[i] = reverse_count

    return reverse_counts


@dataclass
@sf_component(name="stat/reverse_points")
class ReversePointsStat(Feature):
    """Rolling count of price direction reversals.

    Counts how many times price direction changes within the window.
    High values indicate choppy/ranging markets.
    Low values indicate trending markets.

    Outputs:
    - reverse_points: count of reversals
    - reverse_points_norm: normalized by max possible (window - 2)
    """

    source_col: str = "close"
    window: int = 20

    requires = ["{source_col}"]
    outputs = ["{source_col}_reverse_points_{window}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        source = df[self.source_col].to_numpy().astype(np.float64)

        reverse_counts = _calculate_reverse_points(source, self.window)

        max_possible = self.window - 2
        normalized = (
            reverse_counts / max_possible if max_possible > 0 else reverse_counts
        )

        col_raw = f"{self.source_col}_reverse_points_{self.window}"
        col_norm = f"{self.source_col}_reverse_points_norm_{self.window}"

        return df.with_columns(
            [
                pl.Series(name=col_raw, values=reverse_counts),
                pl.Series(name=col_norm, values=normalized),
            ]
        )

    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "window": 20},
        {"source_col": "close", "window": 60},
        {"source_col": "close", "window": 120},
    ]

    @property
    def warmup(self) -> int:
        return self.window * 10


@njit
def _count_consecutive_zeros(values: np.ndarray) -> np.ndarray:
    """Count consecutive zeros (time since spike).

    Args:
        values: Binary array where non-zero indicates spike.

    Returns:
        Array of counts since last non-zero value.
    """
    n = len(values)
    result = np.zeros(n, dtype=np.int32)

    for i in range(1, n):
        if values[i] != 0:
            result[i] = 0
        else:
            result[i] = result[i - 1] + 1

    return result


@dataclass
@sf_component(name="stat/time_since_spike")
class TimeSinceSpikeStat(Feature):
    """Time (bars) since last spike event.

    Takes a binary spike column and calculates how many bars
    have passed since the last spike (non-zero value).

    Useful for:
    - Mean reversion timing
    - Event decay analysis
    - Volatility clustering studies
    """

    source_col: str = "spike"

    requires = ["{source_col}"]
    outputs = ["time_since_{source_col}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        spike_values = df[self.source_col].to_numpy().astype(np.float64)

        time_since = _count_consecutive_zeros(spike_values)

        col_name = f"time_since_{self.source_col}"

        return df.with_columns(pl.Series(name=col_name, values=time_since))

    # This is a dependent indicator - requires pre-computed spike column.
    # Tested manually or via integration tests.
    test_params: ClassVar[list[dict]] = []

    @property
    def warmup(self) -> int:
        return 100


@dataclass
@sf_component(name="stat/volatility_spike")
class VolatilitySpikeStat(Feature):
    """Volatility spike detection using z-score.

    Calculates z-score of price and detects when it exceeds threshold.

    Outputs:
    - volat_zscore: z-score of price
    - volat_spike: 1 if |zscore| > threshold, 0 otherwise
    """

    source_col: str = "close"
    period: int = 60
    threshold: float = 1.0

    requires = ["{source_col}"]
    outputs = ["{source_col}_volat_spike_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        source = df[self.source_col].to_numpy()
        n = len(source)

        zscore = np.full(n, np.nan)

        for i in range(self.period - 1, n):
            window = source[i - self.period + 1 : i + 1]
            mean = np.mean(window)
            std = np.std(window, ddof=1)
            if std > 1e-10:
                zscore[i] = (source[i] - mean) / std

        spike = (np.abs(zscore) > self.threshold).astype(np.int8)

        col_zscore = f"{self.source_col}_volat_zscore_{self.period}"
        col_spike = f"{self.source_col}_volat_spike_{self.period}"

        return df.with_columns(
            [
                pl.Series(name=col_zscore, values=zscore),
                pl.Series(name=col_spike, values=spike),
            ]
        )

    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 60, "threshold": 1.0},
        {"source_col": "close", "period": 120, "threshold": 2.0},
    ]

    @property
    def warmup(self) -> int:
        return self.period * 10


@dataclass
@sf_component(name="stat/volatility_spike_diff")
class VolatilitySpikeDiffStat(Feature):
    """Volatility spike difference between two periods.

    Compares z-scores from two different periods to detect
    relative volatility changes.
    """

    source_col: str = "close"
    first_period: int = 60
    second_period: int = 240
    threshold: float = 0.5

    requires = ["{source_col}"]
    outputs = ["{source_col}_volat_spike_diff_{first_period}_{second_period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        source = df[self.source_col].to_numpy()
        n = len(source)

        zscore1 = np.full(n, np.nan)
        zscore2 = np.full(n, np.nan)

        for i in range(self.first_period - 1, n):
            window = source[i - self.first_period + 1 : i + 1]
            mean = np.mean(window)
            std = np.std(window, ddof=1)
            if std > 1e-10:
                zscore1[i] = (source[i] - mean) / std

        for i in range(self.second_period - 1, n):
            window = source[i - self.second_period + 1 : i + 1]
            mean = np.mean(window)
            std = np.std(window, ddof=1)
            if std > 1e-10:
                zscore2[i] = (source[i] - mean) / std

        diff_zscore = (zscore1 - zscore2) / (np.abs(zscore1) + 1e-10)
        spike = (diff_zscore > self.threshold).astype(np.int8)

        col_diff = (
            f"{self.source_col}_volat_diff_{self.first_period}_{self.second_period}"
        )
        col_spike = f"{self.source_col}_volat_spike_diff_{self.first_period}_{self.second_period}"

        return df.with_columns(
            [
                pl.Series(name=col_diff, values=diff_zscore),
                pl.Series(name=col_spike, values=spike),
            ]
        )

    test_params: ClassVar[list[dict]] = [
        {
            "source_col": "close",
            "first_period": 60,
            "second_period": 240,
            "threshold": 0.5,
        },
    ]

    @property
    def warmup(self) -> int:
        return max(self.first_period, self.second_period) * 10


@dataclass
@sf_component(name="stat/volume_spike")
class VolumeSpikeStat(Feature):
    """Volume spike detection using z-score.

    Detects when volume significantly exceeds normal levels.
    """

    period: int = 60
    threshold: float = 1.0

    requires = ["volume"]
    outputs = ["volume_spike_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        volume = df["volume"].to_numpy()
        n = len(volume)

        zscore = np.full(n, np.nan)

        for i in range(self.period - 1, n):
            window = volume[i - self.period + 1 : i + 1]
            mean = np.mean(window)
            std = np.std(window, ddof=1)
            if std > 1e-10:
                zscore[i] = (volume[i] - mean) / std

        spike = (zscore > self.threshold).astype(np.int8)

        col_zscore = f"volume_zscore_{self.period}"
        col_spike = f"volume_spike_{self.period}"

        return df.with_columns(
            [
                pl.Series(name=col_zscore, values=zscore),
                pl.Series(name=col_spike, values=spike),
            ]
        )

    test_params: ClassVar[list[dict]] = [
        {"period": 60, "threshold": 1.0},
        {"period": 120, "threshold": 2.0},
    ]

    @property
    def warmup(self) -> int:
        return self.period * 10


@dataclass
@sf_component(name="stat/volume_spike_diff")
class VolumeSpikeDiffStat(Feature):
    """Volume spike difference between two periods.

    Compares volume z-scores from two different periods.
    """

    first_period: int = 60
    second_period: int = 240
    threshold: float = 0.5

    requires = ["volume"]
    outputs = ["volume_spike_diff_{first_period}_{second_period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        volume = df["volume"].to_numpy()
        n = len(volume)

        zscore1 = np.full(n, np.nan)
        zscore2 = np.full(n, np.nan)

        for i in range(self.first_period - 1, n):
            window = volume[i - self.first_period + 1 : i + 1]
            mean = np.mean(window)
            std = np.std(window, ddof=1)
            if std > 1e-10:
                zscore1[i] = (volume[i] - mean) / std

        for i in range(self.second_period - 1, n):
            window = volume[i - self.second_period + 1 : i + 1]
            mean = np.mean(window)
            std = np.std(window, ddof=1)
            if std > 1e-10:
                zscore2[i] = (volume[i] - mean) / std

        diff_zscore = (zscore1 - zscore2) / (np.abs(zscore1) + 1e-10)
        spike = (diff_zscore > self.threshold).astype(np.int8)

        col_diff = f"volume_diff_{self.first_period}_{self.second_period}"
        col_spike = f"volume_spike_diff_{self.first_period}_{self.second_period}"

        return df.with_columns(
            [
                pl.Series(name=col_diff, values=diff_zscore),
                pl.Series(name=col_spike, values=spike),
            ]
        )

    test_params: ClassVar[list[dict]] = [
        {"first_period": 60, "second_period": 240, "threshold": 0.5},
    ]

    @property
    def warmup(self) -> int:
        return max(self.first_period, self.second_period) * 10


@dataclass
@sf_component(name="stat/rolling_min")
class RollingMinStat(Feature):
    """Rolling minimum value.

    Tracks the lowest value in the rolling window.
    """

    source_col: str = "close"
    period: int = 14

    requires = ["{source_col}"]
    outputs = ["{source_col}_min_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        col_name = f"{self.source_col}_min_{self.period}"
        return df.with_columns(
            pl.col(self.source_col).rolling_min(window_size=self.period).alias(col_name)
        )

    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 14},
        {"source_col": "close", "period": 60},
        {"source_col": "low", "period": 20},
    ]

    @property
    def warmup(self) -> int:
        return self.period * 10


@dataclass
@sf_component(name="stat/rolling_max")
class RollingMaxStat(Feature):
    """Rolling maximum value.

    Tracks the highest value in the rolling window.
    """

    source_col: str = "close"
    period: int = 14

    requires = ["{source_col}"]
    outputs = ["{source_col}_max_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        col_name = f"{self.source_col}_max_{self.period}"
        return df.with_columns(
            pl.col(self.source_col).rolling_max(window_size=self.period).alias(col_name)
        )

    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 14},
        {"source_col": "close", "period": 60},
        {"source_col": "high", "period": 20},
    ]

    @property
    def warmup(self) -> int:
        return self.period * 10

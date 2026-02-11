"""Trend regime detection - identify trend states and direction."""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import polars as pl
from numba import njit

from signalflow.core import sf_component
from signalflow.feature.base import Feature


@njit
def _williams_alligator_trend(
    confirmation_length: int,
    lips: np.ndarray,
    teeth: np.ndarray,
    jaws: np.ndarray,
    skip: int,
) -> np.ndarray:
    """Calculate trend using Williams Alligator methodology.

    Args:
        confirmation_length: Periods required to confirm a trend.
        lips: Lips moving average values.
        teeth: Teeth moving average values.
        jaws: Jaws moving average values.
        skip: Initial periods to skip.

    Returns:
        Trend indicators: 1=uptrend, -1=downtrend, 0=sideways.
    """
    prev_trend = -2
    current_trend = -2
    nascent_trend = -2
    nascent_trend_start = -1

    trend_indicators = np.zeros(len(lips), dtype=np.int8)

    for i in range(skip, len(lips)):
        if np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaws[i]):
            continue

        if lips[i] > teeth[i] > jaws[i]:
            current_trend = 1
        elif lips[i] < teeth[i] < jaws[i]:
            current_trend = -1
        else:
            current_trend = 0

        if nascent_trend == -2 and current_trend != 0 and prev_trend == current_trend:
            trend_indicators[i] = current_trend
        elif nascent_trend == -2 and current_trend != 0 and prev_trend != current_trend:
            nascent_trend = current_trend
            nascent_trend_start = i
        elif (
            nascent_trend != -2
            and current_trend == nascent_trend
            and i - nascent_trend_start >= confirmation_length
        ):
            trend_indicators[i] = nascent_trend
            nascent_trend = -2
        elif nascent_trend != -2 and current_trend != nascent_trend:
            nascent_trend = current_trend
            nascent_trend_start = i

        prev_trend = current_trend

    return trend_indicators


@njit
def _distance_to_trend_point(trend_indicator: np.ndarray) -> np.ndarray:
    """Calculate distance from each point to trend start.

    Args:
        trend_indicator: Array of trend states.

    Returns:
        Array of distances since trend started.
    """
    n = len(trend_indicator)
    distance = np.zeros(n, dtype=np.int32)
    current_trend = trend_indicator[0]
    current_trend_start = 0

    for i in range(1, n):
        if trend_indicator[i] != current_trend:
            current_trend = trend_indicator[i]
            current_trend_start = i
            distance[i] = 0
        else:
            distance[i] = i - current_trend_start

    return distance


@dataclass
@sf_component(name="trend/alligator_regime")
class WilliamsAlligatorRegime(Feature):
    """Williams Alligator trend regime detector.

    Uses three smoothed moving averages (lips, teeth, jaws) to identify
    trend direction with confirmation periods.

    Outputs:
    - alligator_regime: 1=uptrend, -1=downtrend, 0=sideways
    - alligator_regime_dist: bars since current trend started

    Interpretation:
    - Lips > Teeth > Jaws: uptrend
    - Lips < Teeth < Jaws: downtrend
    - Otherwise: sideways/consolidation

    Reference: Bill Williams, "Trading Chaos"
    """

    lips_length: int = 5
    teeth_length: int = 8
    jaws_length: int = 13
    lips_shift: int = 3
    teeth_shift: int = 5
    jaws_shift: int = 8
    confirmation_length: int = 2

    requires = ["high", "low"]
    outputs = ["alligator_regime", "alligator_regime_dist"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        n = len(high)

        median_price = (high + low) / 2

        # Calculate SMAs
        lips = np.full(n, np.nan)
        teeth = np.full(n, np.nan)
        jaws = np.full(n, np.nan)

        for i in range(self.lips_length - 1, n):
            lips[i] = np.mean(median_price[i - self.lips_length + 1 : i + 1])
        for i in range(self.teeth_length - 1, n):
            teeth[i] = np.mean(median_price[i - self.teeth_length + 1 : i + 1])
        for i in range(self.jaws_length - 1, n):
            jaws[i] = np.mean(median_price[i - self.jaws_length + 1 : i + 1])

        # Apply shifts
        lips_shifted = np.full(n, np.nan)
        teeth_shifted = np.full(n, np.nan)
        jaws_shifted = np.full(n, np.nan)

        lips_shifted[self.lips_shift :] = lips[: -self.lips_shift]
        teeth_shifted[self.teeth_shift :] = teeth[: -self.teeth_shift]
        jaws_shifted[self.jaws_shift :] = jaws[: -self.jaws_shift]

        skip = max(self.lips_shift, self.teeth_shift, self.jaws_shift)

        regime = _williams_alligator_trend(
            self.confirmation_length,
            lips_shifted,
            teeth_shifted,
            jaws_shifted,
            skip,
        )

        distance = _distance_to_trend_point(regime)

        return df.with_columns(
            [
                pl.Series(name="alligator_regime", values=regime),
                pl.Series(name="alligator_regime_dist", values=distance),
            ]
        )

    test_params: ClassVar[list[dict]] = [
        {
            "lips_length": 5,
            "teeth_length": 8,
            "jaws_length": 13,
            "confirmation_length": 2,
        },
        {
            "lips_length": 10,
            "teeth_length": 20,
            "jaws_length": 30,
            "confirmation_length": 3,
        },
    ]

    @property
    def warmup(self) -> int:
        return (
            max(self.lips_length, self.teeth_length, self.jaws_length)
            + max(self.lips_shift, self.teeth_shift, self.jaws_shift)
        ) * 10


@dataclass
@sf_component(name="trend/two_ma_regime")
class TwoMaRegime(Feature):
    """Two Moving Averages trend regime detector.

    Compares fast and slow moving averages to determine trend direction.

    Outputs:
    - two_ma_regime: 1=uptrend (fast > slow), -1=downtrend, 0=equal
    - two_ma_regime_dist: bars since current trend started

    Reference: Classic crossover system
    """

    source_col: str = "close"
    fast_length: int = 10
    slow_length: int = 50

    requires = ["{source_col}"]
    outputs = ["two_ma_regime_{fast_length}_{slow_length}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        source = df[self.source_col].to_numpy()
        n = len(source)

        fast_ma = np.full(n, np.nan)
        slow_ma = np.full(n, np.nan)

        for i in range(self.fast_length - 1, n):
            fast_ma[i] = np.mean(source[i - self.fast_length + 1 : i + 1])
        for i in range(self.slow_length - 1, n):
            slow_ma[i] = np.mean(source[i - self.slow_length + 1 : i + 1])

        regime = np.where(
            fast_ma > slow_ma, 1, np.where(fast_ma < slow_ma, -1, 0)
        ).astype(np.int8)

        distance = _distance_to_trend_point(regime)

        col_regime = f"two_ma_regime_{self.fast_length}_{self.slow_length}"
        col_dist = f"two_ma_regime_dist_{self.fast_length}_{self.slow_length}"

        return df.with_columns(
            [
                pl.Series(name=col_regime, values=regime),
                pl.Series(name=col_dist, values=distance),
            ]
        )

    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "fast_length": 10, "slow_length": 50},
        {"source_col": "close", "fast_length": 20, "slow_length": 100},
        {"source_col": "close", "fast_length": 50, "slow_length": 200},
    ]

    @property
    def warmup(self) -> int:
        return max(self.fast_length, self.slow_length) * 10


@dataclass
@sf_component(name="trend/sma_direction")
class SmaDirection(Feature):
    """SMA direction indicator.

    Binary indicator showing if SMA is increasing from previous period.

    Outputs:
    - sma_dir: 1 if SMA increasing, 0 otherwise
    - sma: the SMA values

    Reference: Basic trend following
    """

    source_col: str = "close"
    period: int = 14

    requires = ["{source_col}"]
    outputs = ["{source_col}_sma_dir_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        source = df[self.source_col].to_numpy()
        n = len(source)

        sma = np.full(n, np.nan)
        for i in range(self.period - 1, n):
            sma[i] = np.mean(source[i - self.period + 1 : i + 1])

        prev_sma = np.roll(sma, 1)
        prev_sma[0] = np.nan

        direction = (sma > prev_sma).astype(np.int8)
        direction[: self.period] = 0

        col_sma = f"{self.source_col}_sma_{self.period}"
        col_dir = f"{self.source_col}_sma_dir_{self.period}"

        return df.with_columns(
            [
                pl.Series(name=col_sma, values=sma),
                pl.Series(name=col_dir, values=direction),
            ]
        )

    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 14},
        {"source_col": "close", "period": 50},
        {"source_col": "close", "period": 200},
    ]

    @property
    def warmup(self) -> int:
        return self.period * 10


@dataclass
@sf_component(name="trend/sma_diff_direction")
class SmaDiffDirection(Feature):
    """SMA difference direction indicator.

    Binary indicator based on change in normalized difference between two SMAs.

    Outputs normalized SMA difference and direction.
    """

    source_col: str = "close"
    first_period: int = 14
    second_period: int = 50

    requires = ["{source_col}"]
    outputs = ["{source_col}_sma_diff_dir_{first_period}_{second_period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        source = df[self.source_col].to_numpy()
        n = len(source)

        first_sma = np.full(n, np.nan)
        second_sma = np.full(n, np.nan)

        for i in range(self.first_period - 1, n):
            first_sma[i] = np.mean(source[i - self.first_period + 1 : i + 1])
        for i in range(self.second_period - 1, n):
            second_sma[i] = np.mean(source[i - self.second_period + 1 : i + 1])

        sma_diff = (first_sma - second_sma) / (first_sma + 1e-10)
        prev_diff = np.roll(sma_diff, 1)
        prev_diff[0] = np.nan

        direction = ((sma_diff - prev_diff) > 0).astype(np.int8)

        col_diff = (
            f"{self.source_col}_sma_diff_{self.first_period}_{self.second_period}"
        )
        col_dir = (
            f"{self.source_col}_sma_diff_dir_{self.first_period}_{self.second_period}"
        )

        return df.with_columns(
            [
                pl.Series(name=col_diff, values=sma_diff),
                pl.Series(name=col_dir, values=direction),
            ]
        )

    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "first_period": 14, "second_period": 50},
        {"source_col": "close", "first_period": 20, "second_period": 100},
    ]

    @property
    def warmup(self) -> int:
        return max(self.first_period, self.second_period) * 10


@njit
def _rolling_linreg_numba(
    close: np.ndarray, window: int
) -> tuple[np.ndarray, np.ndarray]:
    """Fast rolling linear regression using numba.

    Args:
        close: Price array.
        window: Rolling window size.

    Returns:
        Tuple of (slopes, intercepts).
    """
    n = len(close)
    slopes = np.full(n, np.nan)
    intercepts = np.full(n, np.nan)

    x = np.arange(window, dtype=np.float64)
    mean_x = np.mean(x)
    var_x = np.var(x)

    for i in range(window - 1, n):
        y = close[i - window + 1 : i + 1]
        if np.any(np.isnan(y)):
            continue

        mean_y = np.mean(y)

        cov_xy = 0.0
        for j in range(window):
            cov_xy += (x[j] - mean_x) * (y[j] - mean_y)
        cov_xy /= window

        slope = cov_xy / var_x if var_x != 0 else 0.0
        intercept = mean_y - slope * mean_x

        slopes[i] = slope
        intercepts[i] = intercept

    return slopes, intercepts


@dataclass
@sf_component(name="trend/linreg_direction")
class LinRegDirection(Feature):
    """Linear regression slope direction indicator.

    Binary indicator showing if linear regression slope is positive.

    Outputs:
    - linreg_slope: rolling linear regression slope
    - linreg_dir: 1 if slope > 0, 0 otherwise
    """

    source_col: str = "close"
    period: int = 15

    requires = ["{source_col}"]
    outputs = ["{source_col}_linreg_dir_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        source = df[self.source_col].to_numpy().astype(np.float64)

        slopes, intercepts = _rolling_linreg_numba(source, self.period)

        direction = (slopes > 0).astype(np.int8)

        col_slope = f"{self.source_col}_linreg_slope_{self.period}"
        col_dir = f"{self.source_col}_linreg_dir_{self.period}"

        return df.with_columns(
            [
                pl.Series(name=col_slope, values=slopes),
                pl.Series(name=col_dir, values=direction),
            ]
        )

    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 15},
        {"source_col": "close", "period": 30},
        {"source_col": "close", "period": 60},
    ]

    @property
    def warmup(self) -> int:
        return self.period * 10


@dataclass
@sf_component(name="trend/linreg_diff_direction")
class LinRegDiffDirection(Feature):
    """Linear regression difference direction indicator.

    Binary indicator based on difference between two linear regression predictions.
    """

    source_col: str = "close"
    first_period: int = 14
    second_period: int = 60

    requires = ["{source_col}"]
    outputs = ["{source_col}_linreg_diff_dir_{first_period}_{second_period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        source = df[self.source_col].to_numpy().astype(np.float64)

        first_slopes, first_intercepts = _rolling_linreg_numba(
            source, self.first_period
        )
        second_slopes, second_intercepts = _rolling_linreg_numba(
            source, self.second_period
        )

        first_linreg = first_slopes * source + first_intercepts
        second_linreg = second_slopes * source + second_intercepts

        diff = first_linreg - second_linreg
        direction = (diff > 0).astype(np.int8)

        col_diff = (
            f"{self.source_col}_linreg_diff_{self.first_period}_{self.second_period}"
        )
        col_dir = f"{self.source_col}_linreg_diff_dir_{self.first_period}_{self.second_period}"

        return df.with_columns(
            [
                pl.Series(name=col_diff, values=diff),
                pl.Series(name=col_dir, values=direction),
            ]
        )

    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "first_period": 14, "second_period": 60},
        {"source_col": "close", "first_period": 20, "second_period": 100},
    ]

    @property
    def warmup(self) -> int:
        return max(self.first_period, self.second_period) * 10


@dataclass
@sf_component(name="trend/linreg_price_diff")
class LinRegPriceDiff(Feature):
    """Linear regression price difference indicator.

    Binary indicator based on whether close is above or below regression line.
    """

    source_col: str = "close"
    period: int = 15

    requires = ["{source_col}"]
    outputs = ["{source_col}_linreg_price_dir_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        source = df[self.source_col].to_numpy().astype(np.float64)

        slopes, intercepts = _rolling_linreg_numba(source, self.period)

        linreg = slopes * source + intercepts
        diff = source - linreg
        direction = (diff > 0).astype(np.int8)

        col_diff = f"{self.source_col}_linreg_price_diff_{self.period}"
        col_dir = f"{self.source_col}_linreg_price_dir_{self.period}"

        return df.with_columns(
            [
                pl.Series(name=col_diff, values=diff),
                pl.Series(name=col_dir, values=direction),
            ]
        )

    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 15},
        {"source_col": "close", "period": 30},
    ]

    @property
    def warmup(self) -> int:
        return self.period * 10

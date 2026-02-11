"""Divergence-based signal detectors."""

from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
import polars as pl

from signalflow import sf_component
from signalflow.core import Signals, SignalType
from signalflow.detector import SignalDetector
from signalflow.ta.momentum import RsiMom, MacdMom
from signalflow.ta.signals.filters import SignalFilter


def _find_local_extrema(values: np.ndarray, window: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Find local minima and maxima indices.

    Args:
        values: Input array.
        window: Lookback window for extrema detection.

    Returns:
        Tuple of (minima_mask, maxima_mask) boolean arrays.
    """
    n = len(values)
    minima = np.zeros(n, dtype=bool)
    maxima = np.zeros(n, dtype=bool)

    for i in range(window, n - window):
        if np.isnan(values[i]):
            continue

        left_window = values[i - window : i]
        right_window = values[i + 1 : i + window + 1]

        valid_left = left_window[~np.isnan(left_window)]
        valid_right = right_window[~np.isnan(right_window)]

        if len(valid_left) == 0 or len(valid_right) == 0:
            continue

        # Local minimum
        if values[i] <= np.min(valid_left) and values[i] <= np.min(valid_right):
            minima[i] = True

        # Local maximum
        if values[i] >= np.max(valid_left) and values[i] >= np.max(valid_right):
            maxima[i] = True

    return minima, maxima


def _detect_divergence(
    price: np.ndarray,
    indicator: np.ndarray,
    lookback: int = 50,
    min_distance: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect bullish and bearish divergences.

    Bullish divergence: price makes lower low, indicator makes higher low
    Bearish divergence: price makes higher high, indicator makes lower high

    Args:
        price: Price array.
        indicator: Indicator array (RSI, MACD, etc.).
        lookback: Maximum bars to look back for previous extrema.
        min_distance: Minimum bars between extrema.

    Returns:
        Tuple of (bullish_divergence, bearish_divergence) boolean arrays.
    """
    n = len(price)
    bullish = np.zeros(n, dtype=bool)
    bearish = np.zeros(n, dtype=bool)

    price_minima, price_maxima = _find_local_extrema(price)
    ind_minima, ind_maxima = _find_local_extrema(indicator)

    # Find bullish divergence (at price lows)
    for i in range(lookback, n):
        if not price_minima[i]:
            continue

        # Look for previous price low
        for j in range(i - min_distance, max(i - lookback, 0), -1):
            if not price_minima[j]:
                continue

            # Check for divergence: price lower low, indicator higher low
            if price[i] < price[j] and indicator[i] > indicator[j]:
                bullish[i] = True
                break

    # Find bearish divergence (at price highs)
    for i in range(lookback, n):
        if not price_maxima[i]:
            continue

        # Look for previous price high
        for j in range(i - min_distance, max(i - lookback, 0), -1):
            if not price_maxima[j]:
                continue

            # Check for divergence: price higher high, indicator lower high
            if price[i] > price[j] and indicator[i] < indicator[j]:
                bearish[i] = True
                break

    return bullish, bearish


@dataclass
@sf_component(name="ta/divergence_1")
class DivergenceDetector1(SignalDetector):
    """RSI divergence detector.

    Detects bullish and bearish divergences between price and RSI.

    Signal logic:
        - LONG: Bullish divergence (price lower low, RSI higher low)
        - SHORT: Bearish divergence (price higher high, RSI lower high)

    Attributes:
        rsi_period: RSI calculation period.
        lookback: Maximum bars to look back for divergence.
        min_distance: Minimum bars between extrema.
        extrema_window: Window for local extrema detection.
        direction: Signal direction.
        filters: List of SignalFilter instances.

    Example:
        ```python
        from signalflow.ta.signals import DivergenceDetector1

        detector = DivergenceDetector1(
            rsi_period=14,
            lookback=50,
            direction="long"
        )
        signals = detector.run(raw_data_view)
        ```
    """

    rsi_period: int = 14
    lookback: int = 50
    min_distance: int = 5
    extrema_window: int = 5
    direction: str = "long"
    filters: list[SignalFilter] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.direction not in ("long", "short", "both"):
            raise ValueError(
                f"direction must be 'long', 'short', or 'both', got {self.direction}"
            )

        self.rsi_col = f"rsi_{self.rsi_period}"
        self.features = [RsiMom(period=self.rsi_period)]

    def detect(
        self, features: pl.DataFrame, context: dict[str, Any] | None = None
    ) -> Signals:
        """Generate signals based on RSI divergence.

        Args:
            features: DataFrame with computed RSI values.
            context: Optional context dictionary.

        Returns:
            Signals container with detected divergence signals.
        """
        close = features["close"].to_numpy()
        rsi = features[self.rsi_col].to_numpy()
        n = len(close)

        # Detect divergences
        bullish, bearish = _detect_divergence(
            close, rsi, lookback=self.lookback, min_distance=self.min_distance
        )

        # Build signal type array
        signal_type = np.full(n, SignalType.NONE.value)

        if self.direction in ("long", "both"):
            signal_type = np.where(bullish, SignalType.RISE.value, signal_type)

        if self.direction in ("short", "both"):
            signal_type = np.where(bearish, SignalType.FALL.value, signal_type)

        out = features.select([
            self.pair_col,
            self.ts_col,
            pl.Series(name="signal_type", values=signal_type),
            pl.Series(name="signal", values=rsi),
        ])

        # Apply filters
        if self.filters:
            combined_mask = np.ones(len(out), dtype=bool)
            for flt in self.filters:
                filter_mask = flt.apply(features).to_numpy()
                combined_mask = combined_mask & filter_mask

            out = out.with_columns(
                pl.when(pl.Series(values=combined_mask))
                .then(pl.col("signal_type"))
                .otherwise(pl.lit(SignalType.NONE.value))
                .alias("signal_type")
            )

        out = out.filter(pl.col("signal_type") != SignalType.NONE.value)

        return Signals(out)

    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable output."""
        base_warmup = self.rsi_period * 10 + self.lookback
        filter_warmup = max((f.warmup for f in self.filters), default=0)
        return max(base_warmup, filter_warmup)

    test_params: ClassVar[list[dict]] = [
        {"rsi_period": 14, "lookback": 50, "direction": "long"},
        {"rsi_period": 14, "lookback": 50, "direction": "both"},
    ]


@dataclass
@sf_component(name="ta/divergence_2")
class DivergenceDetector2(SignalDetector):
    """MACD divergence detector.

    Detects bullish and bearish divergences between price and MACD histogram.

    Signal logic:
        - LONG: Bullish divergence (price lower low, MACD histogram higher low)
        - SHORT: Bearish divergence (price higher high, MACD histogram lower high)

    Attributes:
        macd_fast: MACD fast period.
        macd_slow: MACD slow period.
        macd_signal: MACD signal period.
        lookback: Maximum bars to look back for divergence.
        min_distance: Minimum bars between extrema.
        direction: Signal direction.
        filters: List of SignalFilter instances.
    """

    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    lookback: int = 50
    min_distance: int = 5
    direction: str = "long"
    filters: list[SignalFilter] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.direction not in ("long", "short", "both"):
            raise ValueError(
                f"direction must be 'long', 'short', or 'both', got {self.direction}"
            )

        self.macd_hist_col = f"macd_hist_{self.macd_fast}_{self.macd_slow}"
        self.features = [
            MacdMom(fast=self.macd_fast, slow=self.macd_slow, signal=self.macd_signal)
        ]

    def detect(
        self, features: pl.DataFrame, context: dict[str, Any] | None = None
    ) -> Signals:
        """Generate signals based on MACD divergence."""
        close = features["close"].to_numpy()
        macd_hist = features[self.macd_hist_col].to_numpy()
        n = len(close)

        # Detect divergences
        bullish, bearish = _detect_divergence(
            close, macd_hist, lookback=self.lookback, min_distance=self.min_distance
        )

        # Build signal type array
        signal_type = np.full(n, SignalType.NONE.value)

        if self.direction in ("long", "both"):
            signal_type = np.where(bullish, SignalType.RISE.value, signal_type)

        if self.direction in ("short", "both"):
            signal_type = np.where(bearish, SignalType.FALL.value, signal_type)

        out = features.select([
            self.pair_col,
            self.ts_col,
            pl.Series(name="signal_type", values=signal_type),
            pl.Series(name="signal", values=macd_hist),
        ])

        # Apply filters
        if self.filters:
            combined_mask = np.ones(len(out), dtype=bool)
            for flt in self.filters:
                filter_mask = flt.apply(features).to_numpy()
                combined_mask = combined_mask & filter_mask

            out = out.with_columns(
                pl.when(pl.Series(values=combined_mask))
                .then(pl.col("signal_type"))
                .otherwise(pl.lit(SignalType.NONE.value))
                .alias("signal_type")
            )

        out = out.filter(pl.col("signal_type") != SignalType.NONE.value)

        return Signals(out)

    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable output."""
        base_warmup = self.macd_slow * 5 + self.lookback
        filter_warmup = max((f.warmup for f in self.filters), default=0)
        return max(base_warmup, filter_warmup)

    test_params: ClassVar[list[dict]] = [
        {"macd_fast": 12, "macd_slow": 26, "lookback": 50, "direction": "long"},
    ]

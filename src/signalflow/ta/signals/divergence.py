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


def _find_local_extrema(
    values: np.ndarray, window: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """Find local minima and maxima indices (causal, no look-ahead).

    An extremum at bar ``i`` requires ``window`` bars on each side to confirm.
    The signal is placed at bar ``i + window`` (the confirmation bar) to avoid
    look-ahead bias.

    Args:
        values: Input array.
        window: Number of bars on each side to confirm an extremum.

    Returns:
        Tuple of (minima_mask, maxima_mask) boolean arrays, shifted to
        confirmation time (no look-ahead).
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

        # Extremum confirmed at bar i+window (when right_window is fully known)
        confirm_idx = i + window

        if values[i] <= np.min(valid_left) and values[i] <= np.min(valid_right):
            minima[confirm_idx] = True

        if values[i] >= np.max(valid_left) and values[i] >= np.max(valid_right):
            maxima[confirm_idx] = True

    return minima, maxima


def _detect_divergence(
    price: np.ndarray,
    indicator: np.ndarray,
    lookback: int = 50,
    min_distance: int = 5,
    extrema_window: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect bullish and bearish divergences (causal, no look-ahead).

    Bullish divergence: price makes lower low, indicator makes higher low
    Bearish divergence: price makes higher high, indicator makes lower high

    Args:
        price: Price array.
        indicator: Indicator array (RSI, MACD, etc.).
        lookback: Maximum bars to look back for previous extrema.
        min_distance: Minimum bars between extrema.
        extrema_window: Window for local extrema detection.

    Returns:
        Tuple of (bullish_divergence, bearish_divergence) boolean arrays.
    """
    n = len(price)
    bullish = np.zeros(n, dtype=bool)
    bearish = np.zeros(n, dtype=bool)

    price_minima, price_maxima = _find_local_extrema(price, window=extrema_window)
    ind_minima, ind_maxima = _find_local_extrema(indicator, window=extrema_window)

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


def _build_signals_df(
    features: pl.DataFrame,
    signal_type: np.ndarray,
    signal_values: np.ndarray,
    filters: list[SignalFilter],
    pair_col: str,
    ts_col: str,
) -> Signals:
    """Build Signals from arrays, apply filters, remove NONE rows."""
    out = features.select(
        [
            pair_col,
            ts_col,
            pl.Series(name="signal_type", values=signal_type),
            pl.Series(name="signal", values=signal_values),
        ]
    )

    if filters:
        combined_mask = np.ones(len(out), dtype=bool)
        for flt in filters:
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


def _detect_multi_pair(
    detector: "SignalDetector",
    features: pl.DataFrame,
    context: dict[str, Any] | None,
) -> Signals:
    """Dispatch _detect_single per pair and concat results."""
    pairs = features[detector.pair_col].unique().sort().to_list()
    if len(pairs) > 1:
        results = []
        for pair in pairs:
            pair_df = features.filter(pl.col(detector.pair_col) == pair)
            sig = detector._detect_single(pair_df, context)
            if len(sig.value) > 0:
                results.append(sig.value)
        if results:
            return Signals(pl.concat(results))
        return Signals(
            features.head(0).select(
                [
                    detector.pair_col,
                    detector.ts_col,
                    pl.lit(0).alias("signal_type"),
                    pl.lit(0.0).alias("signal"),
                ]
            )
        )
    return detector._detect_single(features, context)


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
        return _detect_multi_pair(self, features, context)

    def _detect_single(
        self, features: pl.DataFrame, context: dict[str, Any] | None = None
    ) -> Signals:
        close = features["close"].to_numpy()
        rsi = features[self.rsi_col].to_numpy()
        n = len(close)

        bullish, bearish = _detect_divergence(
            close, rsi,
            lookback=self.lookback,
            min_distance=self.min_distance,
            extrema_window=self.extrema_window,
        )

        signal_type = np.full(n, SignalType.NONE.value)
        if self.direction in ("long", "both"):
            signal_type = np.where(bullish, SignalType.RISE.value, signal_type)
        if self.direction in ("short", "both"):
            signal_type = np.where(bearish, SignalType.FALL.value, signal_type)

        return _build_signals_df(
            features, signal_type, rsi, self.filters, self.pair_col, self.ts_col
        )

    @property
    def warmup(self) -> int:
        base_warmup = self.rsi_period * 10 + self.lookback + self.extrema_window
        filter_warmup = max((f.warmup for f in self.filters), default=0)
        return max(base_warmup, filter_warmup)

    test_params: ClassVar[list[dict]] = [
        {"rsi_period": 14, "lookback": 50, "direction": "long"},
        {"rsi_period": 14, "lookback": 50, "direction": "both"},
    ]


@dataclass
@sf_component(name="ta/divergence_2")
class DivergenceDetector2(SignalDetector):
    """RSI divergence detector with offset subsampling.

    Subsamples every ``offset``-th bar before running extrema / divergence
    detection, then maps signals back to the original timestamps.  This lets
    you keep *1 h-quality* divergence detection while working on 1 m data
    (``offset=60``).

    Signal logic:
        - LONG: Bullish divergence (price lower low, RSI higher low)
        - SHORT: Bearish divergence (price higher high, RSI lower high)

    Attributes:
        rsi_period: RSI calculation period (applied on full data, subsampled for detection).
        lookback: Maximum subsampled bars to look back for divergence.
        min_distance: Minimum subsampled bars between extrema.
        extrema_window: Window for local extrema detection (subsampled).
        offset: Take every N-th bar for analysis. Must be >= 2.
        direction: Signal direction.
        filters: List of SignalFilter instances.

    Example:
        ```python
        from signalflow.ta.signals import DivergenceDetector2

        # 1h-quality divergences on 1m data
        detector = DivergenceDetector2(
            rsi_period=14,
            lookback=50,
            offset=60,
            direction="both",
        )
        signals = detector.run(raw_data_view)
        ```
    """

    rsi_period: int = 14
    lookback: int = 50
    min_distance: int = 5
    extrema_window: int = 5
    offset: int = 60
    direction: str = "long"
    filters: list[SignalFilter] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.direction not in ("long", "short", "both"):
            raise ValueError(
                f"direction must be 'long', 'short', or 'both', got {self.direction}"
            )
        if self.offset < 2:
            raise ValueError(f"offset must be >= 2, got {self.offset}")

        self.rsi_col = f"rsi_{self.rsi_period}"
        self.features = [RsiMom(period=self.rsi_period)]

    def detect(
        self, features: pl.DataFrame, context: dict[str, Any] | None = None
    ) -> Signals:
        return _detect_multi_pair(self, features, context)

    def _detect_single(
        self, features: pl.DataFrame, context: dict[str, Any] | None = None
    ) -> Signals:
        close = features["close"].to_numpy()
        rsi = features[self.rsi_col].to_numpy()
        n = len(close)

        # Subsample every N-th bar
        sub_idx = np.arange(0, n, self.offset)
        close_sub = close[sub_idx]
        rsi_sub = rsi[sub_idx]

        bull_sub, bear_sub = _detect_divergence(
            close_sub, rsi_sub,
            lookback=self.lookback,
            min_distance=self.min_distance,
            extrema_window=self.extrema_window,
        )

        # Map back to original indices
        bullish = np.zeros(n, dtype=bool)
        bearish = np.zeros(n, dtype=bool)
        bullish[sub_idx[bull_sub]] = True
        bearish[sub_idx[bear_sub]] = True

        signal_type = np.full(n, SignalType.NONE.value)
        if self.direction in ("long", "both"):
            signal_type = np.where(bullish, SignalType.RISE.value, signal_type)
        if self.direction in ("short", "both"):
            signal_type = np.where(bearish, SignalType.FALL.value, signal_type)

        return _build_signals_df(
            features, signal_type, rsi, self.filters, self.pair_col, self.ts_col
        )

    @property
    def warmup(self) -> int:
        base_warmup = (self.rsi_period * 10 + self.lookback + self.extrema_window) * self.offset
        filter_warmup = max((f.warmup for f in self.filters), default=0)
        return max(base_warmup, filter_warmup)

    test_params: ClassVar[list[dict]] = [
        {"rsi_period": 14, "lookback": 50, "offset": 60, "direction": "both"},
    ]


@dataclass
@sf_component(name="ta/divergence_3")
class DivergenceDetector3(SignalDetector):
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
    extrema_window: int = 5
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
        return _detect_multi_pair(self, features, context)

    def _detect_single(
        self, features: pl.DataFrame, context: dict[str, Any] | None = None
    ) -> Signals:
        close = features["close"].to_numpy()
        macd_hist = features[self.macd_hist_col].to_numpy()
        n = len(close)

        bullish, bearish = _detect_divergence(
            close, macd_hist,
            lookback=self.lookback,
            min_distance=self.min_distance,
            extrema_window=self.extrema_window,
        )

        signal_type = np.full(n, SignalType.NONE.value)
        if self.direction in ("long", "both"):
            signal_type = np.where(bullish, SignalType.RISE.value, signal_type)
        if self.direction in ("short", "both"):
            signal_type = np.where(bearish, SignalType.FALL.value, signal_type)

        return _build_signals_df(
            features, signal_type, macd_hist, self.filters, self.pair_col, self.ts_col
        )

    @property
    def warmup(self) -> int:
        base_warmup = self.macd_slow * 5 + self.lookback
        filter_warmup = max((f.warmup for f in self.filters), default=0)
        return max(base_warmup, filter_warmup)

    test_params: ClassVar[list[dict]] = [
        {"macd_fast": 12, "macd_slow": 26, "lookback": 50, "direction": "long"},
    ]

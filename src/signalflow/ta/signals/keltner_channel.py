"""Keltner Channel-based signal detectors."""

from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
import polars as pl

from signalflow import sf_component
from signalflow.core import Signals, SignalType
from signalflow.detector import SignalDetector
from signalflow.ta.volatility import KeltnerVol
from signalflow.ta.momentum import RsiMom, MacdMom
from signalflow.ta.signals.filters import SignalFilter


def _rma_sma_init(values: np.ndarray, period: int) -> np.ndarray:
    """RMA with SMA initialization."""
    n = len(values)
    alpha = 1 / period
    rma = np.full(n, np.nan)

    if n < period:
        return rma

    rma[period - 1] = np.mean(values[:period])

    for i in range(period, n):
        rma[i] = alpha * values[i] + (1 - alpha) * rma[i - 1]

    return rma


@dataclass
@sf_component(name="ta/keltner_channel_1")
class KeltnerChannelDetector1(SignalDetector):
    """Keltner Channel detector with RSI z-score condition.

    Generates signals when:
    - Price is below Keltner Channel lower band (oversold)
    - RSI z-score confirms oversold condition

    Signal logic:
        - LONG: close < KC lower AND RSI z-score < threshold
        - SHORT: close > KC upper AND RSI z-score > threshold

    Attributes:
        kc_period: Keltner Channel period.
        kc_multiplier: KC ATR multiplier (scalar).
        rsi_period: RSI calculation period.
        rsi_zscore_window: Window for RSI z-score normalization.
        rsi_zscore_threshold: Z-score threshold for signal.
        use_rsi_condition: Whether to require RSI condition.
        direction: Signal direction - "long", "short", or "both".
        filters: List of SignalFilter instances to apply.

    Example:
        ```python
        from signalflow.ta.signals import KeltnerChannelDetector1

        detector = KeltnerChannelDetector1(
            kc_period=720,
            kc_multiplier=2.0,
            rsi_period=720,
            rsi_zscore_threshold=-1.0,
            direction="long"
        )
        signals = detector.run(raw_data_view)
        ```
    """

    kc_period: int = 720
    kc_multiplier: float = 2.0
    rsi_period: int = 720
    rsi_zscore_window: int = 720
    rsi_zscore_threshold: float = -1.0
    use_rsi_condition: bool = True
    direction: str = "long"
    filters: list[SignalFilter] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.direction not in ("long", "short", "both"):
            raise ValueError(
                f"direction must be 'long', 'short', or 'both', got {self.direction}"
            )

        self.kc_lower_col = f"kc_lower_{self.kc_period}"
        self.kc_upper_col = f"kc_upper_{self.kc_period}"
        self.rsi_col = f"rsi_{self.rsi_period}"

        self.features = [
            KeltnerVol(period=self.kc_period, multiplier=self.kc_multiplier),
            RsiMom(period=self.rsi_period),
        ]

    def detect(
        self, features: pl.DataFrame, context: dict[str, Any] | None = None
    ) -> Signals:
        """Generate signals based on Keltner Channel and RSI conditions.

        Args:
            features: DataFrame with computed KC and RSI values.
            context: Optional context dictionary.

        Returns:
            Signals container with detected signals.
        """
        close = features["close"].to_numpy()
        kc_lower = features[self.kc_lower_col].to_numpy()
        kc_upper = features[self.kc_upper_col].to_numpy()
        rsi = features[self.rsi_col].to_numpy()
        n = len(close)

        # Compute RSI z-score
        rsi_zscore = np.full(n, np.nan)
        for i in range(self.rsi_zscore_window - 1, n):
            window_vals = rsi[i - self.rsi_zscore_window + 1 : i + 1]
            valid = window_vals[~np.isnan(window_vals)]
            if len(valid) > 1:
                mean = np.mean(valid)
                std = np.std(valid, ddof=1)
                if std > 1e-10:
                    rsi_zscore[i] = (rsi[i] - mean) / std

        # KC signal conditions
        below_lower = close < kc_lower
        above_upper = close > kc_upper

        # RSI z-score conditions
        rsi_oversold = rsi_zscore < self.rsi_zscore_threshold
        rsi_overbought = rsi_zscore > -self.rsi_zscore_threshold

        # Build signal type array
        signal_type = np.full(n, SignalType.NONE.value)

        if self.direction in ("long", "both"):
            if self.use_rsi_condition:
                long_signal = below_lower & rsi_oversold
            else:
                long_signal = below_lower
            signal_type = np.where(long_signal, SignalType.RISE.value, signal_type)

        if self.direction in ("short", "both"):
            if self.use_rsi_condition:
                short_signal = above_upper & rsi_overbought
            else:
                short_signal = above_upper
            signal_type = np.where(short_signal, SignalType.FALL.value, signal_type)

        # Use KC difference as signal strength
        kc_diff = kc_lower - close

        out = features.select(
            [
                self.pair_col,
                self.ts_col,
                pl.Series(name="signal_type", values=signal_type),
                pl.Series(name="signal", values=kc_diff),
            ]
        )

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
        base_warmup = max(
            self.kc_period * 5,
            self.rsi_period * 10,
            self.rsi_zscore_window,
        )
        filter_warmup = max((f.warmup for f in self.filters), default=0)
        return max(base_warmup, filter_warmup)

    test_params: ClassVar[list[dict]] = [
        {"kc_period": 20, "rsi_period": 14, "direction": "long"},
        {"kc_period": 720, "rsi_period": 720, "direction": "long"},
    ]


@dataclass
@sf_component(name="ta/keltner_channel_2")
class KeltnerChannelDetector2(SignalDetector):
    """Keltner Channel detector with MACD and RSI conditions.

    Generates signals when:
    - Price is below Keltner Channel lower band
    - MACD is below signal line (bearish momentum)
    - RSI z-score confirms oversold condition

    Signal logic:
        - LONG: close < KC lower AND MACD < signal AND RSI z-score < threshold

    Attributes:
        kc_period: Keltner Channel period.
        kc_multiplier: KC ATR multiplier.
        rsi_period: RSI calculation period.
        rsi_zscore_window: Window for RSI z-score.
        rsi_zscore_threshold: Z-score threshold.
        macd_fast: MACD fast period.
        macd_slow: MACD slow period.
        macd_signal: MACD signal period.
        use_macd_condition: Whether to require MACD condition.
        direction: Signal direction.
        filters: List of SignalFilter instances.

    Example:
        ```python
        from signalflow.ta.signals import KeltnerChannelDetector2

        detector = KeltnerChannelDetector2(
            kc_period=720,
            kc_multiplier=2.0,
            macd_fast=12,
            macd_slow=26,
            macd_signal=9,
            direction="long"
        )
        signals = detector.run(raw_data_view)
        ```
    """

    kc_period: int = 720
    kc_multiplier: float = 2.0
    rsi_period: int = 720
    rsi_zscore_window: int = 720
    rsi_zscore_threshold: float = -1.0
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    use_macd_condition: bool = True
    direction: str = "long"
    filters: list[SignalFilter] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.direction not in ("long", "short", "both"):
            raise ValueError(
                f"direction must be 'long', 'short', or 'both', got {self.direction}"
            )

        self.kc_lower_col = f"kc_lower_{self.kc_period}"
        self.kc_upper_col = f"kc_upper_{self.kc_period}"
        self.rsi_col = f"rsi_{self.rsi_period}"
        self.macd_col = f"macd_{self.macd_fast}_{self.macd_slow}"
        self.macd_signal_col = f"macd_signal_{self.macd_signal}"

        self.features = [
            KeltnerVol(period=self.kc_period, multiplier=self.kc_multiplier),
            RsiMom(period=self.rsi_period),
            MacdMom(fast=self.macd_fast, slow=self.macd_slow, signal=self.macd_signal),
        ]

    def detect(
        self, features: pl.DataFrame, context: dict[str, Any] | None = None
    ) -> Signals:
        """Generate signals based on KC, MACD, and RSI conditions.

        Args:
            features: DataFrame with computed values.
            context: Optional context dictionary.

        Returns:
            Signals container with detected signals.
        """
        close = features["close"].to_numpy()
        kc_lower = features[self.kc_lower_col].to_numpy()
        kc_upper = features[self.kc_upper_col].to_numpy()
        rsi = features[self.rsi_col].to_numpy()
        macd = features[self.macd_col].to_numpy()
        macd_signal = features[self.macd_signal_col].to_numpy()
        n = len(close)

        # Compute RSI z-score
        rsi_zscore = np.full(n, np.nan)
        for i in range(self.rsi_zscore_window - 1, n):
            window_vals = rsi[i - self.rsi_zscore_window + 1 : i + 1]
            valid = window_vals[~np.isnan(window_vals)]
            if len(valid) > 1:
                mean = np.mean(valid)
                std = np.std(valid, ddof=1)
                if std > 1e-10:
                    rsi_zscore[i] = (rsi[i] - mean) / std

        # KC signal conditions
        below_lower = close < kc_lower
        above_upper = close > kc_upper

        # MACD conditions
        macd_bearish = macd < macd_signal
        macd_bullish = macd > macd_signal

        # RSI z-score conditions
        rsi_oversold = rsi_zscore < self.rsi_zscore_threshold
        rsi_overbought = rsi_zscore > -self.rsi_zscore_threshold

        # Build signal type array
        signal_type = np.full(n, SignalType.NONE.value)

        if self.direction in ("long", "both"):
            if self.use_macd_condition:
                long_signal = below_lower & macd_bearish & rsi_oversold
            else:
                long_signal = below_lower & rsi_oversold
            signal_type = np.where(long_signal, SignalType.RISE.value, signal_type)

        if self.direction in ("short", "both"):
            if self.use_macd_condition:
                short_signal = above_upper & macd_bullish & rsi_overbought
            else:
                short_signal = above_upper & rsi_overbought
            signal_type = np.where(short_signal, SignalType.FALL.value, signal_type)

        kc_diff = kc_lower - close

        out = features.select(
            [
                self.pair_col,
                self.ts_col,
                pl.Series(name="signal_type", values=signal_type),
                pl.Series(name="signal", values=kc_diff),
            ]
        )

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
        base_warmup = max(
            self.kc_period * 5,
            self.rsi_period * 10,
            self.rsi_zscore_window,
            self.macd_slow * 5,
        )
        filter_warmup = max((f.warmup for f in self.filters), default=0)
        return max(base_warmup, filter_warmup)

    test_params: ClassVar[list[dict]] = [
        {"kc_period": 20, "rsi_period": 14, "direction": "long"},
        {"kc_period": 720, "rsi_period": 720, "direction": "long"},
    ]

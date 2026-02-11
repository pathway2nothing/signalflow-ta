"""ADX-based regime and trend signal detectors."""

from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
import polars as pl

from signalflow import sf_component
from signalflow.core import Signals, SignalType
from signalflow.detector import SignalDetector
from signalflow.ta.trend import AdxTrend
from signalflow.ta.momentum import RsiMom
from signalflow.ta.signals.filters import SignalFilter


@dataclass
@sf_component(name="ta/adx_regime_1")
class AdxRegimeDetector1(SignalDetector):
    """ADX trend regime detector with DI crossover.

    Uses ADX to confirm trend strength and DI crossover for direction.

    Signal logic:
        - LONG: +DI crosses above -DI AND ADX > threshold (strong uptrend)
        - SHORT: -DI crosses above +DI AND ADX > threshold (strong downtrend)

    Attributes:
        adx_period: ADX calculation period.
        adx_threshold: Minimum ADX for trend confirmation (default 25).
        direction: Signal direction - "long", "short", or "both".
        filters: List of SignalFilter instances.

    Example:
        ```python
        from signalflow.ta.signals import AdxRegimeDetector1

        detector = AdxRegimeDetector1(
            adx_period=14,
            adx_threshold=25,
            direction="long"
        )
        signals = detector.run(raw_data_view)
        ```
    """

    adx_period: int = 14
    adx_threshold: float = 25.0
    direction: str = "long"
    filters: list[SignalFilter] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.direction not in ("long", "short", "both"):
            raise ValueError(
                f"direction must be 'long', 'short', or 'both', got {self.direction}"
            )

        self.adx_col = f"adx_{self.adx_period}"
        self.plus_di_col = f"plus_di_{self.adx_period}"
        self.minus_di_col = f"minus_di_{self.adx_period}"
        self.features = [AdxTrend(period=self.adx_period)]

    def detect(
        self, features: pl.DataFrame, context: dict[str, Any] | None = None
    ) -> Signals:
        """Generate signals based on ADX regime and DI crossover.

        Args:
            features: DataFrame with computed ADX values.
            context: Optional context dictionary.

        Returns:
            Signals container with detected trend signals.
        """
        adx = features[self.adx_col].to_numpy()
        plus_di = features[self.plus_di_col].to_numpy()
        minus_di = features[self.minus_di_col].to_numpy()
        n = len(adx)

        # Previous values for crossover detection
        plus_di_prev = np.roll(plus_di, 1)
        minus_di_prev = np.roll(minus_di, 1)
        plus_di_prev[0] = np.nan
        minus_di_prev[0] = np.nan

        # Crossover conditions
        plus_crosses_above = (plus_di_prev <= minus_di_prev) & (plus_di > minus_di)
        minus_crosses_above = (minus_di_prev <= plus_di_prev) & (minus_di > plus_di)

        # Trend strength condition
        strong_trend = adx > self.adx_threshold

        # Build signal type array
        signal_type = np.full(n, SignalType.NONE.value)

        if self.direction in ("long", "both"):
            long_signal = plus_crosses_above & strong_trend
            signal_type = np.where(long_signal, SignalType.RISE.value, signal_type)

        if self.direction in ("short", "both"):
            short_signal = minus_crosses_above & strong_trend
            signal_type = np.where(short_signal, SignalType.FALL.value, signal_type)

        out = features.select([
            self.pair_col,
            self.ts_col,
            pl.Series(name="signal_type", values=signal_type),
            pl.Series(name="signal", values=adx),
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
        base_warmup = self.adx_period * 10
        filter_warmup = max((f.warmup for f in self.filters), default=0)
        return max(base_warmup, filter_warmup)

    test_params: ClassVar[list[dict]] = [
        {"adx_period": 14, "adx_threshold": 25, "direction": "long"},
        {"adx_period": 14, "adx_threshold": 25, "direction": "both"},
    ]


@dataclass
@sf_component(name="ta/adx_regime_2")
class AdxRegimeDetector2(SignalDetector):
    """ADX regime detector combining trend/range with RSI.

    Uses ADX to determine market regime:
    - ADX > threshold: trending market -> follow momentum
    - ADX < threshold: ranging market -> mean reversion

    Signal logic:
        - In TREND regime (ADX > threshold):
          - LONG: +DI > -DI (uptrend direction)
        - In RANGE regime (ADX < threshold):
          - LONG: RSI < oversold (mean reversion)

    Attributes:
        adx_period: ADX calculation period.
        adx_trend_threshold: ADX threshold for trend regime.
        adx_range_threshold: ADX threshold for range regime.
        rsi_period: RSI calculation period.
        rsi_oversold: RSI oversold threshold.
        rsi_overbought: RSI overbought threshold.
        direction: Signal direction.
        filters: List of SignalFilter instances.
    """

    adx_period: int = 14
    adx_trend_threshold: float = 25.0
    adx_range_threshold: float = 20.0
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    direction: str = "long"
    filters: list[SignalFilter] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.direction not in ("long", "short", "both"):
            raise ValueError(
                f"direction must be 'long', 'short', or 'both', got {self.direction}"
            )

        self.adx_col = f"adx_{self.adx_period}"
        self.plus_di_col = f"plus_di_{self.adx_period}"
        self.minus_di_col = f"minus_di_{self.adx_period}"
        self.rsi_col = f"rsi_{self.rsi_period}"

        self.features = [
            AdxTrend(period=self.adx_period),
            RsiMom(period=self.rsi_period),
        ]

    def detect(
        self, features: pl.DataFrame, context: dict[str, Any] | None = None
    ) -> Signals:
        """Generate signals based on ADX regime with RSI."""
        adx = features[self.adx_col].to_numpy()
        plus_di = features[self.plus_di_col].to_numpy()
        minus_di = features[self.minus_di_col].to_numpy()
        rsi = features[self.rsi_col].to_numpy()
        n = len(adx)

        # Regime detection
        trend_regime = adx > self.adx_trend_threshold
        range_regime = adx < self.adx_range_threshold

        # Trend regime signals (follow direction)
        uptrend = plus_di > minus_di
        downtrend = minus_di > plus_di

        # Range regime signals (mean reversion)
        rsi_oversold = rsi < self.rsi_oversold
        rsi_overbought = rsi > self.rsi_overbought

        # Build signal type array
        signal_type = np.full(n, SignalType.NONE.value)

        if self.direction in ("long", "both"):
            # Trend: follow uptrend, Range: buy oversold
            long_signal = (trend_regime & uptrend) | (range_regime & rsi_oversold)
            signal_type = np.where(long_signal, SignalType.RISE.value, signal_type)

        if self.direction in ("short", "both"):
            # Trend: follow downtrend, Range: sell overbought
            short_signal = (trend_regime & downtrend) | (range_regime & rsi_overbought)
            signal_type = np.where(short_signal, SignalType.FALL.value, signal_type)

        out = features.select([
            self.pair_col,
            self.ts_col,
            pl.Series(name="signal_type", values=signal_type),
            pl.Series(name="signal", values=adx),
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
        base_warmup = max(self.adx_period * 10, self.rsi_period * 10)
        filter_warmup = max((f.warmup for f in self.filters), default=0)
        return max(base_warmup, filter_warmup)

    test_params: ClassVar[list[dict]] = [
        {"adx_period": 14, "rsi_period": 14, "direction": "long"},
    ]

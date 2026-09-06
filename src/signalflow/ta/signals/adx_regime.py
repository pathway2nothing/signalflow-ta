"""ADX-based regime and trend signal detectors."""

from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
import polars as pl

from signalflow.ta._compat import SignalDetector, Signals, SignalType, detector
from signalflow.ta.momentum import RsiMom
from signalflow.ta.signals.filters import SignalFilter
from signalflow.ta.trend import AdxTrend


@dataclass
@detector("detector/adx_di_cross")
class AdxDiCrossDetector(SignalDetector):
    """ADX trend regime detector with DI crossover.

    Uses ADX to confirm trend strength and DI crossover for direction.

    Signal logic:
        - LONG: +DI crosses above -DI AND ADX > threshold (strong uptrend)
        - SHORT: -DI crosses above +DI AND ADX > threshold (strong downtrend)
    """

    adx_period: int = 14
    adx_threshold: float = 25.0
    direction: str = "long"
    filters: list[SignalFilter] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.direction not in ("long", "short", "both"):
            raise ValueError(f"direction must be 'long', 'short', or 'both', got {self.direction}")

        self.adx_col = f"adx_{self.adx_period}"
        self.plus_di_col = f"dmp_{self.adx_period}"
        self.minus_di_col = f"dmn_{self.adx_period}"
        self.features = [AdxTrend(period=self.adx_period)]

    def detect(self, features: pl.DataFrame, context: dict[str, Any] | None = None) -> Signals:
        """Generate signals based on ADX regime and DI crossover."""
        pairs = features[self.pair_col].unique().sort().to_list()
        if len(pairs) > 1:
            results = []
            for pair in pairs:
                pair_df = features.filter(pl.col(self.pair_col) == pair)
                sig = self._detect_single(pair_df, context)
                if len(sig.value) > 0:
                    results.append(sig.value)
            if results:
                return Signals(pl.concat(results))
            return Signals(
                features.head(0).select(
                    [
                        self.pair_col,
                        self.ts_col,
                        pl.lit(0).alias("signal_type"),
                        pl.lit(0.0).alias("score"),
                    ]
                )
            )
        return self._detect_single(features, context)

    def _detect_single(self, features: pl.DataFrame, context: dict[str, Any] | None = None) -> Signals:
        adx = features[self.adx_col].to_numpy()
        plus_di = features[self.plus_di_col].to_numpy()
        minus_di = features[self.minus_di_col].to_numpy()
        n = len(adx)

        plus_di_prev = np.roll(plus_di, 1)
        minus_di_prev = np.roll(minus_di, 1)
        plus_di_prev[0] = np.nan
        minus_di_prev[0] = np.nan

        plus_crosses_above = (plus_di_prev <= minus_di_prev) & (plus_di > minus_di)
        minus_crosses_above = (minus_di_prev <= plus_di_prev) & (minus_di > plus_di)

        strong_trend = adx > self.adx_threshold

        signal_type: np.ndarray = np.full(n, SignalType.NONE.value)

        if self.direction in ("long", "both"):
            long_signal = plus_crosses_above & strong_trend
            signal_type = np.where(long_signal, SignalType.RISE.value, signal_type)

        if self.direction in ("short", "both"):
            short_signal = minus_crosses_above & strong_trend
            signal_type = np.where(short_signal, SignalType.FALL.value, signal_type)

        out = features.select(
            [
                self.pair_col,
                self.ts_col,
                pl.Series(name="signal_type", values=signal_type),
                pl.Series(name="score", values=adx),
            ]
        )

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
@detector("detector/adx_regime_rsi")
class AdxRegimeRsiDetector(SignalDetector):
    """ADX regime detector combining trend/range with RSI.

    Uses ADX to determine market regime:
    - ADX > threshold: trending market -> follow momentum
    - ADX < threshold: ranging market -> mean reversion

    Signal logic:
        - In TREND regime (ADX > threshold):
          - LONG: +DI > -DI (uptrend direction)
        - In RANGE regime (ADX < threshold):
          - LONG: RSI < oversold (mean reversion)
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
            raise ValueError(f"direction must be 'long', 'short', or 'both', got {self.direction}")

        self.adx_col = f"adx_{self.adx_period}"
        self.plus_di_col = f"dmp_{self.adx_period}"
        self.minus_di_col = f"dmn_{self.adx_period}"
        self.rsi_col = f"rsi_{self.rsi_period}"

        self.features = [
            AdxTrend(period=self.adx_period),
            RsiMom(period=self.rsi_period),
        ]

    def detect(self, features: pl.DataFrame, context: dict[str, Any] | None = None) -> Signals:
        """Generate signals based on ADX regime with RSI."""
        pairs = features[self.pair_col].unique().sort().to_list()
        if len(pairs) > 1:
            results = []
            for pair in pairs:
                pair_df = features.filter(pl.col(self.pair_col) == pair)
                sig = self._detect_single(pair_df, context)
                if len(sig.value) > 0:
                    results.append(sig.value)
            if results:
                return Signals(pl.concat(results))
            return Signals(
                features.head(0).select(
                    [
                        self.pair_col,
                        self.ts_col,
                        pl.lit(0).alias("signal_type"),
                        pl.lit(0.0).alias("score"),
                    ]
                )
            )
        return self._detect_single(features, context)

    def _detect_single(self, features: pl.DataFrame, context: dict[str, Any] | None = None) -> Signals:
        adx = features[self.adx_col].to_numpy()
        plus_di = features[self.plus_di_col].to_numpy()
        minus_di = features[self.minus_di_col].to_numpy()
        rsi = features[self.rsi_col].to_numpy()
        n = len(adx)

        trend_regime = adx > self.adx_trend_threshold
        range_regime = adx < self.adx_range_threshold

        uptrend = plus_di > minus_di
        downtrend = minus_di > plus_di

        rsi_oversold = rsi < self.rsi_oversold
        rsi_overbought = rsi > self.rsi_overbought

        signal_type: np.ndarray = np.full(n, SignalType.NONE.value)

        if self.direction in ("long", "both"):
            long_signal = (trend_regime & uptrend) | (range_regime & rsi_oversold)
            signal_type = np.where(long_signal, SignalType.RISE.value, signal_type)

        if self.direction in ("short", "both"):
            short_signal = (trend_regime & downtrend) | (range_regime & rsi_overbought)
            signal_type = np.where(short_signal, SignalType.FALL.value, signal_type)

        out = features.select(
            [
                self.pair_col,
                self.ts_col,
                pl.Series(name="signal_type", values=signal_type),
                pl.Series(name="score", values=adx),
            ]
        )

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

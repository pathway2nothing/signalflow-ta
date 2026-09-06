"""Stochastic oscillator-based signal detectors."""

from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
import polars as pl

from signalflow.ta._compat import SignalDetector, Signals, SignalType, detector
from signalflow.ta.momentum import StochMom
from signalflow.ta.signals.filters import SignalFilter


@dataclass
@detector("detector/stochastic_cross")
class StochasticCrossDetector(SignalDetector):
    """Stochastic oscillator crossover detector.

    Generates signals on %K/%D crossovers in extreme zones.

    Signal logic:
        - LONG: %K crosses above %D in oversold zone (< oversold_threshold)
        - SHORT: %K crosses below %D in overbought zone (> overbought_threshold)
    """

    stoch_period: int = 14
    stoch_smooth_k: int = 3
    stoch_smooth_d: int = 3
    oversold_threshold: float = 20.0
    overbought_threshold: float = 80.0
    direction: str = "long"
    filters: list[SignalFilter] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.direction not in ("long", "short", "both"):
            raise ValueError(f"direction must be 'long', 'short', or 'both', got {self.direction}")

        self.stoch_k_col = f"stoch_k_{self.stoch_period}"
        self.stoch_d_col = f"stoch_d_{self.stoch_period}"
        self.features = [
            StochMom(
                k_period=self.stoch_period,
                smooth_k=self.stoch_smooth_k,
                d_period=self.stoch_smooth_d,
            )
        ]

    def detect(self, features: pl.DataFrame, context: dict[str, Any] | None = None) -> Signals:
        """Generate signals based on Stochastic crossovers."""
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
        stoch_k = features[self.stoch_k_col].to_numpy()
        stoch_d = features[self.stoch_d_col].to_numpy()
        n = len(stoch_k)

        stoch_k_prev = np.roll(stoch_k, 1)
        stoch_d_prev = np.roll(stoch_d, 1)
        stoch_k_prev[0] = np.nan
        stoch_d_prev[0] = np.nan

        k_crosses_above_d = (stoch_k_prev <= stoch_d_prev) & (stoch_k > stoch_d)
        k_crosses_below_d = (stoch_k_prev >= stoch_d_prev) & (stoch_k < stoch_d)

        in_oversold = stoch_k < self.oversold_threshold
        in_overbought = stoch_k > self.overbought_threshold

        signal_type: np.ndarray = np.full(n, SignalType.NONE.value)

        if self.direction in ("long", "both"):
            long_signal = k_crosses_above_d & in_oversold
            signal_type = np.where(long_signal, SignalType.RISE.value, signal_type)

        if self.direction in ("short", "both"):
            short_signal = k_crosses_below_d & in_overbought
            signal_type = np.where(short_signal, SignalType.FALL.value, signal_type)

        out = features.select(
            [
                self.pair_col,
                self.ts_col,
                pl.Series(name="signal_type", values=signal_type),
                pl.Series(name="score", values=stoch_k),
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
        base_warmup = self.stoch_period * 5
        filter_warmup = max((f.warmup for f in self.filters), default=0)
        return max(base_warmup, filter_warmup)

    test_params: ClassVar[list[dict]] = [
        {"stoch_period": 14, "direction": "long"},
        {"stoch_period": 14, "direction": "both"},
    ]


@dataclass
@detector("detector/stochastic_extreme_zscore")
class StochasticExtremeZscoreDetector(SignalDetector):
    """Stochastic oscillator extreme zone detector with z-score.

    Generates signals when Stochastic reaches extreme z-score values.

    Signal logic:
        - Computes z-score of %K
        - LONG: %K z-score < -threshold AND %K < oversold
        - SHORT: %K z-score > threshold AND %K > overbought
    """

    stoch_period: int = 14
    stoch_smooth_k: int = 3
    stoch_smooth_d: int = 3
    zscore_window: int = 100
    zscore_threshold: float = 1.5
    oversold_threshold: float = 25.0
    overbought_threshold: float = 75.0
    direction: str = "long"
    filters: list[SignalFilter] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.direction not in ("long", "short", "both"):
            raise ValueError(f"direction must be 'long', 'short', or 'both', got {self.direction}")

        self.stoch_k_col = f"stoch_k_{self.stoch_period}"
        self.features = [
            StochMom(
                k_period=self.stoch_period,
                smooth_k=self.stoch_smooth_k,
                d_period=self.stoch_smooth_d,
            )
        ]

    def detect(self, features: pl.DataFrame, context: dict[str, Any] | None = None) -> Signals:
        """Generate signals based on Stochastic z-score extremes."""
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
        stoch_k = features[self.stoch_k_col].to_numpy()
        n = len(stoch_k)

        zscore = np.full(n, np.nan)
        for i in range(self.zscore_window - 1, n):
            window_vals = stoch_k[i - self.zscore_window + 1 : i + 1]
            valid = window_vals[~np.isnan(window_vals)]
            if len(valid) > 1:
                mean = np.mean(valid)
                std = np.std(valid, ddof=1)
                if std > 1e-10:
                    zscore[i] = (stoch_k[i] - mean) / std

        oversold_extreme = (zscore < -self.zscore_threshold) & (stoch_k < self.oversold_threshold)
        overbought_extreme = (zscore > self.zscore_threshold) & (stoch_k > self.overbought_threshold)

        signal_type: np.ndarray = np.full(n, SignalType.NONE.value)

        if self.direction in ("long", "both"):
            signal_type = np.where(oversold_extreme, SignalType.RISE.value, signal_type)

        if self.direction in ("short", "both"):
            signal_type = np.where(overbought_extreme, SignalType.FALL.value, signal_type)

        out = features.select(
            [
                self.pair_col,
                self.ts_col,
                pl.Series(name="signal_type", values=signal_type),
                pl.Series(name="score", values=zscore),
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
        base_warmup = max(self.stoch_period * 5, self.zscore_window)
        filter_warmup = max((f.warmup for f in self.filters), default=0)
        return max(base_warmup, filter_warmup)

    test_params: ClassVar[list[dict]] = [
        {"stoch_period": 14, "zscore_window": 100, "direction": "long"},
    ]

"""Money Flow Index-based signal detectors."""

from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
import polars as pl

from signalflow.ta._compat import Signals, SignalType, detector
from signalflow.ta._compat import SignalDetector
from signalflow.ta.signals.filters import SignalFilter
from signalflow.ta.volume import MfiVolume


@dataclass
@detector("ta/mfi_1")
class MfiDetector1(SignalDetector):
    """Money Flow Index extreme zone detector.

    MFI combines price and volume to identify overbought/oversold conditions.
    More reliable than pure price-based oscillators due to volume confirmation.

    Signal logic:
        - LONG: MFI < oversold_threshold (oversold with volume confirmation)
        - SHORT: MFI > overbought_threshold (overbought with volume confirmation)
    """

    mfi_period: int = 14
    oversold_threshold: float = 20.0
    overbought_threshold: float = 80.0
    direction: str = "long"
    filters: list[SignalFilter] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.direction not in ("long", "short", "both"):
            raise ValueError(f"direction must be 'long', 'short', or 'both', got {self.direction}")

        self.mfi_col = f"mfi_{self.mfi_period}"
        self.features = [MfiVolume(period=self.mfi_period)]

    def detect(self, features: pl.DataFrame, context: dict[str, Any] | None = None) -> Signals:
        """Generate signals based on MFI extreme zones."""
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
                        pl.lit(0.0).alias("signal"),
                    ]
                )
            )
        return self._detect_single(features, context)

    def _detect_single(self, features: pl.DataFrame, context: dict[str, Any] | None = None) -> Signals:
        mfi = features[self.mfi_col].to_numpy()
        n = len(mfi)

        oversold = mfi < self.oversold_threshold
        overbought = mfi > self.overbought_threshold

        signal_type: np.ndarray = np.full(n, SignalType.NONE.value)

        if self.direction in ("long", "both"):
            signal_type = np.where(oversold, SignalType.RISE.value, signal_type)

        if self.direction in ("short", "both"):
            signal_type = np.where(overbought, SignalType.FALL.value, signal_type)

        out = features.select(
            [
                self.pair_col,
                self.ts_col,
                pl.Series(name="signal_type", values=signal_type),
                pl.Series(name="signal", values=mfi),
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
        base_warmup = self.mfi_period * 5
        filter_warmup = max((f.warmup for f in self.filters), default=0)
        return max(base_warmup, filter_warmup)

    test_params: ClassVar[list[dict]] = [
        {"mfi_period": 14, "direction": "long"},
        {"mfi_period": 14, "direction": "both"},
    ]


@dataclass
@detector("ta/mfi_2")
class MfiDetector2(SignalDetector):
    """Money Flow Index with z-score and reversal detection.

    Enhanced MFI detector that looks for extreme z-score values
    combined with reversal (MFI turning up from oversold).

    Signal logic:
        - LONG: MFI z-score < -threshold AND MFI starts rising from oversold
        - SHORT: MFI z-score > threshold AND MFI starts falling from overbought
    """

    mfi_period: int = 14
    zscore_window: int = 100
    zscore_threshold: float = 1.5
    oversold_threshold: float = 25.0
    overbought_threshold: float = 75.0
    direction: str = "long"
    filters: list[SignalFilter] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.direction not in ("long", "short", "both"):
            raise ValueError(f"direction must be 'long', 'short', or 'both', got {self.direction}")

        self.mfi_col = f"mfi_{self.mfi_period}"
        self.features = [MfiVolume(period=self.mfi_period)]

    def detect(self, features: pl.DataFrame, context: dict[str, Any] | None = None) -> Signals:
        """Generate signals based on MFI z-score with reversal."""
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
                        pl.lit(0.0).alias("signal"),
                    ]
                )
            )
        return self._detect_single(features, context)

    def _detect_single(self, features: pl.DataFrame, context: dict[str, Any] | None = None) -> Signals:
        mfi = features[self.mfi_col].to_numpy()
        n = len(mfi)

        zscore = np.full(n, np.nan)
        for i in range(self.zscore_window - 1, n):
            window_vals = mfi[i - self.zscore_window + 1 : i + 1]
            valid = window_vals[~np.isnan(window_vals)]
            if len(valid) > 1:
                mean = np.mean(valid)
                std = np.std(valid, ddof=1)
                if std > 1e-10:
                    zscore[i] = (mfi[i] - mean) / std

        mfi_prev = np.roll(mfi, 1)
        mfi_prev[0] = np.nan
        mfi_rising = mfi > mfi_prev
        mfi_falling = mfi < mfi_prev

        long_signal = (zscore < -self.zscore_threshold) & (mfi < self.oversold_threshold) & mfi_rising
        short_signal = (zscore > self.zscore_threshold) & (mfi > self.overbought_threshold) & mfi_falling

        signal_type: np.ndarray = np.full(n, SignalType.NONE.value)

        if self.direction in ("long", "both"):
            signal_type = np.where(long_signal, SignalType.RISE.value, signal_type)

        if self.direction in ("short", "both"):
            signal_type = np.where(short_signal, SignalType.FALL.value, signal_type)

        out = features.select(
            [
                self.pair_col,
                self.ts_col,
                pl.Series(name="signal_type", values=signal_type),
                pl.Series(name="signal", values=zscore),
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
        base_warmup = max(self.mfi_period * 5, self.zscore_window)
        filter_warmup = max((f.warmup for f in self.filters), default=0)
        return max(base_warmup, filter_warmup)

    test_params: ClassVar[list[dict]] = [
        {"mfi_period": 14, "zscore_window": 100, "direction": "long"},
    ]

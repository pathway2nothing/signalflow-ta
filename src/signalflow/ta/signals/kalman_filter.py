"""Kalman filter-based signal detectors."""

from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
import polars as pl

from signalflow.ta._compat import Signals, SignalType, detector
from signalflow.ta._compat import SignalDetector
from signalflow.ta.signals.filters import SignalFilter


def _adaptive_kalman_filter(
    values: np.ndarray,
    window: int = 120,
    process_noise: float = 1e-3,
    measurement_noise: float = 1e-2,
) -> np.ndarray:
    """Apply Adaptive Kalman Filter to a time series.

    The filter adapts its parameters based on local volatility.
    """
    n = len(values)
    filtered = np.full(n, np.nan)

    if n < 2:
        return filtered

    x = values[0]
    p = 1.0

    filtered[0] = x

    for i in range(1, n):
        if np.isnan(values[i]):
            filtered[i] = x
            continue

        start_idx = max(0, i - window)
        local_vals = values[start_idx : i + 1]
        valid_vals = local_vals[~np.isnan(local_vals)]

        local_vol = np.std(valid_vals, ddof=1) if len(valid_vals) > 1 else 1.0

        q = process_noise * (1 + local_vol)
        r = measurement_noise * (1 + local_vol)

        x_pred = x
        p_pred = p + q

        k = p_pred / (p_pred + r)
        x = x_pred + k * (values[i] - x_pred)
        p = (1 - k) * p_pred

        filtered[i] = x

    return filtered


@dataclass
@detector("ta/kalman_filter_1")
class KalmanFilterDetector1(SignalDetector):
    """Adaptive Kalman Filter-based signal detector.

    Generates signals based on deviation from Kalman-filtered price.

    Signal logic:
        - Applies Adaptive Kalman Filter to high price
        - Computes score = kalman_high - close
        - Z-score normalizes the score
        - LONG: z-score > threshold (price below filtered high)
        - Optional uptrend filter
    """

    kf_window: int = 120
    zscore_window: int = 720
    process_noise: float = 1e-3
    measurement_noise: float = 1e-2
    zscore_threshold: float = 1.0
    use_uptrend_filter: bool = False
    uptrend_window: int = 5
    direction: str = "long"
    filters: list[SignalFilter] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.direction not in ("long", "short", "both"):
            raise ValueError(f"direction must be 'long', 'short', or 'both', got {self.direction}")

        self.features = []

    def detect(self, features: pl.DataFrame, context: dict[str, Any] | None = None) -> Signals:
        """Generate signals based on Kalman filter deviation."""
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
        high = features["high"].to_numpy()
        close = features["close"].to_numpy()
        n = len(close)

        kf_high = _adaptive_kalman_filter(
            high,
            window=self.kf_window,
            process_noise=self.process_noise,
            measurement_noise=self.measurement_noise,
        )

        score = kf_high - close

        zscore = np.full(n, np.nan)
        for i in range(self.zscore_window - 1, n):
            window_vals = score[i - self.zscore_window + 1 : i + 1]
            valid = window_vals[~np.isnan(window_vals)]
            if len(valid) > 1:
                mean = np.mean(valid)
                std = np.std(valid, ddof=1)
                if std > 1e-10:
                    zscore[i] = (score[i] - mean) / std

        long_signal = zscore > self.zscore_threshold
        short_signal = zscore < -self.zscore_threshold

        if self.use_uptrend_filter:
            sma = np.full(n, np.nan)
            for i in range(self.uptrend_window - 1, n):
                sma[i] = np.mean(close[i - self.uptrend_window + 1 : i + 1])
            uptrend = close > sma
            long_signal = long_signal & uptrend

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
        base_warmup = max(self.kf_window, self.zscore_window) * 5
        filter_warmup = max((f.warmup for f in self.filters), default=0)
        return max(base_warmup, filter_warmup)

    test_params: ClassVar[list[dict]] = [
        {"kf_window": 60, "zscore_window": 120, "direction": "long"},
        {"kf_window": 120, "zscore_window": 720, "direction": "long"},
    ]

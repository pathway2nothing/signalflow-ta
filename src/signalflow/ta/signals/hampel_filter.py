"""Hampel filter-based signal detectors."""

from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
import polars as pl

from signalflow import sf_component
from signalflow.core import Signals, SignalType
from signalflow.detector import SignalDetector
from signalflow.ta.signals.filters import SignalFilter


def _hampel_filter(values: np.ndarray, window: int, n_sigmas: float) -> np.ndarray:
    """Apply Hampel filter to detect and smooth outliers.

    The Hampel filter uses rolling median and MAD (median absolute deviation)
    to identify outliers and replace them with the median.

    Args:
        values: Input array.
        window: Rolling window size.
        n_sigmas: Number of MAD units for outlier threshold.

    Returns:
        Filtered array with outliers replaced by median.
    """
    n = len(values)
    result = values.copy()
    k = 1.4826  # Scale factor for MAD to approximate std dev

    for i in range(window - 1, n):
        window_vals = values[i - window + 1 : i + 1]
        median = np.median(window_vals)
        mad = k * np.median(np.abs(window_vals - median))

        if mad > 1e-10:
            if np.abs(values[i] - median) > n_sigmas * mad:
                result[i] = median

    return result


def _adaptive_hampel_filter(
    values: np.ndarray,
    window: int,
    n_sigmas: float,
    volatility_window: int,
) -> np.ndarray:
    """Apply adaptive Hampel filter with volatility-based threshold.

    Similar to standard Hampel but uses a separate volatility window
    to compute the MAD, making it more responsive to changing conditions.

    Args:
        values: Input array.
        window: Rolling window for median calculation.
        n_sigmas: Number of MAD units for outlier threshold.
        volatility_window: Window for volatility (MAD) calculation.

    Returns:
        Filtered array with outliers replaced by median.
    """
    n = len(values)
    result = values.copy()
    k = 1.4826

    start_idx = max(window, volatility_window) - 1

    for i in range(start_idx, n):
        # Median from main window
        window_vals = values[i - window + 1 : i + 1]
        median = np.median(window_vals)

        # MAD from volatility window
        vol_window_vals = values[i - volatility_window + 1 : i + 1]
        vol_median = np.median(vol_window_vals)
        mad = k * np.median(np.abs(vol_window_vals - vol_median))

        if mad > 1e-10:
            if np.abs(values[i] - median) > n_sigmas * mad:
                result[i] = median

    return result


@dataclass
@sf_component(name="ta/hampel_filter_1")
class HampelFilterDetector1(SignalDetector):
    """Hampel filter-based anomaly detector.

    Detects price deviations from the Hampel-filtered baseline.

    Signal logic:
        - Computes Hampel filter of close price
        - Calculates deviation score: (close - filtered) / close
        - LONG: score < -threshold (price below filtered baseline)
        - SHORT: score > threshold (price above filtered baseline)

    Attributes:
        window: Hampel filter window size.
        n_sigmas: Number of MAD units for outlier detection.
        threshold: Minimum score deviation for signal.
        direction: Signal direction - "long", "short", or "both".
        filters: List of SignalFilter instances to apply.

    Example:
        ```python
        from signalflow.ta.signals import HampelFilterDetector1

        detector = HampelFilterDetector1(
            window=240,
            n_sigmas=3.0,
            threshold=0.0000001,
            direction="long"
        )
        signals = detector.run(raw_data_view)
        ```
    """

    window: int = 240
    n_sigmas: float = 3.0
    threshold: float = 0.0000001
    direction: str = "long"
    filters: list[SignalFilter] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.direction not in ("long", "short", "both"):
            raise ValueError(
                f"direction must be 'long', 'short', or 'both', got {self.direction}"
            )

        self.features = []

    def detect(
        self, features: pl.DataFrame, context: dict[str, Any] | None = None
    ) -> Signals:
        """Generate signals based on Hampel filter deviation.

        Args:
            features: DataFrame with OHLCV data.
            context: Optional context dictionary.

        Returns:
            Signals container with detected signals.
        """
        close = features["close"].to_numpy()
        n = len(close)

        # Apply Hampel filter
        filtered = _hampel_filter(close, self.window, self.n_sigmas)

        # Calculate deviation score
        score = (close - filtered) / (close + 1e-10)

        # Signal conditions
        oversold = score < -self.threshold
        overbought = score > self.threshold

        # Build signal type array
        signal_type = np.full(n, SignalType.NONE.value)

        if self.direction in ("long", "both"):
            signal_type = np.where(oversold, SignalType.RISE.value, signal_type)

        if self.direction in ("short", "both"):
            signal_type = np.where(overbought, SignalType.FALL.value, signal_type)

        out = features.select(
            [
                self.pair_col,
                self.ts_col,
                pl.Series(name="signal_type", values=signal_type),
                pl.Series(name="signal", values=score),
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
        base_warmup = self.window * 5
        filter_warmup = max((f.warmup for f in self.filters), default=0)
        return max(base_warmup, filter_warmup)

    test_params: ClassVar[list[dict]] = [
        {"window": 60, "n_sigmas": 3.0, "direction": "long"},
        {"window": 240, "n_sigmas": 3.0, "direction": "long"},
    ]


@dataclass
@sf_component(name="ta/hampel_filter_2")
class HampelFilterDetector2(SignalDetector):
    """Adaptive Hampel filter-based anomaly detector.

    Uses separate volatility window for MAD calculation, making it
    more responsive to changing market conditions.

    Signal logic:
        - Computes adaptive Hampel filter with volatility window
        - Calculates deviation score: (close - filtered) / close
        - LONG: score < -threshold (price below filtered baseline)
        - SHORT: score > threshold (price above filtered baseline)

    Attributes:
        window: Hampel filter window for median.
        volatility_window: Window for MAD calculation.
        n_sigmas: Number of MAD units for outlier detection.
        threshold: Minimum score deviation for signal.
        direction: Signal direction.
        filters: List of SignalFilter instances.

    Example:
        ```python
        from signalflow.ta.signals import HampelFilterDetector2

        detector = HampelFilterDetector2(
            window=240,
            volatility_window=30,
            n_sigmas=3.0,
            threshold=0.0000001,
            direction="long"
        )
        signals = detector.run(raw_data_view)
        ```
    """

    window: int = 240
    volatility_window: int = 30
    n_sigmas: float = 3.0
    threshold: float = 0.0000001
    direction: str = "long"
    filters: list[SignalFilter] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.direction not in ("long", "short", "both"):
            raise ValueError(
                f"direction must be 'long', 'short', or 'both', got {self.direction}"
            )

        self.features = []

    def detect(
        self, features: pl.DataFrame, context: dict[str, Any] | None = None
    ) -> Signals:
        """Generate signals based on adaptive Hampel filter deviation.

        Args:
            features: DataFrame with OHLCV data.
            context: Optional context dictionary.

        Returns:
            Signals container with detected signals.
        """
        close = features["close"].to_numpy()
        n = len(close)

        # Apply adaptive Hampel filter
        filtered = _adaptive_hampel_filter(
            close, self.window, self.n_sigmas, self.volatility_window
        )

        # Calculate deviation score
        score = (close - filtered) / (close + 1e-10)

        # Signal conditions
        oversold = score < -self.threshold
        overbought = score > self.threshold

        # Build signal type array
        signal_type = np.full(n, SignalType.NONE.value)

        if self.direction in ("long", "both"):
            signal_type = np.where(oversold, SignalType.RISE.value, signal_type)

        if self.direction in ("short", "both"):
            signal_type = np.where(overbought, SignalType.FALL.value, signal_type)

        out = features.select(
            [
                self.pair_col,
                self.ts_col,
                pl.Series(name="signal_type", values=signal_type),
                pl.Series(name="signal", values=score),
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
        base_warmup = max(self.window, self.volatility_window) * 3
        filter_warmup = max((f.warmup for f in self.filters), default=0)
        return max(base_warmup, filter_warmup)

    test_params: ClassVar[list[dict]] = [
        {"window": 60, "volatility_window": 30, "n_sigmas": 3.0, "direction": "long"},
        {"window": 240, "volatility_window": 30, "n_sigmas": 3.0, "direction": "long"},
    ]

"""RSI anomaly detection signals."""

from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
import polars as pl

from signalflow import sf_component
from signalflow.core import Signals, SignalType
from signalflow.detector import SignalDetector
from signalflow.ta.momentum import RsiMom
from signalflow.ta._normalization import normalize_zscore
from signalflow.ta.signals.filters import SignalFilter


@dataclass
@sf_component(name="ta/rsi_anomaly_1")
class RsiAnomalyDetector1(SignalDetector):
    """RSI statistical anomaly detector.

    Detects extreme RSI deviations using rolling z-score normalization.
    Generates signals when RSI z-score crosses threshold boundaries.

    Signal logic:
        - LONG (RISE): RSI z-score < -threshold (oversold extreme)
        - SHORT (FALL): RSI z-score > threshold (overbought extreme)

    Attributes:
        rsi_period: RSI calculation period.
        zscore_window: Rolling window for z-score normalization.
        threshold: Z-score threshold for signal generation.
        direction: Signal direction - "long", "short", or "both".
        filters: List of SignalFilter instances to apply. All filters must pass.

    Example:
        ```python
        from signalflow.ta.signals import RsiAnomalyDetector1
        from signalflow.ta.signals.filters import PriceUptrendFilter

        # Basic usage
        detector = RsiAnomalyDetector1(rsi_period=14, threshold=1.0)

        # With filters
        detector = RsiAnomalyDetector1(
            rsi_period=720,
            threshold=1.0,
            direction="long",
            filters=[PriceUptrendFilter(window=5)]
        )
        ```
    """

    rsi_period: int = 720
    zscore_window: int = 720
    threshold: float = 1.0
    direction: str = "long"
    filters: list[SignalFilter] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.direction not in ("long", "short", "both"):
            raise ValueError(
                f"direction must be 'long', 'short', or 'both', got {self.direction}"
            )

        self.rsi_col = f"rsi_{self.rsi_period}"
        self.zscore_col = f"rsi_zscore_{self.rsi_period}"
        self.features = [RsiMom(period=self.rsi_period)]

    def detect(
        self, features: pl.DataFrame, context: dict[str, Any] | None = None
    ) -> Signals:
        """Generate signals based on RSI z-score anomalies.

        Args:
            features: DataFrame with computed RSI values.
            context: Optional context dictionary.

        Returns:
            Signals container with detected anomaly signals.
        """
        rsi = features[self.rsi_col].to_numpy()
        zscore = normalize_zscore(rsi, window=self.zscore_window)

        df = features.with_columns(pl.Series(name=self.zscore_col, values=zscore))

        # Base signal conditions
        long_cond = pl.col(self.zscore_col) < -self.threshold
        short_cond = pl.col(self.zscore_col) > self.threshold

        if self.direction == "long":
            signal_expr = pl.when(long_cond).then(pl.lit(SignalType.RISE.value))
        elif self.direction == "short":
            signal_expr = pl.when(short_cond).then(pl.lit(SignalType.FALL.value))
        else:
            signal_expr = (
                pl.when(long_cond)
                .then(pl.lit(SignalType.RISE.value))
                .when(short_cond)
                .then(pl.lit(SignalType.FALL.value))
            )

        out = df.select(
            [
                self.pair_col,
                self.ts_col,
                signal_expr.otherwise(pl.lit(SignalType.NONE.value)).alias(
                    "signal_type"
                ),
                pl.col(self.zscore_col).alias("signal"),
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

        # Filter out NONE signals
        out = out.filter(pl.col("signal_type") != SignalType.NONE.value)

        return Signals(out)

    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable output."""
        base_warmup = max(self.rsi_period * 10, self.zscore_window)
        filter_warmup = max((f.warmup for f in self.filters), default=0)
        return max(base_warmup, filter_warmup)

    test_params: ClassVar[list[dict]] = [
        {"rsi_period": 14, "zscore_window": 60, "threshold": 1.0, "direction": "long"},
        {"rsi_period": 14, "zscore_window": 60, "threshold": 1.0, "direction": "short"},
        {"rsi_period": 14, "zscore_window": 60, "threshold": 1.0, "direction": "both"},
        {
            "rsi_period": 720,
            "zscore_window": 720,
            "threshold": 1.0,
            "direction": "long",
        },
    ]

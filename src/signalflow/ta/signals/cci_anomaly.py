"""CCI anomaly detection signals."""

from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
import polars as pl

from signalflow import sf_component
from signalflow.core import Signals, SignalType
from signalflow.detector import SignalDetector
from signalflow.ta.momentum import CciMom
from signalflow.ta._normalization import normalize_zscore
from signalflow.ta.signals.filters import SignalFilter


@dataclass
@sf_component(name="ta/cci_anomaly_1")
class CciAnomalyDetector1(SignalDetector):
    """CCI statistical anomaly detector.

    Detects extreme CCI deviations using rolling z-score normalization.
    Generates signals when CCI z-score crosses threshold boundaries.

    Signal logic:
        - LONG (RISE): CCI z-score < -threshold (oversold extreme)
        - SHORT (FALL): CCI z-score > threshold (overbought extreme)

    Attributes:
        cci_period: CCI calculation period.
        cci_constant: CCI constant (default 0.015).
        zscore_window: Rolling window for z-score normalization.
        threshold: Z-score threshold for signal generation.
        direction: Signal direction - "long", "short", or "both".
        filters: List of SignalFilter instances to apply.

    Example:
        ```python
        from signalflow.ta.signals import CciAnomalyDetector1
        from signalflow.ta.signals.filters import RsiZscoreFilter

        # Basic usage
        detector = CciAnomalyDetector1(cci_period=180, threshold=1.5)

        # With RSI z-score filter (Sicily001 equivalent)
        detector = CciAnomalyDetector1(
            cci_period=180,
            threshold=1.0,
            direction="long",
            filters=[RsiZscoreFilter(rsi_period=180, threshold=-1.0)]
        )
        ```
    """

    cci_period: int = 180
    cci_constant: float = 0.015
    zscore_window: int = 180
    threshold: float = 1.0
    direction: str = "long"
    filters: list[SignalFilter] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.direction not in ("long", "short", "both"):
            raise ValueError(
                f"direction must be 'long', 'short', or 'both', got {self.direction}"
            )

        self.cci_col = f"cci_{self.cci_period}"
        self.zscore_col = f"cci_zscore_{self.cci_period}"
        self.features = [CciMom(period=self.cci_period, constant=self.cci_constant)]

    def detect(
        self, features: pl.DataFrame, context: dict[str, Any] | None = None
    ) -> Signals:
        """Generate signals based on CCI z-score anomalies.

        Args:
            features: DataFrame with computed CCI values.
            context: Optional context dictionary.

        Returns:
            Signals container with detected anomaly signals.
        """
        cci = features[self.cci_col].to_numpy()
        zscore = normalize_zscore(cci, window=self.zscore_window)

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

        out = out.filter(pl.col("signal_type") != SignalType.NONE.value)

        return Signals(out)

    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable output."""
        base_warmup = max(self.cci_period * 5, self.zscore_window)
        filter_warmup = max((f.warmup for f in self.filters), default=0)
        return max(base_warmup, filter_warmup)

    test_params: ClassVar[list[dict]] = [
        {"cci_period": 20, "zscore_window": 60, "threshold": 1.0, "direction": "long"},
        {"cci_period": 20, "zscore_window": 60, "threshold": 1.0, "direction": "short"},
        {"cci_period": 20, "zscore_window": 60, "threshold": 1.0, "direction": "both"},
        {
            "cci_period": 180,
            "zscore_window": 180,
            "threshold": 1.5,
            "direction": "long",
        },
    ]

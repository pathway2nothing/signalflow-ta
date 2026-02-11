"""Aroon crossover detection signals."""

from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
import polars as pl

from signalflow import sf_component
from signalflow.core import Signals, SignalType
from signalflow.detector import SignalDetector
from signalflow.ta.trend import AroonTrend
from signalflow.ta.signals.filters import SignalFilter


@dataclass
@sf_component(name="ta/aroon_cross_1")
class AroonCrossDetector1(SignalDetector):
    """Aroon crossover signal detector.

    Detects when Aroon Up crosses above Aroon Down (bullish crossover).

    Signal logic:
        - LONG (RISE): aroon_up > aroon_dn AND prev_aroon_up <= prev_aroon_dn

    Attributes:
        period: Aroon calculation period.
        direction: Signal direction - "long", "short", or "both".
        filters: List of SignalFilter instances to apply.

    Example:
        ```python
        from signalflow.ta.signals import AroonCrossDetector1
        from signalflow.ta.signals.filters import RsiZscoreFilter

        # Basic usage
        detector = AroonCrossDetector1(period=25)

        # With RSI z-score filter
        detector = AroonCrossDetector1(
            period=720,
            direction="long",
            filters=[RsiZscoreFilter(rsi_period=720, threshold=-0.5)]
        )
        ```
    """

    period: int = 720
    direction: str = "long"
    filters: list[SignalFilter] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.direction not in ("long", "short", "both"):
            raise ValueError(
                f"direction must be 'long', 'short', or 'both', got {self.direction}"
            )

        self.aroon_up_col = f"aroon_up_{self.period}"
        self.aroon_dn_col = f"aroon_dn_{self.period}"
        self.features = [AroonTrend(period=self.period)]

    def detect(
        self, features: pl.DataFrame, context: dict[str, Any] | None = None
    ) -> Signals:
        """Generate signals based on Aroon crossovers.

        Args:
            features: DataFrame with computed Aroon values.
            context: Optional context dictionary.

        Returns:
            Signals container with detected crossover signals.
        """
        aroon_up = features[self.aroon_up_col].to_numpy()
        aroon_dn = features[self.aroon_dn_col].to_numpy()

        # Previous values for crossover detection
        aroon_up_prev = np.roll(aroon_up, 1)
        aroon_dn_prev = np.roll(aroon_dn, 1)
        aroon_up_prev[0] = np.nan
        aroon_dn_prev[0] = np.nan

        # Crossover conditions
        bull_cross = (aroon_up > aroon_dn) & (aroon_up_prev <= aroon_dn_prev)
        bear_cross = (aroon_up < aroon_dn) & (aroon_up_prev >= aroon_dn_prev)

        # Build signal type array
        signal_type = np.full(len(features), SignalType.NONE.value)

        if self.direction in ("long", "both"):
            signal_type = np.where(bull_cross, SignalType.RISE.value, signal_type)
        if self.direction in ("short", "both"):
            signal_type = np.where(bear_cross, SignalType.FALL.value, signal_type)

        df = features.with_columns(pl.Series(name="_signal_type", values=signal_type))

        out = df.select(
            [
                self.pair_col,
                self.ts_col,
                pl.col("_signal_type").alias("signal_type"),
                pl.col(self.aroon_up_col).alias("signal"),
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
        base_warmup = self.period * 5
        filter_warmup = max((f.warmup for f in self.filters), default=0)
        return max(base_warmup, filter_warmup)

    test_params: ClassVar[list[dict]] = [
        {"period": 25, "direction": "long"},
        {"period": 25, "direction": "short"},
        {"period": 25, "direction": "both"},
        {"period": 720, "direction": "long"},
    ]

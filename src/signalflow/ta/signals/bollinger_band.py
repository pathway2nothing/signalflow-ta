"""Bollinger Band detection signals."""

from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
import polars as pl

from signalflow import sf_component
from signalflow.core import Signals, SignalType
from signalflow.detector import SignalDetector
from signalflow.ta.volatility import BollingerVol
from signalflow.ta.signals.filters import SignalFilter


@dataclass
@sf_component(name="ta/bollinger_band_1")
class BollingerBandDetector1(SignalDetector):
    """Bollinger Band breakout detector.

    Detects when price crosses outside Bollinger Bands.

    Signal logic:
        - LONG (RISE): close < lower band (oversold)
        - SHORT (FALL): close > upper band (overbought)

    Attributes:
        period: BB calculation period.
        std_dev: Number of standard deviations.
        direction: Signal direction - "long", "short", or "both".
        filters: List of SignalFilter instances to apply.

    Example:
        ```python
        from signalflow.ta.signals import BollingerBandDetector1
        from signalflow.ta.signals.filters import RsiZscoreFilter

        # Basic usage
        detector = BollingerBandDetector1(period=720, std_dev=2.0)

        # With RSI z-score filter (Kyoto001 equivalent)
        detector = BollingerBandDetector1(
            period=720,
            std_dev=2.0,
            direction="long",
            filters=[RsiZscoreFilter(rsi_period=720, threshold=-1.0)]
        )
        ```
    """

    period: int = 720
    std_dev: float = 2.0
    direction: str = "long"
    filters: list[SignalFilter] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.direction not in ("long", "short", "both"):
            raise ValueError(
                f"direction must be 'long', 'short', or 'both', got {self.direction}"
            )

        self.bb_upper_col = f"bb_upper_{self.period}"
        self.bb_lower_col = f"bb_lower_{self.period}"
        self.bb_pct_col = f"bb_pct_{self.period}"
        self.features = [BollingerVol(period=self.period, std_dev=self.std_dev)]

    def detect(
        self, features: pl.DataFrame, context: dict[str, Any] | None = None
    ) -> Signals:
        """Generate signals based on BB band crossings.

        Args:
            features: DataFrame with computed BB values.
            context: Optional context dictionary.

        Returns:
            Signals container with detected breakout signals.
        """
        close = features["close"].to_numpy()
        bb_upper = features[self.bb_upper_col].to_numpy()
        bb_lower = features[self.bb_lower_col].to_numpy()

        # Signal conditions
        below_lower = close < bb_lower
        above_upper = close > bb_upper

        # Build signal type array
        signal_type = np.full(len(features), SignalType.NONE.value)

        if self.direction in ("long", "both"):
            signal_type = np.where(below_lower, SignalType.RISE.value, signal_type)
        if self.direction in ("short", "both"):
            signal_type = np.where(above_upper, SignalType.FALL.value, signal_type)

        # Use bb_pct as signal strength
        bb_pct = features[self.bb_pct_col].to_numpy()

        df = features.with_columns(
            [
                pl.Series(name="_signal_type", values=signal_type),
                pl.Series(name="_signal", values=bb_pct),
            ]
        )

        out = df.select(
            [
                self.pair_col,
                self.ts_col,
                pl.col("_signal_type").alias("signal_type"),
                pl.col("_signal").alias("signal"),
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
        base_warmup = self.period * 3
        filter_warmup = max((f.warmup for f in self.filters), default=0)
        return max(base_warmup, filter_warmup)

    test_params: ClassVar[list[dict]] = [
        {"period": 20, "std_dev": 2.0, "direction": "long"},
        {"period": 20, "std_dev": 2.0, "direction": "short"},
        {"period": 20, "std_dev": 2.0, "direction": "both"},
        {"period": 720, "std_dev": 2.0, "direction": "long"},
    ]

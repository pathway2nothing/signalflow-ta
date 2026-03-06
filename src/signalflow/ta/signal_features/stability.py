"""Signal stability and consistency features."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, ClassVar

import polars as pl

from signalflow.signal_feature.base import SignalFeature


@dataclass
class SignalFlipRate(SignalFeature):
    """Fraction of consecutive signal pairs that change direction.

    A high flip rate (rise→fall→rise→fall) indicates an unreliable
    detector.  A low flip rate means signals are directionally stable.
    """

    requires_labels: ClassVar[bool] = False
    outputs: ClassVar[list[str]] = ["flip_rate_{window}"]

    window: int = 20

    def compute(
        self,
        signals: pl.DataFrame,
        labels: pl.DataFrame | None = None,
        context: dict[str, Any] | None = None,
    ) -> pl.DataFrame:
        col = self.output_cols()[0]
        df = signals.sort([self.group_col, self.ts_col])

        # 1 if signal_type differs from previous row, 0 otherwise
        df = df.with_columns(
            (pl.col("signal_type") != pl.col("signal_type").shift(1).over(self.group_col))
            .cast(pl.Int32)
            .alias("_flip"),
        )

        # Rolling mean of flips = flip rate
        df = df.with_columns(
            pl.col("_flip")
            .rolling_mean(window_size=self.window, min_samples=1)
            .over(self.group_col)
            .alias(col),
        )

        return df.select([self.group_col, self.ts_col, col])

    @property
    def warmup(self) -> int:
        return self.window


@dataclass
class SignalStreak(SignalFeature):
    """Length of the current same-direction signal streak.

    Outputs the number of consecutive signals with the same ``signal_type``
    and a directional encoding (1 for rise, -1 for fall).
    """

    requires_labels: ClassVar[bool] = False
    outputs: ClassVar[list[str]] = ["streak_len", "streak_dir"]

    def compute(
        self,
        signals: pl.DataFrame,
        labels: pl.DataFrame | None = None,
        context: dict[str, Any] | None = None,
    ) -> pl.DataFrame:
        df = signals.sort([self.group_col, self.ts_col])

        # Mark where signal_type changes
        df = df.with_columns(
            (pl.col("signal_type") != pl.col("signal_type").shift(1).over(self.group_col))
            .fill_null(True)
            .alias("_change"),
        )

        # Group id for each streak
        df = df.with_columns(
            pl.col("_change").cum_sum().over(self.group_col).alias("_group"),
        )

        # Streak length = row number within each streak group
        df = df.with_columns(
            pl.col(self.ts_col)
            .cum_count()
            .over([self.group_col, "_group"])
            .alias("streak_len"),
        )

        # Direction encoding
        df = df.with_columns(
            pl.when(pl.col("signal_type") == "rise")
            .then(pl.lit(1))
            .when(pl.col("signal_type") == "fall")
            .then(pl.lit(-1))
            .otherwise(pl.lit(0))
            .alias("streak_dir"),
        )

        return df.select([self.group_col, self.ts_col, "streak_len", "streak_dir"])


@dataclass
class SignalEntropy(SignalFeature):
    """Shannon entropy of signal_type distribution in a rolling window.

    High entropy → detector outputs are diverse (confused).
    Low entropy → detector is concentrated on one type (confident).
    Entropy is normalised to [0, 1] by dividing by log(n_types).
    """

    requires_labels: ClassVar[bool] = False
    outputs: ClassVar[list[str]] = ["signal_entropy_{window}"]

    window: int = 30

    def compute(
        self,
        signals: pl.DataFrame,
        labels: pl.DataFrame | None = None,
        context: dict[str, Any] | None = None,
    ) -> pl.DataFrame:
        col = self.output_cols()[0]
        df = signals.sort([self.group_col, self.ts_col])

        # Get unique signal types for encoding
        types = df["signal_type"].unique().drop_nulls().sort().to_list()
        n_types = len(types) if len(types) > 1 else 2
        log_n = math.log(n_types)

        # One-hot encode each type and compute rolling proportion
        entropy_parts: list[pl.Expr] = []
        for t in types:
            indicator_col = f"_is_{t}"
            df = df.with_columns(
                (pl.col("signal_type") == t).cast(pl.Float64).alias(indicator_col),
            )
            p = (
                pl.col(indicator_col)
                .rolling_mean(window_size=self.window, min_samples=1)
                .over(self.group_col)
            )
            # -p * log(p), handling p=0
            entropy_parts.append(
                pl.when(p > 0).then(-p * p.log()).otherwise(pl.lit(0.0))
            )

        # Sum parts and normalise
        if entropy_parts:
            total = entropy_parts[0]
            for part in entropy_parts[1:]:
                total = total + part
            df = df.with_columns(
                (total / log_n).alias(col),
            )
        else:
            df = df.with_columns(pl.lit(0.0).alias(col))

        return df.select([self.group_col, self.ts_col, col])

    @property
    def warmup(self) -> int:
        return self.window


@dataclass
class SignalTypeRatio(SignalFeature):
    """Fraction of rise and fall signals in a rolling window.

    Values near 1.0 for rise_ratio indicate a strong bullish bias.
    Balanced ratios (~0.5) suggest no directional edge.
    """

    requires_labels: ClassVar[bool] = False
    outputs: ClassVar[list[str]] = ["rise_ratio_{window}", "fall_ratio_{window}"]

    window: int = 30

    def compute(
        self,
        signals: pl.DataFrame,
        labels: pl.DataFrame | None = None,
        context: dict[str, Any] | None = None,
    ) -> pl.DataFrame:
        cols = self.output_cols()
        rise_col, fall_col = cols[0], cols[1]
        df = signals.sort([self.group_col, self.ts_col])

        df = df.with_columns(
            (pl.col("signal_type") == "rise").cast(pl.Float64).alias("_is_rise"),
            (pl.col("signal_type") == "fall").cast(pl.Float64).alias("_is_fall"),
        )

        df = df.with_columns(
            pl.col("_is_rise")
            .rolling_mean(window_size=self.window, min_samples=1)
            .over(self.group_col)
            .alias(rise_col),
            pl.col("_is_fall")
            .rolling_mean(window_size=self.window, min_samples=1)
            .over(self.group_col)
            .alias(fall_col),
        )

        return df.select([self.group_col, self.ts_col, rise_col, fall_col])

    @property
    def warmup(self) -> int:
        return self.window

"""Rolling accuracy and false signal rate features."""


from dataclasses import dataclass
from typing import Any, ClassVar

import polars as pl

from signalflow.ta._compat import SignalFeature


@dataclass
class RollingAccuracy(SignalFeature):
    """Hit rate over the last ``window`` resolved signals.

    The most fundamental supervised signal feature - tells the validator
    whether the detector is currently reliable.
    """

    requires_labels: ClassVar[bool] = True
    outputs: ClassVar[list[str]] = ["rolling_acc_{window}"]

    window: int = 50

    def compute(
        self,
        signals: pl.DataFrame,
        labels: pl.DataFrame | None = None,
        context: dict[str, Any] | None = None,
    ) -> pl.DataFrame:
        assert labels is not None
        col = self.output_cols()[0]

        merged = self.prepare_labels(signals, labels)
        merged = self.mask_unresolved(merged)
        df = merged.sort([self.group_col, self.ts_col])

        df = df.with_columns(
            pl.when(pl.col("label").is_not_null())
            .then((pl.col("signal_type") == pl.col("label")).cast(pl.Float64))
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias("_hit"),
        )

        df = df.with_columns(
            pl.col("_hit")
            .rolling_mean(window_size=self.window, min_samples=1)
            .over(self.group_col)
            .alias(col),
        )

        return df.select([self.group_col, self.ts_col, col])

    @property
    def warmup(self) -> int:
        return self.window


@dataclass
class TypeConditionalAccuracy(SignalFeature):
    """Accuracy broken down by signal_type.

    Some detectors are accurate on rise signals but poor on fall, or
    vice versa.  Gives the validator per-type trust.
    """

    requires_labels: ClassVar[bool] = True
    outputs: ClassVar[list[str]] = [
        "cond_acc_rise_{window}",
        "cond_acc_fall_{window}",
    ]

    window: int = 50

    def compute(
        self,
        signals: pl.DataFrame,
        labels: pl.DataFrame | None = None,
        context: dict[str, Any] | None = None,
    ) -> pl.DataFrame:
        assert labels is not None
        cols = self.output_cols()
        rise_col, fall_col = cols[0], cols[1]

        merged = self.prepare_labels(signals, labels)
        merged = self.mask_unresolved(merged)
        df = merged.sort([self.group_col, self.ts_col])

        df = df.with_columns(
            pl.when(
                (pl.col("signal_type") == "rise") & pl.col("label").is_not_null()
            )
            .then((pl.col("label") == "rise").cast(pl.Float64))
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias("_rise_hit"),
            pl.when(
                (pl.col("signal_type") == "fall") & pl.col("label").is_not_null()
            )
            .then((pl.col("label") == "fall").cast(pl.Float64))
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias("_fall_hit"),
        )

        df = df.with_columns(
            pl.col("_rise_hit")
            .rolling_mean(window_size=self.window, min_samples=1)
            .over(self.group_col)
            .alias(rise_col),
            pl.col("_fall_hit")
            .rolling_mean(window_size=self.window, min_samples=1)
            .over(self.group_col)
            .alias(fall_col),
        )

        return df.select([self.group_col, self.ts_col, rise_col, fall_col])

    @property
    def warmup(self) -> int:
        return self.window


@dataclass
class FalseSignalRate(SignalFeature):
    """Rolling false-positive and false-negative rates.

    FPR: fraction of rise signals that were actually falls.
    FNR: fraction of fall signals that were actually rises.
    """

    requires_labels: ClassVar[bool] = True
    outputs: ClassVar[list[str]] = ["fpr_{window}", "fnr_{window}"]

    window: int = 50

    def compute(
        self,
        signals: pl.DataFrame,
        labels: pl.DataFrame | None = None,
        context: dict[str, Any] | None = None,
    ) -> pl.DataFrame:
        assert labels is not None
        cols = self.output_cols()
        fpr_col, fnr_col = cols[0], cols[1]

        merged = self.prepare_labels(signals, labels)
        merged = self.mask_unresolved(merged)
        df = merged.sort([self.group_col, self.ts_col])

        df = df.with_columns(
            pl.when(
                (pl.col("signal_type") == "rise") & pl.col("label").is_not_null()
            )
            .then((pl.col("label") != "rise").cast(pl.Float64))
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias("_fp"),
            pl.when(
                (pl.col("signal_type") == "fall") & pl.col("label").is_not_null()
            )
            .then((pl.col("label") != "fall").cast(pl.Float64))
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias("_fn"),
        )

        df = df.with_columns(
            pl.col("_fp")
            .rolling_mean(window_size=self.window, min_samples=1)
            .over(self.group_col)
            .alias(fpr_col),
            pl.col("_fn")
            .rolling_mean(window_size=self.window, min_samples=1)
            .over(self.group_col)
            .alias(fnr_col),
        )

        return df.select([self.group_col, self.ts_col, fpr_col, fnr_col])

    @property
    def warmup(self) -> int:
        return self.window

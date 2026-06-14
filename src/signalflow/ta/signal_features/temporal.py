"""Temporal pattern signal features."""


from dataclasses import dataclass
from typing import Any, ClassVar

import polars as pl

from signalflow.ta._compat import SignalFeature


@dataclass
class TemporalBias(SignalFeature):
    """Accuracy bias by hour-of-day and day-of-week.

    Some detectors systematically perform better at certain times.
    Outputs the deviation of the current time-slot accuracy from the
    overall rolling accuracy.

    Requires labels with ``t_hit`` for causal masking.
    """

    requires_labels: ClassVar[bool] = True
    outputs: ClassVar[list[str]] = ["hour_acc_bias", "weekday_acc_bias"]

    window: int = 200

    def compute(
        self,
        signals: pl.DataFrame,
        labels: pl.DataFrame | None = None,
        context: dict[str, Any] | None = None,
    ) -> pl.DataFrame:
        assert labels is not None
        hour_col, weekday_col = self.output_cols()

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
            pl.col(self.ts_col).dt.hour().alias("_hour"),
            pl.col(self.ts_col).dt.weekday().alias("_weekday"),
        )

        overall_acc = (
            pl.col("_hit")
            .rolling_mean(window_size=self.window, min_samples=1)
            .over(self.group_col)
        )
        df = df.with_columns(overall_acc.alias("_overall_acc"))

        df = df.with_columns(
            pl.col(self.ts_col).dt.hour().alias("_cur_hour"),
        )

        df = df.with_columns(
            (pl.col("_hit") * 1.0).alias("_hour_hit_val"),
        )

        hour_acc = (
            pl.col("_hit")
            .rolling_mean(window_size=self.window, min_samples=1)
            .over([self.group_col, "_hour"])
        )
        weekday_acc = (
            pl.col("_hit")
            .rolling_mean(window_size=self.window, min_samples=1)
            .over([self.group_col, "_weekday"])
        )

        df = df.with_columns(
            (hour_acc - pl.col("_overall_acc")).alias(hour_col),
            (weekday_acc - pl.col("_overall_acc")).alias(weekday_col),
        )

        return df.select([self.group_col, self.ts_col, hour_col, weekday_col])

    @property
    def warmup(self) -> int:
        return self.window


@dataclass
class SignalAlphaDecay(SignalFeature):
    """Measures how quickly signal alpha decays after emission.

    For each resolved signal, computes the return at multiple horizons
    (1, 5, 10 bars after signal) and tracks the rolling ratio of
    far-horizon return to near-horizon return.

    A ratio near 1 means alpha persists; near 0 means it decays fast.

    Requires ``context["ohlcv"]`` with ``close`` column and labels.
    """

    requires_labels: ClassVar[bool] = True
    outputs: ClassVar[list[str]] = ["alpha_halflife_{window}", "alpha_decay_rate"]

    window: int = 50
    near_horizon: int = 1
    far_horizon: int = 10

    def compute(
        self,
        signals: pl.DataFrame,
        labels: pl.DataFrame | None = None,
        context: dict[str, Any] | None = None,
    ) -> pl.DataFrame:
        assert labels is not None
        cols = self.output_cols()
        halflife_col, decay_col = cols[0], cols[1]

        merged = self.prepare_labels(signals, labels)
        merged = self.mask_unresolved(merged)
        df = merged.sort([self.group_col, self.ts_col])

        ohlcv = context.get("ohlcv") if context else None
        if ohlcv is None or "close" not in ohlcv.columns:
            return df.select([self.group_col, self.ts_col]).with_columns(
                pl.lit(None, dtype=pl.Float64).alias(halflife_col),
                pl.lit(None, dtype=pl.Float64).alias(decay_col),
            )

        ohlcv_sorted = ohlcv.sort([self.group_col, self.ts_col])
        ret_df = ohlcv_sorted.with_columns(
            (
                pl.col("close").shift(-self.near_horizon).over(self.group_col)
                / pl.col("close")
                - 1.0
            ).alias("_ret_near"),
            (
                pl.col("close").shift(-self.far_horizon).over(self.group_col)
                / pl.col("close")
                - 1.0
            ).alias("_ret_far"),
        ).select([self.group_col, self.ts_col, "_ret_near", "_ret_far"])

        df = df.join(ret_df, on=[self.group_col, self.ts_col], how="left")

        direction = (
            pl.when(pl.col("signal_type") == "rise")
            .then(pl.lit(1.0))
            .when(pl.col("signal_type") == "fall")
            .then(pl.lit(-1.0))
            .otherwise(pl.lit(0.0))
        )

        df = df.with_columns(
            pl.when(pl.col("label").is_not_null())
            .then(pl.col("_ret_near") * direction)
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias("_near_signed"),
            pl.when(pl.col("label").is_not_null())
            .then(pl.col("_ret_far") * direction)
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias("_far_signed"),
        )

        near_mean = (
            pl.col("_near_signed")
            .rolling_mean(window_size=self.window, min_samples=1)
            .over(self.group_col)
        )
        far_mean = (
            pl.col("_far_signed")
            .rolling_mean(window_size=self.window, min_samples=1)
            .over(self.group_col)
        )

        df = df.with_columns(
            pl.when(near_mean.abs() > 1e-10)
            .then(far_mean / near_mean)
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias(decay_col),
        )

        df = df.with_columns(
            pl.when((pl.col(decay_col) > 0.01) & (pl.col(decay_col) < 10.0))
            .then(-float(self.far_horizon) / pl.col(decay_col).log())
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias(halflife_col),
        )

        return df.select([self.group_col, self.ts_col, halflife_col, decay_col])

    @property
    def warmup(self) -> int:
        return max(self.window, self.far_horizon)


@dataclass
class SignalLifetime(SignalFeature):
    """How many bars a signal stays valid before being contradicted.

    Lifetime = distance (in bars) to the next signal of the opposite type
    within the same pair.  Short lifetimes indicate the detector is
    flip-flopping; long lifetimes indicate conviction.
    """

    requires_labels: ClassVar[bool] = False
    outputs: ClassVar[list[str]] = [
        "signal_lifetime_mean_{window}",
        "signal_lifetime_std_{window}",
    ]

    window: int = 20

    def compute(
        self,
        signals: pl.DataFrame,
        labels: pl.DataFrame | None = None,
        context: dict[str, Any] | None = None,
    ) -> pl.DataFrame:
        cols = self.output_cols()
        mean_col, std_col = cols[0], cols[1]
        df = signals.sort([self.group_col, self.ts_col])

        df = df.with_columns(
            pl.col(self.ts_col).cum_count().over(self.group_col).cast(pl.Float64).alias("_idx"),
        )

        df = df.with_columns(
            (pl.col("signal_type") != pl.col("signal_type").shift(1).over(self.group_col))
            .fill_null(True)
            .alias("_flip"),
        )

        df = df.with_columns(
            pl.col("_flip").cum_sum().over(self.group_col).alias("_flip_group"),
        )

        group_len = pl.col(self.ts_col).cum_count().over([self.group_col, "_flip_group"])
        df = df.with_columns(group_len.cast(pl.Float64).alias("_cur_len"))

        df = df.with_columns(
            pl.when(pl.col("_flip"))
            .then(pl.col("_cur_len").shift(1).over(self.group_col))
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias("_completed_lifetime"),
        )

        df = df.with_columns(
            pl.col("_completed_lifetime")
            .rolling_mean(window_size=self.window, min_samples=1)
            .over(self.group_col)
            .alias(mean_col),
            pl.col("_completed_lifetime")
            .rolling_std(window_size=self.window, min_samples=2)
            .over(self.group_col)
            .alias(std_col),
        )

        return df.select([self.group_col, self.ts_col, mean_col, std_col])

    @property
    def warmup(self) -> int:
        return self.window

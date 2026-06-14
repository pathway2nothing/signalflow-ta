"""Probability-based signal features."""


from dataclasses import dataclass
from typing import Any, ClassVar

import polars as pl

from signalflow.ta._compat import SignalFeature


@dataclass
class ProbabilityMoments(SignalFeature):
    """Rolling mean, std, and slope of the probability column.

    Declining mean probability → detector losing confidence.
    Rising std → unstable confidence.
    Positive slope → improving confidence trend.
    """

    requires_labels: ClassVar[bool] = False
    outputs: ClassVar[list[str]] = [
        "prob_mean_{window}",
        "prob_std_{window}",
        "prob_slope_{window}",
    ]

    window: int = 20

    def compute(
        self,
        signals: pl.DataFrame,
        labels: pl.DataFrame | None = None,
        context: dict[str, Any] | None = None,
    ) -> pl.DataFrame:
        cols = self.output_cols()
        mean_col, std_col, slope_col = cols[0], cols[1], cols[2]
        df = signals.sort([self.group_col, self.ts_col])

        if "probability" not in df.columns:
            return df.select([self.group_col, self.ts_col]).with_columns(
                pl.lit(None, dtype=pl.Float64).alias(mean_col),
                pl.lit(None, dtype=pl.Float64).alias(std_col),
                pl.lit(None, dtype=pl.Float64).alias(slope_col),
            )

        prob = pl.col("probability").cast(pl.Float64)
        df = df.with_columns(
            prob.rolling_mean(window_size=self.window, min_samples=1)
            .over(self.group_col)
            .alias(mean_col),
            prob.rolling_std(window_size=self.window, min_samples=2)
            .over(self.group_col)
            .alias(std_col),
        )

        half = max(self.window // 2, 1)
        df = df.with_columns(
            (
                prob.rolling_mean(window_size=half, min_samples=1).over(self.group_col)
                - prob.rolling_mean(window_size=self.window, min_samples=1).over(self.group_col)
            ).alias(slope_col),
        )

        return df.select([self.group_col, self.ts_col, mean_col, std_col, slope_col])

    @property
    def warmup(self) -> int:
        return self.window


@dataclass
class CalibrationError(SignalFeature):
    """Difference between declared probability and empirical accuracy.

    If a detector declares probability=0.8 but only hits 60% of the
    time, calibration is off.  A well-calibrated detector has error ≈ 0.
    """

    requires_labels: ClassVar[bool] = True
    outputs: ClassVar[list[str]] = ["calibration_err_{window}"]

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

        rolling_acc = (
            pl.col("_hit")
            .rolling_mean(window_size=self.window, min_samples=1)
            .over(self.group_col)
        )

        prob_expr = pl.col("probability").cast(pl.Float64) if "probability" in df.columns else pl.lit(0.5)

        rolling_prob = (
            prob_expr
            .rolling_mean(window_size=self.window, min_samples=1)
            .over(self.group_col)
        )

        df = df.with_columns(
            (rolling_prob - rolling_acc).abs().alias(col),
        )

        return df.select([self.group_col, self.ts_col, col])

    @property
    def warmup(self) -> int:
        return self.window


@dataclass
class BayesianPosterior(SignalFeature):
    """Bayesian-updated probability combining detector output with base rate.

    Uses Bayes' theorem::

        P(correct | prob) = (prob * base_rate)
                          / (prob * base_rate + (1 - prob) * (1 - base_rate))

    where ``base_rate`` is the rolling empirical accuracy of the detector.
    """

    requires_labels: ClassVar[bool] = True
    outputs: ClassVar[list[str]] = ["bayesian_prob"]

    window: int = 100

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

        base = (
            pl.col("_hit")
            .rolling_mean(window_size=self.window, min_samples=1)
            .over(self.group_col)
        )

        p = pl.col("probability").cast(pl.Float64).clip(0.01, 0.99) if "probability" in df.columns else pl.lit(0.5)

        numerator = p * base
        denominator = numerator + (1.0 - p) * (1.0 - base)

        df = df.with_columns(
            (numerator / denominator).alias(col),
        )

        return df.select([self.group_col, self.ts_col, col])

    @property
    def warmup(self) -> int:
        return self.window

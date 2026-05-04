"""Outcome-based signal features (win/loss streaks)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import polars as pl

from signalflow.signal_feature.base import SignalFeature


@dataclass
class OutcomeStreak(SignalFeature):
    """Current consecutive win/loss streak + serial correlation.

    Tracks how many consecutive correct (or incorrect) signals the
    detector has produced.  A long loss streak is a strong warning
    signal for the validator.

    Outputs:
        outcome_streak: Positive for win streak, negative for loss.
        outcome_autocorr_{window}: Rolling autocorrelation of the hit
            sequence — positive means streaks are likely to continue,
            negative means mean-reversion.
    """

    requires_labels: ClassVar[bool] = True
    outputs: ClassVar[list[str]] = ["outcome_streak", "outcome_autocorr_{window}"]

    window: int = 30

    def compute(
        self,
        signals: pl.DataFrame,
        labels: pl.DataFrame | None = None,
        context: dict[str, Any] | None = None,
    ) -> pl.DataFrame:
        assert labels is not None
        cols = self.output_cols()
        streak_col, autocorr_col = cols[0], cols[1]

        merged = self.prepare_labels(signals, labels)
        merged = self.mask_unresolved(merged)
        df = merged.sort([self.group_col, self.ts_col])

        # Hit: 1 = correct, -1 = wrong, null = unresolved
        df = df.with_columns(
            pl.when(pl.col("label").is_not_null())
            .then(
                pl.when(pl.col("signal_type") == pl.col("label"))
                .then(pl.lit(1))
                .otherwise(pl.lit(-1))
            )
            .otherwise(pl.lit(None, dtype=pl.Int32))
            .alias("_outcome"),
        )

        # Streak: count consecutive same-outcome runs
        # Mark where outcome changes
        df = df.with_columns(
            (
                pl.col("_outcome") != pl.col("_outcome").shift(1).over(self.group_col)
            )
            .fill_null(True)
            .alias("_change"),
        )

        # Group id per streak
        df = df.with_columns(
            pl.col("_change").cum_sum().over(self.group_col).alias("_run_group"),
        )

        # Streak length (signed: positive for wins, negative for losses)
        raw_streak = pl.col(self.ts_col).cum_count().over([self.group_col, "_run_group"])
        df = df.with_columns(
            (raw_streak * pl.col("_outcome")).alias(streak_col),
        )

        # Serial autocorrelation: corr(outcome_t, outcome_{t-1}) in rolling window
        # Approximate with rolling cov / var
        outcome_f = pl.col("_outcome").cast(pl.Float64)
        outcome_lag = pl.col("_outcome").shift(1).over(self.group_col).cast(pl.Float64)

        df = df.with_columns(outcome_lag.alias("_outcome_lag"))

        rolling_cov = (
            (outcome_f * pl.col("_outcome_lag"))
            .rolling_mean(window_size=self.window, min_samples=3)
            .over(self.group_col)
            - outcome_f.rolling_mean(window_size=self.window, min_samples=3).over(self.group_col)
            * pl.col("_outcome_lag")
            .cast(pl.Float64)
            .rolling_mean(window_size=self.window, min_samples=3)
            .over(self.group_col)
        )
        rolling_var = outcome_f.rolling_var(window_size=self.window, min_samples=3).over(self.group_col)

        df = df.with_columns(
            pl.when(rolling_var > 0)
            .then(rolling_cov / rolling_var)
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias(autocorr_col),
        )

        return df.select([self.group_col, self.ts_col, streak_col, autocorr_col])

    @property
    def warmup(self) -> int:
        return self.window

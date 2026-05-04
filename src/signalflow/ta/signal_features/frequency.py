"""Signal frequency and inter-signal distance features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import polars as pl

from signalflow.signal_feature.base import SignalFeature

_ACTIVE_TYPES = {"rise", "fall"}


@dataclass
class SignalFrequency(SignalFeature):
    """Rolling count of active signals within a window.

    Counts how many active (rise/fall) signals were emitted in the last
    ``window`` rows.  A spike in frequency suggests noise or a regime
    change; a drop means the detector went quiet.
    """

    requires_labels: ClassVar[bool] = False
    outputs: ClassVar[list[str]] = ["signal_freq_{window}"]

    window: int = 50

    def compute(
        self,
        signals: pl.DataFrame,
        labels: pl.DataFrame | None = None,
        context: dict[str, Any] | None = None,
    ) -> pl.DataFrame:
        col = self.output_cols()[0]
        df = signals.sort([self.group_col, self.ts_col])

        df = df.with_columns(
            pl.when(pl.col("signal_type").is_in(list(_ACTIVE_TYPES)))
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .alias("_active"),
        )

        df = df.with_columns(
            pl.col("_active")
            .rolling_sum(window_size=self.window, min_samples=1)
            .over(self.group_col)
            .alias(col),
        )

        return df.select([self.group_col, self.ts_col, col])

    @property
    def warmup(self) -> int:
        return self.window


@dataclass
class InterSignalDistance(SignalFeature):
    """Bars since the previous signal + rolling z-score.

    Small ISD (clustered signals) often indicates noisy behaviour.
    Large ISD means the detector fired after a long silence — potentially
    a high-conviction event.
    """

    requires_labels: ClassVar[bool] = False
    outputs: ClassVar[list[str]] = ["isd_bars", "isd_zscore_{zscore_window}"]

    zscore_window: int = 20

    def compute(
        self,
        signals: pl.DataFrame,
        labels: pl.DataFrame | None = None,
        context: dict[str, Any] | None = None,
    ) -> pl.DataFrame:
        cols = self.output_cols()
        isd_col, zscore_col = cols[0], cols[1]
        df = signals.sort([self.group_col, self.ts_col])

        # Row number within each pair
        df = df.with_columns(
            pl.col(self.ts_col).cum_count().over(self.group_col).cast(pl.Float64).alias("_idx"),
        )

        # Distance: difference in row positions between consecutive signals
        df = df.with_columns(
            (pl.col("_idx") - pl.col("_idx").shift(1).over(self.group_col)).alias(isd_col),
        )

        # Rolling z-score of ISD
        rolling_mean = (
            pl.col(isd_col)
            .rolling_mean(window_size=self.zscore_window, min_samples=2)
            .over(self.group_col)
        )
        rolling_std = (
            pl.col(isd_col)
            .rolling_std(window_size=self.zscore_window, min_samples=2)
            .over(self.group_col)
        )
        df = df.with_columns(
            ((pl.col(isd_col) - rolling_mean) / rolling_std).alias(zscore_col),
        )

        return df.select([self.group_col, self.ts_col, isd_col, zscore_col])

    @property
    def warmup(self) -> int:
        return self.zscore_window

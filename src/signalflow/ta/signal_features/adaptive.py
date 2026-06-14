"""Adaptive and nonlinear signal features."""


from dataclasses import dataclass
from typing import Any, ClassVar

import polars as pl

from signalflow.ta._compat import SignalFeature


@dataclass
class SignalClusterQuality(SignalFeature):
    """Accuracy of clustered vs isolated signals.

    A cluster is defined as >= ``min_cluster`` signals within
    ``cluster_gap`` bars.  Separately tracks accuracy for signals
    that belong to a cluster vs those that are isolated.

    Some detectors produce bursts of noise; others confirm via clusters.
    """

    requires_labels: ClassVar[bool] = True
    outputs: ClassVar[list[str]] = [
        "cluster_acc_{window}",
        "isolated_acc_{window}",
        "cluster_ratio",
    ]

    window: int = 50
    cluster_gap: int = 3
    min_cluster: int = 3

    def compute(
        self,
        signals: pl.DataFrame,
        labels: pl.DataFrame | None = None,
        context: dict[str, Any] | None = None,
    ) -> pl.DataFrame:
        assert labels is not None
        cols = self.output_cols()
        cluster_col, isolated_col, ratio_col = cols[0], cols[1], cols[2]

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
            pl.col(self.ts_col).cum_count().over(self.group_col).alias("_idx"),
        )

        df = df.with_columns(
            pl.col("_idx")
            .rolling_sum(window_size=self.cluster_gap, min_samples=1)
            .over(self.group_col)
            .alias("_dummy_sum"),
        )

        df = df.with_columns(
            pl.lit(1)
            .rolling_sum(window_size=self.cluster_gap, min_samples=1)
            .over(self.group_col)
            .alias("_local_density"),
        )

        is_cluster = pl.col("_local_density") >= self.min_cluster

        df = df.with_columns(
            pl.when(is_cluster & pl.col("_hit").is_not_null())
            .then(pl.col("_hit"))
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias("_cluster_hit"),
            pl.when(is_cluster.not_() & pl.col("_hit").is_not_null())
            .then(pl.col("_hit"))
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias("_isolated_hit"),
        )

        df = df.with_columns(
            pl.col("_cluster_hit")
            .rolling_mean(window_size=self.window, min_samples=1)
            .over(self.group_col)
            .alias(cluster_col),
            pl.col("_isolated_hit")
            .rolling_mean(window_size=self.window, min_samples=1)
            .over(self.group_col)
            .alias(isolated_col),
        )

        df = df.with_columns(
            is_cluster.cast(pl.Float64)
            .rolling_mean(window_size=self.window, min_samples=1)
            .over(self.group_col)
            .alias(ratio_col),
        )

        return df.select([self.group_col, self.ts_col, cluster_col, isolated_col, ratio_col])

    @property
    def warmup(self) -> int:
        return max(self.window, self.cluster_gap)


@dataclass
class DrawdownSensitivity(SignalFeature):
    """Accuracy during equity drawdown vs normal periods.

    Some detectors break during drawdowns (they emit noise when the
    strategy is losing).  Tracks separate accuracy for drawdown
    periods vs normal periods.

    Requires ``context["ohlcv"]`` for equity proxy or
    ``context["equity"]`` directly.
    """

    requires_labels: ClassVar[bool] = True
    outputs: ClassVar[list[str]] = [
        "dd_acc_{window}",
        "normal_acc_{window}",
        "dd_sensitivity",
    ]

    window: int = 50
    dd_threshold: float = 0.05

    def compute(
        self,
        signals: pl.DataFrame,
        labels: pl.DataFrame | None = None,
        context: dict[str, Any] | None = None,
    ) -> pl.DataFrame:
        assert labels is not None
        cols = self.output_cols()
        dd_col, normal_col, sens_col = cols[0], cols[1], cols[2]

        merged = self.prepare_labels(signals, labels)
        merged = self.mask_unresolved(merged)
        df = merged.sort([self.group_col, self.ts_col])

        df = df.with_columns(
            pl.when(pl.col("label").is_not_null())
            .then((pl.col("signal_type") == pl.col("label")).cast(pl.Float64))
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias("_hit"),
        )

        equity_df: pl.DataFrame | None = None
        if context and "equity" in context:
            equity_df = context["equity"]
        elif context and "ohlcv" in context:
            ohlcv = context["ohlcv"]
            if "close" in ohlcv.columns:
                equity_df = ohlcv.select([self.group_col, self.ts_col, "close"]).rename({"close": "_equity"})

        if equity_df is None:
            return df.select([self.group_col, self.ts_col]).with_columns(
                pl.lit(None, dtype=pl.Float64).alias(dd_col),
                pl.lit(None, dtype=pl.Float64).alias(normal_col),
                pl.lit(None, dtype=pl.Float64).alias(sens_col),
            )

        if "_equity" not in equity_df.columns:
            equity_df = equity_df.rename({equity_df.columns[-1]: "_equity"})

        equity_df = equity_df.sort([self.group_col, self.ts_col])

        equity_df = equity_df.with_columns(
            pl.col("_equity")
            .cum_max()
            .over(self.group_col)
            .alias("_peak"),
        )

        equity_df = equity_df.with_columns(
            ((pl.col("_peak") - pl.col("_equity")) / pl.col("_peak")).alias("_dd"),
        )

        dd_info = equity_df.select([self.group_col, self.ts_col, "_dd"])
        df = df.join(dd_info, on=[self.group_col, self.ts_col], how="left")

        in_drawdown = pl.col("_dd") >= self.dd_threshold

        df = df.with_columns(
            pl.when(in_drawdown & pl.col("_hit").is_not_null())
            .then(pl.col("_hit"))
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias("_dd_hit"),
            pl.when(in_drawdown.not_() & pl.col("_hit").is_not_null())
            .then(pl.col("_hit"))
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias("_normal_hit"),
        )

        df = df.with_columns(
            pl.col("_dd_hit")
            .rolling_mean(window_size=self.window, min_samples=1)
            .over(self.group_col)
            .alias(dd_col),
            pl.col("_normal_hit")
            .rolling_mean(window_size=self.window, min_samples=1)
            .over(self.group_col)
            .alias(normal_col),
        )

        df = df.with_columns(
            (pl.col(normal_col) - pl.col(dd_col)).alias(sens_col),
        )

        return df.select([self.group_col, self.ts_col, dd_col, normal_col, sens_col])

    @property
    def warmup(self) -> int:
        return self.window


@dataclass
class AdaptiveConfidence(SignalFeature):
    """Exponentially weighted accuracy with adaptive learning rate.

    Unlike RollingAccuracy with a fixed window, this uses an EWM
    (exponentially weighted mean) that gives more weight to recent
    observations.  Reacts faster to regime changes.
    """

    requires_labels: ClassVar[bool] = True
    outputs: ClassVar[list[str]] = ["adaptive_conf"]

    span: int = 30

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
            .ewm_mean(span=self.span, ignore_nulls=True)
            .over(self.group_col)
            .alias(col),
        )

        return df.select([self.group_col, self.ts_col, col])

    @property
    def warmup(self) -> int:
        return self.span


@dataclass
class SignalFragility(SignalFeature):
    """Sensitivity of accuracy to window size.

    Computes accuracy at three window sizes (w/2, w, 2w) and measures
    the spread.  High fragility = accuracy is very window-dependent
    and thus unreliable.
    """

    requires_labels: ClassVar[bool] = True
    outputs: ClassVar[list[str]] = ["fragility_score"]

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

        w_half = max(self.window // 2, 2)
        w_double = self.window * 2

        acc_half = (
            pl.col("_hit")
            .rolling_mean(window_size=w_half, min_samples=1)
            .over(self.group_col)
        )
        acc_mid = (
            pl.col("_hit")
            .rolling_mean(window_size=self.window, min_samples=1)
            .over(self.group_col)
        )
        acc_double = (
            pl.col("_hit")
            .rolling_mean(window_size=w_double, min_samples=1)
            .over(self.group_col)
        )

        df = df.with_columns(
            acc_half.alias("_acc_h"),
            acc_mid.alias("_acc_m"),
            acc_double.alias("_acc_d"),
        )

        max_acc = pl.max_horizontal("_acc_h", "_acc_m", "_acc_d")
        min_acc = pl.min_horizontal("_acc_h", "_acc_m", "_acc_d")

        df = df.with_columns(
            (max_acc - min_acc).alias(col),
        )

        return df.select([self.group_col, self.ts_col, col])

    @property
    def warmup(self) -> int:
        return self.window * 2

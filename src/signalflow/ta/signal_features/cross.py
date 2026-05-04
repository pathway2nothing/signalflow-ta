"""Cross-signal features (between pairs / detectors)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import polars as pl

from signalflow.signal_feature.base import SignalFeature


@dataclass
class SignalCrowding(SignalFeature):
    """Fraction of pairs emitting the same signal type simultaneously.

    If 80% of pairs scream "rise" at the same time, it's likely a macro
    event rather than a pair-specific edge.  High crowding reduces the
    expected alpha of individual signals.

    Computes the ratio of pairs sharing the current signal type at each
    timestamp, plus a rolling z-score for detecting unusual crowding.
    """

    requires_labels: ClassVar[bool] = False
    outputs: ClassVar[list[str]] = ["crowding_ratio", "crowding_zscore_{zscore_window}"]

    zscore_window: int = 30

    def compute(
        self,
        signals: pl.DataFrame,
        labels: pl.DataFrame | None = None,
        context: dict[str, Any] | None = None,
    ) -> pl.DataFrame:
        cols = self.output_cols()
        ratio_col, zscore_col = cols[0], cols[1]
        df = signals.sort([self.group_col, self.ts_col])

        # Count total pairs and same-type pairs per timestamp
        # For each (timestamp, signal_type): how many pairs fired that type?
        type_counts = (
            df.group_by([self.ts_col, "signal_type"])
            .agg(pl.col(self.group_col).count().alias("_type_count"))
        )
        total_counts = (
            df.group_by(self.ts_col)
            .agg(pl.col(self.group_col).count().alias("_total_count"))
        )

        # Join back
        df = df.join(type_counts, on=[self.ts_col, "signal_type"], how="left")
        df = df.join(total_counts, on=self.ts_col, how="left")

        df = df.with_columns(
            (pl.col("_type_count").cast(pl.Float64) / pl.col("_total_count").cast(pl.Float64))
            .alias(ratio_col),
        )

        # Rolling z-score of crowding ratio
        df = df.sort([self.group_col, self.ts_col])
        rolling_mean = (
            pl.col(ratio_col)
            .rolling_mean(window_size=self.zscore_window, min_samples=2)
            .over(self.group_col)
        )
        rolling_std = (
            pl.col(ratio_col)
            .rolling_std(window_size=self.zscore_window, min_samples=2)
            .over(self.group_col)
        )

        df = df.with_columns(
            pl.when(rolling_std > 0)
            .then((pl.col(ratio_col) - rolling_mean) / rolling_std)
            .otherwise(pl.lit(0.0))
            .alias(zscore_col),
        )

        return df.select([self.group_col, self.ts_col, ratio_col, zscore_col])

    @property
    def warmup(self) -> int:
        return self.zscore_window


@dataclass
class CrossPairSpillover(SignalFeature):
    """Does a signal on pair A predict returns on other pairs?

    High spillover (IC between this pair's signal and other pairs'
    returns) means the signal is driven by macro factors, not
    pair-specific alpha.

    Requires ``context["ohlcv"]`` and labels for causal masking.
    """

    requires_labels: ClassVar[bool] = True
    outputs: ClassVar[list[str]] = ["spillover_ic_{window}"]

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

        ohlcv = context.get("ohlcv") if context else None
        if ohlcv is None or "close" not in ohlcv.columns:
            return df.select([self.group_col, self.ts_col]).with_columns(
                pl.lit(None, dtype=pl.Float64).alias(col),
            )

        # Average return across ALL pairs per timestamp
        avg_ret = (
            ohlcv.sort([self.group_col, self.ts_col])
            .with_columns(
                pl.col("close").pct_change().over(self.group_col).alias("_ret"),
            )
            .group_by(self.ts_col)
            .agg(pl.col("_ret").mean().alias("_avg_market_ret"))
        )

        df = df.join(avg_ret, on=self.ts_col, how="left")

        # Causal: only use market ret where label is resolved
        sig = pl.col("signal").cast(pl.Float64)
        mkt = (
            pl.when(pl.col("label").is_not_null())
            .then(pl.col("_avg_market_ret"))
            .otherwise(pl.lit(None, dtype=pl.Float64))
        )

        df = df.with_columns(mkt.alias("_mkt_causal"))

        # Rolling correlation (Pearson) between signal value and avg market return
        mkt_col = pl.col("_mkt_causal")
        rolling_cov = (
            (sig * mkt_col)
            .rolling_mean(window_size=self.window, min_samples=3)
            .over(self.group_col)
            - sig.rolling_mean(window_size=self.window, min_samples=3).over(self.group_col)
            * mkt_col.rolling_mean(window_size=self.window, min_samples=3).over(self.group_col)
        )
        sig_std = sig.rolling_std(window_size=self.window, min_samples=3).over(self.group_col)
        mkt_std = mkt_col.rolling_std(window_size=self.window, min_samples=3).over(self.group_col)

        df = df.with_columns(
            pl.when((sig_std > 0) & (mkt_std > 0))
            .then(rolling_cov / (sig_std * mkt_std))
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias(col),
        )

        return df.select([self.group_col, self.ts_col, col])

    @property
    def warmup(self) -> int:
        return self.window


@dataclass
class SignalDisagreement(SignalFeature):
    """Agreement ratio and accuracy when multiple detectors disagree.

    Requires ``context["all_signals"]`` — a DataFrame with signals from
    ALL detectors, with an extra ``detector`` column.  Computes the
    fraction of detectors agreeing with the current signal, and tracks
    separate accuracy for high-agreement vs low-agreement situations.

    Falls back to nulls if context is missing.
    """

    requires_labels: ClassVar[bool] = True
    outputs: ClassVar[list[str]] = ["agreement_ratio", "disagree_acc_{window}"]

    window: int = 50

    def compute(
        self,
        signals: pl.DataFrame,
        labels: pl.DataFrame | None = None,
        context: dict[str, Any] | None = None,
    ) -> pl.DataFrame:
        assert labels is not None
        cols = self.output_cols()
        agree_col, disagree_acc_col = cols[0], cols[1]

        merged = self.prepare_labels(signals, labels)
        merged = self.mask_unresolved(merged)
        df = merged.sort([self.group_col, self.ts_col])

        df = df.with_columns(
            pl.when(pl.col("label").is_not_null())
            .then((pl.col("signal_type") == pl.col("label")).cast(pl.Float64))
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias("_hit"),
        )

        all_signals = context.get("all_signals") if context else None
        if all_signals is None or "detector" not in all_signals.columns:
            return df.select([self.group_col, self.ts_col]).with_columns(
                pl.lit(None, dtype=pl.Float64).alias(agree_col),
                pl.lit(None, dtype=pl.Float64).alias(disagree_acc_col),
            )

        # Per (pair, timestamp): count detectors and how many agree
        # with the current signal's type
        all_sig = all_signals.sort([self.group_col, self.ts_col])

        # Total detectors per (pair, timestamp)
        det_total = (
            all_sig.group_by([self.group_col, self.ts_col])
            .agg(pl.col("detector").n_unique().alias("_n_detectors"))
        )

        # Count per (pair, timestamp, signal_type)
        det_agree = (
            all_sig.group_by([self.group_col, self.ts_col, "signal_type"])
            .agg(pl.col("detector").n_unique().alias("_n_agree"))
        )

        df = df.join(det_total, on=[self.group_col, self.ts_col], how="left")
        df = df.join(det_agree, on=[self.group_col, self.ts_col, "signal_type"], how="left")

        df = df.with_columns(
            (pl.col("_n_agree").cast(pl.Float64) / pl.col("_n_detectors").cast(pl.Float64))
            .alias(agree_col),
        )

        # Accuracy when agreement < 0.5 (disagreement)
        is_disagree = pl.col(agree_col) < 0.5
        df = df.with_columns(
            pl.when(is_disagree & pl.col("_hit").is_not_null())
            .then(pl.col("_hit"))
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias("_disagree_hit"),
        )

        df = df.with_columns(
            pl.col("_disagree_hit")
            .rolling_mean(window_size=self.window, min_samples=1)
            .over(self.group_col)
            .alias(disagree_acc_col),
        )

        return df.select([self.group_col, self.ts_col, agree_col, disagree_acc_col])

    @property
    def warmup(self) -> int:
        return self.window

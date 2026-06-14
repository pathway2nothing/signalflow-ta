"""Context-aware signal features (signal + market environment)."""


from dataclasses import dataclass
from typing import Any, ClassVar

import polars as pl

from signalflow.ta._compat import SignalFeature


@dataclass
class RegimeSensitivity(SignalFeature):
    """Accuracy broken down by volatility regime.

    Computes rolling realised volatility from OHLCV context, classifies
    each signal as fired during a high-vol or low-vol regime (above /
    below rolling median), and tracks per-regime accuracy separately.

    Requires ``context["ohlcv"]`` DataFrame with ``(pair, timestamp, close)``.
    Falls back to all-null if context is missing.
    """

    requires_labels: ClassVar[bool] = True
    outputs: ClassVar[list[str]] = [
        "acc_high_vol_{window}",
        "acc_low_vol_{window}",
        "regime_spread_{window}",
    ]

    window: int = 50
    vol_window: int = 20

    def compute(
        self,
        signals: pl.DataFrame,
        labels: pl.DataFrame | None = None,
        context: dict[str, Any] | None = None,
    ) -> pl.DataFrame:
        assert labels is not None
        cols = self.output_cols()
        hi_col, lo_col, spread_col = cols[0], cols[1], cols[2]

        merged = self.prepare_labels(signals, labels)
        merged = self.mask_unresolved(merged)
        df = merged.sort([self.group_col, self.ts_col])

        df = df.with_columns(
            pl.when(pl.col("label").is_not_null())
            .then((pl.col("signal_type") == pl.col("label")).cast(pl.Float64))
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias("_hit"),
        )

        ohlcv = context.get("ohlcv") if context else None
        if ohlcv is None or "close" not in ohlcv.columns:
            return df.select([self.group_col, self.ts_col]).with_columns(
                pl.lit(None, dtype=pl.Float64).alias(hi_col),
                pl.lit(None, dtype=pl.Float64).alias(lo_col),
                pl.lit(None, dtype=pl.Float64).alias(spread_col),
            )

        vol_df = (
            ohlcv.sort([self.group_col, self.ts_col])
            .with_columns(
                pl.col("close")
                .pct_change()
                .abs()
                .rolling_std(window_size=self.vol_window, min_samples=2)
                .over(self.group_col)
                .alias("_rvol"),
            )
            .select([self.group_col, self.ts_col, "_rvol"])
        )

        df = df.join(vol_df, on=[self.group_col, self.ts_col], how="left")

        vol_median = (
            pl.col("_rvol")
            .rolling_median(window_size=self.window, min_samples=2)
            .over(self.group_col)
        )
        df = df.with_columns(vol_median.alias("_vol_median"))

        is_high_vol = pl.col("_rvol") >= pl.col("_vol_median")

        df = df.with_columns(
            pl.when(is_high_vol & pl.col("_hit").is_not_null())
            .then(pl.col("_hit"))
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias("_hit_hi"),
            pl.when(is_high_vol.not_() & pl.col("_hit").is_not_null())
            .then(pl.col("_hit"))
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias("_hit_lo"),
        )

        df = df.with_columns(
            pl.col("_hit_hi")
            .rolling_mean(window_size=self.window, min_samples=1)
            .over(self.group_col)
            .alias(hi_col),
            pl.col("_hit_lo")
            .rolling_mean(window_size=self.window, min_samples=1)
            .over(self.group_col)
            .alias(lo_col),
        )

        df = df.with_columns(
            (pl.col(hi_col) - pl.col(lo_col)).alias(spread_col),
        )

        return df.select([self.group_col, self.ts_col, hi_col, lo_col, spread_col])

    @property
    def warmup(self) -> int:
        return max(self.window, self.vol_window)


@dataclass
class VolatilityAdjustedEV(SignalFeature):
    """Expected value normalised by realised volatility at signal time.

    A large EV during high volatility is less impressive than the same
    EV during calm markets.  Divides signed return by local volatility
    to produce a Sharpe-like per-signal metric.

    Requires ``context["ohlcv"]`` and ``ret`` in labels.
    """

    requires_labels: ClassVar[bool] = True
    outputs: ClassVar[list[str]] = ["vol_adj_ev_{window}"]

    window: int = 50
    vol_window: int = 20

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

        direction = (
            pl.when(pl.col("signal_type") == "rise")
            .then(pl.lit(1.0))
            .when(pl.col("signal_type") == "fall")
            .then(pl.lit(-1.0))
            .otherwise(pl.lit(0.0))
        )

        ret_expr = pl.col("ret").cast(pl.Float64) if "ret" in df.columns else pl.lit(0.0)
        df = df.with_columns(
            pl.when(pl.col("label").is_not_null())
            .then(ret_expr * direction)
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias("_signed_ret"),
        )

        ohlcv = context.get("ohlcv") if context else None
        if ohlcv is None or "close" not in ohlcv.columns:
            return df.select([self.group_col, self.ts_col]).with_columns(
                pl.lit(None, dtype=pl.Float64).alias(col),
            )

        vol_df = (
            ohlcv.sort([self.group_col, self.ts_col])
            .with_columns(
                pl.col("close")
                .pct_change()
                .abs()
                .rolling_std(window_size=self.vol_window, min_samples=2)
                .over(self.group_col)
                .alias("_rvol"),
            )
            .select([self.group_col, self.ts_col, "_rvol"])
        )

        df = df.join(vol_df, on=[self.group_col, self.ts_col], how="left")

        df = df.with_columns(
            pl.when(pl.col("_rvol") > 0)
            .then(pl.col("_signed_ret") / pl.col("_rvol"))
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias("_adj_ret"),
        )

        df = df.with_columns(
            pl.col("_adj_ret")
            .rolling_mean(window_size=self.window, min_samples=1)
            .over(self.group_col)
            .alias(col),
        )

        return df.select([self.group_col, self.ts_col, col])

    @property
    def warmup(self) -> int:
        return max(self.window, self.vol_window)


@dataclass
class MomentumAlignment(SignalFeature):
    """Accuracy split by whether the signal aligns with current momentum.

    A rise signal during an uptrend is trend-following; a rise signal
    during a downtrend is mean-reversion.  Tracks accuracy for each.

    Requires ``context["ohlcv"]`` with ``close`` column.
    """

    requires_labels: ClassVar[bool] = True
    outputs: ClassVar[list[str]] = [
        "trend_aligned_acc_{window}",
        "trend_counter_acc_{window}",
    ]

    window: int = 50
    mom_window: int = 20

    def compute(
        self,
        signals: pl.DataFrame,
        labels: pl.DataFrame | None = None,
        context: dict[str, Any] | None = None,
    ) -> pl.DataFrame:
        assert labels is not None
        cols = self.output_cols()
        aligned_col, counter_col = cols[0], cols[1]

        merged = self.prepare_labels(signals, labels)
        merged = self.mask_unresolved(merged)
        df = merged.sort([self.group_col, self.ts_col])

        df = df.with_columns(
            pl.when(pl.col("label").is_not_null())
            .then((pl.col("signal_type") == pl.col("label")).cast(pl.Float64))
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias("_hit"),
        )

        ohlcv = context.get("ohlcv") if context else None
        if ohlcv is None or "close" not in ohlcv.columns:
            return df.select([self.group_col, self.ts_col]).with_columns(
                pl.lit(None, dtype=pl.Float64).alias(aligned_col),
                pl.lit(None, dtype=pl.Float64).alias(counter_col),
            )

        mom_df = (
            ohlcv.sort([self.group_col, self.ts_col])
            .with_columns(
                pl.col("close")
                .pct_change(n=self.mom_window)
                .sign()
                .over(self.group_col)
                .alias("_mom_sign"),
            )
            .select([self.group_col, self.ts_col, "_mom_sign"])
        )

        df = df.join(mom_df, on=[self.group_col, self.ts_col], how="left")

        sig_dir = (
            pl.when(pl.col("signal_type") == "rise")
            .then(pl.lit(1.0))
            .when(pl.col("signal_type") == "fall")
            .then(pl.lit(-1.0))
            .otherwise(pl.lit(0.0))
        )
        is_aligned = sig_dir == pl.col("_mom_sign")

        df = df.with_columns(
            pl.when(is_aligned & pl.col("_hit").is_not_null())
            .then(pl.col("_hit"))
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias("_hit_aligned"),
            pl.when(is_aligned.not_() & pl.col("_hit").is_not_null())
            .then(pl.col("_hit"))
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias("_hit_counter"),
        )

        df = df.with_columns(
            pl.col("_hit_aligned")
            .rolling_mean(window_size=self.window, min_samples=1)
            .over(self.group_col)
            .alias(aligned_col),
            pl.col("_hit_counter")
            .rolling_mean(window_size=self.window, min_samples=1)
            .over(self.group_col)
            .alias(counter_col),
        )

        return df.select([self.group_col, self.ts_col, aligned_col, counter_col])

    @property
    def warmup(self) -> int:
        return max(self.window, self.mom_window)

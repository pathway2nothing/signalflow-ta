"""Performance-based signal features (return-aware, supervised)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import polars as pl

from signalflow.signal_feature.base import SignalFeature


@dataclass
class RollingExpectedValue(SignalFeature):
    """Expected value of a signal: mean return conditioned on resolved signals.

    EV = rolling_mean(ret * direction_sign) over a window of resolved
    signals.  Positive EV means the detector's signals, on average,
    produce positive returns; negative EV means they destroy value.

    Requires ``ret`` column in labels (the actual return from signal
    to resolution).
    """

    requires_labels: ClassVar[bool] = True
    outputs: ClassVar[list[str]] = ["rolling_ev_{window}"]

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

        # Direction sign: rise → +1, fall → -1
        direction = (
            pl.when(pl.col("signal_type") == "rise")
            .then(pl.lit(1.0))
            .when(pl.col("signal_type") == "fall")
            .then(pl.lit(-1.0))
            .otherwise(pl.lit(0.0))
        )

        # Signed return: ret * direction (null if label unresolved or ret missing)
        if "ret" in df.columns:
            ret_expr = pl.col("ret").cast(pl.Float64)
        else:
            # Fallback: use hit/miss as ±1 proxy
            ret_expr = (
                pl.when(pl.col("label").is_not_null())
                .then(
                    pl.when(pl.col("signal_type") == pl.col("label"))
                    .then(pl.lit(1.0))
                    .otherwise(pl.lit(-1.0))
                )
                .otherwise(pl.lit(None, dtype=pl.Float64))
            )

        df = df.with_columns(
            pl.when(pl.col("label").is_not_null())
            .then(ret_expr * direction)
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias("_signed_ret"),
        )

        df = df.with_columns(
            pl.col("_signed_ret")
            .rolling_mean(window_size=self.window, min_samples=1)
            .over(self.group_col)
            .alias(col),
        )

        return df.select([self.group_col, self.ts_col, col])

    @property
    def warmup(self) -> int:
        return self.window


@dataclass
class RollingProfitFactor(SignalFeature):
    """Ratio of gross wins to gross losses over a rolling window.

    PF = sum(positive_rets) / abs(sum(negative_rets)).
    PF > 1 means the detector is net profitable; PF < 1 means net loss.

    Requires ``ret`` column in labels.  Falls back to hit/miss ±1
    proxy when ``ret`` is absent.
    """

    requires_labels: ClassVar[bool] = True
    outputs: ClassVar[list[str]] = ["rolling_pf_{window}"]

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

        # Direction sign
        direction = (
            pl.when(pl.col("signal_type") == "rise")
            .then(pl.lit(1.0))
            .when(pl.col("signal_type") == "fall")
            .then(pl.lit(-1.0))
            .otherwise(pl.lit(0.0))
        )

        if "ret" in df.columns:
            ret_expr = pl.col("ret").cast(pl.Float64)
        else:
            ret_expr = (
                pl.when(pl.col("label").is_not_null())
                .then(
                    pl.when(pl.col("signal_type") == pl.col("label"))
                    .then(pl.lit(1.0))
                    .otherwise(pl.lit(-1.0))
                )
                .otherwise(pl.lit(None, dtype=pl.Float64))
            )

        df = df.with_columns(
            pl.when(pl.col("label").is_not_null())
            .then(ret_expr * direction)
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias("_signed_ret"),
        )

        # Separate wins and losses
        df = df.with_columns(
            pl.when(pl.col("_signed_ret") > 0)
            .then(pl.col("_signed_ret"))
            .otherwise(pl.lit(0.0))
            .alias("_win"),
            pl.when(pl.col("_signed_ret") < 0)
            .then(pl.col("_signed_ret").abs())
            .otherwise(pl.lit(0.0))
            .alias("_loss"),
        )

        # Rolling sums
        gross_win = (
            pl.col("_win")
            .rolling_sum(window_size=self.window, min_samples=1)
            .over(self.group_col)
        )
        gross_loss = (
            pl.col("_loss")
            .rolling_sum(window_size=self.window, min_samples=1)
            .over(self.group_col)
        )

        # Profit factor = gross_win / gross_loss (null if no losses)
        df = df.with_columns(
            pl.when(gross_loss > 0)
            .then(gross_win / gross_loss)
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias(col),
        )

        return df.select([self.group_col, self.ts_col, col])

    @property
    def warmup(self) -> int:
        return self.window


@dataclass
class InformationCoefficient(SignalFeature):
    """Rolling rank correlation between signal value and actual return.

    IC is the Spearman rank correlation between ``signal`` (detector's
    raw score) and ``ret`` (realised return).  A high IC means the
    detector's signal magnitude is informative about return magnitude.

    Uses a Pearson correlation of ranks as an approximation for the
    rolling window.
    """

    requires_labels: ClassVar[bool] = True
    outputs: ClassVar[list[str]] = ["rolling_ic_{window}"]

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

        # Need both signal and ret columns
        has_signal = "signal" in df.columns
        has_ret = "ret" in df.columns

        if not has_signal or not has_ret:
            return df.select([self.group_col, self.ts_col]).with_columns(
                pl.lit(None, dtype=pl.Float64).alias(col),
            )

        # Null out ret where label is unresolved
        df = df.with_columns(
            pl.when(pl.col("label").is_not_null())
            .then(pl.col("ret").cast(pl.Float64))
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias("_ret_causal"),
        )

        # Rolling rank correlation via Pearson on rolling ranks
        # Approximate with rolling covariance / (std_signal * std_ret)
        sig = pl.col("signal").cast(pl.Float64)
        ret = pl.col("_ret_causal")

        rolling_cov = (
            (sig * ret).rolling_mean(window_size=self.window, min_samples=3).over(self.group_col)
            - sig.rolling_mean(window_size=self.window, min_samples=3).over(self.group_col)
            * ret.rolling_mean(window_size=self.window, min_samples=3).over(self.group_col)
        )
        sig_std = sig.rolling_std(window_size=self.window, min_samples=3).over(self.group_col)
        ret_std = ret.rolling_std(window_size=self.window, min_samples=3).over(self.group_col)

        df = df.with_columns(
            pl.when((sig_std > 0) & (ret_std > 0))
            .then(rolling_cov / (sig_std * ret_std))
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias(col),
        )

        return df.select([self.group_col, self.ts_col, col])

    @property
    def warmup(self) -> int:
        return self.window

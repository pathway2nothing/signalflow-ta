"""Information-theoretic signal features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import polars as pl

from signalflow.signal_feature.base import SignalFeature


@dataclass
class SignalSurprise(SignalFeature):
    """KL divergence of current signal distribution from rolling baseline.

    A "surprising" signal (e.g. a fall after 20 consecutive rises) may
    be more informative than a predictable one.  Measures how much the
    instantaneous signal_type distribution diverges from the expected
    distribution based on recent history.

    Approximated as the negative log-probability of the current
    signal_type under the rolling distribution.
    """

    requires_labels: ClassVar[bool] = False
    outputs: ClassVar[list[str]] = ["signal_surprise_{window}"]

    window: int = 30

    def compute(
        self,
        signals: pl.DataFrame,
        labels: pl.DataFrame | None = None,
        context: dict[str, Any] | None = None,
    ) -> pl.DataFrame:
        col = self.output_cols()[0]
        df = signals.sort([self.group_col, self.ts_col])

        # Rolling proportion of each type
        df = df.with_columns(
            (pl.col("signal_type") == "rise").cast(pl.Float64).alias("_is_rise"),
        )

        rise_prob = (
            pl.col("_is_rise")
            .rolling_mean(window_size=self.window, min_samples=1)
            .over(self.group_col)
        )

        # Surprise = -log(p) where p is the probability of the observed type
        # For rise: p = rise_prob; for fall: p = 1 - rise_prob
        # Clamp to avoid log(0)
        df = df.with_columns(rise_prob.alias("_rise_prob"))

        df = df.with_columns(
            pl.when(pl.col("signal_type") == "rise")
            .then(pl.col("_rise_prob").clip(0.01, 0.99))
            .otherwise((1.0 - pl.col("_rise_prob")).clip(0.01, 0.99))
            .alias("_obs_prob"),
        )

        df = df.with_columns(
            (-pl.col("_obs_prob").log()).alias(col),
        )

        return df.select([self.group_col, self.ts_col, col])

    @property
    def warmup(self) -> int:
        return self.window


@dataclass
class MutualInformation(SignalFeature):
    """Discretised mutual information between signal value and return.

    Bins signal values and returns into quantiles and estimates MI
    from their joint distribution over a rolling window.  High MI
    means the signal magnitude is informative about return magnitude.

    Requires ``context["ohlcv"]`` with ``close`` and labels for causal masking.
    """

    requires_labels: ClassVar[bool] = True
    outputs: ClassVar[list[str]] = ["mi_signal_ret_{window}"]

    window: int = 100
    n_bins: int = 5

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

        has_signal = "signal" in df.columns
        has_ret = "ret" in df.columns

        if not has_signal or not has_ret:
            return df.select([self.group_col, self.ts_col]).with_columns(
                pl.lit(None, dtype=pl.Float64).alias(col),
            )

        # Causal ret
        df = df.with_columns(
            pl.when(pl.col("label").is_not_null())
            .then(pl.col("ret").cast(pl.Float64))
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias("_ret_causal"),
        )

        # Bin signal and return into quantiles (using rolling rank / window)
        # Approximate: rolling rank within window / window_size
        sig = pl.col("signal").cast(pl.Float64)
        ret = pl.col("_ret_causal")

        # Pearson |correlation| as MI lower bound proxy:
        # MI >= 0.5 * log(1 / (1 - rho^2))
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
            .otherwise(pl.lit(0.0))
            .alias("_rho"),
        )

        # MI lower bound: 0.5 * ln(1 / (1 - rho^2)), clamped
        df = df.with_columns(
            pl.when(pl.col("_rho").abs() < 0.99)
            .then(
                0.5 * (1.0 / (1.0 - pl.col("_rho") * pl.col("_rho"))).log()
            )
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias(col),
        )

        return df.select([self.group_col, self.ts_col, col])

    @property
    def warmup(self) -> int:
        return self.window


@dataclass
class BayesianSurpriseRate(SignalFeature):
    """Rate of change of Bayesian posterior probability.

    Measures how fast the posterior (base_rate from rolling accuracy)
    is shifting.  Rapid changes indicate an unstable detector that
    the validator should distrust.

    Uses the absolute first derivative of the rolling accuracy as
    a proxy for posterior update magnitude.
    """

    requires_labels: ClassVar[bool] = True
    outputs: ClassVar[list[str]] = ["bayes_surprise_{window}"]

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

        # Rolling accuracy (posterior proxy)
        rolling_acc = (
            pl.col("_hit")
            .rolling_mean(window_size=self.window, min_samples=1)
            .over(self.group_col)
        )
        df = df.with_columns(rolling_acc.alias("_posterior"))

        # First derivative: absolute change between consecutive posteriors
        df = df.with_columns(
            (pl.col("_posterior") - pl.col("_posterior").shift(1).over(self.group_col))
            .abs()
            .alias("_delta_post"),
        )

        # Rolling mean of absolute delta = surprise rate
        df = df.with_columns(
            pl.col("_delta_post")
            .rolling_mean(window_size=self.window, min_samples=1)
            .over(self.group_col)
            .alias(col),
        )

        return df.select([self.group_col, self.ts_col, col])

    @property
    def warmup(self) -> int:
        return self.window

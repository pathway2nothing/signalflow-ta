"""Rolling autocorrelation features."""
from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar

import polars as pl

from signalflow.feature.base import Feature


def _rolling_pearson(x: pl.Expr, y: pl.Expr, window: int):
    """Helper: build rolling Pearson corr terms; returns (xy, xm, ym, xx, yy) expression bundle.

    Caller must combine via: ((xy - xm*ym) / sqrt((xx-xm²)*(yy-ym²))).
    """
    return (
        (x * y).rolling_mean(window),
        x.rolling_mean(window),
        y.rolling_mean(window),
        (x ** 2).rolling_mean(window),
        (y ** 2).rolling_mean(window),
    )


@dataclass
class ReturnAutocorrShort(Feature):
    """Rolling autocorr(returns, lag) over window — mean-reversion vs continuity."""
    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["ret_acf_{lag}_{window}"]
    lag: int = 1
    window: int = 240

    def compute_pair(self, df):
        out = f"ret_acf_{self.lag}_{self.window}"
        ret = pl.col("close").diff()
        ret_l = ret.shift(self.lag)
        df = df.with_columns([ret.alias("_r"), ret_l.alias("_rl")])
        df = df.with_columns([
            (pl.col("_r") * pl.col("_rl")).rolling_mean(self.window).alias("_xy"),
            pl.col("_r").rolling_mean(self.window).alias("_xmean"),
            pl.col("_rl").rolling_mean(self.window).alias("_ymean"),
            (pl.col("_r") ** 2).rolling_mean(self.window).alias("_xx"),
            (pl.col("_rl") ** 2).rolling_mean(self.window).alias("_yy"),
        ])
        denom = ((pl.col("_xx") - pl.col("_xmean") ** 2) * (pl.col("_yy") - pl.col("_ymean") ** 2) + 1e-18).sqrt()
        return df.with_columns(
            ((pl.col("_xy") - pl.col("_xmean") * pl.col("_ymean")) / denom).alias(out)
        ).drop(["_r", "_rl", "_xy", "_xmean", "_ymean", "_xx", "_yy"])


@dataclass
class VolatilityClusterScore(Feature):
    """Rolling autocorr(|returns|) at lag=1. GARCH/vol-clustering proxy."""
    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["vol_cluster_{window}"]
    window: int = 480

    def compute_pair(self, df):
        out = f"vol_cluster_{self.window}"
        a = pl.col("close").diff().abs().alias("_aret")
        df = df.with_columns(a)
        x = pl.col("_aret"); xl = x.shift(1)
        df = df.with_columns([
            (x * xl).rolling_mean(self.window).alias("_xy"),
            x.rolling_mean(self.window).alias("_xm"),
            xl.rolling_mean(self.window).alias("_ym"),
            (x ** 2).rolling_mean(self.window).alias("_xx"),
            (xl ** 2).rolling_mean(self.window).alias("_yy"),
        ])
        denom = ((pl.col("_xx") - pl.col("_xm") ** 2) * (pl.col("_yy") - pl.col("_ym") ** 2) + 1e-18).sqrt()
        return df.with_columns(
            ((pl.col("_xy") - pl.col("_xm") * pl.col("_ym")) / denom).alias(out)
        ).drop(["_aret", "_xy", "_xm", "_ym", "_xx", "_yy"])


@dataclass
class ErrorAutoCorrelation(Feature):
    """Rolling autocorr of (close - SMA(period)) at given lag. Mean-reversion of error."""
    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["err_acf_{period}_{lag}_{window}"]
    period: int = 240
    lag: int = 30
    window: int = 960

    def compute_pair(self, df):
        out = f"err_acf_{self.period}_{self.lag}_{self.window}"
        c = pl.col("close")
        err = (c - c.rolling_mean(self.period)).alias("_e")
        df = df.with_columns(err)
        e = pl.col("_e"); el = e.shift(self.lag)
        df = df.with_columns([
            (e * el).rolling_mean(self.window).alias("_xy"),
            e.rolling_mean(self.window).alias("_xm"),
            el.rolling_mean(self.window).alias("_ym"),
            (e ** 2).rolling_mean(self.window).alias("_xx"),
            (el ** 2).rolling_mean(self.window).alias("_yy"),
        ])
        denom = ((pl.col("_xx") - pl.col("_xm") ** 2) * (pl.col("_yy") - pl.col("_ym") ** 2) + 1e-18).sqrt()
        return df.with_columns(
            ((pl.col("_xy") - pl.col("_xm") * pl.col("_ym")) / denom).alias(out)
        ).drop(["_e", "_xy", "_xm", "_ym", "_xx", "_yy"])

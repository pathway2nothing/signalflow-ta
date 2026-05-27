"""Distribution moment / tail features over rolling returns.

Added from sf-profit iter-20: return skew, kurtosis, tail-ratio variants.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar

import polars as pl

from signalflow.feature.base import Feature


@dataclass
class ReturnSkewWindow(Feature):
    """Skewness of returns in rolling window."""
    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["ret_skew_{window}"]
    window: int = 480

    def compute_pair(self, df):
        out = f"ret_skew_{self.window}"
        r = pl.col("close").diff()
        m = r.rolling_mean(self.window)
        s = r.rolling_std(self.window)
        cubed_mean = ((r - m) ** 3).rolling_mean(self.window)
        return df.with_columns((cubed_mean / (s.pow(3) + 1e-12)).alias(out))


@dataclass
class ReturnKurtosisWindow(Feature):
    """Excess kurtosis of returns. Heavy-tail detector."""
    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["ret_kurt_{window}"]
    window: int = 480

    def compute_pair(self, df):
        out = f"ret_kurt_{self.window}"
        r = pl.col("close").diff()
        m = r.rolling_mean(self.window)
        s = r.rolling_std(self.window)
        fourth = ((r - m) ** 4).rolling_mean(self.window)
        return df.with_columns((fourth / (s.pow(4) + 1e-12) - 3.0).alias(out))


@dataclass
class ReturnTailRatio(Feature):
    """q95(|return|) / q50(|return|) over window. Right-tail concentration."""
    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["ret_tail_ratio_{window}"]
    window: int = 480

    def compute_pair(self, df):
        out = f"ret_tail_ratio_{self.window}"
        a = pl.col("close").diff().abs()
        q95 = a.rolling_quantile(0.95, window_size=self.window)
        q50 = a.rolling_quantile(0.50, window_size=self.window)
        return df.with_columns((q95 / (q50 + 1e-9)).alias(out))

"""Range-based microstructure features."""
from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar

import polars as pl

from signalflow.feature.base import Feature


@dataclass
class RangeExpansion(Feature):
    """(high - low) / SMA(high - low, window) - 1. Range expansion intensity."""
    requires: ClassVar[list[str]] = ["high", "low"]
    outputs: ClassVar[list[str]] = ["range_expansion_{window}"]
    window: int = 240

    def compute_pair(self, df):
        out = f"range_expansion_{self.window}"
        rng = pl.col("high") - pl.col("low")
        return df.with_columns((rng / (rng.rolling_mean(self.window) + 1e-9) - 1).alias(out))


@dataclass
class RangeNormalizedReturn(Feature):
    """(close - close_prev) / range_prev_bar, rolling mean."""
    requires: ClassVar[list[str]] = ["high", "low", "close"]
    outputs: ClassVar[list[str]] = ["range_norm_ret_{window}"]
    window: int = 240

    def compute_pair(self, df):
        out = f"range_norm_ret_{self.window}"
        h, l, c = pl.col("high"), pl.col("low"), pl.col("close")
        prev_range = (h - l).shift(1)
        norm_ret = c.diff() / (prev_range + 1e-9)
        return df.with_columns(norm_ret.rolling_mean(self.window).alias(out))


@dataclass
class RangeFragmentation(Feature):
    """sum(|return|) / (rolling_max(high) - rolling_min(low)). Path-to-net distance."""
    requires: ClassVar[list[str]] = ["close", "high", "low"]
    outputs: ClassVar[list[str]] = ["range_frag_{window}"]
    window: int = 240

    def compute_pair(self, df):
        out = f"range_frag_{self.window}"
        c, h, l = pl.col("close"), pl.col("high"), pl.col("low")
        total_path = c.diff().abs().rolling_sum(self.window)
        rng = h.rolling_max(self.window) - l.rolling_min(self.window)
        return df.with_columns((total_path / (rng + 1e-9)).alias(out))


@dataclass
class HighLowImbalance(Feature):
    """((close - rolling_min) - (rolling_max - close)) / range. In [-1, +1]: signed position in recent range."""
    requires: ClassVar[list[str]] = ["close", "high", "low"]
    outputs: ClassVar[list[str]] = ["hl_imb_{window}"]
    window: int = 240

    def compute_pair(self, df):
        out = f"hl_imb_{self.window}"
        c, h, l = pl.col("close"), pl.col("high"), pl.col("low")
        rmax = h.rolling_max(self.window)
        rmin = l.rolling_min(self.window)
        rng = rmax - rmin
        return df.with_columns((((c - rmin) - (rmax - c)) / (rng + 1e-9)).alias(out))


@dataclass
class SignedRollingRange(Feature):
    """((close - rolling_min(low)) - (rolling_max(high) - close)) / atr. Signed position by ATR."""
    requires: ClassVar[list[str]] = ["high", "low", "close"]
    outputs: ClassVar[list[str]] = ["signed_range_{window}"]
    window: int = 240

    def compute_pair(self, df):
        out = f"signed_range_{self.window}"
        h, l, c = pl.col("high"), pl.col("low"), pl.col("close")
        roll_max = h.rolling_max(self.window)
        roll_min = l.rolling_min(self.window)
        tr = pl.max_horizontal(h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs())
        atr = tr.rolling_mean(self.window)
        signed = ((c - roll_min) - (roll_max - c)) / (atr + 1e-9)
        return df.with_columns(signed.alias(out))

"""Candle-shape microstructure features.

Captures information from per-bar geometry (body, wicks, close position) rolled over a window.
"""
from dataclasses import dataclass
from typing import ClassVar

import polars as pl

from signalflow.ta._compat import Feature


@dataclass
class WickToBodyRatio(Feature):
    """Avg over window of (upper_wick + lower_wick) / |body|. Indecision proxy."""
    requires: ClassVar[list[str]] = ["open", "high", "low", "close"]
    outputs: ClassVar[list[str]] = ["wick_body_{window}"]
    window: int = 240

    def compute_pair(self, df):
        out = f"wick_body_{self.window}"
        o, h, l, c = pl.col("open"), pl.col("high"), pl.col("low"), pl.col("close")
        upper = h - pl.max_horizontal(o, c)
        lower = pl.min_horizontal(o, c) - l
        body = (c - o).abs()
        return df.with_columns(((upper + lower) / (body + 1e-9)).rolling_mean(self.window).alias(out))


@dataclass
class WickToBodyUpper(Feature):
    """Upper wick relative to body, averaged over window."""
    requires: ClassVar[list[str]] = ["open", "high", "close"]
    outputs: ClassVar[list[str]] = ["wick_upper_{window}"]
    window: int = 240

    def compute_pair(self, df):
        out = f"wick_upper_{self.window}"
        o, h, c = pl.col("open"), pl.col("high"), pl.col("close")
        upper = h - pl.max_horizontal(o, c)
        body = (c - o).abs()
        return df.with_columns((upper / (body + 1e-9)).rolling_mean(self.window).alias(out))


@dataclass
class WickToBodyLower(Feature):
    """Lower wick relative to body, averaged over window."""
    requires: ClassVar[list[str]] = ["open", "low", "close"]
    outputs: ClassVar[list[str]] = ["wick_lower_{window}"]
    window: int = 240

    def compute_pair(self, df):
        out = f"wick_lower_{self.window}"
        o, l, c = pl.col("open"), pl.col("low"), pl.col("close")
        lower = pl.min_horizontal(o, c) - l
        body = (c - o).abs()
        return df.with_columns((lower / (body + 1e-9)).rolling_mean(self.window).alias(out))


@dataclass
class WickAsymmetry(Feature):
    """(upper - lower) / (upper + lower). Signed in [-1, +1]. Directional rejection bias."""
    requires: ClassVar[list[str]] = ["open", "high", "low", "close"]
    outputs: ClassVar[list[str]] = ["wick_asym_{window}"]
    window: int = 240

    def compute_pair(self, df):
        out = f"wick_asym_{self.window}"
        o, h, l, c = pl.col("open"), pl.col("high"), pl.col("low"), pl.col("close")
        upper = h - pl.max_horizontal(o, c)
        lower = pl.min_horizontal(o, c) - l
        return df.with_columns(((upper - lower) / (upper + lower + 1e-9)).rolling_mean(self.window).alias(out))


@dataclass
class BodyToRangeRatio(Feature):
    """|body| / (high - low). 0 = doji, 1 = full marubozu. Rolling mean."""
    requires: ClassVar[list[str]] = ["open", "high", "low", "close"]
    outputs: ClassVar[list[str]] = ["body_range_{window}"]
    window: int = 240

    def compute_pair(self, df):
        out = f"body_range_{self.window}"
        o, h, l, c = pl.col("open"), pl.col("high"), pl.col("low"), pl.col("close")
        body = (c - o).abs()
        rng = h - l
        return df.with_columns((body / (rng + 1e-9)).rolling_mean(self.window).alias(out))


@dataclass
class ClosePositionInBar(Feature):
    """(close - low) / (high - low) ∈ [0, 1]. 0 = bearish close, 1 = bullish. Rolling mean."""
    requires: ClassVar[list[str]] = ["high", "low", "close"]
    outputs: ClassVar[list[str]] = ["close_pos_{window}"]
    window: int = 240

    def compute_pair(self, df):
        out = f"close_pos_{self.window}"
        h, l, c = pl.col("high"), pl.col("low"), pl.col("close")
        pos = (c - l) / (h - l + 1e-9)
        return df.with_columns(pos.rolling_mean(self.window).alias(out))


@dataclass
class UpperWickPersistence(Feature):
    """Fraction of bars in window where upper_wick > |body|. Persistent rejection at top."""
    requires: ClassVar[list[str]] = ["open", "high", "close"]
    outputs: ClassVar[list[str]] = ["upper_wick_persist_{window}"]
    window: int = 240

    def compute_pair(self, df):
        out = f"upper_wick_persist_{self.window}"
        o, h, c = pl.col("open"), pl.col("high"), pl.col("close")
        upper = h - pl.max_horizontal(o, c)
        body = (c - o).abs()
        flag = (upper > body).cast(pl.Float32)
        return df.with_columns(flag.rolling_mean(self.window).alias(out))


@dataclass
class LowerWickPersistence(Feature):
    """Fraction of bars where lower_wick > |body|. Persistent rejection at bottom."""
    requires: ClassVar[list[str]] = ["open", "low", "close"]
    outputs: ClassVar[list[str]] = ["lower_wick_persist_{window}"]
    window: int = 240

    def compute_pair(self, df):
        out = f"lower_wick_persist_{self.window}"
        o, l, c = pl.col("open"), pl.col("low"), pl.col("close")
        lower = pl.min_horizontal(o, c) - l
        body = (c - o).abs()
        flag = (lower > body).cast(pl.Float32)
        return df.with_columns(flag.rolling_mean(self.window).alias(out))


@dataclass
class HiLoMedianGap(Feature):
    """(high + low) / 2 - close, rolling mean. Composes median-of-bar with close.

    Captures whether close drifts above/below the bar midpoint persistently.
    """
    requires: ClassVar[list[str]] = ["high", "low", "close"]
    outputs: ClassVar[list[str]] = ["hi_lo_median_gap_{window}"]
    window: int = 240

    def compute_pair(self, df):
        out = f"hi_lo_median_gap_{self.window}"
        h, l, c = pl.col("high"), pl.col("low"), pl.col("close")
        gap = (h + l) / 2 - c
        return df.with_columns(gap.rolling_mean(self.window).alias(out))

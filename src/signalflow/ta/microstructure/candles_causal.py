"""Warmup-invariant variants of candle-shape features.

The originals in :mod:`signalflow.ta.microstructure.candles` use
``polars.rolling_mean`` whose optimised prefix-sum implementation produces
float64 round-off at the ~1e-7 level when the input series starts at
different points. These variants use a strict numpy implementation
(``sliding_window_view`` + per-window mean) that recomputes every window
from scratch, giving bit-identical output regardless of input start.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import polars as pl

from signalflow.feature.base import Feature
from signalflow.ta.stat._causal_helpers import rolling_mean


@dataclass
class WickToBodyRatioCausal(Feature):
    """Warmup-invariant variant of :class:`WickToBodyRatio`."""
    requires: ClassVar[list[str]] = ["open", "high", "low", "close"]
    outputs: ClassVar[list[str]] = ["wick_body_causal_{window}"]
    test_params: ClassVar[list[dict]] = [{"window": 240}, {"window": 480}, {"window": 960}]
    window: int = 240

    def compute_pair(self, df):
        out_col = f"wick_body_causal_{self.window}"
        o = df["open"].to_numpy().astype(np.float64)
        h = df["high"].to_numpy().astype(np.float64)
        l = df["low"].to_numpy().astype(np.float64)
        c = df["close"].to_numpy().astype(np.float64)
        upper = h - np.maximum(o, c)
        lower = np.minimum(o, c) - l
        body = np.abs(c - o)
        per_bar = (upper + lower) / (body + 1e-9)
        out = rolling_mean(per_bar, self.window)
        return df.with_columns(pl.Series(out_col, out, dtype=pl.Float64))


@dataclass
class WickToBodyUpperCausal(Feature):
    """Warmup-invariant variant of :class:`WickToBodyUpper`."""
    requires: ClassVar[list[str]] = ["open", "high", "close"]
    outputs: ClassVar[list[str]] = ["wick_upper_causal_{window}"]
    test_params: ClassVar[list[dict]] = [{"window": 240}, {"window": 480}, {"window": 960}]
    window: int = 240

    def compute_pair(self, df):
        out_col = f"wick_upper_causal_{self.window}"
        o = df["open"].to_numpy().astype(np.float64)
        h = df["high"].to_numpy().astype(np.float64)
        c = df["close"].to_numpy().astype(np.float64)
        upper = h - np.maximum(o, c)
        body = np.abs(c - o)
        per_bar = upper / (body + 1e-9)
        out = rolling_mean(per_bar, self.window)
        return df.with_columns(pl.Series(out_col, out, dtype=pl.Float64))


@dataclass
class WickToBodyLowerCausal(Feature):
    """Warmup-invariant variant of :class:`WickToBodyLower`."""
    requires: ClassVar[list[str]] = ["open", "low", "close"]
    outputs: ClassVar[list[str]] = ["wick_lower_causal_{window}"]
    test_params: ClassVar[list[dict]] = [{"window": 240}, {"window": 480}, {"window": 960}]
    window: int = 240

    def compute_pair(self, df):
        out_col = f"wick_lower_causal_{self.window}"
        o = df["open"].to_numpy().astype(np.float64)
        l = df["low"].to_numpy().astype(np.float64)
        c = df["close"].to_numpy().astype(np.float64)
        lower = np.minimum(o, c) - l
        body = np.abs(c - o)
        per_bar = lower / (body + 1e-9)
        out = rolling_mean(per_bar, self.window)
        return df.with_columns(pl.Series(out_col, out, dtype=pl.Float64))

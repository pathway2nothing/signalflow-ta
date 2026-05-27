"""Brownian bridge tension: maximum deviation from linear interpolation.

Given a price window with known start P_0 and end P_T, the maximum
deviation max_t |P_t - (P_0 + (t/T)*(P_T - P_0))| measures how much the
path "tensions" away from the natural Brownian bridge — direct path from
start to end. A perfect Brownian bridge has Hurst-like scaling; large
deviations indicate one-directional excursions, crowding, or structural
breaks.

Empirically validated in iter-27 of sf-profit (target encoding research):
mean MI_normalised across 6 walk-forward folds = 0.11, std 0.007 — most
stable feature in the iteration (lowest std-to-mean ratio).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import polars as pl
from numpy.lib.stride_tricks import sliding_window_view

from signalflow.core import feature
from signalflow.feature.base import Feature


@dataclass
@feature("stat/brownian_bridge_tension")
class BrownianBridgeTensionStat(Feature):
    """Max deviation from linear interp(start, end) / window std.

    Algorithm:
        For each window of length `period`:
            1. linear path L_i = P_0 + (i/(N-1)) * (P_{N-1} - P_0)
            2. deviation d_i = |P_i - L_i|
            3. tension = max(d) / std(P)

    Captures "tension" or "bowing" of price excursions within a window:
        - tension ≈ 1: typical Brownian-bridge fluctuation
        - tension >> 1: extreme one-directional excursion
        - tension << 1: very straight path

    Attributes:
        period: window length in bars.
        source_col: price column.
    """

    period: int = 60
    source_col: str = "close"

    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["bb_tension_{period}"]
    test_params: ClassVar[list[dict]] = [
        {"period": 60},
        {"period": 240},
        {"period": 480},
    ]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        out_col = f"bb_tension_{self.period}"
        x = df[self.source_col].to_numpy().astype(np.float64)
        n = len(x)
        w = int(self.period)
        if n < w:
            return df.with_columns(pl.lit(np.nan).alias(out_col))
        wins = sliding_window_view(x, w)
        starts = wins[:, 0:1]
        ends = wins[:, -1:]
        t = np.linspace(0, 1, w)[None, :]
        linear = starts + (ends - starts) * t
        dev = np.abs(wins - linear)
        max_dev = dev.max(axis=1)
        sd = wins.std(axis=1) + 1e-12
        tension = max_dev / sd
        pad = np.full(w - 1, np.nan, dtype=np.float64)
        return df.with_columns(pl.Series(out_col, np.concatenate([pad, tension]), dtype=pl.Float64))


@dataclass
@feature("stat/bb_path_roughness")
class BBPathRoughnessStat(Feature):
    """Sum(|Δprice|) / |end − start| over window — tortuosity of the path.

    Ratio of accumulated path length to straight-line displacement. Value 1
    means perfectly straight movement (low roughness); large values mean
    high oscillation within the window.

    Iter-28 stability: mean MI_normalised = 0.275 (#3 in iter-28) with
    std 0.014 across 12 stable triples.
    """

    period: int = 60
    source_col: str = "close"

    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["bb_roughness_{period}"]
    test_params: ClassVar[list[dict]] = [
        {"period": 60},
        {"period": 240},
        {"period": 480},
    ]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        out_col = f"bb_roughness_{self.period}"
        x = df[self.source_col].to_numpy().astype(np.float64)
        n = len(x)
        w = int(self.period)
        if n < w:
            return df.with_columns(pl.lit(np.nan).alias(out_col))
        diffs = np.abs(np.diff(x))
        wins_d = sliding_window_view(diffs, w - 1)
        path_len = wins_d.sum(axis=1)
        direct = np.abs(x[w - 1:] - x[: n - w + 1]) + 1e-12
        roughness = path_len / direct
        pad = np.full(w - 1, np.nan, dtype=np.float64)
        return df.with_columns(pl.Series(out_col, np.concatenate([pad, roughness]), dtype=pl.Float64))


@dataclass
@feature("stat/bb_tension_directional")
class BBTensionDirectionalStat(Feature):
    """Signed BB tension — positive if max deviation above linear, negative if below.

    Iter-28 stability: mean MI_normalised = 0.106 with std 0.024 across 4
    stable triples.
    """

    period: int = 60
    source_col: str = "close"

    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["bb_tension_dir_{period}"]
    test_params: ClassVar[list[dict]] = [
        {"period": 60},
        {"period": 240},
        {"period": 480},
    ]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        out_col = f"bb_tension_dir_{self.period}"
        x = df[self.source_col].to_numpy().astype(np.float64)
        n = len(x)
        w = int(self.period)
        if n < w:
            return df.with_columns(pl.lit(np.nan).alias(out_col))
        wins = sliding_window_view(x, w)
        starts = wins[:, 0:1]
        ends = wins[:, -1:]
        t = np.linspace(0, 1, w)[None, :]
        linear = starts + (ends - starts) * t
        dev = wins - linear
        abs_dev = np.abs(dev)
        max_idx = np.argmax(abs_dev, axis=1)
        signed_max = dev[np.arange(len(dev)), max_idx]
        sd = wins.std(axis=1) + 1e-12
        pad = np.full(w - 1, np.nan, dtype=np.float64)
        return df.with_columns(pl.Series(out_col, np.concatenate([pad, signed_max / sd]), dtype=pl.Float64))


@dataclass
@feature("stat/swing_amp_displacement")
class SwingAmpDisplacementStat(Feature):
    """sum(high − low) over window / |close_end − close_start| — micro-swing tortuosity.

    Range-based path-tortuosity variant. Counts cumulative high-low
    excursions per unit of net displacement; high values = much chop,
    little net move.

    Iter-29 stability: mean MI_normalised = 0.266 on D3, std 0.012 across
    12 stable triples.
    """

    period: int = 15

    requires: ClassVar[list[str]] = ["high", "low", "close"]
    outputs: ClassVar[list[str]] = ["f24_swing_disp_{period}"]
    test_params: ClassVar[list[dict]] = [
        {"period": 15},
        {"period": 60},
        {"period": 240},
    ]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        out_col = f"f24_swing_disp_{self.period}"
        c = df["close"].to_numpy().astype(np.float64)
        h = df["high"].to_numpy().astype(np.float64)
        l = df["low"].to_numpy().astype(np.float64)
        n = len(c); w = int(self.period)
        if n < w:
            return df.with_columns(pl.lit(np.nan).alias(out_col))
        wins = sliding_window_view(h - l, w)
        swings = wins.sum(axis=1)
        disp = np.abs(c[w - 1:] - c[: n - w + 1]) + 1e-12
        out = swings / disp
        pad = np.full(w - 1, np.nan, dtype=np.float64)
        return df.with_columns(pl.Series(out_col, np.concatenate([pad, out]), dtype=pl.Float64))

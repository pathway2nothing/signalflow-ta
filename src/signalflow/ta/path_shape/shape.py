"""Path shape descriptors: roughness, efficiency, tortuosity, simplicity."""
from dataclasses import dataclass
from typing import ClassVar

import polars as pl

from signalflow.ta._compat import Feature


@dataclass
class PathRoughness(Feature):
    """std(|returns|) / mean(|returns|) - coefficient of variation of move size."""
    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["roughness_{window}"]
    window: int = 240

    def compute_pair(self, df):
        out = f"roughness_{self.window}"
        absret = pl.col("close").diff().abs()
        return df.with_columns(
            (absret.rolling_std(self.window) / (absret.rolling_mean(self.window) + 1e-12)).alias(out)
        )


@dataclass
class PathEfficiency(Feature):
    """Kaufman's Efficiency Ratio: net move / total path length ∈ [-1, +1]."""
    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["path_eff_{window}"]
    window: int = 240

    def compute_pair(self, df):
        out = f"path_eff_{self.window}"
        c = pl.col("close")
        net = c - c.shift(self.window)
        total = c.diff().abs().rolling_sum(self.window)
        return df.with_columns((net / (total + 1e-9)).alias(out))


@dataclass
class PathTortuosity(Feature):
    """sum(|diff close|) / (max - min). Path length per net displacement."""
    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["tortuosity_{window}"]
    window: int = 240

    def compute_pair(self, df):
        out = f"tortuosity_{self.window}"
        c = pl.col("close")
        path = c.diff().abs().rolling_sum(self.window)
        spread = c.rolling_max(self.window) - c.rolling_min(self.window)
        return df.with_columns((path / (spread + 1e-9)).alias(out))


@dataclass
class PathSimplicity(Feature):
    """|net_return| / sum_abs_returns ∈ [0, 1]. High = directional move; low = back-and-forth."""
    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["simplicity_{window}"]
    window: int = 240

    def compute_pair(self, df):
        out = f"simplicity_{self.window}"
        d = pl.col("close").diff()
        net = d.rolling_sum(self.window).abs()
        path = d.abs().rolling_sum(self.window)
        return df.with_columns((net / (path + 1e-9)).alias(out))

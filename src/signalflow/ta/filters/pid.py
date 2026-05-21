"""PID-controller terms: integral and derivative of close-vs-SMA error."""
from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import polars as pl

from signalflow.feature.base import Feature


@dataclass
class PIDIntegralTerm(Feature):
    """Rolling sum of (close − SMA(period)) over window, sqrt-normalized.

    PID-I component: integrated control error. Captures persistent
    over/under-shoot of price relative to its filter.
    """
    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["pid_i_{period}_{window}"]
    period: int = 240
    window: int = 1440

    def compute_pair(self, df):
        out = f"pid_i_{self.period}_{self.window}"
        c = pl.col("close")
        err = c - c.rolling_mean(self.period)
        return df.with_columns(
            (err.rolling_sum(self.window) / np.sqrt(self.window)).alias(out)
        )


@dataclass
class PIDDerivativeTerm(Feature):
    """Derivative of (close − SMA(period)): rate of error change.

    PID-D component: how fast the error is evolving.
    """
    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["pid_d_{period}_{lag}"]
    period: int = 240
    lag: int = 30

    def compute_pair(self, df):
        out = f"pid_d_{self.period}_{self.lag}"
        c = pl.col("close")
        err = c - c.rolling_mean(self.period)
        return df.with_columns((err - err.shift(self.lag)).alias(out))

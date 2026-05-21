"""Errors between close and various MA-family smoothers (EMA, DEMA, TEMA, HMA)."""
from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import polars as pl

from signalflow.feature.base import Feature


@dataclass
class AdaptiveEMAError(Feature):
    """close − EMA(period). alpha = 2/(period+1). PID-P proxy."""
    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["ema_err_{period}"]
    period: int = 240

    def compute_pair(self, df):
        out = f"ema_err_{self.period}"
        c = pl.col("close")
        alpha = 2.0 / (self.period + 1)
        ema = c.ewm_mean(alpha=alpha, adjust=False)
        return df.with_columns((c - ema).alias(out))


@dataclass
class DEMAError(Feature):
    """close − DEMA(period). DEMA = 2*EMA − EMA(EMA). Less lag than EMA."""
    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["dema_err_{period}"]
    period: int = 240

    def compute_pair(self, df):
        out = f"dema_err_{self.period}"
        c = pl.col("close")
        a = 2.0 / (self.period + 1)
        ema = c.ewm_mean(alpha=a, adjust=False)
        ema2 = ema.ewm_mean(alpha=a, adjust=False)
        dema = 2 * ema - ema2
        return df.with_columns((c - dema).alias(out))


@dataclass
class TEMAError(Feature):
    """close − TEMA(period). TEMA = 3*EMA − 3*EMA(EMA) + EMA(EMA(EMA))."""
    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["tema_err_{period}"]
    period: int = 240

    def compute_pair(self, df):
        out = f"tema_err_{self.period}"
        c = pl.col("close")
        a = 2.0 / (self.period + 1)
        e1 = c.ewm_mean(alpha=a, adjust=False)
        e2 = e1.ewm_mean(alpha=a, adjust=False)
        e3 = e2.ewm_mean(alpha=a, adjust=False)
        tema = 3 * e1 - 3 * e2 + e3
        return df.with_columns((c - tema).alias(out))


@dataclass
class HMAError(Feature):
    """close − HMA(period). HMA proxy via 2*SMA(half) − SMA(period), smoothed by SMA(sqrt(period))."""
    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["hma_err_{period}"]
    period: int = 240

    def compute_pair(self, df):
        out = f"hma_err_{self.period}"
        c = pl.col("close")
        half = max(2, self.period // 2)
        sq = max(2, int(np.sqrt(self.period)))
        wma_h = c.rolling_mean(half)
        wma_f = c.rolling_mean(self.period)
        raw = 2 * wma_h - wma_f
        hma = raw.rolling_mean(sq)
        return df.with_columns((c - hma).alias(out))

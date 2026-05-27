"""Extended momentum features: directional confirmation, higher-order, normalized.

Added from sf-profit feature_research_lib + iter-15/16/18/20.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar

import polars as pl

from signalflow.feature.base import Feature


@dataclass
class RocSignedLog(Feature):
    """sign(ROC) * log1p(|ROC|). Bounded directional momentum."""
    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["roc_signed_log_{period}"]
    period: int = 60

    def compute_pair(self, df):
        out = f"roc_signed_log_{self.period}"
        c = pl.col("close")
        roc = (c / c.shift(self.period) - 1) * 100.0
        return df.with_columns((roc.sign() * (roc.abs() + 1).log()).alias(out))


@dataclass
class MomPosNeg(Feature):
    """Two one-sided momentum z-scores: positive part and absolute negative part."""
    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["mom_pos_{period}", "mom_neg_{period}"]
    period: int = 60

    def compute_pair(self, df):
        c = pl.col("close")
        roc = (c / c.shift(self.period) - 1) * 100.0
        std = roc.rolling_std(self.period * 5)
        return df.with_columns(
            (pl.when(roc > 0).then(roc).otherwise(0.0) / (std + 1e-9)).alias(f"mom_pos_{self.period}"),
            (pl.when(roc < 0).then(-roc).otherwise(0.0) / (std + 1e-9)).alias(f"mom_neg_{self.period}"),
        )


@dataclass
class MacdNorm(Feature):
    """MACD normalized by ATR: (EMA_fast − EMA_slow) / ATR. Scale-invariant MACD."""
    requires: ClassVar[list[str]] = ["high", "low", "close"]
    outputs: ClassVar[list[str]] = ["macd_norm_{fast}_{slow}_{atr_period}"]
    fast: int = 12
    slow: int = 26
    atr_period: int = 120

    def compute_pair(self, df):
        out = f"macd_norm_{self.fast}_{self.slow}_{self.atr_period}"
        c = pl.col("close")
        macd = c.ewm_mean(span=self.fast, adjust=False) - c.ewm_mean(span=self.slow, adjust=False)
        tr = pl.max_horizontal(
            pl.col("high") - pl.col("low"),
            (pl.col("high") - c.shift(1)).abs(),
            (pl.col("low") - c.shift(1)).abs(),
        )
        atr = tr.rolling_mean(self.atr_period)
        return df.with_columns((macd / (atr + 1e-9)).alias(out))


@dataclass
class RsiSpread(Feature):
    """RSI_short − RSI_long. Multi-timeframe RSI spread."""
    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["rsi_spread_{short}_{long}"]
    short: int = 60
    long: int = 960

    def compute_pair(self, df):
        out = f"rsi_spread_{self.short}_{self.long}"
        diff = pl.col("close").diff()
        gain = pl.when(diff > 0).then(diff).otherwise(0.0)
        loss = pl.when(diff < 0).then(-diff).otherwise(0.0)
        rsi_s = 100 - 100 / (1 + gain.rolling_mean(self.short) / (loss.rolling_mean(self.short) + 1e-9))
        rsi_l = 100 - 100 / (1 + gain.rolling_mean(self.long) / (loss.rolling_mean(self.long) + 1e-9))
        return df.with_columns((rsi_s - rsi_l).alias(out))


@dataclass
class PriceMomentumConfirmation(Feature):
    """sign(ret_short) × sign(ret_medium). +1 = aligned, -1 = conflicting."""
    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["mom_conf_{short}_{medium}"]
    short: int = 60
    medium: int = 480

    def compute_pair(self, df):
        out = f"mom_conf_{self.short}_{self.medium}"
        c = pl.col("close")
        ret_s = (c / c.shift(self.short) - 1)
        ret_m = (c / c.shift(self.medium) - 1)
        return df.with_columns((ret_s.sign() * ret_m.sign()).alias(out))


@dataclass
class VolPriceConfirmation(Feature):
    """sign(price_change) × sign(vol_change). +1 = vol confirms, -1 = divergence."""
    requires: ClassVar[list[str]] = ["close", "volume"]
    outputs: ClassVar[list[str]] = ["volprice_conf_{period}"]
    period: int = 240

    def compute_pair(self, df):
        out = f"volprice_conf_{self.period}"
        c = pl.col("close")
        v = pl.col("volume")
        pc = (c / c.shift(self.period) - 1).sign()
        vc = (v.rolling_mean(self.period // 2) / v.rolling_mean(self.period) - 1).sign()
        return df.with_columns((pc * vc).alias(out))


@dataclass
class TrendPersistence(Feature):
    """Rolling fraction of bars with SMA_short > SMA_long over period."""
    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["trend_persist_{short}_{long}_{period}"]
    short: int = 60
    long: int = 480
    period: int = 1440

    def compute_pair(self, df):
        out = f"trend_persist_{self.short}_{self.long}_{self.period}"
        c = pl.col("close")
        above = (c.rolling_mean(self.short) > c.rolling_mean(self.long)).cast(pl.Float32)
        return df.with_columns(above.rolling_mean(self.period).alias(out))


@dataclass
class PriceAcceleration(Feature):
    """Second derivative: diff(diff(close), lag). Smoothed price curvature."""
    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["price_accel_{lag}_{window}"]
    lag: int = 60
    window: int = 480

    def compute_pair(self, df):
        out = f"price_accel_{self.lag}_{self.window}"
        c = pl.col("close")
        d = c.diff(self.lag)
        accel = d - d.shift(self.lag)
        return df.with_columns(accel.rolling_mean(self.window).alias(out))


@dataclass
class MomentumOfMomentum(Feature):
    """diff of ROC: ROC_t − ROC_{t-lag}. Trend acceleration."""
    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["mom_of_mom_{period}_{lag}"]
    period: int = 240
    lag: int = 60

    def compute_pair(self, df):
        out = f"mom_of_mom_{self.period}_{self.lag}"
        c = pl.col("close")
        roc = (c / c.shift(self.period) - 1)
        return df.with_columns((roc - roc.shift(self.lag)).alias(out))

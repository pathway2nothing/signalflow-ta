"""Extended volatility features: regime ratios, vol-of-vol, percentile ranks.

All scale-invariant or stationary transforms of existing ATR/Parkinson/GarmanKlass measures.
Added from sf-profit iter-3.1, iter-15, iter-16, iter-20 research.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import polars as pl

from signalflow.feature.base import Feature


def _tr(h, l, c):
    """True Range expression for high/low/close polars cols."""
    return pl.max_horizontal(h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs())


def _parkinson_vol(c_high, c_low, period):
    return ((c_high.log() - c_low.log()).pow(2).rolling_mean(period) / (4 * np.log(2))).sqrt()


@dataclass
class NatrRatio(Feature):
    """Volatility regime ratio: NATR_short / NATR_long. Captures vol regime shift."""
    requires: ClassVar[list[str]] = ["high", "low", "close"]
    outputs: ClassVar[list[str]] = ["natr_ratio_{short}_{long}"]
    short: int = 60
    long: int = 960

    def compute_pair(self, df):
        out = f"natr_ratio_{self.short}_{self.long}"
        h, l, c = pl.col("high"), pl.col("low"), pl.col("close")
        tr = _tr(h, l, c)
        return df.with_columns(
            atr_s=tr.rolling_mean(self.short),
            atr_l=tr.rolling_mean(self.long),
        ).with_columns(
            ((pl.col("atr_s") / c) / (pl.col("atr_l") / c + 1e-9)).alias(out)
        ).drop(["atr_s", "atr_l"])


@dataclass
class NatrPctRank(Feature):
    """Rolling pct rank of NATR over lookback. Stationary vol regime indicator (min-max)."""
    requires: ClassVar[list[str]] = ["high", "low", "close"]
    outputs: ClassVar[list[str]] = ["natr_pctrank_{period}_{lookback}"]
    period: int = 240
    lookback: int = 2880

    def compute_pair(self, df):
        out = f"natr_pctrank_{self.period}_{self.lookback}"
        h, l, c = pl.col("high"), pl.col("low"), pl.col("close")
        tr = _tr(h, l, c)
        natr = tr.rolling_mean(self.period) / (c + 1e-9)
        return df.with_columns((
            (natr - natr.rolling_min(self.lookback))
            / (natr.rolling_max(self.lookback) - natr.rolling_min(self.lookback) + 1e-9)
        ).alias(out))


@dataclass
class VolOfVol(Feature):
    """std(ATR_short, window) / mean(ATR_short, window). Coefficient of variation of short-term vol."""
    requires: ClassVar[list[str]] = ["high", "low", "close"]
    outputs: ClassVar[list[str]] = ["vol_of_vol_{short}_{window}"]
    short: int = 60
    window: int = 1440

    def compute_pair(self, df):
        out = f"vol_of_vol_{self.short}_{self.window}"
        h, l, c = pl.col("high"), pl.col("low"), pl.col("close")
        tr = _tr(h, l, c)
        atr = tr.rolling_mean(self.short)
        vov = atr.rolling_std(self.window) / (atr.rolling_mean(self.window) + 1e-9)
        return df.with_columns(vov.alias(out))


@dataclass
class RealizedVolPctRank(Feature):
    """Pct rank of realized vol (std of returns) over lookback."""
    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["rv_pctrank_{period}_{lookback}"]
    period: int = 240
    lookback: int = 2880

    def compute_pair(self, df):
        out = f"rv_pctrank_{self.period}_{self.lookback}"
        ret = pl.col("close").pct_change()
        rv = ret.rolling_std(self.period)
        return df.with_columns((
            (rv - rv.rolling_min(self.lookback))
            / (rv.rolling_max(self.lookback) - rv.rolling_min(self.lookback) + 1e-9)
        ).alias(out))


@dataclass
class RealizedVolRatio(Feature):
    """Realized vol ratio: std(ret, short) / std(ret, long). Vol regime shift indicator."""
    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["rv_ratio_{short}_{long}"]
    short: int = 60
    long: int = 960

    def compute_pair(self, df):
        out = f"rv_ratio_{self.short}_{self.long}"
        ret = pl.col("close").pct_change()
        return df.with_columns(
            (ret.rolling_std(self.short) / (ret.rolling_std(self.long) + 1e-9)).alias(out)
        )


@dataclass
class ParkinsonZScore(Feature):
    """ParkinsonVol z-score over lookback — vol regime deviation."""
    requires: ClassVar[list[str]] = ["high", "low"]
    outputs: ClassVar[list[str]] = ["pk_zscore_{period}_{lookback}"]
    period: int = 240
    lookback: int = 2880

    def compute_pair(self, df):
        out = f"pk_zscore_{self.period}_{self.lookback}"
        pv = _parkinson_vol(pl.col("high"), pl.col("low"), self.period)
        return df.with_columns(
            ((pv - pv.rolling_mean(self.lookback)) / (pv.rolling_std(self.lookback) + 1e-9)).alias(out)
        )


@dataclass
class ParkinsonAccel(Feature):
    """Parkinson vol acceleration: (PV_t − PV_{t-lag}) / PV_{t-lag}."""
    requires: ClassVar[list[str]] = ["high", "low"]
    outputs: ClassVar[list[str]] = ["parkinson_accel_{period}_{lag}"]
    period: int = 240
    lag: int = 240

    def compute_pair(self, df):
        out = f"parkinson_accel_{self.period}_{self.lag}"
        pv = _parkinson_vol(pl.col("high"), pl.col("low"), self.period)
        return df.with_columns(((pv - pv.shift(self.lag)) / (pv.shift(self.lag) + 1e-9)).alias(out))


@dataclass
class ParkinsonVolRatio(Feature):
    """Parkinson_short / Parkinson_long. Regime ratio."""
    requires: ClassVar[list[str]] = ["high", "low"]
    outputs: ClassVar[list[str]] = ["parkinson_ratio_{short}_{long}"]
    short: int = 120
    long: int = 960

    def compute_pair(self, df):
        out = f"parkinson_ratio_{self.short}_{self.long}"
        h, l = pl.col("high"), pl.col("low")
        return df.with_columns((
            _parkinson_vol(h, l, self.short) / (_parkinson_vol(h, l, self.long) + 1e-9)
        ).alias(out))


@dataclass
class AltVolDeviation(Feature):
    """ParkinsonVol_short − ParkinsonVol_long. Additive deviation preserves level info."""
    requires: ClassVar[list[str]] = ["high", "low"]
    outputs: ClassVar[list[str]] = ["pk_dev_{short}_{long}"]
    short: int = 60
    long: int = 960

    def compute_pair(self, df):
        out = f"pk_dev_{self.short}_{self.long}"
        h, l = pl.col("high"), pl.col("low")
        return df.with_columns((
            _parkinson_vol(h, l, self.short) - _parkinson_vol(h, l, self.long)
        ).alias(out))


@dataclass
class GarmanKlassRatio(Feature):
    """GarmanKlass_short / GarmanKlass_long."""
    requires: ClassVar[list[str]] = ["high", "low", "open", "close"]
    outputs: ClassVar[list[str]] = ["gk_ratio_{short}_{long}"]
    short: int = 120
    long: int = 960

    def compute_pair(self, df):
        out = f"gk_ratio_{self.short}_{self.long}"
        h, l, o, c = pl.col("high"), pl.col("low"), pl.col("open"), pl.col("close")
        gk_term = (0.5 * (h.log() - l.log()).pow(2)
                   - (2 * np.log(2) - 1) * (c.log() - o.log()).pow(2))
        gv_s = gk_term.rolling_mean(self.short).sqrt()
        gv_l = gk_term.rolling_mean(self.long).sqrt()
        return df.with_columns((gv_s / (gv_l + 1e-9)).alias(out))


@dataclass
class GarmanKlassPctRank(Feature):
    """Rolling pct rank of GarmanKlass vol over lookback."""
    requires: ClassVar[list[str]] = ["high", "low", "open", "close"]
    outputs: ClassVar[list[str]] = ["gk_pctrank_{period}_{lookback}"]
    period: int = 240
    lookback: int = 2880

    def compute_pair(self, df):
        out = f"gk_pctrank_{self.period}_{self.lookback}"
        h, l, o, c = pl.col("high"), pl.col("low"), pl.col("open"), pl.col("close")
        gk_term = (0.5 * (h.log() - l.log()).pow(2)
                   - (2 * np.log(2) - 1) * (c.log() - o.log()).pow(2))
        gv = gk_term.rolling_mean(self.period).sqrt()
        return df.with_columns((
            (gv - gv.rolling_min(self.lookback))
            / (gv.rolling_max(self.lookback) - gv.rolling_min(self.lookback) + 1e-9)
        ).alias(out))


@dataclass
class PriceZAtr(Feature):
    """(close − SMA_N) / ATR_N. Universal stationarity transform."""
    requires: ClassVar[list[str]] = ["high", "low", "close"]
    outputs: ClassVar[list[str]] = ["price_z_atr_{period}"]
    period: int = 60

    def compute_pair(self, df):
        out = f"price_z_atr_{self.period}"
        c = pl.col("close")
        sma = c.rolling_mean(self.period)
        tr = _tr(pl.col("high"), pl.col("low"), c)
        atr = tr.rolling_mean(self.period)
        return df.with_columns(((c - sma) / (atr + 1e-9)).alias(out))

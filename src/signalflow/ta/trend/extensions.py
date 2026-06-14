"""Extended trend features: directional balance, composites, divergence variants.

Added from sf-profit feature_research_lib + iter-15/16/18.
"""
from dataclasses import dataclass
from typing import ClassVar

import polars as pl

from signalflow.ta._compat import Feature


def _tr(h, l, c):
    return pl.max_horizontal(h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs())


@dataclass
class DiBalance(Feature):
    """(+DI − -DI) / (+DI + -DI). Directional balance ∈ [-1, +1]. Signed ADX-like."""
    requires: ClassVar[list[str]] = ["high", "low", "close"]
    outputs: ClassVar[list[str]] = ["di_balance_{period}"]
    period: int = 60

    def compute_pair(self, df):
        out = f"di_balance_{self.period}"
        up = pl.col("high") - pl.col("high").shift(1)
        dn = pl.col("low").shift(1) - pl.col("low")
        tr = _tr(pl.col("high"), pl.col("low"), pl.col("close"))
        plus_dm = pl.when((up > dn) & (up > 0)).then(up).otherwise(0.0)
        minus_dm = pl.when((dn > up) & (dn > 0)).then(dn).otherwise(0.0)
        atr = tr.rolling_mean(self.period)
        pdi = 100.0 * plus_dm.rolling_mean(self.period) / (atr + 1e-9)
        ndi = 100.0 * minus_dm.rolling_mean(self.period) / (atr + 1e-9)
        return df.with_columns(((pdi - ndi) / (pdi + ndi + 1e-9)).alias(out))


@dataclass
class NatrXDiBalance(Feature):
    """NATR × DI_balance. Vol-conditional directional strength."""
    requires: ClassVar[list[str]] = ["high", "low", "close"]
    outputs: ClassVar[list[str]] = ["natr_x_di_{natr_period}_{adx_period}"]
    natr_period: int = 240
    adx_period: int = 240

    def compute_pair(self, df):
        out = f"natr_x_di_{self.natr_period}_{self.adx_period}"
        h, l, c = pl.col("high"), pl.col("low"), pl.col("close")
        tr = _tr(h, l, c)
        atr = tr.rolling_mean(self.natr_period)
        natr = atr / (c + 1e-9)
        up = h - h.shift(1)
        dn = l.shift(1) - l
        plus_dm = pl.when((up > dn) & (up > 0)).then(up).otherwise(0.0)
        minus_dm = pl.when((dn > up) & (dn > 0)).then(dn).otherwise(0.0)
        atr_dx = tr.rolling_mean(self.adx_period)
        pdi = 100.0 * plus_dm.rolling_mean(self.adx_period) / (atr_dx + 1e-9)
        ndi = 100.0 * minus_dm.rolling_mean(self.adx_period) / (atr_dx + 1e-9)
        di_bal = (pdi - ndi) / (pdi + ndi + 1e-9)
        return df.with_columns((natr * di_bal).alias(out))


@dataclass
class UpDownEntropyAsymmetry(Feature):
    """Entropy of up-runs minus entropy of down-runs over period.

    Captures asymmetric path-shape regimes (bull vs bear path complexity).
    """
    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["ud_entropy_asym_{period}"]
    period: int = 480

    def compute_pair(self, df):
        out = f"ud_entropy_asym_{self.period}"
        ret = pl.col("close").diff()
        up_frac = (ret > 0).cast(pl.Float32).rolling_mean(self.period)
        dn_frac = (ret < 0).cast(pl.Float32).rolling_mean(self.period)
        eps = 1e-9
        ent_up = -up_frac * (up_frac + eps).log()
        ent_dn = -dn_frac * (dn_frac + eps).log()
        return df.with_columns((ent_up - ent_dn).alias(out))


@dataclass
class EntropyRatio(Feature):
    """entropy(short) / entropy(long) of return sign sequence. Local-vs-long randomness ratio."""
    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["entropy_ratio_{short}_{long}"]
    short: int = 120
    long: int = 960

    def compute_pair(self, df):
        out = f"entropy_ratio_{self.short}_{self.long}"
        ret = pl.col("close").diff()
        up_s = (ret > 0).cast(pl.Float32).rolling_mean(self.short)
        up_l = (ret > 0).cast(pl.Float32).rolling_mean(self.long)
        eps = 1e-9
        h_s = -(up_s * (up_s + eps).log() + (1 - up_s) * (1 - up_s + eps).log())
        h_l = -(up_l * (up_l + eps).log() + (1 - up_l) * (1 - up_l + eps).log())
        return df.with_columns((h_s / (h_l + 1e-9)).alias(out))


@dataclass
class RsiDivPolarity(Feature):
    """RSI vs price divergence polarity (continuous).

    Computes -d_rsi * d_price where d_rsi = RSI - RSI_lag and d_price = pct change.
    Positive = bearish divergence (price up, RSI down), negative = bullish divergence.
    """
    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["rsi_div_{period}_{lag}"]
    period: int = 14
    lag: int = 60

    def compute_pair(self, df):
        out = f"rsi_div_{self.period}_{self.lag}"
        c = pl.col("close")
        diff = c.diff()
        gain = pl.when(diff > 0).then(diff).otherwise(0.0)
        loss = pl.when(diff < 0).then(-diff).otherwise(0.0)
        rsi = 100 - 100 / (1 + gain.rolling_mean(self.period) / (loss.rolling_mean(self.period) + 1e-9))
        d_rsi = rsi - rsi.shift(self.lag)
        d_price = (c - c.shift(self.lag)) / c.shift(self.lag) * 100.0
        return df.with_columns((-d_rsi * d_price).alias(out))


@dataclass
class HilbertAmplitudeSlope(Feature):
    """Rate of change of Hilbert envelope = volatility-of-volatility proxy.

    Approximates Hilbert amplitude via |close − SMA| envelope, then takes its slope.
    """
    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["hilb_slope_{period}_{lag}"]
    period: int = 240
    lag: int = 60

    def compute_pair(self, df):
        out = f"hilb_slope_{self.period}_{self.lag}"
        c = pl.col("close")
        envelope = (c - c.rolling_mean(self.period)).abs().rolling_max(self.period)
        return df.with_columns(((envelope - envelope.shift(self.lag)) / (envelope.shift(self.lag) + 1e-9)).alias(out))

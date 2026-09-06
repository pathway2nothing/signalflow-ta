"""Extended volume-price coupling features.

Volume alone has weak signal - these features couple it with price/range/direction.
Added from sf-profit iter-15, iter-16, iter-18, iter-20 research.
"""
from dataclasses import dataclass
from typing import ClassVar

import polars as pl

from signalflow.ta._compat import Feature


@dataclass
class VolumeWeightedReturn(Feature):
    """sum(vol_i × sign(ret_i)) / sum(vol_i). Net directional volume pressure."""
    requires: ClassVar[list[str]] = ["close", "volume"]
    outputs: ClassVar[list[str]] = ["vw_ret_{window}"]
    window: int = 240

    def compute_pair(self, df):
        out = f"vw_ret_{self.window}"
        signed_vol = pl.col("volume") * pl.col("close").diff().sign()
        return df.with_columns(
            (signed_vol.rolling_sum(self.window) / (pl.col("volume").rolling_sum(self.window) + 1e-9)).alias(out)
        )


@dataclass
class VolumeImbalance(Feature):
    """(up_vol − down_vol) / total_vol over period. up_vol = vol if close>open."""
    requires: ClassVar[list[str]] = ["open", "close", "volume"]
    outputs: ClassVar[list[str]] = ["vol_imb_{period}"]
    period: int = 240

    def compute_pair(self, df):
        out = f"vol_imb_{self.period}"
        up_v = pl.when(pl.col("close") > pl.col("open")).then(pl.col("volume")).otherwise(0.0)
        dn_v = pl.when(pl.col("close") < pl.col("open")).then(pl.col("volume")).otherwise(0.0)
        return df.with_columns((
            (up_v.rolling_sum(self.period) - dn_v.rolling_sum(self.period))
            / (pl.col("volume").rolling_sum(self.period) + 1e-9)
        ).alias(out))


@dataclass
class PriceImpactPerUnit(Feature):
    """|close_t − close_{t-window}| / sum(volume_window). Price move per unit of volume."""
    requires: ClassVar[list[str]] = ["close", "volume"]
    outputs: ClassVar[list[str]] = ["price_impact_{window}"]
    window: int = 240

    def compute_pair(self, df):
        out = f"price_impact_{self.window}"
        c = pl.col("close")
        net = (c - c.shift(self.window)).abs()
        total_vol = pl.col("volume").rolling_sum(self.window)
        return df.with_columns((net / (total_vol + 1e-9)).alias(out))


@dataclass
class VWAPDeviation(Feature):
    """close − VWAP(window). Deviation from volume-weighted average price."""
    requires: ClassVar[list[str]] = ["close", "high", "low", "volume"]
    outputs: ClassVar[list[str]] = ["vwap_dev_{window}"]
    window: int = 240

    def compute_pair(self, df):
        out = f"vwap_dev_{self.window}"
        h, l, c = pl.col("high"), pl.col("low"), pl.col("close")
        v = pl.col("volume")
        tp = (h + l + c) / 3
        num = (tp * v).rolling_sum(self.window)
        den = v.rolling_sum(self.window)
        vwap = num / (den + 1e-9)
        return df.with_columns((c - vwap).alias(out))


@dataclass
class SignedVolumeAccumulation(Feature):
    """Cumulative signed volume: sum(vol_i × sign(close_i − close_{i-1})) over window."""
    requires: ClassVar[list[str]] = ["close", "volume"]
    outputs: ClassVar[list[str]] = ["sig_vol_acc_{window}"]
    window: int = 240

    def compute_pair(self, df):
        out = f"sig_vol_acc_{self.window}"
        c = pl.col("close")
        v = pl.col("volume")
        signed_v = v * c.diff().sign()
        return df.with_columns(signed_v.rolling_sum(self.window).alias(out))


@dataclass
class AbsReturnVolumeCorr(Feature):
    """Rolling Pearson corr(|return|, volume). High = large moves with vol confirmation."""
    requires: ClassVar[list[str]] = ["close", "volume"]
    outputs: ClassVar[list[str]] = ["absret_vol_corr_{window}"]
    window: int = 480

    def compute_pair(self, df):
        out = f"absret_vol_corr_{self.window}"
        a = pl.col("close").diff().abs().alias("_a")
        v = pl.col("volume").alias("_v")
        df = df.with_columns([a, v])
        x = pl.col("_a")
        y = pl.col("_v")
        df = df.with_columns([
            (x * y).rolling_mean(self.window).alias("_xy"),
            x.rolling_mean(self.window).alias("_xm"),
            y.rolling_mean(self.window).alias("_ym"),
            (x ** 2).rolling_mean(self.window).alias("_xx"),
            (y ** 2).rolling_mean(self.window).alias("_yy"),
        ])
        denom = ((pl.col("_xx") - pl.col("_xm") ** 2) * (pl.col("_yy") - pl.col("_ym") ** 2) + 1e-18).sqrt()
        return df.with_columns(
            ((pl.col("_xy") - pl.col("_xm") * pl.col("_ym")) / denom).alias(out)
        ).drop(["_a", "_v", "_xy", "_xm", "_ym", "_xx", "_yy"])


@dataclass
class PriceVolumeCorrelation(Feature):
    """Rolling corr(diff(close), volume). Direction-volume agreement."""
    requires: ClassVar[list[str]] = ["close", "volume"]
    outputs: ClassVar[list[str]] = ["price_vol_corr_{window}"]
    window: int = 480

    def compute_pair(self, df):
        out = f"price_vol_corr_{self.window}"
        d = pl.col("close").diff().alias("_d")
        v = pl.col("volume").alias("_v2")
        df = df.with_columns([d, v])
        x = pl.col("_d")
        y = pl.col("_v2")
        df = df.with_columns([
            (x * y).rolling_mean(self.window).alias("_xy"),
            x.rolling_mean(self.window).alias("_xm"),
            y.rolling_mean(self.window).alias("_ym"),
            (x ** 2).rolling_mean(self.window).alias("_xx"),
            (y ** 2).rolling_mean(self.window).alias("_yy"),
        ])
        denom = ((pl.col("_xx") - pl.col("_xm") ** 2) * (pl.col("_yy") - pl.col("_ym") ** 2) + 1e-18).sqrt()
        return df.with_columns(
            ((pl.col("_xy") - pl.col("_xm") * pl.col("_ym")) / denom).alias(out)
        ).drop(["_d", "_v2", "_xy", "_xm", "_ym", "_xx", "_yy"])


@dataclass
class VolumePerRange(Feature):
    """Volume / (high − low), rolling mean. High = absorption."""
    requires: ClassVar[list[str]] = ["high", "low", "volume"]
    outputs: ClassVar[list[str]] = ["vol_per_range_{window}"]
    window: int = 240

    def compute_pair(self, df):
        out = f"vol_per_range_{self.window}"
        rng = pl.col("high") - pl.col("low")
        ratio = pl.col("volume") / (rng + 1e-9)
        return df.with_columns(ratio.rolling_mean(self.window).alias(out))


@dataclass
class VolumeSpike(Feature):
    """volume_t / SMA(volume, window) − 1. Instantaneous spike intensity, centered at 0."""
    requires: ClassVar[list[str]] = ["volume"]
    outputs: ClassVar[list[str]] = ["vol_spike_{window}"]
    window: int = 240

    def compute_pair(self, df):
        out = f"vol_spike_{self.window}"
        v = pl.col("volume")
        return df.with_columns((v / (v.rolling_mean(self.window) + 1e-9) - 1).alias(out))


@dataclass
class VolumeAcceleration(Feature):
    """Diff of (SMA_short − SMA_long) of volume. Volume regime change rate."""
    requires: ClassVar[list[str]] = ["volume"]
    outputs: ClassVar[list[str]] = ["vol_accel_{short}_{long}"]
    short: int = 60
    long: int = 480

    def compute_pair(self, df):
        out = f"vol_accel_{self.short}_{self.long}"
        v = pl.col("volume")
        diff = v.rolling_mean(self.short) - v.rolling_mean(self.long)
        return df.with_columns(diff.diff(self.short).alias(out))


@dataclass
class VolumeZScore(Feature):
    """(volume − mean(N)) / std(N). Stationary volume regime."""
    requires: ClassVar[list[str]] = ["volume"]
    outputs: ClassVar[list[str]] = ["vol_zscore_{window}"]
    window: int = 480

    def compute_pair(self, df):
        out = f"vol_zscore_{self.window}"
        v = pl.col("volume")
        return df.with_columns(
            ((v - v.rolling_mean(self.window)) / (v.rolling_std(self.window) + 1e-9)).alias(out)
        )


@dataclass
class VolumeMomentumRatio(Feature):
    """vol_sma_short / vol_sma_long − 1. Long-vs-short volume regime."""
    requires: ClassVar[list[str]] = ["volume"]
    outputs: ClassVar[list[str]] = ["vol_mom_ratio_{short}_{long}"]
    short: int = 60
    long: int = 480

    def compute_pair(self, df):
        out = f"vol_mom_ratio_{self.short}_{self.long}"
        v = pl.col("volume")
        return df.with_columns(
            (v.rolling_mean(self.short) / (v.rolling_mean(self.long) + 1e-9) - 1).alias(out)
        )


@dataclass
class VolPctRankSignedTrend(Feature):
    """natr_pctrank × sign(SMA_short − SMA_long). High vol percentile × trend direction."""
    requires: ClassVar[list[str]] = ["high", "low", "close"]
    outputs: ClassVar[list[str]] = ["vol_x_trend_{natr_period}_{trend_period}"]
    natr_period: int = 240
    trend_period: int = 120

    def compute_pair(self, df):
        out = f"vol_x_trend_{self.natr_period}_{self.trend_period}"
        h, l, c = pl.col("high"), pl.col("low"), pl.col("close")
        tr = pl.max_horizontal(h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs())
        natr = tr.rolling_mean(self.natr_period) / (c + 1e-9)
        lookback = self.natr_period * 12
        natr_lo = natr.rolling_min(lookback)
        natr_rank = (natr - natr_lo) / (natr.rolling_max(lookback) - natr_lo + 1e-9)
        sma_s = c.rolling_mean(self.trend_period)
        sma_l = c.rolling_mean(self.trend_period * 4)
        trend_sign = (sma_s - sma_l).sign()
        return df.with_columns((natr_rank * trend_sign).alias(out))

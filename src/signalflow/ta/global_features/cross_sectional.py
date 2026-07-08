"""Cross-sectional features across multiple pairs.

Operate on the FULL multi-pair DataFrame (not per-pair), computing rank/dispersion/beta
across pairs at each timestamp. Distinct from existing CrossSectionalStat in stat/cross_sectional.py:
these are richer compositions tested in sf-profit iter-15/16/18/20.

All features expect a DataFrame with `pair` and `ts` columns.
"""
from dataclasses import dataclass
from typing import ClassVar

import polars as pl

from signalflow.ta._compat import Feature


@dataclass
class CrossSectionalReturnRank(Feature):
    """Rank of pair's return-over-window among all pairs at each timestamp ∈ [0, 1]."""
    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["cs_ret_rank_{window}"]
    window: int = 240

    def compute(self, df, context=None):
        out = f"cs_ret_rank_{self.window}"
        df = df.sort(["pair", self.ts_col])
        return df.with_columns(
            (pl.col("close") / pl.col("close").shift(self.window).over("pair") - 1).alias("_ret"),
        ).with_columns(
            (pl.col("_ret").rank(method="average").over(self.ts_col)
             / pl.col("_ret").count().over(self.ts_col)).alias(out)
        ).drop("_ret")


@dataclass
class CrossSectionalAtrRank(Feature):
    """Rank of pair's ATR among all pairs at each timestamp."""
    requires: ClassVar[list[str]] = ["high", "low", "close"]
    outputs: ClassVar[list[str]] = ["cs_atr_rank_{period}"]
    period: int = 240

    def compute(self, df, context=None):
        out = f"cs_atr_rank_{self.period}"
        df = df.sort(["pair", self.ts_col])
        h, l, c = pl.col("high"), pl.col("low"), pl.col("close")
        cp = c.shift(1).over("pair")
        tr = pl.max_horizontal(h - l, (h - cp).abs(), (l - cp).abs())
        df = df.with_columns(tr.rolling_mean(self.period).over("pair").alias("_atr"))
        return df.with_columns(
            (pl.col("_atr").rank(method="average").over(self.ts_col)
             / pl.col("_atr").count().over(self.ts_col)).alias(out)
        ).drop("_atr")


@dataclass
class CrossSectionalAdxRank(Feature):
    """Rank of pair's |returns|-sum (ADX proxy) across pairs."""
    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["cs_adx_rank_{period}"]
    period: int = 240

    def compute(self, df, context=None):
        out = f"cs_adx_rank_{self.period}"
        df = df.sort(["pair", self.ts_col])
        df = df.with_columns(pl.col("close").pct_change().abs().over("pair").alias("_absret"))
        df = df.with_columns(pl.col("_absret").rolling_mean(self.period).over("pair").alias("_dx"))
        return df.with_columns(
            (pl.col("_dx").rank(method="average").over(self.ts_col)
             / pl.col("_dx").count().over(self.ts_col)).alias(out)
        ).drop(["_absret", "_dx"])


@dataclass
class CrossSectionalRangeRank(Feature):
    """Rank of pair's mean (high-low) range across pairs."""
    requires: ClassVar[list[str]] = ["high", "low"]
    outputs: ClassVar[list[str]] = ["cs_range_rank_{period}"]
    period: int = 240

    def compute(self, df, context=None):
        out = f"cs_range_rank_{self.period}"
        df = df.sort(["pair", self.ts_col])
        df = df.with_columns((pl.col("high") - pl.col("low")).rolling_mean(self.period).over("pair").alias("_rng"))
        return df.with_columns(
            (pl.col("_rng").rank(method="average").over(self.ts_col)
             / pl.col("_rng").count().over(self.ts_col)).alias(out)
        ).drop("_rng")


@dataclass
class CrossSectionalRsiRank(Feature):
    """Rank of pair's RSI proxy across pairs."""
    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["cs_rsi_rank_{period}"]
    period: int = 240

    def compute(self, df, context=None):
        out = f"cs_rsi_rank_{self.period}"
        df = df.sort(["pair", self.ts_col])
        ret = pl.col("close").diff().over("pair")
        gain = pl.when(ret > 0).then(ret).otherwise(0.0)
        loss = pl.when(ret < 0).then(-ret).otherwise(0.0)
        df = df.with_columns([
            gain.rolling_mean(self.period).over("pair").alias("_avg_gain"),
            loss.rolling_mean(self.period).over("pair").alias("_avg_loss"),
        ])
        df = df.with_columns(
            (pl.col("_avg_gain") / (pl.col("_avg_gain") + pl.col("_avg_loss") + 1e-9)).alias("_rsi")
        )
        return df.with_columns(
            (pl.col("_rsi").rank(method="average").over(self.ts_col)
             / pl.col("_rsi").count().over(self.ts_col)).alias(out)
        ).drop(["_avg_gain", "_avg_loss", "_rsi"])


@dataclass
class CrossSectionalVolRank(Feature):
    """Rank of pair's rolling realized vol across pairs."""
    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["cs_vol_rank_{period}"]
    period: int = 240

    def compute(self, df, context=None):
        out = f"cs_vol_rank_{self.period}"
        df = df.sort(["pair", self.ts_col])
        df = df.with_columns(pl.col("close").pct_change().over("pair").alias("_ret"))
        df = df.with_columns(pl.col("_ret").rolling_std(self.period).over("pair").alias("_rv"))
        return df.with_columns(
            (pl.col("_rv").rank(method="average").over(self.ts_col)
             / pl.col("_rv").count().over(self.ts_col)).alias(out)
        ).drop(["_ret", "_rv"])


@dataclass
class CrossSectionalReturnAccelRank(Feature):
    """Rank of pair's return acceleration (ROC_short − ROC_long) across pairs."""
    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["cs_retaccel_rank_{short}_{long}"]
    short: int = 60
    long: int = 240

    def compute(self, df, context=None):
        out = f"cs_retaccel_rank_{self.short}_{self.long}"
        df = df.sort(["pair", self.ts_col])
        c = pl.col("close")
        roc_s = (c / c.shift(self.short).over("pair") - 1)
        roc_l = (c / c.shift(self.long).over("pair") - 1)
        df = df.with_columns((roc_s - roc_l).alias("_accel"))
        return df.with_columns(
            (pl.col("_accel").rank(method="average").over(self.ts_col)
             / pl.col("_accel").count().over(self.ts_col)).alias(out)
        ).drop("_accel")


@dataclass
class CrossSectionalDispersion(Feature):
    """Mean of |pair_ret − market_mean_ret| across pairs at each timestamp."""
    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["cs_dispersion_{window}"]
    window: int = 240

    def compute(self, df, context=None):
        out = f"cs_dispersion_{self.window}"
        df = df.sort(["pair", self.ts_col])
        df = df.with_columns(
            (pl.col("close") / pl.col("close").shift(self.window).over("pair") - 1).alias("_ret"),
        )
        return df.with_columns(
            (pl.col("_ret") - pl.col("_ret").mean().over(self.ts_col)).abs().mean().over(self.ts_col).alias(out)
        ).drop("_ret")


@dataclass
class CrossSectionalRetSkew(Feature):
    """Skewness of returns across pairs at each timestamp. NEEDS 5+ pairs."""
    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["cs_ret_skew_{window}"]
    window: int = 240

    def compute(self, df, context=None):
        out = f"cs_ret_skew_{self.window}"
        df = df.sort(["pair", self.ts_col])
        df = df.with_columns(
            (pl.col("close") / pl.col("close").shift(self.window).over("pair") - 1).alias("_ret"),
        )
        m = pl.col("_ret").mean().over(self.ts_col)
        s = pl.col("_ret").std().over(self.ts_col)
        skew = ((pl.col("_ret") - m).pow(3)).mean().over(self.ts_col) / (s.pow(3) + 1e-12)
        return df.with_columns(skew.alias(out)).drop("_ret")


@dataclass
class MarketBreadth(Feature):
    """Fraction of pairs with positive return over window ∈ [0, 1]."""
    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["market_breadth_{window}"]
    window: int = 480

    def compute(self, df, context=None):
        out = f"market_breadth_{self.window}"
        df = df.sort(["pair", self.ts_col])
        df = df.with_columns(
            (pl.col("close") / pl.col("close").shift(self.window).over("pair") - 1).alias("_ret"),
        )
        breadth = (pl.col("_ret") > 0).cast(pl.Float32).mean().over(self.ts_col)
        return df.with_columns(breadth.alias(out)).drop("_ret")


@dataclass
class RelativeStrengthVsMarket(Feature):
    """(pair_ret − market_mean_ret) / market_std_ret over window."""
    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["rs_vs_market_{window}"]
    window: int = 240

    def compute(self, df, context=None):
        out = f"rs_vs_market_{self.window}"
        df = df.sort(["pair", self.ts_col])
        df = df.with_columns(
            (pl.col("close") / pl.col("close").shift(self.window).over("pair") - 1).alias("_ret"),
        )
        m = pl.col("_ret").mean().over(self.ts_col)
        s = pl.col("_ret").std().over(self.ts_col)
        return df.with_columns(((pl.col("_ret") - m) / (s + 1e-9)).alias(out)).drop("_ret")


@dataclass
class DivergenceFromMarketMedian(Feature):
    """pair_close pct vs market_median pct over period."""
    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["div_from_market_{period}"]
    period: int = 240

    def compute(self, df, context=None):
        out = f"div_from_market_{self.period}"
        df = df.sort(["pair", self.ts_col])
        df = df.with_columns(
            (pl.col("close") / pl.col("close").shift(self.period).over("pair") - 1).alias("_ret"),
        )
        med = pl.col("_ret").median().over(self.ts_col)
        return df.with_columns((pl.col("_ret") - med).alias(out)).drop("_ret")


@dataclass
class PairExcessReturn(Feature):
    """pair_ret − market_mean_ret. Excess return vs cross-section."""
    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["pair_excess_ret_{window}"]
    window: int = 240

    def compute(self, df, context=None):
        out = f"pair_excess_ret_{self.window}"
        df = df.sort(["pair", self.ts_col])
        df = df.with_columns(
            (pl.col("close") / pl.col("close").shift(self.window).over("pair") - 1).alias("_ret"),
        )
        m = pl.col("_ret").mean().over(self.ts_col)
        return df.with_columns((pl.col("_ret") - m).alias(out)).drop("_ret")


@dataclass
class CrossSectionalBeta(Feature):
    """Rolling β = cov(pair_ret, market_median_ret) / var(market_median_ret)."""
    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["cs_beta_{window}"]
    window: int = 960

    def compute(self, df, context=None):
        out = f"cs_beta_{self.window}"
        df = df.sort(["pair", self.ts_col])
        df = df.with_columns(pl.col("close").pct_change().over("pair").alias("_ret"))
        df = df.with_columns(pl.col("_ret").median().over(self.ts_col).alias("_mret"))
        df = df.with_columns([
            (pl.col("_ret") * pl.col("_mret")).rolling_mean(self.window).over("pair").alias("_xy"),
            pl.col("_ret").rolling_mean(self.window).over("pair").alias("_xmean"),
            pl.col("_mret").rolling_mean(self.window).over("pair").alias("_ymean"),
            (pl.col("_mret") ** 2).rolling_mean(self.window).over("pair").alias("_yy"),
        ])
        return df.with_columns(
            ((pl.col("_xy") - pl.col("_xmean") * pl.col("_ymean"))
             / (pl.col("_yy") - pl.col("_ymean") ** 2 + 1e-12)).alias(out)
        ).drop(["_ret", "_mret", "_xy", "_xmean", "_ymean", "_yy"])


@dataclass
class AvgPairwiseCorrMarket(Feature):
    """Rolling corr(pair_ret, market_median_ret). Pair's coupling to broad market."""
    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["pair_market_corr_{window}"]
    window: int = 960

    def compute(self, df, context=None):
        out = f"pair_market_corr_{self.window}"
        df = df.sort(["pair", self.ts_col])
        df = df.with_columns(pl.col("close").pct_change().over("pair").alias("_ret"))
        df = df.with_columns(pl.col("_ret").median().over(self.ts_col).alias("_mret"))
        df = df.with_columns([
            (pl.col("_ret") * pl.col("_mret")).rolling_mean(self.window).over("pair").alias("_xy"),
            pl.col("_ret").rolling_mean(self.window).over("pair").alias("_xm"),
            pl.col("_mret").rolling_mean(self.window).over("pair").alias("_ym"),
            (pl.col("_ret") ** 2).rolling_mean(self.window).over("pair").alias("_xx"),
            (pl.col("_mret") ** 2).rolling_mean(self.window).over("pair").alias("_yy"),
        ])
        denom = ((pl.col("_xx") - pl.col("_xm") ** 2) * (pl.col("_yy") - pl.col("_ym") ** 2) + 1e-18).sqrt()
        return df.with_columns(
            ((pl.col("_xy") - pl.col("_xm") * pl.col("_ym")) / denom).alias(out)
        ).drop(["_ret", "_mret", "_xy", "_xm", "_ym", "_xx", "_yy"])


@dataclass
class PairLeadLagCorr(Feature):
    """Rolling corr(pair_ret_t, market_median_ret_{t-lag}). Detects lead/lag relations."""
    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["leadlag_corr_{lag}_{window}"]
    lag: int = 5
    window: int = 1440

    def compute(self, df, context=None):
        out = f"leadlag_corr_{self.lag}_{self.window}"
        df = df.sort(["pair", self.ts_col])
        df = df.with_columns(pl.col("close").pct_change().over("pair").alias("_ret"))
        df = df.with_columns(pl.col("_ret").median().over(self.ts_col).alias("_mret"))
        df = df.with_columns(pl.col("_mret").shift(self.lag).over("pair").alias("_mret_lag"))
        df = df.with_columns([
            (pl.col("_ret") * pl.col("_mret_lag")).rolling_mean(self.window).over("pair").alias("_xy"),
            pl.col("_ret").rolling_mean(self.window).over("pair").alias("_xmean"),
            pl.col("_mret_lag").rolling_mean(self.window).over("pair").alias("_ymean"),
            (pl.col("_ret") ** 2).rolling_mean(self.window).over("pair").alias("_xx"),
            (pl.col("_mret_lag") ** 2).rolling_mean(self.window).over("pair").alias("_yy"),
        ])
        denom = ((pl.col("_xx") - pl.col("_xmean") ** 2) * (pl.col("_yy") - pl.col("_ymean") ** 2) + 1e-18).sqrt()
        return df.with_columns(
            ((pl.col("_xy") - pl.col("_xmean") * pl.col("_ymean")) / denom).alias(out)
        ).drop(["_ret", "_mret", "_mret_lag", "_xy", "_xmean", "_ymean", "_xx", "_yy"])

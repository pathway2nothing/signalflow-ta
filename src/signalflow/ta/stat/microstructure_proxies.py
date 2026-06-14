"""Microstructure-from-OHLCV proxies.

Features that approximate quantities measurable from L2 order-book data
(spread, order-flow imbalance, volume-weighted depth) using only bar-level
OHLCV. Validated in iter-29 and onwards.
"""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import polars as pl

from signalflow.ta._compat import feature
from signalflow.ta._compat import Feature
from signalflow.ta.stat._causal_helpers import log_returns, rolling_mean, rolling_std, truncated_ema


@dataclass
@feature("stat/bid_ask_spread_proxy_ema")
class BidAskSpreadProxyEMAStat(Feature):
    """EMA of z-scored bar-level spread proxy: log(H/L) − |log(C/O)|.

    Captures execution friction / spread tightness from high-low range
    versus open-close magnitude. Wider proxy = wider spread → more friction.

    Iter-29 stability: mean MI_normalised = 0.155 across 19 stable triples.

    Reference: Corwin-Schultz (2012). A simple way to estimate bid-ask
    spreads from daily high and low prices. JoF 67(2):719-760.
    """

    period: int = 60
    tau: int = 30

    requires: ClassVar[list[str]] = ["high", "low", "open", "close"]
    outputs: ClassVar[list[str]] = ["f14_spread_ema_{period}_{tau}"]
    test_params: ClassVar[list[dict]] = [
        {"period": 60, "tau": 30},
        {"period": 240, "tau": 120},
        {"period": 480, "tau": 60},
    ]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        out_col = f"f14_spread_ema_{self.period}_{self.tau}"
        h = df["high"].to_numpy().astype(np.float64)
        l = df["low"].to_numpy().astype(np.float64)
        o = df["open"].to_numpy().astype(np.float64)
        c = df["close"].to_numpy().astype(np.float64)
        sp = np.log(np.maximum(h, 1e-12) / np.maximum(l, 1e-12)) - np.abs(
            np.log(np.maximum(c, 1e-12) / np.maximum(o, 1e-12)))
        m = rolling_mean(sp, self.period)
        s = rolling_std(sp, self.period)
        z = np.where(np.isfinite(s) & (s > 0), (sp - m) / np.maximum(s, 1e-12), 0.0)
        out = truncated_ema(z, self.tau)
        return df.with_columns(pl.Series(out_col, out, dtype=pl.Float64))


@dataclass
@feature("stat/vol_weighted_bb_width")
class VolWeightedBBWidthStat(Feature):
    """(close − volume-weighted mean) / (2·volume-weighted std).

    Bollinger-band stretch where the centerline and band-width are
    volume-weighted instead of equal-weighted. Reduces noise from
    low-conviction bars.

    Iter-29 stability: mean MI_normalised = 0.115 across 8 stable triples.
    """

    period: int = 20

    requires: ClassVar[list[str]] = ["close", "volume"]
    outputs: ClassVar[list[str]] = ["f07_vw_bb_{period}"]
    test_params: ClassVar[list[dict]] = [
        {"period": 20},
        {"period": 60},
        {"period": 240},
    ]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        out_col = f"f07_vw_bb_{self.period}"
        c = df["close"].to_numpy().astype(np.float64)
        v = df["volume"].to_numpy().astype(np.float64)
        cv = c * v
        v_mean = pl.Series(v).rolling_mean(self.period, min_samples=2).to_numpy()
        mid = pl.Series(cv).rolling_mean(self.period, min_samples=2).to_numpy() / np.maximum(v_mean, 1e-12)
        sq = ((c - mid) ** 2) * v
        var = pl.Series(sq).rolling_mean(self.period, min_samples=2).to_numpy() / np.maximum(v_mean, 1e-12)
        sd = np.sqrt(np.maximum(var, 1e-12))
        out = (c - mid) / (2 * sd)
        return df.with_columns(pl.Series(out_col, np.clip(out, -10, 10), dtype=pl.Float64))


@dataclass
@feature("stat/ofi_signed_proxy")
class OFISignedProxyStat(Feature):
    """Cont-Kukanov-Stoikov order-flow imbalance proxy from OHLCV.

    OFI_bar = volume × (2·(close-low)/(high-low) - 1)
    Normalised by rolling-median volume, then EMA-smoothed. Approximates
    aggressive-flow imbalance derivable only from L2 order book.

    Iter-30 stability: top mean MI_normalised = 0.145 on D3.

    Reference: Cont, Kukanov & Stoikov (2014). The price impact of order
    book events. Journal of Financial Econometrics 12(1):47-88, arXiv:1011.6402.
    """

    period: int = 60

    requires: ClassVar[list[str]] = ["high", "low", "close", "volume"]
    outputs: ClassVar[list[str]] = ["f008_ofi_{period}"]
    test_params: ClassVar[list[dict]] = [
        {"period": 60},
        {"period": 240},
        {"period": 480},
    ]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        out_col = f"f008_ofi_{self.period}"
        h = df["high"].to_numpy().astype(np.float64)
        l = df["low"].to_numpy().astype(np.float64)
        c = df["close"].to_numpy().astype(np.float64)
        v = df["volume"].to_numpy().astype(np.float64)
        loc = (c - l) / np.maximum(h - l, 1e-12)
        ofi_bar = v * (2 * loc - 1)
        v_med = pl.Series(v).rolling_median(240, min_samples=2).to_numpy()
        ofi_n = ofi_bar / np.maximum(v_med, 1e-12)
        ofi_n = np.nan_to_num(ofi_n, nan=0.0, posinf=0.0, neginf=0.0)
        out = truncated_ema(ofi_n, self.period)
        return df.with_columns(pl.Series(out_col, np.clip(out, -10, 10), dtype=pl.Float64))


@dataclass
@feature("stat/kalman_innovation_vol")
class KalmanInnovationVolStat(Feature):
    """Std of Kalman-filter innovations (one-step forecast errors).

    Random-walk Kalman filter on close; rolling std of innovations captures
    deviations from local trend that simple ATR misses.

    Iter-30 stability: top WF mean MI_normalised ≈ 0.208 (med std 0.04).

    Reference: Harvey, A. C. (1989). Forecasting, structural time series
    models and the Kalman filter.
    """

    period: int = 60
    q: int = 10

    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["f042_kalman_innov_{period}_{q}"]
    test_params: ClassVar[list[dict]] = [
        {"period": 60, "q": 10},
        {"period": 240, "q": 50},
        {"period": 480, "q": 25},
    ]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        """Warmup-invariant rewrite: steady-state Kalman ≈ EMA, innovations = c - EMA.

        For a random-walk Kalman with process noise q and obs noise 1, the
        steady-state gain K* solves K*² + (q-1)K* - q = 0 ⇒ K* = (q + √(q²+4q)) / 2,
        equivalent to EMA with span ≈ 2/K* − 1. We use truncated_ema instead of
        recursive Kalman to make innovations warmup-invariant.
        """
        out_col = f"f042_kalman_innov_{self.period}_{self.q}"
        c = df["close"].to_numpy().astype(np.float64)
        q_val = self.q / 100.0
        K_star = (q_val + np.sqrt(q_val ** 2 + 4 * q_val)) / 2.0
        tau = max(2, int(2.0 / max(K_star, 1e-6) - 1))
        ema = truncated_ema(c, tau)
        innov = c - ema
        innov = np.where(np.isfinite(innov), innov, 0.0)
        out = rolling_std(innov, self.period)
        return df.with_columns(pl.Series(out_col, out, dtype=pl.Float64))


@dataclass
@feature("stat/vwap_parkinson_stretch")
class VwapParkinsonStretchStat(Feature):
    """(close - VWAP) / Parkinson vol - volume-weighted fair-value stretch.

    VWAP anchors to volume-weighted price; Parkinson normalises by range
    rather than std. Combines Family A position stretch with VWAP reference.

    Iter-29 stability: top mean MI_normalised = 0.242 on D3 across 3 stable
    triples (best mean of any underexploited stretch reference).

    Reference: Parkinson, M. (1980). The extreme value method for estimating
    the variance of the rate of return. Journal of Business 53(1):61-65.
    """

    period: int = 15

    requires: ClassVar[list[str]] = ["high", "low", "close", "volume"]
    outputs: ClassVar[list[str]] = ["f02_vwap_park_{period}"]
    test_params: ClassVar[list[dict]] = [
        {"period": 15},
        {"period": 60},
        {"period": 240},
    ]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        import math
        out_col = f"f02_vwap_park_{self.period}"
        c = df["close"].to_numpy().astype(np.float64)
        h = df["high"].to_numpy().astype(np.float64)
        l = df["low"].to_numpy().astype(np.float64)
        v = df["volume"].to_numpy().astype(np.float64)
        pv = c * v
        v_mean = rolling_mean(v, self.period)
        vwap = rolling_mean(pv, self.period) / np.where(np.isfinite(v_mean) & (v_mean > 0), v_mean, 1e-12)
        rng_sq = np.log(np.maximum(h, 1e-12) / np.maximum(l, 1e-12)) ** 2
        park = np.sqrt(rolling_mean(rng_sq, self.period) / (4 * math.log(2)))
        out = (c - vwap) / np.maximum(park * c, 1e-12)
        return df.with_columns(pl.Series(out_col, np.clip(out, -20, 20), dtype=pl.Float64))


@dataclass
@feature("stat/fisher_information_returns")
class FisherInformationReturnsStat(Feature):
    """Fisher information for normal MLE: n / sigma^2 of returns - inverse variance.

    Iter-27 stability: top mean MI_normalised ≈ 0.126 on C1 market-vol regime.
    Stable for tau-style features but with higher fold-to-fold std.
    """

    period: int = 60
    source_col: str = "close"

    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["fisher_info_returns_{period}"]
    test_params: ClassVar[list[dict]] = [
        {"period": 60},
        {"period": 240},
        {"period": 480},
    ]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        out_col = f"fisher_info_returns_{self.period}"
        c = df[self.source_col].to_numpy().astype(np.float64)
        r = log_returns(c)
        sd = rolling_std(r, self.period)
        fisher = self.period / (sd ** 2 + 1e-20)
        return df.with_columns(pl.Series(out_col, np.log1p(fisher), dtype=pl.Float64))

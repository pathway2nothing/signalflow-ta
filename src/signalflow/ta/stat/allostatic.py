"""Allostatic load: cumulative volatility-stress with exponential decay.

Borrowed from stress biology (McEwen & Stellar 1993): living organisms
accumulate "wear and tear" from repeated stress responses with a decay
constant. Applied to markets, |z-score| of returns is the per-bar stress
event; EMA with characteristic time tau is the body's adaptation.

Empirically validated in iter-27 of sf-profit (target encoding research):
mean MI_normalised across 6 walk-forward folds = 0.18 on the
forward realized-volatility regime label, std 0.02 — one of the most
stable cross-fold features in that experiment.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import polars as pl

from signalflow.core import feature
from signalflow.feature.base import Feature


@dataclass
@feature("stat/allostatic_load")
class AllostaticLoadStat(Feature):
    """Exponentially decaying cumulative stress proxy.

    Algorithm:
        1. log_returns r_t = ln(close_t / close_{t-1})
        2. rolling std σ_t = std(r) over `period` bars
        3. per-bar stress s_t = |r_t| / σ_t  (z-score magnitude)
        4. allostatic load L_t = α * s_t + (1-α) * L_{t-1}
           where α = 1 - exp(-1/tau)

    High L values indicate prolonged elevated stress — markets in extended
    high-volatility regimes. Low values indicate calm.

    Attributes:
        period: window for the rolling std baseline (denominator of z-score).
        tau: characteristic decay time of the EMA (bars).
        source_col: input column.

    Reference:
        McEwen, B. S., & Stellar, E. (1993). Stress and the individual:
        Mechanisms leading to disease. Archives of Internal Medicine.
    """

    period: int = 240
    tau: int = 60
    source_col: str = "close"

    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["allo_load_{period}_{tau}"]
    test_params: ClassVar[list[dict]] = [
        {"period": 240, "tau": 60},
        {"period": 1440, "tau": 240},
        {"period": 480, "tau": 120},
    ]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        out_col = f"allo_load_{self.period}_{self.tau}"
        x = df[self.source_col].to_numpy().astype(np.float64)
        rets = np.diff(np.log(x), prepend=x[0])
        sd = pl.Series(rets).rolling_std(window_size=self.period, min_samples=2).to_numpy()
        z = np.where(sd > 0, np.abs(rets) / sd, 0.0)
        alpha = 1.0 - math.exp(-1.0 / self.tau)
        load = np.zeros_like(z)
        s = 0.0
        for i in range(len(z)):
            s = alpha * z[i] + (1 - alpha) * s
            load[i] = s
        return df.with_columns(pl.Series(out_col, load, dtype=pl.Float64))


@dataclass
@feature("stat/allostatic_load_directional")
class AllostaticLoadDirectionalStat(Feature):
    """Directional allostatic load — signed cumulative stress.

    Like :class:`AllostaticLoadStat` but uses signed z-score (r/σ instead of
    |r|/σ). Captures directional persistence of stress events: positive
    values = bull-stress accumulation, negative = bear-stress.

    Empirically the most informative feature of iter-28 (sf-profit
    target-encoding research): mean MI_normalised across 6 walk-forward
    folds = 0.467 on mean-reversion-event labels with std 0.009 — best
    cross-fold mean of any feature tested across iter-26/27/28 (502 + 60
    feature classes screened).

    Attributes:
        period: window for rolling-σ baseline.
        tau: EMA characteristic time (bars).
        source_col: input column.
    """

    period: int = 240
    tau: int = 60
    source_col: str = "close"

    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["allo_dir_{period}_{tau}"]
    test_params: ClassVar[list[dict]] = [
        {"period": 240, "tau": 60},
        {"period": 480, "tau": 120},
        {"period": 1440, "tau": 240},
    ]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        out_col = f"allo_dir_{self.period}_{self.tau}"
        x = df[self.source_col].to_numpy().astype(np.float64)
        rets = np.diff(np.log(x), prepend=x[0])
        sd = pl.Series(rets).rolling_std(self.period, min_samples=2).to_numpy()
        signed_z = np.where(sd > 0, rets / sd, 0.0)
        alpha = 1.0 - math.exp(-1.0 / self.tau)
        load = np.zeros_like(signed_z)
        s = 0.0
        for i in range(len(signed_z)):
            s = alpha * signed_z[i] + (1 - alpha) * s
            load[i] = s
        return df.with_columns(pl.Series(out_col, load, dtype=pl.Float64))


@dataclass
@feature("stat/allostatic_fast_slow_ratio")
class AllostaticFastSlowRatioStat(Feature):
    """Ratio of fast-tau load to slow-tau load — short vs long stress regime.

    A growing ratio indicates an accelerating stress regime; values near 1
    indicate a stable regime; sub-1 indicates winding-down.

    Iter-28 stability: mean MI_normalised = 0.13 with std 0.02 across 22
    stable triples, all on B1_vol_realized and C1_mkt_vol targets.

    Attributes:
        period: window for rolling-σ baseline.
        tau_fast: short EMA characteristic time.
        tau_slow: long EMA characteristic time.
        source_col: input column.
    """

    period: int = 480
    tau_fast: int = 30
    tau_slow: int = 240
    source_col: str = "close"

    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["allo_fs_{period}_{tau_fast}_{tau_slow}"]
    test_params: ClassVar[list[dict]] = [
        {"period": 480, "tau_fast": 30, "tau_slow": 240},
        {"period": 1440, "tau_fast": 60, "tau_slow": 480},
        {"period": 240, "tau_fast": 15, "tau_slow": 120},
    ]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        out_col = f"allo_fs_{self.period}_{self.tau_fast}_{self.tau_slow}"
        x = df[self.source_col].to_numpy().astype(np.float64)
        rets = np.diff(np.log(x), prepend=x[0])
        sd = pl.Series(rets).rolling_std(self.period, min_samples=2).to_numpy()
        z = np.where(sd > 0, np.abs(rets) / sd, 0.0)

        def _ema(arr, tau):
            a = 1.0 - math.exp(-1.0 / tau)
            out = np.zeros_like(arr)
            s = 0.0
            for i in range(len(arr)):
                s = a * arr[i] + (1 - a) * s
                out[i] = s
            return out

        load_fast = _ema(z, self.tau_fast)
        load_slow = _ema(z, self.tau_slow) + 1e-12
        return df.with_columns(pl.Series(out_col, load_fast / load_slow, dtype=pl.Float64))


@dataclass
@feature("stat/cum_zscore_power_law")
class CumZScorePowerLawStat(Feature):
    """Cumulative z-score with power-law decay kernel.

    EMA replaced by a slow power-law kernel `(i+1)^(-gamma)` giving
    long-memory weights instead of exponential decay. Mimics
    fractional-integration / rough-volatility memory structure.

    Iter-29 stability: mean MI_normalised = 0.292 on D3 mean-reversion,
    std 0.011 across 23 stable triples.
    """

    period: int = 60
    gamma: int = 50  # gamma/100 = decay exponent

    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["f08_pl_cum_z_{period}_{gamma}"]
    test_params: ClassVar[list[dict]] = [
        {"period": 60, "gamma": 50},
        {"period": 240, "gamma": 70},
        {"period": 120, "gamma": 50},
    ]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        from numpy.lib.stride_tricks import sliding_window_view
        out_col = f"f08_pl_cum_z_{self.period}_{self.gamma}"
        c = df["close"].to_numpy().astype(np.float64)
        r = np.diff(np.log(c), prepend=c[0])
        sd = pl.Series(r).rolling_std(self.period, min_samples=2).to_numpy()
        z = np.where(sd > 0, r / sd, 0.0)
        w = (np.arange(1, self.period + 1).astype(np.float64)) ** (-self.gamma / 100.0)
        w /= w.sum()
        n = len(z)
        out = np.full(n, np.nan, dtype=np.float64)
        if n >= self.period:
            wins = sliding_window_view(z, self.period)
            out[self.period - 1:] = wins.dot(w[::-1])
        return df.with_columns(pl.Series(out_col, out, dtype=pl.Float64))


@dataclass
@feature("stat/vol_weighted_zscore_ema")
class VolWeightedZScoreEMAStat(Feature):
    """EMA of (return z-score × normalised volume) — conviction-weighted stress.

    Scales each bar's z-score by its volume relative to recent average,
    then EMA-smooths. Captures momentum from high-conviction bars and
    suppresses noisy low-volume moves.

    Iter-29 stability: mean MI_normalised = 0.249 on D3 mean-reversion,
    std 0.015 across 19 stable triples.

    Reference: Easley, D., Lopez de Prado, M. & O'Hara, M. (2012).
    Flow Toxicity and Liquidity in a High-Frequency World.
    """

    period: int = 30
    tau: int = 60

    requires: ClassVar[list[str]] = ["close", "volume"]
    outputs: ClassVar[list[str]] = ["f10_vw_z_ema_{period}_{tau}"]
    test_params: ClassVar[list[dict]] = [
        {"period": 30, "tau": 60},
        {"period": 240, "tau": 120},
        {"period": 120, "tau": 60},
    ]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        out_col = f"f10_vw_z_ema_{self.period}_{self.tau}"
        c = df["close"].to_numpy().astype(np.float64)
        v = df["volume"].to_numpy().astype(np.float64)
        r = np.diff(np.log(c), prepend=c[0])
        sd = pl.Series(r).rolling_std(self.period, min_samples=2).to_numpy()
        mn = pl.Series(r).rolling_mean(self.period, min_samples=2).to_numpy()
        z = np.where(sd > 0, (r - mn) / sd, 0.0)
        v_mean = pl.Series(v).rolling_mean(self.period, min_samples=2).to_numpy()
        norm_v = v / np.maximum(v_mean, 1e-12)
        wz = z * norm_v
        wz = np.nan_to_num(wz, nan=0.0, posinf=0.0, neginf=0.0)
        if (wz != 0).any():
            lo, hi = np.quantile(wz, [0.01, 0.99])
            wz = np.clip(wz, lo, hi)
        out = pl.Series(wz).ewm_mean(span=self.tau).to_numpy()
        return df.with_columns(pl.Series(out_col, out, dtype=pl.Float64))


@dataclass
@feature("stat/downside_zscore_ema")
class DownsideZScoreEMAStat(Feature):
    """EMA of downside-only return z-score — panic-driven selling stress.

    Iter-29 stability: mean MI_normalised = 0.117 across 10 stable triples.
    """

    period: int = 60
    tau: int = 45

    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["f13_down_z_ema_{period}_{tau}"]
    test_params: ClassVar[list[dict]] = [
        {"period": 60, "tau": 45},
        {"period": 240, "tau": 120},
        {"period": 480, "tau": 60},
    ]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        out_col = f"f13_down_z_ema_{self.period}_{self.tau}"
        c = df["close"].to_numpy().astype(np.float64)
        r = np.diff(np.log(c), prepend=c[0])
        sd = pl.Series(r).rolling_std(self.period, min_samples=2).to_numpy()
        down = np.where(r < 0, r, 0.0)
        z = np.where(sd > 0, down / sd, 0.0)
        out = pl.Series(z).ewm_mean(span=self.tau).to_numpy()
        return df.with_columns(pl.Series(out_col, out, dtype=pl.Float64))

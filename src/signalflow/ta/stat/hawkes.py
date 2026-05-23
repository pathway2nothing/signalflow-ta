"""Hawkes self-excitation intensity proxy.

Hawkes processes (Hawkes 1971) model events whose probability of occurrence
is elevated after recent past events — "self-exciting" dynamics. In
markets, large bars cluster: each |z|>2 event raises the conditional
intensity of the next event with exponential decay.

The intensity λ(t) = μ + Σ_{t_i < t} K · exp(−(t − t_i)/τ) reduces (for
homogeneous background μ=0 and weights K=α) to an exponential moving
average of indicator events. This feature computes that EMA.

Iter-28 stability: mean MI_normalised = 0.186 across 14 stable triples,
mostly on B1_vol_realized labels — high self-excitation precedes / coexists
with high-volatility regimes.

References:
    - Hawkes, A. G. (1971). Spectra of some self-exciting and mutually
      exciting point processes.
    - Bacry, Mastromatteo, Muzy (2015). Hawkes processes in finance.
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
@feature("stat/hawkes_self_excitation")
class HawkesSelfExcitationStat(Feature):
    """EMA of |z|>2 indicator events with characteristic time tau.

    Algorithm:
        1. log returns r_t, rolling-σ baseline over `period` bars.
        2. event_t = 1 if |r_t / σ_t| > 2 else 0.
        3. λ_t = α · event_t + (1 − α) · λ_{t−1}, α = 1 − exp(−1/τ).

    Equivalent to homogeneous Hawkes intensity with exponential kernel
    and unit base rate.

    Attributes:
        period: window for rolling σ baseline.
        tau: characteristic decay time of the kernel (bars).
        source_col: input price column.
    """

    period: int = 480
    tau: int = 60
    source_col: str = "close"

    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["hawkes_{period}_{tau}"]
    test_params: ClassVar[list[dict]] = [
        {"period": 480, "tau": 60},
        {"period": 960, "tau": 120},
        {"period": 1440, "tau": 240},
    ]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        out_col = f"hawkes_{self.period}_{self.tau}"
        x = df[self.source_col].to_numpy().astype(np.float64)
        rets = np.diff(np.log(x), prepend=x[0])
        sd = pl.Series(rets).rolling_std(self.period, min_samples=2).to_numpy()
        z = np.where(sd > 0, np.abs(rets) / sd, 0.0)
        events = (z > 2.0).astype(np.float64)
        alpha = 1.0 - math.exp(-1.0 / self.tau)
        intensity = np.zeros_like(events)
        s = 0.0
        for i in range(len(events)):
            s = alpha * events[i] + (1 - alpha) * s
            intensity[i] = s
        return df.with_columns(pl.Series(out_col, intensity, dtype=pl.Float64))


@dataclass
@feature("stat/marked_hawkes_jumps")
class MarkedHawkesJumpsStat(Feature):
    """Marked Hawkes intensity — EMA of |z|·indicator(|z|>2) (magnitude-weighted).

    Unlike plain Hawkes (indicator only), this weights events by their
    magnitude, so a 5σ event contributes much more than a 2σ one.

    Iter-29 stability: mean MI_normalised = 0.144 across 18 stable triples.

    Reference: Bacry, Delattre, Hoffmann & Muzy (2013).
    """

    period: int = 60
    tau: int = 60

    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["f15_marked_hawkes_{period}_{tau}"]
    test_params: ClassVar[list[dict]] = [
        {"period": 60, "tau": 60},
        {"period": 240, "tau": 120},
        {"period": 480, "tau": 60},
    ]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        out_col = f"f15_marked_hawkes_{self.period}_{self.tau}"
        c = df["close"].to_numpy().astype(np.float64)
        r = np.diff(np.log(c), prepend=c[0])
        sd = pl.Series(r).rolling_std(self.period, min_samples=2).to_numpy()
        z = np.where(sd > 0, r / sd, 0.0)
        mark = np.where(np.abs(z) > 2.0, np.abs(z), 0.0)
        out = pl.Series(mark).ewm_mean(span=self.tau).to_numpy()
        return df.with_columns(pl.Series(out_col, out, dtype=pl.Float64))


@dataclass
@feature("stat/hawkes_vol_climax_intensity")
class HawkesVolClimaxIntensityStat(Feature):
    """EMA of volume-spike events (volume > mean + 3σ over period).

    Captures the arrival rate of large execution waves (institutional
    rebalances, liquidation cascades, retail FOMO bursts).

    Iter-29 stability: mean MI_normalised = 0.136 across 18 stable triples.
    """

    period: int = 120
    tau: int = 120

    requires: ClassVar[list[str]] = ["volume"]
    outputs: ClassVar[list[str]] = ["f17_hawkes_vol_climax_{period}_{tau}"]
    test_params: ClassVar[list[dict]] = [
        {"period": 120, "tau": 120},
        {"period": 480, "tau": 240},
        {"period": 240, "tau": 60},
    ]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        out_col = f"f17_hawkes_vol_climax_{self.period}_{self.tau}"
        v = df["volume"].to_numpy().astype(np.float64)
        mn = pl.Series(v).rolling_mean(self.period, min_samples=2).to_numpy()
        sd = pl.Series(v).rolling_std(self.period, min_samples=2).to_numpy()
        ev = (v > mn + 3 * sd).astype(np.float64)
        out = pl.Series(ev).ewm_mean(span=self.tau).to_numpy()
        return df.with_columns(pl.Series(out_col, out, dtype=pl.Float64))


@dataclass
@feature("stat/hawkes_signed_jumps")
class HawkesSignedJumpsStat(Feature):
    """Directional Hawkes intensity = EMA(+jumps) − EMA(−jumps).

    Separates positive and negative jump-driven self-excitation; sign and
    magnitude carry directional information missed by absolute Hawkes.

    Iter-29 stability: mean MI_normalised = 0.108 across 5 stable triples.
    """

    period: int = 60
    tau: int = 60

    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["f19_signed_jumps_{period}_{tau}"]
    test_params: ClassVar[list[dict]] = [
        {"period": 60, "tau": 60},
        {"period": 240, "tau": 120},
        {"period": 480, "tau": 60},
    ]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        out_col = f"f19_signed_jumps_{self.period}_{self.tau}"
        c = df["close"].to_numpy().astype(np.float64)
        r = np.diff(np.log(c), prepend=c[0])
        sd = pl.Series(r).rolling_std(self.period, min_samples=2).to_numpy()
        z = np.where(sd > 0, r / sd, 0.0)
        jp = (z > 2.0).astype(np.float64); jn = (z < -2.0).astype(np.float64)
        out = (pl.Series(jp).ewm_mean(span=self.tau).to_numpy() -
               pl.Series(jn).ewm_mean(span=self.tau).to_numpy())
        return df.with_columns(pl.Series(out_col, out, dtype=pl.Float64))


@dataclass
@feature("stat/power_law_hawkes")
class PowerLawHawkesStat(Feature):
    """Power-law kernel Hawkes intensity: convolve |z| with t^(-1.2).

    Unlike exponential Hawkes (memory ≈ tau), power-law kernel preserves
    long-memory structure observed empirically in financial volatility
    (Hardiman-Bercot-Bouchaud 2013, EPJB 86).

    Iter-30 stability: top mean MI_normalised = 0.218 on B1_vol_realized —
    best B1 feature among the iter-27/28/29/30 cross-disciplinary set.

    Reference: Hardiman, Bercot & Bouchaud (2013). Critical reflexivity in
    financial markets. arXiv:1302.1405.
    """

    K: int = 480

    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["f029_pl_hawkes_{K}"]
    test_params: ClassVar[list[dict]] = [
        {"K": 480},
        {"K": 1440},
        {"K": 240},
    ]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        from numpy.lib.stride_tricks import sliding_window_view
        out_col = f"f029_pl_hawkes_{self.K}"
        c = df["close"].to_numpy().astype(np.float64)
        r = np.diff(np.log(c), prepend=c[0])
        sd = pl.Series(r).rolling_std(self.K, min_samples=2).to_numpy()
        z = np.where(sd > 0, np.abs(r) / sd, 0.0)
        kernel = (np.arange(1, self.K + 1)) ** (-1.2)
        kernel /= kernel.sum()
        n = len(z)
        out = np.full(n, np.nan, dtype=np.float64)
        if n >= self.K:
            wins = sliding_window_view(z, self.K)
            out_v = wins.dot(kernel[::-1])
            pad = np.full(self.K - 1, np.nan, dtype=np.float64)
            out = np.concatenate([pad, out_v])
        return df.with_columns(pl.Series(out_col, out, dtype=pl.Float64))

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

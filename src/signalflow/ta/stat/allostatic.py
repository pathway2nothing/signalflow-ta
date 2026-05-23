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

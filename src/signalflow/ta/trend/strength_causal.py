"""Warmup-invariant variant of :class:`AdxTrend`.

The original uses Wilder's RMA (recursive moving average with α=1/period),
which is state-recursive and never forgets the cold-start: the same bar T
gives slightly different values depending on how far back the input series
starts. This variant replaces Wilder RMA with a truncated exponential
kernel of width ``5 × period`` - bit-identical regardless of input start,
after warmup.

Reference: Welles Wilder, "New Concepts in Technical Trading Systems".
"""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import polars as pl

from signalflow.ta._compat import Feature
from signalflow.ta.stat._causal_helpers import truncated_ema


@dataclass
class AdxTrendCausal(Feature):
    """ADX (trend strength) with truncated-EMA smoother - warmup-invariant.

    Algorithm (same as Wilder ADX, but RMA replaced by truncated EMA):
        1. True range TR = max(H-L, |H-C_prev|, |L-C_prev|)
        2. +DM = max(H-H_prev, 0) if H_diff > L_diff else 0
           -DM = max(L_prev-L, 0) if L_diff > H_diff else 0
        3. smoothed_TR, smoothed_+DM, smoothed_-DM via truncated_ema(τ=period, window=5·period)
        4. +DI = 100 · smoothed_+DM / smoothed_TR
        5. -DI = 100 · smoothed_-DM / smoothed_TR
        6. DX  = 100 · |+DI − -DI| / (+DI + -DI)
        7. ADX = truncated_ema(DX, τ=period, window=5·period)
    """

    period: int = 14
    normalized: bool = False

    requires: ClassVar[list[str]] = ["high", "low", "close"]
    outputs: ClassVar[list[str]] = ["adx_causal_{period}", "dmp_causal_{period}", "dmn_causal_{period}"]
    test_params: ClassVar[list[dict]] = [
        {"period": 14},
        {"period": 14, "normalized": True},
        {"period": 30},
        {"period": 60},
        {"period": 960},
    ]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        h = df["high"].to_numpy().astype(np.float64)
        l = df["low"].to_numpy().astype(np.float64)
        c = df["close"].to_numpy().astype(np.float64)
        c_prev = np.roll(c, 1); c_prev[0] = c[0]
        h_prev = np.roll(h, 1); h_prev[0] = h[0]
        l_prev = np.roll(l, 1); l_prev[0] = l[0]
        tr = np.maximum.reduce([h - l, np.abs(h - c_prev), np.abs(l - c_prev)])
        tr[0] = h[0] - l[0]
        up = h - h_prev
        dn = l_prev - l
        up[0] = dn[0] = 0
        pdm = np.where((up > dn) & (up > 0), up, 0).astype(np.float64)
        ndm = np.where((dn > up) & (dn > 0), dn, 0).astype(np.float64)
        atr = truncated_ema(tr, self.period)
        s_pdm = truncated_ema(pdm, self.period)
        s_ndm = truncated_ema(ndm, self.period)
        atr_safe = np.where(np.isfinite(atr) & (atr > 0), atr, np.nan)
        dmp = 100.0 * s_pdm / atr_safe
        dmn = 100.0 * s_ndm / atr_safe
        dx = 100.0 * np.abs(dmp - dmn) / np.maximum(dmp + dmn, 1e-12)
        dx = np.nan_to_num(dx, nan=0.0, posinf=0.0, neginf=0.0)
        adx = truncated_ema(dx, self.period)
        if self.normalized:
            adx = adx / 100.0; dmp = dmp / 100.0; dmn = dmn / 100.0
        suffix = ""
        return df.with_columns([
            pl.Series(f"adx_causal_{self.period}{suffix}", adx, dtype=pl.Float64),
            pl.Series(f"dmp_causal_{self.period}{suffix}", dmp, dtype=pl.Float64),
            pl.Series(f"dmn_causal_{self.period}{suffix}", dmn, dtype=pl.Float64),
        ])

    @property
    def warmup(self) -> int:
        return self.period * 10

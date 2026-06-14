"""Omori-Utsu aftershock rate: post-shock relaxation dynamics.

Borrowed from seismology. After a main earthquake, the rate of aftershocks
n(t) decays as a power law n(t) = K / (t + c)^p (Omori 1894, Utsu 1961).
Markets show identical post-shock relaxation: after a 3σ-bar event, smaller
shocks cluster and decay over hours/days.

This feature measures the *current* aftershock rate following the most
recent big-bar event, normalised by the lookforward window.

Empirically validated in iter-27 of sf-profit (target encoding research):
mean MI_normalised across 6 walk-forward folds = 0.18 on the forward
realized-volatility regime label, std 0.02.
"""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import polars as pl

from signalflow.ta._compat import feature
from signalflow.ta._compat import Feature


@dataclass
@feature("stat/aftershock_rate")
class OmoriAftershockRateStat(Feature):
    """Rate of post-shock |z|>1 events normalised by lookforward window.

    Algorithm:
        1. log_returns r_t and rolling σ_t over `period` bars
        2. z_t = r_t / σ_t
        3. mark "main shocks": bars where |z| > 3
        4. mark "aftershocks": bars where |z| > 1
        5. For each bar t, find last main-shock index t*.
           If t - t* <= lookforward, compute:
              rate_t = (# aftershocks in (t*, t]) / (t - t*)
           else rate_t = NaN (no active shock).

    High rate indicates active aftershock sequence; persistent high rate
    over many bars = prolonged regime instability.
    """

    period: int = 480
    lookforward: int = 60
    source_col: str = "close"

    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["omori_rate_{period}_{lookforward}"]
    test_params: ClassVar[list[dict]] = [
        {"period": 480, "lookforward": 60},
        {"period": 1440, "lookforward": 240},
        {"period": 960, "lookforward": 120},
    ]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        """Warmup-invariant: per-bar window scan only on last `lookforward` bars."""
        from signalflow.ta.stat._causal_helpers import log_returns, rolling_std
        out_col = f"omori_rate_{self.period}_{self.lookforward}"
        c = df[self.source_col].to_numpy().astype(np.float64)
        n = len(c)
        rets = log_returns(c)
        s = rolling_std(rets, self.period)
        z = np.where(np.isfinite(s) & (s > 0), rets / np.maximum(s, 1e-12), 0.0)
        big = (np.abs(z) > 3.0).astype(np.int8)
        small = (np.abs(z) > 1.0).astype(np.int8)
        out = np.full(n, np.nan, dtype=np.float64)
        lf = int(self.lookforward)
        for i in range(lf, n):
            win_big = big[i - lf:i]
            if not win_big.any():
                continue
            rel_lb = int(np.where(win_big)[0][-1])
            lb = (i - lf) + rel_lb
            seg = small[lb + 1:i + 1]
            if seg.size > 0:
                out[i] = seg.sum() / seg.size
        return df.with_columns(pl.Series(out_col, out, dtype=pl.Float64))


@dataclass
@feature("stat/aftershock_rate_broad")
class OmoriRateBroadStat(Feature):
    """Aftershock rate with broader thresholds (|z|>0.5 events after |z|>2.5).

    More sensitive to mild post-shock activity than the strict-threshold
    :class:`OmoriAftershockRateStat`. Catches "weak aftershocks" that
    indicate ongoing regime instability without dramatic bars.

    Iter-28 stability: mean MI_normalised = 0.145 with std 0.013 across 14
    stable triples.
    """

    period: int = 480
    lookforward: int = 60
    source_col: str = "close"

    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["omori_broad_{period}_{lookforward}"]
    test_params: ClassVar[list[dict]] = [
        {"period": 480, "lookforward": 60},
        {"period": 1440, "lookforward": 240},
        {"period": 960, "lookforward": 120},
    ]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        """Warmup-invariant Omori broad."""
        from signalflow.ta.stat._causal_helpers import log_returns, rolling_std
        out_col = f"omori_broad_{self.period}_{self.lookforward}"
        c = df[self.source_col].to_numpy().astype(np.float64)
        n = len(c)
        rets = log_returns(c)
        s = rolling_std(rets, self.period)
        z = np.where(np.isfinite(s) & (s > 0), rets / np.maximum(s, 1e-12), 0.0)
        big = (np.abs(z) > 2.5).astype(np.int8)
        small = (np.abs(z) > 0.5).astype(np.int8)
        out = np.full(n, np.nan, dtype=np.float64)
        lf = int(self.lookforward)
        for i in range(lf, n):
            win_big = big[i - lf:i]
            if not win_big.any():
                continue
            rel_lb = int(np.where(win_big)[0][-1])
            lb = (i - lf) + rel_lb
            seg = small[lb + 1:i + 1]
            if seg.size > 0:
                out[i] = seg.sum() / seg.size
        return df.with_columns(pl.Series(out_col, out, dtype=pl.Float64))


@dataclass
@feature("stat/foreshock_buildup")
class ForeshockBuildupStat(Feature):
    """OLS slope of |z| over last `period` bars - building intensity precedes events.

    Earthquake seismology shows foreshocks of slowly increasing magnitude
    precede some main shocks. This feature fits a line to recent stress
    magnitudes; positive slope = build-up, negative = winding-down.

    Iter-28 stability: mean MI_normalised = 0.105 with std 0.022.
    """

    period: int = 30
    source_col: str = "close"

    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["foreshock_{period}"]
    test_params: ClassVar[list[dict]] = [
        {"period": 30},
        {"period": 60},
        {"period": 120},
    ]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        from numpy.lib.stride_tricks import sliding_window_view
        from signalflow.ta.stat._causal_helpers import log_returns, rolling_std

        out_col = f"foreshock_{self.period}"
        c = df[self.source_col].to_numpy().astype(np.float64)
        rets = log_returns(c)
        sd = rolling_std(rets, self.period * 4)
        z = np.where(np.isfinite(sd) & (sd > 0), np.abs(rets) / np.maximum(sd, 1e-12), 0.0)
        n = len(z)
        w = self.period
        if n < w:
            return df.with_columns(pl.lit(np.nan).alias(out_col))
        wins = sliding_window_view(z, w)
        t = np.arange(w, dtype=np.float64)
        t_c = t - t.mean()
        denom = (t_c ** 2).sum()
        ym = wins.mean(axis=1, keepdims=True)
        slope = ((wins - ym) * t_c).sum(axis=1) / denom
        pad = np.full(w - 1, np.nan, dtype=np.float64)
        return df.with_columns(pl.Series(out_col, np.concatenate([pad, slope]), dtype=pl.Float64))


@dataclass
@feature("stat/gutenberg_richter_b_value")
class GutenbergRichterBValueStat(Feature):
    """Gutenberg-Richter b-value (Aki MLE): tail slope of |return| magnitude distribution.

    From seismology: log10(N(>m)) = a - b·m. Computed via Aki's MLE:
    b = log10(e) / (mean(M) - M_min) where M = log|return|.
    Low b indicates heavier tails / more large events relative to small.

    Iter-29 stability: mean MI_normalised = 0.128 across 15 stable triples.

    Reference: Aki, K. (1965). Maximum likelihood estimate of b in the
    formula logN = a - bM and its confidence limits.
    """

    period: int = 240

    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["f28_gr_b_{period}"]
    test_params: ClassVar[list[dict]] = [
        {"period": 240},
        {"period": 480},
        {"period": 960},
    ]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        import math
        from numpy.lib.stride_tricks import sliding_window_view
        out_col = f"f28_gr_b_{self.period}"
        c = df["close"].to_numpy().astype(np.float64)
        n = len(c); w = int(self.period)
        if n < w:
            return df.with_columns(pl.lit(np.nan).alias(out_col))
        r = np.diff(np.log(np.maximum(c, 1e-12)), prepend=np.log(max(float(c[0]), 1e-12)))
        mag = np.log(np.abs(r) + 1e-12)
        wins = sliding_window_view(mag, w)
        m_min = wins.min(axis=1); m_mean = wins.mean(axis=1)
        b = np.log10(math.e) / np.maximum(m_mean - m_min, 1e-12)
        pad = np.full(w - 1, np.nan, dtype=np.float64)
        return df.with_columns(pl.Series(out_col, np.concatenate([pad, b]), dtype=pl.Float64))

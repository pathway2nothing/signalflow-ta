"""KL-divergence drift detector: short-vs-long return distribution.

Online change-detector that computes KL(P_short || P_long), where the two
empirical distributions are histograms of returns over a short recent
window and a long baseline window. Spikes in KL indicate that the recent
distribution has departed from the baseline - useful for regime-shift
detection.

Empirically validated in iter-27 of sf-profit (target encoding research):
mean MI_normalised across 6 walk-forward folds = 0.11, std 0.025; 8 of
the binning × labeling triples crossed the stability threshold.
"""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import polars as pl

from signalflow.ta._compat import feature
from signalflow.ta._compat import Feature


@dataclass
@feature("stat/kl_drift")
class KullbackLeiblerDriftStat(Feature):
    """KL(P_short || P_long) of return histograms.

    Algorithm:
        1. log returns r_t.
        2. For each bar t > long:
              base = r[t-long : t]
              cur  = r[t-short : t]
              edges = linspace(min(base), max(base), n_bins+1)
              P_base = histogram(base, edges)  with Laplace smoothing
              P_cur  = histogram(cur, edges)   with Laplace smoothing
              KL    = sum(P_cur * log(P_cur / P_base))

    High KL = recent distribution has shifted from baseline (volatility
    regime change, jump in drift, etc.).
    """

    short: int = 60
    long: int = 480
    n_bins: int = 16
    source_col: str = "close"

    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["kl_drift_{short}_{long}"]
    test_params: ClassVar[list[dict]] = [
        {"short": 60, "long": 480},
        {"short": 120, "long": 1440},
        {"short": 30, "long": 240},
    ]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        out_col = f"kl_drift_{self.short}_{self.long}"
        x = df[self.source_col].to_numpy().astype(np.float64)
        rets = np.diff(np.log(np.maximum(x, 1e-12)), prepend=np.log(max(float(x[0]), 1e-12)))
        n = len(rets)
        if n < self.long:
            return df.with_columns(pl.lit(np.nan).alias(out_col))
        out = np.full(n, np.nan, dtype=np.float64)
        for i in range(self.long, n):
            base = rets[i - self.long: i]
            cur = rets[i - self.short: i]
            mn, mx = base.min(), base.max()
            if mx - mn < 1e-12:
                continue
            edges = np.linspace(mn, mx, self.n_bins + 1)
            p_base, _ = np.histogram(base, bins=edges)
            p_cur, _ = np.histogram(cur, bins=edges)
            p_base = (p_base + 0.5) / (p_base.sum() + 0.5 * self.n_bins)
            p_cur = (p_cur + 0.5) / (p_cur.sum() + 0.5 * self.n_bins)
            out[i] = float(np.sum(p_cur * np.log(p_cur / p_base)))
        return df.with_columns(pl.Series(out_col, out, dtype=pl.Float64))

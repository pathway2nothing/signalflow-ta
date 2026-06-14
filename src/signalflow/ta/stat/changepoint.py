"""Online change-point detector statistics.

FOCuS-style window-free CUSUM (Romano-Eckley-Fearnhead-Rigaill 2023): the
maximum of `cum_z[t] - cum_z[s]` over `s ∈ [t-W, t]` gives the largest
recent positive drift, a regime-shift indicator.
"""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import polars as pl
from numpy.lib.stride_tricks import sliding_window_view

from signalflow.ta._compat import feature
from signalflow.ta._compat import Feature


@dataclass
@feature("stat/focus_max_stat")
class FocusMaxStatStat(Feature):
    """FOCuS-style CUSUM max statistic over rolling W.

    Iter-30 stability: top mean MI_normalised = 0.106 across 2 stable triples.

    Reference: Romano, Eckley, Fearnhead & Rigaill (2023). Fast online
    changepoint detection via functional pruning CUSUM statistics. Biometrika.
    """

    period: int = 240
    source_col: str = "close"

    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["f037_focus_{period}"]
    test_params: ClassVar[list[dict]] = [
        {"period": 240},
        {"period": 480},
        {"period": 1440},
    ]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        """Warmup-invariant: max-stat computed via rolling cumulative within window only.

        Original cum_z = np.cumsum(z) introduces unbounded path-dependence.
        Equivalent formulation: max over s ∈ [t-w+1, t+1] of Σ_{k=s..t} z[k]
        is computed by taking the maximum *suffix sum* within the rolling window.
        """
        from signalflow.ta.stat._causal_helpers import log_returns, rolling_mean, rolling_std
        out_col = f"f037_focus_{self.period}"
        c = df[self.source_col].to_numpy().astype(np.float64)
        r = log_returns(c)
        mn = rolling_mean(r, self.period)
        sd = rolling_std(r, self.period)
        z = np.where(np.isfinite(sd) & (sd > 0), (r - mn) / np.maximum(sd, 1e-12), 0.0)
        n = len(z); w = self.period
        if n < w:
            return df.with_columns(pl.lit(np.nan).alias(out_col))
        wins = sliding_window_view(z, w)
        rev_cumsum = np.cumsum(wins[:, ::-1], axis=1)
        max_stat = rev_cumsum.max(axis=1)
        pad = np.full(w - 1, np.nan, dtype=np.float64)
        return df.with_columns(pl.Series(out_col, np.concatenate([pad, max_stat]), dtype=pl.Float64))

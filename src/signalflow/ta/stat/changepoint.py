"""Online change-point detector statistics.

FOCuS-style window-free CUSUM (Romano-Eckley-Fearnhead-Rigaill 2023): the
maximum of `cum_z[t] - cum_z[s]` over `s ∈ [t-W, t]` gives the largest
recent positive drift, a regime-shift indicator.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import polars as pl
from numpy.lib.stride_tricks import sliding_window_view

from signalflow.core import feature
from signalflow.feature.base import Feature


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
        out_col = f"f037_focus_{self.period}"
        c = df[self.source_col].to_numpy().astype(np.float64)
        r = np.diff(np.log(c), prepend=c[0])
        mn = pl.Series(r).rolling_mean(self.period, min_samples=2).to_numpy()
        sd = pl.Series(r).rolling_std(self.period, min_samples=2).to_numpy()
        z = np.where(sd > 0, (r - mn) / sd, 0.0)
        cum_z = np.cumsum(z)
        n = len(cum_z); w = self.period
        if n < w:
            return df.with_columns(pl.lit(np.nan).alias(out_col))
        wins = sliding_window_view(cum_z, w)
        max_stat = wins[:, -1:] - wins.min(axis=1, keepdims=True)
        pad = np.full(w - 1, np.nan, dtype=np.float64)
        return df.with_columns(pl.Series(out_col, np.concatenate([pad, max_stat.flatten()]), dtype=pl.Float64))

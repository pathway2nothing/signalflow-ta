"""Kalman filter residual feature."""
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import polars as pl

from signalflow.ta._compat import Feature


@dataclass
class KalmanResidual(Feature):
    """1D Kalman filter residual (innovation) on close.

    State = level. Process noise q, measurement variance fixed = 1.
    Lower q → more smoothing; higher q → tracks faster.
    Output = close_t − x_pred - what the filter didn't anticipate.
    """
    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["kalman_res_{q}"]
    q: float = 0.001

    def compute_pair(self, df):
        out = f"kalman_res_{self.q}"
        c = df.get_column("close").to_numpy().astype(np.float64)
        n = len(c)
        if n < 2:
            return df.with_columns(pl.Series(out, np.full(n, np.nan, dtype=np.float32)))
        x = np.zeros(n)
        p = 1.0
        r = 1.0
        x[0] = c[0]
        res = np.zeros(n)
        for t in range(1, n):
            x_pred = x[t - 1]
            p_pred = p + self.q
            k = p_pred / (p_pred + r)
            x[t] = x_pred + k * (c[t] - x_pred)
            p = (1 - k) * p_pred
            res[t] = c[t] - x_pred
        out_arr = res.astype(np.float32)
        out_arr[0] = np.nan
        return df.with_columns(pl.Series(out, out_arr))

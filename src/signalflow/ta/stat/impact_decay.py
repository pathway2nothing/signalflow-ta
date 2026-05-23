"""Almgren-Chriss-style market-impact decay.

Approximates the temporary price impact of recent order flow as
EMA(r·v) / (std(r) · sum(v)). Order blocks that move price with significant
volume produce large impact; this feature decays it exponentially.

Iter-29 stability: mean MI_normalised = 0.190 on D3 across 10 stable triples.

Reference: Almgren, R. & Chriss, N. (2001). Optimal execution of portfolio
transactions. Journal of Risk.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import polars as pl

from signalflow.core import feature
from signalflow.feature.base import Feature


@dataclass
@feature("stat/almgren_chriss_impact_decay")
class AlmgrenChrissImpactDecayStat(Feature):
    """Order-block impact decay (Almgren-Chriss-style).

    Attributes:
        ema: short EMA span over r·v (impact accumulation).
        period: longer window for vol normalisation and total volume sum.
    """

    ema: int = 10
    period: int = 60

    requires: ClassVar[list[str]] = ["close", "volume"]
    outputs: ClassVar[list[str]] = ["f51_ac_impact_{ema}_{period}"]
    test_params: ClassVar[list[dict]] = [
        {"ema": 10, "period": 60},
        {"ema": 30, "period": 240},
        {"ema": 5, "period": 30},
    ]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        out_col = f"f51_ac_impact_{self.ema}_{self.period}"
        c = df["close"].to_numpy().astype(np.float64)
        v = df["volume"].to_numpy().astype(np.float64)
        r = np.diff(np.log(c), prepend=c[0])
        rv = r * v
        ema_rv = pl.Series(rv).ewm_mean(span=self.ema).to_numpy()
        sd = pl.Series(r).rolling_std(self.period, min_samples=2).to_numpy()
        v_sum = pl.Series(v).rolling_sum(self.ema, min_samples=2).to_numpy()
        out = ema_rv / np.maximum(sd * v_sum, 1e-12)
        return df.with_columns(pl.Series(out_col, np.clip(out, -100, 100), dtype=pl.Float64))

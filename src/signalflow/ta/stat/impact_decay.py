"""Almgren-Chriss-style market-impact decay.

Approximates the temporary price impact of recent order flow as
EMA(r·v) / (std(r) · sum(v)). Order blocks that move price with significant
volume produce large impact; this feature decays it exponentially.

Iter-29 stability: mean MI_normalised = 0.190 on D3 across 10 stable triples.

Reference: Almgren, R. & Chriss, N. (2001). Optimal execution of portfolio
transactions. Journal of Risk.
"""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import polars as pl

from signalflow.ta._compat import Feature, feature
from signalflow.ta.stat._causal_helpers import log_returns, rolling_std, rolling_sum, truncated_ema


@dataclass
@feature("stat/almgren_chriss_impact_decay")
class AlmgrenChrissImpactDecayStat(Feature):
    """Order-block impact decay (Almgren-Chriss-style)."""

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
        r = log_returns(c)
        rv = r * v
        ema_rv = truncated_ema(rv, self.ema)
        sd = rolling_std(r, self.period)
        v_sum = rolling_sum(v, self.ema)
        out = ema_rv / np.maximum(sd * v_sum, 1e-12)
        return df.with_columns(pl.Series(out_col, np.clip(out, -100, 100), dtype=pl.Float64))

"""BTC stress contagion features.

Broadcast a BTC-derived stress signal (e.g. EMA of |z-score| of BTC returns)
to all pairs in the universe - treating BTC as a market-wide stress driver
rather than a per-pair input.

Validated empirically: best mean MI_normalised = 0.160 on forward volatility
regime (B1) with std 0.037 across 6 walk-forward folds, 37 stable triples
in iter-31 of sf-profit (cross-pair research).
"""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import polars as pl

from signalflow.ta._compat import feature
from signalflow.ta._compat import Feature

BTC_PAIR = "BTCUSDT"


@dataclass
@feature("global/btc_stress_contagion")
class BTCStressContagionStat(Feature):
    """EMA of BTC |z-score| returns broadcast across all pairs.

    Algorithm:
        1. From the BTC slice of the input df, compute log returns r_btc
           and rolling-σ baseline over `period`.
        2. z_btc = |r_btc| / σ_btc; clamp to 0 where σ=0.
        3. EMA of z_btc with characteristic time tau:
              stress[t] = α · z_btc[t] + (1 − α) · stress[t-1],   α = 1 − exp(−1/τ).
        4. Broadcast stress series back to all rows (by timestamp).

    Captures systemic stress propagation from the dominant asset to the
    full universe. Predictive of forward volatility regime on altcoins.
    """

    period: int = 240
    tau: int = 60

    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["btc_stress_{period}_{tau}"]
    test_params: ClassVar[list[dict]] = [
        {"period": 240, "tau": 60},
        {"period": 1440, "tau": 240},
        {"period": 480, "tau": 120},
    ]

    def compute(self, df: pl.DataFrame, context=None) -> pl.DataFrame:
        from signalflow.ta.stat._causal_helpers import log_returns, rolling_std, truncated_ema
        out_col = f"btc_stress_{self.period}_{self.tau}"
        df = df.sort(["pair", "timestamp"])
        btc = df.filter(pl.col("pair") == BTC_PAIR).sort("timestamp")
        if btc.height == 0:
            return df.with_columns(pl.lit(np.nan).alias(out_col))
        c_btc = btc["close"].to_numpy().astype(np.float64)
        btc_r = log_returns(c_btc)
        sd = rolling_std(btc_r, self.period)
        z = np.where(np.isfinite(sd) & (sd > 0), np.abs(btc_r) / np.maximum(sd, 1e-12), 0.0)
        stress = truncated_ema(z, self.tau)
        stress_df = pl.DataFrame({"timestamp": btc["timestamp"].to_list(), out_col: stress})
        return df.join(stress_df, on="timestamp", how="left")

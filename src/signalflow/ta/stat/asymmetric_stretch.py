"""Asymmetric SMA stretch with semivariance scaling.

(close - SMA) normalised by downside-semivariance when close < SMA, and by
upside-semivariance when close > SMA. Captures the asymmetric reversion
tendency of crypto markets where downside excursions occur via faster /
larger liquidations than upside accumulations.

Iter-29 stability: mean MI_normalised = 0.352 on D3 mean-reversion-event,
std 0.011 across 6 walk-forward folds — second-highest stable feature
after AllostaticLoadDirectionalStat (SOTA 0.467).

Reference: Barndorff-Nielsen, O. E., Kinnebross, S. & Shephard, N. (2010).
Measuring Downside Risk: Realised Semivariance.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import polars as pl

from signalflow.core import feature
from signalflow.feature.base import Feature


@dataclass
@feature("stat/asymmetric_sma_semi_vol_stretch")
class AsymmetricSemiVolStretchStat(Feature):
    """Price stretch normalised by directional semivariance.

    Algorithm:
        1. log returns r_t.
        2. rolling SMA over `period` of price.
        3. rolling sqrt(mean(r^2 | r<0)) = down_semi over `period`.
        4. rolling sqrt(mean(r^2 | r>0)) = up_semi over `period`.
        5. diff = close - SMA. Pick scale: down_semi if diff < 0 else up_semi.
        6. feature = diff / (scale * close).

    Attributes:
        period: rolling window for SMA and semivariance.
        source_col: input price column.
    """

    period: int = 60
    source_col: str = "close"

    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["f03_asym_sma_semi_{period}"]
    test_params: ClassVar[list[dict]] = [
        {"period": 240},
        {"period": 60},
        {"period": 480},
    ]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        out_col = f"f03_asym_sma_semi_{self.period}"
        c = df[self.source_col].to_numpy().astype(np.float64)
        r = np.diff(np.log(c), prepend=c[0])
        ma = pl.Series(c).rolling_mean(self.period, min_samples=2).to_numpy()
        down_sq = np.where(r < 0, r ** 2, 0.0)
        up_sq = np.where(r > 0, r ** 2, 0.0)
        d_semi = np.sqrt(pl.Series(down_sq).rolling_mean(self.period, min_samples=2).to_numpy())
        u_semi = np.sqrt(pl.Series(up_sq).rolling_mean(self.period, min_samples=2).to_numpy())
        diff = c - ma
        scale = np.where(diff < 0, d_semi, u_semi)
        scale = np.where(scale > 0, scale, 1e-12)
        out = diff / (scale * c)
        return df.with_columns(pl.Series(out_col, np.clip(out, -100, 100), dtype=pl.Float64))

"""Posterior P(reversal | recent z-stretch) — Bayes update on a rare event."""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import polars as pl

from signalflow.core import feature
from signalflow.feature.base import Feature


@dataclass
@feature("probabilistic/posterior_reversal")
class PosteriorReversalProb(Feature):
    """Bayesian ``P(mean-reversion in next bars | current z-stretch)``.

    Treats reversal as a rare event (prior = ``base_rate``); the likelihood
    multiplier is a sigmoid of the current ``|z|`` against
    ``stretch_threshold``. The posterior is then a calibrated probability
    that stays near the base rate during noise and climbs toward 1 only
    when a real overstretch is observed:

        prior        = base_rate
        likelihood_r = σ(likelihood_strength · (|z_now| − stretch_threshold))
        posterior    = prior · L_r / (prior · L_r + (1 − prior) · (1 − L_r))

    Output ∈ ``[0, 1]``, with mean ≈ ``base_rate`` on typical bars.

    Research provenance:
        iter-35 (sf-profit) — best soft-native feature for
        ``soft_D3_multi_horizon`` (soft MI = 0.179 on
        ``stretch_threshold=2.5``, ``z_window=240``).

    Attributes:
        price_col: Source price column. Default: ``"close"``.
        z_window: Trailing window for z-score computation. Default: 240.
        stretch_threshold: |z| above which the likelihood crosses 0.5.
            Default: 2.5.
        base_rate: Prior probability of a reversal event. Default: 0.05.
        likelihood_strength: Sigmoid steepness for the likelihood map.
            Default: 5.0.
    """

    price_col: str = "close"
    z_window: int = 240
    stretch_threshold: float = 2.5
    base_rate: float = 0.05
    likelihood_strength: float = 5.0

    requires: ClassVar[list[str]] = ["{price_col}"]
    outputs: ClassVar[list[str]] = ["post_revert_{z_window}_t{stretch_threshold}"]

    test_params: ClassVar[list[dict]] = [
        {"price_col": "close", "z_window": 240, "stretch_threshold": 2.0},
        {"price_col": "close", "z_window": 240, "stretch_threshold": 2.5},
        {"price_col": "close", "z_window": 1440, "stretch_threshold": 2.0},
    ]

    def __post_init__(self) -> None:
        if self.z_window < 20:
            raise ValueError("z_window must be >= 20")
        if not 0 < self.base_rate < 1:
            raise ValueError("base_rate must be in (0, 1)")
        if self.stretch_threshold <= 0 or self.likelihood_strength <= 0:
            raise ValueError("stretch_threshold and likelihood_strength must be > 0")

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        close = df.get_column(self.price_col).to_numpy().astype(np.float64)
        n = len(close)
        mu = np.full(n, np.nan)
        sd = np.full(n, np.nan)
        for i in range(self.z_window - 1, n):
            seg = close[i - self.z_window + 1 : i + 1]
            v = seg[np.isfinite(seg)]
            if len(v) < self.z_window // 4:
                continue
            mu[i] = v.mean()
            sd[i] = v.std(ddof=1)
        z = (close - mu) / np.where(sd > 1e-12, sd, np.nan)
        abs_z = np.abs(z)
        likelihood = 1.0 / (1.0 + np.exp(-self.likelihood_strength * (abs_z - self.stretch_threshold)))
        prior = self.base_rate
        post = (prior * likelihood) / (
            prior * likelihood + (1 - prior) * (1 - likelihood) + 1e-12
        )
        post = np.where(np.isfinite(z), post, np.nan)
        out_col = f"post_revert_{self.z_window}_t{self.stretch_threshold}"
        return df.with_columns(pl.Series(out_col, post, dtype=pl.Float64))

    @property
    def warmup(self) -> int:
        return self.z_window

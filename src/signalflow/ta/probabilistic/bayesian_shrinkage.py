"""Bayesian-shrinkage z-score → calibrated probability via Φ."""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import polars as pl
from scipy.stats import norm

from signalflow.ta._compat import Feature, feature
from signalflow.ta.probabilistic._helpers import log_returns


@dataclass
@feature("probabilistic/bayesian_shrinkage_z")
class BayesianShrinkageZscore(Feature):
    """z-score with Normal-Inverse-Gamma shrinkage, mapped through Φ.

    Standard rolling z-score is noisy on short windows because both µ and
    σ² are estimated from few observations. This variant pulls the
    estimates toward a long-run prior using a Normal-Inverse-Gamma
    conjugate update, then maps the regularised z through the standard
    Normal CDF to produce a calibrated probability in ``[0, 1]``.

    Posterior:
        µ_post  = (prior_strength · prior_mean + n · sample_mean) / (prior_strength + n)
        σ²_post = (prior_strength · prior_var + (n - 1) · sample_var) / (prior_strength + n - 1)
        z       = (r_t - µ_post) / sqrt(σ²_post)
        output  = Φ(z)

    The prior mean and variance are estimated from the entire history of
    the pair as a fixed bias - equivalent to having observed
    ``prior_strength`` historical bars.

    Output is closer to 0.5 (uninformative) on short noisy windows and
    sharpens to the tails only when there is real evidence.

    Research provenance: iter-35 (sf-profit) reference soft-native
    feature for the iter-33 ``soft_D3_*`` family.
    """

    price_col: str = "close"
    window: int = 240
    prior_strength: float = 60.0

    requires: ClassVar[list[str]] = ["{price_col}"]
    outputs: ClassVar[list[str]] = ["bshrink_z_{window}"]

    test_params: ClassVar[list[dict]] = [
        {"price_col": "close", "window": 240, "prior_strength": 60.0},
        {"price_col": "close", "window": 1440, "prior_strength": 240.0},
    ]

    def __post_init__(self) -> None:
        if self.window < 10:
            raise ValueError("window must be >= 10")
        if self.prior_strength <= 0:
            raise ValueError("prior_strength must be > 0")

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        close = df.get_column(self.price_col).to_numpy().astype(np.float64)
        r = log_returns(close)
        finite = r[np.isfinite(r)]
        n = len(r)
        out = np.full(n, np.nan)
        if len(finite) < 100:
            return df.with_columns(pl.Series(f"bshrink_z_{self.window}", out, dtype=pl.Float64))
        prior_mean = float(finite.mean())
        prior_var = float(finite.var(ddof=1))
        for i in range(self.window - 1, n):
            seg = r[i - self.window + 1 : i + 1]
            v = seg[np.isfinite(seg)]
            n_eff = len(v)
            if n_eff < self.window // 4:
                continue
            m = v.mean()
            s2 = v.var(ddof=1)
            post_mean = (self.prior_strength * prior_mean + n_eff * m) / (self.prior_strength + n_eff)
            post_var = (
                self.prior_strength * prior_var + (n_eff - 1) * s2
            ) / (self.prior_strength + n_eff - 1)
            if post_var <= 0:
                continue
            z = (r[i] - post_mean) / np.sqrt(post_var)
            out[i] = norm.cdf(z)
        return df.with_columns(pl.Series(f"bshrink_z_{self.window}", out, dtype=pl.Float64))

    @property
    def warmup(self) -> int:
        return self.window

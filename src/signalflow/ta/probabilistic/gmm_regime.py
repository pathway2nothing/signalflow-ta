"""GMM vol-regime posteriors - soft regime membership probabilities.

Two variants:
    GMMVolRegime3State - 3 components (low / mid / high)
    GMMVolRegime5State - 5 components anchored at vol quantiles
"""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import polars as pl

from signalflow.ta._compat import Feature, feature
from signalflow.ta.probabilistic._helpers import causal_rolling_logvol, log_returns


def _gmm_posterior(
    close: np.ndarray,
    window: int,
    smoother: int,
    quantiles: tuple[float, ...],
) -> list[np.ndarray]:
    """Causal Gaussian-mixture posterior with quantile-anchored centres.

    Fits a width-only Gaussian mixture each bar: the K regime centres are
    placed at the requested quantiles of the trailing ``window`` of log-vol,
    the common width is the trailing log-vol std, and posteriors are the
    softmax over per-component Gaussian likelihoods.

    Returns one ``np.ndarray`` per quantile, each of length ``len(close)``.
    All arrays sum to 1 elementwise (or NaN during warmup).
    """
    r = log_returns(close)
    log_rv = causal_rolling_logvol(r, smoother)
    n = len(close)
    K = len(quantiles)
    outs = [np.full(n, np.nan) for _ in range(K)]
    quantiles_arr = np.asarray(quantiles)
    for i in range(window - 1, n):
        seg = log_rv[i - window + 1 : i + 1]
        v = seg[~np.isnan(seg)]
        if len(v) < window // 4:
            continue
        centres = np.quantile(v, quantiles_arr)
        sigma = v.std(ddof=1)
        if sigma <= 0:
            continue
        x = log_rv[i]
        if not np.isfinite(x):
            continue
        sims = np.exp(-0.5 * ((x - centres) / sigma) ** 2)
        tot = sims.sum()
        if tot <= 0:
            continue
        for k in range(K):
            outs[k][i] = sims[k] / tot
    return outs


@dataclass
@feature("probabilistic/gmm_vol_regime_3state")
class GMMVolRegime3State(Feature):
    """3-component GMM vol regime posterior.

    Centres anchored at the 0.20, 0.50, 0.80 quantiles of rolling
    log-volatility. Returns three probability columns summing to 1.

    Research provenance:
        iter-34 (sf-profit) reported soft MI ≈ 0.114 against
        ``soft_F1_tail_anomaly`` (best of 5 tested soft-native features
        on a top-30 validated pool subset).
    """

    price_col: str = "close"
    window: int = 1440
    smoother: int = 60

    requires: ClassVar[list[str]] = ["{price_col}"]
    outputs: ClassVar[list[str]] = [
        "volreg3_low_{window}",
        "volreg3_mid_{window}",
        "volreg3_high_{window}",
    ]

    test_params: ClassVar[list[dict]] = [
        {"price_col": "close", "window": 1440, "smoother": 60},
        {"price_col": "close", "window": 2880, "smoother": 60},
    ]

    def __post_init__(self) -> None:
        if self.window < 100:
            raise ValueError("window must be >= 100")
        if self.smoother < 5 or self.smoother > self.window:
            raise ValueError("smoother must be in [5, window]")

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        close = df.get_column(self.price_col).to_numpy().astype(np.float64)
        p_low, p_mid, p_high = _gmm_posterior(close, self.window, self.smoother, (0.20, 0.50, 0.80))
        return df.with_columns(
            pl.Series(f"volreg3_low_{self.window}", p_low, dtype=pl.Float64),
            pl.Series(f"volreg3_mid_{self.window}", p_mid, dtype=pl.Float64),
            pl.Series(f"volreg3_high_{self.window}", p_high, dtype=pl.Float64),
        )

    @property
    def warmup(self) -> int:
        return self.window + self.smoother


@dataclass
@feature("probabilistic/gmm_vol_regime_5state")
class GMMVolRegime5State(Feature):
    """5-component GMM vol regime posterior - finer granularity than 3-state.

    Centres at 0.10 / 0.30 / 0.50 / 0.70 / 0.90 quantiles. Captures vol
    bands that the 3-state version groups together.

    Research provenance:
        iter-35 (sf-profit) - best soft-native feature for the
        ``hmm_vol_2state`` label (soft MI = 0.391 on ``volreg5_q90`` and
        0.385 on ``volreg5_q10``).
    """

    price_col: str = "close"
    window: int = 1440
    smoother: int = 60

    requires: ClassVar[list[str]] = ["{price_col}"]
    outputs: ClassVar[list[str]] = [
        "volreg5_q10_{window}",
        "volreg5_q30_{window}",
        "volreg5_q50_{window}",
        "volreg5_q70_{window}",
        "volreg5_q90_{window}",
    ]

    test_params: ClassVar[list[dict]] = [
        {"price_col": "close", "window": 1440, "smoother": 60},
        {"price_col": "close", "window": 2880, "smoother": 60},
    ]

    def __post_init__(self) -> None:
        if self.window < 100:
            raise ValueError("window must be >= 100")
        if self.smoother < 5 or self.smoother > self.window:
            raise ValueError("smoother must be in [5, window]")

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        close = df.get_column(self.price_col).to_numpy().astype(np.float64)
        outs = _gmm_posterior(close, self.window, self.smoother, (0.10, 0.30, 0.50, 0.70, 0.90))
        return df.with_columns(
            pl.Series(f"volreg5_q10_{self.window}", outs[0], dtype=pl.Float64),
            pl.Series(f"volreg5_q30_{self.window}", outs[1], dtype=pl.Float64),
            pl.Series(f"volreg5_q50_{self.window}", outs[2], dtype=pl.Float64),
            pl.Series(f"volreg5_q70_{self.window}", outs[3], dtype=pl.Float64),
            pl.Series(f"volreg5_q90_{self.window}", outs[4], dtype=pl.Float64),
        )

    @property
    def warmup(self) -> int:
        return self.window + self.smoother

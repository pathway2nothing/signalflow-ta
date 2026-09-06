"""Realised-volatility decomposition features.

These features split the total realised variance into structural components
(continuous vs jump, upside vs downside, sparse vs dense sampling). Each
component carries distinct predictive information about forward dynamics.
"""

import math
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import polars as pl

from signalflow.ta._compat import Feature, feature


@dataclass
@feature("stat/realised_bipower_variance_ratio")
class RealisedBipowerVarianceRatioStat(Feature):
    """BV / RV - fraction of variance from continuous (non-jump) component.

    Realised Variance RV = Σ r_t². Bipower Variation BV = (π/2)·Σ|r_t|·|r_{t-1}|.
    BV is jump-robust (jumps appear in only one of the two adjacent bars
    in the product), so BV/RV measures the continuous-volatility share.

    Low ratios indicate jump-dominated regimes - informative about
    forward jump risk and mean-reversion timing.
    """

    period: int = 15

    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["f30_bv_rv_{period}"]
    test_params: ClassVar[list[dict]] = [
        {"period": 15},
        {"period": 60},
        {"period": 240},
    ]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        out_col = f"f30_bv_rv_{self.period}"
        c = df["close"].to_numpy().astype(np.float64)
        r = np.diff(np.log(np.maximum(c, 1e-12)), prepend=np.log(max(float(c[0]), 1e-12)))
        rv = pl.Series(r ** 2).rolling_sum(self.period, min_samples=2).to_numpy()
        abs_r = np.abs(r)
        bv = (math.pi / 2.0) * pl.Series(abs_r * np.roll(abs_r, 1)).rolling_sum(
            self.period, min_samples=2).to_numpy()
        out = bv / np.maximum(rv, 1e-20)
        return df.with_columns(pl.Series(out_col, np.clip(out, 0, 2), dtype=pl.Float64))


@dataclass
@feature("stat/realised_semivariance_ratio")
class RealisedSemivarianceRatioStat(Feature):
    """Downside semivariance / upside semivariance - directional vol asymmetry.

    Downside RS = Σ r²·(r<0), Upside RS = Σ r²·(r>0). Their ratio quantifies
    structural panic-vs-rally asymmetry. High ratio indicates downside
    volatility dominating - typical of capitulation.

    Iter-29 stability: mean MI_normalised = 0.119 across 11 stable triples.
    """

    period: int = 30

    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["f31_rs_ratio_{period}"]
    test_params: ClassVar[list[dict]] = [
        {"period": 30},
        {"period": 120},
        {"period": 480},
    ]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        out_col = f"f31_rs_ratio_{self.period}"
        c = df["close"].to_numpy().astype(np.float64)
        r = np.diff(np.log(np.maximum(c, 1e-12)), prepend=np.log(max(float(c[0]), 1e-12)))
        down = np.where(r < 0, r ** 2, 0.0)
        up = np.where(r > 0, r ** 2, 0.0)
        rs_neg = pl.Series(down).rolling_sum(self.period, min_samples=2).to_numpy()
        rs_pos = pl.Series(up).rolling_sum(self.period, min_samples=2).to_numpy()
        out = rs_neg / np.maximum(rs_pos, 1e-20)
        return df.with_columns(pl.Series(out_col, np.clip(out, 0, 20), dtype=pl.Float64))


@dataclass
@feature("stat/jump_truncated_variance_ratio")
class JumpTruncatedVarianceRatioStat(Feature):
    """Variance of |z|<2 returns / total variance - diffusive variance share.

    Removes jumps (|z|>=2) and computes the variance share of the
    remaining continuous part. Complementary to BV/RV (which uses adjacent-bar
    bipower; this uses single-bar truncation).

    Iter-29 stability: mean MI_normalised = 0.13 across 12 stable triples.
    """

    period: int = 60

    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["f34_jt_ratio_{period}"]
    test_params: ClassVar[list[dict]] = [
        {"period": 60},
        {"period": 240},
        {"period": 480},
    ]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        out_col = f"f34_jt_ratio_{self.period}"
        c = df["close"].to_numpy().astype(np.float64)
        r = np.diff(np.log(np.maximum(c, 1e-12)), prepend=np.log(max(float(c[0]), 1e-12)))
        sd = pl.Series(r).rolling_std(self.period, min_samples=2).to_numpy()
        z = np.where(sd > 0, r / sd, 0.0)
        r_tr = np.where(np.abs(z) < 2.0, r, 0.0)
        var_tr = pl.Series(r_tr ** 2).rolling_sum(self.period, min_samples=2).to_numpy()
        var_total = pl.Series(r ** 2).rolling_sum(self.period, min_samples=2).to_numpy()
        out = var_tr / np.maximum(var_total, 1e-20)
        return df.with_columns(pl.Series(out_col, np.clip(out, 0, 1), dtype=pl.Float64))


@dataclass
@feature("stat/rv_semivariance_asymmetry")
class RVSemivarianceAsymmetryStat(Feature):
    """Signed semivariance asymmetry (RS+ - RS-) / (RS+ + RS-), range [-1, +1].

    Distinct from RealisedSemivarianceRatioStat (which is the ratio
    down/up); this is the bounded signed measure where positive values
    indicate upside-dominated variance and negative values downside.

    Iter-30 stability: top mean MI_normalised = 0.16 on D3 mean-reversion.

    Reference: Barndorff-Nielsen, Kinnebrock & Shephard (2010); Patton &
    Sheppard (2015) Review of Economics and Statistics 97(3):683-697.
    """

    period: int = 60

    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["f013_rv_asym_{period}"]
    test_params: ClassVar[list[dict]] = [
        {"period": 60},
        {"period": 240},
        {"period": 480},
    ]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        out_col = f"f013_rv_asym_{self.period}"
        c = df["close"].to_numpy().astype(np.float64)
        r = np.diff(np.log(np.maximum(c, 1e-12)), prepend=np.log(max(float(c[0]), 1e-12)))
        down = np.where(r < 0, r ** 2, 0.0)
        up = np.where(r > 0, r ** 2, 0.0)
        rs_n = pl.Series(down).rolling_sum(self.period, min_samples=2).to_numpy()
        rs_p = pl.Series(up).rolling_sum(self.period, min_samples=2).to_numpy()
        out = (rs_p - rs_n) / np.maximum(rs_p + rs_n, 1e-20)
        return df.with_columns(pl.Series(out_col, np.clip(out, -1, 1), dtype=pl.Float64))


@dataclass
@feature("stat/realized_quarticity")
class RealizedQuarticityStat(Feature):
    """Σ r^4 over rolling window - fourth-moment volatility-of-volatility.

    Realised quarticity is the natural estimator of integrated quarticity
    σ⁴, used to standardise tests of realised variance.

    Iter-30 stability: top mean MI_normalised = 0.119 across 9 stable triples.
    """

    period: int = 60

    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["f051_rq_{period}"]
    test_params: ClassVar[list[dict]] = [
        {"period": 60},
        {"period": 240},
        {"period": 480},
    ]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        out_col = f"f051_rq_{self.period}"
        c = df["close"].to_numpy().astype(np.float64)
        r = np.diff(np.log(np.maximum(c, 1e-12)), prepend=np.log(max(float(c[0]), 1e-12)))
        rq = pl.Series(r ** 4).rolling_sum(self.period, min_samples=2).to_numpy() * (self.period / 3.0)
        return df.with_columns(pl.Series(out_col, np.log1p(rq * 1e8), dtype=pl.Float64))


@dataclass
@feature("stat/pre_averaged_bipower_variation")
class PreAveragedBipowerVariationStat(Feature):
    """Pre-averaged bipower variation - k-bar averaged returns then BV.

    Noise-robust BV: averaging over k sub-bars reduces microstructure noise
    before computing bipower variation. Robust to both noise and jumps.

    Iter-30 stability: top mean MI_normalised = 0.112 across 9 stable triples.

    Reference: Jacod, Li, Mykland, Podolskij, Vetter (2009). Microstructure
    noise in the continuous case: the pre-averaging approach. SPA 119:2249-2276.
    """

    period: int = 60
    k: int = 5

    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["f054_pabv_{period}_{k}"]
    test_params: ClassVar[list[dict]] = [
        {"period": 60, "k": 5},
        {"period": 240, "k": 5},
        {"period": 480, "k": 10},
    ]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        import math
        out_col = f"f054_pabv_{self.period}_{self.k}"
        c = df["close"].to_numpy().astype(np.float64)
        c_avg = pl.Series(c).rolling_mean(self.k, min_samples=2).to_numpy()
        r = np.diff(np.log(c_avg), prepend=c_avg[0])
        abs_r = np.abs(r)
        bv = (math.pi / 2.0) * pl.Series(abs_r * np.roll(abs_r, 1)).rolling_sum(
            self.period, min_samples=2).to_numpy()
        return df.with_columns(pl.Series(out_col, np.log1p(bv * 1e8), dtype=pl.Float64))

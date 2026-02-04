# src/signalflow/ta/stat/memory.py
"""Time series memory measures - persistence, mean-reversion, diffusion, oscillator dynamics."""
from dataclasses import dataclass

import numpy as np
import polars as pl

from signalflow import sf_component
from signalflow.feature.base import Feature
from typing import ClassVar


@dataclass
@sf_component(name="stat/hurst")
class HurstStat(Feature):
    """Rolling Hurst Exponent.

    Measures long-term memory of time series.

    H < 0.5: mean-reverting (anti-persistent)
    H = 0.5: random walk (no memory)
    H > 0.5: trending (persistent)

    Uses Rescaled Range (R/S) method.

    Formula:
        R(n) = max(cumsum(x - mean)) - min(cumsum(x - mean))
        S(n) = stdev(x)
        H = log(R/S) / log(n)

    Reference: https://en.wikipedia.org/wiki/Hurst_exponent
    """

    source_col: str = "close"
    period: int = 100

    requires = ["{source_col}"]
    outputs = ["{source_col}_hurst_{period}"]

    def __post_init__(self):
        if self.period < 20:
            raise ValueError(f"period must be >= 20 for reliable Hurst estimate, got {self.period}")

    def _hurst_rs(self, ts: np.ndarray) -> float:
        """Compute Hurst via R/S method."""
        n = len(ts)
        if n < 20:
            return np.nan

        mean = np.mean(ts)
        std = np.std(ts, ddof=1)
        if std < 1e-10:
            return np.nan

        y = np.cumsum(ts - mean)
        r = np.max(y) - np.min(y)

        rs = r / std

        if rs > 0:
            return np.log(rs) / np.log(n)
        return np.nan

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        values = df[self.source_col].to_numpy()
        n = len(values)

        hurst = np.full(n, np.nan)
        for i in range(self.period - 1, n):
            window = values[i - self.period + 1:i + 1]
            hurst[i] = self._hurst_rs(window)

        return df.with_columns(
            pl.Series(name=f"{self.source_col}_hurst_{self.period}", values=hurst)
        )

    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 100},
        {"source_col": "close", "period": 200},
        {"source_col": "close", "period": 500},
    ]

    @property
    def warmup(self) -> int:
        return self.period * 5


@dataclass
@sf_component(name="stat/autocorr")
class AutocorrStat(Feature):
    """Rolling Autocorrelation.

    Correlation of series with its lagged version.

    High positive: trending/momentum
    High negative: mean-reverting
    Near zero: random

    Formula:
        ACF(lag) = corr(x[t], x[t-lag])

    Reference: https://en.wikipedia.org/wiki/Autocorrelation
    """

    source_col: str = "close"
    period: int = 30
    lag: int = 1

    requires = ["{source_col}"]
    outputs = ["{source_col}_acf{lag}_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        values = df[self.source_col].to_numpy()
        n = len(values)

        acf = np.full(n, np.nan)
        for i in range(self.period + self.lag - 1, n):
            x = values[i - self.period + 1:i + 1]
            x_lag = values[i - self.period + 1 - self.lag:i + 1 - self.lag]

            if len(x) == len(x_lag) and len(x) > 0:
                corr = np.corrcoef(x, x_lag)[0, 1]
                acf[i] = corr if not np.isnan(corr) else 0

        return df.with_columns(
            pl.Series(name=f"{self.source_col}_acf{self.lag}_{self.period}", values=acf)
        )

    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 30, "lag": 1},
        {"source_col": "close", "period": 60, "lag": 5},
        {"source_col": "close", "period": 120, "lag": 10},
    ]

    @property
    def warmup(self) -> int:
        return (self.period + self.lag) * 5


@dataclass
@sf_component(name="stat/variance_ratio")
class VarianceRatioStat(Feature):
    """Rolling Variance Ratio.

    Tests random walk hypothesis by comparing variance at different horizons.

    VR = Var(k-period returns) / (k * Var(1-period returns))

    VR ≈ 1: random walk
    VR > 1: positive autocorrelation (momentum)
    VR < 1: negative autocorrelation (mean-reversion)

    Reference: Lo & MacKinlay (1988)
    https://en.wikipedia.org/wiki/Variance_ratio_test
    """

    source_col: str = "close"
    period: int = 50
    k: int = 5

    requires = ["{source_col}"]
    outputs = ["{source_col}_vr{k}_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        values = df[self.source_col].to_numpy()
        n = len(values)

        log_ret = np.diff(np.log(values), prepend=np.nan)

        vr = np.full(n, np.nan)
        for i in range(self.period + self.k - 1, n):
            window_1 = log_ret[i - self.period + 1:i + 1]

            log_ret_k = np.log(values[i - self.period + 1 + self.k:i + 1]) - \
                        np.log(values[i - self.period + 1:i + 1 - self.k])

            var_1 = np.nanvar(window_1, ddof=1)
            var_k = np.nanvar(log_ret_k, ddof=1)

            if var_1 > 1e-10:
                vr[i] = var_k / (self.k * var_1)

        return df.with_columns(
            pl.Series(name=f"{self.source_col}_vr{self.k}_{self.period}", values=vr)
        )

    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 50, "k": 5},
        {"source_col": "close", "period": 100, "k": 10},
        {"source_col": "close", "period": 200, "k": 20},
    ]

    @property
    def warmup(self) -> int:
        return (self.period + self.k) * 5


# --- Diffusion features ---


@dataclass
@sf_component(name="stat/diffusion_coeff")
class DiffusionCoeffStat(Feature):
    """Rolling Diffusion Coefficient.

    D = Var(log-returns) / (2 × Δt)

    Measures local intensity of random movement.
    In a pure random walk, D is constant; changes in D
    signal regime transitions.

    Interpretation:
    - Rising D: volatility expansion, market becoming more random
    - Falling D: volatility contraction, potential trend forming
    - Stable D: stationary regime

    Reference: Einstein diffusion equation, MSD = 2D×t
    """

    source_col: str = "close"
    period: int = 30
    normalized: bool = False
    norm_period: int | None = None

    requires = ["{source_col}"]
    outputs = ["{source_col}_diffcoeff_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        values = df[self.source_col].to_numpy()
        n = len(values)

        log_ret = np.full(n, np.nan)
        for i in range(1, n):
            if values[i - 1] > 0 and values[i] > 0:
                log_ret[i] = np.log(values[i] / values[i - 1])

        dc = np.full(n, np.nan)
        for i in range(self.period, n):
            window = log_ret[i - self.period + 1:i + 1]
            valid = window[~np.isnan(window)]
            if len(valid) > 2:
                dc[i] = np.var(valid, ddof=1) / 2.0

        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window
            norm_window = self.norm_period or get_norm_window(self.period)
            dc = normalize_zscore(dc, window=norm_window)

        suffix = "_norm" if self.normalized else ""
        col_name = f"{self.source_col}_diffcoeff_{self.period}{suffix}"
        return df.with_columns(
            pl.Series(name=col_name, values=dc)
        )

    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 30},
        {"source_col": "close", "period": 60},
        {"source_col": "close", "period": 100},
        {"source_col": "close", "period": 30, "normalized": True},
    ]

    @property
    def warmup(self) -> int:
        base = self.period * 5
        if self.normalized:
            from signalflow.ta._normalization import get_norm_window
            norm_window = self.norm_period or get_norm_window(self.period)
            return base + norm_window
        return base


@dataclass
@sf_component(name="stat/anomalous_diffusion")
class AnomalousDiffusionStat(Feature):
    """Anomalous Diffusion Exponent (α).

    MSD(τ) ~ τ^α  →  α = log(MSD(τ2)/MSD(τ1)) / log(τ2/τ1)

    Estimated by comparing MSD at two time scales within a rolling window.

    Interpretation:
    - α ≈ 1: normal diffusion (random walk)
    - α > 1: super-diffusion (trending / persistent)
    - α < 1: sub-diffusion (mean-reverting / anti-persistent)
    - More responsive than Hurst for short windows

    Reference: Anomalous diffusion in physics
    https://en.wikipedia.org/wiki/Anomalous_diffusion
    """

    source_col: str = "close"
    period: int = 60
    tau_short: int = 1
    tau_long: int = 10
    normalized: bool = False
    norm_period: int | None = None

    requires = ["{source_col}"]
    outputs = ["{source_col}_adiff_{period}"]

    def __post_init__(self):
        if self.tau_long <= self.tau_short:
            raise ValueError(f"tau_long ({self.tau_long}) must be > tau_short ({self.tau_short})")

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        values = df[self.source_col].to_numpy()
        n = len(values)
        log_values = np.log(np.maximum(values, 1e-10))

        adiff = np.full(n, np.nan)
        log_ratio = np.log(self.tau_long / self.tau_short)

        for i in range(self.period + self.tau_long - 1, n):
            window = log_values[i - self.period + 1:i + 1]
            w = len(window)

            # MSD for tau_short
            displacements_short = window[self.tau_short:] - window[:-self.tau_short]
            msd_short = np.mean(displacements_short ** 2) if len(displacements_short) > 0 else 0

            # MSD for tau_long
            displacements_long = window[self.tau_long:] - window[:-self.tau_long]
            msd_long = np.mean(displacements_long ** 2) if len(displacements_long) > 0 else 0

            if msd_short > 1e-20 and msd_long > 1e-20:
                adiff[i] = np.log(msd_long / msd_short) / log_ratio

        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window
            norm_window = self.norm_period or get_norm_window(self.period)
            adiff = normalize_zscore(adiff, window=norm_window)

        suffix = "_norm" if self.normalized else ""
        col_name = f"{self.source_col}_adiff_{self.period}{suffix}"
        return df.with_columns(
            pl.Series(name=col_name, values=adiff)
        )

    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 60, "tau_short": 1, "tau_long": 10},
        {"source_col": "close", "period": 100, "tau_short": 1, "tau_long": 20},
        {"source_col": "close", "period": 200, "tau_short": 5, "tau_long": 50},
        {"source_col": "close", "period": 60, "normalized": True},
    ]

    @property
    def warmup(self) -> int:
        base = (self.period + self.tau_long) * 5
        if self.normalized:
            from signalflow.ta._normalization import get_norm_window
            norm_window = self.norm_period or get_norm_window(self.period)
            return base + norm_window
        return base


@dataclass
@sf_component(name="stat/msd_ratio")
class MsdRatioStat(Feature):
    """Mean Squared Displacement Ratio.

    MSDR = MSD(2τ) / MSD(τ)

    A fast, parameter-light test for diffusion regime.

    Interpretation:
    - MSDR ≈ 2: normal diffusion (random walk)
    - MSDR > 2: super-diffusion (trending)
    - MSDR < 2: sub-diffusion (mean-reverting)
    - Simpler and faster than full anomalous diffusion exponent

    Reference: MSD scaling analysis
    """

    source_col: str = "close"
    period: int = 60
    tau: int = 5
    normalized: bool = False
    norm_period: int | None = None

    requires = ["{source_col}"]
    outputs = ["{source_col}_msdr_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        values = df[self.source_col].to_numpy()
        n = len(values)
        log_values = np.log(np.maximum(values, 1e-10))

        tau2 = 2 * self.tau
        msdr = np.full(n, np.nan)

        for i in range(self.period + tau2 - 1, n):
            window = log_values[i - self.period + 1:i + 1]

            # MSD for tau
            disp_tau = window[self.tau:] - window[:-self.tau]
            msd_tau = np.mean(disp_tau ** 2)

            # MSD for 2*tau
            disp_2tau = window[tau2:] - window[:-tau2]
            msd_2tau = np.mean(disp_2tau ** 2)

            if msd_tau > 1e-20:
                msdr[i] = msd_2tau / msd_tau

        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window
            norm_window = self.norm_period or get_norm_window(self.period)
            msdr = normalize_zscore(msdr, window=norm_window)

        suffix = "_norm" if self.normalized else ""
        col_name = f"{self.source_col}_msdr_{self.period}{suffix}"
        return df.with_columns(
            pl.Series(name=col_name, values=msdr)
        )

    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 60, "tau": 5},
        {"source_col": "close", "period": 100, "tau": 10},
        {"source_col": "close", "period": 200, "tau": 20},
        {"source_col": "close", "period": 60, "tau": 5, "normalized": True},
    ]

    @property
    def warmup(self) -> int:
        base = (self.period + 2 * self.tau) * 5
        if self.normalized:
            from signalflow.ta._normalization import get_norm_window
            norm_window = self.norm_period or get_norm_window(self.period)
            return base + norm_window
        return base


# --- Oscillator dynamics ---


@dataclass
@sf_component(name="stat/spring_constant")
class SpringConstantStat(Feature):
    """Rolling Spring Constant (k) - Mean Reversion Strength.

    Models price as a spring: restoring_force ∝ -k × displacement.

    displacement = ln(Close) - ln(SMA)
    Δdisplacement = displacement[t] - displacement[t-1]
    k = -regression_slope(Δdisplacement ~ displacement[t-1])

    Interpretation:
    - k > 0: mean-reverting (price pulled back toward MA)
    - k ≈ 0: random walk (no restoring force)
    - k < 0: anti-restoring (trend reinforcing, momentum)
    - Higher k: faster mean reversion

    Reference: Ornstein-Uhlenbeck process, Hooke's law
    """

    source_col: str = "close"
    period: int = 60
    ma_period: int = 50
    normalized: bool = False
    norm_period: int | None = None

    requires = ["{source_col}"]
    outputs = ["{source_col}_spring_k_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        values = df[self.source_col].to_numpy()
        n = len(values)

        # MA equilibrium
        ma = np.full(n, np.nan)
        for i in range(self.ma_period - 1, n):
            ma[i] = np.mean(values[i - self.ma_period + 1:i + 1])

        # Log displacement
        log_values = np.log(np.maximum(values, 1e-10))
        log_ma = np.log(np.maximum(ma, 1e-10))
        displacement = log_values - log_ma

        # Change in displacement
        d_displacement = np.full(n, np.nan)
        for i in range(1, n):
            if not np.isnan(displacement[i]) and not np.isnan(displacement[i - 1]):
                d_displacement[i] = displacement[i] - displacement[i - 1]

        # Rolling regression: Δdisp = -k * disp[t-1] + c
        spring_k = np.full(n, np.nan)
        start = max(self.ma_period, self.period)
        for i in range(start, n):
            x = displacement[i - self.period + 1:i]  # displacement[t-1]
            y = d_displacement[i - self.period + 2:i + 1]  # Δdisplacement[t]

            valid = ~(np.isnan(x) | np.isnan(y))
            x_v = x[valid]
            y_v = y[valid]

            if len(x_v) > 5:
                x_mean = np.mean(x_v)
                y_mean = np.mean(y_v)
                ss_xx = np.sum((x_v - x_mean) ** 2)
                if ss_xx > 1e-20:
                    slope = np.sum((x_v - x_mean) * (y_v - y_mean)) / ss_xx
                    spring_k[i] = -slope  # k = -slope (restoring force convention)

        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window
            norm_window = self.norm_period or get_norm_window(self.period)
            spring_k = normalize_zscore(spring_k, window=norm_window)

        suffix = "_norm" if self.normalized else ""
        col_name = f"{self.source_col}_spring_k_{self.period}{suffix}"
        return df.with_columns(
            pl.Series(name=col_name, values=spring_k)
        )

    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 60, "ma_period": 50},
        {"source_col": "close", "period": 100, "ma_period": 100},
        {"source_col": "close", "period": 200, "ma_period": 200},
        {"source_col": "close", "period": 60, "ma_period": 50, "normalized": True},
    ]

    @property
    def warmup(self) -> int:
        base = max(self.ma_period, self.period) * 5
        if self.normalized:
            from signalflow.ta._normalization import get_norm_window
            norm_window = self.norm_period or get_norm_window(self.period)
            return base + norm_window
        return base


@dataclass
@sf_component(name="stat/damping_ratio")
class DampingRatioStat(Feature):
    """Rolling Damping Ratio (ζ) - Oscillation Decay Characterization.

    Estimates damping from successive peak amplitudes of displacement
    from equilibrium (MA).

    Uses logarithmic decrement: δ = ln(A_n / A_{n+1})
    ζ = δ / sqrt(4π² + δ²)

    Interpretation:
    - ζ < 1: underdamped (oscillating around MA, mean-reverting with overshoot)
    - ζ ≈ 1: critically damped (fastest return without oscillation)
    - ζ > 1: overdamped (slow monotonic return, or trending away)
    - Low ζ: more oscillatory, good for mean-reversion strategies
    - High ζ: more trendy, good for momentum strategies

    Reference: Damped harmonic oscillator theory
    https://en.wikipedia.org/wiki/Damping#Damping_ratio
    """

    source_col: str = "close"
    period: int = 60
    ma_period: int = 50
    normalized: bool = False
    norm_period: int | None = None

    requires = ["{source_col}"]
    outputs = ["{source_col}_damping_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        values = df[self.source_col].to_numpy()
        n = len(values)

        # MA equilibrium
        ma = np.full(n, np.nan)
        for i in range(self.ma_period - 1, n):
            ma[i] = np.mean(values[i - self.ma_period + 1:i + 1])

        # Displacement
        log_values = np.log(np.maximum(values, 1e-10))
        log_ma = np.log(np.maximum(ma, 1e-10))
        displacement = log_values - log_ma

        damping = np.full(n, np.nan)
        start = max(self.ma_period, self.period)

        for i in range(start, n):
            window = displacement[i - self.period + 1:i + 1]
            valid = window[~np.isnan(window)]
            if len(valid) < 10:
                continue

            # Find peaks (local maxima of |displacement|)
            abs_disp = np.abs(valid)
            peaks = []
            for j in range(1, len(abs_disp) - 1):
                if abs_disp[j] > abs_disp[j - 1] and abs_disp[j] > abs_disp[j + 1]:
                    peaks.append(abs_disp[j])

            if len(peaks) >= 2:
                # Average logarithmic decrement across successive peaks
                decrements = []
                for j in range(len(peaks) - 1):
                    if peaks[j + 1] > 1e-10 and peaks[j] > 1e-10:
                        decrements.append(np.log(peaks[j] / peaks[j + 1]))

                if len(decrements) > 0:
                    delta = np.mean(decrements)
                    # ζ = δ / sqrt(4π² + δ²)
                    damping[i] = delta / np.sqrt(4 * np.pi ** 2 + delta ** 2)

        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window
            norm_window = self.norm_period or get_norm_window(self.period)
            damping = normalize_zscore(damping, window=norm_window)

        suffix = "_norm" if self.normalized else ""
        col_name = f"{self.source_col}_damping_{self.period}{suffix}"
        return df.with_columns(
            pl.Series(name=col_name, values=damping)
        )

    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 60, "ma_period": 50},
        {"source_col": "close", "period": 100, "ma_period": 100},
        {"source_col": "close", "period": 200, "ma_period": 200},
        {"source_col": "close", "period": 60, "ma_period": 50, "normalized": True},
    ]

    @property
    def warmup(self) -> int:
        base = max(self.ma_period, self.period) * 5
        if self.normalized:
            from signalflow.ta._normalization import get_norm_window
            norm_window = self.norm_period or get_norm_window(self.period)
            return base + norm_window
        return base


@dataclass
@sf_component(name="stat/natural_frequency")
class NaturalFrequencyStat(Feature):
    """Rolling Natural Frequency (ω₀) via zero-crossing rate.

    Estimates the dominant oscillation frequency of price around
    its moving average by counting zero-crossings of displacement.

    ω₀ ≈ π × (zero_crossings / period)

    Interpretation:
    - High ω₀: rapid oscillation around MA (choppy, mean-reverting)
    - Low ω₀: slow oscillation (trending, long cycles)
    - Stable ω₀: consistent market character
    - Changing ω₀: regime transition

    Reference: Zero-crossing rate frequency estimation
    https://en.wikipedia.org/wiki/Zero-crossing_rate
    """

    source_col: str = "close"
    period: int = 60
    ma_period: int = 50
    normalized: bool = False
    norm_period: int | None = None

    requires = ["{source_col}"]
    outputs = ["{source_col}_natfreq_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        values = df[self.source_col].to_numpy()
        n = len(values)

        # MA equilibrium
        ma = np.full(n, np.nan)
        for i in range(self.ma_period - 1, n):
            ma[i] = np.mean(values[i - self.ma_period + 1:i + 1])

        # Displacement
        displacement = values - ma

        natfreq = np.full(n, np.nan)
        start = max(self.ma_period, self.period)

        for i in range(start, n):
            window = displacement[i - self.period + 1:i + 1]
            valid = window[~np.isnan(window)]
            if len(valid) < 4:
                continue

            # Count zero crossings
            signs = np.sign(valid)
            crossings = np.sum(np.abs(np.diff(signs)) > 0)

            # ω₀ = π × crossings / N
            natfreq[i] = np.pi * crossings / len(valid)

        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window
            norm_window = self.norm_period or get_norm_window(self.period)
            natfreq = normalize_zscore(natfreq, window=norm_window)

        suffix = "_norm" if self.normalized else ""
        col_name = f"{self.source_col}_natfreq_{self.period}{suffix}"
        return df.with_columns(
            pl.Series(name=col_name, values=natfreq)
        )

    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 60, "ma_period": 50},
        {"source_col": "close", "period": 100, "ma_period": 100},
        {"source_col": "close", "period": 200, "ma_period": 200},
        {"source_col": "close", "period": 60, "ma_period": 50, "normalized": True},
    ]

    @property
    def warmup(self) -> int:
        base = max(self.ma_period, self.period) * 5
        if self.normalized:
            from signalflow.ta._normalization import get_norm_window
            norm_window = self.norm_period or get_norm_window(self.period)
            return base + norm_window
        return base


# --- Elasticity & Material Science ---


@dataclass
@sf_component(name="stat/plastic_strain")
class PlasticStrainStat(Feature):
    """Plastic Strain Ratio - fraction of non-reversible deformation.

    Measures how much price displacement does NOT revert.

    plastic_ratio = 1 - corr(displacement, -Δdisplacement)

    Interpretation:
    - Ratio near 0: elastic (price reverts after displacement)
    - Ratio near 1: plastic (displacement is permanent, breakout)
    - Rising ratio: transition from mean-reversion to trending
    - Complements spring_constant: k tells strength, plastic_ratio tells regime

    Reference: Plastic deformation in material science
    """

    source_col: str = "close"
    period: int = 60
    ma_period: int = 50
    normalized: bool = False
    norm_period: int | None = None

    requires = ["{source_col}"]
    outputs = ["{source_col}_plastic_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        values = df[self.source_col].to_numpy()
        n = len(values)

        ma = np.full(n, np.nan)
        for i in range(self.ma_period - 1, n):
            ma[i] = np.mean(values[i - self.ma_period + 1:i + 1])

        log_values = np.log(np.maximum(values, 1e-10))
        log_ma = np.log(np.maximum(ma, 1e-10))
        displacement = log_values - log_ma

        d_displacement = np.full(n, np.nan)
        for i in range(1, n):
            if not np.isnan(displacement[i]) and not np.isnan(displacement[i - 1]):
                d_displacement[i] = displacement[i] - displacement[i - 1]

        plastic = np.full(n, np.nan)
        start = max(self.ma_period, self.period)
        for i in range(start, n):
            x = displacement[i - self.period + 1:i]
            y = -d_displacement[i - self.period + 2:i + 1]

            valid = ~(np.isnan(x) | np.isnan(y))
            x_v = x[valid]
            y_v = y[valid]

            if len(x_v) > 5:
                corr_mat = np.corrcoef(x_v, y_v)
                corr = corr_mat[0, 1]
                if not np.isnan(corr):
                    plastic[i] = 1.0 - max(corr, 0)

        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window
            norm_window = self.norm_period or get_norm_window(self.period)
            plastic = normalize_zscore(plastic, window=norm_window)

        suffix = "_norm" if self.normalized else ""
        col_name = f"{self.source_col}_plastic_{self.period}{suffix}"
        return df.with_columns(
            pl.Series(name=col_name, values=plastic)
        )

    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 60, "ma_period": 50},
        {"source_col": "close", "period": 100, "ma_period": 100},
        {"source_col": "close", "period": 200, "ma_period": 200},
        {"source_col": "close", "period": 60, "ma_period": 50, "normalized": True},
    ]

    @property
    def warmup(self) -> int:
        base = max(self.ma_period, self.period) * 5
        if self.normalized:
            from signalflow.ta._normalization import get_norm_window
            norm_window = self.norm_period or get_norm_window(self.period)
            return base + norm_window
        return base


@dataclass
@sf_component(name="stat/escape_velocity")
class EscapeVelocityStat(Feature):
    """Escape Velocity - velocity needed to break free from MA attraction.

    v_escape = sqrt(2 × k × |displacement|)

    Compares current velocity to escape velocity.
    Ratio > 1 means price has enough momentum to break away.

    Outputs the ratio: current_velocity / escape_velocity.

    Interpretation:
    - Ratio > 1: velocity exceeds escape threshold → breakout likely
    - Ratio < 1: velocity insufficient → likely pulled back to MA
    - Ratio near 1: on the edge (critical point)

    Reference: Orbital mechanics escape velocity
    """

    source_col: str = "close"
    period: int = 60
    ma_period: int = 50
    normalized: bool = False
    norm_period: int | None = None

    requires = ["{source_col}"]
    outputs = ["{source_col}_vesc_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        values = df[self.source_col].to_numpy()
        n = len(values)

        ma = np.full(n, np.nan)
        for i in range(self.ma_period - 1, n):
            ma[i] = np.mean(values[i - self.ma_period + 1:i + 1])

        log_values = np.log(np.maximum(values, 1e-10))
        log_ma = np.log(np.maximum(ma, 1e-10))
        displacement = log_values - log_ma

        d_displacement = np.full(n, np.nan)
        for i in range(1, n):
            if not np.isnan(displacement[i]) and not np.isnan(displacement[i - 1]):
                d_displacement[i] = displacement[i] - displacement[i - 1]

        velocity = np.full(n, np.nan)
        for i in range(1, n):
            if values[i - 1] > 0 and values[i] > 0:
                velocity[i] = np.log(values[i] / values[i - 1])

        vesc_ratio = np.full(n, np.nan)
        start = max(self.ma_period, self.period)

        for i in range(start, n):
            # Estimate spring constant k
            x = displacement[i - self.period + 1:i]
            y = d_displacement[i - self.period + 2:i + 1]
            valid = ~(np.isnan(x) | np.isnan(y))
            x_v = x[valid]
            y_v = y[valid]

            if len(x_v) > 5 and not np.isnan(velocity[i]) and not np.isnan(displacement[i]):
                x_mean = np.mean(x_v)
                y_mean = np.mean(y_v)
                ss_xx = np.sum((x_v - x_mean) ** 2)
                if ss_xx > 1e-20:
                    slope = np.sum((x_v - x_mean) * (y_v - y_mean)) / ss_xx
                    k = max(-slope, 1e-20)
                    abs_disp = np.abs(displacement[i])
                    if abs_disp > 1e-10:
                        v_escape = np.sqrt(2 * k * abs_disp)
                        abs_v = np.abs(velocity[i])
                        if v_escape > 1e-20:
                            vesc_ratio[i] = abs_v / v_escape

        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window
            norm_window = self.norm_period or get_norm_window(self.period)
            vesc_ratio = normalize_zscore(vesc_ratio, window=norm_window)

        suffix = "_norm" if self.normalized else ""
        col_name = f"{self.source_col}_vesc_{self.period}{suffix}"
        return df.with_columns(
            pl.Series(name=col_name, values=vesc_ratio)
        )

    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 60, "ma_period": 50},
        {"source_col": "close", "period": 100, "ma_period": 100},
        {"source_col": "close", "period": 200, "ma_period": 200},
        {"source_col": "close", "period": 60, "ma_period": 50, "normalized": True},
    ]

    @property
    def warmup(self) -> int:
        base = max(self.ma_period, self.period) * 5
        if self.normalized:
            from signalflow.ta._normalization import get_norm_window
            norm_window = self.norm_period or get_norm_window(self.period)
            return base + norm_window
        return base


@dataclass
@sf_component(name="stat/correlation_length")
class CorrelationLengthStat(Feature):
    """Correlation Length - distance to first autocorrelation zero-crossing.

    Number of lags until autocorrelation of log-returns first crosses zero.

    Interpretation:
    - Long correlation length: long-range dependencies (trending)
    - Short correlation length: quickly decorrelating (mean-reverting/random)
    - Increasing length: approaching critical point (breakout)
    - Decreasing length: regime settling into randomness

    Reference: Correlation length in statistical mechanics / critical phenomena
    """

    source_col: str = "close"
    period: int = 100
    max_lag: int = 20
    normalized: bool = False
    norm_period: int | None = None

    requires = ["{source_col}"]
    outputs = ["{source_col}_corrlen_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        values = df[self.source_col].to_numpy()
        n = len(values)

        log_ret = np.full(n, np.nan)
        for i in range(1, n):
            if values[i - 1] > 0 and values[i] > 0:
                log_ret[i] = np.log(values[i] / values[i - 1])

        corrlen = np.full(n, np.nan)
        for i in range(self.period + self.max_lag - 1, n):
            window = log_ret[i - self.period + 1:i + 1]
            valid = window[~np.isnan(window)]
            if len(valid) < self.max_lag + 5:
                continue

            # Find first zero-crossing of ACF
            found = False
            for lag in range(1, self.max_lag + 1):
                x = valid[lag:]
                x_lag = valid[:-lag]
                if len(x) > 3:
                    corr = np.corrcoef(x, x_lag)[0, 1]
                    if not np.isnan(corr) and corr <= 0:
                        corrlen[i] = lag
                        found = True
                        break

            if not found:
                corrlen[i] = self.max_lag

        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window
            norm_window = self.norm_period or get_norm_window(self.period)
            corrlen = normalize_zscore(corrlen, window=norm_window)

        suffix = "_norm" if self.normalized else ""
        col_name = f"{self.source_col}_corrlen_{self.period}{suffix}"
        return df.with_columns(
            pl.Series(name=col_name, values=corrlen)
        )

    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 100, "max_lag": 20},
        {"source_col": "close", "period": 200, "max_lag": 30},
        {"source_col": "close", "period": 100, "max_lag": 20, "normalized": True},
    ]

    @property
    def warmup(self) -> int:
        base = (self.period + self.max_lag) * 5
        if self.normalized:
            from signalflow.ta._normalization import get_norm_window
            norm_window = self.norm_period or get_norm_window(self.period)
            return base + norm_window
        return base

# src/signalflow/ta/momentum/kinematics.py
"""Kinematic indicators - higher-order derivatives of price movement."""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import polars as pl

from signalflow.core import feature
from signalflow.feature.base import Feature
from signalflow.ta._numba_kernels import (
    velocity_kernel as _velocity_kernel,
    sma_nb as _sma_nb,
    rolling_mean_nan as _rolling_mean_nan,
)


@dataclass
@feature("momentum/acceleration")
class AccelerationMom(Feature):
    """Price Acceleration (second derivative).

    Rate of change of velocity (ROC of log-returns).

    velocity = ln(Close / Close[lag])
    acceleration = velocity - velocity[lag]

    Interpretation:
    - Positive acceleration + positive velocity: trend strengthening
    - Negative acceleration + positive velocity: trend weakening (early reversal signal)
    - Sign change in acceleration precedes sign change in velocity

    Reference: Kinematic analysis applied to financial time series
    """

    source_col: str = "close"
    lag: int = 1
    smooth: int = 1
    normalized: bool = False
    norm_period: int | None = None

    requires: ClassVar[list[str]] = ["{source_col}"]
    outputs: ClassVar[list[str]] = ["{source_col}_accel_{lag}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        values = df[self.source_col].to_numpy().astype(np.float64)
        n = len(values)

        # Velocity: log-return over lag (vectorized)
        velocity = np.full(n, np.nan)
        safe = (values[self.lag:] > 0) & (values[:-self.lag] > 0)
        velocity[self.lag:] = np.where(safe, np.log(values[self.lag:] / values[:-self.lag]), np.nan)

        # Acceleration: change in velocity (vectorized)
        accel = np.full(n, np.nan)
        valid_mask = ~np.isnan(velocity[self.lag:]) & ~np.isnan(velocity[:-self.lag])
        accel[2 * self.lag:] = np.where(
            valid_mask[self.lag:],
            velocity[2 * self.lag:] - velocity[self.lag:n - self.lag],
            np.nan,
        )

        # Optional smoothing via SMA
        if self.smooth > 1:
            accel = _rolling_mean_nan(accel, self.smooth)

        if self.normalized:
            from signalflow.ta._normalization import get_norm_window, normalize_zscore

            norm_window = self.norm_period or get_norm_window(max(self.lag * 2, 10))
            accel = normalize_zscore(accel, window=norm_window)

        suffix = "_norm" if self.normalized else ""
        col_name = f"{self.source_col}_accel_{self.lag}{suffix}"
        return df.with_columns(pl.Series(name=col_name, values=accel))

    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "lag": 1},
        {"source_col": "close", "lag": 5},
        {"source_col": "close", "lag": 1, "smooth": 5},
        {"source_col": "close", "lag": 1, "normalized": True},
    ]

    @property
    def warmup(self) -> int:
        base = (self.lag * 2 + self.smooth) * 5
        if self.normalized:
            from signalflow.ta._normalization import get_norm_window

            norm_window = self.norm_period or get_norm_window(max(self.lag * 2, 10))
            return base + norm_window
        return base


@dataclass
@feature("momentum/jerk")
class JerkMom(Feature):
    """Price Jerk (third derivative).

    Rate of change of acceleration.

    velocity = ln(Close / Close[lag])
    acceleration = velocity - velocity[lag]
    jerk = acceleration - acceleration[lag]

    Interpretation:
    - Jerk sign change is the earliest signal in the kinematic chain
    - Positive jerk: acceleration is increasing (trend about to strengthen)
    - Negative jerk during uptrend: earliest reversal warning
    - Precedes acceleration sign change, which precedes velocity sign change

    Reference: Kinematic analysis applied to financial time series
    """

    source_col: str = "close"
    lag: int = 1
    smooth: int = 1
    normalized: bool = False
    norm_period: int | None = None

    requires: ClassVar[list[str]] = ["{source_col}"]
    outputs: ClassVar[list[str]] = ["{source_col}_jerk_{lag}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        values = df[self.source_col].to_numpy().astype(np.float64)
        n = len(values)

        # Velocity (vectorized)
        velocity = np.full(n, np.nan)
        safe = (values[self.lag:] > 0) & (values[:-self.lag] > 0)
        velocity[self.lag:] = np.where(safe, np.log(values[self.lag:] / values[:-self.lag]), np.nan)

        # Acceleration (vectorized)
        accel = np.full(n, np.nan)
        vm = ~np.isnan(velocity)
        for i in range(2 * self.lag, n):
            if vm[i] and vm[i - self.lag]:
                accel[i] = velocity[i] - velocity[i - self.lag]

        # Jerk (vectorized)
        jerk = np.full(n, np.nan)
        am = ~np.isnan(accel)
        for i in range(3 * self.lag, n):
            if am[i] and am[i - self.lag]:
                jerk[i] = accel[i] - accel[i - self.lag]

        if self.smooth > 1:
            jerk = _rolling_mean_nan(jerk, self.smooth)

        if self.normalized:
            from signalflow.ta._normalization import get_norm_window, normalize_zscore

            norm_window = self.norm_period or get_norm_window(max(self.lag * 3, 10))
            jerk = normalize_zscore(jerk, window=norm_window)

        suffix = "_norm" if self.normalized else ""
        col_name = f"{self.source_col}_jerk_{self.lag}{suffix}"
        return df.with_columns(pl.Series(name=col_name, values=jerk))

    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "lag": 1},
        {"source_col": "close", "lag": 5},
        {"source_col": "close", "lag": 1, "smooth": 5},
        {"source_col": "close", "lag": 1, "normalized": True},
    ]

    @property
    def warmup(self) -> int:
        base = (self.lag * 3 + self.smooth) * 5
        if self.normalized:
            from signalflow.ta._normalization import get_norm_window

            norm_window = self.norm_period or get_norm_window(max(self.lag * 3, 10))
            return base + norm_window
        return base


@dataclass
@feature("momentum/angular_momentum")
class AngularMomentumMom(Feature):
    """Angular Momentum (L = displacement x velocity).

    Cross-product analogy: displacement from MA times velocity.

    displacement = ln(Close) - ln(SMA)
    velocity = ln(Close / Close[1])
    L = displacement x velocity

    Smoothed over rolling period.

    Interpretation:
    - Large positive L: price above MA and accelerating up (strong uptrend)
    - Large negative L: price below MA and accelerating down (strong downtrend)
    - L near zero: either near equilibrium or velocity near zero
    - L conservation violations signal regime changes
    - Divergence between L and price: weakening trend

    Reference: Rotational mechanics angular momentum
    """

    source_col: str = "close"
    period: int = 20
    ma_period: int = 50
    normalized: bool = False
    norm_period: int | None = None

    requires: ClassVar[list[str]] = ["{source_col}"]
    outputs: ClassVar[list[str]] = ["{source_col}_angmom_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        values = df[self.source_col].to_numpy().astype(np.float64)

        ma = _sma_nb(values, self.ma_period)
        log_values = np.log(np.maximum(values, 1e-10))
        log_ma = np.log(np.maximum(ma, 1e-10))
        displacement = log_values - log_ma

        velocity = _velocity_kernel(values)
        velocity[0] = np.nan

        L_raw = displacement * velocity
        L = _rolling_mean_nan(L_raw, self.period)

        if self.normalized:
            from signalflow.ta._normalization import get_norm_window, normalize_zscore

            norm_window = self.norm_period or get_norm_window(self.ma_period)
            L = normalize_zscore(L, window=norm_window)

        suffix = "_norm" if self.normalized else ""
        col_name = f"{self.source_col}_angmom_{self.period}{suffix}"
        return df.with_columns(pl.Series(name=col_name, values=L))

    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 20, "ma_period": 50},
        {"source_col": "close", "period": 20, "ma_period": 100},
        {"source_col": "close", "period": 50, "ma_period": 200},
        {"source_col": "close", "period": 20, "ma_period": 50, "normalized": True},
    ]

    @property
    def warmup(self) -> int:
        base = max(self.ma_period, self.period) * 5
        if self.normalized:
            from signalflow.ta._normalization import get_norm_window

            norm_window = self.norm_period or get_norm_window(self.ma_period)
            return base + norm_window
        return base


@dataclass
@feature("momentum/torque")
class TorqueMom(Feature):
    """Torque (τ = dL/dt) - rate of change of angular momentum.

    torque = Δ(angular_momentum) / Δt

    Interpretation:
    - Positive torque: angular momentum increasing (trend strengthening)
    - Negative torque: angular momentum decreasing (trend weakening)
    - Torque spike: sudden twist in market dynamics
    - Sign change precedes angular momentum sign change

    Reference: Rotational mechanics torque
    """

    source_col: str = "close"
    period: int = 20
    ma_period: int = 50
    torque_lag: int = 5
    normalized: bool = False
    norm_period: int | None = None

    requires: ClassVar[list[str]] = ["{source_col}"]
    outputs: ClassVar[list[str]] = ["{source_col}_torque_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        values = df[self.source_col].to_numpy().astype(np.float64)
        n = len(values)

        ma = _sma_nb(values, self.ma_period)
        log_values = np.log(np.maximum(values, 1e-10))
        log_ma = np.log(np.maximum(ma, 1e-10))
        displacement = log_values - log_ma

        velocity = _velocity_kernel(values)
        velocity[0] = np.nan

        L_raw = displacement * velocity
        L = _rolling_mean_nan(L_raw, self.period)

        # Torque = dL/dt (vectorized with lag)
        torque = np.full(n, np.nan)
        valid_L = ~np.isnan(L)
        for i in range(self.torque_lag, n):
            if valid_L[i] and valid_L[i - self.torque_lag]:
                torque[i] = (L[i] - L[i - self.torque_lag]) / self.torque_lag

        if self.normalized:
            from signalflow.ta._normalization import get_norm_window, normalize_zscore

            norm_window = self.norm_period or get_norm_window(self.ma_period)
            torque = normalize_zscore(torque, window=norm_window)

        suffix = "_norm" if self.normalized else ""
        col_name = f"{self.source_col}_torque_{self.period}{suffix}"
        return df.with_columns(pl.Series(name=col_name, values=torque))

    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 20, "ma_period": 50},
        {"source_col": "close", "period": 20, "ma_period": 100, "torque_lag": 10},
        {"source_col": "close", "period": 50, "ma_period": 200},
        {"source_col": "close", "period": 20, "ma_period": 50, "normalized": True},
    ]

    @property
    def warmup(self) -> int:
        base = (max(self.ma_period, self.period) + self.torque_lag) * 5
        if self.normalized:
            from signalflow.ta._normalization import get_norm_window

            norm_window = self.norm_period or get_norm_window(self.ma_period)
            return base + norm_window
        return base

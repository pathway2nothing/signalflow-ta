# src/signalflow/ta/momentum/kinematics.py
"""Kinematic indicators - higher-order derivatives of price movement."""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import polars as pl

from signalflow import sf_component
from signalflow.feature.base import Feature


@dataclass
@sf_component(name="momentum/acceleration")
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

    requires = ["{source_col}"]
    outputs = ["{source_col}_accel_{lag}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        values = df[self.source_col].to_numpy()
        n = len(values)

        # Velocity: log-return over lag
        velocity = np.full(n, np.nan)
        for i in range(self.lag, n):
            if values[i - self.lag] > 0 and values[i] > 0:
                velocity[i] = np.log(values[i] / values[i - self.lag])

        # Acceleration: change in velocity
        accel = np.full(n, np.nan)
        for i in range(2 * self.lag, n):
            if not np.isnan(velocity[i]) and not np.isnan(velocity[i - self.lag]):
                accel[i] = velocity[i] - velocity[i - self.lag]

        # Optional smoothing via SMA
        if self.smooth > 1:
            raw = accel.copy()
            accel = np.full(n, np.nan)
            for i in range(2 * self.lag + self.smooth - 1, n):
                window = raw[i - self.smooth + 1 : i + 1]
                valid = window[~np.isnan(window)]
                if len(valid) == self.smooth:
                    accel[i] = np.mean(valid)

        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window

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
@sf_component(name="momentum/jerk")
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

    requires = ["{source_col}"]
    outputs = ["{source_col}_jerk_{lag}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        values = df[self.source_col].to_numpy()
        n = len(values)

        # Velocity
        velocity = np.full(n, np.nan)
        for i in range(self.lag, n):
            if values[i - self.lag] > 0 and values[i] > 0:
                velocity[i] = np.log(values[i] / values[i - self.lag])

        # Acceleration
        accel = np.full(n, np.nan)
        for i in range(2 * self.lag, n):
            if not np.isnan(velocity[i]) and not np.isnan(velocity[i - self.lag]):
                accel[i] = velocity[i] - velocity[i - self.lag]

        # Jerk
        jerk = np.full(n, np.nan)
        for i in range(3 * self.lag, n):
            if not np.isnan(accel[i]) and not np.isnan(accel[i - self.lag]):
                jerk[i] = accel[i] - accel[i - self.lag]

        # Optional smoothing
        if self.smooth > 1:
            raw = jerk.copy()
            jerk = np.full(n, np.nan)
            for i in range(3 * self.lag + self.smooth - 1, n):
                window = raw[i - self.smooth + 1 : i + 1]
                valid = window[~np.isnan(window)]
                if len(valid) == self.smooth:
                    jerk[i] = np.mean(valid)

        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window

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
@sf_component(name="momentum/angular_momentum")
class AngularMomentumMom(Feature):
    """Angular Momentum (L = displacement × velocity).

    Cross-product analogy: displacement from MA times velocity.

    displacement = ln(Close) - ln(SMA)
    velocity = ln(Close / Close[1])
    L = displacement × velocity

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

    requires = ["{source_col}"]
    outputs = ["{source_col}_angmom_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        values = df[self.source_col].to_numpy()
        n = len(values)

        # MA equilibrium
        ma = np.full(n, np.nan)
        for i in range(self.ma_period - 1, n):
            ma[i] = np.mean(values[i - self.ma_period + 1 : i + 1])

        # Displacement
        log_values = np.log(np.maximum(values, 1e-10))
        log_ma = np.log(np.maximum(ma, 1e-10))
        displacement = log_values - log_ma

        # Velocity
        velocity = np.full(n, np.nan)
        for i in range(1, n):
            if values[i - 1] > 0 and values[i] > 0:
                velocity[i] = np.log(values[i] / values[i - 1])

        # Angular momentum = displacement × velocity
        L_raw = displacement * velocity

        # Smooth
        L = np.full(n, np.nan)
        start = max(self.ma_period, self.period)
        for i in range(start, n):
            window = L_raw[i - self.period + 1 : i + 1]
            valid = window[~np.isnan(window)]
            if len(valid) > 0:
                L[i] = np.mean(valid)

        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window

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
@sf_component(name="momentum/torque")
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

    requires = ["{source_col}"]
    outputs = ["{source_col}_torque_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        values = df[self.source_col].to_numpy()
        n = len(values)

        # MA
        ma = np.full(n, np.nan)
        for i in range(self.ma_period - 1, n):
            ma[i] = np.mean(values[i - self.ma_period + 1 : i + 1])

        # Displacement & velocity
        log_values = np.log(np.maximum(values, 1e-10))
        log_ma = np.log(np.maximum(ma, 1e-10))
        displacement = log_values - log_ma

        velocity = np.full(n, np.nan)
        for i in range(1, n):
            if values[i - 1] > 0 and values[i] > 0:
                velocity[i] = np.log(values[i] / values[i - 1])

        # Angular momentum smoothed
        L_raw = displacement * velocity
        L = np.full(n, np.nan)
        start = max(self.ma_period, self.period)
        for i in range(start, n):
            window = L_raw[i - self.period + 1 : i + 1]
            valid = window[~np.isnan(window)]
            if len(valid) > 0:
                L[i] = np.mean(valid)

        # Torque = dL/dt
        torque = np.full(n, np.nan)
        for i in range(start + self.torque_lag, n):
            if not np.isnan(L[i]) and not np.isnan(L[i - self.torque_lag]):
                torque[i] = (L[i] - L[i - self.torque_lag]) / self.torque_lag

        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window

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

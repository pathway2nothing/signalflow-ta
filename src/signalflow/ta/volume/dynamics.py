# src/signalflow/ta/volume/dynamics.py
"""Volume-weighted dynamics - force, impulse, momentum, power from Newtonian mechanics."""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import polars as pl

from signalflow import sf_component
from signalflow.feature.base import Feature


@dataclass
@sf_component(name="volume/market_force")
class MarketForceVolume(Feature):
    """Market Force (F = m × a).

    Volume-weighted price acceleration.

    velocity = ln(Close / Close[1])
    acceleration = velocity - velocity[1]
    force = volume × acceleration

    Interpretation:
    - Large positive force: strong buying with increasing momentum
    - Large negative force: strong selling with increasing momentum
    - Force divergence from price: potential reversal
    - Smoothed version reduces noise while preserving signal

    Reference: Newtonian mechanics applied to market dynamics
    """

    period: int = 14
    normalized: bool = False
    norm_period: int | None = None

    requires = ["close", "volume"]
    outputs = ["mforce_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        close = df["close"].to_numpy()
        volume = df["volume"].to_numpy().astype(np.float64)
        n = len(close)

        # Velocity (log-returns)
        velocity = np.full(n, np.nan)
        for i in range(1, n):
            if close[i - 1] > 0 and close[i] > 0:
                velocity[i] = np.log(close[i] / close[i - 1])

        # Acceleration
        accel = np.full(n, np.nan)
        for i in range(2, n):
            if not np.isnan(velocity[i]) and not np.isnan(velocity[i - 1]):
                accel[i] = velocity[i] - velocity[i - 1]

        # Force = volume * acceleration
        force_raw = volume * accel

        # Smooth with SMA
        force = np.full(n, np.nan)
        for i in range(self.period + 1, n):
            window = force_raw[i - self.period + 1 : i + 1]
            valid = window[~np.isnan(window)]
            if len(valid) > 0:
                force[i] = np.mean(valid)

        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window

            norm_window = self.norm_period or get_norm_window(self.period)
            force = normalize_zscore(force, window=norm_window)

        suffix = "_norm" if self.normalized else ""
        col_name = f"mforce_{self.period}{suffix}"
        return df.with_columns(pl.Series(name=col_name, values=force))

    test_params: ClassVar[list[dict]] = [
        {"period": 14},
        {"period": 30},
        {"period": 60},
        {"period": 14, "normalized": True},
    ]

    @property
    def warmup(self) -> int:
        base = (self.period + 2) * 5
        if self.normalized:
            from signalflow.ta._normalization import get_norm_window

            norm_window = self.norm_period or get_norm_window(self.period)
            return base + norm_window
        return base


@dataclass
@sf_component(name="volume/impulse")
class ImpulseVolume(Feature):
    """Market Impulse (J = Σ F × Δt).

    Cumulative force over a rolling window.

    force = volume × acceleration
    impulse = Σ(force, period)

    Interpretation:
    - Sustained positive impulse: strong buying campaign
    - Sustained negative impulse: strong selling campaign
    - Impulse divergence from price: accumulation/distribution signal
    - Larger impulse = more conviction behind the move

    Reference: Newtonian impulse-momentum theorem
    """

    period: int = 14
    normalized: bool = False
    norm_period: int | None = None

    requires = ["close", "volume"]
    outputs = ["impulse_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        close = df["close"].to_numpy()
        volume = df["volume"].to_numpy().astype(np.float64)
        n = len(close)

        # Velocity
        velocity = np.full(n, np.nan)
        for i in range(1, n):
            if close[i - 1] > 0 and close[i] > 0:
                velocity[i] = np.log(close[i] / close[i - 1])

        # Acceleration
        accel = np.full(n, np.nan)
        for i in range(2, n):
            if not np.isnan(velocity[i]) and not np.isnan(velocity[i - 1]):
                accel[i] = velocity[i] - velocity[i - 1]

        # Force
        force = volume * accel

        # Impulse = rolling sum of force
        impulse = np.full(n, np.nan)
        for i in range(self.period + 1, n):
            window = force[i - self.period + 1 : i + 1]
            valid = window[~np.isnan(window)]
            if len(valid) > 0:
                impulse[i] = np.sum(valid)

        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window

            norm_window = self.norm_period or get_norm_window(self.period)
            impulse = normalize_zscore(impulse, window=norm_window)

        suffix = "_norm" if self.normalized else ""
        col_name = f"impulse_{self.period}{suffix}"
        return df.with_columns(pl.Series(name=col_name, values=impulse))

    test_params: ClassVar[list[dict]] = [
        {"period": 14},
        {"period": 30},
        {"period": 60},
        {"period": 14, "normalized": True},
    ]

    @property
    def warmup(self) -> int:
        base = (self.period + 2) * 5
        if self.normalized:
            from signalflow.ta._normalization import get_norm_window

            norm_window = self.norm_period or get_norm_window(self.period)
            return base + norm_window
        return base


@dataclass
@sf_component(name="volume/market_momentum")
class MarketMomentumVolume(Feature):
    """Market Momentum (p = m × v).

    Volume-weighted velocity. Unlike simple price momentum,
    this weights by "mass" (volume participation).

    velocity = ln(Close / Close[1])
    momentum = volume × velocity

    Smoothed over rolling period.

    Interpretation:
    - High positive: large volume driving price up
    - High negative: large volume driving price down
    - Divergence from OBV: acceleration vs cumulative bias
    - More responsive than OBV to recent activity

    Reference: Newtonian momentum (mass × velocity)
    """

    period: int = 14
    normalized: bool = False
    norm_period: int | None = None

    requires = ["close", "volume"]
    outputs = ["mmom_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        close = df["close"].to_numpy()
        volume = df["volume"].to_numpy().astype(np.float64)
        n = len(close)

        # Velocity
        velocity = np.full(n, np.nan)
        for i in range(1, n):
            if close[i - 1] > 0 and close[i] > 0:
                velocity[i] = np.log(close[i] / close[i - 1])

        # Momentum = volume * velocity
        mom_raw = volume * velocity

        # Smooth with SMA
        mmom = np.full(n, np.nan)
        for i in range(self.period, n):
            window = mom_raw[i - self.period + 1 : i + 1]
            valid = window[~np.isnan(window)]
            if len(valid) > 0:
                mmom[i] = np.mean(valid)

        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window

            norm_window = self.norm_period or get_norm_window(self.period)
            mmom = normalize_zscore(mmom, window=norm_window)

        suffix = "_norm" if self.normalized else ""
        col_name = f"mmom_{self.period}{suffix}"
        return df.with_columns(pl.Series(name=col_name, values=mmom))

    test_params: ClassVar[list[dict]] = [
        {"period": 14},
        {"period": 30},
        {"period": 60},
        {"period": 14, "normalized": True},
    ]

    @property
    def warmup(self) -> int:
        base = (self.period + 1) * 5
        if self.normalized:
            from signalflow.ta._normalization import get_norm_window

            norm_window = self.norm_period or get_norm_window(self.period)
            return base + norm_window
        return base


@dataclass
@sf_component(name="volume/market_power")
class MarketPowerVolume(Feature):
    """Market Power (P = F × v).

    Rate at which work is being done by market participants.

    velocity = ln(Close / Close[1])
    acceleration = velocity - velocity[1]
    force = volume × acceleration
    power = force × velocity

    Smoothed over rolling period.

    Interpretation:
    - High positive power: market accelerating in direction of movement
    - Negative power during uptrend: deceleration warning
    - Power spikes precede significant moves
    - Combines directionality (velocity sign) with conviction (force magnitude)

    Reference: Newtonian power (force × velocity)
    """

    period: int = 14
    normalized: bool = False
    norm_period: int | None = None

    requires = ["close", "volume"]
    outputs = ["mpower_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        close = df["close"].to_numpy()
        volume = df["volume"].to_numpy().astype(np.float64)
        n = len(close)

        # Velocity
        velocity = np.full(n, np.nan)
        for i in range(1, n):
            if close[i - 1] > 0 and close[i] > 0:
                velocity[i] = np.log(close[i] / close[i - 1])

        # Acceleration
        accel = np.full(n, np.nan)
        for i in range(2, n):
            if not np.isnan(velocity[i]) and not np.isnan(velocity[i - 1]):
                accel[i] = velocity[i] - velocity[i - 1]

        # Power = force * velocity = (volume * accel) * velocity
        power_raw = volume * accel * velocity

        # Smooth with SMA
        mpower = np.full(n, np.nan)
        for i in range(self.period + 1, n):
            window = power_raw[i - self.period + 1 : i + 1]
            valid = window[~np.isnan(window)]
            if len(valid) > 0:
                mpower[i] = np.mean(valid)

        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window

            norm_window = self.norm_period or get_norm_window(self.period)
            mpower = normalize_zscore(mpower, window=norm_window)

        suffix = "_norm" if self.normalized else ""
        col_name = f"mpower_{self.period}{suffix}"
        return df.with_columns(pl.Series(name=col_name, values=mpower))

    test_params: ClassVar[list[dict]] = [
        {"period": 14},
        {"period": 30},
        {"period": 60},
        {"period": 14, "normalized": True},
    ]

    @property
    def warmup(self) -> int:
        base = (self.period + 2) * 5
        if self.normalized:
            from signalflow.ta._normalization import get_norm_window

            norm_window = self.norm_period or get_norm_window(self.period)
            return base + norm_window
        return base


@dataclass
@sf_component(name="volume/market_capacitance")
class MarketCapacitanceVolume(Feature):
    """Market Capacitance - volume absorbed per unit price change.

    C = Σ(volume) / |Δprice|  over rolling period

    How much volume is needed to move price by one unit.

    Interpretation:
    - High capacitance: strong S/R, price absorbs volume without moving
    - Low capacitance: thin market, small volume causes large moves
    - Rising capacitance: building congestion zone
    - Falling capacitance: breakout imminent

    Reference: Electrical capacitance (charge stored per unit voltage)
    """

    period: int = 20
    normalized: bool = False
    norm_period: int | None = None

    requires = ["close", "volume"]
    outputs = ["mcap_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        close = df["close"].to_numpy()
        volume = df["volume"].to_numpy().astype(np.float64)
        n = len(close)

        mcap = np.full(n, np.nan)
        for i in range(self.period - 1, n):
            vol_sum = np.sum(volume[i - self.period + 1 : i + 1])
            price_change = np.abs(close[i] - close[i - self.period + 1])
            if price_change > 1e-10:
                mcap[i] = vol_sum / price_change

        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window

            norm_window = self.norm_period or get_norm_window(self.period)
            mcap = normalize_zscore(mcap, window=norm_window)

        suffix = "_norm" if self.normalized else ""
        col_name = f"mcap_{self.period}{suffix}"
        return df.with_columns(pl.Series(name=col_name, values=mcap))

    test_params: ClassVar[list[dict]] = [
        {"period": 20},
        {"period": 50},
        {"period": 100},
        {"period": 20, "normalized": True},
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
@sf_component(name="volume/gravitational_pull")
class GravitationalPullVolume(Feature):
    """Gravitational Pull - volume-weighted attraction to recent price levels.

    G = Σ(volume_i / distance_i²) for recent bars

    Bars with high volume near current price exert stronger pull.

    Interpretation:
    - High pull: strong volume-weighted attraction nearby (S/R zone)
    - Low pull: weak attraction, price free to move
    - Pull increasing: building support/resistance
    - Pull decreasing: S/R weakening

    Reference: Newton's law of gravitation (F ∝ m/r²)
    """

    period: int = 20
    normalized: bool = False
    norm_period: int | None = None

    requires = ["close", "volume"]
    outputs = ["gpull_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        close = df["close"].to_numpy()
        volume = df["volume"].to_numpy().astype(np.float64)
        n = len(close)

        gpull = np.full(n, np.nan)
        for i in range(self.period - 1, n):
            current_price = close[i]
            total_pull = 0.0
            for j in range(i - self.period + 1, i):
                if close[j] > 0 and current_price > 0:
                    distance = np.abs(np.log(current_price / close[j]))
                    if distance > 1e-10:
                        total_pull += volume[j] / (distance**2)
            gpull[i] = total_pull

        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window

            norm_window = self.norm_period or get_norm_window(self.period)
            gpull = normalize_zscore(gpull, window=norm_window)

        suffix = "_norm" if self.normalized else ""
        col_name = f"gpull_{self.period}{suffix}"
        return df.with_columns(pl.Series(name=col_name, values=gpull))

    test_params: ClassVar[list[dict]] = [
        {"period": 20},
        {"period": 50},
        {"period": 100},
        {"period": 20, "normalized": True},
    ]

    @property
    def warmup(self) -> int:
        base = self.period * 5
        if self.normalized:
            from signalflow.ta._normalization import get_norm_window

            norm_window = self.norm_period or get_norm_window(self.period)
            return base + norm_window
        return base

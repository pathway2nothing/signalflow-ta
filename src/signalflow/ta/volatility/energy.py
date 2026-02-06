# src/signalflow/ta/volatility/energy.py
"""Energy-based volatility indicators - mechanical energy model for regime detection."""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import polars as pl

from signalflow.core import sf_component
from signalflow.feature.base import Feature


@dataclass
@sf_component(name="volatility/kinetic_energy")
class KineticEnergyVol(Feature):
    """Kinetic Energy of price movement.

    KE = ½ × v²  where v = ln(Close / Close[1])

    Smoothed over rolling period.

    Interpretation:
    - High KE: rapid price movement (high instantaneous volatility)
    - Low KE: slow/stagnant price (consolidation)
    - KE spikes precede or accompany breakouts
    - Direction-agnostic: captures intensity regardless of trend

    Reference: Classical mechanics ½mv² with m=1
    """

    period: int = 20
    normalized: bool = False
    norm_period: int | None = None

    requires = ["close"]
    outputs = ["ke_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        close = df["close"].to_numpy()
        n = len(close)

        # Velocity (log-returns)
        velocity = np.full(n, np.nan)
        for i in range(1, n):
            if close[i - 1] > 0 and close[i] > 0:
                velocity[i] = np.log(close[i] / close[i - 1])

        # Kinetic energy = 0.5 * v^2
        ke_raw = 0.5 * velocity**2

        # Smooth with rolling mean
        ke = np.full(n, np.nan)
        for i in range(self.period, n):
            window = ke_raw[i - self.period + 1 : i + 1]
            valid = window[~np.isnan(window)]
            if len(valid) > 0:
                ke[i] = np.mean(valid)

        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window

            norm_window = self.norm_period or get_norm_window(self.period)
            ke = normalize_zscore(ke, window=norm_window)

        suffix = "_norm" if self.normalized else ""
        col_name = f"ke_{self.period}{suffix}"
        return df.with_columns(pl.Series(name=col_name, values=ke))

    test_params: ClassVar[list[dict]] = [
        {"period": 20},
        {"period": 50},
        {"period": 100},
        {"period": 20, "normalized": True},
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
@sf_component(name="volatility/potential_energy")
class PotentialEnergyVol(Feature):
    """Potential Energy - displacement from equilibrium (moving average).

    PE = (ln(Close) - ln(MA))²

    Models price as a spring-mass system where the moving average
    is the equilibrium position.

    Interpretation:
    - High PE: price far from equilibrium, strong reversion potential
    - Low PE: price near equilibrium, low stored energy
    - Rising PE: price stretching away from MA
    - PE release (drop) often accompanies snapback moves

    Reference: Harmonic oscillator potential energy ½kx²
    """

    period: int = 20
    ma_period: int = 50
    normalized: bool = False
    norm_period: int | None = None

    requires = ["close"]
    outputs = ["pe_{period}_{ma_period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        close = df["close"].to_numpy()
        n = len(close)

        # Moving average as equilibrium
        ma = np.full(n, np.nan)
        for i in range(self.ma_period - 1, n):
            ma[i] = np.mean(close[i - self.ma_period + 1 : i + 1])

        # Log displacement from equilibrium
        log_close = np.log(np.maximum(close, 1e-10))
        log_ma = np.log(np.maximum(ma, 1e-10))
        displacement = log_close - log_ma

        # Potential energy = displacement^2
        pe_raw = displacement**2

        # Smooth with rolling mean
        pe = np.full(n, np.nan)
        for i in range(max(self.ma_period, self.period) - 1, n):
            start = max(0, i - self.period + 1)
            window = pe_raw[start : i + 1]
            valid = window[~np.isnan(window)]
            if len(valid) > 0:
                pe[i] = np.mean(valid)

        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window

            norm_window = self.norm_period or get_norm_window(self.ma_period)
            pe = normalize_zscore(pe, window=norm_window)

        suffix = "_norm" if self.normalized else ""
        col_name = f"pe_{self.period}_{self.ma_period}{suffix}"
        return df.with_columns(pl.Series(name=col_name, values=pe))

    test_params: ClassVar[list[dict]] = [
        {"period": 20, "ma_period": 50},
        {"period": 20, "ma_period": 100},
        {"period": 50, "ma_period": 200},
        {"period": 20, "ma_period": 50, "normalized": True},
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
@sf_component(name="volatility/total_energy")
class TotalEnergyVol(Feature):
    """Total Mechanical Energy (E = KE + PE).

    Combines kinetic (movement intensity) and potential (displacement) energy.

    KE = ½ × v²
    PE = (ln(Close) - ln(MA))²
    E = KE + PE

    Interpretation:
    - In stable markets, total energy is approximately conserved
    - Sudden increases in E signal regime change (new volatility regime)
    - Sudden decreases in E signal energy dissipation (consolidation)
    - Energy conservation violations are stronger signals than either component

    Reference: Conservation of mechanical energy principle
    """

    period: int = 20
    ma_period: int = 50
    normalized: bool = False
    norm_period: int | None = None

    requires = ["close"]
    outputs = ["te_{period}_{ma_period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        close = df["close"].to_numpy()
        n = len(close)

        # Velocity
        velocity = np.full(n, np.nan)
        for i in range(1, n):
            if close[i - 1] > 0 and close[i] > 0:
                velocity[i] = np.log(close[i] / close[i - 1])

        # KE = 0.5 * v^2
        ke = 0.5 * velocity**2

        # MA equilibrium
        ma = np.full(n, np.nan)
        for i in range(self.ma_period - 1, n):
            ma[i] = np.mean(close[i - self.ma_period + 1 : i + 1])

        # PE = displacement^2
        log_close = np.log(np.maximum(close, 1e-10))
        log_ma = np.log(np.maximum(ma, 1e-10))
        pe = (log_close - log_ma) ** 2

        # Total energy
        te_raw = ke + pe

        # Smooth with rolling mean
        te = np.full(n, np.nan)
        for i in range(max(self.ma_period, self.period) - 1, n):
            start = max(0, i - self.period + 1)
            window = te_raw[start : i + 1]
            valid = window[~np.isnan(window)]
            if len(valid) > 0:
                te[i] = np.mean(valid)

        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window

            norm_window = self.norm_period or get_norm_window(self.ma_period)
            te = normalize_zscore(te, window=norm_window)

        suffix = "_norm" if self.normalized else ""
        col_name = f"te_{self.period}_{self.ma_period}{suffix}"
        return df.with_columns(pl.Series(name=col_name, values=te))

    test_params: ClassVar[list[dict]] = [
        {"period": 20, "ma_period": 50},
        {"period": 20, "ma_period": 100},
        {"period": 50, "ma_period": 200},
        {"period": 20, "ma_period": 50, "normalized": True},
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
@sf_component(name="volatility/energy_flow")
class EnergyFlowVol(Feature):
    """Energy Flow Rate (Power of the system).

    Rate of change of total energy: dE/dt.

    Interpretation:
    - Positive flow: system gaining energy (volatility expanding)
    - Negative flow: system losing energy (volatility contracting)
    - Large absolute flow: rapid regime transition
    - Zero crossing: energy equilibrium point (inflection)

    Reference: Thermodynamic power / energy flow rate
    """

    period: int = 20
    ma_period: int = 50
    flow_lag: int = 5
    normalized: bool = False
    norm_period: int | None = None

    requires = ["close"]
    outputs = ["eflow_{period}_{ma_period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        close = df["close"].to_numpy()
        n = len(close)

        # Velocity
        velocity = np.full(n, np.nan)
        for i in range(1, n):
            if close[i - 1] > 0 and close[i] > 0:
                velocity[i] = np.log(close[i] / close[i - 1])

        # KE
        ke = 0.5 * velocity**2

        # MA equilibrium
        ma = np.full(n, np.nan)
        for i in range(self.ma_period - 1, n):
            ma[i] = np.mean(close[i - self.ma_period + 1 : i + 1])

        # PE
        log_close = np.log(np.maximum(close, 1e-10))
        log_ma = np.log(np.maximum(ma, 1e-10))
        pe = (log_close - log_ma) ** 2

        # Total energy smoothed
        te_raw = ke + pe
        te = np.full(n, np.nan)
        for i in range(max(self.ma_period, self.period) - 1, n):
            start = max(0, i - self.period + 1)
            window = te_raw[start : i + 1]
            valid = window[~np.isnan(window)]
            if len(valid) > 0:
                te[i] = np.mean(valid)

        # Energy flow = dE/dt
        eflow = np.full(n, np.nan)
        for i in range(max(self.ma_period, self.period) - 1 + self.flow_lag, n):
            if not np.isnan(te[i]) and not np.isnan(te[i - self.flow_lag]):
                eflow[i] = te[i] - te[i - self.flow_lag]

        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window

            norm_window = self.norm_period or get_norm_window(self.ma_period)
            eflow = normalize_zscore(eflow, window=norm_window)

        suffix = "_norm" if self.normalized else ""
        col_name = f"eflow_{self.period}_{self.ma_period}{suffix}"
        return df.with_columns(pl.Series(name=col_name, values=eflow))

    test_params: ClassVar[list[dict]] = [
        {"period": 20, "ma_period": 50},
        {"period": 20, "ma_period": 100, "flow_lag": 10},
        {"period": 50, "ma_period": 200},
        {"period": 20, "ma_period": 50, "normalized": True},
    ]

    @property
    def warmup(self) -> int:
        base = (max(self.ma_period, self.period) + self.flow_lag) * 5
        if self.normalized:
            from signalflow.ta._normalization import get_norm_window

            norm_window = self.norm_period or get_norm_window(self.ma_period)
            return base + norm_window
        return base


@dataclass
@sf_component(name="volatility/elastic_strain")
class ElasticStrainVol(Feature):
    """Elastic Strain - relative displacement from equilibrium.

    strain = (Close - MA) / MA

    Reversible deformation analogy: how "stretched" price is from its MA.

    Interpretation:
    - High positive strain: price stretched above MA, snap-back potential
    - High negative strain: price stretched below MA, bounce potential
    - Strain magnitude: tension level
    - Strain rate (change): speed of deformation

    Reference: Hooke's law, elastic deformation
    """

    period: int = 20
    ma_period: int = 50
    normalized: bool = False
    norm_period: int | None = None

    requires = ["close"]
    outputs = ["strain_{period}_{ma_period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        close = df["close"].to_numpy()
        n = len(close)

        ma = np.full(n, np.nan)
        for i in range(self.ma_period - 1, n):
            ma[i] = np.mean(close[i - self.ma_period + 1 : i + 1])

        strain_raw = (close - ma) / np.maximum(ma, 1e-10)

        # Smooth
        strain = np.full(n, np.nan)
        for i in range(max(self.ma_period, self.period) - 1, n):
            window = strain_raw[i - self.period + 1 : i + 1]
            valid = window[~np.isnan(window)]
            if len(valid) > 0:
                strain[i] = np.mean(valid)

        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window

            norm_window = self.norm_period or get_norm_window(self.ma_period)
            strain = normalize_zscore(strain, window=norm_window)

        suffix = "_norm" if self.normalized else ""
        col_name = f"strain_{self.period}_{self.ma_period}{suffix}"
        return df.with_columns(pl.Series(name=col_name, values=strain))

    test_params: ClassVar[list[dict]] = [
        {"period": 20, "ma_period": 50},
        {"period": 20, "ma_period": 100},
        {"period": 50, "ma_period": 200},
        {"period": 20, "ma_period": 50, "normalized": True},
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
@sf_component(name="volatility/temperature")
class TemperatureVol(Feature):
    """Market Temperature - kinetic energy per degree of freedom.

    T = Var(log-returns) × period

    Statistical mechanics analogy: temperature is proportional
    to average kinetic energy.

    Interpretation:
    - High temperature: "hot" market, high random motion
    - Low temperature: "cold" market, low activity
    - Rising temperature: heating up (vol expansion)
    - Falling temperature: cooling down (vol contraction)

    Reference: Boltzmann equipartition theorem
    """

    period: int = 20
    normalized: bool = False
    norm_period: int | None = None

    requires = ["close"]
    outputs = ["mtemp_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        close = df["close"].to_numpy()
        n = len(close)

        log_ret = np.full(n, np.nan)
        for i in range(1, n):
            if close[i - 1] > 0 and close[i] > 0:
                log_ret[i] = np.log(close[i] / close[i - 1])

        temp = np.full(n, np.nan)
        for i in range(self.period, n):
            window = log_ret[i - self.period + 1 : i + 1]
            valid = window[~np.isnan(window)]
            if len(valid) > 2:
                temp[i] = np.var(valid, ddof=1) * self.period

        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window

            norm_window = self.norm_period or get_norm_window(self.period)
            temp = normalize_zscore(temp, window=norm_window)

        suffix = "_norm" if self.normalized else ""
        col_name = f"mtemp_{self.period}{suffix}"
        return df.with_columns(pl.Series(name=col_name, values=temp))

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
@sf_component(name="volatility/heat_capacity")
class HeatCapacityVol(Feature):
    """Heat Capacity - resistance of market to temperature change.

    C_v = Δ(temperature) / Δ(energy_input)

    Approximated as the ratio of temperature change to total energy change.

    Interpretation:
    - High heat capacity: market absorbs energy without changing volatility
    - Low heat capacity: small energy input causes large vol change
    - Useful for predicting volatility regime sensitivity

    Reference: Thermodynamic heat capacity
    """

    period: int = 20
    lag: int = 5
    normalized: bool = False
    norm_period: int | None = None

    requires = ["close"]
    outputs = ["heatcap_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        close = df["close"].to_numpy()
        n = len(close)

        # Log returns
        log_ret = np.full(n, np.nan)
        for i in range(1, n):
            if close[i - 1] > 0 and close[i] > 0:
                log_ret[i] = np.log(close[i] / close[i - 1])

        # Temperature (rolling variance × period)
        temp = np.full(n, np.nan)
        for i in range(self.period, n):
            window = log_ret[i - self.period + 1 : i + 1]
            valid = window[~np.isnan(window)]
            if len(valid) > 2:
                temp[i] = np.var(valid, ddof=1) * self.period

        # Total energy (KE smoothed)
        ke_raw = 0.5 * log_ret**2
        energy = np.full(n, np.nan)
        for i in range(self.period, n):
            window = ke_raw[i - self.period + 1 : i + 1]
            valid = window[~np.isnan(window)]
            if len(valid) > 0:
                energy[i] = np.mean(valid)

        # Heat capacity = ΔT / ΔE
        hcap = np.full(n, np.nan)
        for i in range(self.period + self.lag, n):
            if (
                not np.isnan(temp[i])
                and not np.isnan(temp[i - self.lag])
                and not np.isnan(energy[i])
                and not np.isnan(energy[i - self.lag])
            ):
                d_energy = energy[i] - energy[i - self.lag]
                d_temp = temp[i] - temp[i - self.lag]
                if np.abs(d_energy) > 1e-20:
                    hcap[i] = d_temp / d_energy

        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window

            norm_window = self.norm_period or get_norm_window(self.period)
            hcap = normalize_zscore(hcap, window=norm_window)

        suffix = "_norm" if self.normalized else ""
        col_name = f"heatcap_{self.period}{suffix}"
        return df.with_columns(pl.Series(name=col_name, values=hcap))

    test_params: ClassVar[list[dict]] = [
        {"period": 20},
        {"period": 50, "lag": 10},
        {"period": 100},
        {"period": 20, "normalized": True},
    ]

    @property
    def warmup(self) -> int:
        base = (self.period + self.lag) * 5
        if self.normalized:
            from signalflow.ta._normalization import get_norm_window

            norm_window = self.norm_period or get_norm_window(self.period)
            return base + norm_window
        return base


@dataclass
@sf_component(name="volatility/free_energy")
class FreeEnergyVol(Feature):
    """Helmholtz Free Energy (F = E - T×S).

    Total energy minus "thermal" energy (temperature × entropy).
    The "useful" energy available for directed movement.

    Interpretation:
    - High free energy: market has directed potential (breakout energy)
    - Low free energy: market energy is all "thermal" (random noise)
    - Negative ΔF: spontaneous movement likely
    - Positive ΔF: energy being stored

    Reference: Helmholtz free energy in thermodynamics
    """

    period: int = 20
    entropy_bins: int = 10
    normalized: bool = False
    norm_period: int | None = None

    requires = ["close"]
    outputs = ["fenergy_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        close = df["close"].to_numpy()
        n = len(close)

        log_ret = np.full(n, np.nan)
        for i in range(1, n):
            if close[i - 1] > 0 and close[i] > 0:
                log_ret[i] = np.log(close[i] / close[i - 1])

        fenergy = np.full(n, np.nan)
        for i in range(self.period, n):
            window = log_ret[i - self.period + 1 : i + 1]
            valid = window[~np.isnan(window)]
            if len(valid) < 5:
                continue

            # Energy = mean KE
            energy = np.mean(0.5 * valid**2)

            # Temperature = variance × period
            temperature = np.var(valid, ddof=1) * self.period

            # Entropy via histogram
            hist, _ = np.histogram(valid, bins=self.entropy_bins)
            p = hist / hist.sum()
            p = p[p > 0]
            entropy = -np.sum(p * np.log(p))

            # Free energy = E - T*S
            fenergy[i] = energy - temperature * entropy

        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window

            norm_window = self.norm_period or get_norm_window(self.period)
            fenergy = normalize_zscore(fenergy, window=norm_window)

        suffix = "_norm" if self.normalized else ""
        col_name = f"fenergy_{self.period}{suffix}"
        return df.with_columns(pl.Series(name=col_name, values=fenergy))

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

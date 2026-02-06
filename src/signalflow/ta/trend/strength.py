"""Trend strength indicators - measure how strong a trend is."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl

from signalflow.core import sf_component
from signalflow.feature.base import Feature
from typing import ClassVar


@dataclass
@sf_component(name="trend/adx")
class AdxTrend(Feature):
    """Average Directional Index (ADX).

    Measures trend strength regardless of direction.

    Outputs:
    - adx: trend strength
    - dmp: +DI (positive directional indicator)
    - dmn: -DI (negative directional indicator)

    Bounded [0, 100] (absolute) or [0, 1] (normalized).

    Interpretation (absolute mode):
    - ADX < 20: weak/no trend (ranging)
    - ADX 20-40: developing trend
    - ADX > 40: strong trend
    - +DI > -DI: bullish
    - -DI > +DI: bearish

    Interpretation (normalized mode):
    - ADX < 0.2: weak/no trend (ranging)
    - ADX 0.2-0.4: developing trend
    - ADX > 0.4: strong trend

    Reference: Welles Wilder, "New Concepts in Technical Trading Systems"
    https://www.investopedia.com/terms/a/adx.asp
    """

    period: int = 14
    normalized: bool = False

    requires = ["high", "low", "close"]
    outputs = ["adx_{period}", "dmp_{period}", "dmn_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        close = df["close"].to_numpy()
        n = len(close)

        tr = np.maximum(
            high - low,
            np.maximum(
                np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))
            ),
        )
        tr[0] = high[0] - low[0]

        up = high - np.roll(high, 1)
        dn = np.roll(low, 1) - low
        up[0] = dn[0] = 0

        pdm = np.where((up > dn) & (up > 0), up, 0)
        ndm = np.where((dn > up) & (dn > 0), dn, 0)
        alpha = 1.0 / self.period

        atr = np.full(n, np.nan)
        smooth_pdm = np.full(n, np.nan)
        smooth_ndm = np.full(n, np.nan)

        atr[self.period - 1] = np.mean(tr[: self.period])
        smooth_pdm[self.period - 1] = np.mean(pdm[: self.period])
        smooth_ndm[self.period - 1] = np.mean(ndm[: self.period])

        for i in range(self.period, n):
            atr[i] = alpha * tr[i] + (1 - alpha) * atr[i - 1]
            smooth_pdm[i] = alpha * pdm[i] + (1 - alpha) * smooth_pdm[i - 1]
            smooth_ndm[i] = alpha * ndm[i] + (1 - alpha) * smooth_ndm[i - 1]

        dmp = 100 * smooth_pdm / atr
        dmn = 100 * smooth_ndm / atr

        dx = 100 * np.abs(dmp - dmn) / (dmp + dmn + 1e-10)

        adx = np.full(n, np.nan)
        start = 2 * self.period - 1
        if start < n:
            adx[start] = np.nanmean(dx[self.period : start + 1])
            for i in range(start + 1, n):
                adx[i] = alpha * dx[i] + (1 - alpha) * adx[i - 1]

        # Normalization: [0, 100] → [0, 1]
        if self.normalized:
            adx = adx / 100
            dmp = dmp / 100
            dmn = dmn / 100

        col_adx, col_dmp, col_dmn = self._get_output_names()
        return df.with_columns(
            [
                pl.Series(name=col_adx, values=adx),
                pl.Series(name=col_dmp, values=dmp),
                pl.Series(name=col_dmn, values=dmn),
            ]
        )

    def _get_output_names(self) -> tuple[str, str, str]:
        """Generate output column names with normalization suffix."""
        suffix = "_norm" if self.normalized else ""
        return (
            f"adx_{self.period}{suffix}",
            f"dmp_{self.period}{suffix}",
            f"dmn_{self.period}{suffix}",
        )

    test_params: ClassVar[list[dict]] = [
        {"period": 14},
        {"period": 14, "normalized": True},
        {"period": 30},
        {"period": 60},
    ]

    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable, reproducible output."""
        return self.period * 5


@dataclass
@sf_component(name="trend/aroon")
class AroonTrend(Feature):
    """Aroon Indicator.

    Measures time since highest high / lowest low.

    Outputs:
    - aroon_up: periods since highest high
    - aroon_dn: periods since lowest low
    - aroon_osc: aroon_up - aroon_dn

    Bounded [0, 100] (absolute) or [0, 1] (normalized) for up/dn.
    Bounded [-100, 100] (absolute) or [-1, 1] (normalized) for osc.

    Interpretation (absolute mode):
    - aroon_up > 70: strong uptrend
    - aroon_dn > 70: strong downtrend
    - aroon_osc > 0: bullish
    - aroon_osc < 0: bearish

    Interpretation (normalized mode):
    - aroon_up > 0.7: strong uptrend
    - aroon_dn > 0.7: strong downtrend
    - aroon_osc > 0: bullish
    - aroon_osc < 0: bearish

    Reference: Tushar Chande
    https://www.investopedia.com/terms/a/aroon.asp
    """

    period: int = 25
    normalized: bool = False

    requires = ["high", "low"]
    outputs = ["aroon_up_{period}", "aroon_dn_{period}", "aroon_osc_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        n = len(high)

        aroon_up = np.full(n, np.nan)
        aroon_dn = np.full(n, np.nan)

        for i in range(self.period, n):
            window_high = high[i - self.period : i + 1]
            window_low = low[i - self.period : i + 1]

            periods_from_hh = self.period - np.argmax(window_high[::-1])
            periods_from_ll = self.period - np.argmin(window_low[::-1])

            aroon_up[i] = 100 * (self.period - periods_from_hh) / self.period
            aroon_dn[i] = 100 * (self.period - periods_from_ll) / self.period

        aroon_osc = aroon_up - aroon_dn

        # Normalization: [0, 100] → [0, 1] for up/dn, [-100, 100] → [-1, 1] for osc
        if self.normalized:
            aroon_up = aroon_up / 100
            aroon_dn = aroon_dn / 100
            aroon_osc = aroon_osc / 100

        col_up, col_dn, col_osc = self._get_output_names()
        return df.with_columns(
            [
                pl.Series(name=col_up, values=aroon_up),
                pl.Series(name=col_dn, values=aroon_dn),
                pl.Series(name=col_osc, values=aroon_osc),
            ]
        )

    def _get_output_names(self) -> tuple[str, str, str]:
        """Generate output column names with normalization suffix."""
        suffix = "_norm" if self.normalized else ""
        return (
            f"aroon_up_{self.period}{suffix}",
            f"aroon_dn_{self.period}{suffix}",
            f"aroon_osc_{self.period}{suffix}",
        )

    test_params: ClassVar[list[dict]] = [
        {"period": 25},
        {"period": 25, "normalized": True},
        {"period": 60},
        {"period": 120},
    ]

    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable, reproducible output."""
        return self.period * 5


@dataclass
@sf_component(name="trend/vortex")
class VortexTrend(Feature):
    """Vortex Indicator.

    Captures positive and negative trend movement.

    Outputs:
    - vi_plus: positive vortex
    - vi_minus: negative vortex

    Unbounded. Uses z-score in normalized mode.

    Interpretation:
    - VI+ > VI-: bullish
    - VI- > VI+: bearish
    - Crossovers signal trend changes

    Reference: Etienne Botes & Douglas Siepman
    https://school.stockcharts.com/doku.php?id=technical_indicators:vortex_indicator
    """

    period: int = 14
    normalized: bool = False
    norm_period: int | None = None

    requires = ["high", "low", "close"]
    outputs = ["vi_plus_{period}", "vi_minus_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        close = df["close"].to_numpy()
        n = len(close)

        tr = np.maximum(
            high - low,
            np.maximum(
                np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))
            ),
        )
        tr[0] = high[0] - low[0]

        vm_plus = np.abs(high - np.roll(low, 1))
        vm_minus = np.abs(low - np.roll(high, 1))
        vm_plus[0] = vm_minus[0] = 0

        vi_plus = np.full(n, np.nan)
        vi_minus = np.full(n, np.nan)

        for i in range(self.period - 1, n):
            tr_sum = np.sum(tr[i - self.period + 1 : i + 1])
            if tr_sum > 0:
                vi_plus[i] = np.sum(vm_plus[i - self.period + 1 : i + 1]) / tr_sum
                vi_minus[i] = np.sum(vm_minus[i - self.period + 1 : i + 1]) / tr_sum

        # Normalization: z-score for unbounded oscillator
        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window

            norm_window = self.norm_period or get_norm_window(self.period)
            vi_plus = normalize_zscore(vi_plus, window=norm_window)
            vi_minus = normalize_zscore(vi_minus, window=norm_window)

        col_plus, col_minus = self._get_output_names()
        return df.with_columns(
            [
                pl.Series(name=col_plus, values=vi_plus),
                pl.Series(name=col_minus, values=vi_minus),
            ]
        )

    def _get_output_names(self) -> tuple[str, str]:
        """Generate output column names with normalization suffix."""
        suffix = "_norm" if self.normalized else ""
        return (f"vi_plus_{self.period}{suffix}", f"vi_minus_{self.period}{suffix}")

    test_params: ClassVar[list[dict]] = [
        {"period": 14},
        {"period": 14, "normalized": True},
        {"period": 30},
        {"period": 60},
    ]

    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable, reproducible output."""
        base_warmup = self.period * 5
        if self.normalized:
            from signalflow.ta._normalization import get_norm_window

            norm_window = self.norm_period or get_norm_window(self.period)
            return base_warmup + norm_window
        return base_warmup


@dataclass
@sf_component(name="trend/vhf")
class VhfTrend(Feature):
    """Vertical Horizontal Filter (VHF).

    Identifies trending vs ranging markets.

    VHF = |HCP - LCP| / Σ|ΔClose|

    Unbounded. Uses z-score in normalized mode.

    Higher values = stronger trend
    Lower values = ranging/consolidation

    Reference: Adam White
    https://www.incrediblecharts.com/indicators/vertical_horizontal_filter.php
    """

    period: int = 28
    normalized: bool = False
    norm_period: int | None = None

    requires = ["close"]
    outputs = ["vhf_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        close = df["close"].to_numpy()
        n = len(close)

        diff = np.abs(np.diff(close, prepend=close[0]))

        vhf = np.full(n, np.nan)

        for i in range(self.period - 1, n):
            window = close[i - self.period + 1 : i + 1]
            hcp = np.max(window)
            lcp = np.min(window)

            diff_sum = np.sum(diff[i - self.period + 1 : i + 1])

            if diff_sum > 0:
                vhf[i] = np.abs(hcp - lcp) / diff_sum

        # Normalization: z-score for unbounded oscillator
        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window

            norm_window = self.norm_period or get_norm_window(self.period)
            vhf = normalize_zscore(vhf, window=norm_window)

        col_name = self._get_output_name()
        return df.with_columns(pl.Series(name=col_name, values=vhf))

    def _get_output_name(self) -> str:
        """Generate output column name with normalization suffix."""
        suffix = "_norm" if self.normalized else ""
        return f"vhf_{self.period}{suffix}"

    test_params: ClassVar[list[dict]] = [
        {"period": 28},
        {"period": 28, "normalized": True},
        {"period": 60},
        {"period": 120},
    ]

    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable, reproducible output."""
        base_warmup = self.period * 5
        if self.normalized:
            from signalflow.ta._normalization import get_norm_window

            norm_window = self.norm_period or get_norm_window(self.period)
            return base_warmup + norm_window
        return base_warmup


@dataclass
@sf_component(name="trend/chop")
class ChopTrend(Feature):
    """Choppiness Index (CHOP).

    Determines if market is choppy (ranging) or trending.

    CHOP = 100 * log10(Σ ATR / (HH - LL)) / log10(n)

    Bounded [0, 100] (absolute) or [0, 1] (normalized).

    Interpretation (absolute mode):
    - CHOP > 61.8: choppy/ranging
    - CHOP < 38.2: trending
    - Between: transitional

    Interpretation (normalized mode):
    - CHOP > 0.618: choppy/ranging
    - CHOP < 0.382: trending
    - Between: transitional

    Reference: E.W. Dreiss
    https://www.tradingview.com/scripts/choppinessindex/
    """

    period: int = 14
    normalized: bool = False

    requires = ["high", "low", "close"]
    outputs = ["chop_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        close = df["close"].to_numpy()
        n = len(close)

        tr = np.maximum(
            high - low,
            np.maximum(
                np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))
            ),
        )
        tr[0] = high[0] - low[0]

        chop = np.full(n, np.nan)
        log_period = np.log10(self.period)

        for i in range(self.period - 1, n):
            hh = np.max(high[i - self.period + 1 : i + 1])
            ll = np.min(low[i - self.period + 1 : i + 1])
            tr_sum = np.sum(tr[i - self.period + 1 : i + 1])

            diff = hh - ll
            if diff > 0:
                chop[i] = 100 * np.log10(tr_sum / diff) / log_period

        # Normalization: [0, 100] → [0, 1]
        if self.normalized:
            chop = chop / 100

        col_name = self._get_output_name()
        return df.with_columns(pl.Series(name=col_name, values=chop))

    def _get_output_name(self) -> str:
        """Generate output column name with normalization suffix."""
        suffix = "_norm" if self.normalized else ""
        return f"chop_{self.period}{suffix}"

    test_params: ClassVar[list[dict]] = [
        {"period": 14},
        {"period": 14, "normalized": True},
        {"period": 30},
        {"period": 60},
    ]

    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable, reproducible output."""
        return self.period * 5


@dataclass
@sf_component(name="trend/viscosity")
class ViscosityTrend(Feature):
    """Market Viscosity - resistance to velocity change.

    viscosity = mean(|Δv|) / mean(|v|)

    where v = log-returns (velocity).

    Measures how "sticky" the market is: how much velocity fluctuation
    occurs relative to the average velocity.

    Interpretation:
    - High viscosity: market resists direction changes (sticky trend)
    - Low viscosity: market changes direction easily (whipsaw-prone)
    - Rising viscosity: trend solidifying
    - Falling viscosity: trend becoming fragile

    Reference: Fluid dynamics viscosity analogy
    """

    period: int = 20
    normalized: bool = False
    norm_period: int | None = None

    requires = ["close"]
    outputs = ["viscosity_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        close = df["close"].to_numpy()
        n = len(close)

        # Velocity (log-returns)
        velocity = np.full(n, np.nan)
        for i in range(1, n):
            if close[i - 1] > 0 and close[i] > 0:
                velocity[i] = np.log(close[i] / close[i - 1])

        # Acceleration (change in velocity)
        accel = np.full(n, np.nan)
        for i in range(2, n):
            if not np.isnan(velocity[i]) and not np.isnan(velocity[i - 1]):
                accel[i] = velocity[i] - velocity[i - 1]

        visc = np.full(n, np.nan)
        for i in range(self.period + 1, n):
            v_window = velocity[i - self.period + 1 : i + 1]
            a_window = accel[i - self.period + 1 : i + 1]

            v_valid = v_window[~np.isnan(v_window)]
            a_valid = a_window[~np.isnan(a_window)]

            if len(v_valid) > 0 and len(a_valid) > 0:
                mean_abs_v = np.mean(np.abs(v_valid))
                mean_abs_a = np.mean(np.abs(a_valid))
                if mean_abs_v > 1e-10:
                    visc[i] = mean_abs_a / mean_abs_v

        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window

            norm_window = self.norm_period or get_norm_window(self.period)
            visc = normalize_zscore(visc, window=norm_window)

        suffix = "_norm" if self.normalized else ""
        col_name = f"viscosity_{self.period}{suffix}"
        return df.with_columns(pl.Series(name=col_name, values=visc))

    test_params: ClassVar[list[dict]] = [
        {"period": 20},
        {"period": 20, "normalized": True},
        {"period": 50},
        {"period": 100},
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
@sf_component(name="trend/reynolds")
class ReynoldsTrend(Feature):
    """Market Reynolds Number - laminar vs turbulent regime.

    Re = |mean(v)| × period / std(v)

    Ratio of inertial forces (directed movement) to viscous forces
    (random fluctuation).

    Interpretation:
    - High Re: "laminar" flow (strong, ordered trend)
    - Low Re: "turbulent" flow (chaotic, noisy market)
    - Re transition: regime change between trending and ranging
    - More intuitive than VHF or CHOP for regime classification

    Reference: Reynolds number in fluid dynamics
    https://en.wikipedia.org/wiki/Reynolds_number
    """

    period: int = 20
    normalized: bool = False
    norm_period: int | None = None

    requires = ["close"]
    outputs = ["reynolds_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        close = df["close"].to_numpy()
        n = len(close)

        # Velocity (log-returns)
        velocity = np.full(n, np.nan)
        for i in range(1, n):
            if close[i - 1] > 0 and close[i] > 0:
                velocity[i] = np.log(close[i] / close[i - 1])

        reynolds = np.full(n, np.nan)
        for i in range(self.period, n):
            window = velocity[i - self.period + 1 : i + 1]
            valid = window[~np.isnan(window)]

            if len(valid) > 2:
                mean_v = np.mean(valid)
                std_v = np.std(valid, ddof=1)
                if std_v > 1e-10:
                    reynolds[i] = np.abs(mean_v) * self.period / std_v

        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window

            norm_window = self.norm_period or get_norm_window(self.period)
            reynolds = normalize_zscore(reynolds, window=norm_window)

        suffix = "_norm" if self.normalized else ""
        col_name = f"reynolds_{self.period}{suffix}"
        return df.with_columns(pl.Series(name=col_name, values=reynolds))

    test_params: ClassVar[list[dict]] = [
        {"period": 20},
        {"period": 20, "normalized": True},
        {"period": 50},
        {"period": 100},
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
@sf_component(name="trend/rotational_inertia")
class RotationalInertiaTrend(Feature):
    """Rotational Inertia (Moment of Inertia) - resistance to trend change.

    I = Σ(volume_i × distance_i²)

    where distance_i = ln(close_i / SMA) (log-displacement from mean).

    Interpretation:
    - High inertia: heavy volume far from mean (hard to reverse trend)
    - Low inertia: volume concentrated near mean (easy to change direction)
    - Rising inertia: trend becoming more entrenched
    - Falling inertia: trend becoming fragile

    Reference: Moment of inertia in rotational mechanics
    """

    period: int = 20
    normalized: bool = False
    norm_period: int | None = None

    requires = ["close", "volume"]
    outputs = ["rot_inertia_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        close = df["close"].to_numpy()
        volume = df["volume"].to_numpy().astype(np.float64)
        n = len(close)

        inertia = np.full(n, np.nan)
        for i in range(self.period - 1, n):
            c_window = close[i - self.period + 1 : i + 1]
            v_window = volume[i - self.period + 1 : i + 1]
            sma = np.mean(c_window)

            if sma > 1e-10:
                log_disp = np.log(c_window / sma)
                inertia[i] = np.sum(v_window * log_disp**2)

        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window

            norm_window = self.norm_period or get_norm_window(self.period)
            inertia = normalize_zscore(inertia, window=norm_window)

        suffix = "_norm" if self.normalized else ""
        col_name = f"rot_inertia_{self.period}{suffix}"
        return df.with_columns(pl.Series(name=col_name, values=inertia))

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
@sf_component(name="trend/market_impedance")
class MarketImpedanceTrend(Feature):
    """Market Impedance - opposition to price flow (Z = V/I analogy).

    Z = price_range / volume_flow

    Ratio of price movement to volume over rolling window.
    High impedance = large price change per unit volume.

    Interpretation:
    - High impedance: thin market, small volume causes big moves
    - Low impedance: deep market, price absorbs volume easily
    - Rising impedance: liquidity drying up (breakout setup)
    - Falling impedance: liquidity increasing (range-bound setup)

    Reference: Electrical impedance (Z = V/I)
    """

    period: int = 20
    normalized: bool = False
    norm_period: int | None = None

    requires = ["high", "low", "volume"]
    outputs = ["impedance_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        volume = df["volume"].to_numpy().astype(np.float64)
        n = len(high)

        impedance = np.full(n, np.nan)
        for i in range(self.period - 1, n):
            h_window = high[i - self.period + 1 : i + 1]
            l_window = low[i - self.period + 1 : i + 1]
            v_window = volume[i - self.period + 1 : i + 1]

            price_range = np.sum(h_window - l_window)
            vol_sum = np.sum(v_window)

            if vol_sum > 1e-10:
                impedance[i] = price_range / vol_sum

        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window

            norm_window = self.norm_period or get_norm_window(self.period)
            impedance = normalize_zscore(impedance, window=norm_window)

        suffix = "_norm" if self.normalized else ""
        col_name = f"impedance_{self.period}{suffix}"
        return df.with_columns(pl.Series(name=col_name, values=impedance))

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
@sf_component(name="trend/rc_time_constant")
class RCTimeConstantTrend(Feature):
    """RC Time Constant - market response time.

    τ = R × C

    R (resistance) = impedance (price_range / volume)
    C (capacitance) = volume / |Δprice|

    τ = (Σrange / Σvolume) × (Σvolume / |Δprice_total|)
      = Σrange / |Δprice_total|

    Measures how quickly the market "charges" to new levels.

    Interpretation:
    - High τ: slow response, price changing gradually (trending)
    - Low τ: fast response, price jumping quickly (breakout)
    - Rising τ: market slowing down
    - Falling τ: market speeding up

    Reference: RC circuit time constant (τ = RC)
    """

    period: int = 20
    normalized: bool = False
    norm_period: int | None = None

    requires = ["high", "low", "close"]
    outputs = ["rc_tau_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        close = df["close"].to_numpy()
        n = len(close)

        tau = np.full(n, np.nan)
        for i in range(self.period - 1, n):
            h_window = high[i - self.period + 1 : i + 1]
            l_window = low[i - self.period + 1 : i + 1]

            range_sum = np.sum(h_window - l_window)
            price_change = np.abs(close[i] - close[i - self.period + 1])

            if price_change > 1e-10:
                tau[i] = range_sum / price_change

        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window

            norm_window = self.norm_period or get_norm_window(self.period)
            tau = normalize_zscore(tau, window=norm_window)

        suffix = "_norm" if self.normalized else ""
        col_name = f"rc_tau_{self.period}{suffix}"
        return df.with_columns(pl.Series(name=col_name, values=tau))

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
@sf_component(name="trend/snr")
class SNRTrend(Feature):
    """Signal-to-Noise Ratio - trend clarity.

    SNR = |mean(returns)|² / var(returns)

    Ratio of directional signal power to noise power.

    Interpretation:
    - High SNR: clear directional signal (strong clean trend)
    - Low SNR: noise dominates (choppy, random walk)
    - SNR > 1: signal stronger than noise
    - SNR < 1: noise stronger than signal
    - Better than ADX for ML as it's a continuous ratio

    Reference: Signal-to-noise ratio in signal processing
    """

    period: int = 20
    normalized: bool = False
    norm_period: int | None = None

    requires = ["close"]
    outputs = ["snr_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        close = df["close"].to_numpy()
        n = len(close)

        # Log-returns
        velocity = np.full(n, np.nan)
        for i in range(1, n):
            if close[i - 1] > 0 and close[i] > 0:
                velocity[i] = np.log(close[i] / close[i - 1])

        snr = np.full(n, np.nan)
        for i in range(self.period, n):
            window = velocity[i - self.period + 1 : i + 1]
            valid = window[~np.isnan(window)]

            if len(valid) > 2:
                mean_v = np.mean(valid)
                var_v = np.var(valid, ddof=1)
                if var_v > 1e-15:
                    snr[i] = mean_v**2 / var_v

        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window

            norm_window = self.norm_period or get_norm_window(self.period)
            snr = normalize_zscore(snr, window=norm_window)

        suffix = "_norm" if self.normalized else ""
        col_name = f"snr_{self.period}{suffix}"
        return df.with_columns(pl.Series(name=col_name, values=snr))

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
@sf_component(name="trend/order_parameter")
class OrderParameterTrend(Feature):
    """Order Parameter - degree of collective alignment.

    ψ = |mean(sign(returns))|

    Fraction of bars moving in same direction.
    Like magnetization in Ising model.

    Interpretation:
    - ψ ≈ 1: all bars aligned (strong trend, fully ordered)
    - ψ ≈ 0: mixed directions (no trend, disordered)
    - ψ transition from 0→1: trend emerging (phase transition)
    - ψ transition from 1→0: trend breaking down

    Reference: Order parameter in statistical mechanics / Ising model
    https://en.wikipedia.org/wiki/Phase_transition#Order_parameters
    """

    period: int = 20
    normalized: bool = False

    requires = ["close"]
    outputs = ["order_param_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        close = df["close"].to_numpy()
        n = len(close)

        # Returns sign
        ret_sign = np.full(n, np.nan)
        for i in range(1, n):
            if close[i - 1] > 0:
                ret_sign[i] = np.sign(close[i] - close[i - 1])

        order = np.full(n, np.nan)
        for i in range(self.period, n):
            window = ret_sign[i - self.period + 1 : i + 1]
            valid = window[~np.isnan(window)]
            if len(valid) > 0:
                order[i] = np.abs(np.mean(valid))

        # Already bounded [0, 1]
        if self.normalized:
            order = order  # already [0, 1]

        suffix = "_norm" if self.normalized else ""
        col_name = f"order_param_{self.period}{suffix}"
        return df.with_columns(pl.Series(name=col_name, values=order))

    test_params: ClassVar[list[dict]] = [
        {"period": 20},
        {"period": 50},
        {"period": 100},
        {"period": 20, "normalized": True},
    ]

    @property
    def warmup(self) -> int:
        return self.period * 5


@dataclass
@sf_component(name="trend/susceptibility")
class SusceptibilityTrend(Feature):
    """Market Susceptibility - sensitivity to perturbation.

    χ = var(order_parameter) × period

    Variance of the order parameter over a rolling window.
    Peaks at phase transitions (trend ↔ range).

    Interpretation:
    - High susceptibility: market at critical point (regime change imminent)
    - Low susceptibility: stable regime (either clear trend or clear range)
    - Susceptibility spike: phase transition underway
    - Analogous to critical slowing down in complex systems

    Reference: Magnetic susceptibility near phase transitions
    https://en.wikipedia.org/wiki/Magnetic_susceptibility
    """

    period: int = 20
    chi_window: int = 50
    normalized: bool = False
    norm_period: int | None = None

    requires = ["close"]
    outputs = ["susceptibility_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        close = df["close"].to_numpy()
        n = len(close)

        # Returns sign
        ret_sign = np.full(n, np.nan)
        for i in range(1, n):
            if close[i - 1] > 0:
                ret_sign[i] = np.sign(close[i] - close[i - 1])

        # Order parameter
        order = np.full(n, np.nan)
        for i in range(self.period, n):
            window = ret_sign[i - self.period + 1 : i + 1]
            valid = window[~np.isnan(window)]
            if len(valid) > 0:
                order[i] = np.abs(np.mean(valid))

        # Susceptibility = variance of order parameter × period
        chi = np.full(n, np.nan)
        start = self.period + self.chi_window - 1
        for i in range(start, n):
            window = order[i - self.chi_window + 1 : i + 1]
            valid = window[~np.isnan(window)]
            if len(valid) > 2:
                chi[i] = np.var(valid, ddof=1) * self.period

        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window

            norm_window = self.norm_period or get_norm_window(self.chi_window)
            chi = normalize_zscore(chi, window=norm_window)

        suffix = "_norm" if self.normalized else ""
        col_name = f"susceptibility_{self.period}{suffix}"
        return df.with_columns(pl.Series(name=col_name, values=chi))

    test_params: ClassVar[list[dict]] = [
        {"period": 20, "chi_window": 50},
        {"period": 20, "chi_window": 100},
        {"period": 50, "chi_window": 100},
        {"period": 20, "chi_window": 50, "normalized": True},
    ]

    @property
    def warmup(self) -> int:
        base = (self.period + self.chi_window) * 5
        if self.normalized:
            from signalflow.ta._normalization import get_norm_window

            norm_window = self.norm_period or get_norm_window(self.chi_window)
            return base + norm_window
        return base

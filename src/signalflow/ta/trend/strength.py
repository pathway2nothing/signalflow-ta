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
                np.abs(high - np.roll(close, 1)),
                np.abs(low - np.roll(close, 1))
            )
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
        
        atr[self.period - 1] = np.mean(tr[:self.period])
        smooth_pdm[self.period - 1] = np.mean(pdm[:self.period])
        smooth_ndm[self.period - 1] = np.mean(ndm[:self.period])
        
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
            adx[start] = np.nanmean(dx[self.period:start + 1])
            for i in range(start + 1, n):
                adx[i] = alpha * dx[i] + (1 - alpha) * adx[i - 1]

        # Normalization: [0, 100] → [0, 1]
        if self.normalized:
            adx = adx / 100
            dmp = dmp / 100
            dmn = dmn / 100

        col_adx, col_dmp, col_dmn = self._get_output_names()
        return df.with_columns([
            pl.Series(name=col_adx, values=adx),
            pl.Series(name=col_dmp, values=dmp),
            pl.Series(name=col_dmn, values=dmn),
        ])

    def _get_output_names(self) -> tuple[str, str, str]:
        """Generate output column names with normalization suffix."""
        suffix = "_norm" if self.normalized else ""
        return (
            f"adx_{self.period}{suffix}",
            f"dmp_{self.period}{suffix}",
            f"dmn_{self.period}{suffix}"
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
            window_high = high[i - self.period:i + 1]
            window_low = low[i - self.period:i + 1]
            
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
        return df.with_columns([
            pl.Series(name=col_up, values=aroon_up),
            pl.Series(name=col_dn, values=aroon_dn),
            pl.Series(name=col_osc, values=aroon_osc),
        ])

    def _get_output_names(self) -> tuple[str, str, str]:
        """Generate output column names with normalization suffix."""
        suffix = "_norm" if self.normalized else ""
        return (
            f"aroon_up_{self.period}{suffix}",
            f"aroon_dn_{self.period}{suffix}",
            f"aroon_osc_{self.period}{suffix}"
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
                np.abs(high - np.roll(close, 1)),
                np.abs(low - np.roll(close, 1))
            )
        )
        tr[0] = high[0] - low[0]
        
        vm_plus = np.abs(high - np.roll(low, 1))
        vm_minus = np.abs(low - np.roll(high, 1))
        vm_plus[0] = vm_minus[0] = 0
        
        vi_plus = np.full(n, np.nan)
        vi_minus = np.full(n, np.nan)
        
        for i in range(self.period - 1, n):
            tr_sum = np.sum(tr[i - self.period + 1:i + 1])
            if tr_sum > 0:
                vi_plus[i] = np.sum(vm_plus[i - self.period + 1:i + 1]) / tr_sum
                vi_minus[i] = np.sum(vm_minus[i - self.period + 1:i + 1]) / tr_sum

        # Normalization: z-score for unbounded oscillator
        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window
            norm_window = self.norm_period or get_norm_window(self.period)
            vi_plus = normalize_zscore(vi_plus, window=norm_window)
            vi_minus = normalize_zscore(vi_minus, window=norm_window)

        col_plus, col_minus = self._get_output_names()
        return df.with_columns([
            pl.Series(name=col_plus, values=vi_plus),
            pl.Series(name=col_minus, values=vi_minus),
        ])

    def _get_output_names(self) -> tuple[str, str]:
        """Generate output column names with normalization suffix."""
        suffix = "_norm" if self.normalized else ""
        return (
            f"vi_plus_{self.period}{suffix}",
            f"vi_minus_{self.period}{suffix}"
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
            window = close[i - self.period + 1:i + 1]
            hcp = np.max(window)  
            lcp = np.min(window)  
            
            diff_sum = np.sum(diff[i - self.period + 1:i + 1])

            if diff_sum > 0:
                vhf[i] = np.abs(hcp - lcp) / diff_sum

        # Normalization: z-score for unbounded oscillator
        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window
            norm_window = self.norm_period or get_norm_window(self.period)
            vhf = normalize_zscore(vhf, window=norm_window)

        col_name = self._get_output_name()
        return df.with_columns(
            pl.Series(name=col_name, values=vhf)
        )

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
                np.abs(high - np.roll(close, 1)),
                np.abs(low - np.roll(close, 1))
            )
        )
        tr[0] = high[0] - low[0]
        
        chop = np.full(n, np.nan)
        log_period = np.log10(self.period)
        
        for i in range(self.period - 1, n):
            hh = np.max(high[i - self.period + 1:i + 1])
            ll = np.min(low[i - self.period + 1:i + 1])
            tr_sum = np.sum(tr[i - self.period + 1:i + 1])
            
            diff = hh - ll
            if diff > 0:
                chop[i] = 100 * np.log10(tr_sum / diff) / log_period

        # Normalization: [0, 100] → [0, 1]
        if self.normalized:
            chop = chop / 100

        col_name = self._get_output_name()
        return df.with_columns(
            pl.Series(name=col_name, values=chop)
        )

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

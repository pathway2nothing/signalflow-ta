"""Trend strength indicators - measure how strong a trend is."""
from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl

from signalflow.core import sf_component
from signalflow.feature.base import Feature


@dataclass
@sf_component(name="trend/adx")
class AdxTrend(Feature):
    """Average Directional Index (ADX).
    
    Measures trend strength regardless of direction.
    
    Outputs:
    - adx: trend strength (0-100)
    - dmp: +DI (positive directional indicator)
    - dmn: -DI (negative directional indicator)
    
    Interpretation:
    - ADX < 20: weak/no trend (ranging)
    - ADX 20-40: developing trend
    - ADX > 40: strong trend
    - +DI > -DI: bullish
    - -DI > +DI: bearish
    
    Reference: Welles Wilder, "New Concepts in Technical Trading Systems"
    https://www.investopedia.com/terms/a/adx.asp
    """
    
    period: int = 14
    
    requires = ["high", "low", "close"]
    outputs = ["adx_{period}", "dmp_{period}", "dmn_{period}"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        close = df["close"].to_numpy()
        n = len(close)
        
        # True Range
        tr = np.maximum(
            high - low,
            np.maximum(
                np.abs(high - np.roll(close, 1)),
                np.abs(low - np.roll(close, 1))
            )
        )
        tr[0] = high[0] - low[0]
        
        # Directional Movement
        up = high - np.roll(high, 1)
        dn = np.roll(low, 1) - low
        up[0] = dn[0] = 0
        
        # +DM and -DM
        pdm = np.where((up > dn) & (up > 0), up, 0)
        ndm = np.where((dn > up) & (dn > 0), dn, 0)
        
        # Smoothed TR, +DM, -DM using Wilder's smoothing (RMA)
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
        
        # +DI and -DI
        dmp = 100 * smooth_pdm / atr
        dmn = 100 * smooth_ndm / atr
        
        # DX and ADX
        dx = 100 * np.abs(dmp - dmn) / (dmp + dmn + 1e-10)
        
        adx = np.full(n, np.nan)
        start = 2 * self.period - 1
        if start < n:
            adx[start] = np.nanmean(dx[self.period:start + 1])
            for i in range(start + 1, n):
                adx[i] = alpha * dx[i] + (1 - alpha) * adx[i - 1]
        
        return df.with_columns([
            pl.Series(name=f"adx_{self.period}", values=adx),
            pl.Series(name=f"dmp_{self.period}", values=dmp),
            pl.Series(name=f"dmn_{self.period}", values=dmn),
        ])


@dataclass
@sf_component(name="trend/aroon")
class AroonTrend(Feature):
    """Aroon Indicator.
    
    Measures time since highest high / lowest low.
    
    Outputs:
    - aroon_up: periods since highest high (0-100)
    - aroon_dn: periods since lowest low (0-100)
    - aroon_osc: aroon_up - aroon_dn (-100 to 100)
    
    Interpretation:
    - aroon_up > 70: strong uptrend
    - aroon_dn > 70: strong downtrend
    - aroon_osc > 0: bullish
    - aroon_osc < 0: bearish
    
    Reference: Tushar Chande
    https://www.investopedia.com/terms/a/aroon.asp
    """
    
    period: int = 25
    
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
            
            # Periods since highest high (0 = today, period = oldest)
            periods_from_hh = self.period - np.argmax(window_high[::-1])
            periods_from_ll = self.period - np.argmin(window_low[::-1])
            
            aroon_up[i] = 100 * (self.period - periods_from_hh) / self.period
            aroon_dn[i] = 100 * (self.period - periods_from_ll) / self.period
        
        aroon_osc = aroon_up - aroon_dn
        
        return df.with_columns([
            pl.Series(name=f"aroon_up_{self.period}", values=aroon_up),
            pl.Series(name=f"aroon_dn_{self.period}", values=aroon_dn),
            pl.Series(name=f"aroon_osc_{self.period}", values=aroon_osc),
        ])


@dataclass
@sf_component(name="trend/vortex")
class VortexTrend(Feature):
    """Vortex Indicator.
    
    Captures positive and negative trend movement.
    
    Outputs:
    - vi_plus: positive vortex
    - vi_minus: negative vortex
    
    Interpretation:
    - VI+ > VI-: bullish
    - VI- > VI+: bearish
    - Crossovers signal trend changes
    
    Reference: Etienne Botes & Douglas Siepman
    https://school.stockcharts.com/doku.php?id=technical_indicators:vortex_indicator
    """
    
    period: int = 14
    
    requires = ["high", "low", "close"]
    outputs = ["vi_plus_{period}", "vi_minus_{period}"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        close = df["close"].to_numpy()
        n = len(close)
        
        # True Range
        tr = np.maximum(
            high - low,
            np.maximum(
                np.abs(high - np.roll(close, 1)),
                np.abs(low - np.roll(close, 1))
            )
        )
        tr[0] = high[0] - low[0]
        
        # Vortex Movement
        vm_plus = np.abs(high - np.roll(low, 1))
        vm_minus = np.abs(low - np.roll(high, 1))
        vm_plus[0] = vm_minus[0] = 0
        
        # Rolling sums
        vi_plus = np.full(n, np.nan)
        vi_minus = np.full(n, np.nan)
        
        for i in range(self.period - 1, n):
            tr_sum = np.sum(tr[i - self.period + 1:i + 1])
            if tr_sum > 0:
                vi_plus[i] = np.sum(vm_plus[i - self.period + 1:i + 1]) / tr_sum
                vi_minus[i] = np.sum(vm_minus[i - self.period + 1:i + 1]) / tr_sum
        
        return df.with_columns([
            pl.Series(name=f"vi_plus_{self.period}", values=vi_plus),
            pl.Series(name=f"vi_minus_{self.period}", values=vi_minus),
        ])


@dataclass
@sf_component(name="trend/vhf")
class VhfTrend(Feature):
    """Vertical Horizontal Filter (VHF).
    
    Identifies trending vs ranging markets.
    
    VHF = |HCP - LCP| / Σ|ΔClose|
    
    Higher values = stronger trend
    Lower values = ranging/consolidation
    
    Reference: Adam White
    https://www.incrediblecharts.com/indicators/vertical_horizontal_filter.php
    """
    
    period: int = 28
    
    requires = ["close"]
    outputs = ["vhf_{period}"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        close = df["close"].to_numpy()
        n = len(close)
        
        # Absolute price changes
        diff = np.abs(np.diff(close, prepend=close[0]))
        
        vhf = np.full(n, np.nan)
        
        for i in range(self.period - 1, n):
            window = close[i - self.period + 1:i + 1]
            hcp = np.max(window)  # Highest close
            lcp = np.min(window)  # Lowest close
            
            diff_sum = np.sum(diff[i - self.period + 1:i + 1])
            
            if diff_sum > 0:
                vhf[i] = np.abs(hcp - lcp) / diff_sum
        
        return df.with_columns(
            pl.Series(name=f"vhf_{self.period}", values=vhf)
        )


@dataclass
@sf_component(name="trend/chop")
class ChopTrend(Feature):
    """Choppiness Index (CHOP).
    
    Determines if market is choppy (ranging) or trending.
    
    CHOP = 100 * log10(Σ ATR / (HH - LL)) / log10(n)
    
    Interpretation:
    - CHOP > 61.8: choppy/ranging
    - CHOP < 38.2: trending
    - Between: transitional
    
    Reference: E.W. Dreiss
    https://www.tradingview.com/scripts/choppinessindex/
    """
    
    period: int = 14
    
    requires = ["high", "low", "close"]
    outputs = ["chop_{period}"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        close = df["close"].to_numpy()
        n = len(close)
        
        # True Range
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
        
        return df.with_columns(
            pl.Series(name=f"chop_{self.period}", values=chop)
        )
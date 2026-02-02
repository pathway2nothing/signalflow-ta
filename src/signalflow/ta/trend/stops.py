"""Trailing stop indicators - trend-following with defined exits."""
from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl

from signalflow.core import sf_component
from signalflow.feature.base import Feature


@dataclass
@sf_component(name="trend/psar")
class PsarTrend(Feature):
    """Parabolic SAR (Stop and Reverse).
    
    Trailing stop that accelerates with trend.
    
    SAR = SAR_prev + AF * (EP - SAR_prev)
    AF: acceleration factor (starts at af, increases by af_step to af_max)
    EP: extreme point (highest high / lowest low)
    
    Outputs:
    - psar: SAR value
    - psar_dir: direction (+1 bullish, -1 bearish)
    
    Reference: Welles Wilder, "New Concepts in Technical Trading Systems"
    """
    
    af: float = 0.02        # initial acceleration factor
    af_step: float = 0.02   # AF increment
    af_max: float = 0.2     # maximum AF
    
    requires = ["high", "low", "close"]
    outputs = ["psar", "psar_dir"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        close = df["close"].to_numpy()
        n = len(close)
        
        psar = np.full(n, np.nan)
        direction = np.ones(n)
        
        # Initialize
        psar[0] = close[0]
        af = self.af
        
        # Determine initial trend
        if n > 1 and close[1] > close[0]:
            direction[0] = 1
            ep = high[0]
            psar[0] = low[0]
        else:
            direction[0] = -1
            ep = low[0]
            psar[0] = high[0]
        
        for i in range(1, n):
            prev_psar = psar[i - 1]
            prev_dir = direction[i - 1]
            
            if prev_dir == 1:  # Uptrend
                psar[i] = prev_psar + af * (ep - prev_psar)
                # SAR cannot be above prior two lows
                psar[i] = min(psar[i], low[i - 1])
                if i > 1:
                    psar[i] = min(psar[i], low[i - 2])
                
                if low[i] < psar[i]:  # Reversal to downtrend
                    direction[i] = -1
                    psar[i] = ep
                    ep = low[i]
                    af = self.af
                else:
                    direction[i] = 1
                    if high[i] > ep:
                        ep = high[i]
                        af = min(af + self.af_step, self.af_max)
            else:  # Downtrend
                psar[i] = prev_psar + af * (ep - prev_psar)
                # SAR cannot be below prior two highs
                psar[i] = max(psar[i], high[i - 1])
                if i > 1:
                    psar[i] = max(psar[i], high[i - 2])
                
                if high[i] > psar[i]:  # Reversal to uptrend
                    direction[i] = 1
                    psar[i] = ep
                    ep = high[i]
                    af = self.af
                else:
                    direction[i] = -1
                    if low[i] < ep:
                        ep = low[i]
                        af = min(af + self.af_step, self.af_max)
        
        return df.with_columns([
            pl.Series(name="psar", values=psar),
            pl.Series(name="psar_dir", values=direction),
        ])


@dataclass
@sf_component(name="trend/supertrend")
class SupertrendTrend(Feature):
    """Supertrend Indicator.
    
    Trend-following based on ATR bands.
    
    upper = HL2 + multiplier * ATR
    lower = HL2 - multiplier * ATR
    
    Outputs:
    - supertrend: trend line value
    - supertrend_dir: direction (+1 bullish, -1 bearish)
    
    Reference: Olivier Seban
    """
    
    period: int = 10
    multiplier: float = 3.0
    
    requires = ["high", "low", "close"]
    outputs = ["supertrend_{period}", "supertrend_dir_{period}"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        close = df["close"].to_numpy()
        n = len(close)
        
        # ATR
        tr = np.maximum(
            high - low,
            np.maximum(
                np.abs(high - np.roll(close, 1)),
                np.abs(low - np.roll(close, 1))
            )
        )
        tr[0] = high[0] - low[0]
        
        atr = np.full(n, np.nan)
        atr[self.period - 1] = np.mean(tr[:self.period])
        alpha = 1 / self.period
        for i in range(self.period, n):
            atr[i] = alpha * tr[i] + (1 - alpha) * atr[i - 1]
        
        # HL2
        hl2 = (high + low) / 2
        
        # Basic bands
        basic_upper = hl2 + self.multiplier * atr
        basic_lower = hl2 - self.multiplier * atr
        
        # Final bands
        upper = np.full(n, np.nan)
        lower = np.full(n, np.nan)
        supertrend = np.full(n, np.nan)
        direction = np.ones(n)
        
        upper[self.period - 1] = basic_upper[self.period - 1]
        lower[self.period - 1] = basic_lower[self.period - 1]
        
        for i in range(self.period, n):
            # Upper band with trailing
            if basic_upper[i] < upper[i - 1] or close[i - 1] > upper[i - 1]:
                upper[i] = basic_upper[i]
            else:
                upper[i] = upper[i - 1]
            
            # Lower band with trailing
            if basic_lower[i] > lower[i - 1] or close[i - 1] < lower[i - 1]:
                lower[i] = basic_lower[i]
            else:
                lower[i] = lower[i - 1]
            
            # Direction
            if close[i] > upper[i - 1]:
                direction[i] = 1
            elif close[i] < lower[i - 1]:
                direction[i] = -1
            else:
                direction[i] = direction[i - 1]
            
            # Supertrend value
            supertrend[i] = lower[i] if direction[i] == 1 else upper[i]
        
        return df.with_columns([
            pl.Series(name=f"supertrend_{self.period}", values=supertrend),
            pl.Series(name=f"supertrend_dir_{self.period}", values=direction),
        ])


@dataclass
@sf_component(name="trend/chandelier")
class ChandelierTrend(Feature):
    """Chandelier Exit.
    
    Trailing stop based on ATR from highest high / lowest low.
    
    Long exit: HH - multiplier * ATR
    Short exit: LL + multiplier * ATR
    
    Outputs:
    - chandelier_long: long trailing stop
    - chandelier_short: short trailing stop
    
    Reference: Chuck LeBeau, Alexander Elder
    """
    
    period: int = 22
    multiplier: float = 3.0
    
    requires = ["high", "low", "close"]
    outputs = ["chandelier_long_{period}", "chandelier_short_{period}"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        close = df["close"].to_numpy()
        n = len(close)
        
        # ATR (SMA version for Chandelier)
        tr = np.maximum(
            high - low,
            np.maximum(
                np.abs(high - np.roll(close, 1)),
                np.abs(low - np.roll(close, 1))
            )
        )
        tr[0] = high[0] - low[0]
        
        chandelier_long = np.full(n, np.nan)
        chandelier_short = np.full(n, np.nan)
        
        for i in range(self.period - 1, n):
            atr_val = np.mean(tr[i - self.period + 1:i + 1])
            hh = np.max(high[i - self.period + 1:i + 1])
            ll = np.min(low[i - self.period + 1:i + 1])
            
            chandelier_long[i] = hh - self.multiplier * atr_val
            chandelier_short[i] = ll + self.multiplier * atr_val
        
        return df.with_columns([
            pl.Series(name=f"chandelier_long_{self.period}", values=chandelier_long),
            pl.Series(name=f"chandelier_short_{self.period}", values=chandelier_short),
        ])


@dataclass
@sf_component(name="trend/hilo")
class HiloTrend(Feature):
    """Gann HiLo Activator.
    
    Trend indicator using moving averages of highs and lows.
    
    Switches between:
    - MA(low) when close > MA(high) [uptrend]
    - MA(high) when close < MA(low) [downtrend]
    
    Outputs:
    - hilo: current stop level
    - hilo_dir: direction (+1 bullish, -1 bearish)
    
    Reference: Robert Krausz, Stocks & Commodities 1998
    """
    
    high_period: int = 13
    low_period: int = 21
    ma_type: Literal["sma", "ema"] = "sma"
    
    requires = ["high", "low", "close"]
    outputs = ["hilo", "hilo_dir"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        close = df["close"].to_numpy()
        n = len(close)
        
        # Compute MAs
        if self.ma_type == "ema":
            alpha_h = 2 / (self.high_period + 1)
            alpha_l = 2 / (self.low_period + 1)
            
            high_ma = np.full(n, np.nan)
            low_ma = np.full(n, np.nan)
            high_ma[0] = high[0]
            low_ma[0] = low[0]
            
            for i in range(1, n):
                high_ma[i] = alpha_h * high[i] + (1 - alpha_h) * high_ma[i - 1]
                low_ma[i] = alpha_l * low[i] + (1 - alpha_l) * low_ma[i - 1]
        else:  # SMA
            high_ma = np.full(n, np.nan)
            low_ma = np.full(n, np.nan)
            for i in range(self.high_period - 1, n):
                high_ma[i] = np.mean(high[i - self.high_period + 1:i + 1])
            for i in range(self.low_period - 1, n):
                low_ma[i] = np.mean(low[i - self.low_period + 1:i + 1])
        
        # HiLo logic
        hilo = np.full(n, np.nan)
        direction = np.zeros(n)
        
        max_period = max(self.high_period, self.low_period)
        
        for i in range(max_period, n):
            if close[i] > high_ma[i - 1]:
                hilo[i] = low_ma[i]
                direction[i] = 1
            elif close[i] < low_ma[i - 1]:
                hilo[i] = high_ma[i]
                direction[i] = -1
            else:
                hilo[i] = hilo[i - 1]
                direction[i] = direction[i - 1]
        
        return df.with_columns([
            pl.Series(name="hilo", values=hilo),
            pl.Series(name="hilo_dir", values=direction),
        ])


@dataclass
@sf_component(name="trend/cksp")
class CkspTrend(Feature):
    """Chande Kroll Stop.
    
    ATR-based trailing stop with two-step smoothing.
    
    Step 1: Initial stop = HH - x*ATR or LL + x*ATR
    Step 2: Final stop = max/min of initial over q periods
    
    Outputs:
    - cksp_long: long trailing stop
    - cksp_short: short trailing stop
    
    Reference: Tushar Chande & Stanley Kroll, "The New Technical Trader"
    """
    
    p: int = 10      # ATR period
    x: float = 1.0   # ATR multiplier
    q: int = 9       # smoothing period
    
    requires = ["high", "low", "close"]
    outputs = ["cksp_long", "cksp_short"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        close = df["close"].to_numpy()
        n = len(close)
        
        # ATR using RMA (Wilder's smoothing)
        tr = np.maximum(
            high - low,
            np.maximum(
                np.abs(high - np.roll(close, 1)),
                np.abs(low - np.roll(close, 1))
            )
        )
        tr[0] = high[0] - low[0]
        
        atr = np.full(n, np.nan)
        atr[self.p - 1] = np.mean(tr[:self.p])
        alpha = 1 / self.p
        for i in range(self.p, n):
            atr[i] = alpha * tr[i] + (1 - alpha) * atr[i - 1]
        
        # Initial stops
        long_stop_init = np.full(n, np.nan)
        short_stop_init = np.full(n, np.nan)
        
        for i in range(self.p - 1, n):
            hh = np.max(high[i - self.p + 1:i + 1])
            ll = np.min(low[i - self.p + 1:i + 1])
            long_stop_init[i] = hh - self.x * atr[i]
            short_stop_init[i] = ll + self.x * atr[i]
        
        # Final stops (smoothed)
        cksp_long = np.full(n, np.nan)
        cksp_short = np.full(n, np.nan)
        
        start = self.p + self.q - 2
        for i in range(start, n):
            cksp_long[i] = np.nanmax(long_stop_init[i - self.q + 1:i + 1])
            cksp_short[i] = np.nanmin(short_stop_init[i - self.q + 1:i + 1])
        
        return df.with_columns([
            pl.Series(name="cksp_long", values=cksp_long),
            pl.Series(name="cksp_short", values=cksp_short),
        ])
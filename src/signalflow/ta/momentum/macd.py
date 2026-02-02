"""MACD family indicators."""
from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl

from signalflow import sf_component
from signalflow.feature.base import Feature


@dataclass
@sf_component(name="momentum/macd")
class MacdMom(Feature):
    """Moving Average Convergence Divergence (MACD).
    
    Trend-following momentum indicator.
    
    MACD = EMA(close, fast) - EMA(close, slow)
    Signal = EMA(MACD, signal)
    Histogram = MACD - Signal
    
    Outputs:
    - macd: MACD line
    - macd_signal: signal line
    - macd_hist: histogram (momentum)
    
    Interpretation:
    - MACD above signal: bullish
    - MACD below signal: bearish
    - Histogram shows momentum strength
    - Zero line crossovers: trend change
    
    Reference: Gerald Appel
    https://www.investopedia.com/terms/m/macd.asp
    """
    
    fast: int = 12
    slow: int = 26
    signal: int = 9
    
    requires = ["close"]
    outputs = ["macd_{fast}_{slow}", "macd_signal_{signal}", "macd_hist_{fast}_{slow}"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        close = df["close"].to_numpy()
        n = len(close)
        
        # Fast EMA
        alpha_fast = 2 / (self.fast + 1)
        ema_fast = np.full(n, np.nan)
        ema_fast[0] = close[0]
        for i in range(1, n):
            ema_fast[i] = alpha_fast * close[i] + (1 - alpha_fast) * ema_fast[i - 1]
        
        # Slow EMA
        alpha_slow = 2 / (self.slow + 1)
        ema_slow = np.full(n, np.nan)
        ema_slow[0] = close[0]
        for i in range(1, n):
            ema_slow[i] = alpha_slow * close[i] + (1 - alpha_slow) * ema_slow[i - 1]
        
        # MACD line
        macd = ema_fast - ema_slow
        
        # Signal line
        alpha_sig = 2 / (self.signal + 1)
        signal_line = np.full(n, np.nan)
        signal_line[self.slow - 1] = macd[self.slow - 1]
        for i in range(self.slow, n):
            signal_line[i] = alpha_sig * macd[i] + (1 - alpha_sig) * signal_line[i - 1]
        
        # Histogram
        histogram = macd - signal_line
        
        # Set proper NaN start
        macd[:self.slow - 1] = np.nan
        
        return df.with_columns([
            pl.Series(name=f"macd_{self.fast}_{self.slow}", values=macd),
            pl.Series(name=f"macd_signal_{self.signal}", values=signal_line),
            pl.Series(name=f"macd_hist_{self.fast}_{self.slow}", values=histogram),
        ])


@dataclass
@sf_component(name="momentum/ppo")
class PpoMom(Feature):
    """Percentage Price Oscillator (PPO).
    
    MACD expressed as percentage.
    
    PPO = 100 * (EMA_fast - EMA_slow) / EMA_slow
    Signal = EMA(PPO, signal)
    Histogram = PPO - Signal
    
    Advantage over MACD:
    - Comparable across different price levels
    - Better for comparing securities
    
    Reference: https://school.stockcharts.com/doku.php?id=technical_indicators:price_oscillators_ppo
    """
    
    fast: int = 12
    slow: int = 26
    signal: int = 9
    
    requires = ["close"]
    outputs = ["ppo_{fast}_{slow}", "ppo_signal_{signal}", "ppo_hist_{fast}_{slow}"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        close = df["close"].to_numpy()
        n = len(close)
        
        # EMAs
        alpha_fast = 2 / (self.fast + 1)
        alpha_slow = 2 / (self.slow + 1)
        
        ema_fast = np.full(n, np.nan)
        ema_slow = np.full(n, np.nan)
        ema_fast[0] = close[0]
        ema_slow[0] = close[0]
        
        for i in range(1, n):
            ema_fast[i] = alpha_fast * close[i] + (1 - alpha_fast) * ema_fast[i - 1]
            ema_slow[i] = alpha_slow * close[i] + (1 - alpha_slow) * ema_slow[i - 1]
        
        # PPO
        ppo = 100 * (ema_fast - ema_slow) / (ema_slow + 1e-10)
        
        # Signal line
        alpha_sig = 2 / (self.signal + 1)
        signal_line = np.full(n, np.nan)
        signal_line[self.slow - 1] = ppo[self.slow - 1]
        for i in range(self.slow, n):
            signal_line[i] = alpha_sig * ppo[i] + (1 - alpha_sig) * signal_line[i - 1]
        
        histogram = ppo - signal_line
        ppo[:self.slow - 1] = np.nan
        
        return df.with_columns([
            pl.Series(name=f"ppo_{self.fast}_{self.slow}", values=ppo),
            pl.Series(name=f"ppo_signal_{self.signal}", values=signal_line),
            pl.Series(name=f"ppo_hist_{self.fast}_{self.slow}", values=histogram),
        ])


@dataclass
@sf_component(name="momentum/tsi")
class TsiMom(Feature):
    """True Strength Index (TSI).
    
    Double-smoothed momentum indicator.
    
    PC = close - prev_close
    TSI = 100 * EMA(EMA(PC, slow), fast) / EMA(EMA(|PC|, slow), fast)
    Signal = EMA(TSI, signal)
    
    Outputs:
    - tsi: True Strength Index
    - tsi_signal: signal line
    
    Bounded approximately -100 to +100:
    - Positive: bullish momentum
    - Negative: bearish momentum
    - Signal crossovers: trading signals
    
    Reference: William Blau
    https://www.investopedia.com/terms/t/tsi.asp
    """
    
    fast: int = 13
    slow: int = 25
    signal: int = 13
    
    requires = ["close"]
    outputs = ["tsi_{fast}_{slow}", "tsi_signal_{signal}"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        close = df["close"].to_numpy()
        n = len(close)
        
        # Price change
        pc = np.diff(close, prepend=close[0])
        pc[0] = 0
        abs_pc = np.abs(pc)
        
        alpha_slow = 2 / (self.slow + 1)
        alpha_fast = 2 / (self.fast + 1)
        alpha_sig = 2 / (self.signal + 1)
        
        # Double smooth PC
        pc_ema1 = np.full(n, np.nan)
        pc_ema2 = np.full(n, np.nan)
        abs_pc_ema1 = np.full(n, np.nan)
        abs_pc_ema2 = np.full(n, np.nan)
        
        pc_ema1[0] = pc[0]
        abs_pc_ema1[0] = abs_pc[0]
        
        for i in range(1, n):
            pc_ema1[i] = alpha_slow * pc[i] + (1 - alpha_slow) * pc_ema1[i - 1]
            abs_pc_ema1[i] = alpha_slow * abs_pc[i] + (1 - alpha_slow) * abs_pc_ema1[i - 1]
        
        pc_ema2[0] = pc_ema1[0]
        abs_pc_ema2[0] = abs_pc_ema1[0]
        
        for i in range(1, n):
            pc_ema2[i] = alpha_fast * pc_ema1[i] + (1 - alpha_fast) * pc_ema2[i - 1]
            abs_pc_ema2[i] = alpha_fast * abs_pc_ema1[i] + (1 - alpha_fast) * abs_pc_ema2[i - 1]
        
        # TSI
        tsi = 100 * pc_ema2 / (abs_pc_ema2 + 1e-10)
        
        # Signal
        tsi_signal = np.full(n, np.nan)
        tsi_signal[0] = tsi[0]
        for i in range(1, n):
            tsi_signal[i] = alpha_sig * tsi[i] + (1 - alpha_sig) * tsi_signal[i - 1]
        
        return df.with_columns([
            pl.Series(name=f"tsi_{self.fast}_{self.slow}", values=tsi),
            pl.Series(name=f"tsi_signal_{self.signal}", values=tsi_signal),
        ])


@dataclass
@sf_component(name="momentum/trix")
class TrixMom(Feature):
    """Triple Exponential Average (TRIX).
    
    Rate of change of triple-smoothed EMA.
    
    EMA1 = EMA(close, period)
    EMA2 = EMA(EMA1, period)
    EMA3 = EMA(EMA2, period)
    TRIX = 100 * (EMA3 - EMA3[1]) / EMA3[1]
    
    Outputs:
    - trix: TRIX oscillator
    - trix_signal: signal line
    
    Very smooth indicator:
    - Positive: bullish trend
    - Negative: bearish trend
    - Zero crossings: trend changes
    - Signal crossings: trading signals
    
    Reference: Jack Hutson
    https://www.investopedia.com/terms/t/trix.asp
    """
    
    period: int = 18
    signal: int = 9
    
    requires = ["close"]
    outputs = ["trix_{period}", "trix_signal_{signal}"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        close = df["close"].to_numpy()
        n = len(close)
        
        alpha = 2 / (self.period + 1)
        
        # Triple EMA
        ema1 = np.full(n, np.nan)
        ema2 = np.full(n, np.nan)
        ema3 = np.full(n, np.nan)
        
        ema1[0] = close[0]
        ema2[0] = close[0]
        ema3[0] = close[0]
        
        for i in range(1, n):
            ema1[i] = alpha * close[i] + (1 - alpha) * ema1[i - 1]
            ema2[i] = alpha * ema1[i] + (1 - alpha) * ema2[i - 1]
            ema3[i] = alpha * ema2[i] + (1 - alpha) * ema3[i - 1]
        
        # TRIX = percentage change of EMA3
        trix = np.full(n, np.nan)
        for i in range(1, n):
            if ema3[i - 1] != 0:
                trix[i] = 100 * (ema3[i] - ema3[i - 1]) / ema3[i - 1]
        
        # Signal line
        alpha_sig = 2 / (self.signal + 1)
        trix_signal = np.full(n, np.nan)
        
        # Find first valid trix
        start = 1
        trix_signal[start] = trix[start]
        for i in range(start + 1, n):
            if not np.isnan(trix[i]) and not np.isnan(trix_signal[i - 1]):
                trix_signal[i] = alpha_sig * trix[i] + (1 - alpha_sig) * trix_signal[i - 1]
        
        return df.with_columns([
            pl.Series(name=f"trix_{self.period}", values=trix),
            pl.Series(name=f"trix_signal_{self.signal}", values=trix_signal),
        ])

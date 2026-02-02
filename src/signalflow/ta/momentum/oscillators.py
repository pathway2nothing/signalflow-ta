"""Stochastic and other momentum oscillators."""
from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl

from signalflow import sf_component
from signalflow.feature.base import Feature
from typing import ClassVar


@dataclass
@sf_component(name="momentum/stoch")
class StochMom(Feature):
    """Stochastic Oscillator.
    
    Compares close to high-low range over period.
    
    %K = 100 * (close - lowest_low) / (highest_high - lowest_low)
    %D = SMA(%K, d_period)
    
    Outputs:
    - stoch_k: fast stochastic (smoothed)
    - stoch_d: slow stochastic (signal line)
    
    Interpretation:
    - > 80: overbought
    - < 20: oversold
    - %K crossing %D: signal
    
    Reference: George Lane
    https://www.investopedia.com/terms/s/stochasticoscillator.asp
    """
    
    k_period: int = 14
    d_period: int = 3
    smooth_k: int = 3
    
    requires = ["high", "low", "close"]
    outputs = ["stoch_k_{k_period}", "stoch_d_{k_period}"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        close = df["close"].to_numpy()
        n = len(close)
        
        raw_k = np.full(n, np.nan)
        
        for i in range(self.k_period - 1, n):
            hh = np.max(high[i - self.k_period + 1:i + 1])
            ll = np.min(low[i - self.k_period + 1:i + 1])
            
            if hh != ll:
                raw_k[i] = 100 * (close[i] - ll) / (hh - ll)
        
        stoch_k = np.full(n, np.nan)
        for i in range(self.k_period + self.smooth_k - 2, n):
            stoch_k[i] = np.nanmean(raw_k[i - self.smooth_k + 1:i + 1])
        
        stoch_d = np.full(n, np.nan)
        for i in range(self.k_period + self.smooth_k + self.d_period - 3, n):
            stoch_d[i] = np.nanmean(stoch_k[i - self.d_period + 1:i + 1])
        
        return df.with_columns([
            pl.Series(name=f"stoch_k_{self.k_period}", values=stoch_k),
            pl.Series(name=f"stoch_d_{self.k_period}", values=stoch_d),
        ])
    
    test_params: ClassVar[list[dict]] = [
        {"k_period": 14, "d_period": 3, "smooth_k": 3},    
        {"k_period": 60, "d_period": 10, "smooth_k": 10},  
        {"k_period": 120, "d_period": 20, "smooth_k": 20},
    ]


@dataclass
@sf_component(name="momentum/stochrsi")
class StochRsiMom(Feature):
    """Stochastic RSI.
    
    Stochastic applied to RSI instead of price.
    
    StochRSI = (RSI - lowest_RSI) / (highest_RSI - lowest_RSI)
    
    More sensitive than standard RSI:
    - Reaches extremes more frequently
    - Better for ranging markets
    - Use with trend filter
    
    Reference: Tushar Chande & Stanley Kroll
    https://www.investopedia.com/terms/s/stochrsi.asp
    """
    
    rsi_period: int = 14
    stoch_period: int = 14
    k_period: int = 3
    d_period: int = 3
    
    requires = ["close"]
    outputs = ["stochrsi_k_{rsi_period}", "stochrsi_d_{rsi_period}"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        close = df["close"].to_numpy()
        n = len(close)
        
        diff = np.diff(close, prepend=close[0])
        diff[0] = 0
        
        gains = np.where(diff > 0, diff, 0)
        losses = np.where(diff < 0, -diff, 0)
        
        alpha = 1 / self.rsi_period
        avg_gain = np.full(n, np.nan)
        avg_loss = np.full(n, np.nan)
        
        avg_gain[self.rsi_period] = np.mean(gains[1:self.rsi_period + 1])
        avg_loss[self.rsi_period] = np.mean(losses[1:self.rsi_period + 1])
        
        for i in range(self.rsi_period + 1, n):
            avg_gain[i] = alpha * gains[i] + (1 - alpha) * avg_gain[i - 1]
            avg_loss[i] = alpha * losses[i] + (1 - alpha) * avg_loss[i - 1]
        
        rsi = 100 - (100 / (1 + avg_gain / (avg_loss + 1e-10)))
        
        raw_stoch = np.full(n, np.nan)
        start = self.rsi_period + self.stoch_period - 1
        
        for i in range(start, n):
            rsi_window = rsi[i - self.stoch_period + 1:i + 1]
            rsi_min = np.nanmin(rsi_window)
            rsi_max = np.nanmax(rsi_window)
            
            if rsi_max != rsi_min:
                raw_stoch[i] = 100 * (rsi[i] - rsi_min) / (rsi_max - rsi_min)
        
        stoch_k = np.full(n, np.nan)
        stoch_d = np.full(n, np.nan)
        
        for i in range(start + self.k_period - 1, n):
            stoch_k[i] = np.nanmean(raw_stoch[i - self.k_period + 1:i + 1])
        
        for i in range(start + self.k_period + self.d_period - 2, n):
            stoch_d[i] = np.nanmean(stoch_k[i - self.d_period + 1:i + 1])
        
        return df.with_columns([
            pl.Series(name=f"stochrsi_k_{self.rsi_period}", values=stoch_k),
            pl.Series(name=f"stochrsi_d_{self.rsi_period}", values=stoch_d),
        ])

    test_params: ClassVar[list[dict]] = [
        {"rsi_period": 14, "stoch_period": 14, "k_period": 3, "d_period": 3},
        {"rsi_period": 60, "stoch_period": 60, "k_period": 10, "d_period": 10},
        {"rsi_period": 120, "stoch_period": 120, "k_period": 20, "d_period": 20},
    ]


@dataclass
@sf_component(name="momentum/willr")
class WillrMom(Feature):
    """Williams %R.
    
    Similar to Stochastic but inverted and uses close vs high.
    
    %R = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    Bounded -100 to 0:
    - > -20: overbought
    - < -80: oversold
    
    Reference: Larry Williams
    https://www.investopedia.com/terms/w/williamsr.asp
    """
    
    period: int = 14
    
    requires = ["high", "low", "close"]
    outputs = ["willr_{period}"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        close = df["close"].to_numpy()
        n = len(close)
        
        willr = np.full(n, np.nan)
        
        for i in range(self.period - 1, n):
            hh = np.max(high[i - self.period + 1:i + 1])
            ll = np.min(low[i - self.period + 1:i + 1])
            
            if hh != ll:
                willr[i] = -100 * (hh - close[i]) / (hh - ll)
        
        return df.with_columns(
            pl.Series(name=f"willr_{self.period}", values=willr)
        )
    
    test_params: ClassVar[list[dict]] = [
        {"period": 14},
        {"period": 60},
        {"period": 240},
    ]


@dataclass
@sf_component(name="momentum/cci")
class CciMom(Feature):
    """Commodity Channel Index (CCI).
    
    Measures deviation from statistical mean.
    
    TP = (high + low + close) / 3
    CCI = (TP - SMA(TP)) / (0.015 * MAD(TP))
    
    Unbounded but typically stays within ±100:
    - > 100: overbought / strong uptrend
    - < -100: oversold / strong downtrend
    - Zero line crossovers: trend signals
    
    Reference: Donald Lambert
    https://www.investopedia.com/terms/c/commoditychannelindex.asp
    """
    
    period: int = 20
    constant: float = 0.015
    
    requires = ["high", "low", "close"]
    outputs = ["cci_{period}"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        close = df["close"].to_numpy()
        n = len(close)
        
        tp = (high + low + close) / 3

        cci = np.full(n, np.nan)
        
        for i in range(self.period - 1, n):
            window = tp[i - self.period + 1:i + 1]
            sma = np.mean(window)
            mad = np.mean(np.abs(window - sma))
            
            if mad > 0:
                cci[i] = (tp[i] - sma) / (self.constant * mad)
        
        return df.with_columns(
            pl.Series(name=f"cci_{self.period}", values=cci)
        )
    
    test_params: ClassVar[list[dict]] = [
        {"period": 20, "constant": 0.015},
        {"period": 60, "constant": 0.015},
        {"period": 240, "constant": 0.015},
    ]


@dataclass
@sf_component(name="momentum/uo")
class UoMom(Feature):
    """Ultimate Oscillator (UO).
    
    Multi-timeframe momentum combining three periods.
    
    Uses buying pressure and true range across fast/medium/slow periods
    with weighted average.
    
    BP = close - min(low, prev_close)
    TR = max(high, prev_close) - min(low, prev_close)
    
    Bounded 0-100:
    - > 70: overbought
    - < 30: oversold
    - Divergence trading common
    
    Reference: Larry Williams
    https://www.investopedia.com/terms/u/ultimateoscillator.asp
    """
    
    fast: int = 7
    medium: int = 14
    slow: int = 28
    fast_weight: float = 4.0
    medium_weight: float = 2.0
    slow_weight: float = 1.0
    
    requires = ["high", "low", "close"]
    outputs = ["uo_{fast}_{medium}_{slow}"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        close = df["close"].to_numpy()
        n = len(close)
        
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        
        bp = close - np.minimum(low, prev_close)
        tr = np.maximum(high, prev_close) - np.minimum(low, prev_close)
        
        uo = np.full(n, np.nan)
        
        for i in range(self.slow - 1, n):
            fast_bp = np.sum(bp[i - self.fast + 1:i + 1])
            fast_tr = np.sum(tr[i - self.fast + 1:i + 1])
            
            med_bp = np.sum(bp[i - self.medium + 1:i + 1])
            med_tr = np.sum(tr[i - self.medium + 1:i + 1])
            
            slow_bp = np.sum(bp[i - self.slow + 1:i + 1])
            slow_tr = np.sum(tr[i - self.slow + 1:i + 1])
            
            if fast_tr > 0 and med_tr > 0 and slow_tr > 0:
                fast_avg = fast_bp / fast_tr
                med_avg = med_bp / med_tr
                slow_avg = slow_bp / slow_tr
                
                total_weight = self.fast_weight + self.medium_weight + self.slow_weight
                weighted = (self.fast_weight * fast_avg + 
                           self.medium_weight * med_avg + 
                           self.slow_weight * slow_avg)
                
                uo[i] = 100 * weighted / total_weight
        
        return df.with_columns(
            pl.Series(name=f"uo_{self.fast}_{self.medium}_{self.slow}", values=uo)
        )
    
    test_params: ClassVar[list[dict]] = [
        {"fast": 7, "medium": 14, "slow": 28, "fast_weight": 4.0, "medium_weight": 2.0, "slow_weight": 1.0},
        {"fast": 15, "medium": 30, "slow": 60, "fast_weight": 4.0, "medium_weight": 2.0, "slow_weight": 1.0},
        {"fast": 30, "medium": 60, "slow": 120, "fast_weight": 4.0, "medium_weight": 2.0, "slow_weight": 1.0},
    ]


@dataclass
@sf_component(name="momentum/ao")
class AoMom(Feature):
    """Awesome Oscillator (AO).
    
    Difference between fast and slow SMA of median price.
    
    Median = (high + low) / 2
    AO = SMA(Median, fast) - SMA(Median, slow)
    
    Bill Williams indicator:
    - Positive: bullish momentum
    - Negative: bearish momentum
    - Saucer pattern: continuation signal
    - Twin peaks: reversal signal
    
    Reference: Bill Williams, "Trading Chaos"
    https://www.investopedia.com/terms/a/awesome-oscillator.asp
    """
    
    fast: int = 5
    slow: int = 34
    
    requires = ["high", "low"]
    outputs = ["ao_{fast}_{slow}"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        n = len(high)
        
        median = (high + low) / 2
        
        ao = np.full(n, np.nan)
        
        for i in range(self.slow - 1, n):
            fast_sma = np.mean(median[i - self.fast + 1:i + 1])
            slow_sma = np.mean(median[i - self.slow + 1:i + 1])
            ao[i] = fast_sma - slow_sma
        
        return df.with_columns(
            pl.Series(name=f"ao_{self.fast}_{self.slow}", values=ao)
        )
    test_params: ClassVar[list[dict]] = [
        {"fast": 5, "slow": 34},    
        {"fast": 15, "slow": 100},  
        {"fast": 30, "slow": 200},  
    ]
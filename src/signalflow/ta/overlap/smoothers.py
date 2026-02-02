"""Basic smoothing moving averages."""
from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl

from signalflow import sf_component
from signalflow.feature.base import Feature
from typing import ClassVar


@dataclass
@sf_component(name="smooth/sma")
class SmaSmooth(Feature):
    """Simple Moving Average.
    
    SMA = Σ(close) / n
    
    Equal weight to all observations in window.
    Most lag, but most stable.
    
    Reference: https://en.wikipedia.org/wiki/Moving_average#Simple_moving_average
    """
    
    source_col: str = "close"
    period: int = 20
    
    requires = ["{source_col}"]
    outputs = ["{source_col}_sma_{period}"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            pl.col(self.source_col)
              .rolling_mean(window_size=self.period)
              .alias(f"{self.source_col}_sma_{self.period}")
        )

    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 20},
        {"source_col": "close", "period": 60},
        {"source_col": "close", "period": 240},
    ]

@dataclass
@sf_component(name="smooth/ema")
class EmaSmooth(Feature):
    """Exponential Moving Average.
    
    EMA = α * price + (1 - α) * EMA_prev
    α = 2 / (period + 1)
    
    More weight to recent prices. Less lag than SMA.
    
    Reference: https://en.wikipedia.org/wiki/Exponential_smoothing
    """
    
    source_col: str = "close"
    period: int = 20
    
    requires = ["{source_col}"]
    outputs = ["{source_col}_ema_{period}"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            pl.col(self.source_col)
              .ewm_mean(span=self.period, adjust=False)
              .alias(f"{self.source_col}_ema_{self.period}")
        )
    
    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 20},
        {"source_col": "close", "period": 60},
        {"source_col": "close", "period": 240},
    ]


@dataclass
@sf_component(name="smooth/wma")
class WmaSmooth(Feature):
    """Weighted Moving Average.
    
    WMA = Σ(weight_i * price_i) / Σ(weight_i)
    weights = [1, 2, 3, ..., n]
    
    Linearly increasing weights. Most recent = highest weight.
    
    Reference: https://en.wikipedia.org/wiki/Moving_average#Weighted_moving_average
    """
    
    source_col: str = "close"
    period: int = 20
    
    requires = ["{source_col}"]
    outputs = ["{source_col}_wma_{period}"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        values = df[self.source_col].to_numpy()
        n = len(values)
        
        weights = np.arange(1, self.period + 1, dtype=np.float64)
        weight_sum = weights.sum()
        
        wma = np.full(n, np.nan)
        for i in range(self.period - 1, n):
            window = values[i - self.period + 1:i + 1]
            wma[i] = np.dot(window, weights) / weight_sum
        
        return df.with_columns(
            pl.Series(name=f"{self.source_col}_wma_{self.period}", values=wma)
        )
    
    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 20},
        {"source_col": "close", "period": 60},
        {"source_col": "close", "period": 240},
    ]


@dataclass
@sf_component(name="smooth/rma")
class RmaSmooth(Feature):
    """Wilder's Smoothed Moving Average (RMA).
    
    RMA = α * price + (1 - α) * RMA_prev
    α = 1 / period
    
    Used in RSI, ATR. Smoother than EMA with same period.
    Equivalent to EMA with span = 2*period - 1.
    
    Reference: https://www.investopedia.com/terms/w/wilders-smoothing.asp
    """
    
    source_col: str = "close"
    period: int = 14
    
    requires = ["{source_col}"]
    outputs = ["{source_col}_rma_{period}"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        alpha = 1.0 / self.period
        return df.with_columns(
            pl.col(self.source_col)
              .ewm_mean(alpha=alpha, adjust=False, min_periods=self.period)
              .alias(f"{self.source_col}_rma_{self.period}")
        )
    
    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 14},
        {"source_col": "close", "period": 60},
        {"source_col": "close", "period": 240},
    ]

@dataclass
@sf_component(name="smooth/dema")
class DemaSmooth(Feature):
    """Double Exponential Moving Average.
    
    DEMA = 2 * EMA(price) - EMA(EMA(price))
    
    Reduces lag by subtracting the "lag" component.
    
    Reference: https://www.investopedia.com/terms/d/double-exponential-moving-average.asp
    """
    
    source_col: str = "close"
    period: int = 20
    
    requires = ["{source_col}"]
    outputs = ["{source_col}_dema_{period}"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        col = pl.col(self.source_col)
        ema1 = col.ewm_mean(span=self.period, adjust=False)
        ema2 = ema1.ewm_mean(span=self.period, adjust=False)
        
        return df.with_columns(
            (2 * ema1 - ema2).alias(f"{self.source_col}_dema_{self.period}")
        )
    
    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 20},
        {"source_col": "close", "period": 60},
        {"source_col": "close", "period": 120},
    ]


@dataclass
@sf_component(name="smooth/tema")
class TemaSmooth(Feature):
    """Triple Exponential Moving Average.
    
    TEMA = 3*EMA - 3*EMA(EMA) + EMA(EMA(EMA))
    
    Even less lag than DEMA. May overshoot in choppy markets.
    
    Reference: https://www.investopedia.com/terms/t/triple-exponential-moving-average.asp
    """
    
    source_col: str = "close"
    period: int = 20
    
    requires = ["{source_col}"]
    outputs = ["{source_col}_tema_{period}"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        col = pl.col(self.source_col)
        ema1 = col.ewm_mean(span=self.period, adjust=False)
        ema2 = ema1.ewm_mean(span=self.period, adjust=False)
        ema3 = ema2.ewm_mean(span=self.period, adjust=False)
        
        return df.with_columns(
            (3 * ema1 - 3 * ema2 + ema3).alias(f"{self.source_col}_tema_{self.period}")
        )
    
    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 20},
        {"source_col": "close", "period": 60},
        {"source_col": "close", "period": 120},
    ]


@dataclass
@sf_component(name="smooth/hma")
class HmaSmooth(Feature):
    """Hull Moving Average.
    
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    
    Attempts to eliminate lag while maintaining smoothness.
    Very responsive to price changes.
    
    Reference: https://alanhull.com/hull-moving-average
    """
    
    source_col: str = "close"
    period: int = 20
    
    requires = ["{source_col}"]
    outputs = ["{source_col}_hma_{period}"]
    
    def _wma(self, values: np.ndarray, period: int) -> np.ndarray:
        """Compute WMA."""
        n = len(values)
        weights = np.arange(1, period + 1, dtype=np.float64)
        weight_sum = weights.sum()
        
        result = np.full(n, np.nan)
        for i in range(period - 1, n):
            window = values[i - period + 1:i + 1]
            result[i] = np.dot(window, weights) / weight_sum
        return result
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        values = df[self.source_col].to_numpy()
        
        half_period = int(self.period / 2)
        sqrt_period = int(np.sqrt(self.period))
        
        wma_half = self._wma(values, half_period)
        wma_full = self._wma(values, self.period)
        
        raw_hma = 2 * wma_half - wma_full
        hma = self._wma(raw_hma, sqrt_period)
        
        return df.with_columns(
            pl.Series(name=f"{self.source_col}_hma_{self.period}", values=hma)
        )
    
    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 20},
        {"source_col": "close", "period": 60},
        {"source_col": "close", "period": 144},  
    ]


@dataclass
@sf_component(name="smooth/trima")
class TrimaSmooth(Feature):
    """Triangular Moving Average.
    
    TRIMA = SMA(SMA(price, ceil(n/2)), floor(n/2)+1)
    
    Double-smoothed SMA with triangular weights.
    Very smooth but more lag.
    
    Reference: https://www.investopedia.com/terms/t/triangularmoving-average.asp
    """
    
    source_col: str = "close"
    period: int = 20
    
    requires = ["{source_col}"]
    outputs = ["{source_col}_trima_{period}"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        half = int(np.ceil((self.period + 1) / 2))
        col = pl.col(self.source_col)
        
        sma1 = col.rolling_mean(window_size=half)
        trima = sma1.rolling_mean(window_size=half)
        
        return df.with_columns(
            trima.alias(f"{self.source_col}_trima_{self.period}")
        )
    
    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 20},
        {"source_col": "close", "period": 60},
        {"source_col": "close", "period": 120},
    ]


@dataclass
@sf_component(name="smooth/swma")
class SwmaSmooth(Feature):
    """Symmetric Weighted Moving Average.
    
    Weights form symmetric triangle: [1,2,3,...,n,...,3,2,1]
    Middle values have highest weight.
    
    Reference: https://www.tradingview.com/pine-script-reference/#fun_swma
    """
    
    source_col: str = "close"
    period: int = 4
    
    requires = ["{source_col}"]
    outputs = ["{source_col}_swma_{period}"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        values = df[self.source_col].to_numpy()
        n = len(values)
        
        # Symmetric triangle weights
        half = (self.period + 1) // 2
        if self.period % 2 == 0:
            weights = np.concatenate([np.arange(1, half + 1), np.arange(half, 0, -1)])
        else:
            weights = np.concatenate([np.arange(1, half + 1), np.arange(half - 1, 0, -1)])
        
        weights = weights[:self.period].astype(np.float64)
        weight_sum = weights.sum()
        
        swma = np.full(n, np.nan)
        for i in range(self.period - 1, n):
            window = values[i - self.period + 1:i + 1]
            swma[i] = np.dot(window, weights) / weight_sum
        
        return df.with_columns(
            pl.Series(name=f"{self.source_col}_swma_{self.period}", values=swma)
        )
    
    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 4},   
        {"source_col": "close", "period": 10},
        {"source_col": "close", "period": 20},
    ]


@dataclass
@sf_component(name="smooth/ssf")
class SsfSmooth(Feature):
    """Ehler's Super Smoother Filter.
    
    Digital filter designed to reduce aliasing noise.
    Provides smooth output with minimal lag.
    
    poles=2: Standard 2-pole Butterworth filter
    poles=3: 3-pole filter, even smoother
    
    Reference: Ehlers, J. F. "Cybernetic Analysis for Stocks and Futures"
    """
    
    source_col: str = "close"
    period: int = 10
    poles: Literal[2, 3] = 2
    
    requires = ["{source_col}"]
    outputs = ["{source_col}_ssf_{period}"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        values = df[self.source_col].to_numpy()
        n = len(values)
        ssf = values.copy().astype(np.float64)
        
        if self.poles == 3:
            x = np.pi / self.period
            a0 = np.exp(-x)
            b0 = 2 * a0 * np.cos(np.sqrt(3) * x)
            c0 = a0 * a0
            
            c4 = c0 * c0
            c3 = -c0 * (1 + b0)
            c2 = c0 + b0
            c1 = 1 - c2 - c3 - c4
            
            for i in range(3, n):
                ssf[i] = c1 * values[i] + c2 * ssf[i-1] + c3 * ssf[i-2] + c4 * ssf[i-3]
        else:  # poles == 2
            x = np.pi * np.sqrt(2) / self.period
            a0 = np.exp(-x)
            a1 = -a0 * a0
            b1 = 2 * a0 * np.cos(x)
            c1 = 1 - a1 - b1
            
            for i in range(2, n):
                ssf[i] = c1 * values[i] + b1 * ssf[i-1] + a1 * ssf[i-2]
        
        ssf[:self.period - 1] = np.nan
        
        return df.with_columns(
            pl.Series(name=f"{self.source_col}_ssf_{self.period}", values=ssf)
        )
    
    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 10, "poles": 2},
        {"source_col": "close", "period": 30, "poles": 2},
        {"source_col": "close", "period": 60, "poles": 3},
    ]
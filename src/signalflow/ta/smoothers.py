# src/signalflow/ta/smoothers.py
"""Smoothing features for SignalFlow pipelines.

Moving averages and other smoothing algorithms.

Example:
    >>> pipeline = FeaturePipeline(features=[
    ...     RsiFeature(period=14),
    ...     SmaSmooth(source_col="rsi_14", period=5),
    ...     EmaSmooth(source_col="rsi_14", period=5),
    ... ])
"""
from dataclasses import dataclass

import polars as pl

from signalflow import sf_component
from signalflow.feature.base import Feature


@dataclass
@sf_component(name="smooth/sma")
class SmaSmooth(Feature):
    """Simple Moving Average.
    
    Arithmetic mean over rolling window.
    
    Args:
        source_col: Column to smooth.
        period: Window size. Default: 20.
        min_periods: Minimum observations required. Default: None (= period).
    
    Example:
        >>> SmaFeature(source_col="close", period=20)
        >>> # Output: close_sma_20
        
        >>> SmaFeature(source_col="rsi_14", period=5)
        >>> # Output: rsi_14_sma_5
    """
    
    source_col: str = "close"
    period: int = 20
    min_periods: int | None = None
    
    requires = ["{source_col}"]
    outputs = ["{source_col}_sma_{period}"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        min_p = self.min_periods or self.period
        out_col = f"{self.source_col}_sma_{self.period}"
        
        return df.with_columns(
            pl.col(self.source_col)
              .rolling_mean(window_size=self.period, min_periods=min_p)
              .alias(out_col)
        )


@dataclass
@sf_component(name="smooth/ema")
class EmaSmooth(Feature):
    """Exponential Moving Average.
    
    Weighted average giving more weight to recent values.
    Uses span-based decay: alpha = 2 / (span + 1)
    
    Args:
        source_col: Column to smooth.
        period: Span for EMA calculation. Default: 20.
        adjust: Divide by decaying adjustment factor. Default: True.
    
    Example:
        >>> EmaFeature(source_col="close", period=12)
        >>> # Output: close_ema_12
        
        >>> EmaFeature(source_col="rsi_14", period=5)  
        >>> # Output: rsi_14_ema_5
    """
    
    source_col: str = "close"
    period: int = 20
    adjust: bool = True
    
    requires = ["{source_col}"]
    outputs = ["{source_col}_ema_{period}"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        out_col = f"{self.source_col}_ema_{self.period}"
        
        return df.with_columns(
            pl.col(self.source_col)
              .ewm_mean(span=self.period, adjust=self.adjust)
              .alias(out_col)
        )


@dataclass
@sf_component(name="smooth/wma")
class WmaSmooth(Feature):
    """Weighted Moving Average.
    
    Linear weights: most recent value has weight N, oldest has weight 1.
    
    Args:
        source_col: Column to smooth.
        period: Window size. Default: 20.
    
    Example:
        >>> WmaFeature(source_col="close", period=10)
        >>> # Output: close_wma_10
    """
    
    source_col: str = "close"
    period: int = 20
    
    requires = ["{source_col}"]
    outputs = ["{source_col}_wma_{period}"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        out_col = f"{self.source_col}_wma_{self.period}"
        
        weights = list(range(1, self.period + 1))
        weight_sum = sum(weights)
        
        return df.with_columns(
            pl.col(self.source_col)
              .rolling_map(
                  lambda s: sum(w * v for w, v in zip(weights, s)) / weight_sum,
                  window_size=self.period,
              )
              .alias(out_col)
        )


@dataclass
@sf_component(name="smooth/dema")
class DemaSmooth(Feature):
    """Double Exponential Moving Average.
    
    DEMA = 2 * EMA(price) - EMA(EMA(price))
    Reduces lag compared to regular EMA.
    
    Args:
        source_col: Column to smooth.
        period: EMA span. Default: 20.
    
    Example:
        >>> DemaFeature(source_col="close", period=20)
        >>> # Output: close_dema_20
    """
    
    source_col: str = "close"
    period: int = 20
    
    requires = ["{source_col}"]
    outputs = ["{source_col}_dema_{period}"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        out_col = f"{self.source_col}_dema_{self.period}"
        
        ema1 = pl.col(self.source_col).ewm_mean(span=self.period)
        ema2 = ema1.ewm_mean(span=self.period)
        dema = 2 * ema1 - ema2
        
        return df.with_columns(dema.alias(out_col))


@dataclass
@sf_component(name="smooth/tema")
class TemaSmooth(Feature):
    """Triple Exponential Moving Average.
    
    TEMA = 3*EMA - 3*EMA(EMA) + EMA(EMA(EMA))
    Even less lag than DEMA.
    
    Args:
        source_col: Column to smooth.
        period: EMA span. Default: 20.
    
    Example:
        >>> TemaFeature(source_col="close", period=20)
        >>> # Output: close_tema_20
    """
    
    source_col: str = "close"
    period: int = 20
    
    requires = ["{source_col}"]
    outputs = ["{source_col}_tema_{period}"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        out_col = f"{self.source_col}_tema_{self.period}"
        
        ema1 = pl.col(self.source_col).ewm_mean(span=self.period)
        ema2 = ema1.ewm_mean(span=self.period)
        ema3 = ema2.ewm_mean(span=self.period)
        tema = 3 * ema1 - 3 * ema2 + ema3
        
        return df.with_columns(tema.alias(out_col))


@dataclass
@sf_component(name="smooth/kama")
class KamaFeature(Feature):
    """Kaufman Adaptive Moving Average.
    
    Adapts smoothing based on market efficiency (trend vs noise).
    Fast when trending, slow when ranging.
    
    Args:
        source_col: Column to smooth.
        period: Efficiency ratio period. Default: 10.
        fast_span: Fast EMA span. Default: 2.
        slow_span: Slow EMA span. Default: 30.
    
    Example:
        >>> KamaFeature(source_col="close", period=10)
        >>> # Output: close_kama_10
    """
    
    source_col: str = "close"
    period: int = 10
    fast_span: int = 2
    slow_span: int = 30
    
    requires = ["{source_col}"]
    outputs = ["{source_col}_kama_{period}"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        import numpy as np
        
        out_col = f"{self.source_col}_kama_{self.period}"
        values = df[self.source_col].to_numpy().astype(np.float64)
        n = len(values)
        
        fast_sc = 2 / (self.fast_span + 1)
        slow_sc = 2 / (self.slow_span + 1)
        
        kama = np.full(n, np.nan)
        
        if n <= self.period:
            return df.with_columns(pl.Series(name=out_col, values=kama))
        
        kama[self.period - 1] = values[self.period - 1]
        for i in range(self.period, n):
            change = abs(values[i] - values[i - self.period])
            volatility = sum(abs(values[j] - values[j - 1]) for j in range(i - self.period + 1, i + 1))
            er = change / volatility if volatility != 0 else 0
            sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
            kama[i] = kama[i - 1] + sc * (values[i] - kama[i - 1])
        
        return df.with_columns(pl.Series(name=out_col, values=kama))
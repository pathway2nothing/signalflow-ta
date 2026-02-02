# src/signalflow/ta/stat/memory.py
"""Time series memory measures - persistence, mean-reversion detection."""
from dataclasses import dataclass

import numpy as np
import polars as pl

from signalflow import sf_component
from signalflow.feature.base import Feature
from typing import ClassVar


@dataclass
@sf_component(name="stat/hurst")
class HurstStat(Feature):
    """Rolling Hurst Exponent.
    
    Measures long-term memory of time series.
    
    H < 0.5: mean-reverting (anti-persistent)
    H = 0.5: random walk (no memory)
    H > 0.5: trending (persistent)
    
    Uses Rescaled Range (R/S) method.
    
    Formula:
        R(n) = max(cumsum(x - mean)) - min(cumsum(x - mean))
        S(n) = stdev(x)
        H = log(R/S) / log(n)
    
    Reference: https://en.wikipedia.org/wiki/Hurst_exponent
    """
    
    source_col: str = "close"
    period: int = 100
    
    requires = ["{source_col}"]
    outputs = ["{source_col}_hurst_{period}"]
    
    def __post_init__(self):
        if self.period < 20:
            raise ValueError(f"period must be >= 20 for reliable Hurst estimate, got {self.period}")
    
    def _hurst_rs(self, ts: np.ndarray) -> float:
        """Compute Hurst via R/S method."""
        n = len(ts)
        if n < 20:
            return np.nan
        
        mean = np.mean(ts)
        std = np.std(ts, ddof=1)
        if std < 1e-10:
            return np.nan
        
        y = np.cumsum(ts - mean)
        r = np.max(y) - np.min(y) 
        
        rs = r / std
        
        if rs > 0:
            return np.log(rs) / np.log(n)
        return np.nan
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        values = df[self.source_col].to_numpy()
        n = len(values)
        
        hurst = np.full(n, np.nan)
        for i in range(self.period - 1, n):
            window = values[i - self.period + 1:i + 1]
            hurst[i] = self._hurst_rs(window)
        
        return df.with_columns(
            pl.Series(name=f"{self.source_col}_hurst_{self.period}", values=hurst)
        )
    
    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 100},   
        {"source_col": "close", "period": 200},   
        {"source_col": "close", "period": 500},   
    ]


@dataclass
@sf_component(name="stat/autocorr")
class AutocorrStat(Feature):
    """Rolling Autocorrelation.
    
    Correlation of series with its lagged version.
    
    High positive: trending/momentum
    High negative: mean-reverting
    Near zero: random
    
    Formula:
        ACF(lag) = corr(x[t], x[t-lag])
    
    Reference: https://en.wikipedia.org/wiki/Autocorrelation
    """
    
    source_col: str = "close"
    period: int = 30
    lag: int = 1
    
    requires = ["{source_col}"]
    outputs = ["{source_col}_acf{lag}_{period}"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        values = df[self.source_col].to_numpy()
        n = len(values)
        
        acf = np.full(n, np.nan)
        for i in range(self.period + self.lag - 1, n):
            x = values[i - self.period + 1:i + 1]
            x_lag = values[i - self.period + 1 - self.lag:i + 1 - self.lag]
            
            if len(x) == len(x_lag) and len(x) > 0:
                corr = np.corrcoef(x, x_lag)[0, 1]
                acf[i] = corr if not np.isnan(corr) else 0
        
        return df.with_columns(
            pl.Series(name=f"{self.source_col}_acf{self.lag}_{self.period}", values=acf)
        )

    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 30, "lag": 1},
        {"source_col": "close", "period": 60, "lag": 5},
        {"source_col": "close", "period": 120, "lag": 10},
    ]

@dataclass
@sf_component(name="stat/variance_ratio")
class VarianceRatioStat(Feature):
    """Rolling Variance Ratio.
    
    Tests random walk hypothesis by comparing variance at different horizons.
    
    VR = Var(k-period returns) / (k * Var(1-period returns))
    
    VR ≈ 1: random walk
    VR > 1: positive autocorrelation (momentum)
    VR < 1: negative autocorrelation (mean-reversion)
    
    Reference: Lo & MacKinlay (1988)
    https://en.wikipedia.org/wiki/Variance_ratio_test
    """
    
    source_col: str = "close"
    period: int = 50
    k: int = 5  
    
    requires = ["{source_col}"]
    outputs = ["{source_col}_vr{k}_{period}"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        values = df[self.source_col].to_numpy()
        n = len(values)
        
        log_ret = np.diff(np.log(values), prepend=np.nan)
        
        vr = np.full(n, np.nan)
        for i in range(self.period + self.k - 1, n):
            window_1 = log_ret[i - self.period + 1:i + 1]
            
            log_ret_k = np.log(values[i - self.period + 1 + self.k:i + 1]) - \
                        np.log(values[i - self.period + 1:i + 1 - self.k])
            
            var_1 = np.nanvar(window_1, ddof=1)
            var_k = np.nanvar(log_ret_k, ddof=1)
            
            if var_1 > 1e-10:
                vr[i] = var_k / (self.k * var_1)
        
        return df.with_columns(
            pl.Series(name=f"{self.source_col}_vr{self.k}_{self.period}", values=vr)
        )
    
    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 50, "k": 5},
        {"source_col": "close", "period": 100, "k": 10},
        {"source_col": "close", "period": 200, "k": 20},
    ]
"""Core momentum indicators."""
from dataclasses import dataclass

import numpy as np
import polars as pl

from signalflow import sf_component
from signalflow.feature.base import Feature
from typing import ClassVar

@dataclass
@sf_component(name="momentum/rsi")
class RsiMom(Feature):
    """Relative Strength Index (RSI).
    
    Momentum oscillator measuring speed and magnitude of price changes.
    
    RSI = 100 * avg_gain / (avg_gain + avg_loss)
    
    Where avg_gain/avg_loss use Wilder's smoothing (RMA).
    
    Interpretation:
    - RSI > 70: overbought
    - RSI < 30: oversold
    - Divergence from price: potential reversal
    - Centerline (50) crossovers: trend confirmation
    
    Reference: J. Welles Wilder, "New Concepts in Technical Trading Systems"
    https://www.investopedia.com/terms/r/rsi.asp
    """
    
    period: int = 14
    
    requires = ["close"]
    outputs = ["rsi_{period}"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        close = df["close"].to_numpy()
        n = len(close)
        
        diff = np.diff(close, prepend=close[0])
        diff[0] = 0
        
        gains = np.where(diff > 0, diff, 0)
        losses = np.where(diff < 0, -diff, 0)
        
        alpha = 1 / self.period
        
        avg_gain = np.full(n, np.nan)
        avg_loss = np.full(n, np.nan)
        
        avg_gain[self.period] = np.mean(gains[1:self.period + 1])
        avg_loss[self.period] = np.mean(losses[1:self.period + 1])
        
        for i in range(self.period + 1, n):
            avg_gain[i] = alpha * gains[i] + (1 - alpha) * avg_gain[i - 1]
            avg_loss[i] = alpha * losses[i] + (1 - alpha) * avg_loss[i - 1]
        
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        return df.with_columns(
            pl.Series(name=f"rsi_{self.period}", values=rsi)
        )
    
    test_params: ClassVar[list[dict]] = [
        {"period": 14},
        {"period": 60},
        {"period": 240},
    ]


@dataclass
@sf_component(name="momentum/roc")
class RocMom(Feature):
    """Rate of Change (ROC).
    
    Percentage change over n periods.
    
    ROC = 100 * (close - close[n]) / close[n]
    
    Unbounded oscillator:
    - Positive: price increased
    - Negative: price decreased
    - Zero crossings: momentum shift
    
    Reference: https://www.investopedia.com/terms/r/rateofchange.asp
    """
    
    period: int = 10
    
    requires = ["close"]
    outputs = ["roc_{period}"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        close = df["close"].to_numpy()
        n = len(close)
        
        roc = np.full(n, np.nan)
        
        for i in range(self.period, n):
            roc[i] = 100 * (close[i] - close[i - self.period]) / close[i - self.period]
        
        return df.with_columns(
            pl.Series(name=f"roc_{self.period}", values=roc)
        )

    test_params: ClassVar[list[dict]] = [
        {"period": 14},
        {"period": 60},
        {"period": 240},
    ]


@dataclass
@sf_component(name="momentum/mom")
class MomMom(Feature):
    """Momentum (MOM).
    
    Simple price difference over n periods.
    
    MOM = close - close[n]
    
    Unlike ROC, measures absolute change:
    - Positive: price higher
    - Negative: price lower
    - Acceleration/deceleration shows in slope
    
    Reference: https://www.investopedia.com/terms/m/momentum.asp
    """
    
    period: int = 10
    
    requires = ["close"]
    outputs = ["mom_{period}"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        close = df["close"].to_numpy()
        n = len(close)
        
        mom = np.full(n, np.nan)
        
        for i in range(self.period, n):
            mom[i] = close[i] - close[i - self.period]
        
        return df.with_columns(
            pl.Series(name=f"mom_{self.period}", values=mom)
        )
    
    test_params: ClassVar[list[dict]] = [
        {"period": 14},
        {"period": 60},
        {"period": 240},
    ]
        


@dataclass
@sf_component(name="momentum/cmo")
class CmoMom(Feature):
    """Chande Momentum Oscillator (CMO).
    
    Similar to RSI but uses sum instead of average.
    
    CMO = 100 * (sum_gains - sum_losses) / (sum_gains + sum_losses)
    
    Bounded -100 to +100:
    - CMO > 50: overbought
    - CMO < -50: oversold
    - Closer to 0: less momentum
    
    Reference: Tushar Chande
    https://www.investopedia.com/terms/c/chandemomentumoscillator.asp
    """
    
    period: int = 14
    
    requires = ["close"]
    outputs = ["cmo_{period}"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        close = df["close"].to_numpy()
        n = len(close)
        
        # Price changes
        diff = np.diff(close, prepend=close[0])
        diff[0] = 0
        
        gains = np.where(diff > 0, diff, 0)
        losses = np.where(diff < 0, -diff, 0)
        
        cmo = np.full(n, np.nan)
        
        for i in range(self.period, n):
            sum_gains = np.sum(gains[i - self.period + 1:i + 1])
            sum_losses = np.sum(losses[i - self.period + 1:i + 1])
            
            total = sum_gains + sum_losses
            if total > 0:
                cmo[i] = 100 * (sum_gains - sum_losses) / total
        
        return df.with_columns(
            pl.Series(name=f"cmo_{self.period}", values=cmo)
        )
    
    test_params: ClassVar[list[dict]] = [
        {"period": 14},
        {"period": 60},
        {"period": 240},
    ]
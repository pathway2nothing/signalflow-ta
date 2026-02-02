"""Core momentum indicators with reproducible initialization."""
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import polars as pl

from signalflow import sf_component
from signalflow.feature.base import Feature


def _rma_sma_init(values: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate RMA (Wilder's smoothing) with SMA initialization.
    
    RMA uses alpha = 1/period (unlike EMA which uses 2/(period+1)).
    Initialize with SMA for reproducibility.
    
    Args:
        values: Input array
        period: RMA period
        
    Returns:
        RMA array with first (period-1) values as NaN
    """
    n = len(values)
    alpha = 1 / period
    rma = np.full(n, np.nan)
    
    if n < period:
        return rma
    
    # Initialize with SMA of first `period` values
    rma[period - 1] = np.mean(values[:period])
    
    # Continue with Wilder's smoothing
    for i in range(period, n):
        rma[i] = alpha * values[i] + (1 - alpha) * rma[i - 1]
    
    return rma


@dataclass
@sf_component(name="momentum/rsi")
class RsiMom(Feature):
    """Relative Strength Index (RSI) with reproducible initialization.
    
    Momentum oscillator measuring speed and magnitude of price changes.
    
    RSI = 100 * avg_gain / (avg_gain + avg_loss)
    
    Where avg_gain/avg_loss use Wilder's smoothing (RMA) with SMA initialization.
    This ensures reproducibility regardless of data entry point.
    
    Interpretation:
    - RSI > 70: overbought
    - RSI < 30: oversold
    
    Reference: J. Welles Wilder, "New Concepts in Technical Trading Systems"
    """
    
    period: int = 14
    
    requires = ["close"]
    outputs = ["rsi_{period}"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        close = df["close"].to_numpy()
        n = len(close)
        
        # Price changes
        diff = np.diff(close, prepend=close[0])
        diff[0] = 0
        
        gains = np.where(diff > 0, diff, 0)
        losses = np.where(diff < 0, -diff, 0)
        
        # RMA with SMA initialization for reproducibility
        avg_gain = _rma_sma_init(gains, self.period)
        avg_loss = _rma_sma_init(losses, self.period)
        
        # RSI calculation
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
    
    Percentage change over n periods. Pure lookback, always reproducible.
    
    ROC = 100 * (close - close[n]) / close[n]
    
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
            if close[i - self.period] != 0:
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
    
    Simple price difference over n periods. Pure lookback, always reproducible.
    
    MOM = close - close[n]
    
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
    
    Uses rolling sums, not EMA - always reproducible.
    
    CMO = 100 * (sum_gains - sum_losses) / (sum_gains + sum_losses)
    
    Bounded -100 to +100.
    
    Reference: Tushar Chande
    """
    
    period: int = 14
    
    requires = ["close"]
    outputs = ["cmo_{period}"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        close = df["close"].to_numpy()
        n = len(close)
        
        diff = np.diff(close, prepend=close[0])
        diff[0] = 0
        
        gains = np.where(diff > 0, diff, 0)
        losses = np.where(diff < 0, -diff, 0)
        
        cmo = np.full(n, np.nan)
        
        for i in range(self.period - 1, n):
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
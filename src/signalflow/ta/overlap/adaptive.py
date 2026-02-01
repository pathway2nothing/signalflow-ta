"""Adaptive smoothing algorithms that adjust to market conditions."""
from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl

from signalflow import sf_component
from signalflow.feature.base import Feature


@dataclass
@sf_component(name="smooth/kama")
class KamaSmooth(Feature):
    """Kaufman's Adaptive Moving Average.
    
    Adapts smoothing based on efficiency ratio (trend vs noise).
    
    ER = |price_change| / Σ|price_changes|
    SC = (ER * (fast - slow) + slow)²
    KAMA = SC * price + (1 - SC) * KAMA_prev
    
    Trending: fast response. Ranging: slow response.
    
    Reference: Kaufman, P. "Trading Systems and Methods"
    """
    
    source_col: str = "close"
    period: int = 10
    fast: int = 2
    slow: int = 30
    
    requires = ["{source_col}"]
    outputs = ["{source_col}_kama_{period}"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        values = df[self.source_col].to_numpy()
        n = len(values)
        
        fast_sc = 2 / (self.fast + 1)
        slow_sc = 2 / (self.slow + 1)
        
        kama = np.full(n, np.nan)
        kama[self.period - 1] = values[self.period - 1]
        
        for i in range(self.period, n):
            # Efficiency Ratio
            change = abs(values[i] - values[i - self.period])
            volatility = np.sum(np.abs(np.diff(values[i - self.period:i + 1])))
            
            er = change / volatility if volatility > 0 else 0
            
            # Smoothing constant
            sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
            
            # KAMA
            kama[i] = sc * values[i] + (1 - sc) * kama[i - 1]
        
        return df.with_columns(
            pl.Series(name=f"{self.source_col}_kama_{self.period}", values=kama)
        )


@dataclass
@sf_component(name="smooth/alma")
class AlmaSmooth(Feature):
    """Arnaud Legoux Moving Average.
    
    Uses Gaussian distribution for weights.
    offset controls responsiveness (0=smooth, 1=responsive).
    sigma controls shape of the curve.
    
    Reference: https://www.prorealcode.com/prorealtime-indicators/alma-arnaud-legoux-moving-average/
    """
    
    source_col: str = "close"
    period: int = 10
    offset: float = 0.85  # 0 to 1
    sigma: float = 6.0
    
    requires = ["{source_col}"]
    outputs = ["{source_col}_alma_{period}"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        values = df[self.source_col].to_numpy()
        n = len(values)
        
        m = self.offset * (self.period - 1)
        s = self.period / self.sigma
        
        # Gaussian weights
        weights = np.array([
            np.exp(-((i - m) ** 2) / (2 * s * s))
            for i in range(self.period)
        ])
        
        alma = np.full(n, np.nan)
        for i in range(self.period - 1, n):
            window = values[i - self.period + 1:i + 1]
            alma[i] = np.dot(window, weights) / weights.sum()
        
        return df.with_columns(
            pl.Series(name=f"{self.source_col}_alma_{self.period}", values=alma)
        )


@dataclass
@sf_component(name="smooth/jma")
class JmaSmooth(Feature):
    """Jurik Moving Average.
    
    Proprietary adaptive MA with extremely low lag.
    Uses volatility bands and dynamic smoothing.
    
    phase: -100 to +100 (controls overshoot)
    
    Reference: https://c.mql5.com/forextsd/forum/164/jurik_1.pdf
    """
    
    source_col: str = "close"
    period: int = 7
    phase: float = 0  # -100 to 100
    
    requires = ["{source_col}"]
    outputs = ["{source_col}_jma_{period}"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        values = df[self.source_col].to_numpy().astype(np.float64)
        n = len(values)
        
        jma = np.zeros(n)
        volty = np.zeros(n)
        v_sum = np.zeros(n)
        
        # Initialize
        jma[0] = ma1 = uBand = lBand = values[0]
        kv = det0 = det1 = ma2 = 0.0
        
        # Parameters
        length = 0.5 * (self.period - 1)
        pr = 0.5 if self.phase < -100 else 2.5 if self.phase > 100 else 1.5 + self.phase * 0.01
        length1 = max(np.log(np.sqrt(length)) / np.log(2.0) + 2.0, 0)
        pow1 = max(length1 - 2.0, 0.5)
        length2 = length1 * np.sqrt(length)
        bet = length2 / (length2 + 1)
        beta = 0.45 * (self.period - 1) / (0.45 * (self.period - 1) + 2.0)
        
        sum_length = 10
        
        for i in range(1, n):
            price = values[i]
            
            # Price volatility
            del1 = price - uBand
            del2 = price - lBand
            volty[i] = max(abs(del1), abs(del2)) if abs(del1) != abs(del2) else 0
            
            # Relative price volatility
            start_idx = max(i - sum_length, 0)
            v_sum[i] = v_sum[i-1] + (volty[i] - volty[start_idx]) / sum_length
            
            avg_idx = max(i - 65, 0)
            avg_volty = np.mean(v_sum[avg_idx:i+1])
            d_volty = volty[i] / avg_volty if avg_volty > 0 else 0
            r_volty = max(1.0, min(length1 ** (1/pow1), d_volty))
            
            # Volatility bands
            pow2 = r_volty ** pow1
            kv = bet ** np.sqrt(pow2)
            uBand = price if del1 > 0 else price - kv * del1
            lBand = price if del2 < 0 else price - kv * del2
            
            # Dynamic factor
            power = r_volty ** pow1
            alpha = beta ** power
            
            # 3-stage smoothing
            ma1 = (1 - alpha) * price + alpha * ma1
            det0 = (price - ma1) * (1 - beta) + beta * det0
            ma2 = ma1 + pr * det0
            det1 = (ma2 - jma[i-1]) * (1 - alpha) ** 2 + alpha ** 2 * det1
            jma[i] = jma[i-1] + det1
        
        jma[:self.period - 1] = np.nan
        
        return df.with_columns(
            pl.Series(name=f"{self.source_col}_jma_{self.period}", values=jma)
        )


@dataclass
@sf_component(name="smooth/vidya")
class VidyaSmooth(Feature):
    """Variable Index Dynamic Average.
    
    Adapts based on Chande Momentum Oscillator (CMO).
    High volatility = fast, Low volatility = slow.
    
    VIDYA = α * |CMO| * price + (1 - α * |CMO|) * VIDYA_prev
    
    Reference: Chande, T. "The New Technical Trader"
    """
    
    source_col: str = "close"
    period: int = 14
    
    requires = ["{source_col}"]
    outputs = ["{source_col}_vidya_{period}"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        values = df[self.source_col].to_numpy()
        n = len(values)
        
        alpha = 2 / (self.period + 1)
        
        # Momentum
        mom = np.diff(values, prepend=np.nan)
        pos = np.where(mom > 0, mom, 0)
        neg = np.where(mom < 0, -mom, 0)
        
        vidya = np.full(n, np.nan)
        vidya[self.period] = values[self.period]
        
        for i in range(self.period + 1, n):
            # CMO
            pos_sum = np.sum(pos[i - self.period + 1:i + 1])
            neg_sum = np.sum(neg[i - self.period + 1:i + 1])
            
            cmo = (pos_sum - neg_sum) / (pos_sum + neg_sum) if (pos_sum + neg_sum) > 0 else 0
            abs_cmo = abs(cmo)
            
            vidya[i] = alpha * abs_cmo * values[i] + (1 - alpha * abs_cmo) * vidya[i - 1]
        
        return df.with_columns(
            pl.Series(name=f"{self.source_col}_vidya_{self.period}", values=vidya)
        )


@dataclass
@sf_component(name="smooth/t3")
class T3Smooth(Feature):
    """Tillson T3 Moving Average.
    
    Smoother and more responsive than TEMA.
    Uses volume factor 'a' to control smoothing.
    
    T3 = c1*e6 + c2*e5 + c3*e4 + c4*e3
    where e1..e6 are cascaded EMAs
    
    Reference: Tillson, T. "Technical Analysis of Stocks & Commodities"
    """
    
    source_col: str = "close"
    period: int = 10
    vfactor: float = 0.7  # volume factor, 0 < a < 1
    
    requires = ["{source_col}"]
    outputs = ["{source_col}_t3_{period}"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        a = self.vfactor
        c1 = -a ** 3
        c2 = 3 * a ** 2 + 3 * a ** 3
        c3 = -6 * a ** 2 - 3 * a - 3 * a ** 3
        c4 = 1 + 3 * a + a ** 3 + 3 * a ** 2
        
        col = pl.col(self.source_col)
        e1 = col.ewm_mean(span=self.period, adjust=False)
        e2 = e1.ewm_mean(span=self.period, adjust=False)
        e3 = e2.ewm_mean(span=self.period, adjust=False)
        e4 = e3.ewm_mean(span=self.period, adjust=False)
        e5 = e4.ewm_mean(span=self.period, adjust=False)
        e6 = e5.ewm_mean(span=self.period, adjust=False)
        
        t3 = c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3
        
        return df.with_columns(
            t3.alias(f"{self.source_col}_t3_{self.period}")
        )


@dataclass
@sf_component(name="smooth/zlma")
class ZlmaSmooth(Feature):
    """Zero Lag Moving Average.
    
    Reduces lag by adjusting price before smoothing.
    
    adjusted_price = 2 * price - price.shift(lag)
    ZLMA = EMA(adjusted_price)
    lag = (period - 1) / 2
    
    Reference: Ehlers & Way, "Zero Lag (Well, Almost)"
    """
    
    source_col: str = "close"
    period: int = 20
    ma_type: Literal["ema", "sma"] = "ema"
    
    requires = ["{source_col}"]
    outputs = ["{source_col}_zlma_{period}"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        lag = int((self.period - 1) / 2)
        col = pl.col(self.source_col)
        
        # Adjust price to remove lag
        adjusted = 2 * col - col.shift(lag)
        
        if self.ma_type == "sma":
            zlma = adjusted.rolling_mean(window_size=self.period)
        else:
            zlma = adjusted.ewm_mean(span=self.period, adjust=False)
        
        return df.with_columns(
            zlma.alias(f"{self.source_col}_zlma_{self.period}")
        )


@dataclass
@sf_component(name="smooth/mcginley")
class McGinleySmooth(Feature):
    """McGinley Dynamic Indicator.
    
    Self-adjusting MA that tracks price more closely.
    Speeds up in downtrends, slows in uptrends.
    
    MD = MD_prev + (price - MD_prev) / (k * n * (price/MD_prev)^4)
    
    Reference: McGinley, J. "Journal of Technical Analysis"
    """
    
    source_col: str = "close"
    period: int = 10
    k: float = 0.6  # constant, typically 0.6
    
    requires = ["{source_col}"]
    outputs = ["{source_col}_mcg_{period}"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        values = df[self.source_col].to_numpy()
        n = len(values)
        
        md = np.full(n, np.nan)
        md[0] = values[0]
        
        for i in range(1, n):
            if md[i-1] != 0:
                ratio = values[i] / md[i-1]
                denom = self.k * self.period * (ratio ** 4)
                md[i] = md[i-1] + (values[i] - md[i-1]) / denom
            else:
                md[i] = values[i]
        
        return df.with_columns(
            pl.Series(name=f"{self.source_col}_mcg_{self.period}", values=md)
        )


@dataclass
@sf_component(name="smooth/frama")
class FramaSmooth(Feature):
    """Fractal Adaptive Moving Average.
    
    Uses fractal dimension to adapt smoothing.
    Higher dimension (choppy) = slower. Lower (trending) = faster.
    
    D = (log(N1 + N2) - log(N3)) / log(2)
    α = exp(-4.6 * (D - 1))
    
    Reference: Ehlers, J. "FRAMA"
    """
    
    source_col: str = "close"
    period: int = 16  # must be even
    
    requires = ["{source_col}"]
    outputs = ["{source_col}_frama_{period}"]
    
    def __post_init__(self):
        if self.period % 2 != 0:
            raise ValueError("FRAMA period must be even")
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        values = df[self.source_col].to_numpy()
        n = len(values)
        half = self.period // 2
        
        frama = np.full(n, np.nan)
        frama[self.period - 1] = values[self.period - 1]
        
        for i in range(self.period, n):
            # First half range
            n1 = (np.max(values[i - self.period + 1:i - half + 1]) - 
                  np.min(values[i - self.period + 1:i - half + 1])) / half
            
            # Second half range
            n2 = (np.max(values[i - half + 1:i + 1]) - 
                  np.min(values[i - half + 1:i + 1])) / half
            
            # Full range
            n3 = (np.max(values[i - self.period + 1:i + 1]) - 
                  np.min(values[i - self.period + 1:i + 1])) / self.period
            
            # Fractal dimension
            if n1 + n2 > 0 and n3 > 0:
                d = (np.log(n1 + n2) - np.log(n3)) / np.log(2)
            else:
                d = 1
            
            # Alpha
            alpha = np.exp(-4.6 * (d - 1))
            alpha = np.clip(alpha, 0.01, 1)
            
            frama[i] = alpha * values[i] + (1 - alpha) * frama[i - 1]
        
        return df.with_columns(
            pl.Series(name=f"{self.source_col}_frama_{self.period}", values=frama)
        )
"""Adaptive smoothing algorithms that adjust to market conditions."""
from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl

from signalflow import sf_component
from signalflow.feature.base import Feature
from typing import ClassVar


def _ema_sma_init(values: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate EMA with SMA initialization for reproducibility.

    Args:
        values: Input array (may contain NaN)
        period: EMA period (also used for SMA initialization)

    Returns:
        EMA array with first (period-1) values as NaN
    """
    n = len(values)
    alpha = 2 / (period + 1)
    ema = np.full(n, np.nan)

    if n < period:
        return ema

    # Find first valid (non-NaN) index
    valid_idx = np.where(~np.isnan(values))[0]
    if len(valid_idx) == 0:
        return ema

    first_valid = valid_idx[0]

    # Need at least period values after first valid
    if first_valid + period > n:
        return ema

    # Initialize with SMA of first `period` valid values
    init_idx = first_valid + period - 1
    ema[init_idx] = np.mean(values[first_valid:first_valid + period])

    # Continue with standard EMA
    for i in range(init_idx + 1, n):
        if not np.isnan(values[i]):
            ema[i] = alpha * values[i] + (1 - alpha) * ema[i - 1]

    return ema


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

        if n >= self.period:
            # Initialize with SMA for reproducibility
            kama[self.period - 1] = np.mean(values[:self.period])
        
        for i in range(self.period, n):
            change = abs(values[i] - values[i - self.period])
            volatility = np.sum(np.abs(np.diff(values[i - self.period:i + 1])))
            
            er = change / volatility if volatility > 0 else 0
            
            sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
            
            kama[i] = sc * values[i] + (1 - sc) * kama[i - 1]
        
        return df.with_columns(
            pl.Series(name=f"{self.source_col}_kama_{self.period}", values=kama)
        )

    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 10, "fast": 2, "slow": 30},
        {"source_col": "close", "period": 60, "fast": 5, "slow": 120},
        {"source_col": "close", "period": 120, "fast": 10, "slow": 240},
    ]


    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable, reproducible output."""
        return self.period * 5

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
    offset: float = 0.85  
    sigma: float = 6.0
    
    requires = ["{source_col}"]
    outputs = ["{source_col}_alma_{period}"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        values = df[self.source_col].to_numpy()
        n = len(values)
        
        m = self.offset * (self.period - 1)
        s = self.period / self.sigma
        
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
    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 10, "offset": 0.85, "sigma": 6.0},
        {"source_col": "close", "period": 60, "offset": 0.85, "sigma": 6.0},
        {"source_col": "close", "period": 120, "offset": 0.85, "sigma": 6.0},
    ]


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
        
        jma = np.full(n, np.nan)
        volty = np.zeros(n)
        v_sum = np.zeros(n)

        # Initialize with SMA for reproducibility
        warmup = min(self.period, n)
        if warmup > 0:
            init_val = np.mean(values[:warmup])
        else:
            init_val = 0.0

        jma[warmup - 1] = ma1 = uBand = lBand = init_val
        kv = det0 = det1 = ma2 = 0.0
        
        length = 0.5 * (self.period - 1)
        pr = 0.5 if self.phase < -100 else 2.5 if self.phase > 100 else 1.5 + self.phase * 0.01
        length1 = max(np.log(np.sqrt(length)) / np.log(2.0) + 2.0, 0)
        pow1 = max(length1 - 2.0, 0.5)
        length2 = length1 * np.sqrt(length)
        bet = length2 / (length2 + 1)
        beta = 0.45 * (self.period - 1) / (0.45 * (self.period - 1) + 2.0)
        
        sum_length = 10

        for i in range(warmup, n):
            price = values[i]
            
            del1 = price - uBand
            del2 = price - lBand
            volty[i] = max(abs(del1), abs(del2)) if abs(del1) != abs(del2) else 0
            
            start_idx = max(i - sum_length, 0)
            v_sum[i] = v_sum[i-1] + (volty[i] - volty[start_idx]) / sum_length
            
            avg_idx = max(i - 65, 0)
            avg_volty = np.mean(v_sum[avg_idx:i+1])
            d_volty = volty[i] / avg_volty if avg_volty > 0 else 0
            r_volty = max(1.0, min(length1 ** (1/pow1), d_volty))
            
            pow2 = r_volty ** pow1
            kv = bet ** np.sqrt(pow2)
            uBand = price if del1 > 0 else price - kv * del1
            lBand = price if del2 < 0 else price - kv * del2
            
            power = r_volty ** pow1
            alpha = beta ** power
            
            ma1 = (1 - alpha) * price + alpha * ma1
            det0 = (price - ma1) * (1 - beta) + beta * det0
            ma2 = ma1 + pr * det0
            det1 = (ma2 - jma[i-1]) * (1 - alpha) ** 2 + alpha ** 2 * det1
            jma[i] = jma[i-1] + det1
        
        jma[:self.period - 1] = np.nan
        
        return df.with_columns(
            pl.Series(name=f"{self.source_col}_jma_{self.period}", values=jma)
        )
    
    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 7, "phase": 0},
        {"source_col": "close", "period": 30, "phase": 0},
        {"source_col": "close", "period": 60, "phase": 50},
    ]


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
        
        mom = np.diff(values, prepend=np.nan)
        pos = np.where(mom > 0, mom, 0)
        neg = np.where(mom < 0, -mom, 0)
        
        vidya = np.full(n, np.nan)

        if n > self.period:
            # Initialize with SMA for reproducibility
            vidya[self.period] = np.mean(values[:self.period + 1])
        
        for i in range(self.period + 1, n):
            pos_sum = np.sum(pos[i - self.period + 1:i + 1])
            neg_sum = np.sum(neg[i - self.period + 1:i + 1])
            
            cmo = (pos_sum - neg_sum) / (pos_sum + neg_sum) if (pos_sum + neg_sum) > 0 else 0
            abs_cmo = abs(cmo)
            
            vidya[i] = alpha * abs_cmo * values[i] + (1 - alpha * abs_cmo) * vidya[i - 1]
        
        return df.with_columns(
            pl.Series(name=f"{self.source_col}_vidya_{self.period}", values=vidya)
        )
    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 14},
        {"source_col": "close", "period": 60},
        {"source_col": "close", "period": 120},
    ]   


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
    vfactor: float = 0.7 
    
    requires = ["{source_col}"]
    outputs = ["{source_col}_t3_{period}"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        values = df[self.source_col].to_numpy()

        a = self.vfactor
        c1 = -a ** 3
        c2 = 3 * a ** 2 + 3 * a ** 3
        c3 = -6 * a ** 2 - 3 * a - 3 * a ** 3
        c4 = 1 + 3 * a + a ** 3 + 3 * a ** 2

        # Calculate 6 cascaded EMAs with SMA initialization
        e1 = _ema_sma_init(values, self.period)
        e2 = _ema_sma_init(e1, self.period)
        e3 = _ema_sma_init(e2, self.period)
        e4 = _ema_sma_init(e3, self.period)
        e5 = _ema_sma_init(e4, self.period)
        e6 = _ema_sma_init(e5, self.period)

        t3 = c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3

        return df.with_columns(
            pl.Series(name=f"{self.source_col}_t3_{self.period}", values=t3)
        )
    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 10, "vfactor": 0.7},
        {"source_col": "close", "period": 30, "vfactor": 0.7},
        {"source_col": "close", "period": 60, "vfactor": 0.8},
    ]

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
        
        adjusted = 2 * col - col.shift(lag)
        
        if self.ma_type == "sma":
            zlma = adjusted.rolling_mean(window_size=self.period)
        else:
            zlma = adjusted.ewm_mean(span=self.period, adjust=False)
        
        return df.with_columns(
            zlma.alias(f"{self.source_col}_zlma_{self.period}")
        )

    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 20, "ma_type": "ema"},
        {"source_col": "close", "period": 60, "ma_type": "ema"},
        {"source_col": "close", "period": 120, "ma_type": "sma"},
    ]

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
    k: float = 0.6  
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

    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 10, "k": 0.6},
        {"source_col": "close", "period": 60, "k": 0.6},
        {"source_col": "close", "period": 120, "k": 0.6},
    ]

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
    period: int = 16 
    
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
            n1 = (np.max(values[i - self.period + 1:i - half + 1]) - 
                  np.min(values[i - self.period + 1:i - half + 1])) / half
            n2 = (np.max(values[i - half + 1:i + 1]) - 
                  np.min(values[i - half + 1:i + 1])) / half
            n3 = (np.max(values[i - self.period + 1:i + 1]) - 
                  np.min(values[i - self.period + 1:i + 1])) / self.period
            if n1 + n2 > 0 and n3 > 0:
                d = (np.log(n1 + n2) - np.log(n3)) / np.log(2)
            else:
                d = 1
            alpha = np.exp(-4.6 * (d - 1))
            alpha = np.clip(alpha, 0.01, 1)
            
            frama[i] = alpha * values[i] + (1 - alpha) * frama[i - 1]
        
        return df.with_columns(
            pl.Series(name=f"{self.source_col}_frama_{self.period}", values=frama)
        )
    
    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 16},   
        {"source_col": "close", "period": 60},   
        {"source_col": "close", "period": 120},  
    ]   

    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable, reproducible output."""
        return self.period * 5


    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable, reproducible output."""
        return self.period * 5


    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable, reproducible output."""
        return self.period * 5


    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable, reproducible output."""
        return self.period * 6


    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable, reproducible output."""
        return self.period * 5


    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable, reproducible output."""
        return self.period * 5


    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable, reproducible output."""
        return self.period * 5

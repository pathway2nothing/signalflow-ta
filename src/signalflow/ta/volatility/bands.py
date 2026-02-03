"""Channel and envelope volatility indicators."""
from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl

from signalflow.core import sf_component
from signalflow.feature.base import Feature
from typing import ClassVar


@dataclass
@sf_component(name="volatility/bollinger")
class BollingerVol(Feature):
    """Bollinger Bands.
    
    Volatility bands around a moving average.
    
    Middle = MA(close, period)
    Upper = Middle + std_dev * StdDev(close, period)
    Lower = Middle - std_dev * StdDev(close, period)
    
    Outputs:
    - bb_upper: upper band
    - bb_middle: middle band (MA)
    - bb_lower: lower band
    - bb_width: (upper - lower) / middle * 100 (bandwidth)
    - bb_pct: (close - lower) / (upper - lower) (%B)
    
    Interpretation:
    - Price at upper band: overbought / strong uptrend
    - Price at lower band: oversold / strong downtrend
    - Band squeeze: low volatility, potential breakout
    - Band expansion: high volatility
    
    Reference: John Bollinger
    https://www.investopedia.com/terms/b/bollingerbands.asp
    """
    
    period: int = 20
    std_dev: float = 2.0
    ma_type: Literal["sma", "ema"] = "sma"
    
    requires = ["close"]
    outputs = ["bb_upper_{period}", "bb_middle_{period}", "bb_lower_{period}", 
               "bb_width_{period}", "bb_pct_{period}"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        close = df["close"].to_numpy()
        n = len(close)
        
        middle = np.full(n, np.nan)
        std = np.full(n, np.nan)
        
        if self.ma_type == "ema":
            alpha = 2 / (self.period + 1)

            if n >= self.period:
                # Initialize with SMA for reproducibility
                middle[self.period - 1] = np.mean(close[:self.period])

            for i in range(self.period, n):
                middle[i] = alpha * close[i] + (1 - alpha) * middle[i - 1]

            # Standard deviation still uses rolling window
            for i in range(self.period - 1, n):
                std[i] = np.std(close[i - self.period + 1:i + 1], ddof=0)
        else:  # SMA
            for i in range(self.period - 1, n):
                window = close[i - self.period + 1:i + 1]
                middle[i] = np.mean(window)
                std[i] = np.std(window, ddof=0)
        
        upper = middle + self.std_dev * std
        lower = middle - self.std_dev * std
        
        width = 100 * (upper - lower) / middle
        
        pct = (close - lower) / (upper - lower + 1e-10)
        
        return df.with_columns([
            pl.Series(name=f"bb_upper_{self.period}", values=upper),
            pl.Series(name=f"bb_middle_{self.period}", values=middle),
            pl.Series(name=f"bb_lower_{self.period}", values=lower),
            pl.Series(name=f"bb_width_{self.period}", values=width),
            pl.Series(name=f"bb_pct_{self.period}", values=pct),
        ])
    
    test_params: ClassVar[list[dict]] = [
        {"period": 20, "std_dev": 2.0, "ma_type": "sma"},
        {"period": 30, "std_dev": 2.0, "ma_type": "sma"},
        {"period": 60, "std_dev": 2.5, "ma_type": "ema"},
    ]


@dataclass
@sf_component(name="volatility/keltner")
class KeltnerVol(Feature):
    """Keltner Channels.
    
    Volatility envelope based on ATR.
    
    Basis = EMA(close, period)
    Upper = Basis + multiplier * ATR(period)
    Lower = Basis - multiplier * ATR(period)
    
    Outputs:
    - kc_upper: upper channel
    - kc_basis: middle line (EMA)
    - kc_lower: lower channel
    
    Compared to Bollinger:
    - Uses ATR instead of standard deviation
    - Generally smoother bands
    - Less reactive to sudden price spikes
    
    Reference: Chester Keltner (original), Linda Raschke (modern version)
    https://school.stockcharts.com/doku.php?id=technical_indicators:keltner_channels
    """
    
    period: int = 20
    multiplier: float = 2.0
    ma_type: Literal["ema", "sma"] = "ema"
    use_true_range: bool = True
    
    requires = ["high", "low", "close"]
    outputs = ["kc_upper_{period}", "kc_basis_{period}", "kc_lower_{period}"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        close = df["close"].to_numpy()
        n = len(close)
        
        if self.use_true_range:
            prev_close = np.roll(close, 1)
            prev_close[0] = close[0]
            range_vals = np.maximum(
                high - low,
                np.maximum(
                    np.abs(high - prev_close),
                    np.abs(low - prev_close)
                )
            )
            range_vals[0] = high[0] - low[0]
        else:
            range_vals = high - low
        
        basis = np.full(n, np.nan)
        atr = np.full(n, np.nan)
        
        if self.ma_type == "ema":
            alpha = 2 / (self.period + 1)
            basis[0] = close[0]
            atr[0] = range_vals[0]
            for i in range(1, n):
                basis[i] = alpha * close[i] + (1 - alpha) * basis[i - 1]
                atr[i] = alpha * range_vals[i] + (1 - alpha) * atr[i - 1]
        else:  
            for i in range(self.period - 1, n):
                basis[i] = np.mean(close[i - self.period + 1:i + 1])
                atr[i] = np.mean(range_vals[i - self.period + 1:i + 1])
        
        upper = basis + self.multiplier * atr
        lower = basis - self.multiplier * atr
        
        return df.with_columns([
            pl.Series(name=f"kc_upper_{self.period}", values=upper),
            pl.Series(name=f"kc_basis_{self.period}", values=basis),
            pl.Series(name=f"kc_lower_{self.period}", values=lower),
        ])
        
    test_params: ClassVar[list[dict]] = [
        {"period": 20, "multiplier": 2.0, "ma_type": "ema", "use_true_range": True},
        {"period": 30, "multiplier": 1.5, "ma_type": "ema", "use_true_range": True},
        {"period": 60, "multiplier": 2.0, "ma_type": "sma", "use_true_range": False},
    ]

@dataclass
@sf_component(name="volatility/donchian")
class DonchianVol(Feature):
    """Donchian Channels.
    
    Price channel based on highest high and lowest low.
    
    Upper = Highest High over period
    Lower = Lowest Low over period
    Middle = (Upper + Lower) / 2
    
    Outputs:
    - dc_upper: upper channel (resistance)
    - dc_middle: midline
    - dc_lower: lower channel (support)
    
    Classic breakout indicator:
    - Price breaks above upper: bullish breakout
    - Price breaks below lower: bearish breakout
    - Width shows volatility
    
    Reference: Richard Donchian (Turtle Trading)
    https://school.stockcharts.com/doku.php?id=technical_indicators:donchian_channels
    """
    
    period: int = 20
    
    requires = ["high", "low"]
    outputs = ["dc_upper_{period}", "dc_middle_{period}", "dc_lower_{period}"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        n = len(high)
        
        upper = np.full(n, np.nan)
        lower = np.full(n, np.nan)
        
        for i in range(self.period - 1, n):
            upper[i] = np.max(high[i - self.period + 1:i + 1])
            lower[i] = np.min(low[i - self.period + 1:i + 1])
        
        middle = (upper + lower) / 2
        
        return df.with_columns([
            pl.Series(name=f"dc_upper_{self.period}", values=upper),
            pl.Series(name=f"dc_middle_{self.period}", values=middle),
            pl.Series(name=f"dc_lower_{self.period}", values=lower),
        ])
    
    test_params: ClassVar[list[dict]] = [
        {"period": 20},
        {"period": 55},  
        {"period": 120},
    ]


@dataclass
@sf_component(name="volatility/accbands")
class AccBandsVol(Feature):
    """Acceleration Bands.
    
    Envelope bands that widen with volatility.
    
    HL_Ratio = c * (High - Low) / (High + Low)
    Upper = MA(High * (1 + HL_Ratio), period)
    Lower = MA(Low * (1 - HL_Ratio), period)
    Middle = MA(Close, period)
    
    Outputs:
    - accb_upper: upper band
    - accb_middle: middle band
    - accb_lower: lower band
    
    Breakout indicator:
    - Close above upper band: strong uptrend
    - Close below lower band: strong downtrend
    - Inside bands: consolidation
    
    Reference: Price Headley
    https://www.tradingtechnologies.com/help/x-study/technical-indicator-definitions/acceleration-bands-abands/
    """
    
    period: int = 20
    factor: float = 4.0
    ma_type: Literal["sma", "ema"] = "sma"
    
    requires = ["high", "low", "close"]
    outputs = ["accb_upper_{period}", "accb_middle_{period}", "accb_lower_{period}"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        close = df["close"].to_numpy()
        n = len(close)
        
        hl_range = high - low
        hl_sum = high + low
        hl_ratio = self.factor * hl_range / (hl_sum + 1e-10)
        
        upper_raw = high * (1 + hl_ratio)
        lower_raw = low * (1 - hl_ratio)
        
        upper = np.full(n, np.nan)
        middle = np.full(n, np.nan)
        lower = np.full(n, np.nan)
        
        if self.ma_type == "ema":
            alpha = 2 / (self.period + 1)
            upper[0] = upper_raw[0]
            middle[0] = close[0]
            lower[0] = lower_raw[0]
            for i in range(1, n):
                upper[i] = alpha * upper_raw[i] + (1 - alpha) * upper[i - 1]
                middle[i] = alpha * close[i] + (1 - alpha) * middle[i - 1]
                lower[i] = alpha * lower_raw[i] + (1 - alpha) * lower[i - 1]
        else:  
            for i in range(self.period - 1, n):
                upper[i] = np.mean(upper_raw[i - self.period + 1:i + 1])
                middle[i] = np.mean(close[i - self.period + 1:i + 1])
                lower[i] = np.mean(lower_raw[i - self.period + 1:i + 1])
        
        return df.with_columns([
            pl.Series(name=f"accb_upper_{self.period}", values=upper),
            pl.Series(name=f"accb_middle_{self.period}", values=middle),
            pl.Series(name=f"accb_lower_{self.period}", values=lower),
        ])
    
    test_params: ClassVar[list[dict]] = [
        {"period": 20, "factor": 4.0, "ma_type": "sma"},
        {"period": 30, "factor": 3.0, "ma_type": "sma"},
        {"period": 60, "factor": 4.0, "ma_type": "ema"},
    ]

"""Cumulative volume-price indicators."""
from dataclasses import dataclass

import numpy as np
import polars as pl

from signalflow import sf_component
from signalflow.feature.base import Feature


@dataclass
@sf_component(name="volume/obv")
class ObvVol(Feature):
    """On Balance Volume (OBV).
    
    Cumulative volume based on price direction.
    
    If close > prev_close: OBV += volume
    If close < prev_close: OBV -= volume
    If close == prev_close: OBV unchanged
    
    Interpretation:
    - Rising OBV with rising price: uptrend confirmed
    - Falling OBV with falling price: downtrend confirmed
    - OBV divergence from price: potential reversal
    
    Reference: Joseph Granville, "New Key to Stock Market Profits"
    https://www.investopedia.com/terms/o/onbalancevolume.asp
    """
    
    requires = ["close", "volume"]
    outputs = ["obv"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        close = df["close"].to_numpy()
        volume = df["volume"].to_numpy()
        n = len(close)
        
        # Direction: +1 if up, -1 if down, 0 if unchanged
        direction = np.sign(np.diff(close, prepend=close[0]))
        direction[0] = 0
        
        # OBV = cumsum of signed volume
        obv = np.cumsum(direction * volume)
        
        return df.with_columns(
            pl.Series(name="obv", values=obv)
        )


@dataclass
@sf_component(name="volume/ad")
class AdVol(Feature):
    """Accumulation/Distribution Line (A/D).
    
    Cumulative indicator using close location within range.
    
    CLV = ((Close - Low) - (High - Close)) / (High - Low)
    AD = cumsum(CLV * Volume)
    
    CLV ranges from -1 to +1:
    - Close at high: CLV = +1 (accumulation)
    - Close at low: CLV = -1 (distribution)
    - Close at midpoint: CLV = 0
    
    Interpretation:
    - Rising A/D: accumulation (buying pressure)
    - Falling A/D: distribution (selling pressure)
    - Divergence from price: potential reversal
    
    Reference: Marc Chaikin
    https://school.stockcharts.com/doku.php?id=technical_indicators:accumulation_distribution_line
    """
    
    requires = ["high", "low", "close", "volume"]
    outputs = ["ad"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        close = df["close"].to_numpy()
        volume = df["volume"].to_numpy()
        
        # Close Location Value (Money Flow Multiplier)
        hl_range = high - low
        clv = np.where(
            hl_range > 0,
            ((close - low) - (high - close)) / hl_range,
            0
        )
        
        # Money Flow Volume
        mfv = clv * volume
        
        # Cumulative A/D
        ad = np.cumsum(mfv)
        
        return df.with_columns(
            pl.Series(name="ad", values=ad)
        )


@dataclass
@sf_component(name="volume/pvt")
class PvtVol(Feature):
    """Price-Volume Trend (PVT).
    
    Cumulative indicator using percentage price change.
    
    PVT = cumsum(ROC(close, 1) * volume)
    
    Similar to OBV but weights by magnitude of price change,
    not just direction.
    
    Interpretation:
    - Rising PVT: volume flowing into the asset
    - Falling PVT: volume flowing out
    - Divergence signals potential reversals
    
    Reference: https://www.investopedia.com/terms/p/pvtrend.asp
    """
    
    requires = ["close", "volume"]
    outputs = ["pvt"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        close = df["close"].to_numpy()
        volume = df["volume"].to_numpy()
        
        # Rate of change (percentage)
        roc = np.diff(close, prepend=close[0]) / np.roll(close, 1)
        roc[0] = 0
        
        # Price-Volume
        pv = roc * volume
        
        # Cumulative PVT
        pvt = np.cumsum(pv)
        
        return df.with_columns(
            pl.Series(name="pvt", values=pvt)
        )


@dataclass
@sf_component(name="volume/nvi")
class NviVol(Feature):
    """Negative Volume Index (NVI).
    
    Tracks price changes on days with lower volume.
    
    If volume < prev_volume:
        NVI = NVI_prev + ROC(close)
    Else:
        NVI = NVI_prev (unchanged)
    
    Theory: "Smart money" trades on low-volume days,
    while "uninformed" traders dominate high-volume days.
    
    Interpretation:
    - Rising NVI: smart money accumulating
    - Compare to 255-day EMA for trend signals
    
    Reference: Paul Dysart, Norman Fosback
    https://school.stockcharts.com/doku.php?id=technical_indicators:negative_volume_inde
    """
    
    initial: float = 1000.0
    
    requires = ["close", "volume"]
    outputs = ["nvi"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        close = df["close"].to_numpy()
        volume = df["volume"].to_numpy()
        n = len(close)
        
        # ROC
        roc = np.diff(close, prepend=close[0]) / np.roll(close, 1)
        roc[0] = 0
        
        # Volume direction
        vol_down = volume < np.roll(volume, 1)
        vol_down[0] = False
        
        # NVI changes only on down-volume days
        nvi_change = np.where(vol_down, roc * 100, 0)
        
        # Cumulative with initial value
        nvi = np.full(n, self.initial)
        for i in range(1, n):
            nvi[i] = nvi[i - 1] + nvi_change[i]
        
        return df.with_columns(
            pl.Series(name="nvi", values=nvi)
        )


@dataclass
@sf_component(name="volume/pvi")
class PviVol(Feature):
    """Positive Volume Index (PVI).
    
    Tracks price changes on days with higher volume.
    
    If volume > prev_volume:
        PVI = PVI_prev + ROC(close)
    Else:
        PVI = PVI_prev (unchanged)
    
    Opposite of NVI - tracks "uninformed" crowd behavior.
    
    Interpretation:
    - Use with NVI for complete picture
    - PVI rising: crowd is bullish
    - PVI falling: crowd is bearish
    
    Reference: Paul Dysart, Norman Fosback
    https://www.investopedia.com/terms/p/pvi.asp
    """
    
    initial: float = 1000.0
    
    requires = ["close", "volume"]
    outputs = ["pvi"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        close = df["close"].to_numpy()
        volume = df["volume"].to_numpy()
        n = len(close)
        
        # ROC
        roc = np.diff(close, prepend=close[0]) / np.roll(close, 1)
        roc[0] = 0
        
        # Volume direction
        vol_up = volume > np.roll(volume, 1)
        vol_up[0] = False
        
        # PVI changes only on up-volume days
        pvi_change = np.where(vol_up, roc * 100, 0)
        
        # Cumulative with initial value
        pvi = np.full(n, self.initial)
        for i in range(1, n):
            pvi[i] = pvi[i - 1] + pvi_change[i]
        
        return df.with_columns(
            pl.Series(name="pvi", values=pvi)
        )
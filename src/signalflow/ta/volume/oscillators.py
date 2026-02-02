"""Volume-based oscillators."""
from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl

from signalflow import sf_component
from signalflow.feature.base import Feature


@dataclass
@sf_component(name="volume/mfi")
class MfiVolume(Feature):
    """Money Flow Index (MFI).
    
    Volume-weighted RSI using typical price.
    
    TP = (High + Low + Close) / 3
    Raw Money Flow = TP * Volume
    MFI = 100 * Pos_MF / (Pos_MF + Neg_MF)
    
    Bounded 0-100, similar interpretation to RSI:
    - MFI > 80: overbought
    - MFI < 20: oversold
    - Divergence signals potential reversals
    
    Reference: Gene Quong & Avrum Soudack
    https://www.investopedia.com/terms/m/mfi.asp
    """
    
    period: int = 14
    
    requires = ["high", "low", "close", "volume"]
    outputs = ["mfi_{period}"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        close = df["close"].to_numpy()
        volume = df["volume"].to_numpy()
        n = len(close)
        
        # Typical price
        tp = (high + low + close) / 3
        
        # Raw money flow
        rmf = tp * volume
        
        # Direction
        tp_diff = np.diff(tp, prepend=tp[0])
        
        # Positive and negative money flow
        pos_mf = np.where(tp_diff > 0, rmf, 0)
        neg_mf = np.where(tp_diff < 0, rmf, 0)
        
        # Rolling sums
        mfi = np.full(n, np.nan)
        
        for i in range(self.period - 1, n):
            pos_sum = np.sum(pos_mf[i - self.period + 1:i + 1])
            neg_sum = np.sum(neg_mf[i - self.period + 1:i + 1])
            total = pos_sum + neg_sum
            
            if total > 0:
                mfi[i] = 100 * pos_sum / total
        
        return df.with_columns(
            pl.Series(name=f"mfi_{self.period}", values=mfi)
        )


@dataclass
@sf_component(name="volume/cmf")
class CmfVolume(Feature):
    """Chaikin Money Flow (CMF).
    
    Measures buying/selling pressure over a period.
    
    CLV = ((Close - Low) - (High - Close)) / (High - Low)
    CMF = SUM(CLV * Volume, period) / SUM(Volume, period)
    
    Bounded approximately -1 to +1:
    - CMF > 0: buying pressure (accumulation)
    - CMF < 0: selling pressure (distribution)
    - CMF > 0.25: strong buying
    - CMF < -0.25: strong selling
    
    Reference: Marc Chaikin
    https://school.stockcharts.com/doku.php?id=technical_indicators:chaikin_money_flow_cmf
    """
    
    period: int = 20
    
    requires = ["high", "low", "close", "volume"]
    outputs = ["cmf_{period}"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        close = df["close"].to_numpy()
        volume = df["volume"].to_numpy()
        n = len(close)
        
        # Close Location Value
        hl_range = high - low
        clv = np.where(
            hl_range > 0,
            ((close - low) - (high - close)) / hl_range,
            0
        )
        
        # Money Flow Volume
        mfv = clv * volume
        
        # CMF = rolling sum of MFV / rolling sum of volume
        cmf = np.full(n, np.nan)
        
        for i in range(self.period - 1, n):
            mfv_sum = np.sum(mfv[i - self.period + 1:i + 1])
            vol_sum = np.sum(volume[i - self.period + 1:i + 1])
            
            if vol_sum > 0:
                cmf[i] = mfv_sum / vol_sum
        
        return df.with_columns(
            pl.Series(name=f"cmf_{self.period}", values=cmf)
        )


@dataclass
@sf_component(name="volume/efi")
class EfiVolume(Feature):
    """Elder's Force Index (EFI).
    
    Measures the force behind price movements.
    
    Force = (Close - Prev_Close) * Volume
    EFI = EMA(Force, period)
    
    Unbounded oscillator:
    - Positive: buying force
    - Negative: selling force
    - Magnitude indicates strength
    - Zero crossings signal changes
    
    Reference: Alexander Elder, "Trading for a Living"
    https://www.investopedia.com/terms/f/force-index.asp
    """
    
    period: int = 13
    
    requires = ["close", "volume"]
    outputs = ["efi_{period}"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        close = df["close"].to_numpy()
        volume = df["volume"].to_numpy()
        n = len(close)
        
        # Force = price change * volume
        force = np.diff(close, prepend=close[0]) * volume
        force[0] = 0
        
        # EMA of force
        alpha = 2 / (self.period + 1)
        efi = np.full(n, np.nan)
        efi[0] = force[0]
        
        for i in range(1, n):
            efi[i] = alpha * force[i] + (1 - alpha) * efi[i - 1]
        
        return df.with_columns(
            pl.Series(name=f"efi_{self.period}", values=efi)
        )


@dataclass
@sf_component(name="volume/eom")
class EomVolume(Feature):
    """Ease of Movement (EMV/EOM).
    
    Relates price change to volume.
    
    Distance = ((High + Low)/2) - ((Prev_High + Prev_Low)/2)
    Box_Ratio = (Volume / divisor) / (High - Low)
    EMV = Distance / Box_Ratio
    EOM = SMA(EMV, period)
    
    Interpretation:
    - Positive: price moving up on low volume (easy movement)
    - Negative: price moving down on low volume
    - Near zero: consolidation or high volume moves
    
    Reference: Richard Arms
    https://school.stockcharts.com/doku.php?id=technical_indicators:ease_of_movement_emv
    """
    
    period: int = 14
    divisor: float = 100_000_000
    
    requires = ["high", "low", "volume"]
    outputs = ["eom_{period}"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        volume = df["volume"].to_numpy()
        n = len(high)
        
        # HL2
        hl2 = (high + low) / 2
        prev_hl2 = np.roll(hl2, 1)
        prev_hl2[0] = hl2[0]
        
        # Distance moved
        distance = hl2 - prev_hl2
        
        # Box ratio
        hl_range = high - low
        box_ratio = (volume / self.divisor) / (hl_range + 1e-10)
        
        # EMV
        emv = distance / (box_ratio + 1e-10)
        emv[0] = 0
        
        # SMA
        eom = np.full(n, np.nan)
        for i in range(self.period - 1, n):
            eom[i] = np.mean(emv[i - self.period + 1:i + 1])
        
        return df.with_columns(
            pl.Series(name=f"eom_{self.period}", values=eom)
        )


@dataclass
@sf_component(name="volume/kvo")
class KvoVolume(Feature):
    """Klinger Volume Oscillator (KVO).
    
    Measures long-term money flow with trend detection.
    
    Trend = sign(HLC3 - Prev_HLC3)
    dm = High - Low
    cm = cumulative dm when trend unchanged, else resets
    VF = Volume * |2*(dm/cm) - 1| * Trend * 100
    KVO = EMA(VF, fast) - EMA(VF, slow)
    Signal = EMA(KVO, signal)
    
    Outputs:
    - kvo: oscillator line
    - kvo_signal: signal line
    
    Interpretation:
    - KVO above signal: bullish
    - KVO below signal: bearish
    - Zero crossings: trend confirmation
    
    Reference: Stephen Klinger
    https://www.investopedia.com/terms/k/klingeroscillator.asp
    """
    
    fast: int = 34
    slow: int = 55
    signal: int = 13
    
    requires = ["high", "low", "close", "volume"]
    outputs = ["kvo_{fast}_{slow}", "kvo_signal_{signal}"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        close = df["close"].to_numpy()
        volume = df["volume"].to_numpy()
        n = len(close)
        
        # HLC3
        hlc3 = (high + low + close) / 3
        
        # Trend direction
        hlc3_diff = np.diff(hlc3, prepend=hlc3[0])
        trend = np.sign(hlc3_diff)
        trend[0] = 1
        
        # Signed volume (simplified Klinger)
        sv = volume * trend
        
        # EMA fast and slow
        alpha_fast = 2 / (self.fast + 1)
        alpha_slow = 2 / (self.slow + 1)
        alpha_sig = 2 / (self.signal + 1)
        
        ema_fast = np.full(n, np.nan)
        ema_slow = np.full(n, np.nan)
        
        ema_fast[0] = sv[0]
        ema_slow[0] = sv[0]
        
        for i in range(1, n):
            ema_fast[i] = alpha_fast * sv[i] + (1 - alpha_fast) * ema_fast[i - 1]
            ema_slow[i] = alpha_slow * sv[i] + (1 - alpha_slow) * ema_slow[i - 1]
        
        # KVO
        kvo = ema_fast - ema_slow
        
        # Signal line
        kvo_signal = np.full(n, np.nan)
        kvo_signal[0] = kvo[0]
        
        for i in range(1, n):
            if not np.isnan(kvo[i]) and not np.isnan(kvo_signal[i - 1]):
                kvo_signal[i] = alpha_sig * kvo[i] + (1 - alpha_sig) * kvo_signal[i - 1]
        
        return df.with_columns([
            pl.Series(name=f"kvo_{self.fast}_{self.slow}", values=kvo),
            pl.Series(name=f"kvo_signal_{self.signal}", values=kvo_signal),
        ])


@dataclass
@sf_component(name="volume/vwap")
class VwapVolume(Feature):
    """Volume Weighted Average Price (VWAP).
    
    Average price weighted by volume.
    
    VWAP = cumsum(TP * Volume) / cumsum(Volume)
    
    Note: This is cumulative VWAP (session-based).
    For intraday, reset should occur at session start.
    
    Interpretation:
    - Price above VWAP: bullish
    - Price below VWAP: bearish
    - VWAP acts as support/resistance
    
    Reference: https://www.investopedia.com/terms/v/vwap.asp
    """
    
    requires = ["high", "low", "close", "volume"]
    outputs = ["vwap"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        close = df["close"].to_numpy()
        volume = df["volume"].to_numpy()
        
        # Typical price
        tp = (high + low + close) / 3
        
        # Cumulative VWAP
        cum_tp_vol = np.cumsum(tp * volume)
        cum_vol = np.cumsum(volume)
        
        vwap = cum_tp_vol / (cum_vol + 1e-10)
        
        return df.with_columns(
            pl.Series(name="vwap", values=vwap)
        )


@dataclass
@sf_component(name="volume/vwap_bands")
class VwapBandsVolume(Feature):
    """VWAP with Standard Deviation Bands.
    
    VWAP with rolling standard deviation bands.
    
    Outputs:
    - vwap: volume weighted average price
    - vwap_upper: VWAP + std_dev * rolling_std
    - vwap_lower: VWAP - std_dev * rolling_std
    
    Useful for:
    - Mean reversion strategies
    - Identifying extended moves
    - Dynamic support/resistance
    """
    
    period: int = 20
    std_dev: float = 2.0
    
    requires = ["high", "low", "close", "volume"]
    outputs = ["vwap", "vwap_upper_{period}", "vwap_lower_{period}"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        close = df["close"].to_numpy()
        volume = df["volume"].to_numpy()
        n = len(close)
        
        # Typical price
        tp = (high + low + close) / 3
        
        # Cumulative VWAP
        cum_tp_vol = np.cumsum(tp * volume)
        cum_vol = np.cumsum(volume)
        vwap = cum_tp_vol / (cum_vol + 1e-10)
        
        # Rolling standard deviation of typical price
        upper = np.full(n, np.nan)
        lower = np.full(n, np.nan)
        
        for i in range(self.period - 1, n):
            window = tp[i - self.period + 1:i + 1]
            std = np.std(window, ddof=1)
            upper[i] = vwap[i] + self.std_dev * std
            lower[i] = vwap[i] - self.std_dev * std
        
        return df.with_columns([
            pl.Series(name="vwap", values=vwap),
            pl.Series(name=f"vwap_upper_{self.period}", values=upper),
            pl.Series(name=f"vwap_lower_{self.period}", values=lower),
        ])
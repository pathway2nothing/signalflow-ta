"""Trend detection systems - identify trend presence and direction."""
from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl

from signalflow.core import sf_component
from signalflow.feature.base import Feature
from typing import ClassVar


@dataclass
@sf_component(name="trend/ichimoku")
class IchimokuTrend(Feature):
    """Ichimoku Kinko Hyo (Ichimoku Cloud).

    Comprehensive Japanese trend system.

    Outputs:
    - tenkan_sen: conversion line (fast, default 9)
    - kijun_sen: base line (slow, default 26)
    - senkou_a: leading span A (cloud edge)
    - senkou_b: leading span B (cloud edge)

    Unbounded. Uses z-score in normalized mode.

    Interpretation:
    - Price above cloud: bullish
    - Price below cloud: bearish
    - Tenkan > Kijun: bullish momentum
    - Cloud color (A vs B): trend strength

    Note: Senkou spans shifted forward by kijun periods.
    Chikou span omitted (requires future lookahead).

    Reference: Goichi Hosoda, 1968
    """

    tenkan: int = 9
    kijun: int = 26
    senkou: int = 52
    normalized: bool = False
    norm_period: int | None = None

    requires = ["high", "low"]
    outputs = ["tenkan_sen", "kijun_sen", "senkou_a", "senkou_b"]
    
    def _midprice(self, high: np.ndarray, low: np.ndarray, period: int) -> np.ndarray:
        """Rolling (highest + lowest) / 2."""
        n = len(high)
        result = np.full(n, np.nan)
        for i in range(period - 1, n):
            hh = np.max(high[i - period + 1:i + 1])
            ll = np.min(low[i - period + 1:i + 1])
            result[i] = (hh + ll) / 2
        return result
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        n = len(high)
        
        tenkan_sen = self._midprice(high, low, self.tenkan)
        kijun_sen = self._midprice(high, low, self.kijun)
        
        span_a = (tenkan_sen + kijun_sen) / 2
        senkou_a = np.full(n, np.nan)
        senkou_a[self.kijun:] = span_a[:-self.kijun]
    
        span_b = self._midprice(high, low, self.senkou)
        senkou_b = np.full(n, np.nan)
        senkou_b[self.kijun:] = span_b[:-self.kijun]

        # Normalization: z-score for unbounded oscillator
        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window
            norm_window = self.norm_period or get_norm_window(self.senkou)
            tenkan_sen = normalize_zscore(tenkan_sen, window=norm_window)
            kijun_sen = normalize_zscore(kijun_sen, window=norm_window)
            senkou_a = normalize_zscore(senkou_a, window=norm_window)
            senkou_b = normalize_zscore(senkou_b, window=norm_window)

        col_tenkan, col_kijun, col_senkou_a, col_senkou_b = self._get_output_names()
        return df.with_columns([
            pl.Series(name=col_tenkan, values=tenkan_sen),
            pl.Series(name=col_kijun, values=kijun_sen),
            pl.Series(name=col_senkou_a, values=senkou_a),
            pl.Series(name=col_senkou_b, values=senkou_b),
        ])

    def _get_output_names(self) -> tuple[str, str, str, str]:
        """Generate output column names with normalization suffix."""
        suffix = "_norm" if self.normalized else ""
        return (
            f"tenkan_sen{suffix}",
            f"kijun_sen{suffix}",
            f"senkou_a{suffix}",
            f"senkou_b{suffix}"
        )

    test_params: ClassVar[list[dict]] = [
        {"tenkan": 9, "kijun": 26, "senkou": 52},
        {"tenkan": 9, "kijun": 26, "senkou": 52, "normalized": True},
        {"tenkan": 20, "kijun": 60, "senkou": 120},
        {"tenkan": 45, "kijun": 130, "senkou": 260},
    ]

    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable, reproducible output."""
        base_warmup = self.senkou * 5
        if self.normalized:
            from signalflow.ta._normalization import get_norm_window
            norm_window = self.norm_period or get_norm_window(self.senkou)
            return base_warmup + norm_window
        return base_warmup

@dataclass
@sf_component(name="trend/dpo")
class DpoTrend(Feature):
    """Detrended Price Oscillator.

    Removes trend to identify cycles.

    DPO = Close - SMA(Close, n).shift(n/2 + 1)

    Unbounded. Uses z-score in normalized mode.

    Oscillates around zero:
    - Above zero: price above historical average
    - Below zero: price below historical average

    Use for cycle analysis, not trend following.

    Reference: https://school.stockcharts.com/doku.php?id=technical_indicators:detrended_price_osci
    """

    period: int = 20
    normalized: bool = False
    norm_period: int | None = None

    requires = ["close"]
    outputs = ["dpo_{period}"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        close = df["close"].to_numpy()
        n = len(close)
        
        sma = np.full(n, np.nan)
        for i in range(self.period - 1, n):
            sma[i] = np.mean(close[i - self.period + 1:i + 1])
        
        shift = int(self.period / 2) + 1
        dpo = np.full(n, np.nan)
        
        for i in range(shift + self.period - 1, n):
            dpo[i] = close[i] - sma[i - shift]

        # Normalization: z-score for unbounded oscillator
        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window
            norm_window = self.norm_period or get_norm_window(self.period)
            dpo = normalize_zscore(dpo, window=norm_window)

        col_name = self._get_output_name()
        return df.with_columns(
            pl.Series(name=col_name, values=dpo)
        )

    def _get_output_name(self) -> str:
        """Generate output column name with normalization suffix."""
        suffix = "_norm" if self.normalized else ""
        return f"dpo_{self.period}{suffix}"

    test_params: ClassVar[list[dict]] = [
        {"period": 20},
        {"period": 20, "normalized": True},
        {"period": 60},
        {"period": 120},
    ]

    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable, reproducible output."""
        base_warmup = self.period * 5
        if self.normalized:
            from signalflow.ta._normalization import get_norm_window
            norm_window = self.norm_period or get_norm_window(self.period)
            return base_warmup + norm_window
        return base_warmup


@dataclass
@sf_component(name="trend/qstick")
class QstickTrend(Feature):
    """Q Stick.

    Quantifies candlestick patterns using moving average of (close - open).

    QSTICK = MA(Close - Open, n)

    Unbounded. Uses z-score in normalized mode.

    Interpretation:
    - Positive: bullish candles dominating
    - Negative: bearish candles dominating
    - Rising: increasing bullish pressure
    - Falling: increasing bearish pressure

    Reference: Tushar Chande
    """

    period: int = 10
    ma_type: Literal["sma", "ema"] = "sma"
    normalized: bool = False
    norm_period: int | None = None

    requires = ["open", "close"]
    outputs = ["qstick_{period}"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        open_price = df["open"].to_numpy()
        close = df["close"].to_numpy()
        n = len(close)
        
        diff = close - open_price
        qstick = np.full(n, np.nan)
        
        if self.ma_type == "ema":
            alpha = 2 / (self.period + 1)
            qstick[self.period - 1] = np.mean(diff[:self.period])
            for i in range(self.period, n):
                qstick[i] = alpha * diff[i] + (1 - alpha) * qstick[i - 1]
        else:
            for i in range(self.period - 1, n):
                qstick[i] = np.mean(diff[i - self.period + 1:i + 1])

        # Normalization: z-score for unbounded oscillator
        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window
            norm_window = self.norm_period or get_norm_window(self.period)
            qstick = normalize_zscore(qstick, window=norm_window)

        col_name = self._get_output_name()
        return df.with_columns(
            pl.Series(name=col_name, values=qstick)
        )

    def _get_output_name(self) -> str:
        """Generate output column name with normalization suffix."""
        suffix = "_norm" if self.normalized else ""
        return f"qstick_{self.period}{suffix}"

    test_params: ClassVar[list[dict]] = [
        {"period": 10, "ma_type": "sma"},
        {"period": 10, "ma_type": "sma", "normalized": True},
        {"period": 30, "ma_type": "sma"},
        {"period": 60, "ma_type": "ema"},
    ]

    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable, reproducible output."""
        base_warmup = self.period * 5
        if self.normalized:
            from signalflow.ta._normalization import get_norm_window
            norm_window = self.norm_period or get_norm_window(self.period)
            return base_warmup + norm_window
        return base_warmup

@dataclass
@sf_component(name="trend/ttm")
class TtmTrend(Feature):
    """TTM Trend (John Carter).

    Simple trend identification based on close vs average HL2.

    avg_price = SMA(HL2, n)
    trend = +1 if close > avg_price else -1

    Unbounded. Uses z-score in normalized mode.

    Two consecutive opposite bars signal trend change.

    Reference: John Carter, "Mastering the Trade"
    """

    period: int = 6
    normalized: bool = False
    norm_period: int | None = None

    requires = ["high", "low", "close"]
    outputs = ["ttm_trend_{period}"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        close = df["close"].to_numpy()
        n = len(close)
        
        hl2 = (high + low) / 2        
        avg_hl2 = np.full(n, np.nan)
        for i in range(self.period - 1, n):
            avg_hl2[i] = np.mean(hl2[i - self.period + 1:i + 1])
        
        trend = np.where(close > avg_hl2, 1, -1).astype(float)
        trend[:self.period - 1] = np.nan

        # Normalization: z-score for unbounded oscillator
        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window
            norm_window = self.norm_period or get_norm_window(self.period)
            trend = normalize_zscore(trend, window=norm_window)

        col_name = self._get_output_name()
        return df.with_columns(
            pl.Series(name=col_name, values=trend)
        )

    def _get_output_name(self) -> str:
        """Generate output column name with normalization suffix."""
        suffix = "_norm" if self.normalized else ""
        return f"ttm_trend_{self.period}{suffix}"

    test_params: ClassVar[list[dict]] = [
        {"period": 6},
        {"period": 6, "normalized": True},
        {"period": 15},
        {"period": 30},
    ]

    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable, reproducible output."""
        base_warmup = self.period * 5
        if self.normalized:
            from signalflow.ta._normalization import get_norm_window
            norm_window = self.norm_period or get_norm_window(self.period)
            return base_warmup + norm_window
        return base_warmup


@dataclass
@sf_component(name="trend/atr_trailing")
class AtrTrailingTrend(Feature):
    """ATR Trailing Stop.

    Simple ATR-based trailing stop.

    Long stop: highest close - multiplier * ATR
    Short stop: lowest close + multiplier * ATR

    Outputs:
    - atr_trail_long: long trailing stop
    - atr_trail_short: short trailing stop
    - atr_trail_dir: direction based on close vs stops

    Unbounded. Uses z-score in normalized mode.
    """

    period: int = 14
    multiplier: float = 3.0
    normalized: bool = False
    norm_period: int | None = None

    requires = ["high", "low", "close"]
    outputs = ["atr_trail_long_{period}", "atr_trail_short_{period}", "atr_trail_dir_{period}"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        close = df["close"].to_numpy()
        n = len(close)
        
        tr = np.maximum(
            high - low,
            np.maximum(
                np.abs(high - np.roll(close, 1)),
                np.abs(low - np.roll(close, 1))
            )
        )
        tr[0] = high[0] - low[0]
        
        atr = np.full(n, np.nan)
        atr[self.period - 1] = np.mean(tr[:self.period])
        alpha = 1 / self.period
        for i in range(self.period, n):
            atr[i] = alpha * tr[i] + (1 - alpha) * atr[i - 1]
        
        trail_long = np.full(n, np.nan)
        trail_short = np.full(n, np.nan)
        direction = np.zeros(n)
        
        for i in range(self.period - 1, n):
            hc = np.max(close[max(0, i - self.period + 1):i + 1])
            lc = np.min(close[max(0, i - self.period + 1):i + 1])
            
            trail_long[i] = hc - self.multiplier * atr[i]
            trail_short[i] = lc + self.multiplier * atr[i]
            
            if close[i] > trail_short[i]:
                direction[i] = 1
            elif close[i] < trail_long[i]:
                direction[i] = -1
            elif i > 0:
                direction[i] = direction[i - 1]

        # Normalization: z-score for unbounded oscillator
        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window
            norm_window = self.norm_period or get_norm_window(self.period)
            trail_long = normalize_zscore(trail_long, window=norm_window)
            trail_short = normalize_zscore(trail_short, window=norm_window)
            direction = normalize_zscore(direction, window=norm_window)

        col_long, col_short, col_dir = self._get_output_names()
        return df.with_columns([
            pl.Series(name=col_long, values=trail_long),
            pl.Series(name=col_short, values=trail_short),
            pl.Series(name=col_dir, values=direction),
        ])

    def _get_output_names(self) -> tuple[str, str, str]:
        """Generate output column names with normalization suffix."""
        suffix = "_norm" if self.normalized else ""
        return (
            f"atr_trail_long_{self.period}{suffix}",
            f"atr_trail_short_{self.period}{suffix}",
            f"atr_trail_dir_{self.period}{suffix}"
        )

    test_params: ClassVar[list[dict]] = [
        {"period": 14, "multiplier": 3.0},
        {"period": 14, "multiplier": 3.0, "normalized": True},
        {"period": 30, "multiplier": 2.5},
        {"period": 60, "multiplier": 3.0},
    ]

    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable, reproducible output."""
        base_warmup = self.period * 5
        if self.normalized:
            from signalflow.ta._normalization import get_norm_window
            norm_window = self.norm_period or get_norm_window(self.period)
            return base_warmup + norm_window
        return base_warmup

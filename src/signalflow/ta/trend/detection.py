"""Trend detection systems - identify trend presence and direction."""

from dataclasses import dataclass
from typing import ClassVar, Literal

import numpy as np
import polars as pl

from signalflow.core import feature
from signalflow.feature.base import Feature
from signalflow.ta._numba_kernels import (
    ema_sma_init as _ema_sma_init,
)
from signalflow.ta._numba_kernels import (
    ichimoku_midprice as _ichimoku_midprice,
)
from signalflow.ta._numba_kernels import (
    rma_sma_init as _rma_sma_init,
)
from signalflow.ta._numba_kernels import (
    rolling_max as _rolling_max,
)
from signalflow.ta._numba_kernels import (
    rolling_min as _rolling_min,
)
from signalflow.ta._numba_kernels import (
    sma_nb as _sma_nb,
)


@dataclass
@feature("trend/ichimoku")
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

    requires: ClassVar[list[str]] = ["high", "low"]
    outputs: ClassVar[list[str]] = ["tenkan_sen", "kijun_sen", "senkou_a", "senkou_b"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        n = len(high)

        tenkan_sen = _ichimoku_midprice(high, low, self.tenkan)
        kijun_sen = _ichimoku_midprice(high, low, self.kijun)

        span_a = (tenkan_sen + kijun_sen) / 2
        senkou_a = np.full(n, np.nan)
        senkou_a[self.kijun :] = span_a[: -self.kijun]

        span_b = _ichimoku_midprice(high, low, self.senkou)
        senkou_b = np.full(n, np.nan)
        senkou_b[self.kijun :] = span_b[: -self.kijun]

        # Normalization: z-score for unbounded oscillator
        if self.normalized:
            from signalflow.ta._normalization import get_norm_window, normalize_zscore

            norm_window = self.norm_period or get_norm_window(self.senkou)
            tenkan_sen = normalize_zscore(tenkan_sen, window=norm_window)
            kijun_sen = normalize_zscore(kijun_sen, window=norm_window)
            senkou_a = normalize_zscore(senkou_a, window=norm_window)
            senkou_b = normalize_zscore(senkou_b, window=norm_window)

        col_tenkan, col_kijun, col_senkou_a, col_senkou_b = self._get_output_names()
        return df.with_columns(
            [
                pl.Series(name=col_tenkan, values=tenkan_sen),
                pl.Series(name=col_kijun, values=kijun_sen),
                pl.Series(name=col_senkou_a, values=senkou_a),
                pl.Series(name=col_senkou_b, values=senkou_b),
            ]
        )

    def _get_output_names(self) -> tuple[str, str, str, str]:
        """Generate output column names with normalization suffix."""
        suffix = "_norm" if self.normalized else ""
        return (
            f"tenkan_sen{suffix}",
            f"kijun_sen{suffix}",
            f"senkou_a{suffix}",
            f"senkou_b{suffix}",
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
@feature("trend/dpo")
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

    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["dpo_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        close = df["close"].to_numpy().astype(np.float64)
        n = len(close)

        sma = _sma_nb(close, self.period)

        shift = int(self.period / 2) + 1
        dpo = np.full(n, np.nan)

        for i in range(shift + self.period - 1, n):
            dpo[i] = close[i] - sma[i - shift]

        # Normalization: z-score for unbounded oscillator
        if self.normalized:
            from signalflow.ta._normalization import get_norm_window, normalize_zscore

            norm_window = self.norm_period or get_norm_window(self.period)
            dpo = normalize_zscore(dpo, window=norm_window)

        col_name = self._get_output_name()
        return df.with_columns(pl.Series(name=col_name, values=dpo))

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
@feature("trend/qstick")
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

    requires: ClassVar[list[str]] = ["open", "close"]
    outputs: ClassVar[list[str]] = ["qstick_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        open_price = df["open"].to_numpy().astype(np.float64)
        close = df["close"].to_numpy().astype(np.float64)

        diff = close - open_price

        qstick = _ema_sma_init(diff, self.period) if self.ma_type == "ema" else _sma_nb(diff, self.period)

        # Normalization: z-score for unbounded oscillator
        if self.normalized:
            from signalflow.ta._normalization import get_norm_window, normalize_zscore

            norm_window = self.norm_period or get_norm_window(self.period)
            qstick = normalize_zscore(qstick, window=norm_window)

        col_name = self._get_output_name()
        return df.with_columns(pl.Series(name=col_name, values=qstick))

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
@feature("trend/ttm")
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

    requires: ClassVar[list[str]] = ["high", "low", "close"]
    outputs: ClassVar[list[str]] = ["ttm_trend_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        close = df["close"].to_numpy()
        _n = len(close)

        hl2 = (high + low).astype(np.float64) / 2
        avg_hl2 = _sma_nb(hl2, self.period)

        trend = np.where(close > avg_hl2, 1, -1).astype(float)
        trend[: self.period - 1] = np.nan

        # Normalization: z-score for unbounded oscillator
        if self.normalized:
            from signalflow.ta._normalization import get_norm_window, normalize_zscore

            norm_window = self.norm_period or get_norm_window(self.period)
            trend = normalize_zscore(trend, window=norm_window)

        col_name = self._get_output_name()
        return df.with_columns(pl.Series(name=col_name, values=trend))

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
@feature("trend/atr_trailing")
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

    requires: ClassVar[list[str]] = ["high", "low", "close"]
    outputs: ClassVar[list[str]] = [
        "atr_trail_long_{period}",
        "atr_trail_short_{period}",
        "atr_trail_dir_{period}",
    ]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        close = df["close"].to_numpy().astype(np.float64)
        n = len(close)

        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
        tr[0] = high[0] - low[0]

        atr = _rma_sma_init(tr, self.period)
        hc = _rolling_max(close, self.period)
        lc = _rolling_min(close, self.period)

        trail_long = hc - self.multiplier * atr
        trail_short = lc + self.multiplier * atr

        # Direction requires stateful logic
        direction = np.zeros(n)
        for i in range(self.period - 1, n):
            if close[i] > trail_short[i]:
                direction[i] = 1
            elif close[i] < trail_long[i]:
                direction[i] = -1
            elif i > 0:
                direction[i] = direction[i - 1]

        # Normalization: z-score for unbounded oscillator
        if self.normalized:
            from signalflow.ta._normalization import get_norm_window, normalize_zscore

            norm_window = self.norm_period or get_norm_window(self.period)
            trail_long = normalize_zscore(trail_long, window=norm_window)
            trail_short = normalize_zscore(trail_short, window=norm_window)
            direction = normalize_zscore(direction, window=norm_window)

        col_long, col_short, col_dir = self._get_output_names()
        return df.with_columns(
            [
                pl.Series(name=col_long, values=trail_long),
                pl.Series(name=col_short, values=trail_short),
                pl.Series(name=col_dir, values=direction),
            ]
        )

    def _get_output_names(self) -> tuple[str, str, str]:
        """Generate output column names with normalization suffix."""
        suffix = "_norm" if self.normalized else ""
        return (
            f"atr_trail_long_{self.period}{suffix}",
            f"atr_trail_short_{self.period}{suffix}",
            f"atr_trail_dir_{self.period}{suffix}",
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

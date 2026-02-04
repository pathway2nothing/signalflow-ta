"""Trailing stop indicators - trend-following with defined exits."""
from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl

from signalflow.core import sf_component
from signalflow.feature.base import Feature
from typing import ClassVar


@dataclass
@sf_component(name="trend/psar")
class PsarTrend(Feature):
    """Parabolic SAR (Stop and Reverse).

    Trailing stop that accelerates with trend.

    SAR = SAR_prev + AF * (EP - SAR_prev)
    AF: acceleration factor (starts at af, increases by af_step to af_max)
    EP: extreme point (highest high / lowest low)

    Outputs:
    - psar: SAR value
    - psar_dir: direction (+1 bullish, -1 bearish)

    Unbounded. Uses z-score in normalized mode.

    Reference: Welles Wilder, "New Concepts in Technical Trading Systems"
    """

    af: float = 0.02
    af_step: float = 0.02
    af_max: float = 0.2
    normalized: bool = False
    norm_period: int | None = None

    requires = ["high", "low", "close"]
    outputs = ["psar", "psar_dir"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        close = df["close"].to_numpy()
        n = len(close)
        
        psar = np.full(n, np.nan)
        direction = np.ones(n)
        
        psar[0] = close[0]
        af = self.af
        
        if n > 1 and close[1] > close[0]:
            direction[0] = 1
            ep = high[0]
            psar[0] = low[0]
        else:
            direction[0] = -1
            ep = low[0]
            psar[0] = high[0]
        
        for i in range(1, n):
            prev_psar = psar[i - 1]
            prev_dir = direction[i - 1]
            
            if prev_dir == 1:
                psar[i] = prev_psar + af * (ep - prev_psar)
                psar[i] = min(psar[i], low[i - 1])
                if i > 1:
                    psar[i] = min(psar[i], low[i - 2])
                
                if low[i] < psar[i]:  
                    direction[i] = -1
                    psar[i] = ep
                    ep = low[i]
                    af = self.af
                else:
                    direction[i] = 1
                    if high[i] > ep:
                        ep = high[i]
                        af = min(af + self.af_step, self.af_max)
            else: 
                psar[i] = prev_psar + af * (ep - prev_psar)
                psar[i] = max(psar[i], high[i - 1])
                if i > 1:
                    psar[i] = max(psar[i], high[i - 2])
                
                if high[i] > psar[i]: 
                    direction[i] = 1
                    psar[i] = ep
                    ep = high[i]
                    af = self.af
                else:
                    direction[i] = -1
                    if low[i] < ep:
                        ep = low[i]
                        af = min(af + self.af_step, self.af_max)

        # Normalization: z-score for unbounded oscillator
        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window
            # Use a reasonable default period (20) since PSAR doesn't have an explicit period
            norm_window = self.norm_period or get_norm_window(20)
            psar = normalize_zscore(psar, window=norm_window)
            direction = normalize_zscore(direction, window=norm_window)

        col_psar, col_dir = self._get_output_names()
        return df.with_columns([
            pl.Series(name=col_psar, values=psar),
            pl.Series(name=col_dir, values=direction),
        ])

    def _get_output_names(self) -> tuple[str, str]:
        """Generate output column names with normalization suffix."""
        suffix = "_norm" if self.normalized else ""
        return (
            f"psar{suffix}",
            f"psar_dir{suffix}"
        )

    test_params: ClassVar[list[dict]] = [
        {"af": 0.02, "af_step": 0.02, "af_max": 0.2},
        {"af": 0.02, "af_step": 0.02, "af_max": 0.2, "normalized": True},
        {"af": 0.01, "af_step": 0.01, "af_max": 0.1},
        {"af": 0.025, "af_step": 0.025, "af_max": 0.25},
    ]

    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable, reproducible output."""
        base_warmup = 100  # PSAR needs time to stabilize
        if self.normalized:
            from signalflow.ta._normalization import get_norm_window
            norm_window = self.norm_period or get_norm_window(20)
            return base_warmup + norm_window
        return base_warmup

@dataclass
@sf_component(name="trend/supertrend")
class SupertrendTrend(Feature):
    """Supertrend Indicator.

    Trend-following based on ATR bands.

    upper = HL2 + multiplier * ATR
    lower = HL2 - multiplier * ATR

    Outputs:
    - supertrend: trend line value
    - supertrend_dir: direction (+1 bullish, -1 bearish)

    Unbounded. Uses z-score in normalized mode.

    Reference: Olivier Seban
    """

    period: int = 10
    multiplier: float = 3.0
    normalized: bool = False
    norm_period: int | None = None

    requires = ["high", "low", "close"]
    outputs = ["supertrend_{period}", "supertrend_dir_{period}"]
    
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
        
        hl2 = (high + low) / 2
        
        basic_upper = hl2 + self.multiplier * atr
        basic_lower = hl2 - self.multiplier * atr
        
        upper = np.full(n, np.nan)
        lower = np.full(n, np.nan)
        supertrend = np.full(n, np.nan)
        direction = np.ones(n)
        
        upper[self.period - 1] = basic_upper[self.period - 1]
        lower[self.period - 1] = basic_lower[self.period - 1]
        
        for i in range(self.period, n):
            if basic_upper[i] < upper[i - 1] or close[i - 1] > upper[i - 1]:
                upper[i] = basic_upper[i]
            else:
                upper[i] = upper[i - 1]
            
            if basic_lower[i] > lower[i - 1] or close[i - 1] < lower[i - 1]:
                lower[i] = basic_lower[i]
            else:
                lower[i] = lower[i - 1]
            
            if close[i] > upper[i - 1]:
                direction[i] = 1
            elif close[i] < lower[i - 1]:
                direction[i] = -1
            else:
                direction[i] = direction[i - 1]

            supertrend[i] = lower[i] if direction[i] == 1 else upper[i]

        # Normalization: z-score for unbounded oscillator
        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window
            norm_window = self.norm_period or get_norm_window(self.period)
            supertrend = normalize_zscore(supertrend, window=norm_window)
            direction = normalize_zscore(direction, window=norm_window)

        col_supertrend, col_dir = self._get_output_names()
        return df.with_columns([
            pl.Series(name=col_supertrend, values=supertrend),
            pl.Series(name=col_dir, values=direction),
        ])

    def _get_output_names(self) -> tuple[str, str]:
        """Generate output column names with normalization suffix."""
        suffix = "_norm" if self.normalized else ""
        return (
            f"supertrend_{self.period}{suffix}",
            f"supertrend_dir_{self.period}{suffix}"
        )

    test_params: ClassVar[list[dict]] = [
        {"period": 10, "multiplier": 3.0},
        {"period": 10, "multiplier": 3.0, "normalized": True},
        {"period": 20, "multiplier": 2.5},
        {"period": 30, "multiplier": 3.5},
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
@sf_component(name="trend/chandelier")
class ChandelierTrend(Feature):
    """Chandelier Exit.

    Trailing stop based on ATR from highest high / lowest low.

    Long exit: HH - multiplier * ATR
    Short exit: LL + multiplier * ATR

    Outputs:
    - chandelier_long: long trailing stop
    - chandelier_short: short trailing stop

    Unbounded. Uses z-score in normalized mode.

    Reference: Chuck LeBeau, Alexander Elder
    """

    period: int = 22
    multiplier: float = 3.0
    normalized: bool = False
    norm_period: int | None = None

    requires = ["high", "low", "close"]
    outputs = ["chandelier_long_{period}", "chandelier_short_{period}"]
    
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
        
        chandelier_long = np.full(n, np.nan)
        chandelier_short = np.full(n, np.nan)
        
        for i in range(self.period - 1, n):
            atr_val = np.mean(tr[i - self.period + 1:i + 1])
            hh = np.max(high[i - self.period + 1:i + 1])
            ll = np.min(low[i - self.period + 1:i + 1])
            
            chandelier_long[i] = hh - self.multiplier * atr_val
            chandelier_short[i] = ll + self.multiplier * atr_val

        # Normalization: z-score for unbounded oscillator
        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window
            norm_window = self.norm_period or get_norm_window(self.period)
            chandelier_long = normalize_zscore(chandelier_long, window=norm_window)
            chandelier_short = normalize_zscore(chandelier_short, window=norm_window)

        col_long, col_short = self._get_output_names()
        return df.with_columns([
            pl.Series(name=col_long, values=chandelier_long),
            pl.Series(name=col_short, values=chandelier_short),
        ])

    def _get_output_names(self) -> tuple[str, str]:
        """Generate output column names with normalization suffix."""
        suffix = "_norm" if self.normalized else ""
        return (
            f"chandelier_long_{self.period}{suffix}",
            f"chandelier_short_{self.period}{suffix}"
        )

    test_params: ClassVar[list[dict]] = [
        {"period": 22, "multiplier": 3.0},
        {"period": 22, "multiplier": 3.0, "normalized": True},
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


@dataclass
@sf_component(name="trend/hilo")
class HiloTrend(Feature):
    """Gann HiLo Activator.

    Trend indicator using moving averages of highs and lows.

    Switches between:
    - MA(low) when close > MA(high) [uptrend]
    - MA(high) when close < MA(low) [downtrend]

    Outputs:
    - hilo: current stop level
    - hilo_dir: direction (+1 bullish, -1 bearish)

    Unbounded. Uses z-score in normalized mode.

    Reference: Robert Krausz, Stocks & Commodities 1998
    """

    high_period: int = 13
    low_period: int = 21
    ma_type: Literal["sma", "ema"] = "sma"
    normalized: bool = False
    norm_period: int | None = None

    requires = ["high", "low", "close"]
    outputs = ["hilo", "hilo_dir"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        close = df["close"].to_numpy()
        n = len(close)
        
        if self.ma_type == "ema":
            alpha_h = 2 / (self.high_period + 1)
            alpha_l = 2 / (self.low_period + 1)
            
            high_ma = np.full(n, np.nan)
            low_ma = np.full(n, np.nan)
            high_ma[0] = high[0]
            low_ma[0] = low[0]
            
            for i in range(1, n):
                high_ma[i] = alpha_h * high[i] + (1 - alpha_h) * high_ma[i - 1]
                low_ma[i] = alpha_l * low[i] + (1 - alpha_l) * low_ma[i - 1]
        else:  
            high_ma = np.full(n, np.nan)
            low_ma = np.full(n, np.nan)
            for i in range(self.high_period - 1, n):
                high_ma[i] = np.mean(high[i - self.high_period + 1:i + 1])
            for i in range(self.low_period - 1, n):
                low_ma[i] = np.mean(low[i - self.low_period + 1:i + 1])
        
        hilo = np.full(n, np.nan)
        direction = np.zeros(n)
        
        max_period = max(self.high_period, self.low_period)
        
        for i in range(max_period, n):
            if close[i] > high_ma[i - 1]:
                hilo[i] = low_ma[i]
                direction[i] = 1
            elif close[i] < low_ma[i - 1]:
                hilo[i] = high_ma[i]
                direction[i] = -1
            else:
                hilo[i] = hilo[i - 1]
                direction[i] = direction[i - 1]

        # Normalization: z-score for unbounded oscillator
        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window
            max_period = max(self.high_period, self.low_period)
            norm_window = self.norm_period or get_norm_window(max_period)
            hilo = normalize_zscore(hilo, window=norm_window)
            direction = normalize_zscore(direction, window=norm_window)

        col_hilo, col_dir = self._get_output_names()
        return df.with_columns([
            pl.Series(name=col_hilo, values=hilo),
            pl.Series(name=col_dir, values=direction),
        ])

    def _get_output_names(self) -> tuple[str, str]:
        """Generate output column names with normalization suffix."""
        suffix = "_norm" if self.normalized else ""
        return (
            f"hilo{suffix}",
            f"hilo_dir{suffix}"
        )

    test_params: ClassVar[list[dict]] = [
        {"high_period": 13, "low_period": 21, "ma_type": "sma"},
        {"high_period": 13, "low_period": 21, "ma_type": "sma", "normalized": True},
        {"high_period": 30, "low_period": 45, "ma_type": "sma"},
        {"high_period": 20, "low_period": 30, "ma_type": "ema"},
    ]

    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable, reproducible output."""
        max_period = max(self.high_period, self.low_period)
        base_warmup = max_period * 5
        if self.normalized:
            from signalflow.ta._normalization import get_norm_window
            norm_window = self.norm_period or get_norm_window(max_period)
            return base_warmup + norm_window
        return base_warmup


@dataclass
@sf_component(name="trend/cksp")
class CkspTrend(Feature):
    """Chande Kroll Stop.

    ATR-based trailing stop with two-step smoothing.

    Step 1: Initial stop = HH - x*ATR or LL + x*ATR
    Step 2: Final stop = max/min of initial over q periods

    Outputs:
    - cksp_long: long trailing stop
    - cksp_short: short trailing stop

    Unbounded. Uses z-score in normalized mode.

    Reference: Tushar Chande & Stanley Kroll, "The New Technical Trader"
    """

    p: int = 10
    x: float = 1.0
    q: int = 9
    normalized: bool = False
    norm_period: int | None = None

    requires = ["high", "low", "close"]
    outputs = ["cksp_long", "cksp_short"]
    
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
        atr[self.p - 1] = np.mean(tr[:self.p])
        alpha = 1 / self.p
        for i in range(self.p, n):
            atr[i] = alpha * tr[i] + (1 - alpha) * atr[i - 1]
        
        long_stop_init = np.full(n, np.nan)
        short_stop_init = np.full(n, np.nan)
        
        for i in range(self.p - 1, n):
            hh = np.max(high[i - self.p + 1:i + 1])
            ll = np.min(low[i - self.p + 1:i + 1])
            long_stop_init[i] = hh - self.x * atr[i]
            short_stop_init[i] = ll + self.x * atr[i]
        
        cksp_long = np.full(n, np.nan)
        cksp_short = np.full(n, np.nan)
        
        start = self.p + self.q - 2
        for i in range(start, n):
            cksp_long[i] = np.nanmax(long_stop_init[i - self.q + 1:i + 1])
            cksp_short[i] = np.nanmin(short_stop_init[i - self.q + 1:i + 1])

        # Normalization: z-score for unbounded oscillator
        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window
            norm_window = self.norm_period or get_norm_window(self.p + self.q)
            cksp_long = normalize_zscore(cksp_long, window=norm_window)
            cksp_short = normalize_zscore(cksp_short, window=norm_window)

        col_long, col_short = self._get_output_names()
        return df.with_columns([
            pl.Series(name=col_long, values=cksp_long),
            pl.Series(name=col_short, values=cksp_short),
        ])

    def _get_output_names(self) -> tuple[str, str]:
        """Generate output column names with normalization suffix."""
        suffix = "_norm" if self.normalized else ""
        return (
            f"cksp_long{suffix}",
            f"cksp_short{suffix}"
        )

    test_params: ClassVar[list[dict]] = [
        {"p": 10, "x": 1.0, "q": 9},
        {"p": 10, "x": 1.0, "q": 9, "normalized": True},
        {"p": 20, "x": 1.5, "q": 15},
        {"p": 30, "x": 2.0, "q": 20},
    ]

    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable, reproducible output."""
        base_warmup = (self.p + self.q) * 5
        if self.normalized:
            from signalflow.ta._normalization import get_norm_window
            norm_window = self.norm_period or get_norm_window(self.p + self.q)
            return base_warmup + norm_window
        return base_warmup

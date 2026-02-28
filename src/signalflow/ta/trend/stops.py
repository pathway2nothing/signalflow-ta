"""Trailing stop indicators - trend-following with defined exits."""

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
    psar_kernel as _psar_kernel,
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
from signalflow.ta._numba_kernels import (
    supertrend_kernel as _supertrend_kernel,
)


@dataclass
@feature("trend/psar")
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

    requires: ClassVar[list[str]] = ["high", "low", "close"]
    outputs: ClassVar[list[str]] = ["psar", "psar_dir"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        close = df["close"].to_numpy()
        _n = len(close)

        psar, direction = _psar_kernel(high, low, self.af_step, self.af_max)

        # Normalization: z-score for unbounded oscillator
        if self.normalized:
            from signalflow.ta._normalization import get_norm_window, normalize_zscore

            # Use a reasonable default period (20) since PSAR doesn't have an explicit period
            norm_window = self.norm_period or get_norm_window(20)
            psar = normalize_zscore(psar, window=norm_window)
            direction = normalize_zscore(direction, window=norm_window)

        col_psar, col_dir = self._get_output_names()
        return df.with_columns(
            [
                pl.Series(name=col_psar, values=psar),
                pl.Series(name=col_dir, values=direction),
            ]
        )

    def _get_output_names(self) -> tuple[str, str]:
        """Generate output column names with normalization suffix."""
        suffix = "_norm" if self.normalized else ""
        return (f"psar{suffix}", f"psar_dir{suffix}")

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
@feature("trend/supertrend")
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

    requires: ClassVar[list[str]] = ["high", "low", "close"]
    outputs: ClassVar[list[str]] = ["supertrend_{period}", "supertrend_dir_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        close = df["close"].to_numpy().astype(np.float64)
        _n = len(close)

        supertrend, direction = _supertrend_kernel(high, low, close, self.period, self.multiplier)

        # Normalization: z-score for unbounded oscillator
        if self.normalized:
            from signalflow.ta._normalization import get_norm_window, normalize_zscore

            norm_window = self.norm_period or get_norm_window(self.period)
            supertrend = normalize_zscore(supertrend, window=norm_window)
            direction = normalize_zscore(direction, window=norm_window)

        col_supertrend, col_dir = self._get_output_names()
        return df.with_columns(
            [
                pl.Series(name=col_supertrend, values=supertrend),
                pl.Series(name=col_dir, values=direction),
            ]
        )

    def _get_output_names(self) -> tuple[str, str]:
        """Generate output column names with normalization suffix."""
        suffix = "_norm" if self.normalized else ""
        return (
            f"supertrend_{self.period}{suffix}",
            f"supertrend_dir_{self.period}{suffix}",
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
@feature("trend/chandelier")
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

    requires: ClassVar[list[str]] = ["high", "low", "close"]
    outputs: ClassVar[list[str]] = ["chandelier_long_{period}", "chandelier_short_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        close = df["close"].to_numpy().astype(np.float64)

        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
        tr[0] = high[0] - low[0]

        atr_sma = _sma_nb(tr, self.period)
        hh = _rolling_max(high, self.period)
        ll = _rolling_min(low, self.period)

        chandelier_long = hh - self.multiplier * atr_sma
        chandelier_short = ll + self.multiplier * atr_sma

        # Normalization: z-score for unbounded oscillator
        if self.normalized:
            from signalflow.ta._normalization import get_norm_window, normalize_zscore

            norm_window = self.norm_period or get_norm_window(self.period)
            chandelier_long = normalize_zscore(chandelier_long, window=norm_window)
            chandelier_short = normalize_zscore(chandelier_short, window=norm_window)

        col_long, col_short = self._get_output_names()
        return df.with_columns(
            [
                pl.Series(name=col_long, values=chandelier_long),
                pl.Series(name=col_short, values=chandelier_short),
            ]
        )

    def _get_output_names(self) -> tuple[str, str]:
        """Generate output column names with normalization suffix."""
        suffix = "_norm" if self.normalized else ""
        return (
            f"chandelier_long_{self.period}{suffix}",
            f"chandelier_short_{self.period}{suffix}",
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
@feature("trend/hilo")
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

    requires: ClassVar[list[str]] = ["high", "low", "close"]
    outputs: ClassVar[list[str]] = ["hilo", "hilo_dir"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        close = df["close"].to_numpy()
        n = len(close)

        if self.ma_type == "ema":
            high_ma = _ema_sma_init(high, self.high_period)
            low_ma = _ema_sma_init(low, self.low_period)
        else:
            high_ma = _sma_nb(high, self.high_period)
            low_ma = _sma_nb(low, self.low_period)

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
            from signalflow.ta._normalization import get_norm_window, normalize_zscore

            max_period = max(self.high_period, self.low_period)
            norm_window = self.norm_period or get_norm_window(max_period)
            hilo = normalize_zscore(hilo, window=norm_window)
            direction = normalize_zscore(direction, window=norm_window)

        col_hilo, col_dir = self._get_output_names()
        return df.with_columns(
            [
                pl.Series(name=col_hilo, values=hilo),
                pl.Series(name=col_dir, values=direction),
            ]
        )

    def _get_output_names(self) -> tuple[str, str]:
        """Generate output column names with normalization suffix."""
        suffix = "_norm" if self.normalized else ""
        return (f"hilo{suffix}", f"hilo_dir{suffix}")

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
@feature("trend/cksp")
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

    requires: ClassVar[list[str]] = ["high", "low", "close"]
    outputs: ClassVar[list[str]] = ["cksp_long", "cksp_short"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        close = df["close"].to_numpy().astype(np.float64)
        _n = len(close)

        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
        tr[0] = high[0] - low[0]

        atr = _rma_sma_init(tr, self.p)
        hh = _rolling_max(high, self.p)
        ll = _rolling_min(low, self.p)

        long_stop_init = hh - self.x * atr
        short_stop_init = ll + self.x * atr

        # Second smoothing: rolling max/min over q periods
        cksp_long = _rolling_max(long_stop_init, self.q)
        cksp_short = _rolling_min(short_stop_init, self.q)

        # Normalization: z-score for unbounded oscillator
        if self.normalized:
            from signalflow.ta._normalization import get_norm_window, normalize_zscore

            norm_window = self.norm_period or get_norm_window(self.p + self.q)
            cksp_long = normalize_zscore(cksp_long, window=norm_window)
            cksp_short = normalize_zscore(cksp_short, window=norm_window)

        col_long, col_short = self._get_output_names()
        return df.with_columns(
            [
                pl.Series(name=col_long, values=cksp_long),
                pl.Series(name=col_short, values=cksp_short),
            ]
        )

    def _get_output_names(self) -> tuple[str, str]:
        """Generate output column names with normalization suffix."""
        suffix = "_norm" if self.normalized else ""
        return (f"cksp_long{suffix}", f"cksp_short{suffix}")

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

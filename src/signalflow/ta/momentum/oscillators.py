"""Stochastic and other momentum oscillators with reproducible initialization."""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import polars as pl

from signalflow.core import feature
from signalflow.feature.base import Feature
from signalflow.ta._numba_kernels import cci_kernel as _cci_kernel
from signalflow.ta._numba_kernels import rma_sma_init as _rma_sma_init
from signalflow.ta._numba_kernels import rolling_max as _rolling_max
from signalflow.ta._numba_kernels import rolling_min as _rolling_min
from signalflow.ta._numba_kernels import sma_nb as _sma_nb
from signalflow.ta._numba_kernels import stoch_kernel as _stoch_kernel
from signalflow.ta._numba_kernels import uo_kernel as _uo_kernel


@dataclass
@feature("momentum/stoch")
class StochMom(Feature):
    """Stochastic Oscillator.

    Uses pure lookback (rolling min/max), always reproducible.

    %K = 100 * (close - lowest_low) / (highest_high - lowest_low)
    %D = SMA(%K, d_period)

    Bounded [0, 100] (absolute) or [0, 1] (normalized).

    Reference: George Lane
    """

    k_period: int = 14
    d_period: int = 3
    smooth_k: int = 3
    normalized: bool = False

    requires: ClassVar[list[str]] = ["high", "low", "close"]
    outputs: ClassVar[list[str]] = ["stoch_k_{k_period}", "stoch_d_{k_period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        close = df["close"].to_numpy().astype(np.float64)

        stoch_k, stoch_d = _stoch_kernel(
            high, low, close, self.k_period, self.smooth_k, self.d_period
        )

        # Normalization: [0, 100] → [0, 1] for both outputs
        if self.normalized:
            stoch_k = stoch_k / 100
            stoch_d = stoch_d / 100

        col_k, col_d = self._get_output_names()
        return df.with_columns(
            [
                pl.Series(name=col_k, values=stoch_k),
                pl.Series(name=col_d, values=stoch_d),
            ]
        )

    def _get_output_names(self) -> tuple[str, str]:
        """Generate output column names with normalization suffix."""
        suffix = "_norm" if self.normalized else ""
        return (f"stoch_k_{self.k_period}{suffix}", f"stoch_d_{self.k_period}{suffix}")

    test_params: ClassVar[list[dict]] = [
        {"k_period": 14, "d_period": 3, "smooth_k": 3},
        {"k_period": 14, "d_period": 3, "smooth_k": 3, "normalized": True},
        {"k_period": 60, "d_period": 10, "smooth_k": 10},
        {"k_period": 120, "d_period": 20, "smooth_k": 20},
    ]

    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable, reproducible output."""
        return (self.k_period + self.smooth_k + self.d_period) * 3


@dataclass
@feature("momentum/stochrsi")
class StochRsiMom(Feature):
    """Stochastic RSI with reproducible initialization.

    Stochastic applied to RSI. RSI uses RMA with SMA init for reproducibility.

    StochRSI = (RSI - lowest_RSI) / (highest_RSI - lowest_RSI)

    Bounded [0, 100] (absolute) or [0, 1] (normalized).

    Reference: Tushar Chande & Stanley Kroll
    """

    rsi_period: int = 14
    stoch_period: int = 14
    k_period: int = 3
    d_period: int = 3
    normalized: bool = False

    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["stochrsi_k_{rsi_period}", "stochrsi_d_{rsi_period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        close = df["close"].to_numpy()
        n = len(close)

        # Calculate RSI with reproducible initialization
        diff = np.diff(close, prepend=close[0])
        diff[0] = 0

        gains = np.where(diff > 0, diff, 0)
        losses = np.where(diff < 0, -diff, 0)

        avg_gain = _rma_sma_init(gains, self.rsi_period)
        avg_loss = _rma_sma_init(losses, self.rsi_period)

        rsi = 100 - (100 / (1 + avg_gain / (avg_loss + 1e-10)))

        # Stochastic of RSI using rolling min/max + SMA kernels
        rsi_min = _rolling_min(rsi, self.stoch_period)
        rsi_max = _rolling_max(rsi, self.stoch_period)

        raw_stoch = np.full(n, np.nan)
        start = self.rsi_period + self.stoch_period - 2
        for i in range(start, n):
            if not np.isnan(rsi_min[i]) and not np.isnan(rsi_max[i]):
                if rsi_max[i] != rsi_min[i]:
                    raw_stoch[i] = 100 * (rsi[i] - rsi_min[i]) / (rsi_max[i] - rsi_min[i])
                else:
                    raw_stoch[i] = 50.0

        # Smoothed %K and %D
        stoch_k = _sma_nb(raw_stoch, self.k_period)
        stoch_d = _sma_nb(stoch_k, self.d_period)

        # Normalization: [0, 100] → [0, 1] for both outputs
        if self.normalized:
            stoch_k = stoch_k / 100
            stoch_d = stoch_d / 100

        col_k, col_d = self._get_output_names()
        return df.with_columns(
            [
                pl.Series(name=col_k, values=stoch_k),
                pl.Series(name=col_d, values=stoch_d),
            ]
        )

    def _get_output_names(self) -> tuple[str, str]:
        """Generate output column names with normalization suffix."""
        suffix = "_norm" if self.normalized else ""
        return (
            f"stochrsi_k_{self.rsi_period}{suffix}",
            f"stochrsi_d_{self.rsi_period}{suffix}",
        )

    test_params: ClassVar[list[dict]] = [
        {"rsi_period": 14, "stoch_period": 14, "k_period": 3, "d_period": 3},
        {
            "rsi_period": 14,
            "stoch_period": 14,
            "k_period": 3,
            "d_period": 3,
            "normalized": True,
        },
        {"rsi_period": 60, "stoch_period": 60, "k_period": 10, "d_period": 10},
        {"rsi_period": 120, "stoch_period": 120, "k_period": 20, "d_period": 20},
    ]

    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable, reproducible output."""
        return (self.rsi_period + self.stoch_period + self.k_period + self.d_period) * 3


@dataclass
@feature("momentum/willr")
class WillrMom(Feature):
    """Williams %R.

    Pure lookback - always reproducible.

    %R = -100 * (highest_high - close) / (highest_high - lowest_low)

    Bounded [-100, 0] (absolute) or [0, 1] (normalized).

    Reference: Larry Williams
    """

    period: int = 14
    normalized: bool = False

    requires: ClassVar[list[str]] = ["high", "low", "close"]
    outputs: ClassVar[list[str]] = ["willr_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        close = df["close"].to_numpy().astype(np.float64)

        hh = _rolling_max(high, self.period)
        ll = _rolling_min(low, self.period)

        diff = hh - ll
        willr = np.where(diff != 0, -100 * (hh - close) / diff, -50.0)
        willr[: self.period - 1] = np.nan

        # Normalization: [-100, 0] → [0, 1]
        if self.normalized:
            willr = (willr + 100) / 100

        col_name = self._get_output_name()
        return df.with_columns(pl.Series(name=col_name, values=willr))

    def _get_output_name(self) -> str:
        """Generate output column name with normalization suffix."""
        suffix = "_norm" if self.normalized else ""
        return f"willr_{self.period}{suffix}"

    test_params: ClassVar[list[dict]] = [
        {"period": 14},
        {"period": 14, "normalized": True},
        {"period": 60},
        {"period": 240},
    ]

    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable, reproducible output."""
        return self.period * 3


@dataclass
@feature("momentum/cci")
class CciMom(Feature):
    """Commodity Channel Index (CCI).

    Uses SMA and MAD - always reproducible.

    TP = (high + low + close) / 3
    CCI = (TP - SMA(TP)) / (0.015 * MAD(TP))

    Unbounded (typically ±100 but no theoretical limit). Uses z-score in normalized mode.

    Reference: Donald Lambert
    """

    period: int = 20
    constant: float = 0.015
    normalized: bool = False
    norm_period: int | None = None

    requires: ClassVar[list[str]] = ["high", "low", "close"]
    outputs: ClassVar[list[str]] = ["cci_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        close = df["close"].to_numpy()
        _n = len(close)

        tp = ((high + low + close) / 3).astype(np.float64)

        cci = _cci_kernel(tp, self.period, self.constant)

        # Normalization: z-score for unbounded oscillator
        if self.normalized:
            from signalflow.ta._normalization import get_norm_window, normalize_zscore

            norm_window = self.norm_period or get_norm_window(self.period)
            cci = normalize_zscore(cci, window=norm_window)

        col_name = self._get_output_name()
        return df.with_columns(pl.Series(name=col_name, values=cci))

    def _get_output_name(self) -> str:
        """Generate output column name with normalization suffix."""
        suffix = "_norm" if self.normalized else ""
        return f"cci_{self.period}{suffix}"

    test_params: ClassVar[list[dict]] = [
        {"period": 20, "constant": 0.015},
        {"period": 20, "constant": 0.015, "normalized": True},
        {"period": 60, "constant": 0.015},
        {"period": 240, "constant": 0.015},
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
@feature("momentum/uo")
class UoMom(Feature):
    """Ultimate Oscillator (UO).

    Uses rolling sums - always reproducible.

    Bounded [0, 100] (absolute) or [0, 1] (normalized).

    Reference: Larry Williams
    """

    fast: int = 7
    medium: int = 14
    slow: int = 28
    fast_weight: float = 4.0
    medium_weight: float = 2.0
    slow_weight: float = 1.0
    normalized: bool = False

    requires: ClassVar[list[str]] = ["high", "low", "close"]
    outputs: ClassVar[list[str]] = ["uo_{fast}_{medium}_{slow}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        close = df["close"].to_numpy()
        _n = len(close)

        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]

        bp = (close - np.minimum(low, prev_close)).astype(np.float64)
        tr = (np.maximum(high, prev_close) - np.minimum(low, prev_close)).astype(np.float64)

        uo = _uo_kernel(
            bp, tr,
            self.fast, self.medium, self.slow,
            self.fast_weight, self.medium_weight, self.slow_weight,
        )

        # Normalization: [0, 100] → [0, 1]
        if self.normalized:
            uo = uo / 100

        col_name = self._get_output_name()
        return df.with_columns(pl.Series(name=col_name, values=uo))

    def _get_output_name(self) -> str:
        """Generate output column name with normalization suffix."""
        suffix = "_norm" if self.normalized else ""
        return f"uo_{self.fast}_{self.medium}_{self.slow}{suffix}"

    test_params: ClassVar[list[dict]] = [
        {"fast": 7, "medium": 14, "slow": 28},
        {"fast": 7, "medium": 14, "slow": 28, "normalized": True},
        {"fast": 15, "medium": 30, "slow": 60},
        {"fast": 30, "medium": 60, "slow": 120},
    ]

    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable, reproducible output."""
        return self.slow * 5


@dataclass
@feature("momentum/ao")
class AoMom(Feature):
    """Awesome Oscillator (AO).

    Uses SMA - always reproducible.

    Median = (high + low) / 2
    AO = SMA(Median, fast) - SMA(Median, slow)

    Unbounded oscillator. Uses z-score in normalized mode.

    Reference: Bill Williams
    """

    fast: int = 5
    slow: int = 34
    normalized: bool = False
    norm_period: int | None = None

    requires: ClassVar[list[str]] = ["high", "low"]
    outputs: ClassVar[list[str]] = ["ao_{fast}_{slow}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)

        median = (high + low) / 2
        fast_sma = _sma_nb(median, self.fast)
        slow_sma = _sma_nb(median, self.slow)
        ao = fast_sma - slow_sma

        # Normalization: z-score for unbounded oscillator
        if self.normalized:
            from signalflow.ta._normalization import get_norm_window, normalize_zscore

            norm_window = self.norm_period or get_norm_window(self.slow)
            ao = normalize_zscore(ao, window=norm_window)

        col_name = self._get_output_name()
        return df.with_columns(pl.Series(name=col_name, values=ao))

    def _get_output_name(self) -> str:
        """Generate output column name with normalization suffix."""
        suffix = "_norm" if self.normalized else ""
        return f"ao_{self.fast}_{self.slow}{suffix}"

    test_params: ClassVar[list[dict]] = [
        {"fast": 5, "slow": 34},
        {"fast": 5, "slow": 34, "normalized": True},
        {"fast": 15, "slow": 100},
        {"fast": 30, "slow": 200},
    ]

    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable, reproducible output."""
        base_warmup = self.slow * 5
        if self.normalized:
            from signalflow.ta._normalization import get_norm_window

            norm_window = self.norm_period or get_norm_window(self.slow)
            return base_warmup + norm_window
        return base_warmup

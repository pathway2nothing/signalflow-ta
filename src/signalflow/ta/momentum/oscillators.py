"""Stochastic and other momentum oscillators with reproducible initialization."""
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import polars as pl

from signalflow import sf_component
from signalflow.feature.base import Feature


def _rma_sma_init(values: np.ndarray, period: int) -> np.ndarray:
    """RMA with SMA initialization for reproducibility."""
    n = len(values)
    alpha = 1 / period
    rma = np.full(n, np.nan)
    
    if n < period:
        return rma
    
    rma[period - 1] = np.mean(values[:period])
    
    for i in range(period, n):
        rma[i] = alpha * values[i] + (1 - alpha) * rma[i - 1]
    
    return rma


@dataclass
@sf_component(name="momentum/stoch")
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

    requires = ["high", "low", "close"]
    outputs = ["stoch_k_{k_period}", "stoch_d_{k_period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        close = df["close"].to_numpy()
        n = len(close)

        # Raw %K
        raw_k = np.full(n, np.nan)

        for i in range(self.k_period - 1, n):
            hh = np.max(high[i - self.k_period + 1:i + 1])
            ll = np.min(low[i - self.k_period + 1:i + 1])

            if hh != ll:
                raw_k[i] = 100 * (close[i] - ll) / (hh - ll)
            else:
                raw_k[i] = 50.0  # Neutral when no range

        # Smoothed %K (SMA)
        stoch_k = np.full(n, np.nan)
        start_k = self.k_period + self.smooth_k - 2

        for i in range(start_k, n):
            window = raw_k[i - self.smooth_k + 1:i + 1]
            valid = window[~np.isnan(window)]
            if len(valid) > 0:
                stoch_k[i] = np.mean(valid)

        # %D (SMA of %K)
        stoch_d = np.full(n, np.nan)
        start_d = start_k + self.d_period - 1

        for i in range(start_d, n):
            window = stoch_k[i - self.d_period + 1:i + 1]
            valid = window[~np.isnan(window)]
            if len(valid) > 0:
                stoch_d[i] = np.mean(valid)

        # Normalization: [0, 100] → [0, 1] for both outputs
        if self.normalized:
            stoch_k = stoch_k / 100
            stoch_d = stoch_d / 100

        col_k, col_d = self._get_output_names()
        return df.with_columns([
            pl.Series(name=col_k, values=stoch_k),
            pl.Series(name=col_d, values=stoch_d),
        ])

    def _get_output_names(self) -> tuple[str, str]:
        """Generate output column names with normalization suffix."""
        suffix = "_norm" if self.normalized else ""
        return (
            f"stoch_k_{self.k_period}{suffix}",
            f"stoch_d_{self.k_period}{suffix}"
        )

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
@sf_component(name="momentum/stochrsi")
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

    requires = ["close"]
    outputs = ["stochrsi_k_{rsi_period}", "stochrsi_d_{rsi_period}"]

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

        # Stochastic of RSI (pure lookback - reproducible)
        raw_stoch = np.full(n, np.nan)
        start = self.rsi_period + self.stoch_period - 2

        for i in range(start, n):
            rsi_window = rsi[i - self.stoch_period + 1:i + 1]
            valid_rsi = rsi_window[~np.isnan(rsi_window)]

            if len(valid_rsi) >= 2:
                rsi_min = np.min(valid_rsi)
                rsi_max = np.max(valid_rsi)

                if rsi_max != rsi_min:
                    raw_stoch[i] = 100 * (rsi[i] - rsi_min) / (rsi_max - rsi_min)
                else:
                    raw_stoch[i] = 50.0

        # Smoothed %K (SMA)
        stoch_k = np.full(n, np.nan)
        start_k = start + self.k_period - 1

        for i in range(start_k, n):
            window = raw_stoch[i - self.k_period + 1:i + 1]
            valid = window[~np.isnan(window)]
            if len(valid) > 0:
                stoch_k[i] = np.mean(valid)

        # %D (SMA of %K)
        stoch_d = np.full(n, np.nan)
        start_d = start_k + self.d_period - 1

        for i in range(start_d, n):
            window = stoch_k[i - self.d_period + 1:i + 1]
            valid = window[~np.isnan(window)]
            if len(valid) > 0:
                stoch_d[i] = np.mean(valid)

        # Normalization: [0, 100] → [0, 1] for both outputs
        if self.normalized:
            stoch_k = stoch_k / 100
            stoch_d = stoch_d / 100

        col_k, col_d = self._get_output_names()
        return df.with_columns([
            pl.Series(name=col_k, values=stoch_k),
            pl.Series(name=col_d, values=stoch_d),
        ])

    def _get_output_names(self) -> tuple[str, str]:
        """Generate output column names with normalization suffix."""
        suffix = "_norm" if self.normalized else ""
        return (
            f"stochrsi_k_{self.rsi_period}{suffix}",
            f"stochrsi_d_{self.rsi_period}{suffix}"
        )

    test_params: ClassVar[list[dict]] = [
        {"rsi_period": 14, "stoch_period": 14, "k_period": 3, "d_period": 3},
        {"rsi_period": 14, "stoch_period": 14, "k_period": 3, "d_period": 3, "normalized": True},
        {"rsi_period": 60, "stoch_period": 60, "k_period": 10, "d_period": 10},
        {"rsi_period": 120, "stoch_period": 120, "k_period": 20, "d_period": 20},
    ]

    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable, reproducible output."""
        return (self.rsi_period + self.stoch_period + self.k_period + self.d_period) * 3


@dataclass
@sf_component(name="momentum/willr")
class WillrMom(Feature):
    """Williams %R.

    Pure lookback - always reproducible.

    %R = -100 * (highest_high - close) / (highest_high - lowest_low)

    Bounded [-100, 0] (absolute) or [0, 1] (normalized).

    Reference: Larry Williams
    """

    period: int = 14
    normalized: bool = False

    requires = ["high", "low", "close"]
    outputs = ["willr_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        close = df["close"].to_numpy()
        n = len(close)

        willr = np.full(n, np.nan)

        for i in range(self.period - 1, n):
            hh = np.max(high[i - self.period + 1:i + 1])
            ll = np.min(low[i - self.period + 1:i + 1])

            if hh != ll:
                willr[i] = -100 * (hh - close[i]) / (hh - ll)
            else:
                willr[i] = -50.0

        # Normalization: [-100, 0] → [0, 1]
        if self.normalized:
            willr = (willr + 100) / 100

        col_name = self._get_output_name()
        return df.with_columns(
            pl.Series(name=col_name, values=willr)
        )

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
@sf_component(name="momentum/cci")
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

    requires = ["high", "low", "close"]
    outputs = ["cci_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        close = df["close"].to_numpy()
        n = len(close)

        tp = (high + low + close) / 3
        cci = np.full(n, np.nan)

        for i in range(self.period - 1, n):
            window = tp[i - self.period + 1:i + 1]
            sma = np.mean(window)
            mad = np.mean(np.abs(window - sma))

            if mad > 0:
                cci[i] = (tp[i] - sma) / (self.constant * mad)

        # Normalization: z-score for unbounded oscillator
        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window
            norm_window = self.norm_period or get_norm_window(self.period)
            cci = normalize_zscore(cci, window=norm_window)

        col_name = self._get_output_name()
        return df.with_columns(
            pl.Series(name=col_name, values=cci)
        )

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
@sf_component(name="momentum/uo")
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

    requires = ["high", "low", "close"]
    outputs = ["uo_{fast}_{medium}_{slow}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        close = df["close"].to_numpy()
        n = len(close)

        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]

        bp = close - np.minimum(low, prev_close)
        tr = np.maximum(high, prev_close) - np.minimum(low, prev_close)

        uo = np.full(n, np.nan)

        for i in range(self.slow - 1, n):
            fast_bp = np.sum(bp[i - self.fast + 1:i + 1])
            fast_tr = np.sum(tr[i - self.fast + 1:i + 1])

            med_bp = np.sum(bp[i - self.medium + 1:i + 1])
            med_tr = np.sum(tr[i - self.medium + 1:i + 1])

            slow_bp = np.sum(bp[i - self.slow + 1:i + 1])
            slow_tr = np.sum(tr[i - self.slow + 1:i + 1])

            if fast_tr > 0 and med_tr > 0 and slow_tr > 0:
                fast_avg = fast_bp / fast_tr
                med_avg = med_bp / med_tr
                slow_avg = slow_bp / slow_tr

                total_weight = self.fast_weight + self.medium_weight + self.slow_weight
                weighted = (self.fast_weight * fast_avg +
                           self.medium_weight * med_avg +
                           self.slow_weight * slow_avg)

                uo[i] = 100 * weighted / total_weight

        # Normalization: [0, 100] → [0, 1]
        if self.normalized:
            uo = uo / 100

        col_name = self._get_output_name()
        return df.with_columns(
            pl.Series(name=col_name, values=uo)
        )

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
@sf_component(name="momentum/ao")
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

    requires = ["high", "low"]
    outputs = ["ao_{fast}_{slow}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        n = len(high)

        median = (high + low) / 2
        ao = np.full(n, np.nan)

        for i in range(self.slow - 1, n):
            fast_sma = np.mean(median[i - self.fast + 1:i + 1])
            slow_sma = np.mean(median[i - self.slow + 1:i + 1])
            ao[i] = fast_sma - slow_sma

        # Normalization: z-score for unbounded oscillator
        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window
            norm_window = self.norm_period or get_norm_window(self.slow)
            ao = normalize_zscore(ao, window=norm_window)

        col_name = self._get_output_name()
        return df.with_columns(
            pl.Series(name=col_name, values=ao)
        )

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

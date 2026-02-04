"""MACD family indicators with reproducible EMA initialization."""
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import polars as pl

from signalflow import sf_component
from signalflow.feature.base import Feature


def _ema_sma_init(values: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate EMA with SMA initialization for reproducibility.
    
    Instead of ema[0] = values[0], we initialize with SMA of first `period` values.
    This makes EMA independent of the starting point after warmup.
    
    Args:
        values: Input array
        period: EMA period (also used for SMA initialization)
        
    Returns:
        EMA array with first (period-1) values as NaN
    """
    n = len(values)
    alpha = 2 / (period + 1)
    ema = np.full(n, np.nan)
    
    if n < period:
        return ema
    
    # Initialize with SMA of first `period` values
    ema[period - 1] = np.mean(values[:period])
    
    # Continue with standard EMA
    for i in range(period, n):
        ema[i] = alpha * values[i] + (1 - alpha) * ema[i - 1]
    
    return ema


@dataclass
@sf_component(name="momentum/macd")
class MacdMom(Feature):
    """Moving Average Convergence Divergence (MACD).

    Trend-following momentum indicator with reproducible EMA initialization.

    MACD = EMA(close, fast) - EMA(close, slow)
    Signal = EMA(MACD, signal)
    Histogram = MACD - Signal

    Key improvement: EMA initialized with SMA instead of first value,
    ensuring reproducibility regardless of data entry point.

    Unbounded oscillator. Uses z-score in normalized mode (all 3 outputs).

    Reference: Gerald Appel
    https://www.investopedia.com/terms/m/macd.asp
    """

    fast: int = 12
    slow: int = 26
    signal: int = 9
    normalized: bool = False
    norm_period: int | None = None

    requires = ["close"]
    outputs = ["macd_{fast}_{slow}", "macd_signal_{signal}", "macd_hist_{fast}_{slow}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        close = df["close"].to_numpy()
        n = len(close)

        # EMA with SMA initialization for reproducibility
        ema_fast = _ema_sma_init(close, self.fast)
        ema_slow = _ema_sma_init(close, self.slow)

        # MACD line (valid after slow period)
        macd = ema_fast - ema_slow

        # Signal line - EMA of MACD with SMA init
        # Start after slow period where MACD becomes valid
        alpha_sig = 2 / (self.signal + 1)
        signal_line = np.full(n, np.nan)

        # Find first valid MACD index
        start_idx = self.slow - 1
        signal_start = start_idx + self.signal - 1

        if signal_start < n:
            # Initialize signal with SMA of first `signal` valid MACD values
            signal_line[signal_start] = np.mean(macd[start_idx:signal_start + 1])

            for i in range(signal_start + 1, n):
                signal_line[i] = alpha_sig * macd[i] + (1 - alpha_sig) * signal_line[i - 1]

        histogram = macd - signal_line

        # Clear invalid values
        macd[:start_idx] = np.nan

        # Normalization: z-score for all 3 outputs independently
        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window
            norm_window = self.norm_period or get_norm_window(self.slow)
            macd = normalize_zscore(macd, window=norm_window)
            signal_line = normalize_zscore(signal_line, window=norm_window)
            histogram = normalize_zscore(histogram, window=norm_window)

        col_macd, col_signal, col_hist = self._get_output_names()
        return df.with_columns([
            pl.Series(name=col_macd, values=macd),
            pl.Series(name=col_signal, values=signal_line),
            pl.Series(name=col_hist, values=histogram),
        ])

    def _get_output_names(self) -> tuple[str, str, str]:
        """Generate output column names with normalization suffix."""
        suffix = "_norm" if self.normalized else ""
        return (
            f"macd_{self.fast}_{self.slow}{suffix}",
            f"macd_signal_{self.signal}{suffix}",
            f"macd_hist_{self.fast}_{self.slow}{suffix}"
        )

    test_params: ClassVar[list[dict]] = [
        {"fast": 12, "slow": 26, "signal": 9},
        {"fast": 12, "slow": 26, "signal": 9, "normalized": True},
        {"fast": 24, "slow": 52, "signal": 18},
        {"fast": 48, "slow": 104, "signal": 36},
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

@dataclass
@sf_component(name="momentum/ppo")
class PpoMom(Feature):
    """Percentage Price Oscillator (PPO).

    MACD expressed as percentage with reproducible EMA initialization.

    PPO = 100 * (EMA_fast - EMA_slow) / EMA_slow
    Signal = EMA(PPO, signal)
    Histogram = PPO - Signal

    Unbounded oscillator. Uses z-score in normalized mode (all 3 outputs).

    Reference: https://school.stockcharts.com/doku.php?id=technical_indicators:price_oscillators_ppo
    """

    fast: int = 12
    slow: int = 26
    signal: int = 9
    normalized: bool = False
    norm_period: int | None = None

    requires = ["close"]
    outputs = ["ppo_{fast}_{slow}", "ppo_signal_{signal}", "ppo_hist_{fast}_{slow}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        close = df["close"].to_numpy()
        n = len(close)

        # EMA with SMA initialization
        ema_fast = _ema_sma_init(close, self.fast)
        ema_slow = _ema_sma_init(close, self.slow)

        # PPO as percentage
        ppo = 100 * (ema_fast - ema_slow) / (ema_slow + 1e-10)

        # Signal line
        alpha_sig = 2 / (self.signal + 1)
        signal_line = np.full(n, np.nan)

        start_idx = self.slow - 1
        signal_start = start_idx + self.signal - 1

        if signal_start < n:
            signal_line[signal_start] = np.mean(ppo[start_idx:signal_start + 1])

            for i in range(signal_start + 1, n):
                signal_line[i] = alpha_sig * ppo[i] + (1 - alpha_sig) * signal_line[i - 1]

        histogram = ppo - signal_line
        ppo[:start_idx] = np.nan

        # Normalization: z-score for all 3 outputs independently
        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window
            norm_window = self.norm_period or get_norm_window(self.slow)
            ppo = normalize_zscore(ppo, window=norm_window)
            signal_line = normalize_zscore(signal_line, window=norm_window)
            histogram = normalize_zscore(histogram, window=norm_window)

        col_ppo, col_signal, col_hist = self._get_output_names()
        return df.with_columns([
            pl.Series(name=col_ppo, values=ppo),
            pl.Series(name=col_signal, values=signal_line),
            pl.Series(name=col_hist, values=histogram),
        ])

    def _get_output_names(self) -> tuple[str, str, str]:
        """Generate output column names with normalization suffix."""
        suffix = "_norm" if self.normalized else ""
        return (
            f"ppo_{self.fast}_{self.slow}{suffix}",
            f"ppo_signal_{self.signal}{suffix}",
            f"ppo_hist_{self.fast}_{self.slow}{suffix}"
        )

    test_params: ClassVar[list[dict]] = [
        {"fast": 12, "slow": 26, "signal": 9},
        {"fast": 12, "slow": 26, "signal": 9, "normalized": True},
        {"fast": 24, "slow": 52, "signal": 18},
        {"fast": 48, "slow": 104, "signal": 36},
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


@dataclass
@sf_component(name="momentum/tsi")
class TsiMom(Feature):
    """True Strength Index (TSI).

    Double-smoothed momentum with reproducible EMA initialization.

    PC = close - prev_close
    TSI = 100 * EMA(EMA(PC, slow), fast) / EMA(EMA(|PC|, slow), fast)
    Signal = EMA(TSI, signal)

    Unbounded oscillator. Uses z-score in normalized mode (both outputs).

    Reference: William Blau
    https://www.investopedia.com/terms/t/tsi.asp
    """

    fast: int = 13
    slow: int = 25
    signal: int = 13
    normalized: bool = False
    norm_period: int | None = None

    requires = ["close"]
    outputs = ["tsi_{fast}_{slow}", "tsi_signal_{signal}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        close = df["close"].to_numpy()
        n = len(close)

        # Price change
        pc = np.diff(close, prepend=close[0])
        pc[0] = 0
        abs_pc = np.abs(pc)

        # Double smooth with SMA initialization
        pc_ema1 = _ema_sma_init(pc, self.slow)
        pc_ema2 = _ema_sma_init(pc_ema1, self.fast)

        abs_pc_ema1 = _ema_sma_init(abs_pc, self.slow)
        abs_pc_ema2 = _ema_sma_init(abs_pc_ema1, self.fast)

        # TSI calculation
        tsi = 100 * pc_ema2 / (abs_pc_ema2 + 1e-10)

        # Signal line
        alpha_sig = 2 / (self.signal + 1)
        tsi_signal = np.full(n, np.nan)

        # Find first valid TSI index
        start_idx = self.slow + self.fast - 2
        signal_start = start_idx + self.signal - 1

        if signal_start < n:
            # Find valid values for initialization
            valid_tsi = tsi[start_idx:signal_start + 1]
            valid_tsi = valid_tsi[~np.isnan(valid_tsi)]
            if len(valid_tsi) > 0:
                tsi_signal[signal_start] = np.mean(valid_tsi)

                for i in range(signal_start + 1, n):
                    if not np.isnan(tsi[i]) and not np.isnan(tsi_signal[i - 1]):
                        tsi_signal[i] = alpha_sig * tsi[i] + (1 - alpha_sig) * tsi_signal[i - 1]

        # Normalization: z-score for both outputs independently
        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window
            norm_window = self.norm_period or get_norm_window(self.slow)
            tsi = normalize_zscore(tsi, window=norm_window)
            tsi_signal = normalize_zscore(tsi_signal, window=norm_window)

        col_tsi, col_signal = self._get_output_names()
        return df.with_columns([
            pl.Series(name=col_tsi, values=tsi),
            pl.Series(name=col_signal, values=tsi_signal),
        ])

    def _get_output_names(self) -> tuple[str, str]:
        """Generate output column names with normalization suffix."""
        suffix = "_norm" if self.normalized else ""
        return (
            f"tsi_{self.fast}_{self.slow}{suffix}",
            f"tsi_signal_{self.signal}{suffix}"
        )

    test_params: ClassVar[list[dict]] = [
        {"fast": 13, "slow": 25, "signal": 13},
        {"fast": 13, "slow": 25, "signal": 13, "normalized": True},
        {"fast": 26, "slow": 50, "signal": 26},
        {"fast": 52, "slow": 100, "signal": 52},
    ]

    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable, reproducible output."""
        base_warmup = (self.slow + self.fast) * 5
        if self.normalized:
            from signalflow.ta._normalization import get_norm_window
            norm_window = self.norm_period or get_norm_window(self.slow)
            return base_warmup + norm_window
        return base_warmup


@dataclass
@sf_component(name="momentum/trix")
class TrixMom(Feature):
    """Triple Exponential Average (TRIX).

    Rate of change of triple-smoothed EMA with reproducible initialization.

    EMA1 = EMA(close, period)
    EMA2 = EMA(EMA1, period)
    EMA3 = EMA(EMA2, period)
    TRIX = 100 * (EMA3 - EMA3[1]) / EMA3[1]

    Unbounded oscillator. Uses z-score in normalized mode (both outputs).

    Reference: Jack Hutson
    https://www.investopedia.com/terms/t/trix.asp
    """

    period: int = 18
    signal: int = 9
    normalized: bool = False
    norm_period: int | None = None

    requires = ["close"]
    outputs = ["trix_{period}", "trix_signal_{signal}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        close = df["close"].to_numpy()
        n = len(close)

        # Triple EMA with SMA initialization
        ema1 = _ema_sma_init(close, self.period)
        ema2 = _ema_sma_init(ema1, self.period)
        ema3 = _ema_sma_init(ema2, self.period)

        # TRIX = rate of change of triple EMA
        trix = np.full(n, np.nan)
        for i in range(1, n):
            if not np.isnan(ema3[i]) and not np.isnan(ema3[i - 1]) and ema3[i - 1] != 0:
                trix[i] = 100 * (ema3[i] - ema3[i - 1]) / ema3[i - 1]

        # Signal line
        alpha_sig = 2 / (self.signal + 1)
        trix_signal = np.full(n, np.nan)

        # Find first valid TRIX index (3 * period - 2)
        start_idx = 3 * self.period - 2
        signal_start = start_idx + self.signal - 1

        if signal_start < n:
            valid_trix = trix[start_idx:signal_start + 1]
            valid_trix = valid_trix[~np.isnan(valid_trix)]
            if len(valid_trix) > 0:
                trix_signal[signal_start] = np.mean(valid_trix)

                for i in range(signal_start + 1, n):
                    if not np.isnan(trix[i]) and not np.isnan(trix_signal[i - 1]):
                        trix_signal[i] = alpha_sig * trix[i] + (1 - alpha_sig) * trix_signal[i - 1]

        # Normalization: z-score for both outputs independently
        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window
            norm_window = self.norm_period or get_norm_window(self.period)
            trix = normalize_zscore(trix, window=norm_window)
            trix_signal = normalize_zscore(trix_signal, window=norm_window)

        col_trix, col_signal = self._get_output_names()
        return df.with_columns([
            pl.Series(name=col_trix, values=trix),
            pl.Series(name=col_signal, values=trix_signal),
        ])

    def _get_output_names(self) -> tuple[str, str]:
        """Generate output column names with normalization suffix."""
        suffix = "_norm" if self.normalized else ""
        return (
            f"trix_{self.period}{suffix}",
            f"trix_signal_{self.signal}{suffix}"
        )

    test_params: ClassVar[list[dict]] = [
        {"period": 18, "signal": 9},
        {"period": 18, "signal": 9, "normalized": True},
        {"period": 60, "signal": 30},
        {"period": 120, "signal": 60},
    ]

    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable, reproducible output."""
        base_warmup = self.period * 12
        if self.normalized:
            from signalflow.ta._normalization import get_norm_window
            norm_window = self.norm_period or get_norm_window(self.period)
            return base_warmup + norm_window
        return base_warmup

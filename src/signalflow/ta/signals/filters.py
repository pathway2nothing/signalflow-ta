"""Signal filters for conditional signal generation."""

from dataclasses import dataclass
from abc import ABC, abstractmethod

import numpy as np
import polars as pl


class SignalFilter(ABC):
    """Base class for signal filters."""

    @abstractmethod
    def apply(self, df: pl.DataFrame) -> pl.Series:
        """Apply filter and return boolean mask.

        Args:
            df: DataFrame with OHLCV data.

        Returns:
            Boolean Series where True = signal passes filter.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def warmup(self) -> int:
        """Minimum bars needed for filter calculation."""
        raise NotImplementedError


@dataclass
class PriceUptrendFilter(SignalFilter):
    """Filter signals to only trigger during price uptrend.

    Uptrend defined as: close > SMA(close, window)
    """

    window: int = 5

    def apply(self, df: pl.DataFrame) -> pl.Series:
        close = df["close"].to_numpy()
        n = len(close)

        sma = np.full(n, np.nan)
        for i in range(self.window - 1, n):
            sma[i] = np.mean(close[i - self.window + 1 : i + 1])

        mask = close > sma
        return pl.Series(values=mask)

    @property
    def warmup(self) -> int:
        return self.window


@dataclass
class PriceDowntrendFilter(SignalFilter):
    """Filter signals to only trigger during price downtrend.

    Downtrend defined as: close < SMA(close, window)
    """

    window: int = 5

    def apply(self, df: pl.DataFrame) -> pl.Series:
        close = df["close"].to_numpy()
        n = len(close)

        sma = np.full(n, np.nan)
        for i in range(self.window - 1, n):
            sma[i] = np.mean(close[i - self.window + 1 : i + 1])

        mask = close < sma
        return pl.Series(values=mask)

    @property
    def warmup(self) -> int:
        return self.window


@dataclass
class LowVolatilityFilter(SignalFilter):
    """Filter signals to only trigger during low volatility periods.

    Low volatility defined as: current_vol < coef * mean_vol
    """

    window: int = 60
    coef: float = 1.0

    def apply(self, df: pl.DataFrame) -> pl.Series:
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        close = df["close"].to_numpy()
        n = len(close)

        price = (close + high + low) / 3
        returns = np.diff(price, prepend=price[0]) / (np.roll(price, 1) + 1e-10)
        returns[0] = 0

        volatility = np.full(n, np.nan)
        vol_mean = np.full(n, np.nan)

        for i in range(self.window - 1, n):
            window_returns = returns[i - self.window + 1 : i + 1]
            volatility[i] = np.std(window_returns, ddof=1)

        for i in range(2 * self.window - 2, n):
            vol_window = volatility[i - self.window + 1 : i + 1]
            valid = vol_window[~np.isnan(vol_window)]
            if len(valid) > 0:
                vol_mean[i] = np.mean(valid)

        mask = volatility < self.coef * vol_mean
        return pl.Series(values=mask)

    @property
    def warmup(self) -> int:
        return self.window * 2


@dataclass
class HighVolatilityFilter(SignalFilter):
    """Filter signals to only trigger during high volatility periods.

    High volatility defined as: current_vol > coef * mean_vol
    """

    window: int = 60
    coef: float = 1.0

    def apply(self, df: pl.DataFrame) -> pl.Series:
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        close = df["close"].to_numpy()
        n = len(close)

        price = (close + high + low) / 3
        returns = np.diff(price, prepend=price[0]) / (np.roll(price, 1) + 1e-10)
        returns[0] = 0

        volatility = np.full(n, np.nan)
        vol_mean = np.full(n, np.nan)

        for i in range(self.window - 1, n):
            window_returns = returns[i - self.window + 1 : i + 1]
            volatility[i] = np.std(window_returns, ddof=1)

        for i in range(2 * self.window - 2, n):
            vol_window = volatility[i - self.window + 1 : i + 1]
            valid = vol_window[~np.isnan(vol_window)]
            if len(valid) > 0:
                vol_mean[i] = np.mean(valid)

        mask = volatility > self.coef * vol_mean
        return pl.Series(values=mask)

    @property
    def warmup(self) -> int:
        return self.window * 2


@dataclass
class MeanReversionFilter(SignalFilter):
    """Filter signals to only trigger during mean reversion conditions.

    Mean reversion defined as: price / SMA(price) < coef
    """

    window: int = 720
    coef: float = 0.98

    def apply(self, df: pl.DataFrame) -> pl.Series:
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        close = df["close"].to_numpy()
        n = len(close)

        price = (close + high + low) / 3

        price_sma = np.full(n, np.nan)
        for i in range(self.window - 1, n):
            price_sma[i] = np.mean(price[i - self.window + 1 : i + 1])

        price_ratio = price / (price_sma + 1e-10)
        mask = price_ratio < self.coef
        return pl.Series(values=mask)

    @property
    def warmup(self) -> int:
        return self.window


@dataclass
class MeanExtensionFilter(SignalFilter):
    """Filter signals to only trigger during mean extension conditions.

    Mean extension defined as: price / SMA(price) > coef
    """

    window: int = 720
    coef: float = 1.02

    def apply(self, df: pl.DataFrame) -> pl.Series:
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        close = df["close"].to_numpy()
        n = len(close)

        price = (close + high + low) / 3

        price_sma = np.full(n, np.nan)
        for i in range(self.window - 1, n):
            price_sma[i] = np.mean(price[i - self.window + 1 : i + 1])

        price_ratio = price / (price_sma + 1e-10)
        mask = price_ratio > self.coef
        return pl.Series(values=mask)

    @property
    def warmup(self) -> int:
        return self.window


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
class RsiZscoreFilter(SignalFilter):
    """Filter signals based on RSI z-score threshold.

    Computes RSI, then z-score normalizes it, and filters based on threshold.

    Attributes:
        rsi_period: RSI calculation period.
        zscore_window: Rolling window for z-score.
        threshold: Z-score threshold (negative for oversold, positive for overbought).
        condition: "<" for z-score below threshold, ">" for above.
    """

    rsi_period: int = 720
    zscore_window: int = 720
    threshold: float = -0.5
    condition: str = "<"

    def apply(self, df: pl.DataFrame) -> pl.Series:
        close = df["close"].to_numpy()
        n = len(close)

        # Calculate RSI
        diff = np.diff(close, prepend=close[0])
        diff[0] = 0

        gains = np.where(diff > 0, diff, 0)
        losses = np.where(diff < 0, -diff, 0)

        avg_gain = _rma_sma_init(gains, self.rsi_period)
        avg_loss = _rma_sma_init(losses, self.rsi_period)

        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))

        # Calculate z-score
        zscore = np.full(n, np.nan)
        for i in range(self.zscore_window - 1, n):
            window_vals = rsi[i - self.zscore_window + 1 : i + 1]
            valid = window_vals[~np.isnan(window_vals)]
            if len(valid) > 1:
                mean = np.mean(valid)
                std = np.std(valid, ddof=1)
                if std > 1e-10:
                    zscore[i] = (rsi[i] - mean) / std

        if self.condition == "<":
            mask = zscore < self.threshold
        else:
            mask = zscore > self.threshold

        return pl.Series(values=mask)

    @property
    def warmup(self) -> int:
        return max(self.rsi_period * 10, self.zscore_window)


@dataclass
class BelowBBLowerFilter(SignalFilter):
    """Filter signals when price is below Bollinger Band lower band.

    Indicates oversold condition.
    """

    period: int = 720
    std_dev: float = 2.0

    def apply(self, df: pl.DataFrame) -> pl.Series:
        close = df["close"].to_numpy()
        n = len(close)

        middle = np.full(n, np.nan)
        std = np.full(n, np.nan)

        for i in range(self.period - 1, n):
            window = close[i - self.period + 1 : i + 1]
            middle[i] = np.mean(window)
            std[i] = np.std(window, ddof=0)

        lower = middle - self.std_dev * std
        mask = close < lower

        return pl.Series(values=mask)

    @property
    def warmup(self) -> int:
        return self.period


@dataclass
class AboveBBUpperFilter(SignalFilter):
    """Filter signals when price is above Bollinger Band upper band.

    Indicates overbought condition.
    """

    period: int = 720
    std_dev: float = 2.0

    def apply(self, df: pl.DataFrame) -> pl.Series:
        close = df["close"].to_numpy()
        n = len(close)

        middle = np.full(n, np.nan)
        std = np.full(n, np.nan)

        for i in range(self.period - 1, n):
            window = close[i - self.period + 1 : i + 1]
            middle[i] = np.mean(window)
            std[i] = np.std(window, ddof=0)

        upper = middle + self.std_dev * std
        mask = close > upper

        return pl.Series(values=mask)

    @property
    def warmup(self) -> int:
        return self.period


@dataclass
class CciZscoreFilter(SignalFilter):
    """Filter signals based on CCI z-score threshold.

    Computes CCI, then z-score normalizes it, and filters based on threshold.

    Attributes:
        cci_period: CCI calculation period.
        cci_constant: CCI constant (default 0.015).
        zscore_window: Rolling window for z-score.
        threshold: Z-score threshold.
        condition: "<" for z-score below threshold, ">" for above.
    """

    cci_period: int = 180
    cci_constant: float = 0.015
    zscore_window: int = 180
    threshold: float = -1.0
    condition: str = "<"

    def apply(self, df: pl.DataFrame) -> pl.Series:
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        close = df["close"].to_numpy()
        n = len(close)

        tp = (high + low + close) / 3
        cci = np.full(n, np.nan)

        for i in range(self.cci_period - 1, n):
            window = tp[i - self.cci_period + 1 : i + 1]
            sma = np.mean(window)
            mad = np.mean(np.abs(window - sma))
            if mad > 0:
                cci[i] = (tp[i] - sma) / (self.cci_constant * mad)

        # Calculate z-score
        zscore = np.full(n, np.nan)
        for i in range(self.zscore_window - 1, n):
            window_vals = cci[i - self.zscore_window + 1 : i + 1]
            valid = window_vals[~np.isnan(window_vals)]
            if len(valid) > 1:
                mean = np.mean(valid)
                std = np.std(valid, ddof=1)
                if std > 1e-10:
                    zscore[i] = (cci[i] - mean) / std

        if self.condition == "<":
            mask = zscore < self.threshold
        else:
            mask = zscore > self.threshold

        return pl.Series(values=mask)

    @property
    def warmup(self) -> int:
        return max(self.cci_period * 5, self.zscore_window)


@dataclass
class MacdBelowSignalFilter(SignalFilter):
    """Filter signals when MACD line is below signal line.

    Indicates bearish momentum (used for long entry on pullback).
    """

    fast_period: int = 12
    slow_period: int = 26
    signal_period: int = 9

    def apply(self, df: pl.DataFrame) -> pl.Series:
        close = df["close"].to_numpy()
        n = len(close)

        # Calculate EMAs
        fast_ema = np.full(n, np.nan)
        slow_ema = np.full(n, np.nan)

        fast_alpha = 2 / (self.fast_period + 1)
        slow_alpha = 2 / (self.slow_period + 1)

        # Initialize with SMA
        if n >= self.fast_period:
            fast_ema[self.fast_period - 1] = np.mean(close[: self.fast_period])
        if n >= self.slow_period:
            slow_ema[self.slow_period - 1] = np.mean(close[: self.slow_period])

        for i in range(self.fast_period, n):
            fast_ema[i] = fast_alpha * close[i] + (1 - fast_alpha) * fast_ema[i - 1]
        for i in range(self.slow_period, n):
            slow_ema[i] = slow_alpha * close[i] + (1 - slow_alpha) * slow_ema[i - 1]

        macd_line = fast_ema - slow_ema

        # Signal line (EMA of MACD)
        signal_line = np.full(n, np.nan)
        signal_alpha = 2 / (self.signal_period + 1)

        start_idx = self.slow_period + self.signal_period - 2
        if n > start_idx:
            valid_macd = macd_line[self.slow_period - 1 : start_idx + 1]
            valid_macd = valid_macd[~np.isnan(valid_macd)]
            if len(valid_macd) >= self.signal_period:
                signal_line[start_idx] = np.mean(valid_macd[-self.signal_period :])

        for i in range(start_idx + 1, n):
            if not np.isnan(macd_line[i]) and not np.isnan(signal_line[i - 1]):
                signal_line[i] = (
                    signal_alpha * macd_line[i]
                    + (1 - signal_alpha) * signal_line[i - 1]
                )

        mask = macd_line < signal_line
        return pl.Series(values=mask)

    @property
    def warmup(self) -> int:
        return self.slow_period + self.signal_period


@dataclass
class MacdAboveSignalFilter(SignalFilter):
    """Filter signals when MACD line is above signal line.

    Indicates bullish momentum.
    """

    fast_period: int = 12
    slow_period: int = 26
    signal_period: int = 9

    def apply(self, df: pl.DataFrame) -> pl.Series:
        close = df["close"].to_numpy()
        n = len(close)

        # Calculate EMAs
        fast_ema = np.full(n, np.nan)
        slow_ema = np.full(n, np.nan)

        fast_alpha = 2 / (self.fast_period + 1)
        slow_alpha = 2 / (self.slow_period + 1)

        if n >= self.fast_period:
            fast_ema[self.fast_period - 1] = np.mean(close[: self.fast_period])
        if n >= self.slow_period:
            slow_ema[self.slow_period - 1] = np.mean(close[: self.slow_period])

        for i in range(self.fast_period, n):
            fast_ema[i] = fast_alpha * close[i] + (1 - fast_alpha) * fast_ema[i - 1]
        for i in range(self.slow_period, n):
            slow_ema[i] = slow_alpha * close[i] + (1 - slow_alpha) * slow_ema[i - 1]

        macd_line = fast_ema - slow_ema

        signal_line = np.full(n, np.nan)
        signal_alpha = 2 / (self.signal_period + 1)

        start_idx = self.slow_period + self.signal_period - 2
        if n > start_idx:
            valid_macd = macd_line[self.slow_period - 1 : start_idx + 1]
            valid_macd = valid_macd[~np.isnan(valid_macd)]
            if len(valid_macd) >= self.signal_period:
                signal_line[start_idx] = np.mean(valid_macd[-self.signal_period :])

        for i in range(start_idx + 1, n):
            if not np.isnan(macd_line[i]) and not np.isnan(signal_line[i - 1]):
                signal_line[i] = (
                    signal_alpha * macd_line[i]
                    + (1 - signal_alpha) * signal_line[i - 1]
                )

        mask = macd_line > signal_line
        return pl.Series(values=mask)

    @property
    def warmup(self) -> int:
        return self.slow_period + self.signal_period

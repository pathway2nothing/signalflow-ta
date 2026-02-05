"""True Range and ATR-based volatility indicators."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl

from signalflow.core import sf_component
from signalflow.feature.base import Feature
from typing import ClassVar


@dataclass
@sf_component(name="volatility/true_range")
class TrueRangeVol(Feature):
    """True Range.

    Expands classical range (high - low) to include gaps.

    TR = max(high - low, |high - prev_close|, |low - prev_close|)

    Foundation for ATR and many volatility indicators.

    Reference: Welles Wilder, "New Concepts in Technical Trading Systems"
    """

    normalized: bool = False
    norm_period: int | None = None

    requires = ["high", "low", "close"]
    outputs = ["true_range"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        close = df["close"].to_numpy()
        n = len(close)

        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]

        tr = np.maximum(
            high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close))
        )
        tr[0] = high[0] - low[0]

        # Normalization for unbounded output
        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window

            norm_window = self.norm_period or get_norm_window(20)
            tr = normalize_zscore(tr, window=norm_window)

        col_name = self._get_output_name()
        return df.with_columns(pl.Series(name=col_name, values=tr))

    def _get_output_name(self) -> str:
        """Generate output column name with normalization suffix."""
        suffix = "_norm" if self.normalized else ""
        return f"true_range{suffix}"

    test_params: ClassVar[list[dict]] = [
        {},
        {"normalized": True},
    ]

    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable, reproducible output."""
        base_warmup = 20 * 5
        if self.normalized:
            from signalflow.ta._normalization import get_norm_window

            norm_window = self.norm_period or get_norm_window(20)
            return base_warmup + norm_window
        return base_warmup


@dataclass
@sf_component(name="volatility/atr")
class AtrVol(Feature):
    """Average True Range (ATR).

    Smoothed average of True Range.

    ATR = MA(TR, period)

    Most common volatility measure:
    - Position sizing (risk per trade)
    - Stop loss placement
    - Breakout confirmation

    Reference: Welles Wilder, "New Concepts in Technical Trading Systems"
    https://www.investopedia.com/terms/a/atr.asp
    """

    period: int = 14
    ma_type: Literal["rma", "sma", "ema"] = "rma"
    normalized: bool = False
    norm_period: int | None = None

    requires = ["high", "low", "close"]
    outputs = ["atr_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        close = df["close"].to_numpy()
        n = len(close)

        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]

        tr = np.maximum(
            high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close))
        )
        tr[0] = high[0] - low[0]

        atr = np.full(n, np.nan)

        if self.ma_type == "sma":
            for i in range(self.period - 1, n):
                atr[i] = np.mean(tr[i - self.period + 1 : i + 1])
        elif self.ma_type == "ema":
            alpha = 2 / (self.period + 1)
            atr[self.period - 1] = np.mean(tr[: self.period])
            for i in range(self.period, n):
                atr[i] = alpha * tr[i] + (1 - alpha) * atr[i - 1]
        else:
            alpha = 1 / self.period
            atr[self.period - 1] = np.mean(tr[: self.period])
            for i in range(self.period, n):
                atr[i] = alpha * tr[i] + (1 - alpha) * atr[i - 1]

        # Normalization for unbounded output
        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window

            norm_window = self.norm_period or get_norm_window(self.period)
            atr = normalize_zscore(atr, window=norm_window)

        col_name = self._get_output_name()
        return df.with_columns(pl.Series(name=col_name, values=atr))

    def _get_output_name(self) -> str:
        """Generate output column name with normalization suffix."""
        suffix = "_norm" if self.normalized else ""
        return f"atr_{self.period}{suffix}"

    test_params: ClassVar[list[dict]] = [
        {"period": 14, "ma_type": "rma"},
        {"period": 30, "ma_type": "rma"},
        {"period": 60, "ma_type": "ema"},
        {"period": 14, "ma_type": "rma", "normalized": True},
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
@sf_component(name="volatility/natr")
class NatrVol(Feature):
    """Normalized Average True Range (NATR).

    ATR as percentage of price.

    NATR = (ATR / Close) * 100

    Allows comparison across different price levels:
    - Compare volatility of $10 stock vs $1000 stock
    - Time series comparison when price changes significantly

    Reference: https://www.tradingtechnologies.com/help/x-study/technical-indicator-definitions/normalized-average-true-range-natr/
    """

    period: int = 14
    ma_type: Literal["rma", "sma", "ema"] = "rma"
    normalized: bool = False
    norm_period: int | None = None

    requires = ["high", "low", "close"]
    outputs = ["natr_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        close = df["close"].to_numpy()
        n = len(close)

        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]

        tr = np.maximum(
            high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close))
        )
        tr[0] = high[0] - low[0]

        atr = np.full(n, np.nan)

        if self.ma_type == "rma":
            alpha = 1 / self.period
        elif self.ma_type == "ema":
            alpha = 2 / (self.period + 1)
        else:
            alpha = None

        if alpha is not None:
            atr[self.period - 1] = np.mean(tr[: self.period])
            for i in range(self.period, n):
                atr[i] = alpha * tr[i] + (1 - alpha) * atr[i - 1]
        else:
            for i in range(self.period - 1, n):
                atr[i] = np.mean(tr[i - self.period + 1 : i + 1])

        natr = 100 * atr / close

        # Normalization for unbounded output
        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window

            norm_window = self.norm_period or get_norm_window(self.period)
            natr = normalize_zscore(natr, window=norm_window)

        col_name = self._get_output_name()
        return df.with_columns(pl.Series(name=col_name, values=natr))

    def _get_output_name(self) -> str:
        """Generate output column name with normalization suffix."""
        suffix = "_norm" if self.normalized else ""
        return f"natr_{self.period}{suffix}"

    test_params: ClassVar[list[dict]] = [
        {"period": 14, "ma_type": "rma"},
        {"period": 30, "ma_type": "rma"},
        {"period": 60, "ma_type": "ema"},
        {"period": 14, "ma_type": "rma", "normalized": True},
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

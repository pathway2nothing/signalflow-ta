"""True Range and ATR-based volatility indicators."""

from dataclasses import dataclass
from typing import ClassVar, Literal

import numpy as np
import polars as pl

from signalflow.ta._compat import Feature, feature
from signalflow.ta._numba_kernels import ema_sma_init, rma_sma_init, sma_nb


@dataclass
@feature("volatility/true_range")
class TrueRangeVol(Feature):
    """True Range.

    Expands classical range (high - low) to include gaps.

    TR = max(high - low, |high - prev_close|, |low - prev_close|)

    Foundation for ATR and many volatility indicators.

    Reference: Welles Wilder, "New Concepts in Technical Trading Systems"
    """

    normalized: bool = False
    norm_period: int | None = None

    requires: ClassVar[list[str]] = ["high", "low", "close"]
    outputs: ClassVar[list[str]] = ["true_range"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        close = df["close"].to_numpy()
        len(close)

        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]

        tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
        tr[0] = high[0] - low[0]

        if self.normalized:
            from signalflow.ta._normalization import get_norm_window, normalize_zscore

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
@feature("volatility/atr")
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

    requires: ClassVar[list[str]] = ["high", "low", "close"]
    outputs: ClassVar[list[str]] = ["atr_{period}"]

    is_recursive: ClassVar[bool] = True
    warmup_invariant: ClassVar[bool] = True

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        close = df["close"].to_numpy()
        _n = len(close)

        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]

        tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
        tr[0] = high[0] - low[0]

        tr_f = tr.astype(np.float64)
        if self.ma_type == "sma":
            atr = sma_nb(tr_f, self.period)
        elif self.ma_type == "ema":
            atr = ema_sma_init(tr_f, self.period)
        else:
            atr = rma_sma_init(tr_f, self.period)

        if self.normalized:
            from signalflow.ta._normalization import get_norm_window, normalize_zscore

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
@feature("volatility/natr")
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

    requires: ClassVar[list[str]] = ["high", "low", "close"]
    outputs: ClassVar[list[str]] = ["natr_{period}"]

    is_recursive: ClassVar[bool] = True
    warmup_invariant: ClassVar[bool] = True

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        close = df["close"].to_numpy()
        _n = len(close)

        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]

        tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
        tr[0] = high[0] - low[0]

        tr_f = tr.astype(np.float64)
        if self.ma_type == "rma":
            atr = rma_sma_init(tr_f, self.period)
        elif self.ma_type == "ema":
            atr = ema_sma_init(tr_f, self.period)
        else:
            atr = sma_nb(tr_f, self.period)

        natr = 100 * atr / close

        if self.normalized:
            from signalflow.ta._normalization import get_norm_window, normalize_zscore

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

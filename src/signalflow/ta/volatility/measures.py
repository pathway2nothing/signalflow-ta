"""Other volatility metrics."""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import polars as pl

from signalflow.ta._compat import feature
from signalflow.ta._compat import Feature
from signalflow.ta._numba_kernels import (
    historical_vol_kernel as _historical_vol_kernel,
)
from signalflow.ta._numba_kernels import (
    mass_index_kernel as _mass_index_kernel,
)
from signalflow.ta._numba_kernels import (
    rma_sma_init as _rma_sma_init,
)
from signalflow.ta._numba_kernels import (
    rvi_kernel as _rvi_kernel,
)
from signalflow.ta._numba_kernels import (
    ulcer_index_kernel as _ulcer_index_kernel,
)


@dataclass
@feature("volatility/mass_index")
class MassIndexVol(Feature):
    """Mass Index.

    Non-directional volatility indicator for trend reversals.

    HL_EMA1 = EMA(High - Low, fast)
    HL_EMA2 = EMA(HL_EMA1, fast)
    MASSI = SUM(HL_EMA1 / HL_EMA2, slow)

    Interpretation:
    - "Reversal bulge": MASSI rises above 27, then falls below 26.5
    - Signals potential trend reversal regardless of direction

    Reference: Donald Dorsey
    https://school.stockcharts.com/doku.php?id=technical_indicators:mass_index
    """

    fast: int = 9
    slow: int = 25
    normalized: bool = False
    norm_period: int | None = None

    requires: ClassVar[list[str]] = ["high", "low"]
    outputs: ClassVar[list[str]] = ["massi_{fast}_{slow}"]

    is_recursive: ClassVar[bool] = True
    warmup_invariant: ClassVar[bool] = False

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)

        hl_range = high - low
        massi = _mass_index_kernel(hl_range, self.fast, self.slow)

        if self.normalized:
            from signalflow.ta._normalization import get_norm_window, normalize_zscore

            norm_window = self.norm_period or get_norm_window(self.slow)
            massi = normalize_zscore(massi, window=norm_window)

        col_name = self._get_output_name()
        return df.with_columns(pl.Series(name=col_name, values=massi))

    def _get_output_name(self) -> str:
        """Generate output column name with normalization suffix."""
        suffix = "_norm" if self.normalized else ""
        return f"massi_{self.fast}_{self.slow}{suffix}"

    test_params: ClassVar[list[dict]] = [
        {"fast": 9, "slow": 25},
        {"fast": 9, "slow": 50},
        {"fast": 15, "slow": 40},
        {"fast": 9, "slow": 25, "normalized": True},
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
@feature("volatility/ulcer_index")
class UlcerIndexVol(Feature):
    """Ulcer Index.

    Measures downside volatility (drawdown risk).

    Pct_Drawdown = 100 * (Close - Highest_Close) / Highest_Close
    UI = sqrt(mean(Pct_Drawdown^2, period))

    Focus on downside only:
    - Higher values = larger/longer drawdowns
    - Lower values = smoother equity curve
    - Useful for risk-adjusted returns (Martin Ratio = Return / UI)

    Reference: Peter Martin
    https://www.investopedia.com/terms/u/ulcerindex.asp
    """

    period: int = 14
    normalized: bool = False
    norm_period: int | None = None

    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["ulcer_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        close = df["close"].to_numpy().astype(np.float64)

        ui = _ulcer_index_kernel(close, self.period)

        if self.normalized:
            from signalflow.ta._normalization import get_norm_window, normalize_zscore

            norm_window = self.norm_period or get_norm_window(self.period)
            ui = normalize_zscore(ui, window=norm_window)

        col_name = self._get_output_name()
        return df.with_columns(pl.Series(name=col_name, values=ui))

    def _get_output_name(self) -> str:
        """Generate output column name with normalization suffix."""
        suffix = "_norm" if self.normalized else ""
        return f"ulcer_{self.period}{suffix}"

    test_params: ClassVar[list[dict]] = [
        {"period": 14},
        {"period": 30},
        {"period": 60},
        {"period": 14, "normalized": True},
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
@feature("volatility/rvi")
class RviVol(Feature):
    """Relative Volatility Index (RVI).

    Directional volatility using standard deviation.

    UP_STD = StdDev if close > prev_close else 0
    DN_STD = StdDev if close <= prev_close else 0
    RVI = 100 * EMA(UP_STD) / (EMA(UP_STD) + EMA(DN_STD))

    Interpretation:
    - RVI > 50: upward volatility dominates (bullish)
    - RVI < 50: downward volatility dominates (bearish)
    - Confirm RSI signals or use independently

    Reference: Donald Dorsey, Technical Analysis of Stocks & Commodities, 1993
    """

    period: int = 14
    std_period: int = 10
    normalized: bool = False
    norm_period: int | None = None

    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["rvi_{period}"]

    is_recursive: ClassVar[bool] = True
    warmup_invariant: ClassVar[bool] = True

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        close = df["close"].to_numpy().astype(np.float64)

        rvi = _rvi_kernel(close, self.period, self.std_period)

        if self.normalized:
            rvi = rvi / 100

        col_name = self._get_output_name()
        return df.with_columns(pl.Series(name=col_name, values=rvi))

    def _get_output_name(self) -> str:
        """Generate output column name with normalization suffix."""
        suffix = "_norm" if self.normalized else ""
        return f"rvi_{self.period}{suffix}"

    test_params: ClassVar[list[dict]] = [
        {"period": 14, "std_period": 10},
        {"period": 20, "std_period": 14},
        {"period": 30, "std_period": 20},
        {"period": 14, "std_period": 10, "normalized": True},
    ]

    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable, reproducible output."""
        return (self.std_period + self.period) * 5


@dataclass
@feature("volatility/historical_vol")
class HistoricalVol(Feature):
    """Historical Volatility (Close-to-Close).

    Standard deviation of log returns, annualized.

    Log_Ret = ln(Close / Prev_Close)
    HV = StdDev(Log_Ret, period) * sqrt(annualize)

    Standard volatility measure:
    - Comparable to implied volatility
    - Scale: annualized percentage (e.g., 20% = 0.20)

    Note: For more efficient estimators, see stat/realized.py
    (Parkinson, Garman-Klass, Yang-Zhang)
    """

    period: int = 20
    annualize: int = 252
    normalized: bool = False
    norm_period: int | None = None

    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["hv_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        close = df["close"].to_numpy().astype(np.float64)

        hv = _historical_vol_kernel(close, self.period, self.annualize)

        if self.normalized:
            from signalflow.ta._normalization import get_norm_window, normalize_zscore

            norm_window = self.norm_period or get_norm_window(self.period)
            hv = normalize_zscore(hv, window=norm_window)

        col_name = self._get_output_name()
        return df.with_columns(pl.Series(name=col_name, values=hv))

    def _get_output_name(self) -> str:
        """Generate output column name with normalization suffix."""
        suffix = "_norm" if self.normalized else ""
        return f"hv_{self.period}{suffix}"

    test_params: ClassVar[list[dict]] = [
        {"period": 20, "annualize": 252},
        {"period": 30, "annualize": 252},
        {"period": 60, "annualize": 365},
        {"period": 20, "annualize": 252, "normalized": True},
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
@feature("volatility/atr_percent")
class AtrPercentVol(Feature):
    """ATR as Percentage of Price.

    Similar to NATR but with configurable MA type.

    ATR% = ATR / Close * 100

    Useful for:
    - Cross-asset volatility comparison
    - Relative position sizing
    - Volatility regime detection
    """

    period: int = 14
    normalized: bool = False
    norm_period: int | None = None

    requires: ClassVar[list[str]] = ["high", "low", "close"]
    outputs: ClassVar[list[str]] = ["atr_pct_{period}"]

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

        atr = _rma_sma_init(tr.astype(np.float64), self.period)

        atr_pct = 100 * atr / close

        if self.normalized:
            from signalflow.ta._normalization import get_norm_window, normalize_zscore

            norm_window = self.norm_period or get_norm_window(self.period)
            atr_pct = normalize_zscore(atr_pct, window=norm_window)

        col_name = self._get_output_name()
        return df.with_columns(pl.Series(name=col_name, values=atr_pct))

    def _get_output_name(self) -> str:
        """Generate output column name with normalization suffix."""
        suffix = "_norm" if self.normalized else ""
        return f"atr_pct_{self.period}{suffix}"

    test_params: ClassVar[list[dict]] = [
        {"period": 14},
        {"period": 30},
        {"period": 60},
        {"period": 14, "normalized": True},
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

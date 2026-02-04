"""Other volatility metrics."""
from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl

from signalflow.core import sf_component
from signalflow.feature.base import Feature
from typing import ClassVar


@dataclass
@sf_component(name="volatility/mass_index")
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

    requires = ["high", "low"]
    outputs = ["massi_{fast}_{slow}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        n = len(high)
        
        hl_range = high - low
        
        alpha = 2 / (self.fast + 1)
        
        ema1 = np.full(n, np.nan)
        ema2 = np.full(n, np.nan)
        
        ema1[0] = hl_range[0]
        ema2[0] = hl_range[0]
        
        for i in range(1, n):
            ema1[i] = alpha * hl_range[i] + (1 - alpha) * ema1[i - 1]
            ema2[i] = alpha * ema1[i] + (1 - alpha) * ema2[i - 1]
        
        ratio = ema1 / (ema2 + 1e-10)
        
        massi = np.full(n, np.nan)
        for i in range(self.slow - 1, n):
            massi[i] = np.sum(ratio[i - self.slow + 1:i + 1])

        # Normalization for unbounded output
        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window
            norm_window = self.norm_period or get_norm_window(self.slow)
            massi = normalize_zscore(massi, window=norm_window)

        col_name = self._get_output_name()
        return df.with_columns(
            pl.Series(name=col_name, values=massi)
        )

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
@sf_component(name="volatility/ulcer_index")
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

    requires = ["close"]
    outputs = ["ulcer_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        close = df["close"].to_numpy()
        n = len(close)
        
        ui = np.full(n, np.nan)
        
        for i in range(self.period - 1, n):
            window = close[i - self.period + 1:i + 1]
            highest = np.maximum.accumulate(window)
            pct_dd = 100 * (window - highest) / highest
            ui[i] = np.sqrt(np.mean(pct_dd ** 2))

        # Normalization for unbounded output
        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window
            norm_window = self.norm_period or get_norm_window(self.period)
            ui = normalize_zscore(ui, window=norm_window)

        col_name = self._get_output_name()
        return df.with_columns(
            pl.Series(name=col_name, values=ui)
        )

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
@sf_component(name="volatility/rvi")
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

    requires = ["close"]
    outputs = ["rvi_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        close = df["close"].to_numpy()
        n = len(close)
        
        std = np.full(n, np.nan)
        for i in range(self.std_period - 1, n):
            std[i] = np.std(close[i - self.std_period + 1:i + 1], ddof=1)
        
        diff = np.diff(close, prepend=close[0])
        
        up_std = np.where(diff > 0, std, 0)
        dn_std = np.where(diff <= 0, std, 0)
        
        alpha = 2 / (self.period + 1)
        
        up_ema = np.full(n, np.nan)
        dn_ema = np.full(n, np.nan)

        # Initialize with SMA for reproducibility
        init_idx = self.std_period + self.period - 2

        if n > init_idx:
            # Use SMA of first `period` valid std values
            start_std = self.std_period - 1
            end_init = min(start_std + self.period, n)
            up_ema[init_idx] = np.nanmean(up_std[start_std:end_init])
            dn_ema[init_idx] = np.nanmean(dn_std[start_std:end_init])
        
        for i in range(init_idx + 1, n):
            up_ema[i] = alpha * up_std[i] + (1 - alpha) * up_ema[i - 1]
            dn_ema[i] = alpha * dn_std[i] + (1 - alpha) * dn_ema[i - 1]
        
        rvi = 100 * up_ema / (up_ema + dn_ema + 1e-10)

        # Normalization for bounded output [0,100] → [0,1]
        if self.normalized:
            rvi = rvi / 100

        col_name = self._get_output_name()
        return df.with_columns(
            pl.Series(name=col_name, values=rvi)
        )

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
@sf_component(name="volatility/historical_vol")
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
    annualize: int = 252  # trading days
    normalized: bool = False
    norm_period: int | None = None

    requires = ["close"]
    outputs = ["hv_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        close = df["close"].to_numpy()
        n = len(close)
        
        log_ret = np.log(close / np.roll(close, 1))
        log_ret[0] = 0
        
        hv = np.full(n, np.nan)
        
        for i in range(self.period - 1, n):
            hv[i] = np.std(log_ret[i - self.period + 1:i + 1], ddof=1) * np.sqrt(self.annualize)

        # Normalization for unbounded output
        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window
            norm_window = self.norm_period or get_norm_window(self.period)
            hv = normalize_zscore(hv, window=norm_window)

        col_name = self._get_output_name()
        return df.with_columns(
            pl.Series(name=col_name, values=hv)
        )

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
@sf_component(name="volatility/atr_percent")
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

    requires = ["high", "low", "close"]
    outputs = ["atr_pct_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        close = df["close"].to_numpy()
        n = len(close)
        
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        
        tr = np.maximum(
            high - low,
            np.maximum(
                np.abs(high - prev_close),
                np.abs(low - prev_close)
            )
        )
        tr[0] = high[0] - low[0]
        
        alpha = 1 / self.period
        atr = np.full(n, np.nan)
        atr[self.period - 1] = np.mean(tr[:self.period])
        for i in range(self.period, n):
            atr[i] = alpha * tr[i] + (1 - alpha) * atr[i - 1]
        
        atr_pct = 100 * atr / close

        # Normalization for unbounded output
        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window
            norm_window = self.norm_period or get_norm_window(self.period)
            atr_pct = normalize_zscore(atr_pct, window=norm_window)

        col_name = self._get_output_name()
        return df.with_columns(
            pl.Series(name=col_name, values=atr_pct)
        )

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

"""Core momentum indicators with reproducible initialization."""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import polars as pl

from signalflow.core import feature
from signalflow.feature.base import Feature


from signalflow.ta._numba_kernels import rma_sma_init as _rma_sma_init
from signalflow.ta._numba_kernels import cmo_kernel as _cmo_kernel


@dataclass
@feature("momentum/rsi")
class RsiMom(Feature):
    """Relative Strength Index (RSI) with reproducible initialization.

    Momentum oscillator measuring speed and magnitude of price changes.

    RSI = 100 * avg_gain / (avg_gain + avg_loss)

    Where avg_gain/avg_loss use Wilder's smoothing (RMA) with SMA initialization.
    This ensures reproducibility regardless of data entry point.

    Interpretation (absolute mode):
    - RSI > 70: overbought
    - RSI < 30: oversold

    Interpretation (normalized mode):
    - RSI > 0.7: overbought
    - RSI < 0.3: oversold

    Reference: J. Welles Wilder, "New Concepts in Technical Trading Systems"
    """

    period: int = 14
    normalized: bool = False

    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["rsi_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        close = df["close"].to_numpy()
        len(close)

        # Price changes
        diff = np.diff(close, prepend=close[0])
        diff[0] = 0

        gains = np.where(diff > 0, diff, 0)
        losses = np.where(diff < 0, -diff, 0)

        # RMA with SMA initialization for reproducibility
        avg_gain = _rma_sma_init(gains, self.period)
        avg_loss = _rma_sma_init(losses, self.period)

        # RSI calculation
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))

        # Normalization: [0, 100] → [0, 1]
        if self.normalized:
            rsi = rsi / 100

        col_name = self._get_output_name()
        return df.with_columns(pl.Series(name=col_name, values=rsi))

    def _get_output_name(self) -> str:
        """Generate output column name with normalization suffix."""
        suffix = "_norm" if self.normalized else ""
        return f"rsi_{self.period}{suffix}"

    test_params: ClassVar[list[dict]] = [
        {"period": 14},
        {"period": 14, "normalized": True},
        {"period": 60},
        {"period": 240},
    ]

    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable, reproducible output."""
        return self.period * 10


@dataclass
@feature("momentum/roc")
class RocMom(Feature):
    """Rate of Change (ROC).

    Percentage change over n periods. Pure lookback, always reproducible.

    ROC = 100 * (close - close[n]) / close[n]

    Unbounded oscillator. In normalized mode, uses rolling z-score.

    Reference: https://www.investopedia.com/terms/r/rateofchange.asp
    """

    period: int = 10
    normalized: bool = False
    norm_period: int | None = None

    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["roc_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        close = df["close"].to_numpy()
        n = len(close)

        roc = np.full(n, np.nan)

        for i in range(self.period, n):
            if close[i - self.period] != 0:
                roc[i] = 100 * (close[i] - close[i - self.period]) / close[i - self.period]

        # Normalization: z-score for unbounded oscillator
        if self.normalized:
            from signalflow.ta._normalization import get_norm_window, normalize_zscore

            norm_window = self.norm_period or get_norm_window(self.period)
            roc = normalize_zscore(roc, window=norm_window)

        col_name = self._get_output_name()
        return df.with_columns(pl.Series(name=col_name, values=roc))

    def _get_output_name(self) -> str:
        """Generate output column name with normalization suffix."""
        suffix = "_norm" if self.normalized else ""
        return f"roc_{self.period}{suffix}"

    test_params: ClassVar[list[dict]] = [
        {"period": 14},
        {"period": 14, "normalized": True},
        {"period": 60},
        {"period": 240},
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
@feature("momentum/mom")
class MomMom(Feature):
    """Momentum (MOM).

    Simple price difference over n periods. Pure lookback, always reproducible.

    MOM = close - close[n]

    Unbounded oscillator. In normalized mode, uses rolling z-score.

    Reference: https://www.investopedia.com/terms/m/momentum.asp
    """

    period: int = 10
    normalized: bool = False
    norm_period: int | None = None

    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["mom_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        close = df["close"].to_numpy()
        n = len(close)

        mom = np.full(n, np.nan)

        for i in range(self.period, n):
            mom[i] = close[i] - close[i - self.period]

        # Normalization: z-score for unbounded oscillator
        if self.normalized:
            from signalflow.ta._normalization import get_norm_window, normalize_zscore

            norm_window = self.norm_period or get_norm_window(self.period)
            mom = normalize_zscore(mom, window=norm_window)

        col_name = self._get_output_name()
        return df.with_columns(pl.Series(name=col_name, values=mom))

    def _get_output_name(self) -> str:
        """Generate output column name with normalization suffix."""
        suffix = "_norm" if self.normalized else ""
        return f"mom_{self.period}{suffix}"

    test_params: ClassVar[list[dict]] = [
        {"period": 14},
        {"period": 14, "normalized": True},
        {"period": 60},
        {"period": 240},
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
@feature("momentum/cmo")
class CmoMom(Feature):
    """Chande Momentum Oscillator (CMO).

    Uses rolling sums, not EMA - always reproducible.

    CMO = 100 * (sum_gains - sum_losses) / (sum_gains + sum_losses)

    Bounded -100 to +100 (absolute) or -1 to +1 (normalized).

    Reference: Tushar Chande
    """

    period: int = 14
    normalized: bool = False

    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["cmo_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        close = df["close"].to_numpy()
        n = len(close)

        diff = np.diff(close, prepend=close[0])
        diff[0] = 0

        gains = np.where(diff > 0, diff, 0)
        losses = np.where(diff < 0, -diff, 0)

        cmo = _cmo_kernel(gains, losses, self.period)

        # Normalization: [-100, 100] → [-1, 1]
        if self.normalized:
            cmo = cmo / 100

        col_name = self._get_output_name()
        return df.with_columns(pl.Series(name=col_name, values=cmo))

    def _get_output_name(self) -> str:
        """Generate output column name with normalization suffix."""
        suffix = "_norm" if self.normalized else ""
        return f"cmo_{self.period}{suffix}"

    test_params: ClassVar[list[dict]] = [
        {"period": 14},
        {"period": 14, "normalized": True},
        {"period": 60},
        {"period": 240},
    ]

    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable, reproducible output."""
        return self.period * 5

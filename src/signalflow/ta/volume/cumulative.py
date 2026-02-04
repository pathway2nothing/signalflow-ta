"""Cumulative volume-price indicators."""
from dataclasses import dataclass

import numpy as np
import polars as pl

from signalflow import sf_component
from signalflow.feature.base import Feature
from typing import ClassVar
from numba import njit


@njit
def rolling_sum_numpy(arr: np.ndarray, window: int) -> np.ndarray:
    """Fast rolling sum using numpy.

    Faster than pandas and doesn't require extra dependencies.
    Uses cumsum trick for O(n) performance.
    """
    n = len(arr)
    result = np.empty(n, dtype=np.float64)
    cumsum = np.cumsum(arr)
    result[:window] = cumsum[:window]

    if n > window:
        result[window:] = cumsum[window:] - cumsum[:-window]

    return result


@dataclass
@sf_component(name="volume/obv")
class ObvVolume(Feature):
    """On Balance Volume (OBV) - Windowed version for reproducibility.

    Rolling sum of volume-weighted by price direction over a fixed period.
    This windowed approach makes the indicator reproducible from any entry point.

    Calculation:
    - direction = sign(close - prev_close)
    - signed_volume = direction * volume
    - obv = rolling_sum(signed_volume, period)

    Interpretation:
    - Positive OBV: More volume on up days (accumulation)
    - Negative OBV: More volume on down days (distribution)
    - Rising OBV: Increasing buying pressure
    - Falling OBV: Increasing selling pressure

    Note: This is a windowed version (not cumulative) for reproducibility.
    For classic cumulative OBV, use longer periods (100+) to approximate.

    Reference: Joseph Granville, "New Key to Stock Market Profits"
    https://www.investopedia.com/terms/o/onbalancevolume.asp
    """

    period: int = 20  # Rolling window size
    normalized: bool = False
    norm_period: int | None = None

    requires = ["close", "volume"]
    outputs = ["obv"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        close = df["close"].to_numpy()
        volume = df["volume"].to_numpy()

        # Direction: +1 if up, -1 if down, 0 if unchanged
        direction = np.sign(np.diff(close, prepend=np.nan))
        direction[0] = 0

        # Signed volume
        signed_volume = direction * volume

        # Fast rolling sum using numpy
        obv = rolling_sum_numpy(signed_volume, self.period)

        # Normalization: z-score for unbounded oscillator
        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window
            norm_window = self.norm_period or get_norm_window(self.period)
            obv = normalize_zscore(obv, window=norm_window)

        col_name = self._get_output_name()
        return df.with_columns(
            pl.Series(name=col_name, values=obv)
        )

    def _get_output_name(self) -> str:
        """Generate output column name with normalization suffix."""
        suffix = "_norm" if self.normalized else ""
        return f"obv{suffix}"

    @property
    def warmup(self) -> int:
        """Minimum bars for stable output."""
        base_warmup = self.period
        if self.normalized:
            from signalflow.ta._normalization import get_norm_window
            norm_window = self.norm_period or get_norm_window(self.period)
            return base_warmup + norm_window
        return base_warmup

    test_params: ClassVar[list[dict]] = [
        {},
        {"normalized": True},
    ]


@dataclass
@sf_component(name="volume/ad")
class AdVolume(Feature):
    """Accumulation/Distribution Line (A/D) - Windowed version.

    Rolling sum of money flow volume based on close location within range.
    Windowed approach for reproducibility.

    Calculation:
    - CLV = ((Close - Low) - (High - Close)) / (High - Low)
    - MFV = CLV * Volume
    - AD = rolling_sum(MFV, period)

    CLV ranges from -1 to +1:
    - Close at high: CLV = +1 (accumulation)
    - Close at low: CLV = -1 (distribution)
    - Close at midpoint: CLV = 0

    Interpretation:
    - Positive A/D: accumulation (buying pressure)
    - Negative A/D: distribution (selling pressure)
    - Rising A/D: increasing accumulation
    - Falling A/D: increasing distribution

    Reference: Marc Chaikin
    https://school.stockcharts.com/doku.php?id=technical_indicators:accumulation_distribution_line
    """

    period: int = 20  # Rolling window size
    normalized: bool = False
    norm_period: int | None = None

    requires = ["high", "low", "close", "volume"]
    outputs = ["ad"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        close = df["close"].to_numpy()
        volume = df["volume"].to_numpy()

        # Close Location Value
        hl_range = high - low
        clv = np.where(
            hl_range > 0,
            ((close - low) - (high - close)) / hl_range,
            0
        )

        # Money Flow Volume
        mfv = clv * volume

        # Fast rolling sum using numpy
        ad = rolling_sum_numpy(mfv, self.period)

        # Normalization: z-score for unbounded oscillator
        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window
            norm_window = self.norm_period or get_norm_window(self.period)
            ad = normalize_zscore(ad, window=norm_window)

        col_name = self._get_output_name()
        return df.with_columns(
            pl.Series(name=col_name, values=ad)
        )

    def _get_output_name(self) -> str:
        """Generate output column name with normalization suffix."""
        suffix = "_norm" if self.normalized else ""
        return f"ad{suffix}"

    @property
    def warmup(self) -> int:
        """Minimum bars for stable output."""
        base_warmup = self.period
        if self.normalized:
            from signalflow.ta._normalization import get_norm_window
            norm_window = self.norm_period or get_norm_window(self.period)
            return base_warmup + norm_window
        return base_warmup

    test_params: ClassVar[list[dict]] = [
        {},
        {"normalized": True},
    ]


@dataclass
@sf_component(name="volume/pvt")
class PvtVolume(Feature):
    """Price-Volume Trend (PVT) - Windowed version.

    Rolling sum of volume weighted by percentage price change.
    Similar to OBV but uses magnitude of change, not just direction.

    Calculation:
    - ROC = (close - prev_close) / prev_close
    - PV = ROC * volume
    - PVT = rolling_sum(PV, period)

    Interpretation:
    - Positive PVT: volume flowing in with price increases
    - Negative PVT: volume flowing out with price decreases
    - Rising PVT: increasing volume momentum
    - Divergence signals potential reversals

    Reference: https://www.investopedia.com/terms/p/pvtrend.asp
    """

    period: int = 20  # Rolling window size
    normalized: bool = False
    norm_period: int | None = None

    requires = ["close", "volume"]
    outputs = ["pvt"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        close = df["close"].to_numpy()
        volume = df["volume"].to_numpy()

        # Rate of change
        roc = np.diff(close, prepend=np.nan) / np.roll(close, 1)
        roc[0] = 0

        # Price-volume product
        pv = roc * volume

        # Fast rolling sum using numpy
        pvt = rolling_sum_numpy(pv, self.period)

        # Normalization: z-score for unbounded oscillator
        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window
            norm_window = self.norm_period or get_norm_window(self.period)
            pvt = normalize_zscore(pvt, window=norm_window)

        col_name = self._get_output_name()
        return df.with_columns(
            pl.Series(name=col_name, values=pvt)
        )

    def _get_output_name(self) -> str:
        """Generate output column name with normalization suffix."""
        suffix = "_norm" if self.normalized else ""
        return f"pvt{suffix}"

    @property
    def warmup(self) -> int:
        """Minimum bars for stable output."""
        base_warmup = self.period
        if self.normalized:
            from signalflow.ta._normalization import get_norm_window
            norm_window = self.norm_period or get_norm_window(self.period)
            return base_warmup + norm_window
        return base_warmup

    test_params: ClassVar[list[dict]] = [
        {},
        {"normalized": True},
    ]


@dataclass
@sf_component(name="volume/nvi")
class NviVolume(Feature):
    """Negative Volume Index (NVI) - Windowed version.

    Tracks cumulative price changes on low-volume days over a rolling window.

    Calculation:
    - If volume < prev_volume: change = ROC(close) * 100
    - Else: change = 0
    - NVI = rolling_sum(change, period)

    Theory: "Smart money" trades on low-volume days.

    Interpretation:
    - Positive NVI: Smart money accumulating on low-volume days
    - Negative NVI: Smart money distributing
    - Rising NVI: Increasing smart money confidence

    Reference: Paul Dysart, Norman Fosback
    https://school.stockcharts.com/doku.php?id=technical_indicators:negative_volume_inde
    """

    period: int = 255  # Standard NVI period
    normalized: bool = False
    norm_period: int | None = None

    requires = ["close", "volume"]
    outputs = ["nvi"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        close = df["close"].to_numpy()
        volume = df["volume"].to_numpy()

        # Rate of change
        roc = np.diff(close, prepend=np.nan) / np.roll(close, 1)
        roc[0] = 0

        # Only on volume-down days
        vol_down = volume < np.roll(volume, 1)
        vol_down[0] = False
        nvi_change = np.where(vol_down, roc * 100, 0)

        # Fast rolling sum using numpy
        nvi = rolling_sum_numpy(nvi_change, self.period)

        # Normalization: z-score for unbounded oscillator
        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window
            norm_window = self.norm_period or get_norm_window(self.period)
            nvi = normalize_zscore(nvi, window=norm_window)

        col_name = self._get_output_name()
        return df.with_columns(
            pl.Series(name=col_name, values=nvi)
        )

    def _get_output_name(self) -> str:
        """Generate output column name with normalization suffix."""
        suffix = "_norm" if self.normalized else ""
        return f"nvi{suffix}"

    @property
    def warmup(self) -> int:
        """Minimum bars for stable output."""
        base_warmup = self.period
        if self.normalized:
            from signalflow.ta._normalization import get_norm_window
            norm_window = self.norm_period or get_norm_window(self.period)
            return base_warmup + norm_window
        return base_warmup

    test_params: ClassVar[list[dict]] = [
        {},
        {"normalized": True},
    ]


@dataclass
@sf_component(name="volume/pvi")
class PviVolume(Feature):
    """Positive Volume Index (PVI) - Windowed version.

    Tracks cumulative price changes on high-volume days over a rolling window.

    Calculation:
    - If volume > prev_volume: change = ROC(close) * 100
    - Else: change = 0
    - PVI = rolling_sum(change, period)

    Opposite of NVI - tracks "uninformed" crowd behavior.

    Interpretation:
    - Positive PVI: Crowd is bullish on high-volume days
    - Negative PVI: Crowd is bearish
    - Use with NVI for complete picture

    Reference: Paul Dysart, Norman Fosback
    https://www.investopedia.com/terms/p/pvi.asp
    """

    period: int = 255  # Standard PVI period
    normalized: bool = False
    norm_period: int | None = None

    requires = ["close", "volume"]
    outputs = ["pvi"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        close = df["close"].to_numpy()
        volume = df["volume"].to_numpy()

        # Rate of change
        roc = np.diff(close, prepend=np.nan) / np.roll(close, 1)
        roc[0] = 0

        # Only on volume-up days
        vol_up = volume > np.roll(volume, 1)
        vol_up[0] = False

        pvi_change = np.where(vol_up, roc * 100, 0)

        # Fast rolling sum using numpy
        pvi = rolling_sum_numpy(pvi_change, self.period)

        # Normalization: z-score for unbounded oscillator
        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window
            norm_window = self.norm_period or get_norm_window(self.period)
            pvi = normalize_zscore(pvi, window=norm_window)

        col_name = self._get_output_name()
        return df.with_columns(
            pl.Series(name=col_name, values=pvi)
        )

    def _get_output_name(self) -> str:
        """Generate output column name with normalization suffix."""
        suffix = "_norm" if self.normalized else ""
        return f"pvi{suffix}"

    @property
    def warmup(self) -> int:
        """Minimum bars for stable output."""
        base_warmup = self.period
        if self.normalized:
            from signalflow.ta._normalization import get_norm_window
            norm_window = self.norm_period or get_norm_window(self.period)
            return base_warmup + norm_window
        return base_warmup

    test_params: ClassVar[list[dict]] = [
        {},
        {"normalized": True},
    ]

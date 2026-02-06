"""Basic smoothing moving averages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal

import numpy as np
import polars as pl

from signalflow import sf_component
from signalflow.feature.base import Feature


@dataclass
@sf_component(name="smooth/sma")
class SmaSmooth(Feature):
    """Simple Moving Average.

    SMA = Σ(close) / n

    Equal weight to all observations in window.
    Most lag, but most stable.

    In normalized mode, returns percentage difference from source:
    normalized = (source - sma) / source

    Reference: https://en.wikipedia.org/wiki/Moving_average#Simple_moving_average
    """

    source_col: str = "close"
    period: int = 20
    normalized: bool = False

    requires = ["{source_col}"]
    outputs = ["{source_col}_sma_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        source = df[self.source_col].to_numpy()
        sma = (
            df.select(pl.col(self.source_col).rolling_mean(window_size=self.period))
            .to_series()
            .to_numpy()
        )

        if self.normalized:
            from signalflow.ta._normalization import normalize_ma_pct

            sma = normalize_ma_pct(source, sma)

        col_name = self._get_output_name()
        return df.with_columns(pl.Series(name=col_name, values=sma))

    def _get_output_name(self) -> str:
        """Generate output column name with normalization suffix."""
        suffix = "_norm" if self.normalized else ""
        return f"{self.source_col}_sma_{self.period}{suffix}"

    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 20},
        {"source_col": "close", "period": 20, "normalized": True},
        {"source_col": "close", "period": 60},
        {"source_col": "close", "period": 240},
    ]

    @property
    def warmup(self) -> int:
        return self.period * 3


@dataclass
@sf_component(name="smooth/ema")
class EmaSmooth(Feature):
    """Exponential Moving Average.

    EMA = α * price + (1 - α) * EMA_prev
    α = 2 / (period + 1)

    More weight to recent prices. Less lag than SMA.

    In normalized mode, returns percentage difference from source:
    normalized = (source - ema) / source

    Reference: https://en.wikipedia.org/wiki/Exponential_smoothing
    """

    source_col: str = "close"
    period: int = 20
    normalized: bool = False

    requires = ["{source_col}"]
    outputs = ["{source_col}_ema_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        source = df[self.source_col].to_numpy()
        ema = (
            df.select(pl.col(self.source_col).ewm_mean(span=self.period, adjust=False))
            .to_series()
            .to_numpy()
        )

        if self.normalized:
            from signalflow.ta._normalization import normalize_ma_pct

            ema = normalize_ma_pct(source, ema)

        col_name = self._get_output_name()
        return df.with_columns(pl.Series(name=col_name, values=ema))

    def _get_output_name(self) -> str:
        """Generate output column name with normalization suffix."""
        suffix = "_norm" if self.normalized else ""
        return f"{self.source_col}_ema_{self.period}{suffix}"

    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 20},
        {"source_col": "close", "period": 20, "normalized": True},
        {"source_col": "close", "period": 60},
        {"source_col": "close", "period": 240},
    ]

    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable, reproducible output."""
        return self.period * 5


@dataclass
@sf_component(name="smooth/wma")
class WmaSmooth(Feature):
    """Weighted Moving Average.

    WMA = Σ(weight_i * price_i) / Σ(weight_i)
    weights = [1, 2, 3, ..., n]

    Linearly increasing weights. Most recent = highest weight.

    In normalized mode, returns percentage difference from source:
    normalized = (source - wma) / source

    Reference: https://en.wikipedia.org/wiki/Moving_average#Weighted_moving_average
    """

    source_col: str = "close"
    period: int = 20
    normalized: bool = False

    requires = ["{source_col}"]
    outputs = ["{source_col}_wma_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        source = df[self.source_col].to_numpy()
        n = len(source)

        weights = np.arange(1, self.period + 1, dtype=np.float64)
        weight_sum = weights.sum()

        wma = np.full(n, np.nan)
        for i in range(self.period - 1, n):
            window = source[i - self.period + 1 : i + 1]
            wma[i] = np.dot(window, weights) / weight_sum

        if self.normalized:
            from signalflow.ta._normalization import normalize_ma_pct

            wma = normalize_ma_pct(source, wma)

        col_name = self._get_output_name()
        return df.with_columns(pl.Series(name=col_name, values=wma))

    def _get_output_name(self) -> str:
        """Generate output column name with normalization suffix."""
        suffix = "_norm" if self.normalized else ""
        return f"{self.source_col}_wma_{self.period}{suffix}"

    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 20},
        {"source_col": "close", "period": 20, "normalized": True},
        {"source_col": "close", "period": 60},
        {"source_col": "close", "period": 240},
    ]

    @property
    def warmup(self) -> int:
        return self.period * 3


@dataclass
@sf_component(name="smooth/rma")
class RmaSmooth(Feature):
    """Wilder's Smoothed Moving Average (RMA).

    In normalized mode, returns percentage difference from source:
    normalized = (source - rma) / source
    """

    source_col: str = "close"
    period: int = 14
    normalized: bool = False

    requires = ["{source_col}"]
    outputs = ["{source_col}_rma_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        source = df[self.source_col].to_numpy()
        equivalent_span = 2 * self.period - 1

        rma = (
            df.select(
                pl.col(self.source_col).ewm_mean(span=equivalent_span, adjust=False)
            )
            .to_series()
            .to_numpy()
        )

        if self.normalized:
            from signalflow.ta._normalization import normalize_ma_pct

            rma = normalize_ma_pct(source, rma)

        col_name = self._get_output_name()
        return df.with_columns(pl.Series(name=col_name, values=rma))

    def _get_output_name(self) -> str:
        """Generate output column name with normalization suffix."""
        suffix = "_norm" if self.normalized else ""
        return f"{self.source_col}_rma_{self.period}{suffix}"

    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 14},
        {"source_col": "close", "period": 14, "normalized": True},
        {"source_col": "close", "period": 60},
        {"source_col": "close", "period": 240},
    ]

    @property
    def warmup(self) -> int:
        """RMA needs ~10x period for initialization error to decay below 0.01%."""
        return self.period * 10


@dataclass
@sf_component(name="smooth/dema")
class DemaSmooth(Feature):
    """Double Exponential Moving Average.

    DEMA = 2 * EMA(price) - EMA(EMA(price))

    Reduces lag by subtracting the "lag" component.

    In normalized mode, returns percentage difference from source:
    normalized = (source - dema) / source

    Reference: https://www.investopedia.com/terms/d/double-exponential-moving-average.asp
    """

    source_col: str = "close"
    period: int = 20
    normalized: bool = False

    requires = ["{source_col}"]
    outputs = ["{source_col}_dema_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        source = df[self.source_col].to_numpy()
        ema1 = df[self.source_col].ewm_mean(span=self.period, adjust=False)
        ema2 = ema1.ewm_mean(span=self.period, adjust=False)

        dema = df.select(2 * ema1 - ema2).to_series().to_numpy()

        if self.normalized:
            from signalflow.ta._normalization import normalize_ma_pct

            dema = normalize_ma_pct(source, dema)

        col_name = self._get_output_name()
        return df.with_columns(pl.Series(name=col_name, values=dema))

    def _get_output_name(self) -> str:
        """Generate output column name with normalization suffix."""
        suffix = "_norm" if self.normalized else ""
        return f"{self.source_col}_dema_{self.period}{suffix}"

    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 20},
        {"source_col": "close", "period": 20, "normalized": True},
        {"source_col": "close", "period": 60},
        {"source_col": "close", "period": 120},
    ]

    @property
    def warmup(self) -> int:
        return self.period * 5


@dataclass
@sf_component(name="smooth/tema")
class TemaSmooth(Feature):
    """Triple Exponential Moving Average.

    TEMA = 3*EMA - 3*EMA(EMA) + EMA(EMA(EMA))

    Even less lag than DEMA. May overshoot in choppy markets.

    In normalized mode, returns percentage difference from source:
    normalized = (source - tema) / source

    Reference: https://www.investopedia.com/terms/t/triple-exponential-moving-average.asp
    """

    source_col: str = "close"
    period: int = 20
    normalized: bool = False

    requires = ["{source_col}"]
    outputs = ["{source_col}_tema_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        source = df[self.source_col].to_numpy()
        ema1 = df[self.source_col].ewm_mean(span=self.period, adjust=False)
        ema2 = ema1.ewm_mean(span=self.period, adjust=False)
        ema3 = ema2.ewm_mean(span=self.period, adjust=False)

        tema = df.select(3 * ema1 - 3 * ema2 + ema3).to_series().to_numpy()

        if self.normalized:
            from signalflow.ta._normalization import normalize_ma_pct

            tema = normalize_ma_pct(source, tema)

        col_name = self._get_output_name()
        return df.with_columns(pl.Series(name=col_name, values=tema))

    def _get_output_name(self) -> str:
        """Generate output column name with normalization suffix."""
        suffix = "_norm" if self.normalized else ""
        return f"{self.source_col}_tema_{self.period}{suffix}"

    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 20},
        {"source_col": "close", "period": 20, "normalized": True},
        {"source_col": "close", "period": 60},
        {"source_col": "close", "period": 120},
    ]

    @property
    def warmup(self) -> int:
        return self.period * 6


@dataclass
@sf_component(name="smooth/hma")
class HmaSmooth(Feature):
    """Hull Moving Average.

    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))

    Attempts to eliminate lag while maintaining smoothness.
    Very responsive to price changes.

    In normalized mode, returns percentage difference from source:
    normalized = (source - hma) / source

    Reference: https://alanhull.com/hull-moving-average
    """

    source_col: str = "close"
    period: int = 20
    normalized: bool = False

    requires = ["{source_col}"]
    outputs = ["{source_col}_hma_{period}"]

    def _wma(self, values: np.ndarray, period: int) -> np.ndarray:
        """Compute WMA."""
        n = len(values)
        weights = np.arange(1, period + 1, dtype=np.float64)
        weight_sum = weights.sum()

        result = np.full(n, np.nan)
        for i in range(period - 1, n):
            window = values[i - period + 1 : i + 1]
            result[i] = np.dot(window, weights) / weight_sum
        return result

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        source = df[self.source_col].to_numpy()

        half_period = int(self.period / 2)
        sqrt_period = int(np.sqrt(self.period))

        wma_half = self._wma(source, half_period)
        wma_full = self._wma(source, self.period)

        raw_hma = 2 * wma_half - wma_full
        hma = self._wma(raw_hma, sqrt_period)

        if self.normalized:
            from signalflow.ta._normalization import normalize_ma_pct

            hma = normalize_ma_pct(source, hma)

        col_name = self._get_output_name()
        return df.with_columns(pl.Series(name=col_name, values=hma))

    def _get_output_name(self) -> str:
        """Generate output column name with normalization suffix."""
        suffix = "_norm" if self.normalized else ""
        return f"{self.source_col}_hma_{self.period}{suffix}"

    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 20},
        {"source_col": "close", "period": 20, "normalized": True},
        {"source_col": "close", "period": 60},
        {"source_col": "close", "period": 144},
    ]

    @property
    def warmup(self) -> int:
        return self.period * 3


@dataclass
@sf_component(name="smooth/trima")
class TrimaSmooth(Feature):
    """Triangular Moving Average.

    TRIMA = SMA(SMA(price, ceil(n/2)), floor(n/2)+1)

    Double-smoothed SMA with triangular weights.
    Very smooth but more lag.

    In normalized mode, returns percentage difference from source:
    normalized = (source - trima) / source

    Reference: https://www.investopedia.com/terms/t/triangularmoving-average.asp
    """

    source_col: str = "close"
    period: int = 20
    normalized: bool = False

    requires = ["{source_col}"]
    outputs = ["{source_col}_trima_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        source = df[self.source_col].to_numpy()
        half = int(np.ceil((self.period + 1) / 2))

        sma1 = pl.col(self.source_col).rolling_mean(window_size=half)
        trima = df.select(sma1.rolling_mean(window_size=half)).to_series().to_numpy()

        # Normalization: percentage difference from source
        if self.normalized:
            from signalflow.ta._normalization import normalize_ma_pct

            trima = normalize_ma_pct(source, trima)

        col_name = self._get_output_name()
        return df.with_columns(pl.Series(name=col_name, values=trima))

    def _get_output_name(self) -> str:
        """Generate output column name with normalization suffix."""
        suffix = "_norm" if self.normalized else ""
        return f"{self.source_col}_trima_{self.period}{suffix}"

    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 20},
        {"source_col": "close", "period": 20, "normalized": True},
        {"source_col": "close", "period": 60},
        {"source_col": "close", "period": 120},
    ]

    @property
    def warmup(self) -> int:
        return self.period * 3


@dataclass
@sf_component(name="smooth/swma")
class SwmaSmooth(Feature):
    """Symmetric Weighted Moving Average.

    Weights form symmetric triangle: [1,2,3,...,n,...,3,2,1]
    Middle values have highest weight.

    In normalized mode, returns percentage difference from source:
    normalized = (source - swma) / source

    Reference: https://www.tradingview.com/pine-script-reference/#fun_swma
    """

    source_col: str = "close"
    period: int = 4
    normalized: bool = False

    requires = ["{source_col}"]
    outputs = ["{source_col}_swma_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        source = df[self.source_col].to_numpy()
        n = len(source)

        # Symmetric triangle weights
        half = (self.period + 1) // 2
        if self.period % 2 == 0:
            weights = np.concatenate([np.arange(1, half + 1), np.arange(half, 0, -1)])
        else:
            weights = np.concatenate(
                [np.arange(1, half + 1), np.arange(half - 1, 0, -1)]
            )

        weights = weights[: self.period].astype(np.float64)
        weight_sum = weights.sum()

        swma = np.full(n, np.nan)
        for i in range(self.period - 1, n):
            window = source[i - self.period + 1 : i + 1]
            swma[i] = np.dot(window, weights) / weight_sum

        # Normalization: percentage difference from source
        if self.normalized:
            from signalflow.ta._normalization import normalize_ma_pct

            swma = normalize_ma_pct(source, swma)

        col_name = self._get_output_name()
        return df.with_columns(pl.Series(name=col_name, values=swma))

    def _get_output_name(self) -> str:
        """Generate output column name with normalization suffix."""
        suffix = "_norm" if self.normalized else ""
        return f"{self.source_col}_swma_{self.period}{suffix}"

    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 4},
        {"source_col": "close", "period": 4, "normalized": True},
        {"source_col": "close", "period": 10},
        {"source_col": "close", "period": 20},
    ]

    @property
    def warmup(self) -> int:
        return self.period * 3


@dataclass
@sf_component(name="smooth/ssf")
class SsfSmooth(Feature):
    """Ehler's Super Smoother Filter.

    Digital filter designed to reduce aliasing noise.
    Provides smooth output with minimal lag.

    poles=2: Standard 2-pole Butterworth filter
    poles=3: 3-pole filter, even smoother

    In normalized mode, returns percentage difference from source:
    normalized = (source - ssf) / source

    Reference: Ehlers, J. F. "Cybernetic Analysis for Stocks and Futures"
    """

    source_col: str = "close"
    period: int = 10
    poles: Literal[2, 3] = 2
    normalized: bool = False

    requires = ["{source_col}"]
    outputs = ["{source_col}_ssf_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        source = df[self.source_col].to_numpy()
        n = len(source)
        ssf = source.copy().astype(np.float64)

        if self.poles == 3:
            x = np.pi / self.period
            a0 = np.exp(-x)
            b0 = 2 * a0 * np.cos(np.sqrt(3) * x)
            c0 = a0 * a0

            c4 = c0 * c0
            c3 = -c0 * (1 + b0)
            c2 = c0 + b0
            c1 = 1 - c2 - c3 - c4

            for i in range(3, n):
                ssf[i] = (
                    c1 * source[i] + c2 * ssf[i - 1] + c3 * ssf[i - 2] + c4 * ssf[i - 3]
                )
        else:  # poles == 2
            x = np.pi * np.sqrt(2) / self.period
            a0 = np.exp(-x)
            a1 = -a0 * a0
            b1 = 2 * a0 * np.cos(x)
            c1 = 1 - a1 - b1

            for i in range(2, n):
                ssf[i] = c1 * source[i] + b1 * ssf[i - 1] + a1 * ssf[i - 2]

        ssf[: self.period - 1] = np.nan

        # Normalization: percentage difference from source
        if self.normalized:
            from signalflow.ta._normalization import normalize_ma_pct

            ssf = normalize_ma_pct(source, ssf)

        col_name = self._get_output_name()
        return df.with_columns(pl.Series(name=col_name, values=ssf))

    def _get_output_name(self) -> str:
        """Generate output column name with normalization suffix."""
        suffix = "_norm" if self.normalized else ""
        return f"{self.source_col}_ssf_{self.period}{suffix}"

    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 10, "poles": 2},
        {"source_col": "close", "period": 10, "poles": 2, "normalized": True},
        {"source_col": "close", "period": 30, "poles": 2},
        {"source_col": "close", "period": 60, "poles": 3},
    ]

    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable, reproducible output."""
        return self.period * 3


@dataclass
@sf_component(name="smooth/fft")
class FftSmooth(Feature):
    """FFT Low-Pass Smoother.

    Applies a frequency-domain low-pass filter to remove high-frequency noise.
    For each rolling window the algorithm:

    1. Removes linear trend (avoids spectral leakage at DC).
    2. Computes the real FFT of the detrended window.
    3. Zeros out frequency bins above ``cutoff_ratio * f_nyquist``.
    4. Reconstructs the signal via inverse FFT.
    5. Adds the trend back and takes the last sample as output.

    The computation is **fully deterministic** — no random state, no
    data-dependent initialization — so the output at bar *N* depends only
    on bars [N - period + 1 … N] and is independent of the entry point.

    Parameters:
        source_col: Price column to smooth.
        period: Rolling window size (power-of-2 recommended for FFT speed).
        cutoff_ratio: Fraction of frequency bins to keep, in (0, 1].
            Lower values → heavier smoothing. Default 0.1 keeps the
            lowest 10 % of non-DC frequencies.
        normalized: If True, output percentage difference from source:
            ``(source - fft_smooth) / source``.

    Reference: Oppenheim, A.V. & Schafer, R.W. "Discrete-Time Signal
    Processing", 3rd ed., Pearson, 2010.
    """

    source_col: str = "close"
    period: int = 64
    cutoff_ratio: float = 0.1
    normalized: bool = False

    requires = ["{source_col}"]
    outputs = ["{source_col}_fft_{period}"]

    def __post_init__(self) -> None:
        if not (0.0 < self.cutoff_ratio <= 1.0):
            raise ValueError(f"cutoff_ratio must be in (0, 1], got {self.cutoff_ratio}")
        if self.period < 4:
            raise ValueError(
                f"period must be >= 4 for meaningful FFT, got {self.period}"
            )

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        source = df[self.source_col].to_numpy().astype(np.float64)
        n = len(source)

        result = np.full(n, np.nan)

        # Number of rfft bins (including DC)
        n_bins = self.period // 2 + 1
        # How many non-DC bins to keep (at least 1)
        keep = max(1, int(round((n_bins - 1) * self.cutoff_ratio)))

        for i in range(self.period - 1, n):
            window = source[i - self.period + 1 : i + 1]

            # --- detrend (linear) ---
            x = np.arange(self.period, dtype=np.float64)
            coeffs = np.polyfit(x, window, 1)
            trend_line = np.polyval(coeffs, x)
            detrended = window - trend_line

            # --- forward FFT ---
            spectrum = np.fft.rfft(detrended)

            # --- low-pass: zero out high-frequency bins ---
            # spectrum[0] = DC, spectrum[1:keep+1] = kept, rest = zeroed
            spectrum[keep + 1 :] = 0.0

            # --- inverse FFT ---
            smoothed = np.fft.irfft(spectrum, n=self.period)

            # --- add trend back, take last sample ---
            result[i] = smoothed[-1] + trend_line[-1]

        if self.normalized:
            from signalflow.ta._normalization import normalize_ma_pct

            result = normalize_ma_pct(source, result)

        col_name = self._get_output_name()
        return df.with_columns(pl.Series(name=col_name, values=result))

    def _get_output_name(self) -> str:
        suffix = "_norm" if self.normalized else ""
        return f"{self.source_col}_fft_{self.period}{suffix}"

    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 64, "cutoff_ratio": 0.1},
        {"source_col": "close", "period": 64, "cutoff_ratio": 0.1, "normalized": True},
        {"source_col": "close", "period": 128, "cutoff_ratio": 0.05},
        {"source_col": "close", "period": 32, "cutoff_ratio": 0.2},
    ]

    @property
    def warmup(self) -> int:
        return self.period * 3

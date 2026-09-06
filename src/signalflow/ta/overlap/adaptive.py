"""Adaptive smoothing algorithms that adjust to market conditions."""

from dataclasses import dataclass
from typing import ClassVar, Literal

import numpy as np
import polars as pl

from signalflow.ta._compat import Feature, feature
from signalflow.ta._numba_compat import njit
from signalflow.ta._numba_kernels import (
    ema_sma_init as _ema_sma_init,
)
from signalflow.ta._numba_kernels import (
    frama_kernel as _frama_kernel,
)
from signalflow.ta._numba_kernels import (
    jma_kernel as _jma_kernel,
)
from signalflow.ta._numba_kernels import (
    kama_kernel as _kama_kernel,
)
from signalflow.ta._numba_kernels import (
    mcginley_kernel as _mcginley_kernel,
)
from signalflow.ta._numba_kernels import (
    vidya_kernel as _vidya_kernel,
)


@dataclass
@feature("smooth/kama")
class KamaSmooth(Feature):
    """Kaufman's Adaptive Moving Average.

    Adapts smoothing based on efficiency ratio (trend vs noise).

    ER = |price_change| / Σ|price_changes|
    SC = (ER * (fast - slow) + slow)²
    KAMA = SC * price + (1 - SC) * KAMA_prev

    Trending: fast response. Ranging: slow response.

    In normalized mode, returns percentage difference from source:
    normalized = (source - kama) / source

    Reference: Kaufman, P. "Trading Systems and Methods"
    """

    source_col: str = "close"
    period: int = 10
    fast: int = 2
    slow: int = 30
    normalized: bool = False

    requires: ClassVar[list[str]] = ["{source_col}"]
    outputs: ClassVar[list[str]] = ["{source_col}_kama_{period}"]

    is_recursive: ClassVar[bool] = True
    warmup_invariant: ClassVar[bool] = True

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        source = df[self.source_col].to_numpy().astype(np.float64)

        fast_sc = 2.0 / (self.fast + 1)
        slow_sc = 2.0 / (self.slow + 1)

        kama = _kama_kernel(source, self.period, fast_sc, slow_sc)

        if self.normalized:
            from signalflow.ta._normalization import normalize_ma_pct

            kama = normalize_ma_pct(source, kama)

        col_name = self._get_output_name()
        return df.with_columns(pl.Series(name=col_name, values=kama))

    def _get_output_name(self) -> str:
        """Generate output column name with normalization suffix."""
        suffix = "_norm" if self.normalized else ""
        return f"{self.source_col}_kama_{self.period}{suffix}"

    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 10, "fast": 2, "slow": 30},
        {
            "source_col": "close",
            "period": 10,
            "fast": 2,
            "slow": 30,
            "normalized": True,
        },
        {"source_col": "close", "period": 60, "fast": 5, "slow": 120},
        {"source_col": "close", "period": 120, "fast": 10, "slow": 240},
    ]

    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable, reproducible output."""
        return self.period * 100


@dataclass
@feature("smooth/alma")
class AlmaSmooth(Feature):
    """Arnaud Legoux Moving Average.

    Uses Gaussian distribution for weights.
    offset controls responsiveness (0=smooth, 1=responsive).
    sigma controls shape of the curve.

    In normalized mode, returns percentage difference from source:
    normalized = (source - ma) / source

    Reference: https://www.prorealcode.com/prorealtime-indicators/alma-arnaud-legoux-moving-average/
    """

    source_col: str = "close"
    period: int = 10
    offset: float = 0.85
    sigma: float = 6.0
    normalized: bool = False

    requires: ClassVar[list[str]] = ["{source_col}"]
    outputs: ClassVar[list[str]] = ["{source_col}_alma_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        source = df[self.source_col].to_numpy()
        n = len(source)

        m = self.offset * (self.period - 1)
        s = self.period / self.sigma

        weights = np.array([np.exp(-((i - m) ** 2) / (2 * s * s)) for i in range(self.period)])

        alma = np.full(n, np.nan)
        for i in range(self.period - 1, n):
            window = source[i - self.period + 1 : i + 1]
            alma[i] = np.dot(window, weights) / weights.sum()

        if self.normalized:
            from signalflow.ta._normalization import normalize_ma_pct

            alma = normalize_ma_pct(source, alma)

        col_name = self._get_output_name()
        return df.with_columns(pl.Series(name=col_name, values=alma))

    def _get_output_name(self) -> str:
        """Generate output column name with normalization suffix."""
        suffix = "_norm" if self.normalized else ""
        return f"{self.source_col}_alma_{self.period}{suffix}"

    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 10, "offset": 0.85, "sigma": 6.0},
        {
            "source_col": "close",
            "period": 10,
            "offset": 0.85,
            "sigma": 6.0,
            "normalized": True,
        },
        {"source_col": "close", "period": 60, "offset": 0.85, "sigma": 6.0},
        {"source_col": "close", "period": 120, "offset": 0.85, "sigma": 6.0},
    ]

    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable, reproducible output."""
        return self.period * 3


@dataclass
@feature("smooth/jma")
class JmaSmooth(Feature):
    """Jurik Moving Average.

    Proprietary adaptive MA with extremely low lag.
    Uses volatility bands and dynamic smoothing.

    phase: -100 to +100 (controls overshoot)

    In normalized mode, returns percentage difference from source:
    normalized = (source - ma) / source

    Reference: https://c.mql5.com/forextsd/forum/164/jurik_1.pdf
    """

    source_col: str = "close"
    period: int = 7
    phase: float = 0
    normalized: bool = False

    requires: ClassVar[list[str]] = ["{source_col}"]
    outputs: ClassVar[list[str]] = ["{source_col}_jma_{period}"]

    is_recursive: ClassVar[bool] = True
    warmup_invariant: ClassVar[bool] = True

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        source = df[self.source_col].to_numpy().astype(np.float64)

        jma = _jma_kernel(source, self.period, float(self.phase))

        if self.normalized:
            from signalflow.ta._normalization import normalize_ma_pct

            jma = normalize_ma_pct(source, jma)

        col_name = self._get_output_name()
        return df.with_columns(pl.Series(name=col_name, values=jma))

    def _get_output_name(self) -> str:
        """Generate output column name with normalization suffix."""
        suffix = "_norm" if self.normalized else ""
        return f"{self.source_col}_jma_{self.period}{suffix}"

    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 7, "phase": 0},
        {"source_col": "close", "period": 7, "phase": 0, "normalized": True},
        {"source_col": "close", "period": 30, "phase": 0},
        {"source_col": "close", "period": 60, "phase": 50},
    ]

    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable, reproducible output."""
        return self.period * 10 + 65


@dataclass
@feature("smooth/vidya")
class VidyaSmooth(Feature):
    """Variable Index Dynamic Average.

    Adapts based on Chande Momentum Oscillator (CMO).
    High volatility = fast, Low volatility = slow.

    VIDYA = alpha * |CMO| * price + (1 - alpha * |CMO|) * VIDYA_prev

    In normalized mode, returns percentage difference from source:
    normalized = (source - ma) / source

    Reference: Chande, T. "The New Technical Trader"
    """

    source_col: str = "close"
    period: int = 14
    normalized: bool = False

    requires: ClassVar[list[str]] = ["{source_col}"]
    outputs: ClassVar[list[str]] = ["{source_col}_vidya_{period}"]

    is_recursive: ClassVar[bool] = True
    warmup_invariant: ClassVar[bool] = True

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        source = df[self.source_col].to_numpy().astype(np.float64)

        alpha = 2.0 / (self.period + 1)

        mom = np.diff(source, prepend=np.nan)
        pos = np.where(mom > 0, mom, 0).astype(np.float64)
        neg = np.where(mom < 0, -mom, 0).astype(np.float64)

        vidya = _vidya_kernel(source, pos, neg, self.period, alpha)

        if self.normalized:
            from signalflow.ta._normalization import normalize_ma_pct

            vidya = normalize_ma_pct(source, vidya)

        col_name = self._get_output_name()
        return df.with_columns(pl.Series(name=col_name, values=vidya))

    def _get_output_name(self) -> str:
        """Generate output column name with normalization suffix."""
        suffix = "_norm" if self.normalized else ""
        return f"{self.source_col}_vidya_{self.period}{suffix}"

    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 14},
        {"source_col": "close", "period": 14, "normalized": True},
        {"source_col": "close", "period": 60},
        {"source_col": "close", "period": 120},
    ]

    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable, reproducible output."""
        return self.period * 5


@dataclass
@feature("smooth/t3")
class T3Smooth(Feature):
    """Tillson T3 Moving Average.

    Smoother and more responsive than TEMA.
    Uses volume factor 'a' to control smoothing.

    T3 = c1*e6 + c2*e5 + c3*e4 + c4*e3
    where e1..e6 are cascaded EMAs

    In normalized mode, returns percentage difference from source:
    normalized = (source - ma) / source

    Reference: Tillson, T. "Technical Analysis of Stocks & Commodities"
    """

    source_col: str = "close"
    period: int = 10
    vfactor: float = 0.7
    normalized: bool = False

    requires: ClassVar[list[str]] = ["{source_col}"]
    outputs: ClassVar[list[str]] = ["{source_col}_t3_{period}"]

    is_recursive: ClassVar[bool] = True
    warmup_invariant: ClassVar[bool] = True

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        source = df[self.source_col].to_numpy()

        a = self.vfactor
        c1 = -(a**3)
        c2 = 3 * a**2 + 3 * a**3
        c3 = -6 * a**2 - 3 * a - 3 * a**3
        c4 = 1 + 3 * a + a**3 + 3 * a**2

        e1 = _ema_sma_init(source, self.period)
        e2 = _ema_sma_init(e1, self.period)
        e3 = _ema_sma_init(e2, self.period)
        e4 = _ema_sma_init(e3, self.period)
        e5 = _ema_sma_init(e4, self.period)
        e6 = _ema_sma_init(e5, self.period)

        t3 = c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3

        if self.normalized:
            from signalflow.ta._normalization import normalize_ma_pct

            t3 = normalize_ma_pct(source, t3)

        col_name = self._get_output_name()
        return df.with_columns(pl.Series(name=col_name, values=t3))

    def _get_output_name(self) -> str:
        """Generate output column name with normalization suffix."""
        suffix = "_norm" if self.normalized else ""
        return f"{self.source_col}_t3_{self.period}{suffix}"

    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 10, "vfactor": 0.7},
        {"source_col": "close", "period": 10, "vfactor": 0.7, "normalized": True},
        {"source_col": "close", "period": 30, "vfactor": 0.7},
        {"source_col": "close", "period": 60, "vfactor": 0.8},
    ]

    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable, reproducible output."""
        return self.period * 6


@dataclass
@feature("smooth/zlma")
class ZlmaSmooth(Feature):
    """Zero Lag Moving Average.

    Reduces lag by adjusting price before smoothing.

    adjusted_price = 2 * price - price.shift(lag)
    ZLMA = EMA(adjusted_price)
    lag = (period - 1) / 2

    In normalized mode, returns percentage difference from source:
    normalized = (source - ma) / source

    Reference: Ehlers & Way, "Zero Lag (Well, Almost)"
    """

    source_col: str = "close"
    period: int = 20
    ma_type: Literal["ema", "sma"] = "ema"
    normalized: bool = False

    requires: ClassVar[list[str]] = ["{source_col}"]
    outputs: ClassVar[list[str]] = ["{source_col}_zlma_{period}"]

    is_recursive: ClassVar[bool] = True
    warmup_invariant: ClassVar[bool] = False

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        source = df[self.source_col].to_numpy()
        lag = int((self.period - 1) / 2)

        col_series = df[self.source_col]
        adjusted = 2 * col_series - col_series.shift(lag)

        if self.ma_type == "sma":
            zlma = adjusted.rolling_mean(window_size=self.period)
        else:
            zlma = adjusted.ewm_mean(span=self.period, adjust=False)

        zlma_array = zlma.to_numpy()

        if self.normalized:
            from signalflow.ta._normalization import normalize_ma_pct

            zlma_array = normalize_ma_pct(source, zlma_array)

        col_name = self._get_output_name()
        return df.with_columns(pl.Series(name=col_name, values=zlma_array))

    def _get_output_name(self) -> str:
        """Generate output column name with normalization suffix."""
        suffix = "_norm" if self.normalized else ""
        return f"{self.source_col}_zlma_{self.period}{suffix}"

    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 20, "ma_type": "ema"},
        {"source_col": "close", "period": 20, "ma_type": "ema", "normalized": True},
        {"source_col": "close", "period": 60, "ma_type": "ema"},
        {"source_col": "close", "period": 120, "ma_type": "sma"},
    ]

    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable, reproducible output."""
        return self.period * 5


@dataclass
@feature("smooth/mcginley")
class McGinleySmooth(Feature):
    """McGinley Dynamic Indicator.

    Self-adjusting MA that tracks price more closely.
    Speeds up in downtrends, slows in uptrends.

    MD = MD_prev + (price - MD_prev) / (k * n * (price/MD_prev)^4)

    In normalized mode, returns percentage difference from source:
    normalized = (source - ma) / source

    Reference: McGinley, J. "Journal of Technical Analysis"
    """

    source_col: str = "close"
    period: int = 10
    k: float = 0.6
    normalized: bool = False

    requires: ClassVar[list[str]] = ["{source_col}"]
    outputs: ClassVar[list[str]] = ["{source_col}_mcg_{period}"]

    is_recursive: ClassVar[bool] = True
    warmup_invariant: ClassVar[bool] = False

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        source = df[self.source_col].to_numpy().astype(np.float64)

        md = _mcginley_kernel(source, self.period, float(self.k))

        if self.normalized:
            from signalflow.ta._normalization import normalize_ma_pct

            md = normalize_ma_pct(source, md)

        col_name = self._get_output_name()
        return df.with_columns(pl.Series(name=col_name, values=md))

    def _get_output_name(self) -> str:
        """Generate output column name with normalization suffix."""
        suffix = "_norm" if self.normalized else ""
        return f"{self.source_col}_mcg_{self.period}{suffix}"

    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 10, "k": 0.6},
        {"source_col": "close", "period": 10, "k": 0.6, "normalized": True},
        {"source_col": "close", "period": 60, "k": 0.6},
        {"source_col": "close", "period": 120, "k": 0.6},
    ]

    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable, reproducible output."""
        return self.period * 5


@dataclass
@feature("smooth/frama")
class FramaSmooth(Feature):
    """Fractal Adaptive Moving Average.

    Uses fractal dimension to adapt smoothing.
    Higher dimension (choppy) = slower. Lower (trending) = faster.

    D = (log(N1 + N2) - log(N3)) / log(2)
    alpha = exp(-4.6 * (D - 1))

    In normalized mode, returns percentage difference from source:
    normalized = (source - ma) / source

    Reference: Ehlers, J. "FRAMA"
    """

    source_col: str = "close"
    period: int = 16
    normalized: bool = False

    requires: ClassVar[list[str]] = ["{source_col}"]
    outputs: ClassVar[list[str]] = ["{source_col}_frama_{period}"]

    is_recursive: ClassVar[bool] = True
    warmup_invariant: ClassVar[bool] = False

    def __post_init__(self) -> None:
        if self.period % 2 != 0:
            raise ValueError("FRAMA period must be even")

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        source = df[self.source_col].to_numpy().astype(np.float64)

        frama = _frama_kernel(source, self.period)

        if self.normalized:
            from signalflow.ta._normalization import normalize_ma_pct

            frama = normalize_ma_pct(source, frama)

        col_name = self._get_output_name()
        return df.with_columns(pl.Series(name=col_name, values=frama))

    def _get_output_name(self) -> str:
        """Generate output column name with normalization suffix."""
        suffix = "_norm" if self.normalized else ""
        return f"{self.source_col}_frama_{self.period}{suffix}"

    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 16},
        {"source_col": "close", "period": 16, "normalized": True},
        {"source_col": "close", "period": 60},
        {"source_col": "close", "period": 120},
    ]

    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable, reproducible output."""
        return self.period * 5


@njit
def _kalman_filter_single_window(
    data: np.ndarray, process_noise: float, measurement_noise: float
) -> tuple[float, float]:
    """Apply Kalman filter to a single window segment."""
    n = len(data)
    estimate = data[0]
    estimate_covariance = 1.0

    for i in range(1, n):
        predicted_estimate = estimate
        predicted_covariance = estimate_covariance + process_noise
        innovation = data[i] - predicted_estimate
        innovation_covariance = predicted_covariance + measurement_noise
        kalman_gain = predicted_covariance / innovation_covariance

        estimate = predicted_estimate + kalman_gain * innovation
        estimate_covariance = (1 - kalman_gain) * predicted_covariance

        process_noise = max(1e-2, np.abs(innovation) / 10)

    return estimate, process_noise


@njit
def _kalman_filter_series(
    data: np.ndarray, window: int, process_noise_init: float, measurement_noise: float
) -> np.ndarray:
    """Apply Kalman filter across a rolling window."""
    n = len(data)
    result = np.full(n, np.nan)
    process_noise = process_noise_init

    for i in range(window, n):
        segment = data[i - window : i]
        filtered_value, process_noise = _kalman_filter_single_window(segment, process_noise, measurement_noise)
        result[i] = filtered_value

    return result


@dataclass
@feature("smooth/kalman")
class KalmanSmooth(Feature):
    """Adaptive Kalman Filter smoothing.

    Uses Kalman filter with adaptive process noise that adjusts
    based on innovation (prediction error). Higher innovation
    increases process noise, making the filter more responsive.

    Particularly effective for noisy price data as it optimally
    balances between following the signal and filtering noise.

    In normalized mode, returns percentage difference from source:
    normalized = (source - kalman) / source

    Reference: Kalman, R.E. "A New Approach to Linear Filtering
    and Prediction Problems"
    """

    source_col: str = "close"
    window: int = 120
    process_noise: float = 0.1
    measurement_noise: float = 0.1
    normalized: bool = False

    requires: ClassVar[list[str]] = ["{source_col}"]
    outputs: ClassVar[list[str]] = ["{source_col}_kalman_{window}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        source = df[self.source_col].to_numpy().astype(np.float64)

        kalman = _kalman_filter_series(source, self.window, self.process_noise, self.measurement_noise)

        if self.normalized:
            from signalflow.ta._normalization import normalize_ma_pct

            kalman = normalize_ma_pct(source, kalman)

        col_name = self._get_output_name()
        return df.with_columns(pl.Series(name=col_name, values=kalman))

    def _get_output_name(self) -> str:
        """Generate output column name with normalization suffix."""
        suffix = "_norm" if self.normalized else ""
        return f"{self.source_col}_kalman_{self.window}{suffix}"

    test_params: ClassVar[list[dict]] = [
        {
            "source_col": "close",
            "window": 120,
            "process_noise": 0.1,
            "measurement_noise": 0.1,
        },
        {
            "source_col": "close",
            "window": 120,
            "process_noise": 0.1,
            "measurement_noise": 0.1,
            "normalized": True,
        },
        {
            "source_col": "close",
            "window": 60,
            "process_noise": 0.1,
            "measurement_noise": 0.1,
        },
        {
            "source_col": "close",
            "window": 240,
            "process_noise": 0.05,
            "measurement_noise": 0.1,
        },
    ]

    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable, reproducible output."""
        return self.window * 2

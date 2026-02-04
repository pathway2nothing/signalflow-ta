# src/signalflow/ta/stat/dsp.py
"""Digital signal processing measures for financial time series.

Indicators adapted from audio/acoustics signal processing that capture
spectral dynamics, frequency content changes, and signal characteristics
in financial time series. All computations are causal (no lookahead).

References:
    - Scheirer & Slaney (1997) - Spectral Flux
    - Kedem (1986) - Zero-Crossing Rate
    - Peeters (2004) - Spectral Rolloff
    - Dubnov (2004) - Spectral Flatness (Wiener Entropy)
    - Bogert et al. (1963) - Power Cepstrum
"""
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import polars as pl

from signalflow import sf_component
from signalflow.feature.base import Feature


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detrend_window(window: np.ndarray) -> np.ndarray | None:
    """Remove linear trend from window.

    Returns detrended array, or None if the residual has near-zero variance
    (i.e. the window is effectively constant after detrending).

    Complexity: O(n).
    """
    period = len(window)
    x = np.arange(period, dtype=np.float64)
    coeffs = np.polyfit(x, window, 1)
    detrended = window - np.polyval(coeffs, x)
    if np.std(detrended) < 1e-10:
        return None
    return detrended


def _power_spectrum(detrended: np.ndarray, skip_dc: bool = True
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Compute power spectrum and frequency bins from a detrended window.

    Args:
        detrended: Already-detrended signal (use _detrend_window first).
        skip_dc: If True, remove DC component (index 0).

    Returns:
        (power, freqs) where power = |FFT|^2 and freqs = normalized
        frequency bins from np.fft.rfftfreq.

    Complexity: O(n log n).
    """
    fft_vals = np.fft.rfft(detrended)
    power = np.abs(fft_vals) ** 2
    freqs = np.fft.rfftfreq(len(detrended))
    if skip_dc:
        power = power[1:]
        freqs = freqs[1:]
    return power, freqs


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------


@dataclass
@sf_component(name="stat/spectral_flux")
class SpectralFluxStat(Feature):
    """Rolling Spectral Flux (Scheirer & Slaney, 1997).

    Measures the rate of change of the power spectrum between consecutive
    overlapping windows. High spectral flux indicates that the frequency
    content of the signal is changing rapidly -- a regime shift detector.

    flux = sum( (P_curr - P_prev)^2 )

    where P is the L2-normalized power spectrum.

    Interpretation:
        - Low flux: stable spectral content (steady regime)
        - High flux: rapid spectral change (regime transition)
        - Flux spike: abrupt change in frequency structure
        - Rising flux: increasing market instability

    Parameters:
        source_col: Price column to analyze
        period: FFT window size (power-of-2 recommended)
        normalized: If True, apply rolling z-score normalization

    Reference: Scheirer, E. & Slaney, M. (1997). Construction and
    evaluation of a robust multifeature speech/music discriminator.
    ICASSP, 1331-1334.
    """

    source_col: str = "close"
    period: int = 64
    normalized: bool = False
    norm_period: int | None = None

    requires = ["{source_col}"]
    outputs = ["{source_col}_sflux_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        values = df[self.source_col].to_numpy().astype(np.float64)
        n = len(values)

        flux = np.full(n, np.nan)

        prev_power_norm = None
        for i in range(self.period - 1, n):
            window = values[i - self.period + 1:i + 1]

            detrended = _detrend_window(window)
            if detrended is None:
                prev_power_norm = None
                continue

            power, _ = _power_spectrum(detrended)
            total_power = np.sum(power)
            if total_power < 1e-10:
                prev_power_norm = None
                continue

            # L2-normalize the power spectrum
            norm = np.sqrt(np.sum(power ** 2))
            if norm < 1e-10:
                prev_power_norm = None
                continue
            curr_power_norm = power / norm

            if prev_power_norm is not None and len(prev_power_norm) == len(curr_power_norm):
                diff = curr_power_norm - prev_power_norm
                flux[i] = float(np.sum(diff ** 2))

            prev_power_norm = curr_power_norm

        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window
            norm_window = self.norm_period or get_norm_window(self.period)
            flux = normalize_zscore(flux, window=norm_window)

        suffix = "_norm" if self.normalized else ""
        col_name = f"{self.source_col}_sflux_{self.period}{suffix}"
        return df.with_columns(pl.Series(name=col_name, values=flux))

    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 64},
        {"source_col": "close", "period": 128},
        {"source_col": "close", "period": 32},
        {"source_col": "close", "period": 64, "normalized": True},
    ]

    @property
    def warmup(self) -> int:
        base = self.period * 5
        if self.normalized:
            from signalflow.ta._normalization import get_norm_window
            norm_window = self.norm_period or get_norm_window(self.period)
            return base + norm_window
        return base


@dataclass
@sf_component(name="stat/zero_crossing_rate")
class ZeroCrossingRateStat(Feature):
    """Rolling Zero-Crossing Rate (Kedem, 1986).

    Measures the rate at which the detrended signal changes sign within
    a rolling window. Operates in the time domain (no FFT required).

    ZCR = count(sign(x[t]) != sign(x[t-1])) / (period - 1)

    Interpretation:
        - High ZCR (~0.5): choppy, noisy price action (resembles white noise)
        - Low ZCR (~0): smooth, trending price action
        - Rising ZCR: market becoming choppier
        - Falling ZCR: trend emerging
        - ZCR is inversely related to dominant cycle wavelength

    Output is bounded [0, 1].

    Parameters:
        source_col: Price column to analyze
        period: Rolling window size
        normalized: If True, apply rolling z-score normalization

    Reference: Kedem, B. (1986). Spectral Analysis and Discrimination
    by Zero-Crossings. Proceedings of the IEEE, 74(11), 1477-1493.
    """

    source_col: str = "close"
    period: int = 64
    normalized: bool = False
    norm_period: int | None = None

    requires = ["{source_col}"]
    outputs = ["{source_col}_zcr_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        values = df[self.source_col].to_numpy().astype(np.float64)
        n = len(values)

        zcr = np.full(n, np.nan)
        for i in range(self.period - 1, n):
            window = values[i - self.period + 1:i + 1]

            detrended = _detrend_window(window)
            if detrended is None:
                continue

            # Count sign changes (treat zero as non-negative)
            signs = np.sign(detrended)
            signs[signs == 0] = 1.0
            crossings = np.sum(signs[1:] != signs[:-1])
            zcr[i] = float(crossings) / (self.period - 1)

        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window
            norm_window = self.norm_period or get_norm_window(self.period)
            zcr = normalize_zscore(zcr, window=norm_window)

        suffix = "_norm" if self.normalized else ""
        col_name = f"{self.source_col}_zcr_{self.period}{suffix}"
        return df.with_columns(pl.Series(name=col_name, values=zcr))

    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 64},
        {"source_col": "close", "period": 128},
        {"source_col": "close", "period": 32},
        {"source_col": "close", "period": 64, "normalized": True},
    ]

    @property
    def warmup(self) -> int:
        base = self.period * 5
        if self.normalized:
            from signalflow.ta._normalization import get_norm_window
            norm_window = self.norm_period or get_norm_window(self.period)
            return base + norm_window
        return base


@dataclass
@sf_component(name="stat/spectral_rolloff")
class SpectralRolloffStat(Feature):
    """Rolling Spectral Rolloff (Peeters, 2004).

    The frequency below which a specified percentage of the total spectral
    energy is concentrated. Indicates how the spectral energy is distributed
    across the frequency range.

    rolloff_freq = f_k where sum(P[0:k]) >= rolloff_pct * sum(P)

    Interpretation:
        - High rolloff: energy extends to high frequencies (noisy, choppy)
        - Low rolloff: energy concentrated at low frequencies (smooth, trending)
        - Rising rolloff: increasing high-frequency activity
        - Falling rolloff: low-frequency (trend) dominance increasing

    Output is in normalized frequency units [0, 0.5].

    Parameters:
        source_col: Price column to analyze
        period: FFT window size (power-of-2 recommended)
        rolloff_pct: Percentage threshold (default: 0.85)
        normalized: If True, apply rolling z-score normalization

    Reference: Peeters, G. (2004). A Large Set of Audio Features for
    Sound Description. IRCAM Technical Report.
    """

    source_col: str = "close"
    period: int = 64
    rolloff_pct: float = 0.85
    normalized: bool = False
    norm_period: int | None = None

    requires = ["{source_col}"]
    outputs = ["{source_col}_srolloff_{period}"]

    def __post_init__(self):
        if not (0.0 < self.rolloff_pct < 1.0):
            raise ValueError(
                f"rolloff_pct must be in (0, 1), got {self.rolloff_pct}"
            )

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        values = df[self.source_col].to_numpy().astype(np.float64)
        n = len(values)

        rolloff = np.full(n, np.nan)
        for i in range(self.period - 1, n):
            window = values[i - self.period + 1:i + 1]

            detrended = _detrend_window(window)
            if detrended is None:
                continue

            power, freqs = _power_spectrum(detrended)
            total_power = np.sum(power)
            if total_power < 1e-10:
                continue

            # Find frequency where cumulative power exceeds threshold
            threshold = self.rolloff_pct * total_power
            cumulative = np.cumsum(power)
            idx = np.searchsorted(cumulative, threshold)
            if idx < len(freqs):
                rolloff[i] = float(freqs[idx])
            else:
                rolloff[i] = float(freqs[-1])

        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window
            norm_window = self.norm_period or get_norm_window(self.period)
            rolloff = normalize_zscore(rolloff, window=norm_window)

        suffix = "_norm" if self.normalized else ""
        col_name = f"{self.source_col}_srolloff_{self.period}{suffix}"
        return df.with_columns(pl.Series(name=col_name, values=rolloff))

    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 64, "rolloff_pct": 0.85},
        {"source_col": "close", "period": 128, "rolloff_pct": 0.85},
        {"source_col": "close", "period": 64, "rolloff_pct": 0.95},
        {"source_col": "close", "period": 64, "normalized": True},
    ]

    @property
    def warmup(self) -> int:
        base = self.period * 5
        if self.normalized:
            from signalflow.ta._normalization import get_norm_window
            norm_window = self.norm_period or get_norm_window(self.period)
            return base + norm_window
        return base


@dataclass
@sf_component(name="stat/spectral_flatness")
class SpectralFlatnessStat(Feature):
    """Rolling Spectral Flatness / Wiener Entropy (Dubnov, 2004).

    Ratio of the geometric mean to the arithmetic mean of the power
    spectrum. Measures how "flat" (noise-like) versus "peaked" (tonal)
    the spectral distribution is.

    flatness = geometric_mean(P) / arithmetic_mean(P)
             = exp(mean(log(P))) / mean(P)

    Interpretation:
        - Flatness ~ 1.0: flat spectrum (white noise, choppy market)
        - Flatness ~ 0.0: peaked spectrum (strong periodicity, clear cycle)
        - Rising flatness: market losing periodic structure
        - Falling flatness: dominant cycle emerging
        - Complement to spectral entropy (flatness focuses on
          peak-to-average ratio, entropy on overall distribution shape)

    Output is bounded [0, 1].

    Parameters:
        source_col: Price column to analyze
        period: FFT window size (power-of-2 recommended)
        normalized: If True, apply rolling z-score normalization

    Reference: Dubnov, S. (2004). Generalization of Spectral Flatness
    Measure for Non-Gaussian Linear Processes. IEEE Signal Processing
    Letters, 11(8), 698-701.
    """

    source_col: str = "close"
    period: int = 64
    normalized: bool = False
    norm_period: int | None = None

    requires = ["{source_col}"]
    outputs = ["{source_col}_sflat_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        values = df[self.source_col].to_numpy().astype(np.float64)
        n = len(values)

        flatness = np.full(n, np.nan)
        for i in range(self.period - 1, n):
            window = values[i - self.period + 1:i + 1]

            detrended = _detrend_window(window)
            if detrended is None:
                continue

            power, _ = _power_spectrum(detrended)
            if len(power) == 0:
                continue

            arith_mean = np.mean(power)
            if arith_mean < 1e-20:
                continue

            # Geometric mean via log to avoid overflow/underflow
            # Add epsilon to avoid log(0)
            log_power = np.log(power + 1e-20)
            geom_mean = np.exp(np.mean(log_power))

            flatness[i] = float(geom_mean / arith_mean)

        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window
            norm_window = self.norm_period or get_norm_window(self.period)
            flatness = normalize_zscore(flatness, window=norm_window)

        suffix = "_norm" if self.normalized else ""
        col_name = f"{self.source_col}_sflat_{self.period}{suffix}"
        return df.with_columns(pl.Series(name=col_name, values=flatness))

    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 64},
        {"source_col": "close", "period": 128},
        {"source_col": "close", "period": 32},
        {"source_col": "close", "period": 64, "normalized": True},
    ]

    @property
    def warmup(self) -> int:
        base = self.period * 5
        if self.normalized:
            from signalflow.ta._normalization import get_norm_window
            norm_window = self.norm_period or get_norm_window(self.period)
            return base + norm_window
        return base


@dataclass
@sf_component(name="stat/power_cepstrum")
class PowerCepstrumStat(Feature):
    """Rolling Power Cepstrum (Bogert, Healy & Tukey, 1963).

    The power cepstrum is the squared magnitude of the inverse FFT of
    the log power spectrum. It detects periodicity within the spectrum
    itself -- "spectrum of the spectrum." The dominant peak in the
    cepstrum (the dominant quefrency) reveals the strongest repeating
    spectral pattern.

    cepstrum = |IFFT(log(|FFT(x)|^2))|^2

    Output is the magnitude of the dominant cepstral peak (excluding
    the DC quefrency and very short quefrencies below min_quefrency).

    Interpretation:
        - High peak: strong repeating spectral pattern (dominant cycle)
        - Low peak: no clear spectral periodicity
        - Peak rising: repeating pattern strengthening
        - Peak falling: cycle structure weakening
        - The quefrency of the peak indicates the pseudo-period of
          the dominant spectral pattern

    Parameters:
        source_col: Price column to analyze
        period: FFT window size (power-of-2 recommended)
        min_quefrency: Minimum quefrency to consider (skip very short
            pseudo-periods, default: 2)
        normalized: If True, apply rolling z-score normalization

    Reference: Bogert, B.P., Healy, M.J.R. & Tukey, J.W. (1963).
    The Quefrency Alanysis of Time Series for Echoes. Proceedings of
    the Symposium on Time Series Analysis, 209-243.
    """

    source_col: str = "close"
    period: int = 64
    min_quefrency: int = 2
    normalized: bool = False
    norm_period: int | None = None

    requires = ["{source_col}"]
    outputs = ["{source_col}_cepstrum_{period}"]

    def __post_init__(self):
        if self.min_quefrency < 1:
            raise ValueError(
                f"min_quefrency must be >= 1, got {self.min_quefrency}"
            )
        if self.min_quefrency >= self.period // 2:
            raise ValueError(
                f"min_quefrency must be < period // 2, got "
                f"min_quefrency={self.min_quefrency}, period={self.period}"
            )

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        values = df[self.source_col].to_numpy().astype(np.float64)
        n = len(values)

        cepstrum_peak = np.full(n, np.nan)
        for i in range(self.period - 1, n):
            window = values[i - self.period + 1:i + 1]

            detrended = _detrend_window(window)
            if detrended is None:
                continue

            # Full FFT power spectrum (include DC for proper cepstrum)
            fft_vals = np.fft.rfft(detrended)
            power = np.abs(fft_vals) ** 2

            if np.sum(power) < 1e-10:
                continue

            # Log power spectrum (add epsilon to avoid log(0))
            log_power = np.log(power + 1e-20)

            # Inverse FFT of log power spectrum = cepstrum
            cepstrum = np.fft.irfft(log_power)
            cepstrum_mag = np.abs(cepstrum)

            # Find dominant peak in valid quefrency range
            max_quefrency = self.period // 2
            if self.min_quefrency < max_quefrency:
                search_range = cepstrum_mag[self.min_quefrency:max_quefrency]
                if len(search_range) > 0:
                    cepstrum_peak[i] = float(np.max(search_range))

        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window
            norm_window = self.norm_period or get_norm_window(self.period)
            cepstrum_peak = normalize_zscore(cepstrum_peak, window=norm_window)

        suffix = "_norm" if self.normalized else ""
        col_name = f"{self.source_col}_cepstrum_{self.period}{suffix}"
        return df.with_columns(pl.Series(name=col_name, values=cepstrum_peak))

    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 64},
        {"source_col": "close", "period": 128},
        {"source_col": "close", "period": 32},
        {"source_col": "close", "period": 64, "normalized": True},
    ]

    @property
    def warmup(self) -> int:
        base = self.period * 5
        if self.normalized:
            from signalflow.ta._normalization import get_norm_window
            norm_window = self.norm_period or get_norm_window(self.period)
            return base + norm_window
        return base

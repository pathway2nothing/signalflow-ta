# src/signalflow/ta/stat/cycle.py
"""Cycle analysis via Hilbert Transform - instantaneous amplitude, phase, frequency."""
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import polars as pl
from scipy.signal import hilbert

from signalflow import sf_component
from signalflow.feature.base import Feature


def _detrend_and_hilbert(values: np.ndarray, period: int, idx: int) -> tuple[float, float]:
    """Apply Hilbert transform to a detrended window, return amplitude and phase."""
    window = values[idx - period + 1:idx + 1]

    # Detrend: remove linear trend to isolate oscillatory component
    x = np.arange(period, dtype=np.float64)
    coeffs = np.polyfit(x, window, 1)
    detrended = window - np.polyval(coeffs, x)

    # Check for near-zero variance
    if np.std(detrended) < 1e-10:
        return np.nan, np.nan

    analytic = hilbert(detrended)
    amplitude = np.abs(analytic[-1])
    phase = np.angle(analytic[-1])
    return amplitude, phase


@dataclass
@sf_component(name="stat/inst_amplitude")
class InstAmplitudeStat(Feature):
    """Instantaneous Amplitude via Hilbert Transform.

    Envelope of the detrended price signal.

    Detrend price (remove linear trend), apply Hilbert transform,
    take absolute value of the analytic signal.

    Interpretation:
    - High amplitude: large oscillations (volatile, active market)
    - Low amplitude: small oscillations (quiet, consolidating)
    - Rising amplitude: volatility expansion
    - Falling amplitude: volatility contraction

    Reference: Hilbert transform / analytic signal
    https://en.wikipedia.org/wiki/Hilbert_transform
    """

    source_col: str = "close"
    period: int = 40
    normalized: bool = False
    norm_period: int | None = None

    requires = ["{source_col}"]
    outputs = ["{source_col}_hamp_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        values = df[self.source_col].to_numpy().astype(np.float64)
        n = len(values)

        amplitude = np.full(n, np.nan)
        for i in range(self.period - 1, n):
            amp, _ = _detrend_and_hilbert(values, self.period, i)
            amplitude[i] = amp

        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window
            norm_window = self.norm_period or get_norm_window(self.period)
            amplitude = normalize_zscore(amplitude, window=norm_window)

        suffix = "_norm" if self.normalized else ""
        col_name = f"{self.source_col}_hamp_{self.period}{suffix}"
        return df.with_columns(
            pl.Series(name=col_name, values=amplitude)
        )

    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 40},
        {"source_col": "close", "period": 80},
        {"source_col": "close", "period": 120},
        {"source_col": "close", "period": 40, "normalized": True},
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
@sf_component(name="stat/inst_phase")
class InstPhaseStat(Feature):
    """Instantaneous Phase via Hilbert Transform.

    Phase angle of the analytic signal (detrended price).

    Output in radians [-π, π].

    Interpretation:
    - Phase near 0 or ±π: price near equilibrium crossings
    - Phase near π/2: price at local peak of oscillation
    - Phase near -π/2: price at local trough of oscillation
    - Phase velocity (rate of change) indicates cycle speed

    Reference: Hilbert transform / analytic signal
    https://en.wikipedia.org/wiki/Analytic_signal
    """

    source_col: str = "close"
    period: int = 40
    normalized: bool = False

    requires = ["{source_col}"]
    outputs = ["{source_col}_hphase_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        values = df[self.source_col].to_numpy().astype(np.float64)
        n = len(values)

        phase = np.full(n, np.nan)
        for i in range(self.period - 1, n):
            _, ph = _detrend_and_hilbert(values, self.period, i)
            phase[i] = ph

        # Phase is bounded [-π, π] → normalize to [-1, 1]
        if self.normalized:
            phase = phase / np.pi

        suffix = "_norm" if self.normalized else ""
        col_name = f"{self.source_col}_hphase_{self.period}{suffix}"
        return df.with_columns(
            pl.Series(name=col_name, values=phase)
        )

    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 40},
        {"source_col": "close", "period": 80},
        {"source_col": "close", "period": 120},
        {"source_col": "close", "period": 40, "normalized": True},
    ]

    @property
    def warmup(self) -> int:
        return self.period * 5


@dataclass
@sf_component(name="stat/inst_frequency")
class InstFrequencyStat(Feature):
    """Instantaneous Frequency via Hilbert Transform.

    Rate of change of instantaneous phase: ω = Δφ/Δt.

    Uses unwrapped phase difference to handle ±π boundary.

    Interpretation:
    - High frequency: rapid oscillation cycles (choppy market)
    - Low frequency: slow cycles (trending or consolidating)
    - Frequency increase: market speeding up
    - Frequency decrease: market slowing down
    - Stable frequency: dominant cycle active

    Reference: Instantaneous frequency
    https://en.wikipedia.org/wiki/Instantaneous_phase_and_frequency
    """

    source_col: str = "close"
    period: int = 40
    normalized: bool = False
    norm_period: int | None = None

    requires = ["{source_col}"]
    outputs = ["{source_col}_hfreq_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        values = df[self.source_col].to_numpy().astype(np.float64)
        n = len(values)

        # Compute phase for full series
        phase = np.full(n, np.nan)
        for i in range(self.period - 1, n):
            _, ph = _detrend_and_hilbert(values, self.period, i)
            phase[i] = ph

        # Instantaneous frequency = d(phase)/dt with unwrapping
        freq = np.full(n, np.nan)
        for i in range(self.period, n):
            if not np.isnan(phase[i]) and not np.isnan(phase[i - 1]):
                dp = phase[i] - phase[i - 1]
                # Unwrap: handle ±π boundary
                if dp > np.pi:
                    dp -= 2 * np.pi
                elif dp < -np.pi:
                    dp += 2 * np.pi
                freq[i] = dp

        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window
            norm_window = self.norm_period or get_norm_window(self.period)
            freq = normalize_zscore(freq, window=norm_window)

        suffix = "_norm" if self.normalized else ""
        col_name = f"{self.source_col}_hfreq_{self.period}{suffix}"
        return df.with_columns(
            pl.Series(name=col_name, values=freq)
        )

    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 40},
        {"source_col": "close", "period": 80},
        {"source_col": "close", "period": 120},
        {"source_col": "close", "period": 40, "normalized": True},
    ]

    @property
    def warmup(self) -> int:
        base = (self.period + 1) * 5
        if self.normalized:
            from signalflow.ta._normalization import get_norm_window
            norm_window = self.norm_period or get_norm_window(self.period)
            return base + norm_window
        return base


@dataclass
@sf_component(name="stat/phase_acceleration")
class PhaseAccelerationStat(Feature):
    """Phase Acceleration (second derivative of phase).

    Rate of change of instantaneous frequency: α = Δω/Δt.

    Interpretation:
    - Positive acceleration: cycles speeding up
    - Negative acceleration: cycles slowing down
    - Sign changes: frequency inflection points (cycle regime change)
    - Precedes frequency changes, which precede amplitude changes

    Reference: Phase dynamics in signal processing
    """

    source_col: str = "close"
    period: int = 40
    normalized: bool = False
    norm_period: int | None = None

    requires = ["{source_col}"]
    outputs = ["{source_col}_phaseaccel_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        values = df[self.source_col].to_numpy().astype(np.float64)
        n = len(values)

        # Compute phase
        phase = np.full(n, np.nan)
        for i in range(self.period - 1, n):
            _, ph = _detrend_and_hilbert(values, self.period, i)
            phase[i] = ph

        # Frequency (unwrapped phase diff)
        freq = np.full(n, np.nan)
        for i in range(self.period, n):
            if not np.isnan(phase[i]) and not np.isnan(phase[i - 1]):
                dp = phase[i] - phase[i - 1]
                if dp > np.pi:
                    dp -= 2 * np.pi
                elif dp < -np.pi:
                    dp += 2 * np.pi
                freq[i] = dp

        # Phase acceleration = d(freq)/dt
        phaseaccel = np.full(n, np.nan)
        for i in range(self.period + 1, n):
            if not np.isnan(freq[i]) and not np.isnan(freq[i - 1]):
                phaseaccel[i] = freq[i] - freq[i - 1]

        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window
            norm_window = self.norm_period or get_norm_window(self.period)
            phaseaccel = normalize_zscore(phaseaccel, window=norm_window)

        suffix = "_norm" if self.normalized else ""
        col_name = f"{self.source_col}_phaseaccel_{self.period}{suffix}"
        return df.with_columns(
            pl.Series(name=col_name, values=phaseaccel)
        )

    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 40},
        {"source_col": "close", "period": 80},
        {"source_col": "close", "period": 120},
        {"source_col": "close", "period": 40, "normalized": True},
    ]

    @property
    def warmup(self) -> int:
        base = (self.period + 2) * 5
        if self.normalized:
            from signalflow.ta._normalization import get_norm_window
            norm_window = self.norm_period or get_norm_window(self.period)
            return base + norm_window
        return base


@dataclass
@sf_component(name="stat/constructive_interference")
class ConstructiveInterferenceStat(Feature):
    """Constructive Interference - phase-aligned amplitude boost.

    Uses two Hilbert transforms at different periods.
    When phases align (cos(phase_diff) > 0), amplitudes reinforce.

    interference = amp_fast × amp_slow × cos(phase_fast - phase_slow)

    Smoothed over rolling window.

    Interpretation:
    - Large positive: both cycles pushing in same direction (strong move)
    - Large negative: cycles opposing (cancellation, choppy)
    - Near zero: one cycle dominant or orthogonal phases
    - Peaks coincide with strong directional moves

    Reference: Wave superposition principle
    """

    source_col: str = "close"
    fast_period: int = 20
    slow_period: int = 50
    smooth: int = 5
    normalized: bool = False
    norm_period: int | None = None

    requires = ["{source_col}"]
    outputs = ["{source_col}_cinterf_{fast_period}_{slow_period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        values = df[self.source_col].to_numpy().astype(np.float64)
        n = len(values)

        # Compute fast and slow Hilbert transforms
        amp_fast = np.full(n, np.nan)
        phase_fast = np.full(n, np.nan)
        amp_slow = np.full(n, np.nan)
        phase_slow = np.full(n, np.nan)

        for i in range(self.fast_period - 1, n):
            a, p = _detrend_and_hilbert(values, self.fast_period, i)
            amp_fast[i] = a
            phase_fast[i] = p

        for i in range(self.slow_period - 1, n):
            a, p = _detrend_and_hilbert(values, self.slow_period, i)
            amp_slow[i] = a
            phase_slow[i] = p

        # Interference = amp_fast * amp_slow * cos(phase_diff)
        interf_raw = np.full(n, np.nan)
        start = self.slow_period - 1
        for i in range(start, n):
            if (not np.isnan(amp_fast[i]) and not np.isnan(amp_slow[i])
                    and not np.isnan(phase_fast[i]) and not np.isnan(phase_slow[i])):
                phase_diff = phase_fast[i] - phase_slow[i]
                interf_raw[i] = amp_fast[i] * amp_slow[i] * np.cos(phase_diff)

        # Smooth
        if self.smooth > 1:
            interf = np.full(n, np.nan)
            for i in range(start + self.smooth - 1, n):
                window = interf_raw[i - self.smooth + 1:i + 1]
                valid = window[~np.isnan(window)]
                if len(valid) > 0:
                    interf[i] = np.mean(valid)
        else:
            interf = interf_raw

        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window
            norm_window = self.norm_period or get_norm_window(self.slow_period)
            interf = normalize_zscore(interf, window=norm_window)

        suffix = "_norm" if self.normalized else ""
        col_name = f"{self.source_col}_cinterf_{self.fast_period}_{self.slow_period}{suffix}"
        return df.with_columns(
            pl.Series(name=col_name, values=interf)
        )

    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "fast_period": 20, "slow_period": 50},
        {"source_col": "close", "fast_period": 10, "slow_period": 30},
        {"source_col": "close", "fast_period": 30, "slow_period": 80},
        {"source_col": "close", "fast_period": 20, "slow_period": 50, "normalized": True},
    ]

    @property
    def warmup(self) -> int:
        base = (self.slow_period + self.smooth) * 5
        if self.normalized:
            from signalflow.ta._normalization import get_norm_window
            norm_window = self.norm_period or get_norm_window(self.slow_period)
            return base + norm_window
        return base


@dataclass
@sf_component(name="stat/beat_frequency")
class BeatFrequencyStat(Feature):
    """Beat Frequency - difference between two cycle frequencies.

    beat_freq = |inst_freq_fast - inst_freq_slow|

    When two cycles with different frequencies combine, they produce
    "beats" at the difference frequency.

    Interpretation:
    - High beat frequency: cycles diverging rapidly (instability)
    - Low beat frequency: cycles synchronized (stable regime)
    - Beat frequency approaching zero: resonance condition
    - Rising beat frequency: regime transition underway

    Reference: Beat frequency in wave physics
    https://en.wikipedia.org/wiki/Beat_(acoustics)
    """

    source_col: str = "close"
    fast_period: int = 20
    slow_period: int = 50
    normalized: bool = False
    norm_period: int | None = None

    requires = ["{source_col}"]
    outputs = ["{source_col}_beatfreq_{fast_period}_{slow_period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        values = df[self.source_col].to_numpy().astype(np.float64)
        n = len(values)

        # Compute instantaneous frequencies for both periods
        phase_fast = np.full(n, np.nan)
        phase_slow = np.full(n, np.nan)

        for i in range(self.fast_period - 1, n):
            _, p = _detrend_and_hilbert(values, self.fast_period, i)
            phase_fast[i] = p

        for i in range(self.slow_period - 1, n):
            _, p = _detrend_and_hilbert(values, self.slow_period, i)
            phase_slow[i] = p

        # Instantaneous frequencies (unwrapped phase diffs)
        def _inst_freq(phase_arr, start_idx):
            freq = np.full(n, np.nan)
            for i in range(start_idx, n):
                if not np.isnan(phase_arr[i]) and not np.isnan(phase_arr[i - 1]):
                    dp = phase_arr[i] - phase_arr[i - 1]
                    if dp > np.pi:
                        dp -= 2 * np.pi
                    elif dp < -np.pi:
                        dp += 2 * np.pi
                    freq[i] = dp
            return freq

        freq_fast = _inst_freq(phase_fast, self.fast_period)
        freq_slow = _inst_freq(phase_slow, self.slow_period)

        # Beat frequency = |freq_fast - freq_slow|
        beat = np.full(n, np.nan)
        start = self.slow_period
        for i in range(start, n):
            if not np.isnan(freq_fast[i]) and not np.isnan(freq_slow[i]):
                beat[i] = np.abs(freq_fast[i] - freq_slow[i])

        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window
            norm_window = self.norm_period or get_norm_window(self.slow_period)
            beat = normalize_zscore(beat, window=norm_window)

        suffix = "_norm" if self.normalized else ""
        col_name = f"{self.source_col}_beatfreq_{self.fast_period}_{self.slow_period}{suffix}"
        return df.with_columns(
            pl.Series(name=col_name, values=beat)
        )

    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "fast_period": 20, "slow_period": 50},
        {"source_col": "close", "fast_period": 10, "slow_period": 30},
        {"source_col": "close", "fast_period": 30, "slow_period": 80},
        {"source_col": "close", "fast_period": 20, "slow_period": 50, "normalized": True},
    ]

    @property
    def warmup(self) -> int:
        base = (self.slow_period + 1) * 5
        if self.normalized:
            from signalflow.ta._normalization import get_norm_window
            norm_window = self.norm_period or get_norm_window(self.slow_period)
            return base + norm_window
        return base


@dataclass
@sf_component(name="stat/standing_wave_ratio")
class StandingWaveRatioStat(Feature):
    """Standing Wave Ratio (SWR) - max/min amplitude ratio.

    SWR = max(amplitude) / min(amplitude) over rolling window

    Measures how "resonant" the market is - extreme SWR means
    large amplitude variation (nodes and antinodes).

    Interpretation:
    - SWR ≈ 1: traveling wave (uniform trend, no reflection)
    - SWR >> 1: standing wave (strong reflection, range-bound with spikes)
    - Rising SWR: increasing resonance or support/resistance interaction
    - Falling SWR: trend developing, less reflection

    Reference: Standing wave ratio in wave physics
    https://en.wikipedia.org/wiki/Standing_wave_ratio
    """

    source_col: str = "close"
    period: int = 40
    swr_window: int = 20
    normalized: bool = False
    norm_period: int | None = None

    requires = ["{source_col}"]
    outputs = ["{source_col}_swr_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        values = df[self.source_col].to_numpy().astype(np.float64)
        n = len(values)

        # Compute instantaneous amplitude
        amplitude = np.full(n, np.nan)
        for i in range(self.period - 1, n):
            amp, _ = _detrend_and_hilbert(values, self.period, i)
            amplitude[i] = amp

        # SWR = max(amp) / min(amp) over rolling window
        swr = np.full(n, np.nan)
        start = self.period - 1 + self.swr_window - 1
        for i in range(start, n):
            window = amplitude[i - self.swr_window + 1:i + 1]
            valid = window[~np.isnan(window)]
            if len(valid) >= 2:
                min_amp = np.min(valid)
                max_amp = np.max(valid)
                if min_amp > 1e-10:
                    swr[i] = max_amp / min_amp

        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window
            norm_window = self.norm_period or get_norm_window(self.period)
            swr = normalize_zscore(swr, window=norm_window)

        suffix = "_norm" if self.normalized else ""
        col_name = f"{self.source_col}_swr_{self.period}{suffix}"
        return df.with_columns(
            pl.Series(name=col_name, values=swr)
        )

    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 40, "swr_window": 20},
        {"source_col": "close", "period": 60, "swr_window": 30},
        {"source_col": "close", "period": 80, "swr_window": 40},
        {"source_col": "close", "period": 40, "swr_window": 20, "normalized": True},
    ]

    @property
    def warmup(self) -> int:
        base = (self.period + self.swr_window) * 5
        if self.normalized:
            from signalflow.ta._normalization import get_norm_window
            norm_window = self.norm_period or get_norm_window(self.period)
            return base + norm_window
        return base


@dataclass
@sf_component(name="stat/spectral_centroid")
class SpectralCentroidStat(Feature):
    """Spectral Centroid - center of mass of frequency spectrum.

    centroid = Σ(f_i × P_i) / Σ(P_i)

    where f_i = frequency bin, P_i = power at that frequency.

    Uses FFT on rolling detrended window.

    Interpretation:
    - High centroid: dominant energy at high frequencies (choppy)
    - Low centroid: dominant energy at low frequencies (smooth trend)
    - Rising centroid: market becoming choppier
    - Falling centroid: market becoming smoother/trending

    Reference: Spectral centroid in signal processing
    https://en.wikipedia.org/wiki/Spectral_centroid
    """

    source_col: str = "close"
    period: int = 64
    normalized: bool = False
    norm_period: int | None = None

    requires = ["{source_col}"]
    outputs = ["{source_col}_scentroid_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        values = df[self.source_col].to_numpy().astype(np.float64)
        n = len(values)

        centroid = np.full(n, np.nan)
        for i in range(self.period - 1, n):
            window = values[i - self.period + 1:i + 1]

            # Detrend
            x = np.arange(self.period, dtype=np.float64)
            coeffs = np.polyfit(x, window, 1)
            detrended = window - np.polyval(coeffs, x)

            if np.std(detrended) < 1e-10:
                continue

            # FFT power spectrum (positive frequencies only)
            fft_vals = np.fft.rfft(detrended)
            power = np.abs(fft_vals) ** 2
            freqs = np.fft.rfftfreq(self.period)

            # Skip DC component
            power = power[1:]
            freqs = freqs[1:]

            total_power = np.sum(power)
            if total_power > 1e-10:
                centroid[i] = np.sum(freqs * power) / total_power

        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window
            norm_window = self.norm_period or get_norm_window(self.period)
            centroid = normalize_zscore(centroid, window=norm_window)

        suffix = "_norm" if self.normalized else ""
        col_name = f"{self.source_col}_scentroid_{self.period}{suffix}"
        return df.with_columns(
            pl.Series(name=col_name, values=centroid)
        )

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
@sf_component(name="stat/spectral_entropy")
class SpectralEntropyStat(Feature):
    """Spectral Entropy - disorder of frequency distribution.

    H = -Σ(p_i × log(p_i))

    where p_i = normalized power spectrum (probability distribution).

    Interpretation:
    - High entropy: energy spread across many frequencies (noisy, complex)
    - Low entropy: energy concentrated in few frequencies (periodic, simple)
    - Rising entropy: market becoming more complex/chaotic
    - Falling entropy: dominant cycle emerging
    - Normalized by log(N) to bound in [0, 1]

    Reference: Spectral entropy in signal processing
    https://en.wikipedia.org/wiki/Spectral_density#Power_spectral_density
    """

    source_col: str = "close"
    period: int = 64
    normalized: bool = False

    requires = ["{source_col}"]
    outputs = ["{source_col}_sentropy_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        values = df[self.source_col].to_numpy().astype(np.float64)
        n = len(values)

        sentropy = np.full(n, np.nan)
        for i in range(self.period - 1, n):
            window = values[i - self.period + 1:i + 1]

            # Detrend
            x = np.arange(self.period, dtype=np.float64)
            coeffs = np.polyfit(x, window, 1)
            detrended = window - np.polyval(coeffs, x)

            if np.std(detrended) < 1e-10:
                continue

            # FFT power spectrum (positive frequencies, skip DC)
            fft_vals = np.fft.rfft(detrended)
            power = np.abs(fft_vals[1:]) ** 2

            total_power = np.sum(power)
            if total_power > 1e-10:
                # Normalize to probability distribution
                p = power / total_power
                # Entropy (avoid log(0))
                p_safe = p[p > 1e-15]
                entropy = -np.sum(p_safe * np.log(p_safe))
                # Normalize by max possible entropy
                max_entropy = np.log(len(power))
                if max_entropy > 0:
                    sentropy[i] = entropy / max_entropy

        # Already bounded [0, 1], use bounded normalization
        if self.normalized:
            sentropy = sentropy  # already [0, 1]

        suffix = "_norm" if self.normalized else ""
        col_name = f"{self.source_col}_sentropy_{self.period}{suffix}"
        return df.with_columns(
            pl.Series(name=col_name, values=sentropy)
        )

    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 64},
        {"source_col": "close", "period": 128},
        {"source_col": "close", "period": 32},
        {"source_col": "close", "period": 64, "normalized": True},
    ]

    @property
    def warmup(self) -> int:
        return self.period * 5

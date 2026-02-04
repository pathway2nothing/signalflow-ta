# src/signalflow/ta/stat/complexity.py
"""Complexity and information-theoretic measures for time series.

Indicators from nonlinear dynamics, information theory, and complexity science
that capture predictability, regularity, and regime transitions in financial
time series. All computations are causal (no lookahead) and reproducible.

References:
    - Bandt & Pompe (2002) - Permutation Entropy
    - Richman & Moorman (2000) - Sample Entropy
    - Lempel & Ziv (1976) - Lempel-Ziv Complexity
    - Frieden (2004) - Fisher Information
    - Peng et al. (1994) - Detrended Fluctuation Analysis
"""
from dataclasses import dataclass
from math import factorial, log2

import numpy as np
import polars as pl

from signalflow import sf_component
from signalflow.feature.base import Feature
from typing import ClassVar


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log_returns(values: np.ndarray) -> np.ndarray:
    """Compute log-returns with NaN for first element."""
    lr = np.full(len(values), np.nan)
    for i in range(1, len(values)):
        if values[i] > 0 and values[i - 1] > 0:
            lr[i] = np.log(values[i] / values[i - 1])
    return lr


def _permutation_entropy(x: np.ndarray, m: int) -> float:
    """Bandt-Pompe permutation entropy of order m.

    Counts ordinal patterns of length m in the series and computes
    the Shannon entropy of their distribution, normalized to [0, 1].

    Complexity: O(n * m)  (m is small, typically 3-5).
    """
    n = len(x)
    if n < m:
        return np.nan

    counts: dict[tuple, int] = {}
    total = 0

    for i in range(n - m + 1):
        # Ordinal pattern: rank of each element in the sub-sequence
        window = x[i:i + m]
        if np.any(np.isnan(window)):
            continue
        pattern = tuple(np.argsort(np.argsort(window)))
        counts[pattern] = counts.get(pattern, 0) + 1
        total += 1

    if total == 0:
        return np.nan

    # Shannon entropy of the pattern distribution
    h = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            h -= p * np.log2(p)

    # Normalize by max possible entropy: log2(m!)
    h_max = log2(factorial(m))
    if h_max < 1e-10:
        return np.nan
    return h / h_max


def _sample_entropy(x: np.ndarray, m: int, r: float) -> float:
    """Sample entropy (SampEn) of embedding dimension m and tolerance r.

    Counts template matches of length m and m+1 (excluding self-matches).
    SampEn = -ln(A / B)  where A = matches of length m+1, B = matches of length m.

    Lower SampEn = more regular/predictable.
    Higher SampEn = more complex/random.

    Complexity: O(n^2) within window — acceptable for typical windows (50-200).
    """
    n = len(x)
    if n < m + 2:
        return np.nan

    # Build templates
    # Count B (matches of length m) and A (matches of length m+1)
    b_count = 0
    a_count = 0

    for i in range(n - m):
        for j in range(i + 1, n - m):
            # Check m-length match
            if np.max(np.abs(x[i:i + m] - x[j:j + m])) <= r:
                b_count += 1
                # Check (m+1)-length match
                if i + m < n and j + m < n:
                    if abs(x[i + m] - x[j + m]) <= r:
                        a_count += 1

    if b_count == 0:
        return np.nan

    if a_count == 0:
        # Convention: return large value (no m+1 matches found)
        return np.log(b_count)

    return -np.log(a_count / b_count)


def _lempel_ziv_complexity(binary_seq: np.ndarray) -> float:
    """Lempel-Ziv complexity of a binary sequence.

    Counts the number of distinct sub-patterns when parsing left to right.
    Normalized by n / log2(n) to give a value near 1.0 for random sequences.

    Complexity: O(n^2) worst case, typically O(n log n).
    """
    n = len(binary_seq)
    if n < 2:
        return np.nan

    s = binary_seq.astype(int).tolist()
    complexity = 1
    l = 1  # current prefix length
    k = 1  # current component length
    k_max = 1

    while l + k <= n:
        # Check if s[l:l+k] is in s[0:l+k-1]
        sub = s[l:l + k]
        found = False
        for start in range(l + k - k):
            if s[start:start + k] == sub:
                found = True
                break

        if found:
            k += 1
            if k > k_max:
                k_max = k
        else:
            complexity += 1
            l += k_max if k_max > k else k
            k = 1
            k_max = 1

    # Normalize
    if n < 2:
        return np.nan
    norm = n / np.log2(n)
    return complexity / norm


def _fisher_information(x: np.ndarray, bins: int) -> float:
    """Fisher information estimated from histogram of the series.

    FI = sum( (dp/dx)^2 / p(x) )

    Approximated using finite differences on the empirical PDF.
    Spikes at regime transitions where the distribution sharpens.

    Complexity: O(n + bins).
    """
    n = len(x)
    if n < bins:
        return np.nan

    valid = x[~np.isnan(x)]
    if len(valid) < bins:
        return np.nan

    hist, bin_edges = np.histogram(valid, bins=bins, density=True)
    dx = bin_edges[1] - bin_edges[0]
    if dx < 1e-20:
        return np.nan

    fi = 0.0
    for i in range(len(hist) - 1):
        p = (hist[i] + hist[i + 1]) / 2
        if p > 1e-20:
            dp = hist[i + 1] - hist[i]
            fi += (dp ** 2) / p

    return fi / dx


def _dfa_exponent(x: np.ndarray, min_box: int = 4, max_box: int | None = None) -> float:
    """Detrended Fluctuation Analysis scaling exponent (alpha).

    Generalizes Hurst exponent to handle non-stationary series.

    alpha ~ 0.5: uncorrelated (random walk of cumulative sum)
    alpha > 0.5: long-range positive correlations (persistent / trending)
    alpha < 0.5: long-range negative correlations (anti-persistent / mean-reverting)
    alpha ~ 1.0: 1/f noise
    alpha ~ 1.5: Brownian motion

    Complexity: O(n * num_scales).
    """
    n = len(x)
    if max_box is None:
        max_box = n // 4

    if max_box < min_box + 2 or n < min_box * 4:
        return np.nan

    # Cumulative sum of deviations from mean (profile)
    mean_x = np.nanmean(x)
    profile = np.nancumsum(x - mean_x)

    # Generate box sizes (log-spaced)
    box_sizes = np.unique(np.logspace(
        np.log10(min_box), np.log10(max_box), num=min(15, max_box - min_box + 1)
    ).astype(int))
    box_sizes = box_sizes[(box_sizes >= min_box) & (box_sizes <= max_box)]

    if len(box_sizes) < 3:
        return np.nan

    fluctuations = []
    valid_boxes = []

    for box_size in box_sizes:
        n_boxes = n // box_size
        if n_boxes < 1:
            continue

        rms_sum = 0.0
        count = 0
        for b in range(n_boxes):
            segment = profile[b * box_size:(b + 1) * box_size]
            # Linear detrending
            x_axis = np.arange(box_size)
            # Fast linear fit: slope = (n*sum(xy) - sum(x)*sum(y)) / (n*sum(x^2) - sum(x)^2)
            sx = np.sum(x_axis)
            sy = np.sum(segment)
            sxy = np.dot(x_axis, segment)
            sx2 = np.sum(x_axis ** 2)
            denom = box_size * sx2 - sx * sx
            if abs(denom) < 1e-20:
                continue
            slope = (box_size * sxy - sx * sy) / denom
            intercept = (sy - slope * sx) / box_size
            trend = slope * x_axis + intercept
            residual = segment - trend
            rms_sum += np.sum(residual ** 2)
            count += box_size

        if count > 0:
            f_val = np.sqrt(rms_sum / count)
            if f_val > 1e-20:
                fluctuations.append(f_val)
                valid_boxes.append(box_size)

    if len(valid_boxes) < 3:
        return np.nan

    # Log-log regression: F(n) ~ n^alpha
    log_n = np.log(np.array(valid_boxes, dtype=float))
    log_f = np.log(np.array(fluctuations, dtype=float))

    # Linear regression for slope (alpha)
    n_pts = len(log_n)
    sx = np.sum(log_n)
    sy = np.sum(log_f)
    sxy = np.dot(log_n, log_f)
    sx2 = np.sum(log_n ** 2)
    denom = n_pts * sx2 - sx * sx
    if abs(denom) < 1e-20:
        return np.nan

    alpha = (n_pts * sxy - sx * sy) / denom
    return alpha


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------


@dataclass
@sf_component(name="stat/permutation_entropy")
class PermutationEntropyStat(Feature):
    """Rolling Permutation Entropy (Bandt & Pompe, 2002).

    Measures the complexity of a time series by analyzing the diversity
    of ordinal patterns (rank permutations) in sliding windows of
    log-returns.

    Ordinal patterns capture the *relative* ordering of consecutive values,
    making this measure robust to monotonic transformations and outliers.

    Formula:
        For embedding dimension m, count all m! possible ordinal patterns.
        H_perm = -sum(p_i * log2(p_i)) / log2(m!)

    Normalized to [0, 1]:
        0 = perfectly predictable (single repeating pattern)
        1 = maximally complex (all patterns equally likely)

    Interpretation:
        - Low PE: regular, predictable dynamics (trending, repeating cycles)
        - High PE: complex, noisy dynamics (random walk, choppy market)
        - Dropping PE: emerging pattern / trend formation
        - Rising PE: increasing randomness / regime breakdown

    Parameters:
        source_col: Price column to compute log-returns from
        period: Rolling window size for computing entropy
        m: Embedding dimension (pattern length, typically 3-5)
        normalized: If True, apply rolling z-score normalization

    Reference: Bandt, C. & Pompe, B. (2002). Permutation Entropy:
    A Natural Complexity Measure for Time Series.
    Physical Review Letters, 88(17), 174102.
    """

    source_col: str = "close"
    period: int = 100
    m: int = 3
    normalized: bool = False
    norm_period: int | None = None

    requires = ["{source_col}"]
    outputs = ["{source_col}_perm_entropy_{period}"]

    def __post_init__(self):
        if self.m < 2 or self.m > 7:
            raise ValueError(f"m must be in [2, 7], got {self.m}")
        if self.period < self.m + 10:
            raise ValueError(f"period must be >= m + 10, got period={self.period}, m={self.m}")

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        values = df[self.source_col].to_numpy()
        n = len(values)

        log_ret = _log_returns(values)

        pe = np.full(n, np.nan)
        for i in range(self.period, n):
            window = log_ret[i - self.period + 1:i + 1]
            valid = window[~np.isnan(window)]
            if len(valid) >= self.m + 5:
                pe[i] = _permutation_entropy(valid, self.m)

        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window
            norm_window = self.norm_period or get_norm_window(self.period)
            pe = normalize_zscore(pe, window=norm_window)

        suffix = "_norm" if self.normalized else ""
        col_name = f"{self.source_col}_perm_entropy_{self.period}{suffix}"
        return df.with_columns(pl.Series(name=col_name, values=pe))

    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 100, "m": 3},
        {"source_col": "close", "period": 200, "m": 4},
        {"source_col": "close", "period": 100, "m": 3, "normalized": True},
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
@sf_component(name="stat/sample_entropy")
class SampleEntropyStat(Feature):
    """Rolling Sample Entropy (Richman & Moorman, 2000).

    Measures the regularity / self-similarity of a time series.
    Counts how often patterns that are similar for m consecutive points
    remain similar when extended to m+1 points.

    SampEn = -ln(A / B)
    where B = number of template matches of length m,
          A = number of template matches of length m+1.

    Unlike Approximate Entropy (ApEn), SampEn does not count self-matches,
    making it less biased and more consistent.

    Interpretation:
        - Low SampEn: highly regular, self-similar (predictable)
        - High SampEn: complex, irregular (unpredictable)
        - Dropping SampEn: market becoming more predictable (pattern forming)
        - Rising SampEn: market becoming more random

    Parameters:
        source_col: Price column to compute log-returns from
        period: Rolling window size
        m: Template length (embedding dimension, typically 2)
        r_mult: Tolerance as multiple of std(window) (typically 0.2)
        normalized: If True, apply rolling z-score normalization

    Reference: Richman, J.S. & Moorman, J.R. (2000). Physiological
    time-series analysis using approximate entropy and sample entropy.
    American Journal of Physiology, 278(6), H2039-H2049.
    """

    source_col: str = "close"
    period: int = 80
    m: int = 2
    r_mult: float = 0.2
    normalized: bool = False
    norm_period: int | None = None

    requires = ["{source_col}"]
    outputs = ["{source_col}_sampen_{period}"]

    def __post_init__(self):
        if self.m < 1 or self.m > 4:
            raise ValueError(f"m must be in [1, 4], got {self.m}")
        if self.period < 30:
            raise ValueError(f"period must be >= 30 for reliable SampEn, got {self.period}")

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        values = df[self.source_col].to_numpy()
        n = len(values)

        log_ret = _log_returns(values)

        se = np.full(n, np.nan)
        for i in range(self.period, n):
            window = log_ret[i - self.period + 1:i + 1]
            valid = window[~np.isnan(window)]
            if len(valid) < self.m + 10:
                continue

            # Tolerance = r_mult * std of window
            std = np.std(valid, ddof=1)
            if std < 1e-10:
                continue

            r = self.r_mult * std
            se[i] = _sample_entropy(valid, self.m, r)

        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window
            norm_window = self.norm_period or get_norm_window(self.period)
            se = normalize_zscore(se, window=norm_window)

        suffix = "_norm" if self.normalized else ""
        col_name = f"{self.source_col}_sampen_{self.period}{suffix}"
        return df.with_columns(pl.Series(name=col_name, values=se))

    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 80, "m": 2, "r_mult": 0.2},
        {"source_col": "close", "period": 120, "m": 2, "r_mult": 0.15},
        {"source_col": "close", "period": 80, "m": 2, "normalized": True},
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
@sf_component(name="stat/lempel_ziv")
class LempelZivStat(Feature):
    """Rolling Lempel-Ziv Complexity (Lempel & Ziv, 1976).

    Measures the compressibility of a binary sequence derived from
    log-returns (positive return = 1, negative = 0).

    Counts the number of distinct sub-patterns encountered when parsing
    the binary string left-to-right. Normalized by n/log2(n) so that
    a perfectly random sequence yields ~1.0.

    This is the theoretical basis of data compression (LZ77/LZ78/gzip).
    Applied to financial series, it captures how "algorithmically complex"
    the recent price movement is.

    Interpretation:
        - LZC near 1.0: complex, random (hard to compress, unpredictable)
        - LZC << 1.0: structured, repetitive (easy to compress, exploitable)
        - Dropping LZC: emerging structure (pattern, trend)
        - Rising LZC: increasing randomness (breakdown of pattern)

    Parameters:
        source_col: Price column to compute log-returns from
        period: Rolling window size for computing complexity
        normalized: If True, apply rolling z-score normalization

    Reference: Lempel, A. & Ziv, J. (1976). On the Complexity of Finite
    Sequences. IEEE Transactions on Information Theory, 22(1), 75-81.
    """

    source_col: str = "close"
    period: int = 100
    normalized: bool = False
    norm_period: int | None = None

    requires = ["{source_col}"]
    outputs = ["{source_col}_lzc_{period}"]

    def __post_init__(self):
        if self.period < 20:
            raise ValueError(f"period must be >= 20 for reliable LZC, got {self.period}")

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        values = df[self.source_col].to_numpy()
        n = len(values)

        log_ret = _log_returns(values)

        lzc = np.full(n, np.nan)
        for i in range(self.period, n):
            window = log_ret[i - self.period + 1:i + 1]
            valid = window[~np.isnan(window)]
            if len(valid) < 20:
                continue

            # Binarize: positive return = 1, non-positive = 0
            binary = (valid > 0).astype(int)
            lzc[i] = _lempel_ziv_complexity(binary)

        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window
            norm_window = self.norm_period or get_norm_window(self.period)
            lzc = normalize_zscore(lzc, window=norm_window)

        suffix = "_norm" if self.normalized else ""
        col_name = f"{self.source_col}_lzc_{self.period}{suffix}"
        return df.with_columns(pl.Series(name=col_name, values=lzc))

    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 100},
        {"source_col": "close", "period": 200},
        {"source_col": "close", "period": 100, "normalized": True},
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
@sf_component(name="stat/fisher_information")
class FisherInformationStat(Feature):
    """Rolling Fisher Information (Frieden, 2004).

    Measures the "sharpness" of the probability distribution of
    log-returns within a rolling window. Fisher Information is high
    when the distribution is concentrated (peaked) and low when it
    is spread out (diffuse).

    FI = integral( (dp/dx)^2 / p(x) dx )

    Approximated using histogram-based finite differences on the
    empirical PDF of log-returns.

    Fisher Information is uniquely sensitive to regime transitions:
    it spikes when the market shifts from one statistical regime to
    another, because the distribution temporarily sharpens before
    re-spreading.

    Interpretation:
        - High FI: concentrated distribution (one dominant behavior)
        - Low FI: diffuse distribution (mixed behaviors, uncertainty)
        - FI spike: regime transition (distribution shifting rapidly)
        - Stable FI: stationary regime

    Parameters:
        source_col: Price column to compute log-returns from
        period: Rolling window size
        bins: Number of histogram bins for PDF estimation
        normalized: If True, apply rolling z-score normalization

    Reference: Frieden, B.R. (2004). Science from Fisher Information.
    Cambridge University Press.
    """

    source_col: str = "close"
    period: int = 100
    bins: int = 20
    normalized: bool = False
    norm_period: int | None = None

    requires = ["{source_col}"]
    outputs = ["{source_col}_fisher_info_{period}"]

    def __post_init__(self):
        if self.bins < 5:
            raise ValueError(f"bins must be >= 5, got {self.bins}")
        if self.period < self.bins * 2:
            raise ValueError(f"period must be >= 2 * bins, got period={self.period}, bins={self.bins}")

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        values = df[self.source_col].to_numpy()
        n = len(values)

        log_ret = _log_returns(values)

        fi = np.full(n, np.nan)
        for i in range(self.period, n):
            window = log_ret[i - self.period + 1:i + 1]
            valid = window[~np.isnan(window)]
            if len(valid) >= self.bins * 2:
                fi[i] = _fisher_information(valid, self.bins)

        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window
            norm_window = self.norm_period or get_norm_window(self.period)
            fi = normalize_zscore(fi, window=norm_window)

        suffix = "_norm" if self.normalized else ""
        col_name = f"{self.source_col}_fisher_info_{self.period}{suffix}"
        return df.with_columns(pl.Series(name=col_name, values=fi))

    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 100, "bins": 20},
        {"source_col": "close", "period": 200, "bins": 30},
        {"source_col": "close", "period": 100, "bins": 20, "normalized": True},
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
@sf_component(name="stat/dfa")
class DfaExponentStat(Feature):
    """Rolling DFA Exponent - Detrended Fluctuation Analysis (Peng et al., 1994).

    Generalizes the Hurst exponent to handle non-stationary time series.
    Unlike R/S analysis (used by HurstStat), DFA explicitly removes local
    trends before measuring fluctuations, making it more robust for
    financial data which is inherently non-stationary.

    Method:
        1. Compute the cumulative sum (profile) of deviations from mean
        2. Divide profile into boxes of size s
        3. Detrend each box (subtract linear fit)
        4. Compute RMS of residuals F(s)
        5. Repeat for multiple box sizes s
        6. DFA exponent alpha = slope of log(F(s)) vs log(s)

    Interpretation:
        - alpha ~ 0.5: uncorrelated (random walk in returns)
        - alpha > 0.5: persistent long-range correlations (trending)
        - alpha < 0.5: anti-persistent correlations (mean-reverting)
        - alpha ~ 1.0: 1/f noise (long memory)
        - alpha ~ 1.5: Brownian motion (integrated random walk)

    Advantages over Hurst (R/S):
        - Handles non-stationarity (detrending removes local trends)
        - More accurate for short time series
        - Less sensitive to outliers

    Parameters:
        source_col: Price column to compute log-returns from
        period: Rolling window size
        min_box: Smallest box size for DFA (default: 4)
        normalized: If True, apply rolling z-score normalization

    Reference: Peng, C.K. et al. (1994). Mosaic organization of DNA
    nucleotides. Physical Review E, 49(2), 1685-1689.
    """

    source_col: str = "close"
    period: int = 200
    min_box: int = 4
    normalized: bool = False
    norm_period: int | None = None

    requires = ["{source_col}"]
    outputs = ["{source_col}_dfa_{period}"]

    def __post_init__(self):
        if self.period < 50:
            raise ValueError(f"period must be >= 50 for reliable DFA, got {self.period}")
        if self.min_box < 3:
            raise ValueError(f"min_box must be >= 3, got {self.min_box}")

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        values = df[self.source_col].to_numpy()
        n = len(values)

        log_ret = _log_returns(values)

        dfa = np.full(n, np.nan)
        for i in range(self.period, n):
            window = log_ret[i - self.period + 1:i + 1]
            valid = window[~np.isnan(window)]
            if len(valid) >= 50:
                dfa[i] = _dfa_exponent(valid, min_box=self.min_box)

        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window
            norm_window = self.norm_period or get_norm_window(self.period)
            dfa = normalize_zscore(dfa, window=norm_window)

        suffix = "_norm" if self.normalized else ""
        col_name = f"{self.source_col}_dfa_{self.period}{suffix}"
        return df.with_columns(pl.Series(name=col_name, values=dfa))

    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 200, "min_box": 4},
        {"source_col": "close", "period": 500, "min_box": 6},
        {"source_col": "close", "period": 200, "min_box": 4, "normalized": True},
    ]

    @property
    def warmup(self) -> int:
        base = self.period * 5
        if self.normalized:
            from signalflow.ta._normalization import get_norm_window
            norm_window = self.norm_period or get_norm_window(self.period)
            return base + norm_window
        return base

# src/signalflow/ta/stat/information.py
"""Information-theoretic measures for time series.

Indicators from information theory and information geometry that capture
distributional divergence, nonlinear dependencies, and information dynamics
in financial time series. All computations are causal (no lookahead).

References:
    - Kullback & Leibler (1951) - KL Divergence
    - Lin (1991) - Jensen-Shannon Divergence
    - Rényi (1961) - Rényi Entropy
    - Kraskov et al. (2004) - Mutual Information estimation
    - Schreiber (2000) - Transfer Entropy / Information Gain
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


def _log_returns(values: np.ndarray) -> np.ndarray:
    """Compute log-returns with NaN for first element."""
    lr = np.full(len(values), np.nan)
    for i in range(1, len(values)):
        if values[i] > 0 and values[i - 1] > 0:
            lr[i] = np.log(values[i] / values[i - 1])
    return lr


def _histogram_pdf(x: np.ndarray, bins: int) -> np.ndarray:
    """Compute normalized histogram (empirical PDF).

    Returns probability array of length `bins`.
    Uses data-adaptive bin edges to handle varying ranges.
    """
    hist, _ = np.histogram(x, bins=bins, density=False)
    total = hist.sum()
    if total == 0:
        return np.full(bins, np.nan)
    return hist / total


def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL divergence D_KL(P || Q).

    Uses additive smoothing (Laplace) to avoid log(0).
    D_KL = sum( p_i * log2(p_i / q_i) )

    Complexity: O(bins).
    """
    eps = 1e-10
    p_s = p + eps
    q_s = q + eps
    p_s = p_s / p_s.sum()
    q_s = q_s / q_s.sum()
    return float(np.sum(p_s * np.log2(p_s / q_s)))


def _jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence JSD(P || Q).

    Symmetric, bounded [0, 1] (using log base 2).
    JSD = 0.5 * D_KL(P || M) + 0.5 * D_KL(Q || M)
    where M = 0.5 * (P + Q).

    Complexity: O(bins).
    """
    eps = 1e-10
    p_s = p + eps
    q_s = q + eps
    p_s = p_s / p_s.sum()
    q_s = q_s / q_s.sum()
    m = 0.5 * (p_s + q_s)
    return float(
        0.5 * np.sum(p_s * np.log2(p_s / m)) + 0.5 * np.sum(q_s * np.log2(q_s / m))
    )


def _renyi_entropy(x: np.ndarray, bins: int, alpha: float) -> float:
    """Rényi entropy of order alpha.

    H_alpha = (1 / (1 - alpha)) * log2( sum(p_i^alpha) )

    Special cases:
        alpha -> 1: Shannon entropy
        alpha = 2: collision entropy (-log2(sum(p^2)))
        alpha = 0: Hartley entropy (log2 of support size)
        alpha -> inf: min-entropy (-log2(max(p)))

    Normalized to [0, 1] by dividing by log2(bins).

    Complexity: O(n + bins).
    """
    p = _histogram_pdf(x, bins)
    if np.any(np.isnan(p)):
        return np.nan

    # Remove zero bins
    p_nz = p[p > 0]
    if len(p_nz) == 0:
        return np.nan

    h_max = np.log2(bins)
    if h_max < 1e-10:
        return np.nan

    if abs(alpha - 1.0) < 1e-10:
        # Shannon entropy limit
        h = -float(np.sum(p_nz * np.log2(p_nz)))
        return h / h_max

    sum_p_alpha = float(np.sum(p_nz**alpha))
    if sum_p_alpha <= 0:
        return np.nan

    h = (1.0 / (1.0 - alpha)) * np.log2(sum_p_alpha)
    return h / h_max


def _auto_mutual_information(x: np.ndarray, lag: int, bins: int) -> float:
    """Auto-mutual information between x(t) and x(t-lag).

    MI(X; X_lag) = sum_ij p(x_i, x_lag_j) * log2( p(x_i, x_lag_j) / (p(x_i) * p(x_lag_j)) )

    Estimated via 2D histogram (binned estimator).
    Captures nonlinear dependencies invisible to autocorrelation.

    Complexity: O(n + bins^2).
    """
    n = len(x)
    if n <= lag:
        return np.nan

    x_current = x[lag:]
    x_lagged = x[: n - lag]

    # Remove pairs with NaN
    valid = ~(np.isnan(x_current) | np.isnan(x_lagged))
    x_c = x_current[valid]
    x_l = x_lagged[valid]

    if len(x_c) < bins * 2:
        return np.nan

    # Joint histogram
    joint_hist, _, _ = np.histogram2d(x_c, x_l, bins=bins)
    joint_total = joint_hist.sum()
    if joint_total == 0:
        return np.nan

    p_joint = joint_hist / joint_total

    # Marginals
    p_x = p_joint.sum(axis=1)
    p_y = p_joint.sum(axis=0)

    # MI calculation
    mi = 0.0
    for i in range(bins):
        for j in range(bins):
            if p_joint[i, j] > 1e-10 and p_x[i] > 1e-10 and p_y[j] > 1e-10:
                mi += p_joint[i, j] * np.log2(p_joint[i, j] / (p_x[i] * p_y[j]))

    return mi


def _relative_information_gain(x: np.ndarray, sub_window: int, bins: int) -> float:
    """Relative information gain between first and second half of a window.

    Measures how much the distribution has changed over the window
    using symmetric JSD, then normalizes by window length.

    High values = distribution changing rapidly (regime shift).
    Low values = stationary distribution.

    Complexity: O(n + bins).
    """
    n = len(x)
    if n < sub_window * 2:
        return np.nan

    # Split into two consecutive sub-windows
    first_half = x[:sub_window]
    second_half = x[n - sub_window :]

    valid_first = first_half[~np.isnan(first_half)]
    valid_second = second_half[~np.isnan(second_half)]

    if len(valid_first) < bins or len(valid_second) < bins:
        return np.nan

    # Use common bin edges for fair comparison
    all_valid = np.concatenate([valid_first, valid_second])
    _, bin_edges = np.histogram(all_valid, bins=bins)

    hist_first, _ = np.histogram(valid_first, bins=bin_edges, density=False)
    hist_second, _ = np.histogram(valid_second, bins=bin_edges, density=False)

    total_first = hist_first.sum()
    total_second = hist_second.sum()
    if total_first == 0 or total_second == 0:
        return np.nan

    p_first = hist_first / total_first
    p_second = hist_second / total_second

    return _jensen_shannon_divergence(p_first, p_second)


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------


@dataclass
@sf_component(name="stat/kl_divergence")
class KLDivergenceStat(Feature):
    """Rolling KL Divergence (Kullback & Leibler, 1951).

    Measures the asymmetric divergence between the distribution of
    recent log-returns (short window) and a longer-term baseline
    distribution. Detects when the current market regime deviates
    from its recent history.

    D_KL(P_recent || Q_baseline) = sum( P(x) * log2(P(x) / Q(x)) )

    Interpretation:
        - KL ~ 0: recent behavior matches baseline (stable regime)
        - KL >> 0: recent behavior diverges from baseline (regime change)
        - KL spike: abrupt distributional shift
        - Rising KL: gradual regime transition

    Note: KL divergence is asymmetric — it measures how "surprising"
    the recent data would be under the baseline model.

    Parameters:
        source_col: Price column to compute log-returns from
        period: Baseline window size (longer-term reference)
        short_period: Recent window size (default: period // 4)
        bins: Number of histogram bins for PDF estimation
        normalized: If True, apply rolling z-score normalization

    Reference: Kullback, S. & Leibler, R.A. (1951). On Information
    and Sufficiency. Annals of Mathematical Statistics, 22(1), 79-86.
    """

    source_col: str = "close"
    period: int = 200
    short_period: int | None = None
    bins: int = 20
    normalized: bool = False
    norm_period: int | None = None

    requires = ["{source_col}"]
    outputs = ["{source_col}_kl_div_{period}"]

    def __post_init__(self):
        if self.short_period is None:
            self.short_period = self.period // 4
        if self.bins < 5:
            raise ValueError(f"bins must be >= 5, got {self.bins}")
        if self.period < self.short_period * 2:
            raise ValueError(
                f"period must be >= 2 * short_period, got period={self.period}, "
                f"short_period={self.short_period}"
            )
        if self.short_period < self.bins:
            raise ValueError(
                f"short_period must be >= bins, got short_period={self.short_period}, "
                f"bins={self.bins}"
            )

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        values = df[self.source_col].to_numpy()
        n = len(values)

        log_ret = _log_returns(values)

        kl = np.full(n, np.nan)
        for i in range(self.period, n):
            baseline = log_ret[i - self.period + 1 : i + 1]
            recent = log_ret[i - self.short_period + 1 : i + 1]

            valid_base = baseline[~np.isnan(baseline)]
            valid_recent = recent[~np.isnan(recent)]

            if len(valid_base) < self.bins * 2 or len(valid_recent) < self.bins:
                continue

            # Common bin edges from baseline
            _, bin_edges = np.histogram(valid_base, bins=self.bins)

            hist_base, _ = np.histogram(valid_base, bins=bin_edges, density=False)
            hist_recent, _ = np.histogram(valid_recent, bins=bin_edges, density=False)

            total_base = hist_base.sum()
            total_recent = hist_recent.sum()
            if total_base == 0 or total_recent == 0:
                continue

            p_recent = hist_recent / total_recent
            q_base = hist_base / total_base

            kl[i] = _kl_divergence(p_recent, q_base)

        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window

            norm_window = self.norm_period or get_norm_window(self.period)
            kl = normalize_zscore(kl, window=norm_window)

        suffix = "_norm" if self.normalized else ""
        col_name = f"{self.source_col}_kl_div_{self.period}{suffix}"
        return df.with_columns(pl.Series(name=col_name, values=kl))

    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 200, "bins": 20},
        {"source_col": "close", "period": 400, "bins": 30},
        {"source_col": "close", "period": 200, "bins": 20, "normalized": True},
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
@sf_component(name="stat/js_divergence")
class JSDivergenceStat(Feature):
    """Rolling Jensen-Shannon Divergence (Lin, 1991).

    Symmetric and bounded version of KL divergence that measures the
    similarity between the distributions of recent and baseline
    log-returns. Unlike KL, JSD is a proper metric (symmetric and
    satisfies triangle inequality when square-rooted).

    JSD(P || Q) = 0.5 * D_KL(P || M) + 0.5 * D_KL(Q || M)
    where M = 0.5 * (P + Q)

    Bounded in [0, 1] (using log base 2).
    sqrt(JSD) is a proper distance metric.

    Interpretation:
        - JSD ~ 0: distributions are identical (stable regime)
        - JSD ~ 1: distributions are maximally different (regime break)
        - JSD spike: abrupt regime change
        - Gradual JSD rise: slow distributional drift

    Advantages over KL divergence:
        - Symmetric: JSD(P||Q) = JSD(Q||P)
        - Bounded: always in [0, 1]
        - Always finite: no division-by-zero issues
        - More numerically stable

    Parameters:
        source_col: Price column to compute log-returns from
        period: Baseline window size
        short_period: Recent window size (default: period // 4)
        bins: Number of histogram bins
        normalized: If True, apply rolling z-score normalization

    Reference: Lin, J. (1991). Divergence Measures Based on the
    Shannon Entropy. IEEE Transactions on Information Theory, 37(1), 145-151.
    """

    source_col: str = "close"
    period: int = 200
    short_period: int | None = None
    bins: int = 20
    normalized: bool = False
    norm_period: int | None = None

    requires = ["{source_col}"]
    outputs = ["{source_col}_js_div_{period}"]

    def __post_init__(self):
        if self.short_period is None:
            self.short_period = self.period // 4
        if self.bins < 5:
            raise ValueError(f"bins must be >= 5, got {self.bins}")
        if self.period < self.short_period * 2:
            raise ValueError(
                f"period must be >= 2 * short_period, got period={self.period}, "
                f"short_period={self.short_period}"
            )
        if self.short_period < self.bins:
            raise ValueError(
                f"short_period must be >= bins, got short_period={self.short_period}, "
                f"bins={self.bins}"
            )

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        values = df[self.source_col].to_numpy()
        n = len(values)

        log_ret = _log_returns(values)

        jsd = np.full(n, np.nan)
        for i in range(self.period, n):
            baseline = log_ret[i - self.period + 1 : i + 1]
            recent = log_ret[i - self.short_period + 1 : i + 1]

            valid_base = baseline[~np.isnan(baseline)]
            valid_recent = recent[~np.isnan(recent)]

            if len(valid_base) < self.bins * 2 or len(valid_recent) < self.bins:
                continue

            # Common bin edges from full window
            all_valid = np.concatenate([valid_base, valid_recent])
            _, bin_edges = np.histogram(all_valid, bins=self.bins)

            hist_base, _ = np.histogram(valid_base, bins=bin_edges, density=False)
            hist_recent, _ = np.histogram(valid_recent, bins=bin_edges, density=False)

            total_base = hist_base.sum()
            total_recent = hist_recent.sum()
            if total_base == 0 or total_recent == 0:
                continue

            p_recent = hist_recent / total_recent
            q_base = hist_base / total_base

            jsd[i] = _jensen_shannon_divergence(p_recent, q_base)

        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window

            norm_window = self.norm_period or get_norm_window(self.period)
            jsd = normalize_zscore(jsd, window=norm_window)

        suffix = "_norm" if self.normalized else ""
        col_name = f"{self.source_col}_js_div_{self.period}{suffix}"
        return df.with_columns(pl.Series(name=col_name, values=jsd))

    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 200, "bins": 20},
        {"source_col": "close", "period": 400, "bins": 30},
        {"source_col": "close", "period": 200, "bins": 20, "normalized": True},
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
@sf_component(name="stat/renyi_entropy")
class RenyiEntropyStat(Feature):
    """Rolling Rényi Entropy (Rényi, 1961).

    Generalization of Shannon entropy with a tunable order parameter
    alpha that controls sensitivity to different parts of the
    probability distribution of log-returns.

    H_alpha = (1 / (1 - alpha)) * log2( sum(p_i^alpha) )

    Normalized to [0, 1] by dividing by log2(bins).

    Key alpha values:
        alpha -> 0: Hartley entropy (counts non-zero bins, max sensitivity
                    to rare events)
        alpha -> 1: Shannon entropy (standard information measure)
        alpha = 2:  Collision entropy / Rényi-2 (-log2(sum(p^2)),
                    related to Herfindahl index)
        alpha -> inf: Min-entropy (-log2(max(p)), most conservative,
                     determined by most probable outcome)

    For financial series:
        - alpha < 1: emphasizes rare events (tail-sensitive)
        - alpha = 1: balanced view (Shannon)
        - alpha > 1: emphasizes common events (mode-sensitive)
        - alpha = 2: captures concentration / dominance of one regime

    Interpretation:
        - High Rényi: diverse behaviors (complex, uncertain)
        - Low Rényi: concentrated behavior (one dominant pattern)
        - Dropping Rényi (alpha=2): single regime becoming dominant
        - Low Rényi (alpha=0.5) but high Shannon: heavy-tailed distribution

    Parameters:
        source_col: Price column to compute log-returns from
        period: Rolling window size
        alpha: Rényi order parameter (default: 2.0 for collision entropy)
        bins: Number of histogram bins
        normalized: If True, apply rolling z-score normalization

    Reference: Rényi, A. (1961). On Measures of Entropy and Information.
    Proceedings of the 4th Berkeley Symposium on Mathematical Statistics
    and Probability, 1, 547-561.
    """

    source_col: str = "close"
    period: int = 100
    alpha: float = 2.0
    bins: int = 20
    normalized: bool = False
    norm_period: int | None = None

    requires = ["{source_col}"]
    outputs = ["{source_col}_renyi_{period}"]

    def __post_init__(self):
        if self.alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {self.alpha}")
        if self.bins < 5:
            raise ValueError(f"bins must be >= 5, got {self.bins}")
        if self.period < self.bins * 2:
            raise ValueError(
                f"period must be >= 2 * bins, got period={self.period}, bins={self.bins}"
            )

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        values = df[self.source_col].to_numpy()
        n = len(values)

        log_ret = _log_returns(values)

        renyi = np.full(n, np.nan)
        for i in range(self.period, n):
            window = log_ret[i - self.period + 1 : i + 1]
            valid = window[~np.isnan(window)]
            if len(valid) >= self.bins * 2:
                renyi[i] = _renyi_entropy(valid, self.bins, self.alpha)

        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window

            norm_window = self.norm_period or get_norm_window(self.period)
            renyi = normalize_zscore(renyi, window=norm_window)

        suffix = "_norm" if self.normalized else ""
        col_name = f"{self.source_col}_renyi_{self.period}{suffix}"
        return df.with_columns(pl.Series(name=col_name, values=renyi))

    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 100, "alpha": 2.0, "bins": 20},
        {"source_col": "close", "period": 200, "alpha": 0.5, "bins": 20},
        {"source_col": "close", "period": 100, "alpha": 2.0, "normalized": True},
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
@sf_component(name="stat/auto_mutual_info")
class AutoMutualInfoStat(Feature):
    """Rolling Auto-Mutual Information (Fraser & Swinney, 1986).

    Measures the mutual information between the log-return series
    and its own lagged version. Unlike autocorrelation (which captures
    only linear dependencies), AMI captures any statistical dependency
    including nonlinear relationships.

    MI(X_t; X_{t-lag}) = sum p(x,y) * log2( p(x,y) / (p(x) * p(y)) )

    Estimated via 2D histogram (binned estimator).

    Interpretation:
        - High AMI: strong dependency between current and lagged values
                   (predictable, possibly nonlinear structure)
        - Low AMI: weak dependency (less predictable)
        - AMI >> autocorrelation: significant nonlinear structure exists
        - AMI spike: nonlinear pattern emerging
        - AMI at lag 1 dropping: loss of temporal structure

    AMI is also used to determine optimal embedding delay for
    phase space reconstruction (Takens' theorem) — the first
    minimum of AMI gives the optimal lag.

    Parameters:
        source_col: Price column to compute log-returns from
        period: Rolling window size
        lag: Time lag for mutual information (default: 1)
        bins: Number of histogram bins per dimension
        normalized: If True, apply rolling z-score normalization

    Reference: Fraser, A.M. & Swinney, H.L. (1986). Independent
    coordinates for strange attractors from mutual information.
    Physical Review A, 33(2), 1134-1140.
    """

    source_col: str = "close"
    period: int = 100
    lag: int = 1
    bins: int = 15
    normalized: bool = False
    norm_period: int | None = None

    requires = ["{source_col}"]
    outputs = ["{source_col}_ami_{period}"]

    def __post_init__(self):
        if self.lag < 1:
            raise ValueError(f"lag must be >= 1, got {self.lag}")
        if self.bins < 5:
            raise ValueError(f"bins must be >= 5, got {self.bins}")
        if self.period < self.lag + self.bins * 2:
            raise ValueError(
                f"period must be >= lag + 2*bins, got period={self.period}, "
                f"lag={self.lag}, bins={self.bins}"
            )

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        values = df[self.source_col].to_numpy()
        n = len(values)

        log_ret = _log_returns(values)

        ami = np.full(n, np.nan)
        for i in range(self.period, n):
            window = log_ret[i - self.period + 1 : i + 1]
            valid = window[~np.isnan(window)]
            if len(valid) >= self.lag + self.bins * 2:
                ami[i] = _auto_mutual_information(valid, self.lag, self.bins)

        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window

            norm_window = self.norm_period or get_norm_window(self.period)
            ami = normalize_zscore(ami, window=norm_window)

        suffix = "_norm" if self.normalized else ""
        col_name = f"{self.source_col}_ami_{self.period}{suffix}"
        return df.with_columns(pl.Series(name=col_name, values=ami))

    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 100, "lag": 1, "bins": 15},
        {"source_col": "close", "period": 200, "lag": 5, "bins": 20},
        {"source_col": "close", "period": 100, "lag": 1, "normalized": True},
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
@sf_component(name="stat/info_gain")
class RelativeInfoGainStat(Feature):
    """Rolling Relative Information Gain (Schreiber, 2000).

    Measures how rapidly the probability distribution of log-returns
    is changing over time. Computes Jensen-Shannon divergence between
    the first and second halves of the rolling window.

    High information gain = distribution changing rapidly (regime shift).
    Low information gain = stationary distribution (stable regime).

    This is conceptually related to transfer entropy (Schreiber, 2000)
    but applied as a self-referential measure: how much information
    does the recent past add compared to the older past?

    Interpretation:
        - High IG: rapid distributional change (transition period)
        - Low IG: stable distribution (steady regime)
        - IG spike: abrupt regime change
        - Sustained high IG: prolonged instability
        - IG dropping after spike: new regime stabilizing

    Parameters:
        source_col: Price column to compute log-returns from
        period: Full rolling window size
        bins: Number of histogram bins
        normalized: If True, apply rolling z-score normalization

    Reference: Schreiber, T. (2000). Measuring Information Transfer.
    Physical Review Letters, 85(2), 461-464.
    """

    source_col: str = "close"
    period: int = 100
    bins: int = 20
    normalized: bool = False
    norm_period: int | None = None

    requires = ["{source_col}"]
    outputs = ["{source_col}_info_gain_{period}"]

    def __post_init__(self):
        if self.bins < 5:
            raise ValueError(f"bins must be >= 5, got {self.bins}")
        if self.period < self.bins * 4:
            raise ValueError(
                f"period must be >= 4 * bins, got period={self.period}, bins={self.bins}"
            )

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        values = df[self.source_col].to_numpy()
        n = len(values)

        log_ret = _log_returns(values)

        ig = np.full(n, np.nan)
        sub_window = self.period // 2

        for i in range(self.period, n):
            window = log_ret[i - self.period + 1 : i + 1]
            valid = window[~np.isnan(window)]
            if len(valid) >= sub_window * 2:
                ig[i] = _relative_information_gain(valid, sub_window, self.bins)

        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window

            norm_window = self.norm_period or get_norm_window(self.period)
            ig = normalize_zscore(ig, window=norm_window)

        suffix = "_norm" if self.normalized else ""
        col_name = f"{self.source_col}_info_gain_{self.period}{suffix}"
        return df.with_columns(pl.Series(name=col_name, values=ig))

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

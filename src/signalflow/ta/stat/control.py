# src/signalflow/ta/stat/control.py
"""Control theory and systems engineering indicators for time series.

Treats price as the output of a dynamical system and applies control-theoretic
techniques to detect regime changes, quantify predictability, and track
changing market dynamics. All computations are causal (no lookahead) and
reproducible.

References:
    - Harvey (1989) - Kalman Filter & Structural Time Series
    - Ljung (1999) - System Identification (AR models)
    - Rosenstein, Collins & De Luca (1993) - Maximum Lyapunov Exponent
    - Astrom & Murray (2008) - PID Control / Feedback Systems
    - Geman, Bienenstock & Doursat (1992) - Bias-Variance Decomposition
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


def _kalman_innovation_variance(
    returns: np.ndarray, process_noise_ratio: float
) -> float:
    """Normalized Innovation Statistic (NIS) from a 1-D local-level Kalman filter.

    Model:
        x[t] = x[t-1] + w,  w ~ N(0, Q)
        z[t] = x[t]   + v,  v ~ N(0, R)

    NIS = mean(e[t]^2 / S[t]) over the second half of the window.
    Under correct model: NIS ~ 1.0.  NIS >> 1 = model breakdown / regime shift.

    Complexity: O(n).
    """
    n = len(returns)
    if n < 20:
        return np.nan

    R = np.var(returns)
    if R < 1e-20:
        return np.nan
    Q = process_noise_ratio * R

    x_hat = returns[0]
    P = R

    innovations_sq = np.empty(n - 1)
    innovation_vars = np.empty(n - 1)

    for t in range(1, n):
        # Predict
        x_pred = x_hat
        P_pred = P + Q
        # Innovation
        e = returns[t] - x_pred
        S = P_pred + R
        # Store
        innovations_sq[t - 1] = e * e
        innovation_vars[t - 1] = S
        # Update
        K = P_pred / S
        x_hat = x_pred + K * e
        P = (1.0 - K) * P_pred

    # NIS over second half (after filter convergence)
    half = (n - 1) // 2
    if half < 5:
        return np.nan

    nis_values = innovations_sq[half:] / innovation_vars[half:]
    return float(np.mean(nis_values))


def _ar_coefficient(returns: np.ndarray, order: int) -> float:
    """First AR coefficient from OLS fit of AR(p) model.

    r[t] = a1*r[t-1] + a2*r[t-2] + ... + ap*r[t-p] + eps

    Returns a1 (dominant lag coefficient).

    Complexity: O(n*p) for matrix build + O(p^3) for solve (negligible for p<=5).
    """
    n = len(returns)
    if n < order + 10:
        return np.nan

    # Build design matrix and target
    y = returns[order:]
    m = len(y)

    if order == 1:
        # Fast scalar path
        x = returns[order - 1 : n - 1]
        xx = np.dot(x, x)
        if abs(xx) < 1e-20:
            return np.nan
        xy = np.dot(x, y)
        return float(xy / xx)

    # General case: build X matrix
    X = np.empty((m, order))
    for k in range(order):
        X[:, k] = returns[order - k - 1 : n - k - 1]

    XtX = X.T @ X
    Xty = X.T @ y

    # Check for near-singular
    diag_prod = np.prod(np.diag(XtX))
    if abs(diag_prod) < 1e-40:
        return np.nan

    try:
        coeffs = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
        return np.nan

    return float(coeffs[0])


def _max_lyapunov(returns: np.ndarray, embed_dim: int, tau: int) -> float:
    """Maximum Lyapunov exponent via the Rosenstein et al. (1993) algorithm.

    1. Build delay-embedding vectors: v[i] = [r[i], r[i+tau], ..., r[i+(m-1)*tau]]
    2. For each point, find nearest neighbor (excluding temporal neighbors)
    3. Track log-divergence of each pair over time
    4. MLE = slope of mean log-divergence vs step number

    Complexity: O(n_embed^2) per window (brute-force nearest-neighbor).
    """
    n = len(returns)
    n_embed = n - (embed_dim - 1) * tau
    if n_embed < 20:
        return np.nan

    # Build embedding matrix
    embedded = np.empty((n_embed, embed_dim))
    for d in range(embed_dim):
        embedded[:, d] = returns[d * tau : d * tau + n_embed]

    max_steps = min(n_embed // 4, 10)
    if max_steps < 2:
        return np.nan

    divergence = np.zeros(max_steps)
    counts = np.zeros(max_steps, dtype=np.int64)
    min_sep = max(tau, 1)  # minimum temporal separation

    for i in range(n_embed - max_steps):
        # Find nearest neighbor
        min_dist = np.inf
        nn_idx = -1
        for j in range(n_embed - max_steps):
            if abs(i - j) <= min_sep:
                continue
            dist = 0.0
            for d in range(embed_dim):
                diff = embedded[i, d] - embedded[j, d]
                dist += diff * diff
            dist = np.sqrt(dist)
            if 1e-20 < dist < min_dist:
                min_dist = dist
                nn_idx = j

        if nn_idx < 0:
            continue

        # Track divergence
        for k in range(max_steps):
            ii = i + k
            jj = nn_idx + k
            if ii < n_embed and jj < n_embed:
                d_k = 0.0
                for d in range(embed_dim):
                    diff = embedded[ii, d] - embedded[jj, d]
                    d_k += diff * diff
                d_k = np.sqrt(d_k)
                if d_k > 1e-20:
                    divergence[k] += np.log(d_k)
                    counts[k] += 1

    # Average log-divergence
    valid_mask = counts > 0
    if np.sum(valid_mask) < 2:
        return np.nan

    steps = np.arange(max_steps, dtype=np.float64)[valid_mask]
    vals = (divergence[valid_mask]) / (counts[valid_mask].astype(np.float64))

    if len(steps) < 2:
        return np.nan

    # Linear regression for slope (MLE)
    n_pts = len(steps)
    sx = np.sum(steps)
    sy = np.sum(vals)
    sxy = np.dot(steps, vals)
    sx2 = np.sum(steps * steps)
    denom = n_pts * sx2 - sx * sx
    if abs(denom) < 1e-20:
        return np.nan

    slope = (n_pts * sxy - sx * sy) / denom
    return float(slope)


def _pid_error_signal(returns: np.ndarray, kp: float, ki: float, kd: float) -> float:
    """RMS of PID composite error signal on log-returns.

    Error = return - mean (P), cumulative error with exponential decay (I),
    first difference of error (D).  Output = RMS of (kp*P + ki*I + kd*D).

    Complexity: O(n).
    """
    n = len(returns)
    if n < 10:
        return np.nan

    # Reference: mean return (equilibrium)
    ref = np.mean(returns)
    errors = returns - ref

    # Integral term with exponential decay (anti-windup)
    decay = 0.95
    integral = 0.0
    integral_vals = np.empty(n)
    for t in range(n):
        integral = decay * integral + errors[t]
        integral_vals[t] = integral

    # Derivative term
    derivative = np.zeros(n)
    for t in range(1, n):
        derivative[t] = errors[t] - errors[t - 1]

    # PID composite
    pid = kp * errors + ki * integral_vals + kd * derivative

    valid = pid[~np.isnan(pid)]
    if len(valid) < 5:
        return np.nan

    return float(np.sqrt(np.mean(valid * valid)))


def _prediction_error_decomp(returns: np.ndarray, forecast_horizon: int) -> float:
    """Bias ratio from bias-variance decomposition of linear prediction errors.

    Train linear model on first 2/3 of window, evaluate on last 1/3.
    Output = bias^2 / (bias^2 + variance), bounded [0, 1].

    High = systematic model failure (regime change).
    Low  = noise-dominated errors (efficient market).

    Complexity: O(n).
    """
    n = len(returns)
    train_n = (2 * n) // 3
    eval_n = n - train_n
    if train_n < 10 or eval_n < 5:
        return np.nan

    train = returns[:train_n]
    evaluate = returns[train_n:]

    # Fit linear model on training data: y = slope * x + intercept
    x_train = np.arange(train_n, dtype=np.float64)
    sx = np.sum(x_train)
    sy = np.sum(train)
    sxy = np.dot(x_train, train)
    sx2 = np.sum(x_train * x_train)
    denom = train_n * sx2 - sx * sx
    if abs(denom) < 1e-20:
        return np.nan

    slope = (train_n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / train_n

    # Predict for evaluation portion
    x_eval = np.arange(train_n, train_n + eval_n, dtype=np.float64)
    predicted = slope * x_eval + intercept

    # Prediction errors
    errors = evaluate - predicted

    # Bias-variance decomposition
    bias = np.mean(errors)
    bias_sq = bias * bias
    variance = np.var(errors, ddof=1) if len(errors) > 1 else 0.0
    total = bias_sq + variance

    if total < 1e-20:
        return np.nan

    return float(bias_sq / total)


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------


@dataclass
@sf_component(name="stat/kalman_innovation")
class KalmanInnovationStat(Feature):
    """Rolling Kalman Innovation Statistic (Harvey, 1989).

    Runs a 1-D local-level (random walk + noise) Kalman filter on
    log-returns within each rolling window and outputs the Normalized
    Innovation Statistic (NIS).

    Model:
        State:       x[t] = x[t-1] + w,   w ~ N(0, Q)
        Observation: z[t] = x[t]   + v,   v ~ N(0, R)

    NIS = mean(e[t]^2 / S[t])  over the second half of the window,
    where e[t] is the innovation (prediction error) and S[t] is the
    innovation variance.

    Under correct model assumptions, NIS ~ 1.0 (chi-squared test).

    Interpretation:
        - NIS ~ 1.0: random-walk model fits well (stable regime)
        - NIS >> 1.0: filter is surprised — innovations exceed expectations.
          Signals model breakdown / regime shift / structural break.
        - NIS << 1.0: filter is over-estimating uncertainty.
          Returns are more predictable than expected (compression).
        - NIS spikes correspond to moments where the optimal filter's
          assumptions are violated — early regime transition detector.

    Parameters:
        source_col: Price column to compute log-returns from
        period: Rolling window size
        process_noise: Q/R ratio (process noise relative to observation noise)
        normalized: If True, apply rolling z-score normalization

    Reference: Harvey, A.C. (1989). Forecasting, Structural Time Series
    Models and the Kalman Filter. Cambridge University Press.
    """

    source_col: str = "close"
    period: int = 100
    process_noise: float = 0.01
    normalized: bool = False
    norm_period: int | None = None

    requires = ["{source_col}"]
    outputs = ["{source_col}_kalman_innov_{period}"]

    def __post_init__(self):
        if self.period < 30:
            raise ValueError(
                f"period must be >= 30 for Kalman filter convergence, got {self.period}"
            )
        if self.process_noise <= 0 or self.process_noise > 10.0:
            raise ValueError(
                f"process_noise must be in (0, 10], got {self.process_noise}"
            )

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        values = df[self.source_col].to_numpy()
        n = len(values)

        log_ret = _log_returns(values)

        result = np.full(n, np.nan)
        for i in range(self.period, n):
            window = log_ret[i - self.period + 1 : i + 1]
            valid = window[~np.isnan(window)]
            if len(valid) >= 20:
                result[i] = _kalman_innovation_variance(valid, self.process_noise)

        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window

            norm_window = self.norm_period or get_norm_window(self.period)
            result = normalize_zscore(result, window=norm_window)

        suffix = "_norm" if self.normalized else ""
        col_name = f"{self.source_col}_kalman_innov_{self.period}{suffix}"
        return df.with_columns(pl.Series(name=col_name, values=result))

    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 100, "process_noise": 0.01},
        {"source_col": "close", "period": 200, "process_noise": 0.1},
        {
            "source_col": "close",
            "period": 100,
            "process_noise": 0.01,
            "normalized": True,
        },
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
@sf_component(name="stat/ar_coefficient")
class ARCoefficientStat(Feature):
    """Rolling AR Coefficient — System Identification (Ljung, 1999).

    Fits an AR(p) autoregressive model to the log-returns in each
    rolling window via ordinary least squares and outputs the first
    coefficient a1, which captures the dominant autoregressive dynamics.

    Model: r[t] = a1*r[t-1] + a2*r[t-2] + ... + ap*r[t-p] + epsilon

    Tracking a1 over time implements "system identification" — the core
    control-theoretic idea of continuously re-estimating the parameters
    of the underlying dynamical system as it evolves.

    Interpretation:
        - a1 > 0: positive autocorrelation — momentum / trending
        - a1 < 0: negative autocorrelation — mean-reversion
        - a1 ~ 0: no linear predictability from immediate past
        - |a1| changing over time: autoregressive structure is shifting,
          useful for adaptive strategy selection
        - a1 approaching ±1: extreme serial dependence (rare, signals
          instability in the AR model)

    Parameters:
        source_col: Price column to compute log-returns from
        period: Rolling window size
        ar_order: AR model order (1-5)
        normalized: If True, apply rolling z-score normalization

    Reference: Ljung, L. (1999). System Identification: Theory for
    the User. 2nd ed. Prentice Hall.
    """

    source_col: str = "close"
    period: int = 100
    ar_order: int = 1
    normalized: bool = False
    norm_period: int | None = None

    requires = ["{source_col}"]
    outputs = ["{source_col}_ar_coeff_{period}"]

    def __post_init__(self):
        if self.ar_order < 1 or self.ar_order > 5:
            raise ValueError(f"ar_order must be in [1, 5], got {self.ar_order}")
        min_period = self.ar_order * 10 + 10
        if self.period < min_period:
            raise ValueError(
                f"period must be >= {min_period} for ar_order={self.ar_order}, "
                f"got {self.period}"
            )

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        values = df[self.source_col].to_numpy()
        n = len(values)

        log_ret = _log_returns(values)

        result = np.full(n, np.nan)
        for i in range(self.period, n):
            window = log_ret[i - self.period + 1 : i + 1]
            valid = window[~np.isnan(window)]
            if len(valid) >= self.ar_order + 10:
                result[i] = _ar_coefficient(valid, self.ar_order)

        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window

            norm_window = self.norm_period or get_norm_window(self.period)
            result = normalize_zscore(result, window=norm_window)

        suffix = "_norm" if self.normalized else ""
        col_name = f"{self.source_col}_ar_coeff_{self.period}{suffix}"
        return df.with_columns(pl.Series(name=col_name, values=result))

    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 100, "ar_order": 1},
        {"source_col": "close", "period": 200, "ar_order": 3},
        {"source_col": "close", "period": 100, "ar_order": 1, "normalized": True},
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
@sf_component(name="stat/lyapunov_exponent")
class LyapunovExponentStat(Feature):
    """Rolling Maximum Lyapunov Exponent (Rosenstein et al., 1993).

    Estimates the maximum Lyapunov exponent (MLE) from log-returns using
    time-delay embedding and nearest-neighbor divergence tracking.

    Method:
        1. Construct delay-embedding vectors from log-returns
        2. For each point, find its nearest neighbor in phase space
           (excluding temporal neighbors)
        3. Track how pairs diverge over time: d(k) = ||v[i+k] - v[nn(i)+k]||
        4. Average log(d(k)) across all reference points
        5. MLE = slope of mean log-divergence vs step number k

    This is fundamentally different from DFA:
        - DFA measures long-range correlation structure (memory)
        - Lyapunov exponent measures deterministic chaos (trajectory divergence)

    Interpretation:
        - MLE > 0: sensitive dependence on initial conditions (chaos).
          Nearby market states diverge exponentially — inherently
          unpredictable beyond short horizons.
        - MLE ~ 0: edge of chaos / quasi-periodic behavior.
        - MLE < 0: trajectories converge — system attracted to fixed point
          or limit cycle. Strong mean-reversion / equilibrium-seeking.
        - MLE transition from negative to positive: market shifting from
          stable to chaotic regime — early warning signal.

    Parameters:
        source_col: Price column to compute log-returns from
        period: Rolling window size (larger = more reliable, slower)
        embed_dim: Embedding dimension (typically 2-5)
        tau: Time delay for embedding (typically 1)
        normalized: If True, apply rolling z-score normalization

    Reference: Rosenstein, M.T., Collins, J.J. & De Luca, C.J. (1993).
    A practical method for calculating largest Lyapunov exponents from
    small data sets. Physica D, 65(1-2), 117-134.
    """

    source_col: str = "close"
    period: int = 200
    embed_dim: int = 3
    tau: int = 1
    normalized: bool = False
    norm_period: int | None = None

    requires = ["{source_col}"]
    outputs = ["{source_col}_lyapunov_{period}"]

    def __post_init__(self):
        if self.period < 50:
            raise ValueError(
                f"period must be >= 50 for reliable Lyapunov estimation, "
                f"got {self.period}"
            )
        if self.embed_dim < 2 or self.embed_dim > 7:
            raise ValueError(f"embed_dim must be in [2, 7], got {self.embed_dim}")
        if self.tau < 1 or self.tau > 10:
            raise ValueError(f"tau must be in [1, 10], got {self.tau}")
        min_period = self.embed_dim * self.tau * 5
        if self.period < min_period:
            raise ValueError(
                f"period must be >= embed_dim * tau * 5 = {min_period}, "
                f"got {self.period}"
            )

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        values = df[self.source_col].to_numpy()
        n = len(values)

        log_ret = _log_returns(values)

        result = np.full(n, np.nan)
        for i in range(self.period, n):
            window = log_ret[i - self.period + 1 : i + 1]
            valid = window[~np.isnan(window)]
            min_len = (self.embed_dim - 1) * self.tau + 20
            if len(valid) >= min_len:
                result[i] = _max_lyapunov(valid, self.embed_dim, self.tau)

        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window

            norm_window = self.norm_period or get_norm_window(self.period)
            result = normalize_zscore(result, window=norm_window)

        suffix = "_norm" if self.normalized else ""
        col_name = f"{self.source_col}_lyapunov_{self.period}{suffix}"
        return df.with_columns(pl.Series(name=col_name, values=result))

    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 200, "embed_dim": 3, "tau": 1},
        {"source_col": "close", "period": 300, "embed_dim": 4, "tau": 2},
        {"source_col": "close", "period": 200, "embed_dim": 3, "normalized": True},
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
@sf_component(name="stat/pid_error")
class PIDErrorStat(Feature):
    """Rolling PID Error Signal (Astrom & Murray, 2008).

    Implements a PID (Proportional-Integral-Derivative) tracking controller
    on log-returns. The "setpoint" is the mean return (equilibrium), and the
    output is the RMS of the composite PID error signal.

    Components:
        P (proportional): error = return - mean  (instantaneous deviation)
        I (integral):      cumulative error with exponential decay (sustained drift)
        D (derivative):    first difference of error (acceleration of deviation)

    PID signal = kp * P + ki * I + kd * D
    Output = RMS(PID signal) over the window

    Unlike a simple z-score, the PID formulation captures multi-scale
    deviation dynamics: level (P), persistence (I), and rate of change (D)
    in a single composite signal.

    Interpretation:
        - High PID error: price deviating strongly, persistently, and/or
          rapidly from equilibrium — breakout or regime shift.
        - Low PID error: price well-tracked by equilibrium — stable,
          mean-reverting behavior.
        - Spikes indicate moments where all three PID components align,
          signaling the strongest form of deviation from equilibrium.

    Parameters:
        source_col: Price column to compute log-returns from
        period: Rolling window size
        kp: Proportional gain (deviation magnitude)
        ki: Integral gain (accumulated drift)
        kd: Derivative gain (deviation acceleration)
        normalized: If True, apply rolling z-score normalization

    Reference: Astrom, K.J. & Murray, R.M. (2008). Feedback Systems:
    An Introduction for Scientists and Engineers. Princeton University Press.
    """

    source_col: str = "close"
    period: int = 100
    kp: float = 1.0
    ki: float = 0.1
    kd: float = 0.05
    normalized: bool = False
    norm_period: int | None = None

    requires = ["{source_col}"]
    outputs = ["{source_col}_pid_error_{period}"]

    def __post_init__(self):
        if self.period < 20:
            raise ValueError(f"period must be >= 20, got {self.period}")
        if self.kp < 0 or self.ki < 0 or self.kd < 0:
            raise ValueError(
                f"gains must be non-negative, got kp={self.kp}, ki={self.ki}, kd={self.kd}"
            )
        if self.kp == 0 and self.ki == 0 and self.kd == 0:
            raise ValueError("at least one gain (kp, ki, kd) must be > 0")

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        values = df[self.source_col].to_numpy()
        n = len(values)

        log_ret = _log_returns(values)

        result = np.full(n, np.nan)
        for i in range(self.period, n):
            window = log_ret[i - self.period + 1 : i + 1]
            valid = window[~np.isnan(window)]
            if len(valid) >= 10:
                result[i] = _pid_error_signal(valid, self.kp, self.ki, self.kd)

        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window

            norm_window = self.norm_period or get_norm_window(self.period)
            result = normalize_zscore(result, window=norm_window)

        suffix = "_norm" if self.normalized else ""
        col_name = f"{self.source_col}_pid_error_{self.period}{suffix}"
        return df.with_columns(pl.Series(name=col_name, values=result))

    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 100, "kp": 1.0, "ki": 0.1, "kd": 0.05},
        {"source_col": "close", "period": 200, "kp": 1.0, "ki": 0.5, "kd": 0.1},
        {
            "source_col": "close",
            "period": 100,
            "kp": 1.0,
            "ki": 0.1,
            "kd": 0.05,
            "normalized": True,
        },
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
@sf_component(name="stat/prediction_error_decomp")
class PredictionErrorDecompositionStat(Feature):
    """Rolling Prediction Error Decomposition (Geman et al., 1992).

    Decomposes the prediction error of a linear model into bias and variance
    components (bias-variance decomposition). In control theory, this
    corresponds to analyzing whether system identification errors are due
    to systematic misspecification (bias) or stochastic noise (variance).

    Method:
        1. Split rolling window into training (first 2/3) and evaluation (last 1/3)
        2. Fit linear model (returns ~ linear trend) on training portion
        3. Generate predictions for evaluation portion
        4. Compute bias = mean(error)^2 and variance = var(error)
        5. Output = bias / (bias + variance), bounded [0, 1]

    Interpretation:
        - Bias ratio ~ 0: errors dominated by variance (noise). The linear
          model captures the trend correctly but cannot predict randomness.
          Market is "efficient" relative to the simple model.
        - Bias ratio ~ 1: errors dominated by systematic bias. The model
          is fundamentally wrong — market dynamics have shifted. Strong
          regime change signal.
        - Transition low → high: dynamics changing faster than model adapts.
          Early warning of structural break.
        - Transition high → low: market settling into new regime that
          the model is beginning to capture.

    Parameters:
        source_col: Price column to compute log-returns from
        period: Rolling window size
        forecast_horizon: Steps-ahead prediction (1-5)
        normalized: If True, apply rolling z-score normalization

    Reference: Geman, S., Bienenstock, E. & Doursat, R. (1992). Neural
    Networks and the Bias/Variance Dilemma. Neural Computation, 4(1), 1-58.
    """

    source_col: str = "close"
    period: int = 100
    forecast_horizon: int = 1
    normalized: bool = False
    norm_period: int | None = None

    requires = ["{source_col}"]
    outputs = ["{source_col}_pred_err_decomp_{period}"]

    def __post_init__(self):
        if self.period < 30:
            raise ValueError(f"period must be >= 30, got {self.period}")
        if self.forecast_horizon < 1 or self.forecast_horizon > 5:
            raise ValueError(
                f"forecast_horizon must be in [1, 5], got {self.forecast_horizon}"
            )

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        values = df[self.source_col].to_numpy()
        n = len(values)

        log_ret = _log_returns(values)

        result = np.full(n, np.nan)
        for i in range(self.period, n):
            window = log_ret[i - self.period + 1 : i + 1]
            valid = window[~np.isnan(window)]
            if len(valid) >= 20:
                result[i] = _prediction_error_decomp(valid, self.forecast_horizon)

        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window

            norm_window = self.norm_period or get_norm_window(self.period)
            result = normalize_zscore(result, window=norm_window)

        suffix = "_norm" if self.normalized else ""
        col_name = f"{self.source_col}_pred_err_decomp_{self.period}{suffix}"
        return df.with_columns(pl.Series(name=col_name, values=result))

    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 100, "forecast_horizon": 1},
        {"source_col": "close", "period": 200, "forecast_horizon": 3},
        {
            "source_col": "close",
            "period": 100,
            "forecast_horizon": 1,
            "normalized": True,
        },
    ]

    @property
    def warmup(self) -> int:
        base = self.period * 5
        if self.normalized:
            from signalflow.ta._normalization import get_norm_window

            norm_window = self.norm_period or get_norm_window(self.period)
            return base + norm_window
        return base

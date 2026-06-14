"""Extended regression-based features.

Goes beyond the basic LinRegSlopeStat/InterceptStat/ResidualStat (in regression.py):
- Configurable-window OLS slope/intercept/R²/residual std
- Slope acceleration and slope ratio across timescales
- Polynomial-fit residuals
- Higher moments of residuals (skew, kurtosis)
- Normalized variants

All features added from sf-profit iter-15/18/20 feature research.
"""
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import polars as pl

from signalflow.ta._compat import Feature


def _linreg_slope_intercept(c: np.ndarray, w: int):
    """Numpy helper: per-window (slope, intercept) over array c with window w.

    Returns (slope_arr, intercept_arr, windows, x_centered, y_mean).
    """
    from numpy.lib.stride_tricks import sliding_window_view as swv
    x = np.arange(w, dtype=np.float64)
    xm = x.mean()
    xc = x - xm
    xv = (xc ** 2).sum()
    windows = swv(c, window_shape=w)
    ym = windows.mean(axis=1)
    yc = windows - ym[:, None]
    slope = (yc * xc).sum(axis=1) / xv
    intercept = ym - slope * xm
    return slope, intercept, windows, xc, ym


@dataclass
class LinRegSlopeWindow(Feature):
    """OLS slope of close ~ t over configurable window. Trend strength."""
    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["linreg_slope_{window}"]
    window: int = 240

    def compute_pair(self, df):
        out = f"linreg_slope_{self.window}"
        c = df.get_column("close").to_numpy().astype(np.float64)
        n = len(c)
        if n < self.window:
            return df.with_columns(pl.Series(out, np.full(n, np.nan, dtype=np.float32)))
        slope, _, _, _, _ = _linreg_slope_intercept(c, self.window)
        full = np.full(n, np.nan, dtype=np.float32)
        full[self.window - 1:] = slope.astype(np.float32)
        return df.with_columns(pl.Series(out, full))


@dataclass
class LinRegR2(Feature):
    """R² of OLS close ~ t over window. Fit quality / trend coherence."""
    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["linreg_r2_{window}"]
    window: int = 240

    def compute_pair(self, df):
        out = f"linreg_r2_{self.window}"
        c = df.get_column("close").to_numpy().astype(np.float64)
        n = len(c)
        if n < self.window:
            return df.with_columns(pl.Series(out, np.full(n, np.nan, dtype=np.float32)))
        from numpy.lib.stride_tricks import sliding_window_view as swv
        w = self.window
        x = np.arange(w, dtype=np.float64)
        xm = x.mean()
        xv = ((x - xm) ** 2).sum()
        windows = swv(c, window_shape=w)
        ym = windows.mean(axis=1)
        yv = ((windows - ym[:, None]) ** 2).sum(axis=1)
        xy = (windows * x).sum(axis=1) - w * ym * xm
        slopes = xy / xv
        ss_reg = slopes ** 2 * xv
        r2 = ss_reg / (yv + 1e-12)
        full = np.full(n, np.nan, dtype=np.float32)
        full[w - 1:] = r2.astype(np.float32)
        return df.with_columns(pl.Series(out, full))


@dataclass
class LinRegResidualStd(Feature):
    """Std of residuals from OLS close ~ t fit. Non-trend volatility."""
    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["linreg_resstd_{window}"]
    window: int = 240

    def compute_pair(self, df):
        out = f"linreg_resstd_{self.window}"
        c = df.get_column("close").to_numpy().astype(np.float64)
        n = len(c)
        if n < self.window:
            return df.with_columns(pl.Series(out, np.full(n, np.nan, dtype=np.float32)))
        slope, intercept, windows, _, _ = _linreg_slope_intercept(c, self.window)
        x = np.arange(self.window, dtype=np.float64)
        pred = slope[:, None] * x + intercept[:, None]
        res = windows - pred
        rstd = res.std(axis=1)
        full = np.full(n, np.nan, dtype=np.float32)
        full[self.window - 1:] = rstd.astype(np.float32)
        return df.with_columns(pl.Series(out, full))


@dataclass
class LinRegSlopeChange(Feature):
    """slope_now − slope_lag. Trend acceleration."""
    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["linreg_slopechg_{window}_{lag}"]
    window: int = 240
    lag: int = 60

    def compute_pair(self, df):
        out = f"linreg_slopechg_{self.window}_{self.lag}"
        c = df.get_column("close").to_numpy().astype(np.float64)
        n = len(c)
        if n < self.window + self.lag:
            return df.with_columns(pl.Series(out, np.full(n, np.nan, dtype=np.float32)))
        slope, _, _, _, _ = _linreg_slope_intercept(c, self.window)
        full = np.full(n, np.nan)
        full[self.window - 1:] = slope
        shifted = np.full(n, np.nan)
        shifted[self.lag:] = full[:-self.lag]
        chg = (full - shifted).astype(np.float32)
        return df.with_columns(pl.Series(out, chg))


@dataclass
class LinRegSlopeAcceleration(Feature):
    """(slope_now − slope_lag) / lag. Normalized second derivative."""
    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["linreg_slope_accel_{window}_{lag}"]
    window: int = 240
    lag: int = 60

    def compute_pair(self, df):
        out = f"linreg_slope_accel_{self.window}_{self.lag}"
        c = df.get_column("close").to_numpy().astype(np.float64)
        n = len(c)
        if n < self.window + self.lag * 2:
            return df.with_columns(pl.Series(out, np.full(n, np.nan, dtype=np.float32)))
        slope, _, _, _, _ = _linreg_slope_intercept(c, self.window)
        full = np.full(n, np.nan)
        full[self.window - 1:] = slope
        prev = np.full(n, np.nan)
        prev[self.lag:] = full[:-self.lag]
        accel = ((full - prev) / self.lag).astype(np.float32)
        return df.with_columns(pl.Series(out, accel))


@dataclass
class Poly2ResidualStd(Feature):
    """Std of residuals from 2nd-order polynomial fit. Non-linearity measure."""
    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["poly2_resstd_{window}"]
    window: int = 240

    def compute_pair(self, df):
        out = f"poly2_resstd_{self.window}"
        c = df.get_column("close").to_numpy().astype(np.float64)
        n = len(c)
        if n < self.window:
            return df.with_columns(pl.Series(out, np.full(n, np.nan, dtype=np.float32)))
        from numpy.lib.stride_tricks import sliding_window_view as swv
        w = self.window
        x = np.arange(w, dtype=np.float64)
        X = np.column_stack([np.ones(w), x, x * x])
        XtX_inv = np.linalg.inv(X.T @ X)
        pinv = XtX_inv @ X.T
        H = X @ pinv
        windows = swv(c, window_shape=w)
        IH = np.eye(w) - H
        residuals = windows @ IH.T
        rstd = residuals.std(axis=1)
        full = np.full(n, np.nan, dtype=np.float32)
        full[w - 1:] = rstd.astype(np.float32)
        return df.with_columns(pl.Series(out, full))


@dataclass
class LinRegResidualSkew(Feature):
    """Skewness of residuals from linear fit."""
    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["linreg_resskew_{window}"]
    window: int = 480

    def compute_pair(self, df):
        out = f"linreg_resskew_{self.window}"
        c = df.get_column("close").to_numpy().astype(np.float64)
        n = len(c)
        if n < self.window:
            return df.with_columns(pl.Series(out, np.full(n, np.nan, dtype=np.float32)))
        slope, intercept, windows, _, _ = _linreg_slope_intercept(c, self.window)
        x = np.arange(self.window, dtype=np.float64)
        pred = slope[:, None] * x + intercept[:, None]
        res = windows - pred
        rm = res.mean(axis=1)
        rstd = res.std(axis=1)
        third = ((res - rm[:, None]) ** 3).mean(axis=1)
        skew = third / (rstd ** 3 + 1e-12)
        full = np.full(n, np.nan, dtype=np.float32)
        full[self.window - 1:] = skew.astype(np.float32)
        return df.with_columns(pl.Series(out, full))


@dataclass
class LinRegResidualKurtosis(Feature):
    """Excess kurtosis of residuals. Heavy-tail regime in deviations from trend."""
    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["linreg_reskurt_{window}"]
    window: int = 480

    def compute_pair(self, df):
        out = f"linreg_reskurt_{self.window}"
        c = df.get_column("close").to_numpy().astype(np.float64)
        n = len(c)
        if n < self.window:
            return df.with_columns(pl.Series(out, np.full(n, np.nan, dtype=np.float32)))
        slope, intercept, windows, _, _ = _linreg_slope_intercept(c, self.window)
        x = np.arange(self.window, dtype=np.float64)
        pred = slope[:, None] * x + intercept[:, None]
        res = windows - pred
        rm = res.mean(axis=1)
        rstd = res.std(axis=1)
        fourth = ((res - rm[:, None]) ** 4).mean(axis=1)
        kurt = fourth / (rstd ** 4 + 1e-12) - 3.0
        full = np.full(n, np.nan, dtype=np.float32)
        full[self.window - 1:] = kurt.astype(np.float32)
        return df.with_columns(pl.Series(out, full))


@dataclass
class LinRegSlopeRatio(Feature):
    """sign-preserving log-magnitude ratio of slope_short to slope_long."""
    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["linreg_slope_ratio_{short}_{long}"]
    short: int = 60
    long: int = 960

    def compute_pair(self, df):
        out = f"linreg_slope_ratio_{self.short}_{self.long}"
        c = df.get_column("close").to_numpy().astype(np.float64)
        n = len(c)
        if n < self.long:
            return df.with_columns(pl.Series(out, np.full(n, np.nan, dtype=np.float32)))
        ss, _, _, _, _ = _linreg_slope_intercept(c, self.short)
        sl, _, _, _, _ = _linreg_slope_intercept(c, self.long)
        full_s = np.full(n, np.nan); full_l = np.full(n, np.nan)
        full_s[self.short - 1:] = ss
        full_l[self.long - 1:] = sl
        sign_prod = np.sign(full_s) * np.sign(full_l)
        ratio = np.log(np.abs(full_s) + 1e-9) - np.log(np.abs(full_l) + 1e-9)
        return df.with_columns(pl.Series(out, (sign_prod * ratio).astype(np.float32)))


@dataclass
class LinRegInterceptNormalized(Feature):
    """LinReg intercept normalized by close.

    (intercept_N − close) / close - scale-invariant offset from trend line.
    """
    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["intercept_norm_{period}"]
    period: int = 240

    def compute_pair(self, df):
        out = f"intercept_norm_{self.period}"
        c = df.get_column("close").to_numpy().astype(np.float64)
        n = len(c)
        if n < self.period:
            return df.with_columns(pl.Series(out, np.full(n, np.nan, dtype=np.float32)))
        _, intercept, _, _, _ = _linreg_slope_intercept(c, self.period)
        full = np.full(n, np.nan, dtype=np.float64)
        full[self.period - 1:] = intercept
        norm = ((full - c) / np.maximum(c, 1e-9)).astype(np.float32)
        return df.with_columns(pl.Series(out, norm))


@dataclass
class LinRegSlopeNormalized(Feature):
    """LinReg slope normalized by close: slope/close. Scale-invariant trend strength."""
    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["slope_norm_{period}"]
    period: int = 240

    def compute_pair(self, df):
        out = f"slope_norm_{self.period}"
        c = df.get_column("close").to_numpy().astype(np.float64)
        n = len(c)
        if n < self.period:
            return df.with_columns(pl.Series(out, np.full(n, np.nan, dtype=np.float32)))
        slope, _, _, _, _ = _linreg_slope_intercept(c, self.period)
        full = np.full(n, np.nan, dtype=np.float64)
        full[self.period - 1:] = slope
        norm = (full / np.maximum(c, 1e-9)).astype(np.float32)
        return df.with_columns(pl.Series(out, norm))


@dataclass
class LinRegFitQuality(Feature):
    """1 − (residual_std / y_std). Robust fit quality alternative to R²."""
    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["fit_quality_{period}"]
    period: int = 240

    def compute_pair(self, df):
        out = f"fit_quality_{self.period}"
        c = df.get_column("close").to_numpy().astype(np.float64)
        n = len(c)
        if n < self.period:
            return df.with_columns(pl.Series(out, np.full(n, np.nan, dtype=np.float32)))
        slope, intercept, windows, _, _ = _linreg_slope_intercept(c, self.period)
        x = np.arange(self.period, dtype=np.float64)
        pred = slope[:, None] * x + intercept[:, None]
        res = windows - pred
        rstd = res.std(axis=1)
        ystd = windows.std(axis=1)
        q = 1 - (rstd / (ystd + 1e-9))
        full = np.full(n, np.nan, dtype=np.float32)
        full[self.period - 1:] = q.astype(np.float32)
        return df.with_columns(pl.Series(out, full))

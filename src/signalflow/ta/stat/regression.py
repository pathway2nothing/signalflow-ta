# src/signalflow/ta/stat/regression.py
"""Linear regression and correlation measures."""

from dataclasses import dataclass

import numpy as np
import polars as pl

from signalflow import sf_component
from signalflow.feature.base import Feature
from typing import ClassVar


@dataclass
@sf_component(name="stat/correlation")
class CorrelationStat(Feature):
    """Rolling Pearson Correlation between two columns.

    Measures linear relationship strength.

    Output: [-1, 1]
    -1: perfect negative
    +1: perfect positive
     0: no linear relationship

    Reference: https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
    """

    source_col: str = "close"
    target_col: str = "volume"
    period: int = 30

    requires = ["{source_col}", "{target_col}"]
    outputs = ["{source_col}_{target_col}_corr_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        out_col = f"{self.source_col}_{self.target_col}_corr_{self.period}"

        return df.with_columns(
            pl.corr(self.source_col, self.target_col)
            .rolling(window_size=self.period)
            .alias(out_col)
        )

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        x = df[self.source_col].to_numpy()
        y = df[self.target_col].to_numpy()
        n = len(x)

        corr = np.full(n, np.nan)
        for i in range(self.period - 1, n):
            x_win = x[i - self.period + 1 : i + 1]
            y_win = y[i - self.period + 1 : i + 1]

            if not (np.any(np.isnan(x_win)) or np.any(np.isnan(y_win))):
                c = np.corrcoef(x_win, y_win)[0, 1]
                corr[i] = c if not np.isnan(c) else 0

        return df.with_columns(
            pl.Series(
                name=f"{self.source_col}_{self.target_col}_corr_{self.period}",
                values=corr,
            )
        )

    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "target_col": "volume", "period": 30},
        {"source_col": "close", "target_col": "volume", "period": 60},
        {"source_col": "high", "target_col": "low", "period": 30},
    ]

    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable, reproducible output."""
        return (
            getattr(
                self, "period", getattr(self, "length", getattr(self, "window", 20))
            )
            * 5
        )


@dataclass
@sf_component(name="stat/beta")
class BetaStat(Feature):
    """Rolling Beta (regression slope) against benchmark.

    Beta = Cov(asset, benchmark) / Var(benchmark)

    Measures sensitivity to benchmark:
    β > 1: more volatile than benchmark
    β < 1: less volatile
    β < 0: inverse relationship

    Reference: https://en.wikipedia.org/wiki/Beta_(finance)
    """

    source_col: str = "close"
    benchmark_col: str = "benchmark"
    period: int = 30

    requires = ["{source_col}", "{benchmark_col}"]
    outputs = ["{source_col}_beta_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        x = df[self.benchmark_col].to_numpy()
        y = df[self.source_col].to_numpy()
        n = len(x)

        x_ret = np.diff(x, prepend=np.nan) / np.where(
            x[:-1] != 0, np.append(x[:-1], 1), 1
        )
        y_ret = np.diff(y, prepend=np.nan) / np.where(
            y[:-1] != 0, np.append(y[:-1], 1), 1
        )
        x_ret = np.append(np.nan, x_ret[1:])
        y_ret = np.append(np.nan, y_ret[1:])

        beta = np.full(n, np.nan)
        for i in range(self.period - 1, n):
            x_win = x_ret[i - self.period + 1 : i + 1]
            y_win = y_ret[i - self.period + 1 : i + 1]

            mask = ~(np.isnan(x_win) | np.isnan(y_win))
            if np.sum(mask) > 2:
                cov = np.cov(x_win[mask], y_win[mask])[0, 1]
                var = np.var(x_win[mask], ddof=1)
                if var > 1e-10:
                    beta[i] = cov / var

        return df.with_columns(
            pl.Series(name=f"{self.source_col}_beta_{self.period}", values=beta)
        )

    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "benchmark_col": "volume", "period": 30},
        {"source_col": "close", "benchmark_col": "volume", "period": 60},
        {"source_col": "close", "benchmark_col": "volume", "period": 120},
    ]

    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable, reproducible output."""
        return self.period * 5


@dataclass
@sf_component(name="stat/rsquared")
class RSquaredStat(Feature):
    """Rolling R-Squared (coefficient of determination).

    R² = 1 - SS_res / SS_tot

    Measures how well linear trend fits data.
    R² = 1: perfect fit
    R² = 0: no better than mean

    Reference: https://en.wikipedia.org/wiki/Coefficient_of_determination
    """

    source_col: str = "close"
    period: int = 30

    requires = ["{source_col}"]
    outputs = ["{source_col}_r2_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        values = df[self.source_col].to_numpy()
        n = len(values)

        r2 = np.full(n, np.nan)
        x = np.arange(self.period)

        for i in range(self.period - 1, n):
            y = values[i - self.period + 1 : i + 1]

            if not np.any(np.isnan(y)):
                # Linear regression
                coeffs = np.polyfit(x, y, 1)
                y_pred = np.polyval(coeffs, x)

                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)

                if ss_tot > 1e-10:
                    r2[i] = 1 - ss_res / ss_tot

        return df.with_columns(
            pl.Series(name=f"{self.source_col}_r2_{self.period}", values=r2)
        )

    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 30},
        {"source_col": "close", "period": 60},
        {"source_col": "close", "period": 120},
    ]

    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable, reproducible output."""
        return self.period * 5


@dataclass
@sf_component(name="stat/linreg_slope")
class LinRegSlopeStat(Feature):
    """Rolling Linear Regression Slope.

    Slope of best-fit line through window.
    Positive = uptrend, Negative = downtrend.
    Magnitude indicates trend strength.

    Formula:
        slope = Σ(x - x̄)(y - ȳ) / Σ(x - x̄)²

    Reference: https://en.wikipedia.org/wiki/Simple_linear_regression
    """

    source_col: str = "close"
    period: int = 30

    requires = ["{source_col}"]
    outputs = ["{source_col}_slope_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        values = df[self.source_col].to_numpy()
        n = len(values)

        slope = np.full(n, np.nan)
        x = np.arange(self.period)

        for i in range(self.period - 1, n):
            y = values[i - self.period + 1 : i + 1]
            if not np.any(np.isnan(y)):
                coeffs = np.polyfit(x, y, 1)
                slope[i] = coeffs[0]

        return df.with_columns(
            pl.Series(name=f"{self.source_col}_slope_{self.period}", values=slope)
        )

    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 30},
        {"source_col": "close", "period": 60},
        {"source_col": "close", "period": 120},
    ]

    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable, reproducible output."""
        return self.period * 5


@dataclass
@sf_component(name="stat/linreg_intercept")
class LinRegInterceptStat(Feature):
    """Rolling Linear Regression Intercept.

    Y-intercept of best-fit line.

    Reference: https://en.wikipedia.org/wiki/Simple_linear_regression
    """

    source_col: str = "close"
    period: int = 30

    requires = ["{source_col}"]
    outputs = ["{source_col}_intercept_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        values = df[self.source_col].to_numpy()
        n = len(values)

        intercept = np.full(n, np.nan)
        x = np.arange(self.period)

        for i in range(self.period - 1, n):
            y = values[i - self.period + 1 : i + 1]
            if not np.any(np.isnan(y)):
                coeffs = np.polyfit(x, y, 1)
                intercept[i] = coeffs[1]

        return df.with_columns(
            pl.Series(
                name=f"{self.source_col}_intercept_{self.period}", values=intercept
            )
        )

    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 30},
        {"source_col": "close", "period": 60},
        {"source_col": "close", "period": 120},
    ]

    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable, reproducible output."""
        return self.period * 5


@dataclass
@sf_component(name="stat/linreg_residual")
class LinRegResidualStat(Feature):
    """Rolling Linear Regression Residual.

    Distance from current value to regression line.

    residual = y_actual - y_predicted

    Positive: above trend
    Negative: below trend

    Useful for mean-reversion signals.
    """

    source_col: str = "close"
    period: int = 30

    requires = ["{source_col}"]
    outputs = ["{source_col}_residual_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        values = df[self.source_col].to_numpy()
        n = len(values)

        residual = np.full(n, np.nan)
        x = np.arange(self.period)

        for i in range(self.period - 1, n):
            y = values[i - self.period + 1 : i + 1]
            if not np.any(np.isnan(y)):
                coeffs = np.polyfit(x, y, 1)
                y_pred = coeffs[0] * (self.period - 1) + coeffs[1]
                residual[i] = y[-1] - y_pred

        return df.with_columns(
            pl.Series(name=f"{self.source_col}_residual_{self.period}", values=residual)
        )

    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 30},
        {"source_col": "close", "period": 60},
        {"source_col": "close", "period": 120},
    ]

    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable, reproducible output."""
        return self.period * 5

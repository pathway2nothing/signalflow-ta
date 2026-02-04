# src/signalflow/ta/stat/distribution.py
"""Distribution shape measures - position, moments, information."""
from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.stats import kurtosis as sp_kurtosis, skew as sp_skew

from signalflow import sf_component
from signalflow.feature.base import Feature
from typing import ClassVar


@dataclass
@sf_component(name="stat/median")
class MedianStat(Feature):
    """Rolling Median.
    
    Middle value in sorted window.
    More robust to outliers than mean.
    
    Reference: https://en.wikipedia.org/wiki/Median
    """
    
    source_col: str = "close"
    period: int = 30
    
    requires = ["{source_col}"]
    outputs = ["{source_col}_median_{period}"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            pl.col(self.source_col)
              .rolling_median(window_size=self.period)
              .alias(f"{self.source_col}_median_{self.period}")
        )
    
    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 30},
        {"source_col": "close", "period": 60},
        {"source_col": "close", "period": 240},
    ]



    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable, reproducible output."""
        return getattr(self, "period", getattr(self, "length", getattr(self, "window", 20))) * 5

@dataclass
@sf_component(name="stat/quantile")
class QuantileStat(Feature):
    """Rolling Quantile.
    
    Value below which q fraction of data falls.
    q=0.5 is median, q=0.25 is Q1, q=0.75 is Q3.
    
    Reference: https://en.wikipedia.org/wiki/Quantile
    """
    
    source_col: str = "close"
    period: int = 30
    q: float = 0.5
    
    requires = ["{source_col}"]
    outputs = ["{source_col}_q{q}_{period}"]
    
    def __post_init__(self):
        if not 0 < self.q < 1:
            raise ValueError(f"q must be in (0, 1), got {self.q}")
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        q_str = str(self.q).replace(".", "")
        return df.with_columns(
            pl.col(self.source_col)
              .rolling_quantile(quantile=self.q, window_size=self.period)
              .alias(f"{self.source_col}_q{q_str}_{self.period}")
        )
    
    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 30, "q": 0.5},
        {"source_col": "close", "period": 60, "q": 0.25},
        {"source_col": "close", "period": 240, "q": 0.75},
    ]

    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable, reproducible output."""
        return self.period * 5


@dataclass
@sf_component(name="stat/pctrank")
class PctRankStat(Feature):
    """Rolling Percentile Rank.
    
    PCTRANK = (count of values < current) / n * 100
    
    Where current value stands relative to window history.
    Output: 0-100 (0 = lowest in window, 100 = highest).
    
    Reference: https://en.wikipedia.org/wiki/Percentile_rank
    """
    
    source_col: str = "close"
    period: int = 30
    
    requires = ["{source_col}"]
    outputs = ["{source_col}_pctrank_{period}"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        values = df[self.source_col].to_numpy()
        n = len(values)
        
        pctrank = np.full(n, np.nan)
        for i in range(self.period - 1, n):
            window = values[i - self.period + 1:i + 1]
            current = values[i]
            pctrank[i] = (np.sum(window < current) / self.period) * 100
        
        return df.with_columns(
            pl.Series(name=f"{self.source_col}_pctrank_{self.period}", values=pctrank)
        )
    
    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 30},
        {"source_col": "close", "period": 60},
        {"source_col": "close", "period": 240},
    ]

    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable, reproducible output."""
        return self.period * 5


@dataclass
@sf_component(name="stat/minmax")
class MinMaxStat(Feature):
    """Rolling Min-Max Normalization.
    
    MINMAX = (x - min) / (max - min)
    
    Scales to [0, 1] range within window.
    0 = at window minimum, 1 = at window maximum.
    
    Reference: https://en.wikipedia.org/wiki/Feature_scaling
    """
    
    source_col: str = "close"
    period: int = 30
    
    requires = ["{source_col}"]
    outputs = ["{source_col}_minmax_{period}"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        col = pl.col(self.source_col)
        min_val = col.rolling_min(window_size=self.period)
        max_val = col.rolling_max(window_size=self.period)
        
        return df.with_columns(
            ((col - min_val) / (max_val - min_val))
            .alias(f"{self.source_col}_minmax_{self.period}")
        )
    
    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 30},
        {"source_col": "close", "period": 60},
        {"source_col": "close", "period": 240},
    ]

    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable, reproducible output."""
        return self.period * 5

@dataclass
@sf_component(name="stat/skew")
class SkewStat(Feature):
    """Rolling Skewness.
    
    SKEW = E[(x - mean)³] / σ³
    
    Measures asymmetry of distribution:
    - skew > 0: right tail longer (positive skew)
    - skew < 0: left tail longer (negative skew)
    - skew ≈ 0: symmetric
    
    Reference: https://en.wikipedia.org/wiki/Skewness
    """
    
    source_col: str = "close"
    period: int = 30
    
    requires = ["{source_col}"]
    outputs = ["{source_col}_skew_{period}"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        values = df[self.source_col].to_numpy()
        n = len(values)

        skew = np.full(n, np.nan)
        for i in range(self.period - 1, n):
            window = values[i - self.period + 1:i + 1]
            if not np.any(np.isnan(window)):
                skew[i] = sp_skew(window, bias=False)

        return df.with_columns(
            pl.Series(name=f"{self.source_col}_skew_{self.period}", values=skew)
        )
    
    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 30},
        {"source_col": "close", "period": 60},
        {"source_col": "close", "period": 240},
    ]

    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable, reproducible output."""
        return self.period * 5


@dataclass
@sf_component(name="stat/kurtosis")
class KurtosisStat(Feature):
    """Rolling Excess Kurtosis.
    
    KURT = E[(x - mean)⁴] / σ⁴ - 3
    
    Measures tail heaviness (outlier propensity):
    - kurt > 0: heavy tails, leptokurtic (more outliers)
    - kurt < 0: light tails, platykurtic (fewer outliers)
    - kurt ≈ 0: normal distribution, mesokurtic
    
    Reference: https://en.wikipedia.org/wiki/Kurtosis
    """
    
    source_col: str = "close"
    period: int = 30
    
    requires = ["{source_col}"]
    outputs = ["{source_col}_kurt_{period}"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        values = df[self.source_col].to_numpy()
        n = len(values)
        
        kurt = np.full(n, np.nan)
        for i in range(self.period - 1, n):
            window = values[i - self.period + 1:i + 1]
            if not np.any(np.isnan(window)):
                kurt[i] = sp_kurtosis(window, fisher=True)
        
        return df.with_columns(
            pl.Series(name=f"{self.source_col}_kurt_{self.period}", values=kurt)
        )
    
    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 30},
        {"source_col": "close", "period": 60},
        {"source_col": "close", "period": 240},
    ]

    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable, reproducible output."""
        return self.period * 5

@dataclass
@sf_component(name="stat/entropy")
class EntropyStat(Feature):
    """Rolling Shannon Entropy.
    
    H = -Σ(p * log(p))  where p = x / Σx
    
    Measures unpredictability/randomness:
    - Higher entropy = more random/uniform
    - Lower entropy = more predictable/concentrated
    
    Note: source_col values must be positive.
    
    Reference: https://en.wikipedia.org/wiki/Entropy_(information_theory)
    """
    
    source_col: str = "close"
    period: int = 10
    base: float = 2.0
    
    requires = ["{source_col}"]
    outputs = ["{source_col}_entropy_{period}"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        values = df[self.source_col].to_numpy()
        n = len(values)
        
        entropy = np.full(n, np.nan)
        log_base = np.log(self.base)
        
        for i in range(self.period - 1, n):
            window = values[i - self.period + 1:i + 1]
            window_sum = np.sum(window)
            if window_sum > 0:
                p = window / window_sum
                p = p[p > 0]
                entropy[i] = -np.sum(p * np.log(p) / log_base)
        
        return df.with_columns(
            pl.Series(name=f"{self.source_col}_entropy_{self.period}", values=entropy)
        )

    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 10, "base": 2.0},
        {"source_col": "close", "period": 30, "base": 2.0},
        {"source_col": "close", "period": 60, "base": 2.718281828},  # ≈ e для natural log
    ]

    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable, reproducible output."""
        return self.period * 5

@dataclass
@sf_component(name="stat/jarque_bera")
class JarqueBeraStat(Feature):
    """Rolling Jarque-Bera Test Statistic.
    
    JB = n/6 * (S² + K²/4)
    
    Tests for normality using skewness (S) and kurtosis (K).
    Higher values = less normal distribution.
    
    Useful for detecting regime changes.
    
    Reference: https://en.wikipedia.org/wiki/Jarque%E2%80%93Bera_test
    """
    
    source_col: str = "close"
    period: int = 30
    
    requires = ["{source_col}"]
    outputs = ["{source_col}_jb_{period}"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        values = df[self.source_col].to_numpy()
        n_vals = len(values)
        
        jb = np.full(n_vals, np.nan)
        for i in range(self.period - 1, n_vals):
            window = values[i - self.period + 1:i + 1]
            if not np.any(np.isnan(window)):
                n = len(window)
                mean = np.mean(window)
                std = np.std(window, ddof=1)
                
                if std > 1e-10:
                    # Skewness
                    skew = np.mean(((window - mean) / std) ** 3)
                    # Excess kurtosis  
                    kurt = np.mean(((window - mean) / std) ** 4) - 3
                    
                    jb[i] = (n / 6) * (skew ** 2 + (kurt ** 2) / 4)
        
        return df.with_columns(
            pl.Series(name=f"{self.source_col}_jb_{self.period}", values=jb)
        )

    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 30},
        {"source_col": "close", "period": 60},
        {"source_col": "close", "period": 240},
    ]

    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable, reproducible output."""
        return self.period * 5

@dataclass
@sf_component(name="stat/mode_distance")
class ModeDistanceStat(Feature):
    """Rolling Distance from Mode (most frequent value region).
    
    Approximates mode using histogram and returns distance 
    from current value to mode bin center.
    
    Useful for mean-reversion signals.
    """
    
    source_col: str = "close"
    period: int = 30
    n_bins: int = 10
    
    requires = ["{source_col}"]
    outputs = ["{source_col}_mode_dist_{period}"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        values = df[self.source_col].to_numpy()
        n = len(values)
        
        mode_dist = np.full(n, np.nan)
        for i in range(self.period - 1, n):
            window = values[i - self.period + 1:i + 1]
            
            if not np.any(np.isnan(window)):
                # Histogram to find mode
                hist, bin_edges = np.histogram(window, bins=self.n_bins)
                mode_bin = np.argmax(hist)
                mode_center = (bin_edges[mode_bin] + bin_edges[mode_bin + 1]) / 2
                
                mode_dist[i] = values[i] - mode_center
        
        return df.with_columns(
            pl.Series(name=f"{self.source_col}_mode_dist_{self.period}", values=mode_dist)
        )
    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 30, "n_bins": 10},
        {"source_col": "close", "period": 60, "n_bins": 10},
        {"source_col": "close", "period": 120, "n_bins": 15},
    ]

    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable, reproducible output."""
        return self.period * 5

@dataclass
@sf_component(name="stat/above_mean_ratio")
class AboveMeanRatioStat(Feature):
    """Rolling Ratio of Values Above Mean.
    
    RATIO = count(x > mean) / n
    
    Indicates distribution balance:
    - ratio > 0.5: more values above mean (negative skew tendency)
    - ratio < 0.5: more values below mean (positive skew tendency)
    - ratio ≈ 0.5: balanced
    """
    
    source_col: str = "close"
    period: int = 30
    
    requires = ["{source_col}"]
    outputs = ["{source_col}_above_mean_{period}"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        values = df[self.source_col].to_numpy()
        n = len(values)
        
        ratio = np.full(n, np.nan)
        for i in range(self.period - 1, n):
            window = values[i - self.period + 1:i + 1]
            mean = np.mean(window)
            ratio[i] = np.sum(window > mean) / self.period
        
        return df.with_columns(
            pl.Series(name=f"{self.source_col}_above_mean_{self.period}", values=ratio)
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

# src/signalflow/ta/stat/dispersion.py
"""Dispersion measures - spread around central tendency."""
from dataclasses import dataclass

import numpy as np
import polars as pl

from signalflow import sf_component
from signalflow.feature.base import Feature
from typing import ClassVar


@dataclass
@sf_component(name="stat/variance")
class VarianceStat(Feature):
    """Rolling Variance.
    
    VAR = Σ(x - mean)² / (n - ddof)
    
    Reference: https://en.wikipedia.org/wiki/Variance
    """
    
    source_col: str = "close"
    period: int = 30
    ddof: int = 1
    
    requires = ["{source_col}"]
    outputs = ["{source_col}_var_{period}"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            pl.col(self.source_col)
              .rolling_var(window_size=self.period, ddof=self.ddof)
              .alias(f"{self.source_col}_var_{self.period}")
        )
    
    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 30, "ddof": 1},
        {"source_col": "close", "period": 60, "ddof": 1},
        {"source_col": "close", "period": 240, "ddof": 0},
    ]



    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable, reproducible output."""
        return getattr(self, "period", getattr(self, "length", getattr(self, "window", 20))) * 5

@dataclass
@sf_component(name="stat/std")
class StdStat(Feature):
    """Rolling Standard Deviation.
    
    STD = √VAR
    
    Reference: https://en.wikipedia.org/wiki/Standard_deviation
    """
    
    source_col: str = "close"
    period: int = 30
    ddof: int = 1
    
    requires = ["{source_col}"]
    outputs = ["{source_col}_std_{period}"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            pl.col(self.source_col)
              .rolling_std(window_size=self.period, ddof=self.ddof)
              .alias(f"{self.source_col}_std_{self.period}")
        )
    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 30, "ddof": 1},
        {"source_col": "close", "period": 60, "ddof": 1},
        {"source_col": "close", "period": 240, "ddof": 1},
    ]

@dataclass
@sf_component(name="stat/mad")
class MadStat(Feature):
    """Rolling Mean Absolute Deviation.
    
    MAD = Σ|x - mean| / n
    
    More robust to outliers than standard deviation.
    
    Reference: https://en.wikipedia.org/wiki/Average_absolute_deviation
    """
    
    source_col: str = "close"
    period: int = 30
    
    requires = ["{source_col}"]
    outputs = ["{source_col}_mad_{period}"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        values = df[self.source_col].to_numpy()
        n = len(values)
        
        mad = np.full(n, np.nan)
        for i in range(self.period - 1, n):
            window = values[i - self.period + 1:i + 1]
            mad[i] = np.mean(np.abs(window - np.mean(window)))
        
        return df.with_columns(
            pl.Series(name=f"{self.source_col}_mad_{self.period}", values=mad)
        )
    
    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 30},
        {"source_col": "close", "period": 60},
        {"source_col": "close", "period": 240},
    ]


@dataclass
@sf_component(name="stat/zscore")
class ZscoreStat(Feature):
    """Rolling Z-Score.
    
    Z = (x - mean) / stdev
    
    Measures how many std deviations from rolling mean.
    Output typically in range [-3, +3].
    
    Reference: https://en.wikipedia.org/wiki/Standard_score
    """
    
    source_col: str = "close"
    period: int = 30
    
    requires = ["{source_col}"]
    outputs = ["{source_col}_zscore_{period}"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        col = pl.col(self.source_col)
        mean = col.rolling_mean(window_size=self.period)
        std = col.rolling_std(window_size=self.period)
        
        return df.with_columns(
            ((col - mean) / std).alias(f"{self.source_col}_zscore_{self.period}")
        )
    
    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 30},
        {"source_col": "close", "period": 60},
        {"source_col": "close", "period": 240},
    ]

@dataclass
@sf_component(name="stat/cv")
class CvStat(Feature):
    """Rolling Coefficient of Variation.
    
    CV = stdev / |mean|
    
    Normalized dispersion. Useful for comparing volatility 
    across assets with different price levels.
    
    Reference: https://en.wikipedia.org/wiki/Coefficient_of_variation
    """
    
    source_col: str = "close"
    period: int = 30
    
    requires = ["{source_col}"]
    outputs = ["{source_col}_cv_{period}"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        col = pl.col(self.source_col)
        std = col.rolling_std(window_size=self.period)
        mean = col.rolling_mean(window_size=self.period).abs()
        
        return df.with_columns(
            (std / mean).alias(f"{self.source_col}_cv_{self.period}")
        )
    
    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 30},
        {"source_col": "close", "period": 60},
        {"source_col": "close", "period": 240},
    ]

@dataclass
@sf_component(name="stat/range")
class RangeStat(Feature):
    """Rolling Range.
    
    RANGE = max - min
    
    Simple measure of spread in window.
    
    Reference: https://en.wikipedia.org/wiki/Range_(statistics)
    """
    
    source_col: str = "close"
    period: int = 30
    
    requires = ["{source_col}"]
    outputs = ["{source_col}_range_{period}"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        col = pl.col(self.source_col)
        return df.with_columns(
            (col.rolling_max(window_size=self.period) - 
             col.rolling_min(window_size=self.period))
            .alias(f"{self.source_col}_range_{self.period}")
        )
    
    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 30},
        {"source_col": "close", "period": 60},
        {"source_col": "close", "period": 240},
    ]

@dataclass
@sf_component(name="stat/iqr")
class IqrStat(Feature):
    """Rolling Interquartile Range.
    
    IQR = Q3 - Q1 = P75 - P25
    
    Robust measure of spread, ignores outliers.
    
    Reference: https://en.wikipedia.org/wiki/Interquartile_range
    """
    
    source_col: str = "close"
    period: int = 30
    
    requires = ["{source_col}"]
    outputs = ["{source_col}_iqr_{period}"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        col = pl.col(self.source_col)
        q3 = col.rolling_quantile(quantile=0.75, window_size=self.period)
        q1 = col.rolling_quantile(quantile=0.25, window_size=self.period)
        
        return df.with_columns(
            (q3 - q1).alias(f"{self.source_col}_iqr_{self.period}")
        )
    
    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 30},
        {"source_col": "close", "period": 60},
        {"source_col": "close", "period": 240},
    ]

@dataclass
@sf_component(name="stat/aad")
class AadStat(Feature):
    """Rolling Average Absolute Deviation from Median.
    
    AAD = Σ|x - median| / n
    
    More robust than MAD (uses median instead of mean).
    
    Reference: https://en.wikipedia.org/wiki/Average_absolute_deviation
    """
    
    source_col: str = "close"
    period: int = 30
    
    requires = ["{source_col}"]
    outputs = ["{source_col}_aad_{period}"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        values = df[self.source_col].to_numpy()
        n = len(values)
        
        aad = np.full(n, np.nan)
        for i in range(self.period - 1, n):
            window = values[i - self.period + 1:i + 1]
            aad[i] = np.mean(np.abs(window - np.median(window)))
        
        return df.with_columns(
            pl.Series(name=f"{self.source_col}_aad_{self.period}", values=aad)
        )
    
    test_params: ClassVar[list[dict]] = [
        {"source_col": "close", "period": 30},
        {"source_col": "close", "period": 60},
        {"source_col": "close", "period": 240},
    ]


@dataclass
@sf_component(name="stat/robust_zscore")
class RobustZscoreStat(Feature):
    """Rolling Robust Z-Score (using median and MAD).
    
    RZ = (x - median) / (1.4826 * MAD)
    
    1.4826 is scale factor for normal distribution consistency.
    More robust to outliers than standard z-score.
    
    Reference: https://en.wikipedia.org/wiki/Median_absolute_deviation
    """
    
    source_col: str = "close"
    period: int = 30
    
    requires = ["{source_col}"]
    outputs = ["{source_col}_robz_{period}"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        values = df[self.source_col].to_numpy()
        n = len(values)
        
        robz = np.full(n, np.nan)
        scale = 1.4826  # consistency constant for normal
        
        for i in range(self.period - 1, n):
            window = values[i - self.period + 1:i + 1]
            median = np.median(window)
            mad = np.median(np.abs(window - median))
            
            if mad > 1e-10:
                robz[i] = (values[i] - median) / (scale * mad)
        
        return df.with_columns(
            pl.Series(name=f"{self.source_col}_robz_{self.period}", values=robz)
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


    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable, reproducible output."""
        return getattr(self, "period", getattr(self, "length", getattr(self, "window", 20))) * 5


    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable, reproducible output."""
        return getattr(self, "period", getattr(self, "length", getattr(self, "window", 20))) * 5


    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable, reproducible output."""
        return getattr(self, "period", getattr(self, "length", getattr(self, "window", 20))) * 5


    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable, reproducible output."""
        return getattr(self, "period", getattr(self, "length", getattr(self, "window", 20))) * 5


    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable, reproducible output."""
        return getattr(self, "period", getattr(self, "length", getattr(self, "window", 20))) * 5


    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable, reproducible output."""
        return getattr(self, "period", getattr(self, "length", getattr(self, "window", 20))) * 5


    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable, reproducible output."""
        return getattr(self, "period", getattr(self, "length", getattr(self, "window", 20))) * 5

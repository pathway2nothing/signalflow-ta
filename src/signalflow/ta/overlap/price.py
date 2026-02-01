"""Price transforms and typical price calculations."""
from dataclasses import dataclass

import numpy as np
import polars as pl

from signalflow import sf_component
from signalflow.feature.base import Feature


@dataclass
@sf_component(name="price/hl2")
class Hl2Price(Feature):
    """High-Low Midpoint.
    
    HL2 = (high + low) / 2
    
    Simple price midpoint. Useful as MA source.
    """
    
    requires = ["high", "low"]
    outputs = ["hl2"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            ((pl.col("high") + pl.col("low")) / 2).alias("hl2")
        )


@dataclass
@sf_component(name="price/hlc3")
class Hlc3Price(Feature):
    """Typical Price.
    
    HLC3 = (high + low + close) / 3
    
    Common price representation. Used in many indicators.
    """
    
    requires = ["high", "low", "close"]
    outputs = ["hlc3"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3).alias("hlc3")
        )


@dataclass
@sf_component(name="price/ohlc4")
class Ohlc4Price(Feature):
    """OHLC Average.
    
    OHLC4 = (open + high + low + close) / 4
    
    Full bar average price.
    """
    
    requires = ["open", "high", "low", "close"]
    outputs = ["ohlc4"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            ((pl.col("open") + pl.col("high") + pl.col("low") + pl.col("close")) / 4).alias("ohlc4")
        )


@dataclass
@sf_component(name="price/wcp")
class WcpPrice(Feature):
    """Weighted Close Price.
    
    WCP = (high + low + 2*close) / 4
    
    Gives extra weight to close.
    """
    
    requires = ["high", "low", "close"]
    outputs = ["wcp"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            ((pl.col("high") + pl.col("low") + 2 * pl.col("close")) / 4).alias("wcp")
        )


@dataclass
@sf_component(name="price/typical")
class TypicalPrice(Feature):
    """Typical Price with configurable weights.
    
    Generalized typical price calculation.
    
    TP = (w_h*high + w_l*low + w_c*close) / (w_h + w_l + w_c)
    """
    
    weight_high: float = 1.0
    weight_low: float = 1.0
    weight_close: float = 1.0
    
    requires = ["high", "low", "close"]
    outputs = ["typical"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        total_weight = self.weight_high + self.weight_low + self.weight_close
        
        return df.with_columns(
            ((self.weight_high * pl.col("high") + 
              self.weight_low * pl.col("low") + 
              self.weight_close * pl.col("close")) / total_weight).alias("typical")
        )


@dataclass
@sf_component(name="price/midpoint")
class MidpointPrice(Feature):
    """Rolling Midpoint.
    
    MIDPOINT = (highest + lowest) / 2
    
    Center of price channel over period.
    """
    
    period: int = 14
    source_col: str = "close"
    
    requires = ["{source_col}"]
    outputs = ["{source_col}_midpoint_{period}"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        col = pl.col(self.source_col)
        highest = col.rolling_max(window_size=self.period)
        lowest = col.rolling_min(window_size=self.period)
        
        return df.with_columns(
            ((highest + lowest) / 2).alias(f"{self.source_col}_midpoint_{self.period}")
        )


@dataclass
@sf_component(name="price/midprice")
class MidpricePrice(Feature):
    """Rolling Midprice (High-Low based).
    
    MIDPRICE = (highest_high + lowest_low) / 2
    
    Donchian channel midline.
    """
    
    period: int = 14
    
    requires = ["high", "low"]
    outputs = ["midprice_{period}"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        hh = pl.col("high").rolling_max(window_size=self.period)
        ll = pl.col("low").rolling_min(window_size=self.period)
        
        return df.with_columns(
            ((hh + ll) / 2).alias(f"midprice_{self.period}")
        )


@dataclass
@sf_component(name="price/log_price")
class LogPrice(Feature):
    """Log-transformed price.
    
    LOG_PRICE = log(price)
    
    Useful for percentage-based analysis.
    Makes price changes scale-invariant.
    """
    
    source_col: str = "close"
    
    requires = ["{source_col}"]
    outputs = ["{source_col}_log"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            pl.col(self.source_col).log().alias(f"{self.source_col}_log")
        )


@dataclass  
@sf_component(name="price/pct_from_high")
class PctFromHighPrice(Feature):
    """Percentage distance from rolling high.
    
    PCT = (price - highest) / highest * 100
    
    Always negative or zero. Useful for drawdown analysis.
    """
    
    source_col: str = "close"
    period: int = 20
    
    requires = ["{source_col}"]
    outputs = ["{source_col}_pct_from_high_{period}"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        col = pl.col(self.source_col)
        highest = col.rolling_max(window_size=self.period)
        
        return df.with_columns(
            ((col - highest) / highest * 100).alias(f"{self.source_col}_pct_from_high_{self.period}")
        )


@dataclass
@sf_component(name="price/pct_from_low")
class PctFromLowPrice(Feature):
    """Percentage distance from rolling low.
    
    PCT = (price - lowest) / lowest * 100
    
    Always positive or zero. Useful for rally analysis.
    """
    
    source_col: str = "close"
    period: int = 20
    
    requires = ["{source_col}"]
    outputs = ["{source_col}_pct_from_low_{period}"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        col = pl.col(self.source_col)
        lowest = col.rolling_min(window_size=self.period)
        
        return df.with_columns(
            ((col - lowest) / lowest * 100).alias(f"{self.source_col}_pct_from_low_{self.period}")
        )
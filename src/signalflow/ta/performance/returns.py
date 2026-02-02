"""Return calculations - log returns and related transforms."""
from dataclasses import dataclass
from typing import Optional

import numpy as np
import polars as pl

from signalflow.core import sf_component
from signalflow.feature import Feature
from typing import ClassVar


@dataclass
@sf_component(name="perf/log_ret")
class LogReturn(Feature):
    """Logarithmic returns.
    
    Log return = ln(price_t / price_{t-period}) = ln(price_t) - ln(price_{t-period})
    
    Properties:
        - Time additive: sum of log returns = total log return
        - Approximately equal to simple return for small changes
        - Symmetric: +10% and -10% have equal magnitude
        - Better statistical properties (closer to normal distribution)
    
    Parameters:
        source: Column name (must exist in df)
        period: Lookback period (default 1)
    
    Outputs:
        logret_{period}_{source}: Log returns
    
    Example:
        >>> LogReturn(source="close", period=1)
        # Output: logret_1_close
    """
    
    source: str = "close"
    period: int = 1
    
    def __post_init__(self):
        self.requires = [self.source]
        self.outputs = [f"logret_{self.period}_{self.source}"]
        
        if self.period < 1:
            raise ValueError("period must be >= 1")
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        result = (
            pl.col(self.source)
            .log()
            .diff(self.period)
            .alias(self.outputs[0])
        )
        return df.with_columns(result)
    
    test_params: ClassVar[list[dict]] = [
        {"source": "close", "period": 1},
        {"source": "close", "period": 60},
        {"source": "close", "period": 240},
    ]   


@dataclass
@sf_component(name="perf/pct_ret")
class PctReturn(Feature):
    """Simple (arithmetic) returns.
    
    Percentage return = (price_t - price_{t-period}) / price_{t-period}
                  = price_t / price_{t-period} - 1
    
    Parameters:
        source: Column name
        period: Lookback period
    
    Outputs:
        pct_ret_{period}_{source}: Simple returns
    """
    
    source: str = "close"
    period: int = 1
    
    def __post_init__(self):
        self.requires = [self.source]
        self.outputs = [f"pct_ret_{self.period}_{self.source}"]
        
        if self.period < 1:
            raise ValueError("period must be >= 1")
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        result = (
            pl.col(self.source)
            .pct_change(self.period)
            .alias(self.outputs[0])
        )
        return df.with_columns(result)
    
    test_params: ClassVar[list[dict]] = [
        {"source": "close", "period": 1},
        {"source": "close", "period": 60},
        {"source": "close", "period": 240},
    ]   
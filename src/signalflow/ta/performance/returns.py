"""Return calculations - log returns and related transforms."""

from dataclasses import dataclass
from typing import ClassVar

import polars as pl

from signalflow.ta._compat import feature
from signalflow.ta._compat import Feature


@dataclass
@feature("perf/log_ret")
class LogReturn(Feature):
    """Logarithmic returns.

    Log return = ln(price_t / price_{t-period}) = ln(price_t) - ln(price_{t-period})

    Properties:
        - Time additive: sum of log returns = total log return
        - Approximately equal to simple return for small changes
        - Symmetric: +10% and -10% have equal magnitude
        - Better statistical properties (closer to normal distribution)
    """

    source: str = "close"
    period: int = 1

    def __post_init__(self) -> None:
        self.requires = [self.source]
        self.outputs = [f"logret_{self.period}_{self.source}"]

        if self.period < 1:
            raise ValueError("period must be >= 1")

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        result = pl.col(self.source).log().diff(self.period).alias(self.outputs[0])
        return df.with_columns(result)

    test_params: ClassVar[list[dict]] = [
        {"source": "close", "period": 1},
        {"source": "close", "period": 60},
        {"source": "close", "period": 240},
    ]

    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable, reproducible output."""
        return getattr(self, "period", getattr(self, "length", getattr(self, "window", 20))) * 5


@dataclass
@feature("perf/pct_ret")
class PctReturn(Feature):
    """Simple (arithmetic) returns.

    Percentage return = (price_t - price_{t-period}) / price_{t-period}
                  = price_t / price_{t-period} - 1
    """

    source: str = "close"
    period: int = 1

    def __post_init__(self) -> None:
        self.requires = [self.source]
        self.outputs = [f"pct_ret_{self.period}_{self.source}"]

        if self.period < 1:
            raise ValueError("period must be >= 1")

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        result = pl.col(self.source).pct_change(self.period).alias(self.outputs[0])
        return df.with_columns(result)

    test_params: ClassVar[list[dict]] = [
        {"source": "close", "period": 1},
        {"source": "close", "period": 60},
        {"source": "close", "period": 240},
    ]

    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable, reproducible output."""
        return getattr(self, "period", getattr(self, "length", getattr(self, "window", 20))) * 5

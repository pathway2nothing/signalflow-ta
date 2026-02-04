"""Gap analysis indicators."""
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import polars as pl

from signalflow.core import sf_component
from signalflow.feature.base import Feature


@dataclass
@sf_component(name="volatility/gap")
class GapVol(Feature):
    """Gap Analysis.

    Analyzes overnight/inter-bar gaps.

    Outputs:
    - gap_val: Open - PrevClose
    - gap_pct: 100 * (Open - PrevClose) / PrevClose
    - gap_fill_pct: % of gap filled during the bar
    - gap_run_ratio: (Close - Open) / GapVal (continuation factor)
    - gap_range_ratio: |GapVal| / (High - Low) (gap vs range)
    - is_gap_up: 1 if gap > threshold, else 0
    - is_gap_down: 1 if gap < -threshold, else 0

    Reference:
    https://www.investopedia.com/terms/g/gap.asp
    """

    min_gap_pct: float = 0.0  # Minimum % change to be considered a gap
    
    requires = ["open", "high", "low", "close"]
    outputs = ["gap_val", "gap_pct", "gap_fill_pct", "gap_run_ratio", "gap_range_ratio", "is_gap_up", "is_gap_down"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        open_ = df["open"].to_numpy()
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        close = df["close"].to_numpy()
        n = len(close)

        prev_close = np.roll(close, 1)
        prev_close[0] = open_[0]  # No gap on first bar

        gap_val = open_ - prev_close
        gap_pct = 100 * gap_val / prev_close
        
        # Gap Fill Percentage
        # If Gap Up: (Open - Low) / GapVal
        # If Gap Down: (High - Open) / abs(GapVal)
        # 100% means fully filled (and possibly more).
        gap_fill_pct = np.zeros(n)
        
        is_up = gap_val > 0
        is_down = gap_val < 0
        
        # Avoid division by zero
        gap_val_safe = np.where(np.abs(gap_val) < 1e-10, 1e-10, gap_val)

        # Fill calculation
        fill_up = (open_ - low) / gap_val_safe
        fill_down = (high - open_) / np.abs(gap_val_safe)
        
        gap_fill_pct = np.where(is_up, fill_up, gap_fill_pct)
        gap_fill_pct = np.where(is_down, fill_down, gap_fill_pct)
        # Convert to percentage
        gap_fill_pct *= 100

        # Run Ratio: (Close - Open) / GapVal
        # positive = continuation, negative = fade
        gap_run_ratio = (close - open_) / gap_val_safe

        # Range Ratio: |GapVal| / (High - Low)
        # Indicates dominance of gap vs intraday volatility
        day_range = (high - low)
        gap_range_ratio = np.abs(gap_val) / np.where(day_range == 0, 1e-10, day_range)
        
        # Threshold logic
        is_gap_up_signal = np.where(gap_pct > self.min_gap_pct, 1.0, 0.0)
        is_gap_down_signal = np.where(gap_pct < -self.min_gap_pct, 1.0, 0.0)

        return df.with_columns([
            pl.Series(name="gap_val", values=gap_val),
            pl.Series(name="gap_pct", values=gap_pct),
            pl.Series(name="gap_fill_pct", values=gap_fill_pct),
            pl.Series(name="gap_run_ratio", values=gap_run_ratio),
            pl.Series(name="gap_range_ratio", values=gap_range_ratio),
            pl.Series(name="is_gap_up", values=is_gap_up_signal),
            pl.Series(name="is_gap_down", values=is_gap_down_signal),
        ])

    test_params: ClassVar[list[dict]] = [
        {"min_gap_pct": 0.0},
        {"min_gap_pct": 0.5},
    ]

    @property
    def warmup(self) -> int:
        return 1

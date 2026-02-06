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
    normalized: bool = False
    norm_period: int | None = None

    requires = ["open", "high", "low", "close"]
    outputs = [
        "gap_val",
        "gap_pct",
        "gap_fill_pct",
        "gap_run_ratio",
        "gap_range_ratio",
        "is_gap_up",
        "is_gap_down",
    ]

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
        day_range = high - low
        gap_range_ratio = np.abs(gap_val) / np.where(day_range == 0, 1e-10, day_range)

        # Threshold logic
        is_gap_up_signal = np.where(gap_pct > self.min_gap_pct, 1.0, 0.0)
        is_gap_down_signal = np.where(gap_pct < -self.min_gap_pct, 1.0, 0.0)

        # Normalization for unbounded outputs
        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window

            norm_window = self.norm_period or get_norm_window(20)
            gap_val = normalize_zscore(gap_val, window=norm_window)
            gap_pct = normalize_zscore(gap_pct, window=norm_window)
            gap_fill_pct = normalize_zscore(gap_fill_pct, window=norm_window)
            gap_run_ratio = normalize_zscore(gap_run_ratio, window=norm_window)
            gap_range_ratio = normalize_zscore(gap_range_ratio, window=norm_window)

        output_names = self._get_output_names()
        return df.with_columns(
            [
                pl.Series(name=output_names[0], values=gap_val),
                pl.Series(name=output_names[1], values=gap_pct),
                pl.Series(name=output_names[2], values=gap_fill_pct),
                pl.Series(name=output_names[3], values=gap_run_ratio),
                pl.Series(name=output_names[4], values=gap_range_ratio),
                pl.Series(name=output_names[5], values=is_gap_up_signal),
                pl.Series(name=output_names[6], values=is_gap_down_signal),
            ]
        )

    def _get_output_names(self) -> list[str]:
        """Generate output column names with normalization suffix."""
        suffix = "_norm" if self.normalized else ""
        return [
            f"gap_val{suffix}",
            f"gap_pct{suffix}",
            f"gap_fill_pct{suffix}",
            f"gap_run_ratio{suffix}",
            f"gap_range_ratio{suffix}",
            "is_gap_up",
            "is_gap_down",
        ]

    test_params: ClassVar[list[dict]] = [
        {"min_gap_pct": 0.0},
        {"min_gap_pct": 0.5},
        {"min_gap_pct": 0.0, "normalized": True},
    ]

    @property
    def warmup(self) -> int:
        base_warmup = 20
        if self.normalized:
            from signalflow.ta._normalization import get_norm_window

            norm_window = self.norm_period or get_norm_window(20)
            return base_warmup + norm_window
        return base_warmup

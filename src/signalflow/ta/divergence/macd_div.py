"""MACD Divergence Detector

Identifies regular and hidden divergences between price and MACD histogram.
"""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import polars as pl

from signalflow.ta._compat import feature
from signalflow.ta.divergence.base import DivergenceBase
from signalflow.ta.momentum import MacdMom


@feature("divergence/macd")
@dataclass
class MacdDivergence(DivergenceBase):
    """MACD Divergence Detector

    Identifies regular and hidden divergences between price and MACD histogram.

    MACD (Moving Average Convergence Divergence) divergences are particularly
    powerful because MACD is a trend-following momentum indicator that combines
    moving averages. Histogram divergences occur when the MACD histogram
    (MACD line - Signal line) diverges from price action.
    """

    fast: int = 12
    """Fast EMA period"""

    slow: int = 26
    """Slow EMA period"""

    signal: int = 9
    """Signal line EMA period"""

    requires: ClassVar[list[str]] = ["high", "low", "close"]
    outputs: ClassVar[list[str]] = [
        "macd_{fast}_{slow}",
        "macd_signal_{signal}",
        "macd_hist_{fast}_{slow}",
        "macd_div_bullish",
        "macd_div_bearish",
        "macd_div_hidden_bullish",
        "macd_div_hidden_bearish",
        "macd_div_strength",
    ]
    test_params: ClassVar[list[dict]] = [
        {"fast": 12, "slow": 26, "signal": 9, "pivot_window": 5},
        {"fast": 8, "slow": 17, "signal": 9, "pivot_window": 4},
        {"fast": 19, "slow": 39, "signal": 9, "pivot_window": 7},
    ]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute MACD and detect all types of divergences."""
        macd_indicator = MacdMom(fast=self.fast, slow=self.slow, signal=self.signal)
        df = macd_indicator.compute_pair(df)

        hist_col = f"macd_hist_{self.fast}_{self.slow}"

        close = df["close"].to_numpy()
        macd_hist = df[hist_col].to_numpy()
        len(close)

        price_highs_idx, price_lows_idx = self.find_pivots(close)
        hist_highs_idx, hist_lows_idx = self.find_pivots(macd_hist)

        bullish_div = self.detect_regular_bullish_divergence(close, price_lows_idx, macd_hist, hist_lows_idx)

        bearish_div = self.detect_regular_bearish_divergence(close, price_highs_idx, macd_hist, hist_highs_idx)

        hidden_bullish_div = self.detect_hidden_bullish_divergence(close, price_lows_idx, macd_hist, hist_lows_idx)

        hidden_bearish_div = self.detect_hidden_bearish_divergence(close, price_highs_idx, macd_hist, hist_highs_idx)

        all_divs = bullish_div | bearish_div | hidden_bullish_div | hidden_bearish_div

        strength = self.calculate_divergence_strength(
            close,
            macd_hist,
            all_divs,
            indicator_range=None,
            lookback_for_range=self.lookback,
        )

        strength = self._apply_crossover_boost(df, strength, bullish_div, bearish_div)

        df = df.with_columns(
            [
                pl.Series("macd_div_bullish", bullish_div),
                pl.Series("macd_div_bearish", bearish_div),
                pl.Series("macd_div_hidden_bullish", hidden_bullish_div),
                pl.Series("macd_div_hidden_bearish", hidden_bearish_div),
                pl.Series("macd_div_strength", strength),
            ]
        )

        return df

    def _apply_crossover_boost(
        self,
        df: pl.DataFrame,
        strength: np.ndarray,
        bullish_div: np.ndarray,
        bearish_div: np.ndarray,
    ) -> np.ndarray:
        """Boost divergence strength when MACD line crosses signal line.

        Crossovers provide additional confirmation:
        - Bullish crossover (MACD crosses above signal) confirms bullish divergence
        - Bearish crossover (MACD crosses below signal) confirms bearish divergence
        """
        boosted = strength.copy()

        macd_col = f"macd_{self.fast}_{self.slow}"
        signal_col = f"macd_signal_{self.signal}"

        macd_line = df[macd_col].to_numpy()
        signal_line = df[signal_col].to_numpy()

        bullish_cross = (macd_line[1:] > signal_line[1:]) & (macd_line[:-1] <= signal_line[:-1])
        bearish_cross = (macd_line[1:] < signal_line[1:]) & (macd_line[:-1] >= signal_line[:-1])

        bullish_cross = np.concatenate([[False], bullish_cross])
        bearish_cross = np.concatenate([[False], bearish_cross])

        crossover_window = 5

        for idx in np.where(bullish_div)[0]:
            window_start = max(0, idx - crossover_window)
            window_end = idx + 1

            if np.any(bullish_cross[window_start:window_end]):
                boosted[idx] += 10

        for idx in np.where(bearish_div)[0]:
            window_start = max(0, idx - crossover_window)
            window_end = idx + 1

            if np.any(bearish_cross[window_start:window_end]):
                boosted[idx] += 10

        return np.clip(boosted, 0, 100)

    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable, reproducible output."""
        return self.slow * 5 + self.pivot_window * 2 + self.lookback

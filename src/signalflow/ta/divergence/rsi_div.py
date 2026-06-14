"""RSI Divergence Detector

Identifies regular and hidden divergences between price and RSI momentum indicator.
"""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import polars as pl

from signalflow.ta._compat import feature
from signalflow.ta.divergence.base import DivergenceBase
from signalflow.ta.momentum import RsiMom


@feature("divergence/rsi")
@dataclass
class RsiDivergence(DivergenceBase):
    """RSI Divergence Detector

    Identifies regular and hidden divergences between price and RSI (Relative Strength Index).
    """

    rsi_period: int = 14
    """Period for RSI calculation"""

    rsi_overbought: float = 70.0
    """RSI level considered overbought"""

    rsi_oversold: float = 30.0
    """RSI level considered oversold"""

    requires: ClassVar[list[str]] = ["high", "low", "close"]
    outputs: ClassVar[list[str]] = [
        "rsi_{rsi_period}",
        "rsi_div_bullish",
        "rsi_div_bearish",
        "rsi_div_hidden_bullish",
        "rsi_div_hidden_bearish",
        "rsi_div_strength",
    ]
    test_params: ClassVar[list[dict]] = [
        {"rsi_period": 14, "pivot_window": 5, "min_pivot_distance": 10},
        {"rsi_period": 21, "pivot_window": 7, "min_pivot_distance": 15},
        {"rsi_period": 9, "pivot_window": 3, "min_pivot_distance": 8},
    ]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute RSI and detect all types of divergences."""
        rsi_indicator = RsiMom(period=self.rsi_period)
        df = rsi_indicator.compute_pair(df)
        rsi_col = f"rsi_{self.rsi_period}"

        close = df["close"].to_numpy()
        rsi = df[rsi_col].to_numpy()
        len(close)

        price_highs_idx, price_lows_idx = self.find_pivots(close)
        rsi_highs_idx, rsi_lows_idx = self.find_pivots(rsi)

        bullish_div = self.detect_regular_bullish_divergence(close, price_lows_idx, rsi, rsi_lows_idx)

        bearish_div = self.detect_regular_bearish_divergence(close, price_highs_idx, rsi, rsi_highs_idx)

        hidden_bullish_div = self.detect_hidden_bullish_divergence(close, price_lows_idx, rsi, rsi_lows_idx)

        hidden_bearish_div = self.detect_hidden_bearish_divergence(close, price_highs_idx, rsi, rsi_highs_idx)

        all_divs = bullish_div | bearish_div | hidden_bullish_div | hidden_bearish_div
        strength = self.calculate_divergence_strength(
            close,
            rsi,
            all_divs,
            indicator_range=(0, 100),
        )

        strength = self._apply_rsi_extremity_boost(rsi, strength, bullish_div, bearish_div)

        df = df.with_columns(
            [
                pl.Series("rsi_div_bullish", bullish_div),
                pl.Series("rsi_div_bearish", bearish_div),
                pl.Series("rsi_div_hidden_bullish", hidden_bullish_div),
                pl.Series("rsi_div_hidden_bearish", hidden_bearish_div),
                pl.Series("rsi_div_strength", strength),
            ]
        )

        return df

    def _apply_rsi_extremity_boost(
        self,
        rsi: np.ndarray,
        strength: np.ndarray,
        bullish_div: np.ndarray,
        bearish_div: np.ndarray,
    ) -> np.ndarray:
        """Boost divergence strength when RSI is in extreme zones.

        Bullish divergences are stronger when RSI < oversold (30)
        Bearish divergences are stronger when RSI > overbought (70)
        """
        boosted = strength.copy()

        oversold_mask = bullish_div & (rsi < self.rsi_oversold)
        if np.any(oversold_mask):
            oversold_depth = (self.rsi_oversold - rsi[oversold_mask]) / self.rsi_oversold
            boost = oversold_depth * 15
            boosted[oversold_mask] += boost

        overbought_mask = bearish_div & (rsi > self.rsi_overbought)
        if np.any(overbought_mask):
            overbought_depth = (rsi[overbought_mask] - self.rsi_overbought) / (100 - self.rsi_overbought)
            boost = overbought_depth * 15
            boosted[overbought_mask] += boost

        return np.clip(boosted, 0, 100)

    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable, reproducible output."""
        return self.rsi_period * 10 + self.pivot_window * 2 + self.lookback

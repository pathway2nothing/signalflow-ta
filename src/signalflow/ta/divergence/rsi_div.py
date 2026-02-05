"""
RSI Divergence Detector

Identifies regular and hidden divergences between price and RSI momentum indicator.
"""

import numpy as np
import polars as pl
from dataclasses import dataclass
from typing import ClassVar
from signalflow.core import sf_component
from signalflow.ta.divergence.base import DivergenceBase
from signalflow.ta.momentum import RsiMom


@sf_component(name="divergence/rsi")
@dataclass
class RsiDivergence(DivergenceBase):
    """
    RSI Divergence Detector

    Identifies regular and hidden divergences between price and RSI (Relative Strength Index).

    Divergence Types:
    -----------------
    1. Regular Bullish: Price makes lower low (LL), RSI makes higher low (HL)
       - Signal: Potential bullish reversal from oversold condition
       - Strength increased when RSI is below 30

    2. Regular Bearish: Price makes higher high (HH), RSI makes lower high (LH)
       - Signal: Potential bearish reversal from overbought condition
       - Strength increased when RSI is above 70

    3. Hidden Bullish: Price makes higher low (HL), RSI makes lower low (LL)
       - Signal: Bullish trend continuation
       - Confirms uptrend strength

    4. Hidden Bearish: Price makes lower high (LH), RSI makes higher high (HH)
       - Signal: Bearish trend continuation
       - Confirms downtrend strength

    Parameters
    ----------
    rsi_period : int, default 14
        Period for RSI calculation
    rsi_overbought : float, default 70
        RSI level considered overbought (increases bearish divergence strength)
    rsi_oversold : float, default 30
        RSI level considered oversold (increases bullish divergence strength)
    pivot_window : int, default 5
        Window size for pivot detection
    min_pivot_distance : int, default 10
        Minimum bars between consecutive pivots
    lookback : int, default 100
        How far back to look for divergences
    min_divergence_magnitude : float, default 0.02
        Minimum price divergence magnitude (2%)

    Returns
    -------
    DataFrame with columns:
    - rsi_{rsi_period} : RSI values
    - rsi_div_bullish : Regular bullish divergence detected
    - rsi_div_bearish : Regular bearish divergence detected
    - rsi_div_hidden_bullish : Hidden bullish divergence detected
    - rsi_div_hidden_bearish : Hidden bearish divergence detected
    - rsi_div_strength : Divergence strength score (0-100)

    References
    ----------
    - Wilder, J. Wells (1978). "New Concepts in Technical Trading Systems"
    - Cardwell, Andrew. "RSI Divergences and Reversals"
    - https://www.investopedia.com/terms/d/divergence.asp
    - https://school.stockcharts.com/doku.php?id=chart_analysis:rsi_divergences

    Examples
    --------
    >>> from signalflow.ta.divergence import RsiDivergence
    >>> import polars as pl
    >>>
    >>> # Create detector with default parameters
    >>> rsi_div = RsiDivergence(rsi_period=14, pivot_window=5)
    >>>
    >>> # Compute divergences
    >>> df = rsi_div.compute_pair(df)
    >>>
    >>> # Filter for high-confidence bullish signals
    >>> signals = df.filter(
    ...     (pl.col("rsi_div_bullish") == True) &
    ...     (pl.col("rsi_div_strength") > 60)
    ... )
    """

    # RSI parameters
    rsi_period: int = 14
    """Period for RSI calculation"""

    rsi_overbought: float = 70.0
    """RSI level considered overbought"""

    rsi_oversold: float = 30.0
    """RSI level considered oversold"""

    requires = ["high", "low", "close"]

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
        """
        Compute RSI and detect all types of divergences.

        Parameters
        ----------
        df : pl.DataFrame
            Input dataframe with OHLC data

        Returns
        -------
        df : pl.DataFrame
            Dataframe with RSI and divergence columns added
        """
        # 1. Calculate RSI using existing momentum indicator
        rsi_indicator = RsiMom(period=self.rsi_period)
        df = rsi_indicator.compute_pair(df)
        rsi_col = f"rsi_{self.rsi_period}"

        # 2. Extract numpy arrays for processing
        close = df["close"].to_numpy()
        rsi = df[rsi_col].to_numpy()
        n = len(close)

        # 3. Find pivots in price and RSI
        price_highs_idx, price_lows_idx = self.find_pivots(close)
        rsi_highs_idx, rsi_lows_idx = self.find_pivots(rsi)

        # 4. Detect regular bullish divergence (Price LL, RSI HL)
        bullish_div = self.detect_regular_bullish_divergence(
            close, price_lows_idx, rsi, rsi_lows_idx
        )

        # 5. Detect regular bearish divergence (Price HH, RSI LH)
        bearish_div = self.detect_regular_bearish_divergence(
            close, price_highs_idx, rsi, rsi_highs_idx
        )

        # 6. Detect hidden bullish divergence (Price HL, RSI LL)
        hidden_bullish_div = self.detect_hidden_bullish_divergence(
            close, price_lows_idx, rsi, rsi_lows_idx
        )

        # 7. Detect hidden bearish divergence (Price LH, RSI HH)
        hidden_bearish_div = self.detect_hidden_bearish_divergence(
            close, price_highs_idx, rsi, rsi_highs_idx
        )

        # 8. Calculate divergence strength
        all_divs = bullish_div | bearish_div | hidden_bullish_div | hidden_bearish_div
        strength = self.calculate_divergence_strength(
            close,
            rsi,
            all_divs,
            indicator_range=(0, 100),  # RSI ranges from 0-100
        )

        # 9. Boost strength for extreme RSI levels
        strength = self._apply_rsi_extremity_boost(
            rsi, strength, bullish_div, bearish_div
        )

        # 10. Add divergence columns to dataframe
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
        """
        Boost divergence strength when RSI is in extreme zones.

        Bullish divergences are stronger when RSI < oversold (30)
        Bearish divergences are stronger when RSI > overbought (70)

        Parameters
        ----------
        rsi : np.ndarray
            RSI values
        strength : np.ndarray
            Base strength scores
        bullish_div : np.ndarray
            Bullish divergence flags
        bearish_div : np.ndarray
            Bearish divergence flags

        Returns
        -------
        boosted_strength : np.ndarray
            Strength scores with extremity boost applied
        """
        boosted = strength.copy()

        # Boost bullish divergences in oversold zone
        oversold_mask = bullish_div & (rsi < self.rsi_oversold)
        if np.any(oversold_mask):
            # More oversold = bigger boost (up to +15 points)
            oversold_depth = (
                self.rsi_oversold - rsi[oversold_mask]
            ) / self.rsi_oversold
            boost = oversold_depth * 15
            boosted[oversold_mask] += boost

        # Boost bearish divergences in overbought zone
        overbought_mask = bearish_div & (rsi > self.rsi_overbought)
        if np.any(overbought_mask):
            # More overbought = bigger boost (up to +15 points)
            overbought_depth = (rsi[overbought_mask] - self.rsi_overbought) / (
                100 - self.rsi_overbought
            )
            boost = overbought_depth * 15
            boosted[overbought_mask] += boost

        return np.clip(boosted, 0, 100)

    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable, reproducible output."""
        return self.rsi_period * 10 + self.pivot_window * 2 + self.lookback

"""
MACD Divergence Detector

Identifies regular and hidden divergences between price and MACD histogram.
"""

import numpy as np
import polars as pl
from dataclasses import dataclass
from typing import ClassVar
from signalflow.core import sf_component
from signalflow.ta.divergence.base import DivergenceBase
from signalflow.ta.momentum import MacdMom


@sf_component(name="divergence/macd")
@dataclass
class MacdDivergence(DivergenceBase):
    """
    MACD Divergence Detector

    Identifies regular and hidden divergences between price and MACD histogram.

    MACD (Moving Average Convergence Divergence) divergences are particularly
    powerful because MACD is a trend-following momentum indicator that combines
    moving averages. Histogram divergences occur when the MACD histogram
    (MACD line - Signal line) diverges from price action.

    Divergence Types:
    -----------------
    1. Regular Bullish: Price LL, MACD Histogram HL
       - Signal: Potential trend reversal to upside
       - Often occurs at market bottoms

    2. Regular Bearish: Price HH, MACD Histogram LH
       - Signal: Potential trend reversal to downside
       - Often occurs at market tops

    3. Hidden Bullish: Price HL, MACD Histogram LL
       - Signal: Uptrend continuation
       - Indicates strong underlying momentum

    4. Hidden Bearish: Price LH, MACD Histogram HH
       - Signal: Downtrend continuation
       - Indicates strong downside momentum

    Key Advantages of MACD Divergences:
    -----------------------------------
    - Earlier signals than RSI (MACD is more responsive)
    - Histogram makes divergences visually clear
    - Works well in trending markets
    - Less prone to false signals in ranging markets

    Parameters
    ----------
    fast_period : int, default 12
        Fast EMA period for MACD
    slow_period : int, default 26
        Slow EMA period for MACD
    signal_period : int, default 9
        Signal line EMA period
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
    - macd_{fast_period}_{slow_period}_{signal_period} : MACD line
    - macd_signal_{fast_period}_{slow_period}_{signal_period} : Signal line
    - macd_hist_{fast_period}_{slow_period}_{signal_period} : Histogram
    - macd_div_bullish : Regular bullish divergence
    - macd_div_bearish : Regular bearish divergence
    - macd_div_hidden_bullish : Hidden bullish divergence
    - macd_div_hidden_bearish : Hidden bearish divergence
    - macd_div_strength : Divergence strength score (0-100)

    References
    ----------
    - Appel, Gerald (2005). "Technical Analysis: Power Tools for Active Investors"
    - https://www.investopedia.com/terms/m/macd.asp
    - https://school.stockcharts.com/doku.php?id=technical_indicators:moving_average_convergence_divergence_macd

    Examples
    --------
    >>> from signalflow.ta.divergence import MacdDivergence
    >>> import polars as pl
    >>>
    >>> # Create detector with standard MACD parameters
    >>> macd_div = MacdDivergence(fast_period=12, slow_period=26, signal_period=9)
    >>>
    >>> # Compute divergences
    >>> df = macd_div.compute_pair(df)
    >>>
    >>> # Filter for strong bearish signals at potential tops
    >>> signals = df.filter(
    ...     (pl.col("macd_div_bearish") == True) &
    ...     (pl.col("macd_div_strength") > 65)
    ... )
    """

    # MACD parameters
    fast: int = 12
    """Fast EMA period"""

    slow: int = 26
    """Slow EMA period"""

    signal: int = 9
    """Signal line EMA period"""

    requires = ["high", "low", "close"]

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
        """
        Compute MACD and detect all types of divergences.

        Parameters
        ----------
        df : pl.DataFrame
            Input dataframe with OHLC data

        Returns
        -------
        df : pl.DataFrame
            Dataframe with MACD and divergence columns added
        """
        # 1. Calculate MACD using existing momentum indicator
        macd_indicator = MacdMom(
            fast=self.fast,
            slow=self.slow,
            signal=self.signal
        )
        df = macd_indicator.compute_pair(df)

        # 2. Get column names
        macd_col = f"macd_{self.fast}_{self.slow}"
        signal_col = f"macd_signal_{self.signal}"
        hist_col = f"macd_hist_{self.fast}_{self.slow}"

        # 3. Extract numpy arrays
        close = df["close"].to_numpy()
        macd_hist = df[hist_col].to_numpy()
        n = len(close)

        # 4. Find pivots in price and MACD histogram
        price_highs_idx, price_lows_idx = self.find_pivots(close)
        hist_highs_idx, hist_lows_idx = self.find_pivots(macd_hist)

        # 5. Detect regular bullish divergence (Price LL, Histogram HL)
        bullish_div = self.detect_regular_bullish_divergence(
            close, price_lows_idx, macd_hist, hist_lows_idx
        )

        # 6. Detect regular bearish divergence (Price HH, Histogram LH)
        bearish_div = self.detect_regular_bearish_divergence(
            close, price_highs_idx, macd_hist, hist_highs_idx
        )

        # 7. Detect hidden bullish divergence (Price HL, Histogram LL)
        hidden_bullish_div = self.detect_hidden_bullish_divergence(
            close, price_lows_idx, macd_hist, hist_lows_idx
        )

        # 8. Detect hidden bearish divergence (Price LH, Histogram HH)
        hidden_bearish_div = self.detect_hidden_bearish_divergence(
            close, price_highs_idx, macd_hist, hist_highs_idx
        )

        # 9. Calculate divergence strength
        all_divs = bullish_div | bearish_div | hidden_bullish_div | hidden_bearish_div

        # Use dynamic range calculation (causal - no look-ahead)
        strength = self.calculate_divergence_strength(
            close, macd_hist, all_divs,
            indicator_range=None,  # Calculate dynamically per bar
            lookback_for_range=self.lookback
        )

        # 10. Boost strength for crossovers (additional confirmation)
        strength = self._apply_crossover_boost(df, strength, bullish_div, bearish_div)

        # 11. Add divergence columns to dataframe
        df = df.with_columns([
            pl.Series("macd_div_bullish", bullish_div),
            pl.Series("macd_div_bearish", bearish_div),
            pl.Series("macd_div_hidden_bullish", hidden_bullish_div),
            pl.Series("macd_div_hidden_bearish", hidden_bearish_div),
            pl.Series("macd_div_strength", strength),
        ])

        return df

    def _apply_crossover_boost(
        self,
        df: pl.DataFrame,
        strength: np.ndarray,
        bullish_div: np.ndarray,
        bearish_div: np.ndarray
    ) -> np.ndarray:
        """
        Boost divergence strength when MACD line crosses signal line.

        Crossovers provide additional confirmation:
        - Bullish crossover (MACD crosses above signal) confirms bullish divergence
        - Bearish crossover (MACD crosses below signal) confirms bearish divergence

        Parameters
        ----------
        df : pl.DataFrame
            Dataframe with MACD columns
        strength : np.ndarray
            Base strength scores
        bullish_div : np.ndarray
            Bullish divergence flags
        bearish_div : np.ndarray
            Bearish divergence flags

        Returns
        -------
        boosted_strength : np.ndarray
            Strength scores with crossover boost applied
        """
        boosted = strength.copy()

        # Get MACD and signal line
        macd_col = f"macd_{self.fast}_{self.slow}"
        signal_col = f"macd_signal_{self.signal}"

        macd_line = df[macd_col].to_numpy()
        signal_line = df[signal_col].to_numpy()

        # Detect crossovers (look at previous bar for cross)
        bullish_cross = (macd_line[1:] > signal_line[1:]) & (macd_line[:-1] <= signal_line[:-1])
        bearish_cross = (macd_line[1:] < signal_line[1:]) & (macd_line[:-1] >= signal_line[:-1])

        # Pad with False at start to match length
        bullish_cross = np.concatenate([[False], bullish_cross])
        bearish_cross = np.concatenate([[False], bearish_cross])

        # Apply boost for confirmed divergences
        # Look for crossover within lookback window (CAUSAL - only backward)
        crossover_window = 5

        for idx in np.where(bullish_div)[0]:
            window_start = max(0, idx - crossover_window)
            # Only look backward - not forward (causal)
            window_end = idx + 1

            if np.any(bullish_cross[window_start:window_end]):
                boosted[idx] += 10  # +10 points for crossover confirmation

        for idx in np.where(bearish_div)[0]:
            window_start = max(0, idx - crossover_window)
            # Only look backward - not forward (causal)
            window_end = idx + 1

            if np.any(bearish_cross[window_start:window_end]):
                boosted[idx] += 10  # +10 points for crossover confirmation

        return np.clip(boosted, 0, 100)


    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable, reproducible output."""
        return self.slow * 5 + self.pivot_window * 2 + self.lookback

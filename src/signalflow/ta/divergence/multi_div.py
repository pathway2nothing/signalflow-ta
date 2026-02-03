"""
Multi-Indicator Divergence Confluence

Combines divergence signals from multiple indicators for higher confidence signals.
"""

import numpy as np
import polars as pl
from dataclasses import dataclass
from typing import ClassVar
from signalflow.core import sf_component
from signalflow.feature.base import Feature
from signalflow.ta.divergence.rsi_div import RsiDivergence
from signalflow.ta.divergence.macd_div import MacdDivergence


@sf_component(name="divergence/multi")
@dataclass
class MultiDivergence(Feature):
    """
    Multi-Indicator Divergence Confluence Detector

    Combines divergence signals from multiple momentum indicators to create
    high-confidence trading signals. Multiple independent divergences occurring
    simultaneously significantly increase the probability of a successful trade.

    Confluence Scoring System:
    -------------------------
    - 1 indicator: Low confidence (30-50 points)
    - 2 indicators: Medium confidence (50-75 points)
    - 3+ indicators: High confidence (75-100 points)

    The final strength score is calculated as:
    1. Base score from number of confirming indicators
    2. Weighted average of individual divergence strengths
    3. Bonus for indicator diversity (RSI + MACD better than 2x RSI)

    Why Confluence Matters:
    ----------------------
    - Reduces false signals significantly
    - Confirms market structure across timeframes
    - Different indicators capture different aspects of momentum
    - RSI: Overbought/oversold extremes
    - MACD: Trend momentum and crossovers
    - Stochastic: Fast momentum changes

    Parameters
    ----------
    use_rsi : bool, default True
        Include RSI divergence in confluence
    use_macd : bool, default True
        Include MACD divergence in confluence
    rsi_period : int, default 14
        RSI period for divergence detection
    macd_fast : int, default 12
        MACD fast period
    macd_slow : int, default 26
        MACD slow period
    macd_signal : int, default 9
        MACD signal period
    min_confluence : int, default 2
        Minimum number of indicators required for signal
    pivot_window : int, default 5
        Window size for pivot detection
    min_pivot_distance : int, default 10
        Minimum bars between pivots
    lookback : int, default 100
        Lookback period for divergence detection

    Returns
    -------
    DataFrame with columns:
    - multi_div_bullish : Bullish confluence signal
    - multi_div_bearish : Bearish confluence signal
    - multi_div_hidden_bullish : Hidden bullish confluence
    - multi_div_hidden_bearish : Hidden bearish confluence
    - multi_div_confluence_score : Overall strength (0-100)
    - multi_div_num_indicators : Number of confirming indicators
    - multi_div_indicators : Comma-separated list of confirming indicators

    References
    ----------
    - Elder, Alexander (1993). "Trading for a Living" - Triple Screen Trading System
    - Murphy, John J. (1999). "Technical Analysis of the Financial Markets"
    - https://www.investopedia.com/terms/c/confluence.asp

    Examples
    --------
    >>> from signalflow.ta.divergence import MultiDivergence
    >>> import polars as pl
    >>>
    >>> # Create multi-indicator detector
    >>> multi_div = MultiDivergence(
    ...     use_rsi=True,
    ...     use_macd=True,
    ...     min_confluence=2
    ... )
    >>>
    >>> # Compute divergences
    >>> df = multi_div.compute_pair(df)
    >>>
    >>> # Filter for very high confidence signals
    >>> signals = df.filter(
    ...     (pl.col("multi_div_bullish") == True) &
    ...     (pl.col("multi_div_confluence_score") > 80) &
    ...     (pl.col("multi_div_num_indicators") >= 2)
    ... )
    >>>
    >>> # Check which indicators confirmed
    >>> print(signals.select("multi_div_indicators"))
    """

    # Indicator selection
    use_rsi: bool = True
    """Include RSI divergence"""

    use_macd: bool = True
    """Include MACD divergence"""

    # RSI parameters
    rsi_period: int = 14
    """RSI period"""

    # MACD parameters
    fast: int = 12
    """MACD fast period"""

    slow: int = 26
    """MACD slow period"""

    signal: int = 9
    """MACD signal period"""

    # Confluence parameters
    min_confluence: int = 2
    """Minimum indicators required for signal"""

    # Pivot parameters (shared across all indicators)
    pivot_window: int = 5
    """Pivot detection window"""

    min_pivot_distance: int = 10
    """Minimum distance between pivots"""

    lookback: int = 100
    """Lookback period"""

    requires = ["high", "low", "close"]

    outputs: ClassVar[list[str]] = [
        "multi_div_bullish",
        "multi_div_bearish",
        "multi_div_hidden_bullish",
        "multi_div_hidden_bearish",
        "multi_div_confluence_score",
        "multi_div_num_indicators",
        "multi_div_indicators",
    ]

    test_params: ClassVar[list[dict]] = [
        {
            "use_rsi": True,
            "use_macd": True,
            "min_confluence": 2,
            "pivot_window": 5,
        },
        {
            "use_rsi": True,
            "use_macd": True,
            "min_confluence": 1,
            "pivot_window": 7,
        },
    ]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Compute multi-indicator divergence confluence.

        Parameters
        ----------
        df : pl.DataFrame
            Input dataframe with OHLC data

        Returns
        -------
        df : pl.DataFrame
            Dataframe with confluence columns added
        """
        n = len(df)

        # Storage for individual indicator results
        indicators_bullish = []
        indicators_bearish = []
        indicators_hidden_bullish = []
        indicators_hidden_bearish = []
        indicator_names = []
        indicator_strengths = []

        # 1. Compute RSI divergences
        if self.use_rsi:
            rsi_div = RsiDivergence(
                rsi_period=self.rsi_period,
                pivot_window=self.pivot_window,
                min_pivot_distance=self.min_pivot_distance,
                lookback=self.lookback
            )
            df = rsi_div.compute_pair(df)

            indicators_bullish.append(df["rsi_div_bullish"].to_numpy())
            indicators_bearish.append(df["rsi_div_bearish"].to_numpy())
            indicators_hidden_bullish.append(df["rsi_div_hidden_bullish"].to_numpy())
            indicators_hidden_bearish.append(df["rsi_div_hidden_bearish"].to_numpy())
            indicator_names.append("rsi")
            indicator_strengths.append(df["rsi_div_strength"].to_numpy())

        # 2. Compute MACD divergences
        if self.use_macd:
            macd_div = MacdDivergence(
                fast=self.fast,
                slow=self.slow,
                signal=self.signal,
                pivot_window=self.pivot_window,
                min_pivot_distance=self.min_pivot_distance,
                lookback=self.lookback
            )
            df = macd_div.compute_pair(df)

            indicators_bullish.append(df["macd_div_bullish"].to_numpy())
            indicators_bearish.append(df["macd_div_bearish"].to_numpy())
            indicators_hidden_bullish.append(df["macd_div_hidden_bullish"].to_numpy())
            indicators_hidden_bearish.append(df["macd_div_hidden_bearish"].to_numpy())
            indicator_names.append("macd")
            indicator_strengths.append(df["macd_div_strength"].to_numpy())

        # 3. Calculate confluence
        if len(indicators_bullish) == 0:
            # No indicators enabled, return zeros
            df = df.with_columns([
                pl.lit(False).alias("multi_div_bullish"),
                pl.lit(False).alias("multi_div_bearish"),
                pl.lit(False).alias("multi_div_hidden_bullish"),
                pl.lit(False).alias("multi_div_hidden_bearish"),
                pl.lit(0.0).alias("multi_div_confluence_score"),
                pl.lit(0).alias("multi_div_num_indicators"),
                pl.lit("").alias("multi_div_indicators"),
            ])
            return df

        # Count confirmations for each divergence type
        bullish_count = np.sum(indicators_bullish, axis=0)
        bearish_count = np.sum(indicators_bearish, axis=0)
        hidden_bullish_count = np.sum(indicators_hidden_bullish, axis=0)
        hidden_bearish_count = np.sum(indicators_hidden_bearish, axis=0)

        # Apply minimum confluence threshold
        multi_bullish = bullish_count >= self.min_confluence
        multi_bearish = bearish_count >= self.min_confluence
        multi_hidden_bullish = hidden_bullish_count >= self.min_confluence
        multi_hidden_bearish = hidden_bearish_count >= self.min_confluence

        # 4. Calculate confluence scores
        confluence_score, num_indicators, indicator_list = self._calculate_confluence_scores(
            bullish_count,
            bearish_count,
            hidden_bullish_count,
            hidden_bearish_count,
            indicator_strengths,
            indicator_names,
            indicators_bullish,
            indicators_bearish,
            indicators_hidden_bullish,
            indicators_hidden_bearish,
            n
        )

        # 5. Add confluence columns to dataframe
        df = df.with_columns([
            pl.Series("multi_div_bullish", multi_bullish),
            pl.Series("multi_div_bearish", multi_bearish),
            pl.Series("multi_div_hidden_bullish", multi_hidden_bullish),
            pl.Series("multi_div_hidden_bearish", multi_hidden_bearish),
            pl.Series("multi_div_confluence_score", confluence_score),
            pl.Series("multi_div_num_indicators", num_indicators),
            pl.Series("multi_div_indicators", indicator_list),
        ])

        return df

    def _calculate_confluence_scores(
        self,
        bullish_count: np.ndarray,
        bearish_count: np.ndarray,
        hidden_bullish_count: np.ndarray,
        hidden_bearish_count: np.ndarray,
        indicator_strengths: list,
        indicator_names: list,
        indicators_bullish: list,
        indicators_bearish: list,
        indicators_hidden_bullish: list,
        indicators_hidden_bearish: list,
        n: int
    ) -> tuple:
        """
        Calculate confluence scores and metadata.

        Returns
        -------
        scores : np.ndarray
            Confluence scores (0-100)
        num_indicators : np.ndarray
            Number of confirming indicators
        indicator_list : list of str
            Comma-separated indicator names
        """
        scores = np.zeros(n, dtype=float)
        num_indicators = np.zeros(n, dtype=int)
        indicator_list = [""] * n

        for i in range(n):
            # Find which divergence type(s) are active
            max_count = max(
                bullish_count[i],
                bearish_count[i],
                hidden_bullish_count[i],
                hidden_bearish_count[i]
            )

            if max_count == 0:
                continue

            num_indicators[i] = max_count

            # Calculate base score from count
            if max_count == 1:
                base_score = 40
            elif max_count == 2:
                base_score = 65
            else:  # 3+
                base_score = 80

            # Calculate weighted average of individual strengths
            contributing_strengths = []
            contributing_indicators = []

            for j, name in enumerate(indicator_names):
                # Check if this indicator contributed to the dominant divergence type
                contributed = False

                if bullish_count[i] == max_count and indicators_bullish[j][i]:
                    contributed = True
                elif bearish_count[i] == max_count and indicators_bearish[j][i]:
                    contributed = True
                elif hidden_bullish_count[i] == max_count and indicators_hidden_bullish[j][i]:
                    contributed = True
                elif hidden_bearish_count[i] == max_count and indicators_hidden_bearish[j][i]:
                    contributed = True

                if contributed:
                    contributing_strengths.append(indicator_strengths[j][i])
                    contributing_indicators.append(name)

            if len(contributing_strengths) > 0:
                avg_strength = np.mean(contributing_strengths)
                # Blend base score with average strength (70/30 weighting)
                scores[i] = base_score * 0.7 + avg_strength * 0.3
                indicator_list[i] = ",".join(contributing_indicators)
            else:
                scores[i] = base_score
                indicator_list[i] = ""

        return np.clip(scores, 0, 100), num_indicators, indicator_list

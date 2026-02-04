"""
Base Divergence Detection Class

Foundation for all divergence detectors with common pivot detection
and divergence pattern recognition logic.
"""

import numpy as np
import polars as pl
from dataclasses import dataclass
from typing import ClassVar, Tuple
from signalflow.feature.base import Feature
from signalflow.ta.divergence.pivot import (
    find_pivots_window,
    find_pivots_scipy,
    calculate_slope
)


@dataclass
class DivergenceBase(Feature):
    """
    Base class for divergence detection.

    Provides common functionality for detecting price-indicator divergences:
    - Pivot point detection (local highs and lows)
    - Divergence pattern recognition (regular and hidden)
    - Divergence strength calculation

    Divergence Types:
    -----------------
    Regular Bullish: Price makes lower low (LL), indicator makes higher low (HL)
        Signal: Potential reversal up from oversold
    Regular Bearish: Price makes higher high (HH), indicator makes lower high (LH)
        Signal: Potential reversal down from overbought
    Hidden Bullish: Price makes higher low (HL), indicator makes lower low (LL)
        Signal: Trend continuation up
    Hidden Bearish: Price makes lower high (LH), indicator makes higher high (HH)
        Signal: Trend continuation down
    """

    # Pivot detection parameters
    pivot_window: int = 5
    """Window size for pivot detection (bars on each side)"""

    min_pivot_distance: int = 10
    """Minimum bars between consecutive pivots"""

    pivot_method: str = "window"
    """Pivot detection method: 'window' or 'scipy'"""

    # Divergence detection parameters
    lookback: int = 100
    """How many bars back to look for divergences"""

    min_divergence_magnitude: float = 0.02
    """Minimum divergence magnitude (as fraction, e.g., 0.02 = 2%)"""

    pivot_align_tolerance: int = 5
    """Maximum bar difference to consider price and indicator pivots aligned"""

    # Strength calculation parameters
    strength_window: int = 14
    """Window for calculating divergence strength context"""

    requires = ["high", "low", "close"]

    def find_pivots(
        self,
        series: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find local highs and lows in a series.

        Parameters
        ----------
        series : np.ndarray
            Input time series

        Returns
        -------
        highs_idx : np.ndarray
            Indices of local maxima
        lows_idx : np.ndarray
            Indices of local minima
        """
        if self.pivot_method == "scipy":
            return find_pivots_scipy(
                series,
                order=self.pivot_window,
                min_distance=self.min_pivot_distance
            )
        else:  # "window"
            return find_pivots_window(
                series,
                window=self.pivot_window,
                min_distance=self.min_pivot_distance
            )

    def detect_regular_bullish_divergence(
        self,
        price: np.ndarray,
        price_lows_idx: np.ndarray,
        indicator: np.ndarray,
        indicator_lows_idx: np.ndarray
    ) -> np.ndarray:
        """
        Detect regular bullish divergence: Price LL, Indicator HL.

        CAUSAL IMPLEMENTATION: Processes bar-by-bar to ensure no look-ahead bias.
        At each bar, checks if a divergence pattern can be confirmed based only
        on pivots that have been confirmed up to that bar.

        Parameters
        ----------
        price : np.ndarray
            Price series
        price_lows_idx : np.ndarray
            Indices of price lows
        indicator : np.ndarray
            Indicator series
        indicator_lows_idx : np.ndarray
            Indices of indicator lows

        Returns
        -------
        divergence : np.ndarray
            Integer array (0 or 1) indicating divergence at each bar
        """
        n = len(price)
        divergence = np.zeros(n, dtype=np.int8)

        if len(price_lows_idx) < 2 or len(indicator_lows_idx) < 2:
            return divergence

        # Process bar-by-bar to ensure causality
        # At bar i, we can only use pivots that are confirmed (with window-bar delay)
        # A pivot at index p is confirmed at bar (p + pivot_window)
        # So at bar i, we can only use pivots where p + pivot_window <= i
        # which means p <= i - pivot_window

        # Track which pivot pairs we've already marked to avoid duplicates
        marked_pairs = set()

        for i in range(n):
            # Get pivots confirmed up to current bar i
            max_pivot_idx = i - self.pivot_window
            confirmed_price_lows = price_lows_idx[price_lows_idx <= max_pivot_idx]
            confirmed_indicator_lows = indicator_lows_idx[indicator_lows_idx <= max_pivot_idx]

            if len(confirmed_price_lows) < 2:
                continue

            # Look for divergence in the lookback window
            lookback_start = max(0, i - self.lookback)
            recent_price_lows = confirmed_price_lows[confirmed_price_lows >= lookback_start]

            if len(recent_price_lows) < 2:
                continue

            # Check most recent pair of lows for divergence
            idx_current = recent_price_lows[-1]
            idx_previous = recent_price_lows[-2]

            # Skip if we've already marked this pair
            pair_key = (int(idx_previous), int(idx_current))
            if pair_key in marked_pairs:
                continue

            # Price makes lower low
            if price[idx_current] < price[idx_previous]:
                # Find corresponding indicator lows
                ind_current = self._find_closest_pivot(
                    idx_current, confirmed_indicator_lows, self.pivot_align_tolerance
                )
                ind_previous = self._find_closest_pivot(
                    idx_previous, confirmed_indicator_lows, self.pivot_align_tolerance
                )

                if ind_current is not None and ind_previous is not None:
                    # Indicator makes higher low
                    if indicator[ind_current] > indicator[ind_previous]:
                        # Check magnitude
                        price_change = abs(price[idx_current] - price[idx_previous]) / price[idx_previous]
                        if price_change >= self.min_divergence_magnitude:
                            # Mark divergence at current bar (where we detect it)
                            divergence[i] = 1
                            marked_pairs.add(pair_key)

        return divergence

    def detect_regular_bearish_divergence(
        self,
        price: np.ndarray,
        price_highs_idx: np.ndarray,
        indicator: np.ndarray,
        indicator_highs_idx: np.ndarray
    ) -> np.ndarray:
        """
        Detect regular bearish divergence: Price HH, Indicator LH.

        CAUSAL IMPLEMENTATION: Processes bar-by-bar to ensure no look-ahead bias.

        Parameters
        ----------
        price : np.ndarray
            Price series
        price_highs_idx : np.ndarray
            Indices of price highs
        indicator : np.ndarray
            Indicator series
        indicator_highs_idx : np.ndarray
            Indices of indicator highs

        Returns
        -------
        divergence : np.ndarray
            Integer array (0 or 1) indicating divergence at each bar
        """
        n = len(price)
        divergence = np.zeros(n, dtype=np.int8)

        if len(price_highs_idx) < 2 or len(indicator_highs_idx) < 2:
            return divergence

        # Process bar-by-bar to ensure causality
        # Track which pivot pairs we've already marked to avoid duplicates
        marked_pairs = set()

        for i in range(n):
            # Get pivots confirmed up to current bar i
            max_pivot_idx = i - self.pivot_window
            confirmed_price_highs = price_highs_idx[price_highs_idx <= max_pivot_idx]
            confirmed_indicator_highs = indicator_highs_idx[indicator_highs_idx <= max_pivot_idx]

            if len(confirmed_price_highs) < 2:
                continue

            # Look for divergence in the lookback window
            lookback_start = max(0, i - self.lookback)
            recent_price_highs = confirmed_price_highs[confirmed_price_highs >= lookback_start]

            if len(recent_price_highs) < 2:
                continue

            # Check most recent pair of highs for divergence
            idx_current = recent_price_highs[-1]
            idx_previous = recent_price_highs[-2]

            # Skip if we've already marked this pair
            pair_key = (int(idx_previous), int(idx_current))
            if pair_key in marked_pairs:
                continue

            # Price makes higher high
            if price[idx_current] > price[idx_previous]:
                # Find corresponding indicator highs
                ind_current = self._find_closest_pivot(
                    idx_current, confirmed_indicator_highs, self.pivot_align_tolerance
                )
                ind_previous = self._find_closest_pivot(
                    idx_previous, confirmed_indicator_highs, self.pivot_align_tolerance
                )

                if ind_current is not None and ind_previous is not None:
                    # Indicator makes lower high
                    if indicator[ind_current] < indicator[ind_previous]:
                        # Check magnitude
                        price_change = abs(price[idx_current] - price[idx_previous]) / price[idx_previous]
                        if price_change >= self.min_divergence_magnitude:
                            # Mark divergence at current bar
                            divergence[i] = 1
                            marked_pairs.add(pair_key)

        return divergence

    def detect_hidden_bullish_divergence(
        self,
        price: np.ndarray,
        price_lows_idx: np.ndarray,
        indicator: np.ndarray,
        indicator_lows_idx: np.ndarray
    ) -> np.ndarray:
        """
        Detect hidden bullish divergence: Price HL, Indicator LL.

        Hidden divergences signal trend continuation rather than reversal.

        CAUSAL IMPLEMENTATION: Processes bar-by-bar to ensure no look-ahead bias.

        Parameters
        ----------
        price : np.ndarray
            Price series
        price_lows_idx : np.ndarray
            Indices of price lows
        indicator : np.ndarray
            Indicator series
        indicator_lows_idx : np.ndarray
            Indices of indicator lows

        Returns
        -------
        divergence : np.ndarray
            Integer array (0 or 1) indicating divergence at each bar
        """
        n = len(price)
        divergence = np.zeros(n, dtype=np.int8)

        if len(price_lows_idx) < 2 or len(indicator_lows_idx) < 2:
            return divergence

        # Process bar-by-bar to ensure causality
        # Track which pivot pairs we've already marked to avoid duplicates
        marked_pairs = set()

        for i in range(n):
            # Get pivots confirmed up to current bar i
            max_pivot_idx = i - self.pivot_window
            confirmed_price_lows = price_lows_idx[price_lows_idx <= max_pivot_idx]
            confirmed_indicator_lows = indicator_lows_idx[indicator_lows_idx <= max_pivot_idx]

            if len(confirmed_price_lows) < 2:
                continue

            # Look for divergence in the lookback window
            lookback_start = max(0, i - self.lookback)
            recent_price_lows = confirmed_price_lows[confirmed_price_lows >= lookback_start]

            if len(recent_price_lows) < 2:
                continue

            # Check most recent pair of lows for divergence
            idx_current = recent_price_lows[-1]
            idx_previous = recent_price_lows[-2]

            # Skip if we've already marked this pair
            pair_key = (int(idx_previous), int(idx_current))
            if pair_key in marked_pairs:
                continue

            # Price makes higher low
            if price[idx_current] > price[idx_previous]:
                ind_current = self._find_closest_pivot(
                    idx_current, confirmed_indicator_lows, self.pivot_align_tolerance
                )
                ind_previous = self._find_closest_pivot(
                    idx_previous, confirmed_indicator_lows, self.pivot_align_tolerance
                )

                if ind_current is not None and ind_previous is not None:
                    # Indicator makes lower low
                    if indicator[ind_current] < indicator[ind_previous]:
                        price_change = abs(price[idx_current] - price[idx_previous]) / price[idx_previous]
                        if price_change >= self.min_divergence_magnitude:
                            # Mark divergence at current bar
                            divergence[i] = 1
                            marked_pairs.add(pair_key)

        return divergence

    def detect_hidden_bearish_divergence(
        self,
        price: np.ndarray,
        price_highs_idx: np.ndarray,
        indicator: np.ndarray,
        indicator_highs_idx: np.ndarray
    ) -> np.ndarray:
        """
        Detect hidden bearish divergence: Price LH, Indicator HH.

        Hidden divergences signal trend continuation rather than reversal.

        CAUSAL IMPLEMENTATION: Processes bar-by-bar to ensure no look-ahead bias.

        Parameters
        ----------
        price : np.ndarray
            Price series
        price_highs_idx : np.ndarray
            Indices of price highs
        indicator : np.ndarray
            Indicator series
        indicator_highs_idx : np.ndarray
            Indices of indicator highs

        Returns
        -------
        divergence : np.ndarray
            Integer array (0 or 1) indicating divergence at each bar
        """
        n = len(price)
        divergence = np.zeros(n, dtype=np.int8)

        if len(price_highs_idx) < 2 or len(indicator_highs_idx) < 2:
            return divergence

        # Process bar-by-bar to ensure causality
        # Track which pivot pairs we've already marked to avoid duplicates
        marked_pairs = set()

        for i in range(n):
            # Get pivots confirmed up to current bar i
            max_pivot_idx = i - self.pivot_window
            confirmed_price_highs = price_highs_idx[price_highs_idx <= max_pivot_idx]
            confirmed_indicator_highs = indicator_highs_idx[indicator_highs_idx <= max_pivot_idx]

            if len(confirmed_price_highs) < 2:
                continue

            # Look for divergence in the lookback window
            lookback_start = max(0, i - self.lookback)
            recent_price_highs = confirmed_price_highs[confirmed_price_highs >= lookback_start]

            if len(recent_price_highs) < 2:
                continue

            # Check most recent pair of highs for divergence
            idx_current = recent_price_highs[-1]
            idx_previous = recent_price_highs[-2]

            # Skip if we've already marked this pair
            pair_key = (int(idx_previous), int(idx_current))
            if pair_key in marked_pairs:
                continue

            # Price makes lower high
            if price[idx_current] < price[idx_previous]:
                ind_current = self._find_closest_pivot(
                    idx_current, confirmed_indicator_highs, self.pivot_align_tolerance
                )
                ind_previous = self._find_closest_pivot(
                    idx_previous, confirmed_indicator_highs, self.pivot_align_tolerance
                )

                if ind_current is not None and ind_previous is not None:
                    # Indicator makes higher high
                    if indicator[ind_current] > indicator[ind_previous]:
                        price_change = abs(price[idx_current] - price[idx_previous]) / price[idx_previous]
                        if price_change >= self.min_divergence_magnitude:
                            # Mark divergence at current bar
                            divergence[i] = 1
                            marked_pairs.add(pair_key)

        return divergence

    def calculate_divergence_strength(
        self,
        price: np.ndarray,
        indicator: np.ndarray,
        divergence_idx: np.ndarray,
        indicator_range: Tuple[float, float] = None,
        lookback_for_range: int = None
    ) -> np.ndarray:
        """
        Calculate strength score for detected divergences.

        Strength is based on:
        - Magnitude of price move
        - Magnitude of indicator divergence
        - Indicator position in range (oversold/overbought)

        CAUSAL IMPLEMENTATION: If indicator_range is None, calculates range
        dynamically for each bar using only historical data.

        Parameters
        ----------
        price : np.ndarray
            Price series
        indicator : np.ndarray
            Indicator series
        divergence_idx : np.ndarray
            Indices where divergence was detected
        indicator_range : tuple of (float, float), optional
            (min, max) normal range for indicator. If None, calculated dynamically.
        lookback_for_range : int, optional
            Lookback period for dynamic range calculation. Uses self.lookback if None.

        Returns
        -------
        strength : np.ndarray
            Strength scores (0-100) for each bar
        """
        n = len(price)
        strength = np.zeros(n, dtype=float)

        if lookback_for_range is None:
            lookback_for_range = self.lookback

        use_dynamic_range = indicator_range is None

        for idx in np.where(divergence_idx)[0]:
            if idx < self.strength_window:
                continue

            # Calculate score components
            window_start = idx - self.strength_window

            # 1. Price volatility in window
            price_window = price[window_start:idx + 1]
            price_range = np.max(price_window) - np.min(price_window)
            price_volatility = price_range / np.mean(price_window)

            # 2. Indicator extremity
            ind_value = indicator[idx]

            # Calculate indicator range causally (only using data up to current bar)
            if use_dynamic_range:
                range_start = max(0, idx - lookback_for_range + 1)
                indicator_window = indicator[range_start:idx + 1]
                # Filter out NaN values
                indicator_window = indicator_window[~np.isnan(indicator_window)]
                if len(indicator_window) > 0:
                    indicator_min = np.min(indicator_window)
                    indicator_max = np.max(indicator_window)
                else:
                    indicator_min = indicator_max = 0
            else:
                indicator_min, indicator_max = indicator_range

            if indicator_max > indicator_min and not np.isnan(ind_value):
                ind_normalized = (ind_value - indicator_min) / (indicator_max - indicator_min)
                ind_extremity = min(ind_normalized, 1 - ind_normalized)  # Distance from center
            else:
                ind_extremity = 0.5

            # 3. Combined score
            base_score = 50
            volatility_bonus = min(price_volatility * 100, 25)
            extremity_bonus = ind_extremity * 25

            strength[idx] = base_score + volatility_bonus + extremity_bonus

        return np.clip(strength, 0, 100)

    def _find_closest_pivot(
        self,
        target_idx: int,
        pivot_indices: np.ndarray,
        tolerance: int
    ) -> int:
        """
        Find the pivot index closest to target within tolerance.

        Parameters
        ----------
        target_idx : int
            Target bar index
        pivot_indices : np.ndarray
            Array of pivot indices
        tolerance : int
            Maximum distance to consider

        Returns
        -------
        closest_idx : int or None
            Closest pivot index within tolerance, or None if not found
        """
        if len(pivot_indices) == 0:
            return None

        distances = np.abs(pivot_indices - target_idx)
        min_dist_idx = np.argmin(distances)

        if distances[min_dist_idx] <= tolerance:
            return pivot_indices[min_dist_idx]

        return None


    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable, reproducible output."""
        return self.pivot_window * 2 + self.lookback

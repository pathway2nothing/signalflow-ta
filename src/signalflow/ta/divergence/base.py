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

        # Look at last few price lows within lookback
        recent_start = max(0, n - self.lookback)
        recent_price_lows = price_lows_idx[price_lows_idx >= recent_start]

        if len(recent_price_lows) < 2:
            return divergence

        # Check last 2 lows for divergence
        for i in range(len(recent_price_lows) - 1, 0, -1):
            idx_current = recent_price_lows[i]
            idx_previous = recent_price_lows[i - 1]

            # Price makes lower low
            if price[idx_current] < price[idx_previous]:
                # Find corresponding indicator lows
                ind_current = self._find_closest_pivot(
                    idx_current, indicator_lows_idx, self.pivot_align_tolerance
                )
                ind_previous = self._find_closest_pivot(
                    idx_previous, indicator_lows_idx, self.pivot_align_tolerance
                )

                if ind_current is not None and ind_previous is not None:
                    # Indicator makes higher low
                    if indicator[ind_current] > indicator[ind_previous]:
                        # Check magnitude
                        price_change = abs(price[idx_current] - price[idx_previous]) / price[idx_previous]
                        if price_change >= self.min_divergence_magnitude:
                            divergence[idx_current] = 1
                            break  # Only mark the most recent divergence

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

        # Look at last few price highs within lookback
        recent_start = max(0, n - self.lookback)
        recent_price_highs = price_highs_idx[price_highs_idx >= recent_start]

        if len(recent_price_highs) < 2:
            return divergence

        # Check last 2 highs for divergence
        for i in range(len(recent_price_highs) - 1, 0, -1):
            idx_current = recent_price_highs[i]
            idx_previous = recent_price_highs[i - 1]

            # Price makes higher high
            if price[idx_current] > price[idx_previous]:
                # Find corresponding indicator highs
                ind_current = self._find_closest_pivot(
                    idx_current, indicator_highs_idx, self.pivot_align_tolerance
                )
                ind_previous = self._find_closest_pivot(
                    idx_previous, indicator_highs_idx, self.pivot_align_tolerance
                )

                if ind_current is not None and ind_previous is not None:
                    # Indicator makes lower high
                    if indicator[ind_current] < indicator[ind_previous]:
                        # Check magnitude
                        price_change = abs(price[idx_current] - price[idx_previous]) / price[idx_previous]
                        if price_change >= self.min_divergence_magnitude:
                            divergence[idx_current] = 1
                            break  # Only mark the most recent divergence

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

        recent_start = max(0, n - self.lookback)
        recent_price_lows = price_lows_idx[price_lows_idx >= recent_start]

        if len(recent_price_lows) < 2:
            return divergence

        # Check for pattern: Price HL, Indicator LL
        for i in range(len(recent_price_lows) - 1, 0, -1):
            idx_current = recent_price_lows[i]
            idx_previous = recent_price_lows[i - 1]

            # Price makes higher low
            if price[idx_current] > price[idx_previous]:
                ind_current = self._find_closest_pivot(
                    idx_current, indicator_lows_idx, self.pivot_align_tolerance
                )
                ind_previous = self._find_closest_pivot(
                    idx_previous, indicator_lows_idx, self.pivot_align_tolerance
                )

                if ind_current is not None and ind_previous is not None:
                    # Indicator makes lower low
                    if indicator[ind_current] < indicator[ind_previous]:
                        price_change = abs(price[idx_current] - price[idx_previous]) / price[idx_previous]
                        if price_change >= self.min_divergence_magnitude:
                            divergence[idx_current] = 1
                            break

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

        recent_start = max(0, n - self.lookback)
        recent_price_highs = price_highs_idx[price_highs_idx >= recent_start]

        if len(recent_price_highs) < 2:
            return divergence

        # Check for pattern: Price LH, Indicator HH
        for i in range(len(recent_price_highs) - 1, 0, -1):
            idx_current = recent_price_highs[i]
            idx_previous = recent_price_highs[i - 1]

            # Price makes lower high
            if price[idx_current] < price[idx_previous]:
                ind_current = self._find_closest_pivot(
                    idx_current, indicator_highs_idx, self.pivot_align_tolerance
                )
                ind_previous = self._find_closest_pivot(
                    idx_previous, indicator_highs_idx, self.pivot_align_tolerance
                )

                if ind_current is not None and ind_previous is not None:
                    # Indicator makes higher high
                    if indicator[ind_current] > indicator[ind_previous]:
                        price_change = abs(price[idx_current] - price[idx_previous]) / price[idx_previous]
                        if price_change >= self.min_divergence_magnitude:
                            divergence[idx_current] = 1
                            break

        return divergence

    def calculate_divergence_strength(
        self,
        price: np.ndarray,
        indicator: np.ndarray,
        divergence_idx: np.ndarray,
        indicator_range: Tuple[float, float]
    ) -> np.ndarray:
        """
        Calculate strength score for detected divergences.

        Strength is based on:
        - Magnitude of price move
        - Magnitude of indicator divergence
        - Indicator position in range (oversold/overbought)

        Parameters
        ----------
        price : np.ndarray
            Price series
        indicator : np.ndarray
            Indicator series
        divergence_idx : np.ndarray
            Indices where divergence was detected
        indicator_range : tuple of (float, float)
            (min, max) normal range for indicator

        Returns
        -------
        strength : np.ndarray
            Strength scores (0-100) for each bar
        """
        n = len(price)
        strength = np.zeros(n, dtype=float)

        indicator_min, indicator_max = indicator_range

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
            if indicator_max > indicator_min:
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

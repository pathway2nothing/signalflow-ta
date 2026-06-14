"""Base Divergence Detection Class

Foundation for all divergence detectors with common pivot detection
and divergence pattern recognition logic.
"""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from signalflow.ta._compat import Feature
from signalflow.ta.divergence.pivot import (
    find_pivots_scipy,
    find_pivots_window,
)


@dataclass
class DivergenceBase(Feature):
    """Base class for divergence detection.

    Provides common functionality for detecting price-indicator divergences:
    - Pivot point detection (local highs and lows)
    - Divergence pattern recognition (regular and hidden)
    - Divergence strength calculation
    """

    pivot_window: int = 5
    """Window size for pivot detection (bars on each side)"""

    min_pivot_distance: int = 10
    """Minimum bars between consecutive pivots"""

    pivot_method: str = "window"
    """Pivot detection method: 'window' or 'scipy'"""

    lookback: int = 100
    """How many bars back to look for divergences"""

    min_divergence_magnitude: float = 0.02
    """Minimum divergence magnitude (as fraction, e.g., 0.02 = 2%)"""

    pivot_align_tolerance: int = 5
    """Maximum bar difference to consider price and indicator pivots aligned"""

    strength_window: int = 14
    """Window for calculating divergence strength context"""

    requires: ClassVar[list[str]] = ["high", "low", "close"]

    def find_pivots(self, series: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Find local highs and lows in a series."""
        if self.pivot_method == "scipy":
            return find_pivots_scipy(series, order=self.pivot_window, min_distance=self.min_pivot_distance)
        else:
            return find_pivots_window(series, window=self.pivot_window, min_distance=self.min_pivot_distance)

    def detect_regular_bullish_divergence(
        self,
        price: np.ndarray,
        price_lows_idx: np.ndarray,
        indicator: np.ndarray,
        indicator_lows_idx: np.ndarray,
    ) -> np.ndarray:
        """Detect regular bullish divergence: Price LL, Indicator HL.

        CAUSAL IMPLEMENTATION: Processes bar-by-bar to ensure no look-ahead bias.
        At each bar, checks if a divergence pattern can be confirmed based only
        on pivots that have been confirmed up to that bar.
        """
        n = len(price)
        divergence = np.zeros(n, dtype=np.int8)

        if len(price_lows_idx) < 2 or len(indicator_lows_idx) < 2:
            return divergence


        marked_pairs = set()

        for i in range(n):
            max_pivot_idx = i - self.pivot_window
            confirmed_price_lows = price_lows_idx[price_lows_idx <= max_pivot_idx]
            confirmed_indicator_lows = indicator_lows_idx[indicator_lows_idx <= max_pivot_idx]

            if len(confirmed_price_lows) < 2:
                continue

            lookback_start = max(0, i - self.lookback)
            recent_price_lows = confirmed_price_lows[confirmed_price_lows >= lookback_start]

            if len(recent_price_lows) < 2:
                continue

            idx_current = recent_price_lows[-1]
            idx_previous = recent_price_lows[-2]

            pair_key = (int(idx_previous), int(idx_current))
            if pair_key in marked_pairs:
                continue

            if price[idx_current] < price[idx_previous]:
                ind_current = self._find_closest_pivot(
                    idx_current, confirmed_indicator_lows, self.pivot_align_tolerance
                )
                ind_previous = self._find_closest_pivot(
                    idx_previous, confirmed_indicator_lows, self.pivot_align_tolerance
                )

                if (
                    ind_current is not None
                    and ind_previous is not None
                    and indicator[ind_current] > indicator[ind_previous]
                ):
                    price_change = abs(price[idx_current] - price[idx_previous]) / price[idx_previous]
                    if price_change >= self.min_divergence_magnitude:
                        divergence[i] = 1
                        marked_pairs.add(pair_key)

        return divergence

    def detect_regular_bearish_divergence(
        self,
        price: np.ndarray,
        price_highs_idx: np.ndarray,
        indicator: np.ndarray,
        indicator_highs_idx: np.ndarray,
    ) -> np.ndarray:
        """Detect regular bearish divergence: Price HH, Indicator LH.

        CAUSAL IMPLEMENTATION: Processes bar-by-bar to ensure no look-ahead bias.
        """
        n = len(price)
        divergence = np.zeros(n, dtype=np.int8)

        if len(price_highs_idx) < 2 or len(indicator_highs_idx) < 2:
            return divergence

        marked_pairs = set()

        for i in range(n):
            max_pivot_idx = i - self.pivot_window
            confirmed_price_highs = price_highs_idx[price_highs_idx <= max_pivot_idx]
            confirmed_indicator_highs = indicator_highs_idx[indicator_highs_idx <= max_pivot_idx]

            if len(confirmed_price_highs) < 2:
                continue

            lookback_start = max(0, i - self.lookback)
            recent_price_highs = confirmed_price_highs[confirmed_price_highs >= lookback_start]

            if len(recent_price_highs) < 2:
                continue

            idx_current = recent_price_highs[-1]
            idx_previous = recent_price_highs[-2]

            pair_key = (int(idx_previous), int(idx_current))
            if pair_key in marked_pairs:
                continue

            if price[idx_current] > price[idx_previous]:
                ind_current = self._find_closest_pivot(
                    idx_current, confirmed_indicator_highs, self.pivot_align_tolerance
                )
                ind_previous = self._find_closest_pivot(
                    idx_previous, confirmed_indicator_highs, self.pivot_align_tolerance
                )

                if (
                    ind_current is not None
                    and ind_previous is not None
                    and indicator[ind_current] < indicator[ind_previous]
                ):
                    price_change = abs(price[idx_current] - price[idx_previous]) / price[idx_previous]
                    if price_change >= self.min_divergence_magnitude:
                        divergence[i] = 1
                        marked_pairs.add(pair_key)

        return divergence

    def detect_hidden_bullish_divergence(
        self,
        price: np.ndarray,
        price_lows_idx: np.ndarray,
        indicator: np.ndarray,
        indicator_lows_idx: np.ndarray,
    ) -> np.ndarray:
        """Detect hidden bullish divergence: Price HL, Indicator LL.

        Hidden divergences signal trend continuation rather than reversal.

        CAUSAL IMPLEMENTATION: Processes bar-by-bar to ensure no look-ahead bias.
        """
        n = len(price)
        divergence = np.zeros(n, dtype=np.int8)

        if len(price_lows_idx) < 2 or len(indicator_lows_idx) < 2:
            return divergence

        marked_pairs = set()

        for i in range(n):
            max_pivot_idx = i - self.pivot_window
            confirmed_price_lows = price_lows_idx[price_lows_idx <= max_pivot_idx]
            confirmed_indicator_lows = indicator_lows_idx[indicator_lows_idx <= max_pivot_idx]

            if len(confirmed_price_lows) < 2:
                continue

            lookback_start = max(0, i - self.lookback)
            recent_price_lows = confirmed_price_lows[confirmed_price_lows >= lookback_start]

            if len(recent_price_lows) < 2:
                continue

            idx_current = recent_price_lows[-1]
            idx_previous = recent_price_lows[-2]

            pair_key = (int(idx_previous), int(idx_current))
            if pair_key in marked_pairs:
                continue

            if price[idx_current] > price[idx_previous]:
                ind_current = self._find_closest_pivot(
                    idx_current, confirmed_indicator_lows, self.pivot_align_tolerance
                )
                ind_previous = self._find_closest_pivot(
                    idx_previous, confirmed_indicator_lows, self.pivot_align_tolerance
                )

                if (
                    ind_current is not None
                    and ind_previous is not None
                    and indicator[ind_current] < indicator[ind_previous]
                ):
                    price_change = abs(price[idx_current] - price[idx_previous]) / price[idx_previous]
                    if price_change >= self.min_divergence_magnitude:
                        divergence[i] = 1
                        marked_pairs.add(pair_key)

        return divergence

    def detect_hidden_bearish_divergence(
        self,
        price: np.ndarray,
        price_highs_idx: np.ndarray,
        indicator: np.ndarray,
        indicator_highs_idx: np.ndarray,
    ) -> np.ndarray:
        """Detect hidden bearish divergence: Price LH, Indicator HH.

        Hidden divergences signal trend continuation rather than reversal.

        CAUSAL IMPLEMENTATION: Processes bar-by-bar to ensure no look-ahead bias.
        """
        n = len(price)
        divergence = np.zeros(n, dtype=np.int8)

        if len(price_highs_idx) < 2 or len(indicator_highs_idx) < 2:
            return divergence

        marked_pairs = set()

        for i in range(n):
            max_pivot_idx = i - self.pivot_window
            confirmed_price_highs = price_highs_idx[price_highs_idx <= max_pivot_idx]
            confirmed_indicator_highs = indicator_highs_idx[indicator_highs_idx <= max_pivot_idx]

            if len(confirmed_price_highs) < 2:
                continue

            lookback_start = max(0, i - self.lookback)
            recent_price_highs = confirmed_price_highs[confirmed_price_highs >= lookback_start]

            if len(recent_price_highs) < 2:
                continue

            idx_current = recent_price_highs[-1]
            idx_previous = recent_price_highs[-2]

            pair_key = (int(idx_previous), int(idx_current))
            if pair_key in marked_pairs:
                continue

            if price[idx_current] < price[idx_previous]:
                ind_current = self._find_closest_pivot(
                    idx_current, confirmed_indicator_highs, self.pivot_align_tolerance
                )
                ind_previous = self._find_closest_pivot(
                    idx_previous, confirmed_indicator_highs, self.pivot_align_tolerance
                )

                if (
                    ind_current is not None
                    and ind_previous is not None
                    and indicator[ind_current] > indicator[ind_previous]
                ):
                    price_change = abs(price[idx_current] - price[idx_previous]) / price[idx_previous]
                    if price_change >= self.min_divergence_magnitude:
                        divergence[i] = 1
                        marked_pairs.add(pair_key)

        return divergence

    def calculate_divergence_strength(
        self,
        price: np.ndarray,
        indicator: np.ndarray,
        divergence_idx: np.ndarray,
        indicator_range: tuple[float, float] | None = None,
        lookback_for_range: int | None = None,
    ) -> np.ndarray:
        """Calculate strength score for detected divergences.

        Strength is based on:
        - Magnitude of price move
        - Magnitude of indicator divergence
        - Indicator position in range (oversold/overbought)

        CAUSAL IMPLEMENTATION: If indicator_range is None, calculates range
        dynamically for each bar using only historical data.
        """
        n = len(price)
        strength = np.zeros(n, dtype=float)

        if lookback_for_range is None:
            lookback_for_range = self.lookback

        use_dynamic_range = indicator_range is None

        for idx in np.where(divergence_idx)[0]:
            if idx < self.strength_window:
                continue

            window_start = idx - self.strength_window

            price_window = price[window_start : idx + 1]
            price_range = np.max(price_window) - np.min(price_window)
            price_volatility = price_range / np.mean(price_window)

            ind_value = indicator[idx]

            if use_dynamic_range:
                range_start = max(0, idx - lookback_for_range + 1)
                indicator_window = indicator[range_start : idx + 1]
                indicator_window = indicator_window[~np.isnan(indicator_window)]
                if len(indicator_window) > 0:
                    indicator_min = np.min(indicator_window)
                    indicator_max = np.max(indicator_window)
                else:
                    indicator_min = indicator_max = 0
            else:
                assert indicator_range is not None
                indicator_min, indicator_max = indicator_range

            if indicator_max > indicator_min and not np.isnan(ind_value):
                ind_normalized = (ind_value - indicator_min) / (indicator_max - indicator_min)
                ind_extremity = min(ind_normalized, 1 - ind_normalized)
            else:
                ind_extremity = 0.5

            base_score = 50
            volatility_bonus = min(price_volatility * 100, 25)
            extremity_bonus = ind_extremity * 25

            strength[idx] = base_score + volatility_bonus + extremity_bonus

        return np.clip(strength, 0, 100)

    def _find_closest_pivot(self, target_idx: int, pivot_indices: np.ndarray, tolerance: int) -> int | None:
        """Find the pivot index closest to target within tolerance."""
        if len(pivot_indices) == 0:
            return None

        distances = np.abs(pivot_indices - target_idx)
        min_dist_idx = np.argmin(distances)

        if distances[min_dist_idx] <= tolerance:
            return int(pivot_indices[min_dist_idx])

        return None

    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable, reproducible output."""
        return self.pivot_window * 2 + self.lookback

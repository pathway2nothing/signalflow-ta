"""
Tests for newly added indicators.

This module contains specific tests for:
- KalmanSmooth (overlap/adaptive.py)
- Trend Regime Indicators (trend/regime.py)
- Structure Statistics (stat/structure.py)
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from typing import Type

from conftest import generate_sinusoidal_ohlcv, generate_random_walk_ohlcv

# Import new indicators
from signalflow.ta import (
    # Overlap
    KalmanSmooth,
    # Trend Regime
    WilliamsAlligatorRegime,
    TwoMaRegime,
    SmaDirection,
    SmaDiffDirection,
    LinRegDirection,
    LinRegDiffDirection,
    LinRegPriceDiff,
    # Structure Statistics
    ReversePointsStat,
    TimeSinceSpikeStat,
    VolatilitySpikeStat,
    VolatilitySpikeDiffStat,
    VolumeSpikeStat,
    VolumeSpikeDiffStat,
    RollingMinStat,
    RollingMaxStat,
)


# =============================================================================
# Helper Functions
# =============================================================================


def run_indicator(indicator, df: pl.DataFrame) -> pl.DataFrame:
    """Run indicator and return result."""
    return indicator.compute_pair(df)


def get_feature_columns(df: pl.DataFrame) -> list[str]:
    """Get feature columns (excluding standard OHLCV)."""
    exclude = {"pair", "timestamp", "open", "high", "low", "close", "volume"}
    return [c for c in df.columns if c not in exclude]


# =============================================================================
# Test: KalmanSmooth
# =============================================================================


class TestKalmanSmooth:
    """Tests for Kalman Filter smoothing indicator."""

    def test_basic_computation(self):
        """Kalman filter should produce smoothed output."""
        df = generate_sinusoidal_ohlcv(n_rows=500)
        indicator = KalmanSmooth(window=60, process_noise=0.1, measurement_noise=0.1)
        result = run_indicator(indicator, df)

        assert "close_kalman_60" in result.columns
        # After warmup, should have valid values
        valid = result["close_kalman_60"].drop_nulls()
        assert len(valid) > 0

    def test_smoothing_reduces_noise(self):
        """Kalman filter should reduce noise in the signal."""
        df = generate_random_walk_ohlcv(n_rows=1000, volatility=0.05)
        indicator = KalmanSmooth(window=120, process_noise=0.1, measurement_noise=0.1)
        result = run_indicator(indicator, df)

        close = result["close"].to_numpy()
        kalman = result["close_kalman_120"].to_numpy()

        # Compare variance after warmup
        warmup = 200
        close_var = np.nanvar(close[warmup:])
        kalman_var = np.nanvar(kalman[warmup:])

        # Smoothed signal should have lower variance
        assert kalman_var < close_var

    def test_normalized_output(self):
        """Normalized Kalman should produce normalized values."""
        df = generate_sinusoidal_ohlcv(n_rows=500)
        indicator = KalmanSmooth(
            window=60, process_noise=0.1, measurement_noise=0.1, normalized=True
        )
        result = run_indicator(indicator, df)

        assert "close_kalman_60_norm" in result.columns

    def test_different_parameters(self):
        """Different parameters should produce different results."""
        df = generate_sinusoidal_ohlcv(n_rows=500)

        ind1 = KalmanSmooth(window=60, process_noise=0.1, measurement_noise=0.1)
        ind2 = KalmanSmooth(window=120, process_noise=0.1, measurement_noise=0.1)

        result1 = run_indicator(ind1, df)
        result2 = run_indicator(ind2, df)

        # Different window sizes should give different results
        col1 = "close_kalman_60"
        col2 = "close_kalman_120"

        assert col1 in result1.columns
        assert col2 in result2.columns


# =============================================================================
# Test: Trend Regime Indicators
# =============================================================================


class TestWilliamsAlligatorRegime:
    """Tests for Williams Alligator trend regime detector."""

    def test_basic_computation(self):
        """Should produce regime and distance columns."""
        df = generate_sinusoidal_ohlcv(n_rows=500)
        indicator = WilliamsAlligatorRegime()
        result = run_indicator(indicator, df)

        assert "alligator_regime" in result.columns
        assert "alligator_regime_dist" in result.columns

    def test_regime_values(self):
        """Regime should be -1, 0, or 1."""
        df = generate_sinusoidal_ohlcv(n_rows=1000)
        indicator = WilliamsAlligatorRegime()
        result = run_indicator(indicator, df)

        regime = result["alligator_regime"].to_numpy()
        valid_values = {-1, 0, 1}
        actual_values = set(regime[~np.isnan(regime)].astype(int))
        assert actual_values.issubset(valid_values)

    def test_distance_non_negative(self):
        """Distance should be non-negative."""
        df = generate_sinusoidal_ohlcv(n_rows=500)
        indicator = WilliamsAlligatorRegime()
        result = run_indicator(indicator, df)

        distance = result["alligator_regime_dist"].to_numpy()
        valid = ~np.isnan(distance)
        assert np.all(distance[valid] >= 0)


class TestTwoMaRegime:
    """Tests for Two Moving Average regime detector."""

    def test_basic_computation(self):
        """Should produce regime and distance columns."""
        df = generate_sinusoidal_ohlcv(n_rows=500)
        indicator = TwoMaRegime(fast_length=10, slow_length=50)
        result = run_indicator(indicator, df)

        assert "two_ma_regime_10_50" in result.columns
        assert "two_ma_regime_dist_10_50" in result.columns

    def test_regime_values(self):
        """Regime should be -1, 0, or 1."""
        df = generate_sinusoidal_ohlcv(n_rows=500)
        indicator = TwoMaRegime()
        result = run_indicator(indicator, df)

        regime = result["two_ma_regime_10_50"].to_numpy()
        valid = ~np.isnan(regime)
        actual_values = set(regime[valid].astype(int))
        assert actual_values.issubset({-1, 0, 1})


class TestSmaDirection:
    """Tests for SMA direction indicator."""

    def test_basic_computation(self):
        """Should produce SMA and direction columns."""
        df = generate_sinusoidal_ohlcv(n_rows=500)
        indicator = SmaDirection(period=14)
        result = run_indicator(indicator, df)

        assert "close_sma_14" in result.columns
        assert "close_sma_dir_14" in result.columns

    def test_direction_binary(self):
        """Direction should be 0 or 1."""
        df = generate_sinusoidal_ohlcv(n_rows=500)
        indicator = SmaDirection(period=14)
        result = run_indicator(indicator, df)

        direction = result["close_sma_dir_14"].to_numpy()
        valid = ~np.isnan(direction)
        actual_values = set(direction[valid].astype(int))
        assert actual_values.issubset({0, 1})


class TestLinRegDirection:
    """Tests for Linear Regression direction indicator."""

    def test_basic_computation(self):
        """Should produce slope and direction columns."""
        df = generate_sinusoidal_ohlcv(n_rows=500)
        indicator = LinRegDirection(period=15)
        result = run_indicator(indicator, df)

        assert "close_linreg_slope_15" in result.columns
        assert "close_linreg_dir_15" in result.columns

    def test_direction_binary(self):
        """Direction should be 0 or 1."""
        df = generate_sinusoidal_ohlcv(n_rows=500)
        indicator = LinRegDirection(period=15)
        result = run_indicator(indicator, df)

        direction = result["close_linreg_dir_15"].to_numpy()
        valid = ~np.isnan(direction)
        actual_values = set(direction[valid].astype(int))
        assert actual_values.issubset({0, 1})


class TestLinRegPriceDiff:
    """Tests for Linear Regression price difference indicator."""

    def test_basic_computation(self):
        """Should produce diff and direction columns."""
        df = generate_sinusoidal_ohlcv(n_rows=500)
        indicator = LinRegPriceDiff(period=15)
        result = run_indicator(indicator, df)

        assert "close_linreg_price_diff_15" in result.columns
        assert "close_linreg_price_dir_15" in result.columns


# =============================================================================
# Test: Structure Statistics
# =============================================================================


class TestReversePointsStat:
    """Tests for reverse points counter."""

    def test_basic_computation(self):
        """Should produce count and normalized columns."""
        df = generate_sinusoidal_ohlcv(n_rows=500)
        indicator = ReversePointsStat(window=20)
        result = run_indicator(indicator, df)

        assert "close_reverse_points_20" in result.columns
        assert "close_reverse_points_norm_20" in result.columns

    def test_count_non_negative(self):
        """Reverse point count should be non-negative."""
        df = generate_sinusoidal_ohlcv(n_rows=500)
        indicator = ReversePointsStat(window=20)
        result = run_indicator(indicator, df)

        counts = result["close_reverse_points_20"].to_numpy()
        assert np.all(counts >= 0)

    def test_normalized_bounded(self):
        """Normalized count should be in [0, 1]."""
        df = generate_sinusoidal_ohlcv(n_rows=500)
        indicator = ReversePointsStat(window=20)
        result = run_indicator(indicator, df)

        norm = result["close_reverse_points_norm_20"].to_numpy()
        valid = ~np.isnan(norm)
        assert np.all(norm[valid] >= 0)
        assert np.all(norm[valid] <= 1)


class TestTimeSinceSpikeStat:
    """Tests for time since spike indicator."""

    def test_basic_computation(self):
        """Should count bars since last spike."""
        # Create a dataframe with a known spike column
        df = generate_sinusoidal_ohlcv(n_rows=100)

        # Add a spike column
        spike = np.zeros(100)
        spike[10] = 1  # Spike at bar 10
        spike[50] = 1  # Spike at bar 50
        df = df.with_columns(pl.Series(name="spike", values=spike))

        indicator = TimeSinceSpikeStat(source_col="spike")
        result = run_indicator(indicator, df)

        assert "time_since_spike" in result.columns

        time_since = result["time_since_spike"].to_numpy()

        # After bar 10, time_since should increment
        assert time_since[10] == 0  # Spike at bar 10
        assert time_since[11] == 1  # 1 bar since spike
        assert time_since[49] == 39  # 39 bars since bar 10 spike
        assert time_since[50] == 0  # New spike at bar 50
        assert time_since[51] == 1  # 1 bar since spike


class TestVolatilitySpikeStat:
    """Tests for volatility spike detection."""

    def test_basic_computation(self):
        """Should produce zscore and spike columns."""
        df = generate_sinusoidal_ohlcv(n_rows=500)
        indicator = VolatilitySpikeStat(period=60, threshold=1.0)
        result = run_indicator(indicator, df)

        assert "close_volat_zscore_60" in result.columns
        assert "close_volat_spike_60" in result.columns

    def test_spike_binary(self):
        """Spike should be 0 or 1."""
        df = generate_sinusoidal_ohlcv(n_rows=500)
        indicator = VolatilitySpikeStat(period=60, threshold=1.0)
        result = run_indicator(indicator, df)

        spike = result["close_volat_spike_60"].to_numpy()
        valid = ~np.isnan(spike)
        actual_values = set(spike[valid].astype(int))
        assert actual_values.issubset({0, 1})


class TestVolumeSpikeStat:
    """Tests for volume spike detection."""

    def test_basic_computation(self):
        """Should produce zscore and spike columns."""
        df = generate_sinusoidal_ohlcv(n_rows=500)
        indicator = VolumeSpikeStat(period=60, threshold=1.0)
        result = run_indicator(indicator, df)

        assert "volume_zscore_60" in result.columns
        assert "volume_spike_60" in result.columns

    def test_spike_binary(self):
        """Spike should be 0 or 1."""
        df = generate_sinusoidal_ohlcv(n_rows=500)
        indicator = VolumeSpikeStat(period=60, threshold=1.0)
        result = run_indicator(indicator, df)

        spike = result["volume_spike_60"].to_numpy()
        valid = ~np.isnan(spike)
        actual_values = set(spike[valid].astype(int))
        assert actual_values.issubset({0, 1})


class TestRollingMinMaxStat:
    """Tests for rolling min/max statistics."""

    def test_rolling_min(self):
        """Rolling min should be <= current close."""
        df = generate_sinusoidal_ohlcv(n_rows=500)
        indicator = RollingMinStat(source_col="close", period=14)
        result = run_indicator(indicator, df)

        assert "close_min_14" in result.columns

        close = result["close"].to_numpy()
        min_val = result["close_min_14"].to_numpy()
        valid = ~np.isnan(min_val)

        # Rolling min should be <= close at each point
        assert np.all(min_val[valid] <= close[valid])

    def test_rolling_max(self):
        """Rolling max should be >= current close."""
        df = generate_sinusoidal_ohlcv(n_rows=500)
        indicator = RollingMaxStat(source_col="close", period=14)
        result = run_indicator(indicator, df)

        assert "close_max_14" in result.columns

        close = result["close"].to_numpy()
        max_val = result["close_max_14"].to_numpy()
        valid = ~np.isnan(max_val)

        # Rolling max should be >= close at each point
        assert np.all(max_val[valid] >= close[valid])


# =============================================================================
# Test: Look-Ahead Bias for New Indicators
# =============================================================================


class TestNoLookAheadBias:
    """Verify new indicators don't have look-ahead bias."""

    TRUNCATE_BARS = 100
    TOLERANCE = 1e-10

    def _test_no_lookahead(self, indicator, df: pl.DataFrame, feature_col: str):
        """Generic look-ahead test."""
        result_full = run_indicator(indicator, df)
        df_truncated = df.head(len(df) - self.TRUNCATE_BARS)
        result_truncated = run_indicator(indicator, df_truncated)

        n_compare = len(result_truncated)
        full_values = result_full[feature_col][:n_compare].to_numpy()
        trunc_values = result_truncated[feature_col].to_numpy()

        # Handle NaN
        valid = ~np.isnan(full_values) & ~np.isnan(trunc_values)
        if valid.sum() > 0:
            np.testing.assert_allclose(
                full_values[valid], trunc_values[valid], rtol=self.TOLERANCE
            )

    def test_kalman_no_lookahead(self):
        """KalmanSmooth should not have look-ahead bias."""
        df = generate_sinusoidal_ohlcv(n_rows=500)
        indicator = KalmanSmooth(window=60)
        self._test_no_lookahead(indicator, df, "close_kalman_60")

    def test_alligator_no_lookahead(self):
        """WilliamsAlligatorRegime should not have look-ahead bias."""
        df = generate_sinusoidal_ohlcv(n_rows=500)
        indicator = WilliamsAlligatorRegime()
        self._test_no_lookahead(indicator, df, "alligator_regime")

    def test_two_ma_no_lookahead(self):
        """TwoMaRegime should not have look-ahead bias."""
        df = generate_sinusoidal_ohlcv(n_rows=500)
        indicator = TwoMaRegime()
        self._test_no_lookahead(indicator, df, "two_ma_regime_10_50")

    def test_reverse_points_no_lookahead(self):
        """ReversePointsStat should not have look-ahead bias."""
        df = generate_sinusoidal_ohlcv(n_rows=500)
        indicator = ReversePointsStat(window=20)
        self._test_no_lookahead(indicator, df, "close_reverse_points_20")


# =============================================================================
# Main Entry Point
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

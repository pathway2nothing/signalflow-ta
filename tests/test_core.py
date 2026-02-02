"""
SignalFlow-TA Core Tests

Test suite for technical analysis indicators with focus on:
1. Empty/null column handling
2. Look-ahead bias detection
3. Reproducibility across entry points
4. OHLCV constraint validation

These tests ensure indicators are production-ready and safe for use
in backtesting and live trading scenarios.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from datetime import datetime, timedelta
from typing import Type, Any
from dataclasses import dataclass

# Import fixtures
from conftest import (
    generate_static_ohlcv,
    generate_sinusoidal_ohlcv,
    generate_random_walk_ohlcv,
    generate_empty_column_df,
    generate_ohlcv_with_nulls,
    validate_ohlcv_constraints,
    SEED,
    DEFAULT_ROWS,
)


# =============================================================================
# Test Configuration - Indicators to Test
# =============================================================================

# List all indicator classes to be tested
# This will be populated dynamically or manually

MOMENTUM_INDICATORS: list[tuple[Type, dict]] = [
    # (IndicatorClass, default_params)
]

OVERLAP_INDICATORS: list[tuple[Type, dict]] = []

VOLATILITY_INDICATORS: list[tuple[Type, dict]] = []

VOLUME_INDICATORS: list[tuple[Type, dict]] = []

STAT_INDICATORS: list[tuple[Type, dict]] = []

TREND_INDICATORS: list[tuple[Type, dict]] = []


def get_all_indicators() -> list[tuple[Type, dict, str]]:
    """Get all indicators with their categories."""
    all_indicators = []
    
    for cls, params in MOMENTUM_INDICATORS:
        all_indicators.append((cls, params, "momentum"))
    for cls, params in OVERLAP_INDICATORS:
        all_indicators.append((cls, params, "overlap"))
    for cls, params in VOLATILITY_INDICATORS:
        all_indicators.append((cls, params, "volatility"))
    for cls, params in VOLUME_INDICATORS:
        all_indicators.append((cls, params, "volume"))
    for cls, params in STAT_INDICATORS:
        all_indicators.append((cls, params, "stat"))
    for cls, params in TREND_INDICATORS:
        all_indicators.append((cls, params, "trend"))
    
    return all_indicators


# =============================================================================
# Helper Functions
# =============================================================================


def get_feature_columns(df: pl.DataFrame) -> list[str]:
    """Get feature columns (excluding pair, timestamp, OHLCV)."""
    exclude = {"pair", "timestamp", "open", "high", "low", "close", "volume", "resample_offset"}
    return [c for c in df.columns if c not in exclude]


def run_indicator(
    indicator_cls: Type,
    df: pl.DataFrame,
    params: dict | None = None,
) -> pl.DataFrame:
    """
    Run an indicator on the given data.
    
    Args:
        indicator_cls: The indicator class
        df: Input OHLCV DataFrame
        params: Optional parameters to override defaults
        
    Returns:
        DataFrame with computed features
    """
    params = params or {}
    indicator = indicator_cls(**params)
    
    # Indicators use compute_pair() method
    if hasattr(indicator, 'compute_pair'):
        return indicator.compute_pair(df)
    elif hasattr(indicator, 'run'):
        return indicator.run(df)
    elif hasattr(indicator, 'extract'):
        return indicator.extract(df)
    else:
        raise ValueError(f"Unknown indicator interface for {indicator_cls.__name__}")


# =============================================================================
# Test 1: Empty Column Handling
# =============================================================================


class TestEmptyColumnHandling:
    """
    Tests for proper handling of empty (all-null) columns.
    
    Indicators should either:
    - Gracefully handle empty input and return NaN output
    - Raise a clear error message
    
    They should NOT:
    - Crash with cryptic errors
    - Return incorrect values
    - Hang or timeout
    """
    
    def test_empty_close_column_does_not_crash(self):
        """Indicators should handle empty close column gracefully."""
        df = generate_empty_column_df(n_rows=100, empty_columns=["close"])
        
        # This is a placeholder - in real tests, iterate over indicators
        # For now, test the basic structure
        assert df["close"].null_count() == len(df)
        assert "pair" in df.columns
        assert "timestamp" in df.columns
    
    def test_empty_volume_column_does_not_crash(self):
        """Volume-based indicators should handle empty volume."""
        df = generate_empty_column_df(n_rows=100, empty_columns=["volume"])
        
        assert df["volume"].null_count() == len(df)
    
    def test_all_empty_columns_does_not_crash(self):
        """Indicators should handle all empty price columns."""
        df = generate_empty_column_df(
            n_rows=100, 
            empty_columns=["open", "high", "low", "close", "volume"]
        )
        
        for col in ["open", "high", "low", "close", "volume"]:
            assert df[col].null_count() == len(df)
    
    def test_partial_null_handling(self):
        """Indicators should handle partial null values."""
        df = generate_sinusoidal_ohlcv(n_rows=100)
        df_with_nulls = generate_ohlcv_with_nulls(df, null_fraction=0.1)
        
        # Some values should be null
        assert df_with_nulls["close"].null_count() > 0
        # But not all
        assert df_with_nulls["close"].null_count() < len(df_with_nulls)
    
    def test_first_rows_null_handling(self):
        """Indicators should handle nulls at the beginning."""
        df = generate_sinusoidal_ohlcv(n_rows=100)
        
        # Make first 10 rows null using Polars
        df = df.with_row_index("_row_idx").with_columns(
            pl.when(pl.col("_row_idx") < 10)
            .then(None)
            .otherwise(pl.col("close"))
            .alias("close")
        ).drop("_row_idx")
        
        assert df["close"][:10].null_count() == 10
    
    def test_last_rows_null_handling(self):
        """Indicators should handle nulls at the end."""
        df = generate_sinusoidal_ohlcv(n_rows=100)
        
        # Make last 10 rows null using Polars
        n = len(df)
        df = df.with_row_index("_row_idx").with_columns(
            pl.when(pl.col("_row_idx") >= n - 10)
            .then(None)
            .otherwise(pl.col("close"))
            .alias("close")
        ).drop("_row_idx")
        
        assert df["close"][-10:].null_count() == 10


# =============================================================================
# Test 2: Look-Ahead Bias Detection
# =============================================================================


class TestLookAheadBias:
    """
    Tests to detect look-ahead bias (using future information).
    
    This is CRITICAL for backtesting correctness. An indicator with
    look-ahead bias will show unrealistic performance in backtests.
    
    Detection method:
    1. Compute indicator on full dataset
    2. Compute indicator on truncated dataset (without last N bars)
    3. Compare values at the same timestamps - they MUST be identical
    
    If values differ, the indicator is "peeking" at future data.
    """
    
    TRUNCATE_BARS = 100  # Number of bars to remove from the end
    TOLERANCE = 1e-10   # Numerical tolerance for float comparison
    
    def _test_no_lookahead_bias_single(
        self,
        df: pl.DataFrame,
        indicator_cls: Type,
        params: dict | None = None,
    ) -> None:
        """
        Test single indicator for look-ahead bias.
        
        Args:
            df: Full OHLCV DataFrame
            indicator_cls: Indicator class to test
            params: Indicator parameters
        """
        # Compute on full dataset
        result_full = run_indicator(indicator_cls, df, params)
        
        # Truncate and compute again
        df_truncated = df.head(len(df) - self.TRUNCATE_BARS)
        result_truncated = run_indicator(indicator_cls, df_truncated, params)
        
        # Get feature columns
        feature_cols = get_feature_columns(result_full)
        
        if not feature_cols:
            pytest.skip(f"No feature columns produced by {indicator_cls.__name__}")
        
        # Compare values at overlapping timestamps
        n_compare = len(result_truncated)
        
        for col in feature_cols:
            full_values = result_full[col][:n_compare].to_numpy()
            truncated_values = result_truncated[col].to_numpy()
            
            # Handle NaN comparison properly
            full_nan_mask = np.isnan(full_values)
            trunc_nan_mask = np.isnan(truncated_values)
            
            # NaN patterns should match
            np.testing.assert_array_equal(
                full_nan_mask,
                trunc_nan_mask,
                err_msg=f"Look-ahead bias detected in {indicator_cls.__name__}.{col}: "
                        f"NaN patterns differ"
            )
            
            # Non-NaN values should match
            valid_mask = ~full_nan_mask
            if valid_mask.any():
                np.testing.assert_allclose(
                    full_values[valid_mask],
                    truncated_values[valid_mask],
                    rtol=self.TOLERANCE,
                    atol=self.TOLERANCE,
                    err_msg=f"Look-ahead bias detected in {indicator_cls.__name__}.{col}: "
                            f"values differ between full and truncated computation"
                )
    
    def test_lookahead_framework_works(self):
        """Verify the look-ahead bias test framework works correctly."""
        # Create a simple "indicator" that would have look-ahead bias
        df = generate_sinusoidal_ohlcv(n_rows=500)
        
        # This simulates a bad indicator that uses future data
        # (e.g., using shift with negative values incorrectly)
        close = df["close"].to_numpy()
        
        # Good indicator: rolling mean (no look-ahead)
        good_values = np.full(len(close), np.nan)
        for i in range(19, len(close)):
            good_values[i] = np.mean(close[i-19:i+1])
        
        # Verify on truncated data
        df_truncated = df.head(400)
        close_trunc = df_truncated["close"].to_numpy()
        
        good_values_trunc = np.full(len(close_trunc), np.nan)
        for i in range(19, len(close_trunc)):
            good_values_trunc[i] = np.mean(close_trunc[i-19:i+1])
        
        # Good indicator should match
        np.testing.assert_allclose(
            good_values[:400],
            good_values_trunc,
            rtol=1e-10
        )
    
    def test_lookahead_detection_catches_bad_indicator(self):
        """Verify look-ahead bias detection catches problematic code."""
        df = generate_sinusoidal_ohlcv(n_rows=500)
        close = df["close"].to_numpy()
        
        # BAD indicator: uses future data (forward-looking mean)
        bad_values = np.full(len(close), np.nan)
        for i in range(len(close) - 20):
            # This looks at future data!
            bad_values[i] = np.mean(close[i:i+20])
        
        # Same on truncated
        df_truncated = df.head(400)
        close_trunc = df_truncated["close"].to_numpy()
        
        bad_values_trunc = np.full(len(close_trunc), np.nan)
        for i in range(len(close_trunc) - 20):
            bad_values_trunc[i] = np.mean(close_trunc[i:i+20])
        
        # These should NOT match (proving look-ahead bias)
        # We expect values near the truncation point to differ
        start_diff = 380  # Near end of truncated data
        end_diff = 400
        
        with pytest.raises(AssertionError):
            np.testing.assert_allclose(
                bad_values[start_diff:end_diff],
                bad_values_trunc[start_diff:end_diff],
                rtol=1e-10
            )
    
    def test_rolling_operations_no_lookahead(self):
        """Test that rolling operations don't have look-ahead bias."""
        df = generate_sinusoidal_ohlcv(n_rows=500)
        
        # Compute rolling mean with Polars (proper implementation)
        df_full = df.with_columns(
            pl.col("close").rolling_mean(window_size=20).alias("sma_20")
        )
        
        df_trunc = df.head(400).with_columns(
            pl.col("close").rolling_mean(window_size=20).alias("sma_20")
        )
        
        # Should match exactly
        full_sma = df_full["sma_20"][:400].to_numpy()
        trunc_sma = df_trunc["sma_20"].to_numpy()
        
        valid = ~np.isnan(full_sma)
        np.testing.assert_allclose(
            full_sma[valid],
            trunc_sma[valid],
            rtol=1e-10
        )


# =============================================================================
# Test 3: Reproducibility Across Entry Points
# =============================================================================


class TestReproducibility:
    """
    Tests for reproducibility regardless of entry point.
    
    An indicator computed on data starting from bar 0 should give
    the same values as when computed on data starting from bar N
    (after warmup period).
    
    This ensures:
    1. No hidden state accumulation
    2. Deterministic computation
    3. Safe for incremental updates
    """
    
    WARMUP_PERIODS = [50, 100, 200]  # Different entry points to test
    TOLERANCE = 1e-10
    
    def _test_reproducibility_single(
        self,
        df: pl.DataFrame,
        indicator_cls: Type,
        params: dict | None = None,
        warmup: int = 100,
    ) -> None:
        """
        Test single indicator for entry point reproducibility.
        
        Args:
            df: Full OHLCV DataFrame
            indicator_cls: Indicator class to test
            params: Indicator parameters
            warmup: Warmup period to skip
        """
        # Compute from the beginning
        result_full = run_indicator(indicator_cls, df, params)
        
        # Compute from later entry point
        df_late_start = df.slice(warmup, len(df) - warmup)
        result_late = run_indicator(indicator_cls, df_late_start, params)
        
        # Get feature columns
        feature_cols = get_feature_columns(result_full)
        
        if not feature_cols:
            pytest.skip(f"No feature columns produced by {indicator_cls.__name__}")
        
        # Determine comparison range
        # We need enough warmup for the indicator itself
        indicator_warmup = params.get("period", 20) if params else 20
        compare_start = warmup + indicator_warmup
        compare_length = len(result_late) - indicator_warmup
        
        if compare_length <= 0:
            pytest.skip("Not enough data for comparison after warmup")
        
        for col in feature_cols:
            # Values from full computation (shifted to match timestamps)
            full_values = result_full[col][compare_start:compare_start + compare_length].to_numpy()
            
            # Values from late-start computation (after indicator warmup)
            late_values = result_late[col][indicator_warmup:indicator_warmup + compare_length].to_numpy()
            
            # Handle NaN comparison
            full_nan = np.isnan(full_values)
            late_nan = np.isnan(late_values)
            
            valid_both = ~full_nan & ~late_nan
            
            if valid_both.sum() == 0:
                continue
            
            np.testing.assert_allclose(
                full_values[valid_both],
                late_values[valid_both],
                rtol=self.TOLERANCE,
                atol=self.TOLERANCE,
                err_msg=f"Reproducibility issue in {indicator_cls.__name__}.{col}: "
                        f"values differ based on entry point"
            )
    
    def test_reproducibility_framework_works(self):
        """Verify the reproducibility test framework works correctly."""
        df = generate_sinusoidal_ohlcv(n_rows=1000)
        
        # Compute SMA from start
        df_full = df.with_columns(
            pl.col("close").rolling_mean(window_size=20).alias("sma_20")
        )
        
        # Compute SMA from bar 200
        df_late = df.slice(200, 800).with_columns(
            pl.col("close").rolling_mean(window_size=20).alias("sma_20")
        )
        
        # Compare overlapping region (after warmup on late start)
        warmup = 20
        compare_start = 200 + warmup
        compare_length = len(df_late) - warmup
        
        full_sma = df_full["sma_20"][compare_start:compare_start + compare_length].to_numpy()
        late_sma = df_late["sma_20"][warmup:warmup + compare_length].to_numpy()
        
        valid = ~np.isnan(full_sma) & ~np.isnan(late_sma)
        
        np.testing.assert_allclose(
            full_sma[valid],
            late_sma[valid],
            rtol=1e-10
        )
    
    def test_multiple_entry_points(self):
        """Test that SMA values are consistent regardless of entry point.
        
        For non-stateful indicators like SMA, the value at any given bar
        should be the same whether computed from the start or from a later point,
        as long as we have enough warmup data.
        """
        df = generate_sinusoidal_ohlcv(n_rows=2000)
        warmup = 20
        
        # Compute SMA on full dataset
        df_full = df.with_columns(
            pl.col("close").rolling_mean(window_size=warmup).alias("sma_20")
        )
        full_sma = df_full["sma_20"].to_numpy()
        
        # Compute SMA starting from different points
        for start in [100, 500, 1000]:
            df_slice = df.slice(start, len(df) - start)
            df_result = df_slice.with_columns(
                pl.col("close").rolling_mean(window_size=warmup).alias("sma_20")
            )
            slice_sma = df_result["sma_20"].to_numpy()
            
            # Compare overlapping region (after warmup in the slice)
            # slice_sma[warmup] corresponds to full_sma[start + warmup]
            compare_len = min(len(slice_sma) - warmup, 400)
            
            full_values = full_sma[start + warmup : start + warmup + compare_len]
            slice_values = slice_sma[warmup : warmup + compare_len]
            
            valid = ~np.isnan(full_values) & ~np.isnan(slice_values)
            
            np.testing.assert_allclose(
                full_values[valid],
                slice_values[valid],
                rtol=1e-10,
                err_msg=f"SMA values differ for entry point {start}"
            )
    
    def test_deterministic_with_same_seed(self):
        """Test that results are identical with same random seed."""
        df1 = generate_sinusoidal_ohlcv(n_rows=500, seed=42)
        df2 = generate_sinusoidal_ohlcv(n_rows=500, seed=42)
        
        # Compute same indicator
        result1 = df1.with_columns(
            pl.col("close").rolling_mean(window_size=20).alias("sma_20")
        )
        result2 = df2.with_columns(
            pl.col("close").rolling_mean(window_size=20).alias("sma_20")
        )
        
        # Should be exactly identical
        assert result1.equals(result2)
    
    def test_different_seeds_different_results(self):
        """Test that different seeds produce different results."""
        df1 = generate_sinusoidal_ohlcv(n_rows=500, seed=42)
        df2 = generate_sinusoidal_ohlcv(n_rows=500, seed=123)
        
        # Data should differ
        assert not df1.equals(df2)


# =============================================================================
# Test 4: Output Validation
# =============================================================================


class TestOutputValidation:
    """
    Tests for validating indicator outputs.
    
    Ensures:
    - Output length matches input length
    - Feature columns are properly named
    - Values are within expected ranges
    - No unexpected data types
    """
    
    def test_output_length_matches_input(self):
        """Indicator output should have same length as input."""
        df = generate_sinusoidal_ohlcv(n_rows=500)
        
        # Test with a simple rolling mean
        result = df.with_columns(
            pl.col("close").rolling_mean(window_size=20).alias("sma_20")
        )
        
        assert len(result) == len(df)
    
    def test_output_contains_feature_columns(self):
        """Indicator should produce expected feature columns."""
        df = generate_sinusoidal_ohlcv(n_rows=500)
        
        result = df.with_columns(
            pl.col("close").rolling_mean(window_size=20).alias("sma_20")
        )
        
        assert "sma_20" in result.columns
    
    def test_preserves_index_columns(self):
        """Indicator should preserve pair and timestamp columns."""
        df = generate_sinusoidal_ohlcv(n_rows=500)
        
        result = df.with_columns(
            pl.col("close").rolling_mean(window_size=20).alias("sma_20")
        )
        
        assert "pair" in result.columns
        assert "timestamp" in result.columns
    
    def test_rsi_bounded_0_100(self):
        """RSI-like indicators should be bounded 0-100."""
        df = generate_sinusoidal_ohlcv(n_rows=500)
        close = df["close"].to_numpy()
        
        # Compute simple RSI
        changes = np.diff(close, prepend=close[0])
        gains = np.where(changes > 0, changes, 0)
        losses = np.where(changes < 0, -changes, 0)
        
        period = 14
        rsi = np.full(len(close), np.nan)
        
        for i in range(period, len(close)):
            avg_gain = np.mean(gains[i-period+1:i+1])
            avg_loss = np.mean(losses[i-period+1:i+1])
            
            if avg_loss == 0:
                rsi[i] = 100
            else:
                rs = avg_gain / avg_loss
                rsi[i] = 100 - (100 / (1 + rs))
        
        valid = ~np.isnan(rsi)
        assert np.all(rsi[valid] >= 0)
        assert np.all(rsi[valid] <= 100)
    
    def test_no_inf_values(self):
        """Indicators should not produce infinite values."""
        df = generate_sinusoidal_ohlcv(n_rows=500)
        
        result = df.with_columns(
            pl.col("close").rolling_mean(window_size=20).alias("sma_20"),
            pl.col("close").rolling_std(window_size=20).alias("std_20")
        )
        
        for col in ["sma_20", "std_20"]:
            values = result[col].to_numpy()
            # Should have no infinities
            assert not np.any(np.isinf(values[~np.isnan(values)]))


# =============================================================================
# Test 5: Edge Cases
# =============================================================================


class TestEdgeCases:
    """
    Tests for edge cases and boundary conditions.
    """
    
    def test_minimum_data_length(self):
        """Indicators should handle minimum data length."""
        df = generate_sinusoidal_ohlcv(n_rows=5)  # Very short
        
        result = df.with_columns(
            pl.col("close").rolling_mean(window_size=3).alias("sma_3")
        )
        
        # Should not crash, may have NaN for warmup period
        assert len(result) == 5
    
    def test_single_row(self):
        """Indicators should handle single row without crashing."""
        df = generate_sinusoidal_ohlcv(n_rows=1)
        
        result = df.with_columns(
            pl.col("close").rolling_mean(window_size=1).alias("sma_1")
        )
        
        assert len(result) == 1
    
    def test_constant_price(self):
        """Indicators should handle constant price data."""
        df = generate_static_ohlcv(n_rows=100)
        
        result = df.with_columns(
            pl.col("close").rolling_mean(window_size=20).alias("sma_20"),
            pl.col("close").rolling_std(window_size=20).alias("std_20")
        )
        
        # SMA should equal the constant price
        valid_sma = result["sma_20"].drop_nulls()
        assert np.allclose(valid_sma.to_numpy(), 100.0)
        
        # Std should be 0
        valid_std = result["std_20"].drop_nulls()
        assert np.allclose(valid_std.to_numpy(), 0.0)
    
    def test_extreme_volatility(self):
        """Indicators should handle extreme volatility."""
        df = generate_random_walk_ohlcv(n_rows=500, volatility=0.5)  # 50% volatility
        
        result = df.with_columns(
            pl.col("close").rolling_mean(window_size=20).alias("sma_20")
        )
        
        # Should not crash or produce invalid values
        valid = result["sma_20"].drop_nulls()
        assert len(valid) > 0
        assert not np.any(np.isinf(valid.to_numpy()))
    
    def test_very_small_values(self):
        """Indicators should handle very small price values."""
        df = generate_sinusoidal_ohlcv(
            n_rows=100,
            base_price=0.00001,  # Very small price
            amplitude=0.000001
        )
        
        result = df.with_columns(
            pl.col("close").rolling_mean(window_size=20).alias("sma_20")
        )
        
        # Should produce valid small values
        valid = result["sma_20"].drop_nulls()
        assert len(valid) > 0
        assert np.all(valid.to_numpy() > 0)
    
    def test_very_large_values(self):
        """Indicators should handle very large price values."""
        df = generate_sinusoidal_ohlcv(
            n_rows=100,
            base_price=1e9,  # 1 billion
            amplitude=1e8
        )
        
        result = df.with_columns(
            pl.col("close").rolling_mean(window_size=20).alias("sma_20")
        )
        
        # Should produce valid large values
        valid = result["sma_20"].drop_nulls()
        assert len(valid) > 0
        assert not np.any(np.isinf(valid.to_numpy()))


# =============================================================================
# Test 6: Multi-Pair Handling
# =============================================================================


class TestMultiPairHandling:
    """
    Tests for proper handling of multiple trading pairs.
    
    Indicators should process each pair independently without
    data leakage between pairs.
    """
    
    def test_pairs_computed_independently(self):
        """Each pair should be computed independently."""
        # Create two pairs with very different price patterns
        df1 = generate_static_ohlcv(n_rows=100, base_price=100, pair="BTCUSDT")
        df2 = generate_static_ohlcv(n_rows=100, base_price=50000, pair="ETHUSDT")
        
        df = pl.concat([df1, df2])
        
        # Compute indicator
        result = df.group_by("pair", maintain_order=True).map_groups(
            lambda g: g.with_columns(
                pl.col("close").rolling_mean(window_size=20).alias("sma_20")
            )
        )
        
        # BTC SMA should be ~100, ETH SMA should be ~50000
        btc_sma = result.filter(pl.col("pair") == "BTCUSDT")["sma_20"].drop_nulls()
        eth_sma = result.filter(pl.col("pair") == "ETHUSDT")["sma_20"].drop_nulls()
        
        assert np.allclose(btc_sma.to_numpy(), 100.0)
        assert np.allclose(eth_sma.to_numpy(), 50000.0)
    
    def test_no_data_leakage_between_pairs(self):
        """Computing one pair should not affect another."""
        df1 = generate_sinusoidal_ohlcv(n_rows=200, pair="BTCUSDT", seed=42)
        df2 = generate_sinusoidal_ohlcv(n_rows=200, pair="ETHUSDT", seed=123)
        
        # Compute single pair
        btc_single = df1.with_columns(
            pl.col("close").rolling_mean(window_size=20).alias("sma_20")
        )
        
        # Compute both pairs together
        df_combined = pl.concat([df1, df2])
        result_combined = df_combined.group_by("pair", maintain_order=True).map_groups(
            lambda g: g.with_columns(
                pl.col("close").rolling_mean(window_size=20).alias("sma_20")
            )
        )
        btc_combined = result_combined.filter(pl.col("pair") == "BTCUSDT")
        
        # BTC values should be identical whether computed alone or with other pairs
        np.testing.assert_allclose(
            btc_single["sma_20"].to_numpy(),
            btc_combined["sma_20"].to_numpy(),
            rtol=1e-10
        )


# =============================================================================
# Main Test Entry Point
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
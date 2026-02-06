"""
SignalFlow-TA Indicator Tests

Simple, clean tests for all technical analysis indicators.
Tests only two critical properties:
1. Reproducibility - same results from different entry points
2. No look-ahead bias - future data doesn't affect past values
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from conftest import generate_test_ohlcv, SEED
from indicator_registry import IndicatorConfig


# =============================================================================
# Test Configuration
# =============================================================================

# Test periods - simple and clear
LONG_PERIOD = 50000  # Long time series
SHORT_PERIOD = 40000  # Short time series (LONG_PERIOD - TEST_LENGTH)
TEST_LENGTH = 1000  # Comparison window


# =============================================================================
# Helper Functions
# =============================================================================


def run_indicator(config: IndicatorConfig, df: pl.DataFrame) -> pl.DataFrame:
    """Run indicator on data and return result."""
    indicator = config.cls(**config.params)
    return indicator.compute_pair(df)


def get_output_columns(result: pl.DataFrame) -> list[str]:
    """Get output columns (all except standard OHLCV columns)."""
    standard = {
        "pair",
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "resample_offset",
    }
    return [col for col in result.columns if col not in standard]


def skip_if_no_config(config):
    """Skip test if no config available."""
    if config is None:
        pytest.skip("No indicators available - install signalflow.ta")


# =============================================================================
# Reproducibility Test
# =============================================================================


class TestReproducibility:
    """
    Test that indicators produce identical results from different entry points.

    Critical for backtesting: Dec values must match whether we compute Jan-Dec or Jul-Dec.

    Method:
    1. Compute indicator on LONG_PERIOD data
    2. Compute indicator on SHORT_PERIOD data (later start)
    3. Compare last TEST_LENGTH values
    4. Both series end at same point, but start at different points
    5. Tolerance: 0.01% (1e-4)
    """

    TOLERANCE = 1e-4  # 0.01%

    def test_reproducibility(self, config: IndicatorConfig):
        """Indicator must give same results regardless of entry point."""
        skip_if_no_config(config)

        # Generate long data
        df_long = generate_test_ohlcv(n_rows=LONG_PERIOD, seed=SEED)

        # Take short data from the end of long data (same end point, later start)
        df_short = df_long.tail(SHORT_PERIOD)

        # Compute on both
        result_long = run_indicator(config, df_long)
        result_short = run_indicator(config, df_short)

        # Get output columns
        out_cols = get_output_columns(result_long)

        if not out_cols:
            pytest.skip(f"{config.name} produces no output columns")

        # Compare last TEST_LENGTH values from both
        # Long: take last TEST_LENGTH rows
        # Short: take last TEST_LENGTH rows
        # These should be identical

        failures = []

        for col in out_cols:
            if col not in result_short.columns:
                continue

            # Last TEST_LENGTH values
            vals_long = result_long[col][-TEST_LENGTH:].to_numpy()
            vals_short = result_short[col][-TEST_LENGTH:].to_numpy()

            # Check valid (non-NaN) values
            valid = ~np.isnan(vals_long) & ~np.isnan(vals_short)

            if valid.sum() < 100:
                continue  # Not enough valid data

            vals_long_valid = vals_long[valid]
            vals_short_valid = vals_short[valid]

            # Relative difference
            with np.errstate(divide="ignore", invalid="ignore"):
                scale = np.maximum(np.abs(vals_long_valid), np.abs(vals_short_valid))
                scale = np.where(scale < 1e-10, 1.0, scale)
                rel_diff = np.abs(vals_long_valid - vals_short_valid) / scale

            max_diff = np.nanmax(rel_diff)

            if max_diff > self.TOLERANCE:
                worst_idx = np.nanargmax(rel_diff)
                failures.append(
                    {
                        "column": col,
                        "max_diff_pct": max_diff * 100,
                        "long_value": vals_long_valid[worst_idx],
                        "short_value": vals_short_valid[worst_idx],
                    }
                )

        if failures:
            msg = [
                f"\nREPRODUCIBILITY FAILURE: {config.name}",
                f"Indicator values differ based on entry point!",
                f"",
                f"Test setup:",
                f"  Long period: {LONG_PERIOD} bars",
                f"  Short period: {SHORT_PERIOD} bars",
                f"  Comparison: last {TEST_LENGTH} bars",
                f"  Tolerance: 0.01%",
                f"",
            ]

            for f in failures:
                msg.extend(
                    [
                        f"Column: {f['column']}",
                        f"  Max diff: {f['max_diff_pct']:.6f}%",
                        f"  Long value: {f['long_value']:.8f}",
                        f"  Short value: {f['short_value']:.8f}",
                        f"",
                    ]
                )

            pytest.fail("\n".join(msg))


# =============================================================================
# Look-Ahead Bias Test
# =============================================================================


class TestLookAhead:
    """
    Test that indicators don't use future data.

    Critical for backtesting: value at bar N must not depend on data after bar N.

    Method:
    1. Compute indicator on LONG_PERIOD data
    2. Compute indicator on SHORT_PERIOD data (truncated earlier)
    3. Compare TEST_LENGTH values before end of short series
    4. Values must be EXACTLY identical (no tolerance)
    """

    TOLERANCE = 0.0  # No tolerance - must be exact

    def test_no_lookahead(self, config: IndicatorConfig):
        """Indicator must not use future data."""
        skip_if_no_config(config)

        # Generate long data
        df_long = generate_test_ohlcv(n_rows=LONG_PERIOD, seed=SEED)

        # Truncate to short (same start, earlier end)
        df_short = df_long.head(SHORT_PERIOD)

        # Compute on both
        result_long = run_indicator(config, df_long)
        result_short = run_indicator(config, df_short)

        # Get output columns
        out_cols = get_output_columns(result_long)

        if not out_cols:
            pytest.skip(f"{config.name} produces no output columns")

        # Compare TEST_LENGTH values before end of short series
        # We compare from (SHORT_PERIOD - TEST_LENGTH) to SHORT_PERIOD
        # in both series - these must be IDENTICAL

        start_idx = SHORT_PERIOD - TEST_LENGTH
        end_idx = SHORT_PERIOD

        failures = []

        for col in out_cols:
            if col not in result_short.columns:
                continue

            # Get overlapping region
            vals_long = result_long[col][start_idx:end_idx].to_numpy()
            vals_short = result_short[col][start_idx:end_idx].to_numpy()

            # Check NaN patterns match
            nan_long = np.isnan(vals_long)
            nan_short = np.isnan(vals_short)

            if not np.array_equal(nan_long, nan_short):
                failures.append(f"{col}: NaN patterns differ")
                continue

            # Check values match exactly where valid
            valid = ~nan_long

            if valid.any():
                vals_long_valid = vals_long[valid]
                vals_short_valid = vals_short[valid]

                diff = np.abs(vals_long_valid - vals_short_valid)
                max_diff = np.max(diff)

                if max_diff > self.TOLERANCE:
                    worst_idx = np.argmax(diff)
                    failures.append(
                        f"{col}: max diff = {max_diff:.2e} at index {worst_idx} "
                        f"(long={vals_long_valid[worst_idx]:.8f}, short={vals_short_valid[worst_idx]:.8f})"
                    )

        if failures:
            msg = [
                f"\nLOOK-AHEAD BIAS DETECTED: {config.name}",
                f"Indicator uses future data!",
                f"",
                f"Test setup:",
                f"  Long period: {LONG_PERIOD} bars",
                f"  Short period: {SHORT_PERIOD} bars (truncated)",
                f"  Comparison: {TEST_LENGTH} bars before truncation point",
                f"  Tolerance: 0 (must be exact)",
                f"",
                f"Failures:",
            ]

            for failure in failures:
                msg.append(f"  - {failure}")

            pytest.fail("\n".join(msg))


# =============================================================================
# Basic Output Validation
# =============================================================================


class TestBasicValidation:
    """Basic sanity checks for indicator outputs."""

    def test_output_length_preserved(self, config: IndicatorConfig):
        """Output should have same length as input."""
        skip_if_no_config(config)

        df = generate_test_ohlcv(n_rows=500, seed=SEED)
        result = run_indicator(config, df)

        assert len(result) == len(df), (
            f"{config.name}: output length {len(result)} != input length {len(df)}"
        )

    def test_produces_output_columns(self, config: IndicatorConfig):
        """Indicator should produce output columns."""
        skip_if_no_config(config)

        df = generate_test_ohlcv(n_rows=500, seed=SEED)
        result = run_indicator(config, df)

        out_cols = get_output_columns(result)

        assert len(out_cols) > 0, f"{config.name}: no output columns produced"

    def test_no_inf_values(self, config: IndicatorConfig):
        """Output should not contain infinite values."""
        skip_if_no_config(config)

        df = generate_test_ohlcv(n_rows=500, seed=SEED)
        result = run_indicator(config, df)

        out_cols = get_output_columns(result)

        for col in out_cols:
            values = result[col].to_numpy()
            valid = ~np.isnan(values)

            if valid.any():
                assert not np.any(np.isinf(values[valid])), (
                    f"{config.name}.{col}: contains infinite values"
                )


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

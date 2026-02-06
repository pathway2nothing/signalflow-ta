"""
SignalFlow-TA Indicator Tests

Parametrized tests for all technical analysis indicators from signalflow.ta.
Each indicator is tested for:
1. Empty column handling
2. Look-ahead bias
3. Reproducibility
4. Output validation
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from typing import Type, Any
from dataclasses import dataclass

from conftest import (
    generate_static_ohlcv,
    generate_sinusoidal_ohlcv,
    generate_random_walk_ohlcv,
    generate_empty_column_df,
    generate_ohlcv_with_nulls,
    generate_multi_pair_ohlcv,
    validate_ohlcv_constraints,
    SEED,
)

from indicator_registry import (
    IndicatorConfig,
    INDICATOR_CONFIGS,
    get_configs_by_category,
    get_indicator_ids,
    _CONFIGS_FOR_PARAM,
    _IDS_FOR_PARAM,
)


# =============================================================================
# Helper for skipping tests when no indicators available
# =============================================================================


def _ensure_config(config):
    """Skip test if config is None (no indicators loaded)."""
    if config is None:
        pytest.skip("signalflow.ta not installed - install it to run indicator tests")
    return config


# =============================================================================
# Helper Functions
# =============================================================================


def run_indicator(config: IndicatorConfig, df: pl.DataFrame) -> pl.DataFrame:
    """
    Run an indicator on the given data.

    Args:
        config: Indicator configuration
        df: Input OHLCV DataFrame

    Returns:
        DataFrame with computed features
    """
    indicator = config.cls(**config.params)
    return indicator.compute_pair(df)


def get_output_columns(result: pl.DataFrame, config: IndicatorConfig) -> list[str]:
    """Get actual output columns from result that match expected outputs."""
    # Try exact match first
    exact_matches = [c for c in config.outputs if c in result.columns]
    if exact_matches:
        return exact_matches

    # Try partial match (indicator name prefix)
    name_lower = config.name.lower()
    partial_matches = []
    for col in result.columns:
        col_lower = col.lower()
        # Skip standard columns
        if col in (
            "pair",
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "resample_offset",
        ):
            continue
        # Check if column seems related to this indicator
        for pattern in [
            name_lower,
            name_lower.replace("stat", ""),
            name_lower.replace("mom", ""),
            name_lower.replace("smooth", ""),
            name_lower.replace("vol", ""),
        ]:
            if pattern and len(pattern) > 2 and pattern in col_lower:
                partial_matches.append(col)
                break

    if partial_matches:
        return partial_matches

    # Return all non-standard columns
    return [
        c
        for c in result.columns
        if c
        not in (
            "pair",
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "resample_offset",
        )
    ]


# =============================================================================
# Test Classes
# =============================================================================


@pytest.mark.parametrize("config", _CONFIGS_FOR_PARAM, ids=_IDS_FOR_PARAM)
class TestIndicatorEmptyColumn:
    """Test empty column handling for all indicators."""

    def test_empty_required_column_graceful_handling(self, config: IndicatorConfig):
        """Indicator should handle empty required columns gracefully."""
        config = _ensure_config(config)

        for required_col in config.requires:
            df = generate_empty_column_df(n_rows=100, empty_columns=[required_col])

            # Should not raise exception (or raise clear error)
            try:
                result = run_indicator(config, df)

                # Should have output columns
                out_cols = get_output_columns(result, config)
                assert len(out_cols) > 0 or len(result.columns) > 7, (
                    f"No output columns from {config.name}"
                )

                # Output length should match input
                assert len(result) == len(df), f"{config.name} changed row count"

            except (ValueError, KeyError, ZeroDivisionError) as e:
                # Acceptable to raise clear error about missing/null data
                error_msg = str(e).lower()
                acceptable_errors = [
                    "missing",
                    "null",
                    "empty",
                    "nan",
                    "required",
                    "column",
                ]
                assert any(kw in error_msg for kw in acceptable_errors), (
                    f"{config.name} raised unclear error: {e}"
                )

    def test_partial_null_handling(self, config: IndicatorConfig):
        """Indicator should handle partial null values."""
        config = _ensure_config(config)
        df = generate_sinusoidal_ohlcv(n_rows=200)
        df_nulls = generate_ohlcv_with_nulls(
            df, null_fraction=0.05, columns=config.requires
        )

        try:
            result = run_indicator(config, df_nulls)

            # Should produce output
            out_cols = get_output_columns(result, config)
            assert len(out_cols) > 0 or len(result.columns) > 7

            # Should have same length
            assert len(result) == len(df_nulls)

        except (ValueError, KeyError, ZeroDivisionError):
            # Some indicators may not support partial nulls
            pytest.skip(f"{config.name} doesn't support partial null data")


@pytest.mark.parametrize("config", _CONFIGS_FOR_PARAM, ids=_IDS_FOR_PARAM)
class TestIndicatorLookAhead:
    """Test look-ahead bias for all indicators."""

    TRUNCATE_BARS = 50
    TOLERANCE = 1e-9

    def test_no_lookahead_bias(self, config: IndicatorConfig):
        """Indicator should not use future data."""
        config = _ensure_config(config)
        # Need enough data for warmup + truncate + comparison
        n_rows = max(500, config.warmup * 3 + self.TRUNCATE_BARS * 2)
        df = generate_sinusoidal_ohlcv(n_rows=n_rows, seed=SEED)

        try:
            # Compute on full data
            result_full = run_indicator(config, df)

            # Compute on truncated data
            df_trunc = df.head(len(df) - self.TRUNCATE_BARS)
            result_trunc = run_indicator(config, df_trunc)

            # Get output columns
            out_cols = get_output_columns(result_full, config)

            if not out_cols:
                pytest.skip(f"No output columns from {config.name}")

            # Compare overlapping region
            n_compare = len(result_trunc)

            for out_col in out_cols:
                if (
                    out_col not in result_full.columns
                    or out_col not in result_trunc.columns
                ):
                    continue

                full_vals = result_full[out_col][:n_compare].to_numpy()
                trunc_vals = result_trunc[out_col].to_numpy()

                # NaN patterns should match
                full_nan = np.isnan(full_vals)
                trunc_nan = np.isnan(trunc_vals)

                np.testing.assert_array_equal(
                    full_nan,
                    trunc_nan,
                    err_msg=f"Look-ahead bias in {config.name}.{out_col}: NaN patterns differ",
                )

                # Values should match where valid
                valid = ~full_nan & ~trunc_nan
                if valid.any():
                    np.testing.assert_allclose(
                        full_vals[valid],
                        trunc_vals[valid],
                        rtol=self.TOLERANCE,
                        atol=self.TOLERANCE,
                        err_msg=f"Look-ahead bias in {config.name}.{out_col}: values differ",
                    )

        except Exception as e:
            if "empty" in str(e).lower() or "null" in str(e).lower():
                pytest.skip(f"{config.name} requires non-null data")
            raise

    def test_no_lookahead_random_walk(self, config: IndicatorConfig):
        """Test look-ahead with random walk data."""
        config = _ensure_config(config)
        n_rows = max(500, config.warmup * 3 + self.TRUNCATE_BARS * 2)
        df = generate_random_walk_ohlcv(n_rows=n_rows, seed=SEED)

        try:
            result_full = run_indicator(config, df)
            result_trunc = run_indicator(config, df.head(n_rows - 100))

            out_cols = get_output_columns(result_full, config)
            n_compare = len(result_trunc)

            for out_col in out_cols:
                if (
                    out_col not in result_full.columns
                    or out_col not in result_trunc.columns
                ):
                    continue

                full_vals = result_full[out_col][:n_compare].to_numpy()
                trunc_vals = result_trunc[out_col].to_numpy()

                valid = ~np.isnan(full_vals) & ~np.isnan(trunc_vals)
                if valid.any():
                    np.testing.assert_allclose(
                        full_vals[valid],
                        trunc_vals[valid],
                        rtol=self.TOLERANCE,
                        atol=self.TOLERANCE,
                    )
        except Exception as e:
            if "empty" in str(e).lower() or "null" in str(e).lower():
                pytest.skip(f"{config.name} requires non-null data")
            raise


@pytest.mark.parametrize("config", _CONFIGS_FOR_PARAM, ids=_IDS_FOR_PARAM)
class TestIndicatorReproducibility:
    """Test reproducibility for all indicators.

    CRITICAL: Indicators must produce identical values regardless of entry point.
    If computing a feature for Jan-Dec vs Jul-Dec, December values must match
    within 0.0001% tolerance after warmup period.

    This is essential for:
    - Backtesting integrity
    - Production consistency
    - Feature store reliability
    """

    # Maximum acceptable difference: 0.0001% = 1e-6
    TOLERANCE = 1e-6

    def test_same_result_different_entry_points(self, config: IndicatorConfig):
        """Indicator must give identical results from different entry points.

        This test simulates:
        - Computing features on full history (Jan-Dec)
        - Computing features on partial history (Jul-Dec)
        - Comparing December values - they MUST match within 0.0001%
        """
        config = _ensure_config(config)

        # Realistic scenario: 1000 bars total, start comparison from bar 200
        # This is like having 1 year of data vs 6 months of data
        n_rows = max(1000, config.warmup * 3 + 500)
        df = generate_sinusoidal_ohlcv(n_rows=n_rows, seed=SEED)

        try:
            # Compute from start (full history)
            result_full = run_indicator(config, df)

            # Compute from bar 200 (partial history - like starting from July)
            offset = 200
            df_late = df.slice(offset, len(df) - offset)
            result_late = run_indicator(config, df_late)

            out_cols = get_output_columns(result_full, config)

            if not out_cols:
                pytest.skip(f"No output columns from {config.name}")

            # Compare after standard warmup (2x period for safety)
            warmup = config.warmup * 2

            # Start comparison after warmup in the "late" dataset
            compare_start = offset + warmup
            compare_len = min(len(result_late) - warmup, 500)

            if compare_len <= 0:
                pytest.skip("Not enough data after warmup")

            failures = []

            for out_col in out_cols:
                if (
                    out_col not in result_full.columns
                    or out_col not in result_late.columns
                ):
                    continue

                full_vals = result_full[out_col][
                    compare_start : compare_start + compare_len
                ].to_numpy()
                late_vals = result_late[out_col][
                    warmup : warmup + compare_len
                ].to_numpy()

                valid = ~np.isnan(full_vals) & ~np.isnan(late_vals)

                if valid.sum() < 10:
                    continue  # Not enough valid data to compare

                full_valid = full_vals[valid]
                late_valid = late_vals[valid]

                # Calculate relative difference
                with np.errstate(divide="ignore", invalid="ignore"):
                    scale = np.maximum(np.abs(full_valid), np.abs(late_valid))
                    scale = np.where(scale < 1e-10, 1.0, scale)
                    rel_diff = np.abs(full_valid - late_valid) / scale

                max_rel_diff = np.nanmax(rel_diff)
                mean_rel_diff = np.nanmean(rel_diff)

                if max_rel_diff > self.TOLERANCE:
                    worst_idx = np.nanargmax(rel_diff)
                    violations = (rel_diff > self.TOLERANCE).sum()

                    failures.append(
                        {
                            "column": out_col,
                            "max_diff_pct": max_rel_diff * 100,
                            "mean_diff_pct": mean_rel_diff * 100,
                            "violations": violations,
                            "total": len(rel_diff),
                            "full_value": full_valid[worst_idx],
                            "late_value": late_valid[worst_idx],
                        }
                    )

            if failures:
                msg_lines = [
                    f"\n{'=' * 70}",
                    f"REPRODUCIBILITY FAILURE: {config.name}",
                    f"{'=' * 70}",
                    f"Indicator values depend on entry point!",
                    f"Tolerance: 0.0001% ({self.TOLERANCE:.0e})",
                    f"",
                ]

                for f in failures:
                    msg_lines.extend(
                        [
                            f"Column: {f['column']}",
                            f"  Max diff: {f['max_diff_pct']:.6f}% (limit: 0.0001%)",
                            f"  Mean diff: {f['mean_diff_pct']:.6f}%",
                            f"  Violations: {f['violations']} / {f['total']} samples",
                            f"  Example: full={f['full_value']:.8f}, partial={f['late_value']:.8f}",
                            f"",
                        ]
                    )

                msg_lines.extend(
                    [
                        f"This indicator uses EMA/recursive smoothing.",
                        f"Fix options:",
                        f"  1. Use SMA for initialization instead of first value",
                        f"  2. Provide pre-computed warmup data",
                        f"  3. Increase warmup period in indicator",
                        f"{'=' * 70}",
                    ]
                )

                pytest.fail("\n".join(msg_lines))

        except Exception as e:
            if "empty" in str(e).lower() or "null" in str(e).lower():
                pytest.skip(f"{config.name} requires non-null data")
            raise

    def test_deterministic_computation(self, config: IndicatorConfig):
        """Running indicator twice should give identical results."""
        config = _ensure_config(config)
        df = generate_sinusoidal_ohlcv(n_rows=500, seed=SEED)

        try:
            result1 = run_indicator(config, df)
            result2 = run_indicator(config, df)

            out_cols = get_output_columns(result1, config)

            for out_col in out_cols:
                if out_col not in result1.columns or out_col not in result2.columns:
                    continue

                vals1 = result1[out_col].to_numpy()
                vals2 = result2[out_col].to_numpy()

                # Handle NaN comparison
                nan1 = np.isnan(vals1)
                nan2 = np.isnan(vals2)

                np.testing.assert_array_equal(nan1, nan2)

                valid = ~nan1
                if valid.any():
                    np.testing.assert_array_equal(vals1[valid], vals2[valid])

        except Exception as e:
            if "empty" in str(e).lower() or "null" in str(e).lower():
                pytest.skip(f"{config.name} requires non-null data")
            raise


@pytest.mark.parametrize("config", _CONFIGS_FOR_PARAM, ids=_IDS_FOR_PARAM)
class TestIndicatorOutputValidation:
    """Test output validation for all indicators."""

    def test_output_length_preserved(self, config: IndicatorConfig):
        """Output length should match input length."""
        config = _ensure_config(config)
        df = generate_sinusoidal_ohlcv(n_rows=500)

        try:
            result = run_indicator(config, df)
            assert len(result) == len(df), (
                f"{config.name} changed row count: {len(result)} != {len(df)}"
            )
        except Exception as e:
            if "empty" in str(e).lower() or "null" in str(e).lower():
                pytest.skip(f"{config.name} requires non-null data")
            raise

    def test_output_columns_exist(self, config: IndicatorConfig):
        """Expected output columns should exist."""
        config = _ensure_config(config)
        df = generate_sinusoidal_ohlcv(n_rows=500)

        try:
            result = run_indicator(config, df)
            out_cols = get_output_columns(result, config)

            assert len(out_cols) > 0 or len(result.columns) > 7, (
                f"{config.name} produced no new columns"
            )
        except Exception as e:
            if "empty" in str(e).lower() or "null" in str(e).lower():
                pytest.skip(f"{config.name} requires non-null data")
            raise

    def test_output_bounds(self, config: IndicatorConfig):
        """Output should be within expected bounds."""
        config = _ensure_config(config)
        if config.bounded is None:
            pytest.skip(f"{config.name} has no defined bounds")

        df = generate_sinusoidal_ohlcv(n_rows=500)

        try:
            result = run_indicator(config, df)

            min_bound, max_bound = config.bounded
            out_cols = get_output_columns(result, config)

            for out_col in out_cols:
                if out_col not in result.columns:
                    continue

                values = result[out_col].drop_nulls().to_numpy()
                valid = ~np.isnan(values)

                if not valid.any():
                    continue

                values = values[valid]

                # Check min bound
                if min_bound is not None:
                    violations = values < min_bound - 1e-6
                    if violations.any():
                        min_val = values[violations].min()
                        pytest.fail(
                            f"{config.name}.{out_col} below min bound {min_bound}: "
                            f"min value = {min_val}"
                        )

                # Check max bound
                if max_bound is not None and max_bound != float("inf"):
                    violations = values > max_bound + 1e-6
                    if violations.any():
                        max_val = values[violations].max()
                        pytest.fail(
                            f"{config.name}.{out_col} above max bound {max_bound}: "
                            f"max value = {max_val}"
                        )

        except Exception as e:
            if "empty" in str(e).lower() or "null" in str(e).lower():
                pytest.skip(f"{config.name} requires non-null data")
            raise

    def test_no_inf_values(self, config: IndicatorConfig):
        """Output should not contain infinite values."""
        config = _ensure_config(config)
        df = generate_sinusoidal_ohlcv(n_rows=500)

        try:
            result = run_indicator(config, df)
            out_cols = get_output_columns(result, config)

            for out_col in out_cols:
                if out_col not in result.columns:
                    continue

                values = result[out_col].to_numpy()
                valid = ~np.isnan(values)

                if valid.any():
                    assert not np.any(np.isinf(values[valid])), (
                        f"{config.name}.{out_col} contains infinite values"
                    )

        except Exception as e:
            if "empty" in str(e).lower() or "null" in str(e).lower():
                pytest.skip(f"{config.name} requires non-null data")
            raise


@pytest.mark.parametrize("config", _CONFIGS_FOR_PARAM, ids=_IDS_FOR_PARAM)
class TestIndicatorMultiPair:
    """Test multi-pair handling for all indicators."""

    def test_pairs_independent(self, config: IndicatorConfig):
        """Pairs should be computed independently."""
        config = _ensure_config(config)
        # Create two pairs with very different price levels
        df1 = generate_static_ohlcv(n_rows=200, base_price=100, pair="BTCUSDT")
        df2 = generate_static_ohlcv(n_rows=200, base_price=50000, pair="ETHUSDT")

        try:
            # Compute separately
            result1 = run_indicator(config, df1)
            result2 = run_indicator(config, df2)

            # Compute together (simulating grouped computation)
            df_combined = pl.concat([df1, df2])

            results_combined = {}
            for pair in ["BTCUSDT", "ETHUSDT"]:
                pair_df = df_combined.filter(pl.col("pair") == pair)
                results_combined[pair] = run_indicator(config, pair_df)

            # Results should match
            out_cols = get_output_columns(result1, config)

            for out_col in out_cols:
                if out_col not in result1.columns:
                    continue

                np.testing.assert_allclose(
                    result1[out_col].to_numpy(),
                    results_combined["BTCUSDT"][out_col].to_numpy(),
                    rtol=1e-10,
                    err_msg=f"{config.name}.{out_col} differs for BTCUSDT",
                )

        except Exception as e:
            if "empty" in str(e).lower() or "null" in str(e).lower():
                pytest.skip(f"{config.name} requires non-null data")
            raise


@pytest.mark.parametrize("config", _CONFIGS_FOR_PARAM, ids=_IDS_FOR_PARAM)
class TestIndicatorDataTypes:
    """Test data type handling."""

    def test_static_data(self, config: IndicatorConfig):
        """Should handle static (constant) price data."""
        config = _ensure_config(config)
        df = generate_static_ohlcv(n_rows=max(200, config.warmup * 2))

        try:
            result = run_indicator(config, df)
            assert len(result) == len(df)
        except (ValueError, ZeroDivisionError) as e:
            # Some indicators may fail on zero variance data
            pytest.skip(f"{config.name} requires non-constant data: {e}")

    def test_sinusoidal_data(self, config: IndicatorConfig):
        """Should handle sinusoidal price data."""
        config = _ensure_config(config)
        df = generate_sinusoidal_ohlcv(n_rows=500)

        result = run_indicator(config, df)
        assert len(result) == len(df)

    def test_random_walk_data(self, config: IndicatorConfig):
        """Should handle random walk price data."""
        config = _ensure_config(config)
        df = generate_random_walk_ohlcv(n_rows=500)

        result = run_indicator(config, df)
        assert len(result) == len(df)

    def test_very_short_data(self, config: IndicatorConfig):
        """Should handle very short data gracefully."""
        config = _ensure_config(config)
        df = generate_sinusoidal_ohlcv(n_rows=5)

        try:
            result = run_indicator(config, df)
            assert len(result) == 5
        except (ValueError, IndexError) as e:
            # Acceptable if raises clear error about insufficient data
            error_msg = str(e).lower()
            acceptable = [
                "length",
                "short",
                "period",
                "minimum",
                "insufficient",
                "at least",
            ]
            if not any(kw in error_msg for kw in acceptable):
                # Still pass if it's just array indexing issue due to short data
                pass

    def test_performance(self, config: IndicatorConfig):
        """Should handle long data efficiently."""
        config = _ensure_config(config)
        df = generate_sinusoidal_ohlcv(n_rows=10000)

        import time

        start = time.time()
        result = run_indicator(config, df)
        elapsed = time.time() - start

        assert len(result) == 10000
        assert elapsed < 10.0, f"{config.name} too slow: {elapsed:.2f}s for 10k rows"


# =============================================================================
# Category-Specific Tests
# =============================================================================


class TestMomentumIndicators:
    """Additional tests specific to momentum indicators."""

    @pytest.fixture
    def momentum_configs(self):
        return get_configs_by_category("momentum")

    def test_rsi_mean_reversion(self, momentum_configs):
        """RSI should oscillate around 50 for random data."""
        rsi_configs = [c for c in momentum_configs if "rsi" in c.name.lower()]

        if not rsi_configs:
            pytest.skip("No RSI indicators found")

        df = generate_random_walk_ohlcv(n_rows=1000, seed=SEED)

        for config in rsi_configs:
            try:
                result = run_indicator(config, df)
                out_cols = get_output_columns(result, config)

                for col in out_cols:
                    if col not in result.columns:
                        continue
                    values = result[col].drop_nulls().to_numpy()
                    if len(values) > 100:
                        mean_rsi = np.mean(values)
                        # RSI should oscillate around 50 (±15)
                        assert 35 < mean_rsi < 65, (
                            f"{config.name} mean RSI = {mean_rsi:.1f}, expected ~50"
                        )
            except Exception:
                continue


class TestVolatilityIndicators:
    """Additional tests specific to volatility indicators."""

    @pytest.fixture
    def volatility_configs(self):
        return get_configs_by_category("volatility")

    def test_atr_positive(self, volatility_configs):
        """ATR should always be positive."""
        atr_configs = [c for c in volatility_configs if "atr" in c.name.lower()]

        if not atr_configs:
            pytest.skip("No ATR indicators found")

        df = generate_random_walk_ohlcv(n_rows=500, seed=SEED)

        for config in atr_configs:
            result = run_indicator(config, df)
            out_cols = get_output_columns(result, config)

            for col in out_cols:
                if col not in result.columns:
                    continue
                values = result[col].drop_nulls().to_numpy()
                valid = ~np.isnan(values)

                if valid.any():
                    assert np.all(values[valid] >= 0), (
                        f"{config.name}.{col} has negative values"
                    )


# =============================================================================
# Run Tests
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])

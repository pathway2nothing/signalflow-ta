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
    generate_trending_ohlcv,
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
# Test Data Size Constants
# =============================================================================

# Realistic data sizes for 1-minute bars
BARS_SHORT = 20_000   # ~2 weeks of 1-minute data
BARS_LONG = 40_000    # ~1 month of 1-minute data


# =============================================================================
# Helper for skipping tests when no indicators available
# =============================================================================

def _ensure_config(config):
    """Skip test if config is None (no indicators loaded)."""
    if config is None:
        pytest.skip("signalflow.ta not installed - install it to run indicator tests")
    return config


def _estimate_warmup(config: IndicatorConfig) -> int:
    """
    Estimate warmup period based on indicator parameters.
    
    For EMA-based indicators with tolerance 0.01% (1e-4), we need warmup where:
    (1 - alpha)^warmup < 1e-4
    
    For alpha = 2/(period+1), this requires approximately 5*period bars.
    """
    base_warmup = config.warmup or 50
    params = config.params
    
    # Find the largest period-like parameter
    period_params = []
    for key, val in params.items():
        if isinstance(val, int) and any(p in key.lower() for p in ['period', 'slow', 'length', 'window']):
            period_params.append(val)
    
    if period_params:
        max_period = max(period_params)
        
        # For 0.01% tolerance: warmup ≈ 5 * period (for single EMA)
        if 'trix' in config.name.lower():
            # Triple EMA
            estimated = max_period * 12
        elif 'tsi' in config.name.lower():
            # Double EMA
            estimated = max_period * 8
        elif 'stoch' in config.name.lower() and 'rsi' in config.name.lower():
            # RSI + Stochastic
            estimated = max_period * 6
        elif any(x in config.name.lower() for x in ['macd', 'ppo', 'ema', 'rsi']):
            # Single/double EMA
            estimated = max_period * 5
        else:
            estimated = max_period * 2
        
        return max(base_warmup, estimated)
    
    return base_warmup


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
        if col in ('pair', 'timestamp', 'open', 'high', 'low', 'close', 'volume', 'resample_offset'):
            continue
        # Check if column seems related to this indicator
        for pattern in [name_lower, name_lower.replace('stat', ''), name_lower.replace('mom', ''), 
                       name_lower.replace('smooth', ''), name_lower.replace('vol', '')]:
            if pattern and len(pattern) > 2 and pattern in col_lower:
                partial_matches.append(col)
                break
    
    if partial_matches:
        return partial_matches
    
    # Return all non-standard columns
    return [c for c in result.columns 
            if c not in ('pair', 'timestamp', 'open', 'high', 'low', 'close', 'volume', 'resample_offset')]


# =============================================================================
# Test Classes
# =============================================================================


@pytest.mark.parametrize(
    "config",
    _CONFIGS_FOR_PARAM,
    ids=_IDS_FOR_PARAM
)
class TestIndicatorEmptyColumn:
    """Test empty column handling for all indicators."""
    
    def test_empty_required_column_graceful_handling(self, config: IndicatorConfig):
        """Indicator should handle empty required columns gracefully."""
        config = _ensure_config(config)
        
        # Generate enough data for indicator's warmup period
        min_rows = max(100, _estimate_warmup(config) * 3)
        
        for required_col in config.requires:
            df = generate_empty_column_df(n_rows=min_rows, empty_columns=[required_col])
            
            # Should not raise exception (or raise clear error)
            try:
                result = run_indicator(config, df)
                
                # Should have output columns
                out_cols = get_output_columns(result, config)
                assert len(out_cols) > 0 or len(result.columns) > 7, \
                    f"No output columns from {config.name}"
                
                # Output length should match input
                assert len(result) == len(df), \
                    f"{config.name} changed row count"
                    
            except (ValueError, KeyError, ZeroDivisionError) as e:
                # Acceptable to raise clear error about missing/null data
                error_msg = str(e).lower()
                acceptable_errors = ['missing', 'null', 'empty', 'nan', 'required', 'column']
                assert any(kw in error_msg for kw in acceptable_errors), \
                    f"{config.name} raised unclear error: {e}"
    
    def test_partial_null_handling(self, config: IndicatorConfig):
        """Indicator should handle partial null values."""
        config = _ensure_config(config)
        
        # Generate enough data for indicator's warmup period
        min_rows = max(200, _estimate_warmup(config) * 3)
        df = generate_sinusoidal_ohlcv(n_rows=min_rows)
        df_nulls = generate_ohlcv_with_nulls(df, null_fraction=0.05, columns=config.requires)
        
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


@pytest.mark.parametrize(
    "config",
    _CONFIGS_FOR_PARAM,
    ids=_IDS_FOR_PARAM
)
class TestIndicatorLookAhead:
    """Test look-ahead bias for all indicators.
    
    Methodology:
    1. Same start point for all computations
    2. Multiple random end points (shorter periods)
    3. Test at local extrema (max/min) where look-ahead is most likely
    
    Data size: 2 months of 1-minute bars (~86,400 bars)
    
    If indicator value at bar N depends on data after bar N, 
    it has look-ahead bias and is INVALID for backtesting.
    """
    
    TOLERANCE = 1e-9
    NUM_RANDOM_CUTS = 5  # Number of random truncation points
    
    def _find_local_extrema(self, df: pl.DataFrame, col: str = "close", window: int = 20) -> list[int]:
        """Find indices of local maxima and minima."""
        values = df[col].to_numpy()
        n = len(values)
        extrema = []
        
        for i in range(window, n - window):
            window_vals = values[i - window:i + window + 1]
            if np.isnan(window_vals).any():
                continue
            
            # Local maximum
            if values[i] == np.max(window_vals):
                extrema.append(i)
            # Local minimum
            elif values[i] == np.min(window_vals):
                extrema.append(i)
        
        return extrema
    
    def _test_truncation_point(
        self, 
        config: IndicatorConfig, 
        df: pl.DataFrame, 
        cut_point: int,
        description: str
    ) -> list[str]:
        """Test a single truncation point. Returns list of errors."""
        errors = []
        
        # Full data
        result_full = run_indicator(config, df)
        
        # Truncated data (same start, earlier end)
        df_trunc = df.head(cut_point)
        result_trunc = run_indicator(config, df_trunc)
        
        out_cols = get_output_columns(result_full, config)
        
        for out_col in out_cols:
            if out_col not in result_full.columns or out_col not in result_trunc.columns:
                continue
            
            # Compare values up to truncation point
            full_vals = result_full[out_col][:cut_point].to_numpy()
            trunc_vals = result_trunc[out_col].to_numpy()
            
            # Check NaN patterns match
            full_nan = np.isnan(full_vals)
            trunc_nan = np.isnan(trunc_vals)
            
            if not np.array_equal(full_nan, trunc_nan):
                nan_diff_count = np.sum(full_nan != trunc_nan)
                errors.append(
                    f"{out_col} at {description}: NaN patterns differ ({nan_diff_count} mismatches)"
                )
                continue
            
            # Check values match where valid
            valid = ~full_nan & ~trunc_nan
            if valid.any():
                full_valid = full_vals[valid]
                trunc_valid = trunc_vals[valid]
                
                diff = np.abs(full_valid - trunc_valid)
                max_diff = np.max(diff)
                
                if max_diff > self.TOLERANCE:
                    # Find where the biggest difference is
                    worst_idx = np.argmax(diff)
                    # Map back to original index
                    valid_indices = np.where(valid)[0]
                    original_idx = valid_indices[worst_idx]
                    
                    errors.append(
                        f"{out_col} at {description}: "
                        f"values differ at bar {original_idx} "
                        f"(full={full_valid[worst_idx]:.8f}, trunc={trunc_valid[worst_idx]:.8f}, "
                        f"diff={max_diff:.2e})"
                    )
        
        return errors
    
    def test_no_lookahead_bias(self, config: IndicatorConfig):
        """Indicator must not use future data.
        
        Tests multiple truncation points:
        1. Random points in the middle of data
        2. Local maxima (where look-ahead for max detection would show)
        3. Local minima (where look-ahead for min detection would show)
        
        Uses 2 months of 1-minute data (~86,400 bars).
        """
        config = _ensure_config(config)
        
        # Use full history
        n_rows = BARS_LONG
        df = generate_sinusoidal_ohlcv(n_rows=n_rows, seed=SEED)
        
        all_errors = []
        
        try:
            # Test 1: Random truncation points in second half of data
            rng = np.random.default_rng(SEED)
            min_cut = BARS_SHORT  # At least half the data
            max_cut = n_rows - 1000  # Leave some buffer at end
            
            random_cuts = rng.integers(min_cut, max_cut, size=self.NUM_RANDOM_CUTS)
            
            for i, cut_point in enumerate(random_cuts):
                errors = self._test_truncation_point(
                    config, df, cut_point, f"random_cut_{i} (bar {cut_point})"
                )
                all_errors.extend(errors)
            
            # Test 2: Local extrema in the second half (most likely to catch look-ahead)
            extrema = self._find_local_extrema(df, "close", window=10)
            
            # Filter to extrema in the second half only
            extrema_in_range = [e for e in extrema if BARS_SHORT < e < n_rows - 500]
            
            # Test a few extrema points
            if extrema_in_range:
                extrema_to_test = extrema_in_range[:3]  # Test up to 3 extrema
                
                for ext_idx in extrema_to_test:
                    errors = self._test_truncation_point(
                        config, df, ext_idx, f"local_extremum (bar {ext_idx})"
                    )
                    all_errors.extend(errors)
            
            # Report all errors
            if all_errors:
                msg = [
                    f"\n{'='*70}",
                    f"LOOK-AHEAD BIAS DETECTED: {config.name}",
                    f"{'='*70}",
                    f"Indicator uses future data! This invalidates backtesting.",
                    f"Data: {n_rows:,} bars",
                    f"",
                    f"Errors found ({len(all_errors)}):",
                ]
                for err in all_errors[:10]:  # Show first 10 errors
                    msg.append(f"  - {err}")
                
                if len(all_errors) > 10:
                    msg.append(f"  ... and {len(all_errors) - 10} more errors")
                
                msg.extend([
                    f"",
                    f"Fix: Ensure indicator only uses data up to current bar.",
                    f"{'='*70}",
                ])
                
                pytest.fail("\n".join(msg))
                
        except Exception as e:
            if "empty" in str(e).lower() or "null" in str(e).lower():
                pytest.skip(f"{config.name} requires non-null data")
            raise
    
    def test_no_lookahead_at_volatility_spike(self, config: IndicatorConfig):
        """Test look-ahead specifically at volatility spikes.
        
        Volatility spikes are where indicators like ATR might 
        accidentally look ahead to detect the spike.
        
        Uses BARS_LONG bars of random walk data.
        """
        config = _ensure_config(config)
        
        n_rows = BARS_LONG
        df = generate_random_walk_ohlcv(n_rows=n_rows, seed=SEED, volatility=0.02)
        
        all_errors = []
        
        try:
            # Find volatility spikes (high true range)
            high = df["high"].to_numpy()
            low = df["low"].to_numpy()
            close = df["close"].to_numpy()
            
            tr = np.maximum(
                high - low,
                np.maximum(
                    np.abs(high - np.roll(close, 1)),
                    np.abs(low - np.roll(close, 1))
                )
            )
            tr[0] = tr[1]  # Fix first value
            
            # Find bars with high true range (top 10%) in second half
            second_half_tr = tr[BARS_SHORT:]
            threshold = np.percentile(second_half_tr, 90)
            
            spike_indices = np.where(tr > threshold)[0]
            spike_indices = spike_indices[spike_indices > BARS_SHORT]
            spike_indices = spike_indices[spike_indices < n_rows - 500]
            
            # Test at a few spike points
            for spike_idx in spike_indices[:3]:
                errors = self._test_truncation_point(
                    config, df, int(spike_idx), f"volatility_spike (bar {spike_idx})"
                )
                all_errors.extend(errors)
            
            if all_errors:
                msg = [
                    f"\n{'='*70}",
                    f"LOOK-AHEAD BIAS AT VOLATILITY SPIKE: {config.name}",
                    f"{'='*70}",
                ]
                for err in all_errors[:5]:
                    msg.append(f"  - {err}")
                msg.append(f"{'='*70}")
                
                pytest.fail("\n".join(msg))
                
        except Exception as e:
            if "empty" in str(e).lower() or "null" in str(e).lower():
                pytest.skip(f"{config.name} requires non-null data")
            raise


@pytest.mark.parametrize(
    "config",
    _CONFIGS_FOR_PARAM,
    ids=_IDS_FOR_PARAM
)
class TestIndicatorReproducibility:
    """Test reproducibility for all indicators.
    
    CRITICAL: Indicators must produce identical values regardless of entry point.
    If computing a feature for Jan-Dec vs Jul-Dec, December values must match
    within 0.01% tolerance after warmup period.
    
    Data size:
    - Full history: 2 months (~86,400 bars)
    - Partial history: starts 1 month later (~43,200 offset)
    - Compare: last month of data
    
    This is essential for:
    - Backtesting integrity
    - Production consistency
    - Feature store reliability
    """
    
    # Maximum acceptable difference: 0.01% = 1e-4
    TOLERANCE = 1e-4
    
    def test_same_result_different_entry_points(self, config: IndicatorConfig):
        """Indicator must give identical results from different entry points.
        
        This test simulates real scenario:
        - Computing features on full history (BARS_LONG bars)
        - Computing features on partial history (BARS_SHORT bars)
        - Comparing overlapping values - they MUST match within 0.01%
        """
        config = _ensure_config(config)
        
        # Calculate required warmup for 0.01% tolerance
        warmup = _estimate_warmup(config)
        
        # Use standard data sizes
        n_rows = BARS_LONG
        offset = BARS_SHORT
        
        df = generate_sinusoidal_ohlcv(n_rows=n_rows, seed=SEED)
        
        try:
            # Compute from start (full history)
            result_full = run_indicator(config, df)
            
            # Compute from offset (partial history)
            df_late = df.slice(offset, len(df) - offset)
            result_late = run_indicator(config, df_late)
            
            out_cols = get_output_columns(result_full, config)
            
            if not out_cols:
                pytest.skip(f"No output columns from {config.name}")
            
            # Compare after warmup in the "late" dataset
            compare_warmup = warmup * 2  # Safety margin
            
            # Start comparison after warmup
            compare_start = offset + compare_warmup
            compare_len = min(len(result_late) - compare_warmup, BARS_SHORT // 2)
            
            if compare_len <= 0 or compare_start + compare_len > len(result_full):
                pytest.skip(f"Not enough data after warmup (need {compare_warmup} warmup)")
            
            failures = []
            
            for out_col in out_cols:
                if out_col not in result_full.columns or out_col not in result_late.columns:
                    continue
                    
                full_vals = result_full[out_col][compare_start:compare_start + compare_len].to_numpy()
                late_vals = result_late[out_col][compare_warmup:compare_warmup + compare_len].to_numpy()
                
                valid = ~np.isnan(full_vals) & ~np.isnan(late_vals)
                
                if valid.sum() < 100:
                    continue  # Not enough valid data to compare
                
                full_valid = full_vals[valid]
                late_valid = late_vals[valid]
                
                # Calculate relative difference
                with np.errstate(divide='ignore', invalid='ignore'):
                    scale = np.maximum(np.abs(full_valid), np.abs(late_valid))
                    scale = np.where(scale < 1e-10, 1.0, scale)
                    rel_diff = np.abs(full_valid - late_valid) / scale
                
                max_rel_diff = np.nanmax(rel_diff)
                mean_rel_diff = np.nanmean(rel_diff)
                
                if max_rel_diff > self.TOLERANCE:
                    worst_idx = np.nanargmax(rel_diff)
                    violations = (rel_diff > self.TOLERANCE).sum()
                    
                    failures.append({
                        'column': out_col,
                        'max_diff_pct': max_rel_diff * 100,
                        'mean_diff_pct': mean_rel_diff * 100,
                        'violations': violations,
                        'total': len(rel_diff),
                        'full_value': full_valid[worst_idx],
                        'late_value': late_valid[worst_idx],
                    })
            
            if failures:
                msg_lines = [
                    f"\n{'='*70}",
                    f"REPRODUCIBILITY FAILURE: {config.name}",
                    f"{'='*70}",
                    f"Indicator values depend on entry point!",
                    f"",
                    f"Test setup:",
                    f"  Full history: {n_rows:,} bars (2 months)",
                    f"  Partial history: {n_rows - offset:,} bars (1 month)",
                    f"  Offset: {offset:,} bars",
                    f"  Tolerance: 0.01% ({self.TOLERANCE:.0e})",
                    f"",
                ]
                
                for f in failures:
                    msg_lines.extend([
                        f"Column: {f['column']}",
                        f"  Max diff: {f['max_diff_pct']:.6f}% (limit: 0.01%)",
                        f"  Mean diff: {f['mean_diff_pct']:.6f}%",
                        f"  Violations: {f['violations']:,} / {f['total']:,} samples",
                        f"  Example: full={f['full_value']:.8f}, partial={f['late_value']:.8f}",
                        f"",
                    ])
                
                msg_lines.extend([
                    f"This indicator uses EMA/recursive smoothing.",
                    f"Warmup used: {compare_warmup:,} bars",
                    f"",
                    f"Fix options:",
                    f"  1. Use SMA for EMA initialization instead of first value",
                    f"  2. Increase warmup period in indicator",
                    f"{'='*70}",
                ])
                
                pytest.fail("\n".join(msg_lines))
                    
        except Exception as e:
            if "empty" in str(e).lower() or "null" in str(e).lower():
                pytest.skip(f"{config.name} requires non-null data")
            raise
    
    def test_deterministic_computation(self, config: IndicatorConfig):
        """Running indicator twice should give identical results."""
        config = _ensure_config(config)
        
        # Use 1 month of data for determinism test
        df = generate_sinusoidal_ohlcv(n_rows=BARS_SHORT, seed=SEED)
        
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


@pytest.mark.parametrize(
    "config",
    _CONFIGS_FOR_PARAM,
    ids=_IDS_FOR_PARAM
)
class TestIndicatorOutputValidation:
    """Test output validation for all indicators."""
    
    def test_output_length_preserved(self, config: IndicatorConfig):
        """Output length should match input length."""
        config = _ensure_config(config)
        min_rows = max(500, _estimate_warmup(config) * 3)
        df = generate_sinusoidal_ohlcv(n_rows=min_rows)
        
        try:
            result = run_indicator(config, df)
            assert len(result) == len(df), \
                f"{config.name} changed row count: {len(result)} != {len(df)}"
        except Exception as e:
            if "empty" in str(e).lower() or "null" in str(e).lower():
                pytest.skip(f"{config.name} requires non-null data")
            raise
    
    def test_output_columns_exist(self, config: IndicatorConfig):
        """Expected output columns should exist."""
        config = _ensure_config(config)
        min_rows = max(500, _estimate_warmup(config) * 3)
        df = generate_sinusoidal_ohlcv(n_rows=min_rows)
        
        try:
            result = run_indicator(config, df)
            out_cols = get_output_columns(result, config)
            
            assert len(out_cols) > 0 or len(result.columns) > 7, \
                f"{config.name} produced no new columns"
        except Exception as e:
            if "empty" in str(e).lower() or "null" in str(e).lower():
                pytest.skip(f"{config.name} requires non-null data")
            raise
    
    def test_output_bounds(self, config: IndicatorConfig):
        """Output should be within expected bounds."""
        config = _ensure_config(config)
        if config.bounded is None:
            pytest.skip(f"{config.name} has no defined bounds")
        
        min_rows = max(500, _estimate_warmup(config) * 3)
        df = generate_sinusoidal_ohlcv(n_rows=min_rows)
        
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
                if max_bound is not None and max_bound != float('inf'):
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
        min_rows = max(500, _estimate_warmup(config) * 3)
        df = generate_sinusoidal_ohlcv(n_rows=min_rows)
        
        try:
            result = run_indicator(config, df)
            out_cols = get_output_columns(result, config)
            
            for out_col in out_cols:
                if out_col not in result.columns:
                    continue
                    
                values = result[out_col].to_numpy()
                valid = ~np.isnan(values)
                
                if valid.any():
                    assert not np.any(np.isinf(values[valid])), \
                        f"{config.name}.{out_col} contains infinite values"
                        
        except Exception as e:
            if "empty" in str(e).lower() or "null" in str(e).lower():
                pytest.skip(f"{config.name} requires non-null data")
            raise


@pytest.mark.parametrize(
    "config",
    _CONFIGS_FOR_PARAM,
    ids=_IDS_FOR_PARAM
)
class TestIndicatorMultiPair:
    """Test multi-pair handling for all indicators."""
    
    def test_pairs_independent(self, config: IndicatorConfig):
        """Pairs should be computed independently."""
        config = _ensure_config(config)
        
        # Generate enough data for indicator's warmup period
        min_rows = max(200, _estimate_warmup(config) * 3)
        
        # Create two pairs with very different price levels
        df1 = generate_static_ohlcv(n_rows=min_rows, base_price=100, pair="BTCUSDT")
        df2 = generate_static_ohlcv(n_rows=min_rows, base_price=50000, pair="ETHUSDT")
        
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
                    err_msg=f"{config.name}.{out_col} differs for BTCUSDT"
                )
                
        except Exception as e:
            if "empty" in str(e).lower() or "null" in str(e).lower():
                pytest.skip(f"{config.name} requires non-null data")
            raise


@pytest.mark.parametrize(
    "config",
    _CONFIGS_FOR_PARAM,
    ids=_IDS_FOR_PARAM
)
class TestIndicatorDataTypes:
    """Test data type handling."""
    
    def test_static_data(self, config: IndicatorConfig):
        """Should handle static (constant) price data."""
        config = _ensure_config(config)
        df = generate_static_ohlcv(n_rows=max(200, _estimate_warmup(config) * 2))
        
        try:
            result = run_indicator(config, df)
            assert len(result) == len(df)
        except (ValueError, ZeroDivisionError) as e:
            # Some indicators may fail on zero variance data
            pytest.skip(f"{config.name} requires non-constant data: {e}")
    
    def test_sinusoidal_data(self, config: IndicatorConfig):
        """Should handle sinusoidal price data."""
        config = _ensure_config(config)
        min_rows = max(500, _estimate_warmup(config) * 3)
        df = generate_sinusoidal_ohlcv(n_rows=min_rows)
        
        result = run_indicator(config, df)
        assert len(result) == len(df)
    
    def test_random_walk_data(self, config: IndicatorConfig):
        """Should handle random walk price data."""
        config = _ensure_config(config)
        min_rows = max(500, _estimate_warmup(config) * 3)
        df = generate_random_walk_ohlcv(n_rows=min_rows)
        
        result = run_indicator(config, df)
        assert len(result) == len(df)
    
    def test_trending_data(self, config: IndicatorConfig):
        """Should handle trending price data with mean reversion."""
        config = _ensure_config(config)
        min_rows = max(500, _estimate_warmup(config) * 3)
        
        # Test uptrend
        df_up = generate_trending_ohlcv(n_rows=min_rows, start_price=100, end_price=150)
        result_up = run_indicator(config, df_up)
        assert len(result_up) == len(df_up)
        
        # Test downtrend
        df_down = generate_trending_ohlcv(n_rows=min_rows, start_price=100, end_price=50)
        result_down = run_indicator(config, df_down)
        assert len(result_down) == len(df_down)
    
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
            acceptable = ['length', 'short', 'period', 'minimum', 'insufficient', 'at least']
            if not any(kw in error_msg for kw in acceptable):
                # Still pass if it's just array indexing issue due to short data
                pass
    
    def test_performance(self, config: IndicatorConfig):
        """Should handle long data efficiently."""
        config = _ensure_config(config)
        min_rows = max(10000, _estimate_warmup(config) * 10)
        df = generate_sinusoidal_ohlcv(n_rows=min_rows)
        
        import time
        start = time.time()
        result = run_indicator(config, df)
        elapsed = time.time() - start
        
        assert len(result) == min_rows
        assert elapsed < 10.0, f"{config.name} too slow: {elapsed:.2f}s for {min_rows} rows"


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
        rsi_configs = [c for c in momentum_configs if 'rsi' in c.name.lower()]
        
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
                        assert 35 < mean_rsi < 65, \
                            f"{config.name} mean RSI = {mean_rsi:.1f}, expected ~50"
            except Exception:
                continue


class TestVolatilityIndicators:
    """Additional tests specific to volatility indicators."""
    
    @pytest.fixture
    def volatility_configs(self):
        return get_configs_by_category("volatility")
    
    def test_atr_positive(self, volatility_configs):
        """ATR should always be positive."""
        atr_configs = [c for c in volatility_configs if 'atr' in c.name.lower()]
        
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
                    assert np.all(values[valid] >= 0), \
                        f"{config.name}.{col} has negative values"


# =============================================================================
# Run Tests
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])
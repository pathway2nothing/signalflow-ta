"""
SignalFlow-TA Test Fixtures

Provides single synthetic test data generator for testing technical indicators.
Uses sine wave with noise and trend for realistic price movement.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from datetime import datetime, timedelta


# =============================================================================
# Constants
# =============================================================================

SEED = 42
DEFAULT_PAIR = "BTCUSDT"
DEFAULT_ROWS = 1000


# =============================================================================
# Single Test Data Generator
# =============================================================================

def generate_test_ohlcv(
    n_rows: int,
    base_price: float = 100.0,
    amplitude: float = 10.0,
    period_bars: int = 100,
    noise_level: float = 0.02,
    trend: float = 0.0001,
    pair: str = DEFAULT_PAIR,
    seed: int = SEED,
) -> pl.DataFrame:
    """
    Generate test OHLCV data: sine wave + noise + trend.

    Single data generator for all tests - no magic coefficients or multiple patterns.

    Args:
        n_rows: Number of rows to generate
        base_price: Center price value (default $100)
        amplitude: Sine wave amplitude (default $10)
        period_bars: Bars per complete sine cycle (default 100)
        noise_level: Price noise as fraction (default 2%)
        trend: Linear trend per bar (default 0.01%)
        pair: Trading pair name
        seed: Random seed for reproducibility

    Returns:
        pl.DataFrame with columns: pair, timestamp, open, high, low, close, volume
    """
    rng = np.random.default_rng(seed)

    # Generate timestamps
    start = datetime(2024, 1, 1, 0, 0, 0)
    timestamps = [start + timedelta(minutes=i) for i in range(n_rows)]

    # Generate base price: sine + trend
    t = np.arange(n_rows)
    trend_component = base_price * trend * t
    sine_component = amplitude * np.sin(2 * np.pi * t / period_bars)
    base_wave = base_price + sine_component + trend_component

    # Add noise
    noise = rng.normal(0, base_price * noise_level, n_rows)
    close_prices = base_wave + noise

    # Generate OHLC with realistic relationships
    open_prices = np.zeros(n_rows)
    high_prices = np.zeros(n_rows)
    low_prices = np.zeros(n_rows)

    open_prices[0] = close_prices[0]
    for i in range(1, n_rows):
        # Open = previous close (continuous)
        open_prices[i] = close_prices[i - 1]

    # Generate high/low with intrabar volatility
    for i in range(n_rows):
        intrabar_range = abs(close_prices[i] - open_prices[i]) + base_price * noise_level
        high_prices[i] = max(open_prices[i], close_prices[i]) + abs(rng.normal(0, intrabar_range * 0.5))
        low_prices[i] = min(open_prices[i], close_prices[i]) - abs(rng.normal(0, intrabar_range * 0.5))

    # Ensure all prices positive
    min_price = low_prices.min()
    if min_price < 0.01:
        shift = abs(min_price) + 1
        open_prices += shift
        high_prices += shift
        low_prices += shift
        close_prices += shift

    # Generate volume (random with some correlation to price change)
    base_volume = 1000.0
    price_changes = np.abs(np.diff(close_prices, prepend=close_prices[0]))
    volume_multiplier = 1 + (price_changes / close_prices) * 5
    volumes = np.abs(rng.normal(base_volume, base_volume * 0.3, n_rows)) * volume_multiplier

    return pl.DataFrame({
        "pair": [pair] * n_rows,
        "timestamp": timestamps,
        "open": open_prices,
        "high": high_prices,
        "low": low_prices,
        "close": close_prices,
        "volume": volumes,
    })


# =============================================================================
# Pytest Fixtures
# =============================================================================

@pytest.fixture
def test_data() -> pl.DataFrame:
    """Standard test data for all tests."""
    return generate_test_ohlcv(n_rows=1000)


# =============================================================================
# Pytest Configuration Hooks
# =============================================================================

def pytest_addoption(parser):
    """Add custom command-line options for test configuration."""
    parser.addoption(
        "--max-params",
        action="store",
        type=int,
        default=None,
        help="Maximum number of parameter combinations to test per indicator (default: all)"
    )
    parser.addoption(
        "--feature-groups",
        action="store",
        default=None,
        help="Comma-separated list of feature groups to test (e.g., 'momentum,overlap'). "
             "Available: momentum, overlap, trend, volatility, volume, stat, performance, other"
    )


def pytest_configure(config):
    """Store configuration options for use in tests."""
    config.test_max_params = config.getoption("--max-params")
    config.test_feature_groups = config.getoption("--feature-groups")


def pytest_generate_tests(metafunc):
    """Dynamically generate test parameters based on command-line options."""
    if "config" in metafunc.fixturenames:
        from indicator_registry import filter_configs_by_options, INDICATOR_CONFIGS

        filtered_configs, filtered_ids = filter_configs_by_options(
            INDICATOR_CONFIGS,
            pytest_config=metafunc.config
        )

        metafunc.parametrize(
            "config",
            filtered_configs,
            ids=filtered_ids,
            indirect=False
        )

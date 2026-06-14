"""SignalFlow-TA Test Fixtures

Provides single synthetic test data generator for testing technical indicators.
Uses sine wave with noise and trend for realistic price movement.
"""


import os

import signalflow

_ta = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "signalflow"))
if _ta not in signalflow.__path__:
    signalflow.__path__.append(_ta)

collect_ignore = [
    "test_signal_features.py",
    "test_signal_features_v2.py",
    "test_signals.py",
    "test_indicators.py",
]

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest


SEED = 42
DEFAULT_PAIR = "BTCUSDT"
DEFAULT_ROWS = 1000


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
    """Generate test OHLCV data: sine wave + noise + trend.

    Single data generator for all tests - no magic coefficients or multiple patterns.
    """
    rng = np.random.default_rng(seed)

    start = datetime(2024, 1, 1, 0, 0, 0)
    timestamps = [start + timedelta(minutes=i) for i in range(n_rows)]

    t = np.arange(n_rows)
    trend_component = base_price * trend * t
    sine_component = amplitude * np.sin(2 * np.pi * t / period_bars)
    base_wave = base_price + sine_component + trend_component

    noise = rng.normal(0, base_price * noise_level, n_rows)
    close_prices = base_wave + noise

    open_prices = np.zeros(n_rows)
    high_prices = np.zeros(n_rows)
    low_prices = np.zeros(n_rows)

    open_prices[0] = close_prices[0]
    for i in range(1, n_rows):
        open_prices[i] = close_prices[i - 1]

    for i in range(n_rows):
        intrabar_range = abs(close_prices[i] - open_prices[i]) + base_price * noise_level
        high_prices[i] = max(open_prices[i], close_prices[i]) + abs(rng.normal(0, intrabar_range * 0.5))
        low_prices[i] = min(open_prices[i], close_prices[i]) - abs(rng.normal(0, intrabar_range * 0.5))

    min_price = low_prices.min()
    if min_price < 0.01:
        shift = abs(min_price) + 1
        open_prices += shift
        high_prices += shift
        low_prices += shift
        close_prices += shift

    base_volume = 1000.0
    price_changes = np.abs(np.diff(close_prices, prepend=close_prices[0]))
    volume_multiplier = 1 + (price_changes / close_prices) * 5
    volumes = np.abs(rng.normal(base_volume, base_volume * 0.3, n_rows)) * volume_multiplier

    return pl.DataFrame(
        {
            "pair": [pair] * n_rows,
            "timestamp": timestamps,
            "open": open_prices,
            "high": high_prices,
            "low": low_prices,
            "close": close_prices,
            "volume": volumes,
        }
    )


generate_sinusoidal_ohlcv = generate_test_ohlcv


def generate_static_ohlcv(
    n_rows: int,
    base_price: float = 100.0,
    pair: str = DEFAULT_PAIR,
    seed: int = SEED,
) -> pl.DataFrame:
    """Generate constant-price OHLCV data (no noise, no trend)."""
    return generate_test_ohlcv(
        n_rows=n_rows,
        base_price=base_price,
        amplitude=0.0,
        noise_level=0.0,
        trend=0.0,
        pair=pair,
        seed=seed,
    )


def generate_random_walk_ohlcv(
    n_rows: int,
    base_price: float = 100.0,
    volatility: float = 0.02,
    pair: str = DEFAULT_PAIR,
    seed: int = SEED,
) -> pl.DataFrame:
    """Generate random-walk OHLCV data."""
    rng = np.random.default_rng(seed)
    start = datetime(2024, 1, 1, 0, 0, 0)
    timestamps = [start + timedelta(minutes=i) for i in range(n_rows)]

    returns = rng.normal(0, volatility, n_rows)
    close_prices = base_price * np.exp(np.cumsum(returns))

    open_prices = np.empty(n_rows)
    open_prices[0] = close_prices[0]
    open_prices[1:] = close_prices[:-1]

    high_prices = np.maximum(open_prices, close_prices) * (1 + np.abs(rng.normal(0, volatility * 0.5, n_rows)))
    low_prices = np.minimum(open_prices, close_prices) * (1 - np.abs(rng.normal(0, volatility * 0.5, n_rows)))

    volumes = np.abs(rng.normal(1000, 300, n_rows))

    return pl.DataFrame(
        {
            "pair": [pair] * n_rows,
            "timestamp": timestamps,
            "open": open_prices,
            "high": high_prices,
            "low": low_prices,
            "close": close_prices,
            "volume": volumes,
        }
    )


def generate_empty_column_df(
    n_rows: int,
    empty_columns: list[str],
    pair: str = DEFAULT_PAIR,
    seed: int = SEED,
) -> pl.DataFrame:
    """Generate OHLCV data with specified columns set to all-null."""
    df = generate_test_ohlcv(n_rows=n_rows, pair=pair, seed=seed)
    return df.with_columns([pl.lit(None).cast(pl.Float64).alias(col) for col in empty_columns])


def generate_ohlcv_with_nulls(
    df: pl.DataFrame,
    null_fraction: float = 0.1,
    seed: int = SEED,
) -> pl.DataFrame:
    """Inject random nulls into OHLCV price columns."""
    rng = np.random.default_rng(seed)
    n = len(df)
    price_cols = ["open", "high", "low", "close"]
    exprs = []
    for col in price_cols:
        mask = rng.random(n) < null_fraction
        exprs.append(pl.when(pl.Series(mask)).then(None).otherwise(pl.col(col)).alias(col))
    return df.with_columns(exprs)


def validate_ohlcv_constraints(df: pl.DataFrame) -> bool:
    """Validate that OHLCV constraints hold (high >= low, etc.)."""
    return (
        df.filter(pl.col("high") < pl.col("low")).height == 0
        and df.filter(pl.col("high") < pl.col("open")).height == 0
        and df.filter(pl.col("high") < pl.col("close")).height == 0
        and df.filter(pl.col("low") > pl.col("open")).height == 0
        and df.filter(pl.col("low") > pl.col("close")).height == 0
    )


@pytest.fixture
def test_data() -> pl.DataFrame:
    """Standard test data for all tests."""
    return generate_test_ohlcv(n_rows=1000)


def pytest_addoption(parser):
    """Add custom command-line options for test configuration."""
    parser.addoption(
        "--max-params",
        action="store",
        type=int,
        default=None,
        help="Maximum number of parameter combinations to test per indicator (default: all)",
    )
    parser.addoption(
        "--feature-groups",
        action="store",
        default=None,
        help="Comma-separated list of feature groups to test (e.g., 'momentum,overlap'). "
        "Available: momentum, overlap, trend, volatility, volume, stat, performance, other",
    )


def pytest_configure(config):
    """Store configuration options for use in tests."""
    config.test_max_params = config.getoption("--max-params")
    config.test_feature_groups = config.getoption("--feature-groups")


def pytest_generate_tests(metafunc):
    """Dynamically generate test parameters based on command-line options."""
    if "config" in metafunc.fixturenames:
        from indicator_registry import INDICATOR_CONFIGS, filter_configs_by_options

        filtered_configs, filtered_ids = filter_configs_by_options(INDICATOR_CONFIGS, pytest_config=metafunc.config)

        metafunc.parametrize("config", filtered_configs, ids=filtered_ids, indirect=False)

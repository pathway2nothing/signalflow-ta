"""
SignalFlow-TA Test Fixtures

Provides synthetic and real market data for testing technical indicators.

Data Types:
1. Real data - 1 month of actual market data (fetched from Binance)
2. Synthetic data with preserved OHLCV constraints:
   - Static price ($100 flat line)
   - Sinusoidal price pattern
   - Random volume
   
All data maintains OHLCV invariants:
   - low <= open, close <= high
   - ohlcv >= 0
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from datetime import datetime, timedelta
from typing import Literal


# =============================================================================
# Constants
# =============================================================================

SEED = 42
DEFAULT_ROWS = 1000
DEFAULT_PAIR = "BTCUSDT"


# =============================================================================
# Data Generators
# =============================================================================


def generate_timestamps(
    n_rows: int,
    start: datetime | None = None,
    freq_minutes: int = 1,
) -> list[datetime]:
    """Generate sequential timestamps."""
    if start is None:
        start = datetime(2024, 1, 1, 0, 0, 0)
    
    return [start + timedelta(minutes=i * freq_minutes) for i in range(n_rows)]


def generate_static_ohlcv(
    n_rows: int = DEFAULT_ROWS,
    base_price: float = 100.0,
    base_volume: float = 1000.0,
    pair: str = DEFAULT_PAIR,
    seed: int = SEED,
) -> pl.DataFrame:
    """
    Generate static price data with random volume.
    
    Price stays constant at base_price, volume varies randomly.
    Maintains OHLCV invariants: low <= open, close <= high and all >= 0.
    
    Args:
        n_rows: Number of rows to generate
        base_price: Static price value (default $100)
        base_volume: Mean volume (default 1000)
        pair: Trading pair name
        seed: Random seed for reproducibility
        
    Returns:
        pl.DataFrame with columns: pair, timestamp, open, high, low, close, volume
    """
    rng = np.random.default_rng(seed)
    
    timestamps = generate_timestamps(n_rows)
    
    # Static OHLC - all same value
    open_prices = np.full(n_rows, base_price)
    high_prices = np.full(n_rows, base_price)
    low_prices = np.full(n_rows, base_price)
    close_prices = np.full(n_rows, base_price)
    
    # Random volume (positive only)
    volumes = np.abs(rng.normal(base_volume, base_volume * 0.3, n_rows))
    
    return pl.DataFrame({
        "pair": [pair] * n_rows,
        "timestamp": timestamps,
        "open": open_prices,
        "high": high_prices,
        "low": low_prices,
        "close": close_prices,
        "volume": volumes,
    })


def generate_sinusoidal_ohlcv(
    n_rows: int = DEFAULT_ROWS,
    base_price: float = 100.0,
    amplitude: float = 10.0,
    period_bars: int = 100,
    volatility: float = 0.02,
    base_volume: float = 1000.0,
    pair: str = DEFAULT_PAIR,
    seed: int = SEED,
) -> pl.DataFrame:
    """
    Generate sinusoidal price data with realistic OHLCV structure.
    
    Price follows a sine wave pattern with added noise.
    Maintains OHLCV invariants: low <= open, close <= high and all >= 0.
    
    Args:
        n_rows: Number of rows to generate
        base_price: Center price value (default $100)
        amplitude: Price swing amplitude (default $10)
        period_bars: Bars per complete cycle (default 100)
        volatility: Intra-bar volatility as fraction of price
        base_volume: Mean volume (default 1000)
        pair: Trading pair name
        seed: Random seed for reproducibility
        
    Returns:
        pl.DataFrame with columns: pair, timestamp, open, high, low, close, volume
    """
    rng = np.random.default_rng(seed)
    
    timestamps = generate_timestamps(n_rows)
    
    # Generate sinusoidal base price
    t = np.arange(n_rows)
    base_wave = base_price + amplitude * np.sin(2 * np.pi * t / period_bars)
    
    # Add small random noise to base
    noise = rng.normal(0, base_price * volatility * 0.1, n_rows)
    mid_prices = base_wave + noise
    
    # Generate OHLC from mid price
    open_prices = np.zeros(n_rows)
    close_prices = np.zeros(n_rows)
    high_prices = np.zeros(n_rows)
    low_prices = np.zeros(n_rows)
    
    # First bar
    open_prices[0] = mid_prices[0]
    close_prices[0] = mid_prices[0] * (1 + rng.normal(0, volatility))
    
    for i in range(1, n_rows):
        # Open = previous close (continuous)
        open_prices[i] = close_prices[i - 1]
        
        # Close moves toward sinusoidal trend with noise
        trend_direction = mid_prices[i] - mid_prices[i - 1]
        close_prices[i] = open_prices[i] + trend_direction + rng.normal(0, base_price * volatility)
    
    # Generate high/low ensuring constraints
    for i in range(n_rows):
        bar_range = abs(close_prices[i] - open_prices[i])
        extra_range = max(bar_range * 0.3, base_price * volatility)
        
        high_prices[i] = max(open_prices[i], close_prices[i]) + abs(rng.normal(0, extra_range))
        low_prices[i] = min(open_prices[i], close_prices[i]) - abs(rng.normal(0, extra_range))
    
    # Ensure all prices are positive
    min_price = min(low_prices.min(), 0.01)
    if min_price < 0.01:
        shift = abs(min_price) + 1
        open_prices += shift
        high_prices += shift
        low_prices += shift
        close_prices += shift
    
    # Random volume
    volumes = np.abs(rng.normal(base_volume, base_volume * 0.5, n_rows))
    
    return pl.DataFrame({
        "pair": [pair] * n_rows,
        "timestamp": timestamps,
        "open": open_prices,
        "high": high_prices,
        "low": low_prices,
        "close": close_prices,
        "volume": volumes,
    })


def generate_random_walk_ohlcv(
    n_rows: int = DEFAULT_ROWS,
    start_price: float = 100.0,
    volatility: float = 0.02,
    drift: float = 0.0001,
    base_volume: float = 1000.0,
    pair: str = DEFAULT_PAIR,
    seed: int = SEED,
) -> pl.DataFrame:
    """
    Generate random walk price data (geometric Brownian motion).
    
    More realistic price movement pattern for testing.
    Maintains OHLCV invariants: low <= open, close <= high and all >= 0.
    
    Args:
        n_rows: Number of rows to generate
        start_price: Initial price (default $100)
        volatility: Daily volatility (default 2%)
        drift: Expected drift per bar (default 0.01%)
        base_volume: Mean volume (default 1000)
        pair: Trading pair name
        seed: Random seed for reproducibility
        
    Returns:
        pl.DataFrame with columns: pair, timestamp, open, high, low, close, volume
    """
    rng = np.random.default_rng(seed)
    
    timestamps = generate_timestamps(n_rows)
    
    # Generate log returns
    log_returns = rng.normal(drift, volatility, n_rows)
    
    # Generate prices from log returns
    close_prices = np.zeros(n_rows)
    close_prices[0] = start_price
    
    for i in range(1, n_rows):
        close_prices[i] = close_prices[i - 1] * np.exp(log_returns[i])
    
    # Generate OHLC
    open_prices = np.zeros(n_rows)
    high_prices = np.zeros(n_rows)
    low_prices = np.zeros(n_rows)
    
    open_prices[0] = start_price
    
    for i in range(1, n_rows):
        open_prices[i] = close_prices[i - 1]
    
    # Generate high/low
    for i in range(n_rows):
        intra_vol = volatility * 0.5
        high_extra = abs(rng.normal(0, close_prices[i] * intra_vol))
        low_extra = abs(rng.normal(0, close_prices[i] * intra_vol))
        
        high_prices[i] = max(open_prices[i], close_prices[i]) + high_extra
        low_prices[i] = min(open_prices[i], close_prices[i]) - low_extra
        
        # Ensure low is positive
        low_prices[i] = max(low_prices[i], 0.01)
    
    # Random volume
    volumes = np.abs(rng.normal(base_volume, base_volume * 0.5, n_rows))
    
    return pl.DataFrame({
        "pair": [pair] * n_rows,
        "timestamp": timestamps,
        "open": open_prices,
        "high": high_prices,
        "low": low_prices,
        "close": close_prices,
        "volume": volumes,
    })


def generate_trending_ohlcv(
    n_rows: int = DEFAULT_ROWS,
    start_price: float = 100.0,
    end_price: float = 150.0,
    volatility: float = 0.02,
    mean_reversion: float = 0.05,
    base_volume: float = 1000.0,
    pair: str = DEFAULT_PAIR,
    seed: int = SEED,
) -> pl.DataFrame:
    """
    Generate trending price data with mean reversion (Ornstein-Uhlenbeck process).
    
    Price follows a trend from start_price to end_price with noise and 
    mean reversion that pulls price back toward the trend line.
    Good for testing trend-following indicators.
    
    Maintains OHLCV invariants: low <= open, close <= high and all >= 0.
    
    Args:
        n_rows: Number of rows to generate
        start_price: Initial price (default $100)
        end_price: Target final price (default $150 for uptrend, use <start for downtrend)
        volatility: Price volatility (default 2%)
        mean_reversion: Speed of reversion to trend (0-1, higher = faster reversion)
        base_volume: Mean volume (default 1000)
        pair: Trading pair name
        seed: Random seed for reproducibility
        
    Returns:
        pl.DataFrame with columns: pair, timestamp, open, high, low, close, volume
        
    Example:
        # Uptrend from $100 to $200
        df = generate_trending_ohlcv(start_price=100, end_price=200)
        
        # Downtrend from $100 to $50
        df = generate_trending_ohlcv(start_price=100, end_price=50)
    """
    rng = np.random.default_rng(seed)
    
    timestamps = generate_timestamps(n_rows)
    
    # Calculate trend line (linear interpolation in log space for percentage growth)
    log_start = np.log(start_price)
    log_end = np.log(end_price)
    t = np.linspace(0, 1, n_rows)
    trend_line = np.exp(log_start + (log_end - log_start) * t)
    
    # Generate prices with mean reversion to trend (Ornstein-Uhlenbeck-like)
    close_prices = np.zeros(n_rows)
    close_prices[0] = start_price
    
    for i in range(1, n_rows):
        # Random shock
        shock = rng.normal(0, close_prices[i-1] * volatility)
        
        # Mean reversion pull toward trend line
        deviation = close_prices[i-1] - trend_line[i-1]
        reversion_pull = -mean_reversion * deviation
        
        # New price = previous + shock + reversion
        close_prices[i] = close_prices[i-1] + shock + reversion_pull
        
        # Ensure positive
        close_prices[i] = max(close_prices[i], 0.01)
    
    # Generate OHLC
    open_prices = np.zeros(n_rows)
    high_prices = np.zeros(n_rows)
    low_prices = np.zeros(n_rows)
    
    open_prices[0] = start_price
    
    for i in range(1, n_rows):
        open_prices[i] = close_prices[i - 1]
    
    # Generate high/low
    for i in range(n_rows):
        intra_vol = volatility * 0.5
        high_extra = abs(rng.normal(0, close_prices[i] * intra_vol))
        low_extra = abs(rng.normal(0, close_prices[i] * intra_vol))
        
        high_prices[i] = max(open_prices[i], close_prices[i]) + high_extra
        low_prices[i] = min(open_prices[i], close_prices[i]) - low_extra
        
        # Ensure low is positive
        low_prices[i] = max(low_prices[i], 0.01)
    
    # Volume increases with volatility (more volume on big moves)
    price_changes = np.abs(np.diff(close_prices, prepend=close_prices[0]))
    volume_multiplier = 1 + (price_changes / close_prices) * 10  # Higher volume on bigger moves
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


def generate_multi_pair_ohlcv(
    pairs: list[str],
    n_rows_per_pair: int = DEFAULT_ROWS,
    data_type: Literal["static", "sinusoidal", "random_walk", "trending"] = "sinusoidal",
    seed: int = SEED,
    **kwargs,
) -> pl.DataFrame:
    """
    Generate OHLCV data for multiple trading pairs.
    
    Args:
        pairs: List of pair names
        n_rows_per_pair: Number of rows per pair
        data_type: Type of price data to generate
        seed: Base random seed
        **kwargs: Additional arguments for generators
        
    Returns:
        pl.DataFrame with data for all pairs
    """
    generators = {
        "static": generate_static_ohlcv,
        "sinusoidal": generate_sinusoidal_ohlcv,
        "random_walk": generate_random_walk_ohlcv,
        "trending": generate_trending_ohlcv,
    }
    
    gen_func = generators[data_type]
    dfs = []
    
    for i, pair in enumerate(pairs):
        # Use different seed for each pair for variety
        df = gen_func(
            n_rows=n_rows_per_pair,
            pair=pair,
            seed=seed + i * 1000,
            **kwargs,
        )
        dfs.append(df)
    
    return pl.concat(dfs)


def generate_ohlcv_with_nulls(
    base_df: pl.DataFrame,
    null_fraction: float = 0.05,
    columns: list[str] | None = None,
    seed: int = SEED,
) -> pl.DataFrame:
    """
    Add null values to OHLCV data for testing null handling.
    
    Args:
        base_df: Base DataFrame to add nulls to
        null_fraction: Fraction of values to make null
        columns: Columns to add nulls to (default: close, volume)
        seed: Random seed
        
    Returns:
        DataFrame with some null values (Polars null, not NaN)
    """
    if columns is None:
        columns = ["close", "volume"]
    
    rng = np.random.default_rng(seed)
    n_rows = len(base_df)
    
    result = base_df.clone()
    
    for col in columns:
        if col in result.columns:
            mask = rng.random(n_rows) < null_fraction
            # Use Polars when().then().otherwise() to create proper null values
            result = result.with_columns(
                pl.when(pl.lit(pl.Series(mask)))
                .then(None)
                .otherwise(pl.col(col))
                .alias(col)
            )
    
    return result


def generate_empty_column_df(
    n_rows: int = DEFAULT_ROWS,
    pair: str = DEFAULT_PAIR,
    empty_columns: list[str] | None = None,
) -> pl.DataFrame:
    """
    Generate DataFrame with empty (all-null) columns for testing.
    
    Args:
        n_rows: Number of rows
        pair: Trading pair name  
        empty_columns: Which columns should be empty (default: close)
        
    Returns:
        DataFrame with specified empty columns (Polars null values)
    """
    if empty_columns is None:
        empty_columns = ["close"]
    
    timestamps = generate_timestamps(n_rows)
    
    # Start with valid structure
    data = {
        "pair": [pair] * n_rows,
        "timestamp": timestamps,
        "open": np.full(n_rows, 100.0),
        "high": np.full(n_rows, 101.0),
        "low": np.full(n_rows, 99.0),
        "close": np.full(n_rows, 100.0),
        "volume": np.full(n_rows, 1000.0),
    }
    
    df = pl.DataFrame(data)
    
    # Make specified columns empty with proper Polars null values
    for col in empty_columns:
        if col in df.columns:
            df = df.with_columns(
                pl.lit(None).cast(pl.Float64).alias(col)
            )
    
    return df


# =============================================================================
# Pytest Fixtures
# =============================================================================


@pytest.fixture
def static_ohlcv() -> pl.DataFrame:
    """Static price ($100) OHLCV data."""
    return generate_static_ohlcv()


@pytest.fixture
def sinusoidal_ohlcv() -> pl.DataFrame:
    """Sinusoidal price pattern OHLCV data."""
    return generate_sinusoidal_ohlcv()


@pytest.fixture
def random_walk_ohlcv() -> pl.DataFrame:
    """Random walk price OHLCV data."""
    return generate_random_walk_ohlcv()


@pytest.fixture
def trending_ohlcv() -> pl.DataFrame:
    """Trending price OHLCV data (uptrend with mean reversion)."""
    return generate_trending_ohlcv()


@pytest.fixture
def downtrend_ohlcv() -> pl.DataFrame:
    """Downtrend price OHLCV data."""
    return generate_trending_ohlcv(start_price=100.0, end_price=50.0)


@pytest.fixture
def multi_pair_ohlcv() -> pl.DataFrame:
    """Multi-pair OHLCV data."""
    return generate_multi_pair_ohlcv(
        pairs=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        n_rows_per_pair=500,
    )


@pytest.fixture
def ohlcv_with_nulls(sinusoidal_ohlcv: pl.DataFrame) -> pl.DataFrame:
    """OHLCV data with some null values."""
    return generate_ohlcv_with_nulls(sinusoidal_ohlcv)


@pytest.fixture
def empty_close_ohlcv() -> pl.DataFrame:
    """OHLCV data with empty close column."""
    return generate_empty_column_df(empty_columns=["close"])


@pytest.fixture
def empty_volume_ohlcv() -> pl.DataFrame:
    """OHLCV data with empty volume column."""
    return generate_empty_column_df(empty_columns=["volume"])


@pytest.fixture
def short_ohlcv() -> pl.DataFrame:
    """Very short OHLCV data (10 rows) for edge case testing."""
    return generate_sinusoidal_ohlcv(n_rows=10)


@pytest.fixture
def long_ohlcv() -> pl.DataFrame:
    """Long OHLCV data (10000 rows) for performance testing."""
    return generate_sinusoidal_ohlcv(n_rows=10000)


# =============================================================================
# OHLCV Validation Helpers
# =============================================================================


def validate_ohlcv_constraints(df: pl.DataFrame) -> bool:
    """
    Validate that OHLCV data maintains required constraints.
    
    Constraints:
        - low <= open <= high
        - low <= close <= high
        - All values >= 0
        
    Returns:
        True if all constraints satisfied
    """
    # Skip null values in validation
    df_valid = df.drop_nulls(subset=["open", "high", "low", "close"])
    
    if len(df_valid) == 0:
        return True
    
    # Check low <= high
    if (df_valid["low"] > df_valid["high"]).any():
        return False
    
    # Check low <= open <= high
    if (df_valid["open"] < df_valid["low"]).any():
        return False
    if (df_valid["open"] > df_valid["high"]).any():
        return False
    
    # Check low <= close <= high
    if (df_valid["close"] < df_valid["low"]).any():
        return False
    if (df_valid["close"] > df_valid["high"]).any():
        return False
    
    # Check all non-negative
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df_valid.columns:
            if (df_valid[col] < 0).any():
                return False
    
    return True


# =============================================================================
# Test the fixtures themselves
# =============================================================================


class TestFixtures:
    """Tests for data generator fixtures."""
    
    def test_static_ohlcv_constraints(self, static_ohlcv):
        """Static OHLCV should maintain constraints."""
        assert validate_ohlcv_constraints(static_ohlcv)
        assert len(static_ohlcv) == DEFAULT_ROWS
        
    def test_sinusoidal_ohlcv_constraints(self, sinusoidal_ohlcv):
        """Sinusoidal OHLCV should maintain constraints."""
        assert validate_ohlcv_constraints(sinusoidal_ohlcv)
        assert len(sinusoidal_ohlcv) == DEFAULT_ROWS
        
    def test_random_walk_ohlcv_constraints(self, random_walk_ohlcv):
        """Random walk OHLCV should maintain constraints."""
        assert validate_ohlcv_constraints(random_walk_ohlcv)
        assert len(random_walk_ohlcv) == DEFAULT_ROWS
    
    def test_trending_ohlcv_constraints(self, trending_ohlcv):
        """Trending OHLCV should maintain constraints."""
        assert validate_ohlcv_constraints(trending_ohlcv)
        assert len(trending_ohlcv) == DEFAULT_ROWS
    
    def test_trending_ohlcv_has_uptrend(self, trending_ohlcv):
        """Trending OHLCV should show upward trend."""
        close = trending_ohlcv["close"].to_numpy()
        # Compare first 10% vs last 10% average
        first_avg = np.mean(close[:len(close)//10])
        last_avg = np.mean(close[-len(close)//10:])
        assert last_avg > first_avg, "Uptrend should have higher end price"
    
    def test_downtrend_ohlcv_has_downtrend(self, downtrend_ohlcv):
        """Downtrend OHLCV should show downward trend."""
        close = downtrend_ohlcv["close"].to_numpy()
        first_avg = np.mean(close[:len(close)//10])
        last_avg = np.mean(close[-len(close)//10:])
        assert last_avg < first_avg, "Downtrend should have lower end price"
        
    def test_multi_pair_has_all_pairs(self, multi_pair_ohlcv):
        """Multi-pair data should contain all pairs."""
        pairs = multi_pair_ohlcv["pair"].unique().to_list()
        assert set(pairs) == {"BTCUSDT", "ETHUSDT", "SOLUSDT"}
        
    def test_static_price_is_constant(self, static_ohlcv):
        """Static price should be constant."""
        assert static_ohlcv["close"].std() == 0
        assert static_ohlcv["close"].mean() == 100.0
        
    def test_sinusoidal_has_variation(self, sinusoidal_ohlcv):
        """Sinusoidal price should have variation."""
        assert sinusoidal_ohlcv["close"].std() > 0
        
    def test_reproducibility_with_seed(self):
        """Same seed should produce same data."""
        df1 = generate_sinusoidal_ohlcv(seed=123)
        df2 = generate_sinusoidal_ohlcv(seed=123)
        
        assert df1.equals(df2)
        
    def test_different_seeds_produce_different_data(self):
        """Different seeds should produce different data."""
        df1 = generate_sinusoidal_ohlcv(seed=123)
        df2 = generate_sinusoidal_ohlcv(seed=456)
        
        assert not df1.equals(df2)
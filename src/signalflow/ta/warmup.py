"""
Automatic warmup period calculation for technical indicators.

Provides utilities to determine the minimum number of bars required
for an indicator to produce stable, reproducible values.
"""

from typing import Any
import inspect


def get_indicator_warmup(indicator: Any) -> int:
    """
    Calculate warmup period for a technical indicator.

    Warmup is the minimum number of bars needed for the indicator
    to produce stable values that are independent of the starting point.

    Parameters
    ----------
    indicator : Feature
        Technical indicator instance

    Returns
    -------
    warmup : int
        Minimum bars required for stable output

    Notes
    -----
    Warmup multipliers by indicator type:
    - SMA/WMA: 1x period (minimal warmup)
    - EMA/RMA: 5x period (EMA convergence to 0.01% tolerance)
    - RSI: 10x period (RMA-based, needs extra convergence)
    - MACD: 5x slow period
    - Stochastic: 3x (k_period + d_period)
    - StochRSI: 6x period (RSI + Stochastic)
    - TRIX: 12x period (triple EMA)
    - TSI: 8x period (double EMA)
    - T3: 6x period (cascaded EMAs)
    - ADX: 5x period (EMA-based smoothing)
    - ATR: 5x period (EMA smoothing)
    - Bollinger Bands: 2x period (SMA-based)
    - KAMA/Adaptive: 5x period (EMA-like)
    - Divergence indicators: pivot_window * 2 + lookback
    """
    # Get indicator class name
    class_name = indicator.__class__.__name__

    # Extract common parameters
    params = {}
    for attr in ['period', 'length', 'window', 'slow', 'fast', 'k_period',
                 'd_period', 'rsi_period', 'stoch_period', 'pivot_window', 'lookback']:
        if hasattr(indicator, attr):
            params[attr] = getattr(indicator, attr)

    # Default warmup
    base_period = params.get('period', params.get('length', params.get('window', 20)))

    # Specific warmup rules by indicator type
    indicator_lower = class_name.lower()

    # RSI and RMA-based indicators (10x period for full convergence)
    if 'rsi' in indicator_lower and 'stoch' not in indicator_lower:
        return base_period * 10

    # StochRSI (RSI + Stochastic)
    if 'stochrsi' in indicator_lower:
        rsi_period = params.get('rsi_period', 14)
        stoch_period = params.get('stoch_period', 14)
        return rsi_period * 6 + stoch_period * 3

    # TRIX (triple EMA)
    if 'trix' in indicator_lower:
        return base_period * 12

    # TSI (double EMA)
    if 'tsi' in indicator_lower:
        return base_period * 8

    # T3 (6 cascaded EMAs)
    if 't3' in indicator_lower:
        return base_period * 6

    # MACD and PPO
    if 'macd' in indicator_lower or 'ppo' in indicator_lower:
        slow = params.get('slow', 26)
        return slow * 5

    # Stochastic oscillators
    if 'stoch' in indicator_lower and 'rsi' not in indicator_lower:
        k = params.get('k_period', 14)
        d = params.get('d_period', 3)
        smooth = params.get('smooth_k', 1)
        return (k + d + smooth) * 3

    # ADX and directional indicators
    if 'adx' in indicator_lower or 'di' in indicator_lower:
        return base_period * 5

    # ATR and volatility with EMA smoothing
    if 'atr' in indicator_lower or 'natr' in indicator_lower:
        return base_period * 5

    # Bollinger Bands and SMA-based
    if 'bollinger' in indicator_lower or 'bb' in indicator_lower:
        return base_period * 2

    # KAMA and adaptive indicators (EMA-like convergence)
    if 'kama' in indicator_lower or 'adaptive' in indicator_lower:
        return base_period * 5

    # JMA (Jurik)
    if 'jma' in indicator_lower or 'jurik' in indicator_lower:
        return base_period * 5

    # VIDYA and T3
    if 'vidya' in indicator_lower:
        return base_period * 5

    # Divergence indicators
    if 'divergence' in indicator_lower or 'div' in class_name:
        pivot_window = params.get('pivot_window', 5)
        lookback = params.get('lookback', 100)
        rsi_period = params.get('rsi_period', params.get('period', 14))
        # Need: RSI warmup + pivot confirmation + lookback
        return rsi_period * 10 + pivot_window * 2 + lookback

    # EMA-based indicators (5x period for 0.01% tolerance)
    if any(x in indicator_lower for x in ['ema', 'tema', 'dema', 'zlma', 'hull']):
        return base_period * 5

    # WMA and linear-weighted
    if 'wma' in indicator_lower or 'weighted' in indicator_lower:
        return base_period

    # SMA and simple rolling
    if 'sma' in indicator_lower or 'simple' in indicator_lower:
        return base_period

    # Ichimoku
    if 'ichimoku' in indicator_lower:
        if hasattr(indicator, 'senkou_period'):
            return getattr(indicator, 'senkou_period') * 2
        return 52 * 2  # Default senkou_b period

    # Default: use EMA-like warmup (conservative)
    return base_period * 5


def add_warmup_property(cls):
    """
    Class decorator to add warmup property to indicator classes.

    Usage:
        @add_warmup_property
        @dataclass
        class MyIndicator(Feature):
            period: int = 14
            ...
    """
    # Store original __init__ if it exists
    original_init = cls.__init__ if hasattr(cls, '__init__') else None

    # Add warmup as a cached property
    @property
    def warmup(self):
        """Minimum bars needed for stable, reproducible output."""
        if not hasattr(self, '_cached_warmup'):
            self._cached_warmup = get_indicator_warmup(self)
        return self._cached_warmup

    cls.warmup = warmup
    return cls

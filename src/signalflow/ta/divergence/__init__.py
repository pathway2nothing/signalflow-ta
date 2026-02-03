"""
Divergence Detection Module

Automatic detection of price-indicator divergences for trading signals.

Available Detectors:
- RsiDivergence: RSI-based divergence detection
- MacdDivergence: MACD histogram divergence detection
- StochDivergence: Stochastic oscillator divergence detection
- VolumeDivergence: Volume-based divergence detection
- MultiDivergence: Multi-indicator confluence detection

Divergence Types:
- Regular Bullish: Price LL, Indicator HL (reversal up)
- Regular Bearish: Price HH, Indicator LH (reversal down)
- Hidden Bullish: Price HL, Indicator LL (trend continuation up)
- Hidden Bearish: Price LH, Indicator HH (trend continuation down)
"""

from signalflow.ta.divergence.base import DivergenceBase
from signalflow.ta.divergence.rsi_div import RsiDivergence
from signalflow.ta.divergence.macd_div import MacdDivergence

__all__ = [
    "DivergenceBase",
    "RsiDivergence",
    "MacdDivergence",
]

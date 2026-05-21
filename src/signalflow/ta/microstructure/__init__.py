"""Microstructure / candle-geometry indicators.

Captures information from candlestick shape (body, wicks, range) rather than
price level or smoothed indicators. Strong signal for indecision/rejection regimes.

Most classes added from sf-profit feature research (iter-15, iter-18, iter-18-extra, iter-20).
"""
from signalflow.ta.microstructure.candles import (
    BodyToRangeRatio,
    ClosePositionInBar,
    HiLoMedianGap,
    LowerWickPersistence,
    UpperWickPersistence,
    WickAsymmetry,
    WickToBodyLower,
    WickToBodyRatio,
    WickToBodyUpper,
)
from signalflow.ta.microstructure.ranges import (
    HighLowImbalance,
    RangeExpansion,
    RangeFragmentation,
    RangeNormalizedReturn,
    SignedRollingRange,
)

__all__ = [
    # candles
    "BodyToRangeRatio",
    "ClosePositionInBar",
    "HiLoMedianGap",
    "LowerWickPersistence",
    "UpperWickPersistence",
    "WickAsymmetry",
    "WickToBodyLower",
    "WickToBodyRatio",
    "WickToBodyUpper",
    # ranges
    "HighLowImbalance",
    "RangeExpansion",
    "RangeFragmentation",
    "RangeNormalizedReturn",
    "SignedRollingRange",
]

"""Trend indicators - strength, direction, stops.

Modules:
    strength - Trend strength measures (ADX, Aroon, Vortex, VHF, CHOP)
    stops - Trailing stops and trend-following (PSAR, Supertrend, Chandelier, HiLo)
    detection - Trend detection systems (Ichimoku, DPO, QStick)
    regime - Trend regime detectors (Alligator, TwoMA, SMA/LinReg direction)
"""

from signalflow.ta.trend.detection import (
    DpoTrend,
    IchimokuTrend,
    QstickTrend,
    TtmTrend,
)
from signalflow.ta.trend.regime import (
    LinRegDiffDirection,
    LinRegDirection,
    LinRegPriceDiff,
    SmaDiffDirection,
    SmaDirection,
    TwoMaRegime,
    WilliamsAlligatorRegime,
)
from signalflow.ta.trend.stops import (
    ChandelierTrend,
    CkspTrend,
    HiloTrend,
    PsarTrend,
    SupertrendTrend,
)
from signalflow.ta.trend.strength import (
    AdxTrend,
    AroonTrend,
    ChopTrend,
    MarketImpedanceTrend,
    OrderParameterTrend,
    RCTimeConstantTrend,
    ReynoldsTrend,
    RotationalInertiaTrend,
    SNRTrend,
    SusceptibilityTrend,
    VhfTrend,
    ViscosityTrend,
    VortexTrend,
)

__all__ = [
    # Strength
    "AdxTrend",
    "AroonTrend",
    "VortexTrend",
    "VhfTrend",
    "ChopTrend",
    "ViscosityTrend",
    "ReynoldsTrend",
    "RotationalInertiaTrend",
    "MarketImpedanceTrend",
    "RCTimeConstantTrend",
    "SNRTrend",
    "OrderParameterTrend",
    "SusceptibilityTrend",
    # Stops
    "PsarTrend",
    "SupertrendTrend",
    "ChandelierTrend",
    "HiloTrend",
    "CkspTrend",
    # Detection
    "IchimokuTrend",
    "DpoTrend",
    "QstickTrend",
    "TtmTrend",
    # Regime
    "WilliamsAlligatorRegime",
    "TwoMaRegime",
    "SmaDirection",
    "SmaDiffDirection",
    "LinRegDirection",
    "LinRegDiffDirection",
    "LinRegPriceDiff",
]

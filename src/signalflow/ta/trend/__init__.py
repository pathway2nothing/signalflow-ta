"""Trend indicators - strength, direction, stops.

Modules:
    strength - Trend strength measures (ADX, Aroon, Vortex, VHF, CHOP)
    stops - Trailing stops and trend-following (PSAR, Supertrend, Chandelier, HiLo)
    detection - Trend detection systems (Ichimoku, DPO, QStick)
    regime - Trend regime detectors (Alligator, TwoMA, SMA/LinReg direction)
"""

from signalflow.ta.trend.strength import (
    AdxTrend,
    AroonTrend,
    VortexTrend,
    VhfTrend,
    ChopTrend,
    ViscosityTrend,
    ReynoldsTrend,
    RotationalInertiaTrend,
    MarketImpedanceTrend,
    RCTimeConstantTrend,
    SNRTrend,
    OrderParameterTrend,
    SusceptibilityTrend,
)
from signalflow.ta.trend.stops import (
    PsarTrend,
    SupertrendTrend,
    ChandelierTrend,
    HiloTrend,
    CkspTrend,
)
from signalflow.ta.trend.detection import (
    IchimokuTrend,
    DpoTrend,
    QstickTrend,
    TtmTrend,
)
from signalflow.ta.trend.regime import (
    WilliamsAlligatorRegime,
    TwoMaRegime,
    SmaDirection,
    SmaDiffDirection,
    LinRegDirection,
    LinRegDiffDirection,
    LinRegPriceDiff,
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

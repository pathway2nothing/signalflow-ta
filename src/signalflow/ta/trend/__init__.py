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
from signalflow.ta.trend.strength_causal import AdxTrendCausal
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
from signalflow.ta.trend.extensions import (
    DiBalance,
    EntropyRatio,
    HilbertAmplitudeSlope,
    NatrXDiBalance,
    RsiDivPolarity,
    UpDownEntropyAsymmetry,
)

__all__ = [
    "AdxTrend",
    "AdxTrendCausal",
    "AroonTrend",
    "ChandelierTrend",
    "ChopTrend",
    "CkspTrend",
    "DpoTrend",
    "HiloTrend",
    "IchimokuTrend",
    "LinRegDiffDirection",
    "LinRegDirection",
    "LinRegPriceDiff",
    "MarketImpedanceTrend",
    "OrderParameterTrend",
    "PsarTrend",
    "QstickTrend",
    "RCTimeConstantTrend",
    "ReynoldsTrend",
    "RotationalInertiaTrend",
    "SNRTrend",
    "SmaDiffDirection",
    "SmaDirection",
    "SupertrendTrend",
    "SusceptibilityTrend",
    "TtmTrend",
    "TwoMaRegime",
    "VhfTrend",
    "ViscosityTrend",
    "VortexTrend",
    "WilliamsAlligatorRegime",
    "DiBalance",
    "EntropyRatio",
    "HilbertAmplitudeSlope",
    "NatrXDiBalance",
    "RsiDivPolarity",
    "UpDownEntropyAsymmetry",
]

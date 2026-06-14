"""Momentum indicators - measure speed and magnitude of price changes.

Modules:
    core - Basic momentum (RSI, ROC, MOM, CMO)
    oscillators - Stochastic family (Stoch, StochRSI, Williams %R, CCI)
    macd - MACD family (MACD, PPO, TSI, TRIX)
"""

from signalflow.ta.momentum.core import (
    CmoMom,
    MomMom,
    RocMom,
    RsiMom,
)
from signalflow.ta.momentum.kinematics import (
    AccelerationMom,
    AngularMomentumMom,
    JerkMom,
    TorqueMom,
)
from signalflow.ta.momentum.macd import (
    MacdMom,
    PpoMom,
    TrixMom,
    TsiMom,
)
from signalflow.ta.momentum.oscillators import (
    AoMom,
    CciMom,
    StochMom,
    StochRsiMom,
    UoMom,
    WillrMom,
)
from signalflow.ta.momentum.extensions import (
    MacdNorm,
    MomPosNeg,
    MomentumOfMomentum,
    PriceAcceleration,
    PriceMomentumConfirmation,
    RocSignedLog,
    RsiSpread,
    TrendPersistence,
    VolPriceConfirmation,
)

__all__ = [
    "AccelerationMom",
    "AngularMomentumMom",
    "AoMom",
    "CciMom",
    "CmoMom",
    "JerkMom",
    "MacdMom",
    "MomMom",
    "PpoMom",
    "RocMom",
    "RsiMom",
    "StochMom",
    "StochRsiMom",
    "TorqueMom",
    "TrixMom",
    "TsiMom",
    "UoMom",
    "WillrMom",
    "MacdNorm",
    "MomPosNeg",
    "MomentumOfMomentum",
    "PriceAcceleration",
    "PriceMomentumConfirmation",
    "RocSignedLog",
    "RsiSpread",
    "TrendPersistence",
    "VolPriceConfirmation",
]

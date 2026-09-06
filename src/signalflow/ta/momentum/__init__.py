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
from signalflow.ta.momentum.extensions import (
    MacdNorm,
    MomentumOfMomentum,
    MomPosNeg,
    PriceAcceleration,
    PriceMomentumConfirmation,
    RocSignedLog,
    RsiSpread,
    TrendPersistence,
    VolPriceConfirmation,
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

__all__ = [
    "AccelerationMom",
    "AngularMomentumMom",
    "AoMom",
    "CciMom",
    "CmoMom",
    "JerkMom",
    "MacdMom",
    "MacdNorm",
    "MomMom",
    "MomPosNeg",
    "MomentumOfMomentum",
    "PpoMom",
    "PriceAcceleration",
    "PriceMomentumConfirmation",
    "RocMom",
    "RocSignedLog",
    "RsiMom",
    "RsiSpread",
    "StochMom",
    "StochRsiMom",
    "TorqueMom",
    "TrendPersistence",
    "TrixMom",
    "TsiMom",
    "UoMom",
    "VolPriceConfirmation",
    "WillrMom",
]

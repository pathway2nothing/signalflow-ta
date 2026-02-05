# src/signalflow/ta/momentum/__init__.py
"""Momentum indicators - measure speed and magnitude of price changes.

Modules:
    core - Basic momentum (RSI, ROC, MOM, CMO)
    oscillators - Stochastic family (Stoch, StochRSI, Williams %R, CCI)
    macd - MACD family (MACD, PPO, TSI, TRIX)
"""

from signalflow.ta.momentum.core import (
    RsiMom,
    RocMom,
    MomMom,
    CmoMom,
)
from signalflow.ta.momentum.oscillators import (
    StochMom,
    StochRsiMom,
    WillrMom,
    CciMom,
    UoMom,
    AoMom,
)
from signalflow.ta.momentum.macd import (
    MacdMom,
    PpoMom,
    TsiMom,
    TrixMom,
)
from signalflow.ta.momentum.kinematics import (
    AccelerationMom,
    JerkMom,
    AngularMomentumMom,
    TorqueMom,
)

__all__ = [
    # Core
    "RsiMom",
    "RocMom",
    "MomMom",
    "CmoMom",
    # Oscillators
    "StochMom",
    "StochRsiMom",
    "WillrMom",
    "CciMom",
    "UoMom",
    "AoMom",
    # MACD family
    "MacdMom",
    "PpoMom",
    "TsiMom",
    "TrixMom",
    # Kinematics
    "AccelerationMom",
    "JerkMom",
    "AngularMomentumMom",
    "TorqueMom",
]

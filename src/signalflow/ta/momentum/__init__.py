# src/signalflow/ta/momentum/__init__.py
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

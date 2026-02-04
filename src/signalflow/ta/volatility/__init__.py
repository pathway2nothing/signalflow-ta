# src/signalflow/ta/volatility/__init__.py
"""Volatility indicators - measure price variability.

Modules:
    range - True Range and ATR variants
    bands - Channel/envelope indicators (Bollinger, Keltner, Donchian)
    measures - Other volatility metrics (Mass Index, Ulcer Index, RVI)
"""

from signalflow.ta.volatility.range import (
    TrueRangeVol,
    AtrVol,
    NatrVol,
)
from signalflow.ta.volatility.bands import (
    BollingerVol,
    KeltnerVol,
    DonchianVol,
    AccBandsVol,
)
from signalflow.ta.volatility.measures import (
    MassIndexVol,
    UlcerIndexVol,
    RviVol,
)
from signalflow.ta.volatility.gaps import GapVol
from signalflow.ta.volatility.energy import (
    KineticEnergyVol,
    PotentialEnergyVol,
    TotalEnergyVol,
    EnergyFlowVol,
    ElasticStrainVol,
    TemperatureVol,
    HeatCapacityVol,
    FreeEnergyVol,
)

__all__ = [
    # Range
    "TrueRangeVol",
    "AtrVol",
    "NatrVol",
    # Bands
    "BollingerVol",
    "KeltnerVol",
    "DonchianVol",
    "AccBandsVol",
    # Measures
    "MassIndexVol",
    "UlcerIndexVol",
    "RviVol",
    "GapVol",
    # Energy
    "KineticEnergyVol",
    "PotentialEnergyVol",
    "TotalEnergyVol",
    "EnergyFlowVol",
    "ElasticStrainVol",
    "TemperatureVol",
    "HeatCapacityVol",
    "FreeEnergyVol",
]
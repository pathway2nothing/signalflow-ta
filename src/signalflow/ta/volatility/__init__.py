# src/signalflow/ta/volatility/__init__.py
"""Volatility indicators - measure price variability.

Modules:
    range - True Range and ATR variants
    bands - Channel/envelope indicators (Bollinger, Keltner, Donchian)
    measures - Other volatility metrics (Mass Index, Ulcer Index, RVI)
"""

from signalflow.ta.volatility.bands import (
    AccBandsVol,
    BollingerVol,
    DonchianVol,
    KeltnerVol,
)
from signalflow.ta.volatility.energy import (
    ElasticStrainVol,
    EnergyFlowVol,
    FreeEnergyVol,
    HeatCapacityVol,
    KineticEnergyVol,
    PotentialEnergyVol,
    TemperatureVol,
    TotalEnergyVol,
)
from signalflow.ta.volatility.gaps import GapVol
from signalflow.ta.volatility.measures import (
    MassIndexVol,
    RviVol,
    UlcerIndexVol,
)
from signalflow.ta.volatility.range import (
    AtrVol,
    NatrVol,
    TrueRangeVol,
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

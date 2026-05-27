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
from signalflow.ta.volatility.extensions import (
    AltVolDeviation,
    GarmanKlassPctRank,
    GarmanKlassRatio,
    NatrPctRank,
    NatrRatio,
    ParkinsonAccel,
    ParkinsonVolRatio,
    ParkinsonZScore,
    PriceZAtr,
    RealizedVolPctRank,
    RealizedVolRatio,
    VolOfVol,
)

__all__ = [
    "AccBandsVol",
    "AtrVol",
    # Bands
    "BollingerVol",
    "DonchianVol",
    "ElasticStrainVol",
    "EnergyFlowVol",
    "FreeEnergyVol",
    "GapVol",
    "HeatCapacityVol",
    "KeltnerVol",
    # Energy
    "KineticEnergyVol",
    # Measures
    "MassIndexVol",
    "NatrVol",
    "PotentialEnergyVol",
    "RviVol",
    "TemperatureVol",
    "TotalEnergyVol",
    # Range
    "TrueRangeVol",
    "UlcerIndexVol",
    # Extensions (sf-profit iter-3.1/15/16/20)
    "AltVolDeviation",
    "GarmanKlassPctRank",
    "GarmanKlassRatio",
    "NatrPctRank",
    "NatrRatio",
    "ParkinsonAccel",
    "ParkinsonVolRatio",
    "ParkinsonZScore",
    "PriceZAtr",
    "RealizedVolPctRank",
    "RealizedVolRatio",
    "VolOfVol",
]

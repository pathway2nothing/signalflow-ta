"""Preset pipes for volatility indicators."""
from __future__ import annotations

from signalflow.feature.base import Feature
from signalflow.ta.volatility import (
    TrueRangeVol, AtrVol, NatrVol,
    BollingerVol, KeltnerVol, DonchianVol, AccBandsVol,
    MassIndexVol, UlcerIndexVol, RviVol, GapVol,
    KineticEnergyVol, PotentialEnergyVol, TotalEnergyVol,
    EnergyFlowVol, ElasticStrainVol,
    TemperatureVol, HeatCapacityVol, FreeEnergyVol,
)


def volatility_range_pipe(*, normalized: bool = False) -> list[Feature]:
    """Range-based volatility: TR, ATR, NATR."""
    return [
        TrueRangeVol(normalized=normalized),
        AtrVol(period=14, normalized=normalized),
        NatrVol(period=14, normalized=normalized),
    ]


def volatility_bands_pipe(*, normalized: bool = False) -> list[Feature]:
    """Band/envelope indicators: Bollinger, Keltner, Donchian, AccBands."""
    return [
        BollingerVol(period=20, std_dev=2.0, normalized=normalized),
        KeltnerVol(period=20, multiplier=2.0, normalized=normalized),
        DonchianVol(period=20, normalized=normalized),
        AccBandsVol(period=20, normalized=normalized),
    ]


def volatility_measures_pipe(*, normalized: bool = False) -> list[Feature]:
    """Volatility measures: Mass Index, Ulcer Index, RVI, Gap."""
    return [
        MassIndexVol(normalized=normalized),
        UlcerIndexVol(period=14, normalized=normalized),
        RviVol(period=14, normalized=normalized),
        GapVol(normalized=normalized),
    ]


def volatility_energy_pipe(*, normalized: bool = False) -> list[Feature]:
    """Physics energy-based volatility indicators."""
    return [
        KineticEnergyVol(period=20, normalized=normalized),
        PotentialEnergyVol(period=20, normalized=normalized),
        TotalEnergyVol(period=20, normalized=normalized),
        EnergyFlowVol(period=20, normalized=normalized),
        ElasticStrainVol(period=20, normalized=normalized),
        TemperatureVol(period=20, normalized=normalized),
        HeatCapacityVol(period=20, normalized=normalized),
        FreeEnergyVol(period=20, normalized=normalized),
    ]


def volatility_pipe(*, normalized: bool = False) -> list[Feature]:
    """All volatility indicators."""
    return [
        *volatility_range_pipe(normalized=normalized),
        *volatility_bands_pipe(normalized=normalized),
        *volatility_measures_pipe(normalized=normalized),
        *volatility_energy_pipe(normalized=normalized),
    ]

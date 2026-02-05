"""Preset pipes for volume indicators."""

from __future__ import annotations

from signalflow.feature.base import Feature
from signalflow.ta.volume import (
    ObvVolume,
    AdVolume,
    PvtVolume,
    NviVolume,
    PviVolume,
    MfiVolume,
    CmfVolume,
    EfiVolume,
    EomVolume,
    KvoVolume,
    MarketForceVolume,
    ImpulseVolume,
    MarketMomentumVolume,
    MarketPowerVolume,
    MarketCapacitanceVolume,
    GravitationalPullVolume,
)


def volume_cumulative_pipe(*, normalized: bool = False) -> list[Feature]:
    """Cumulative volume-price indicators: OBV, A/D, PVT, NVI, PVI."""
    return [
        ObvVolume(period=20, normalized=normalized),
        AdVolume(period=20, normalized=normalized),
        PvtVolume(period=20, normalized=normalized),
        NviVolume(normalized=normalized),
        PviVolume(normalized=normalized),
    ]


def volume_oscillators_pipe(*, normalized: bool = False) -> list[Feature]:
    """Volume oscillators: MFI, CMF, EFI, EOM, KVO."""
    return [
        MfiVolume(period=14, normalized=normalized),
        CmfVolume(period=20, normalized=normalized),
        EfiVolume(period=13, normalized=normalized),
        EomVolume(period=14, normalized=normalized),
        KvoVolume(normalized=normalized),
    ]


def volume_dynamics_pipe(*, normalized: bool = False) -> list[Feature]:
    """Physics-inspired volume dynamics."""
    return [
        MarketForceVolume(period=14, normalized=normalized),
        ImpulseVolume(period=14, normalized=normalized),
        MarketMomentumVolume(period=14, normalized=normalized),
        MarketPowerVolume(period=14, normalized=normalized),
        MarketCapacitanceVolume(period=20, normalized=normalized),
        GravitationalPullVolume(period=20, normalized=normalized),
    ]


def volume_pipe(*, normalized: bool = False) -> list[Feature]:
    """All volume indicators."""
    return [
        *volume_cumulative_pipe(normalized=normalized),
        *volume_oscillators_pipe(normalized=normalized),
        *volume_dynamics_pipe(normalized=normalized),
    ]

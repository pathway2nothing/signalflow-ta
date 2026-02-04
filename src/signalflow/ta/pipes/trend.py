"""Preset pipes for trend indicators."""
from __future__ import annotations

from signalflow.feature.base import Feature
from signalflow.ta.trend import (
    AdxTrend, AroonTrend, VortexTrend, VhfTrend, ChopTrend,
    ViscosityTrend, ReynoldsTrend, RotationalInertiaTrend,
    MarketImpedanceTrend, RCTimeConstantTrend, SNRTrend,
    OrderParameterTrend, SusceptibilityTrend,
    PsarTrend, SupertrendTrend, ChandelierTrend, HiloTrend, CkspTrend,
    IchimokuTrend, DpoTrend, QstickTrend, TtmTrend,
)


def trend_strength_pipe(*, normalized: bool = False) -> list[Feature]:
    """Trend strength measures: ADX, Aroon, Vortex, VHF, CHOP + physics-inspired."""
    return [
        AdxTrend(period=14, normalized=normalized),
        AroonTrend(period=25, normalized=normalized),
        VortexTrend(period=14, normalized=normalized),
        VhfTrend(period=28, normalized=normalized),
        ChopTrend(period=14, normalized=normalized),
        ViscosityTrend(period=20, normalized=normalized),
        ReynoldsTrend(period=20, normalized=normalized),
        RotationalInertiaTrend(period=20, normalized=normalized),
        MarketImpedanceTrend(period=20, normalized=normalized),
        RCTimeConstantTrend(period=20, normalized=normalized),
        SNRTrend(period=20, normalized=normalized),
        OrderParameterTrend(period=20, normalized=normalized),
        SusceptibilityTrend(period=20, normalized=normalized),
    ]


def trend_stops_pipe(*, normalized: bool = False) -> list[Feature]:
    """Trailing stops: PSAR, Supertrend, Chandelier, HiLo, CKSP."""
    return [
        PsarTrend(normalized=normalized),
        SupertrendTrend(period=10, multiplier=3.0, normalized=normalized),
        ChandelierTrend(period=22, multiplier=3.0, normalized=normalized),
        HiloTrend(normalized=normalized),
        CkspTrend(normalized=normalized),
    ]


def trend_detection_pipe(*, normalized: bool = False) -> list[Feature]:
    """Trend detection: Ichimoku, DPO, QStick, TTM."""
    return [
        IchimokuTrend(normalized=normalized),
        DpoTrend(period=20, normalized=normalized),
        QstickTrend(period=10, normalized=normalized),
        TtmTrend(normalized=normalized),
    ]


def trend_pipe(*, normalized: bool = False) -> list[Feature]:
    """All trend indicators."""
    return [
        *trend_strength_pipe(normalized=normalized),
        *trend_stops_pipe(normalized=normalized),
        *trend_detection_pipe(normalized=normalized),
    ]

"""Preset pipes for momentum indicators."""

from __future__ import annotations

from signalflow.feature.base import Feature
from signalflow.ta.momentum import (
    RsiMom,
    RocMom,
    MomMom,
    CmoMom,
    StochMom,
    StochRsiMom,
    WillrMom,
    CciMom,
    UoMom,
    AoMom,
    MacdMom,
    PpoMom,
    TsiMom,
    TrixMom,
    AccelerationMom,
    JerkMom,
    AngularMomentumMom,
    TorqueMom,
)


def momentum_core_pipe(*, normalized: bool = False) -> list[Feature]:
    """Core momentum indicators: RSI, ROC, MOM, CMO."""
    return [
        RsiMom(period=14, normalized=normalized),
        RocMom(period=10, normalized=normalized),
        MomMom(period=10, normalized=normalized),
        CmoMom(period=14, normalized=normalized),
    ]


def momentum_oscillators_pipe(*, normalized: bool = False) -> list[Feature]:
    """Stochastic family: Stoch, StochRSI, Williams %R, CCI, UO, AO."""
    return [
        StochMom(k_period=14, d_period=3, smooth_k=3, normalized=normalized),
        StochRsiMom(
            rsi_period=14,
            stoch_period=14,
            k_period=3,
            d_period=3,
            normalized=normalized,
        ),
        WillrMom(period=14, normalized=normalized),
        CciMom(period=20, normalized=normalized),
        UoMom(normalized=normalized),
        AoMom(normalized=normalized),
    ]


def momentum_macd_pipe(*, normalized: bool = False) -> list[Feature]:
    """MACD family: MACD, PPO, TSI, TRIX."""
    return [
        MacdMom(fast=12, slow=26, signal=9, normalized=normalized),
        PpoMom(fast=12, slow=26, signal=9, normalized=normalized),
        TsiMom(fast=13, slow=25, signal=13, normalized=normalized),
        TrixMom(period=18, signal=9, normalized=normalized),
    ]


def momentum_kinematics_pipe(
    *,
    source_col: str = "close",
    normalized: bool = False,
) -> list[Feature]:
    """Physics-inspired kinematics: Acceleration, Jerk, Angular Momentum, Torque."""
    return [
        AccelerationMom(source_col=source_col, normalized=normalized),
        JerkMom(source_col=source_col, normalized=normalized),
        AngularMomentumMom(source_col=source_col, normalized=normalized),
        TorqueMom(source_col=source_col, normalized=normalized),
    ]


def momentum_pipe(
    *,
    source_col: str = "close",
    normalized: bool = False,
) -> list[Feature]:
    """All momentum indicators."""
    return [
        *momentum_core_pipe(normalized=normalized),
        *momentum_oscillators_pipe(normalized=normalized),
        *momentum_macd_pipe(normalized=normalized),
        *momentum_kinematics_pipe(source_col=source_col, normalized=normalized),
    ]

"""Preset pipes for overlap indicators (smoothers & price transforms)."""

from __future__ import annotations

from signalflow.feature.base import Feature
from signalflow.ta.overlap import (
    SmaSmooth,
    EmaSmooth,
    WmaSmooth,
    RmaSmooth,
    DemaSmooth,
    TemaSmooth,
    HmaSmooth,
    TrimaSmooth,
    SwmaSmooth,
    SsfSmooth,
    FftSmooth,
    KamaSmooth,
    AlmaSmooth,
    JmaSmooth,
    VidyaSmooth,
    T3Smooth,
    ZlmaSmooth,
    McGinleySmooth,
    FramaSmooth,
    Hl2Price,
    Hlc3Price,
    Ohlc4Price,
    WcpPrice,
    MidpointPrice,
    MidpricePrice,
    TypicalPrice,
)


def smoothers_pipe(
    *,
    source_col: str = "close",
    normalized: bool = False,
) -> list[Feature]:
    """All smoothing moving averages with standard periods.

    Includes: SMA, EMA, WMA, RMA, DEMA, TEMA, HMA, TRIMA, SWMA, SSF, FFT,
    KAMA, ALMA, JMA, VIDYA, T3, ZLMA, McGinley, FRAMA.
    """
    return [
        SmaSmooth(source_col=source_col, period=20, normalized=normalized),
        EmaSmooth(source_col=source_col, period=20, normalized=normalized),
        WmaSmooth(source_col=source_col, period=20, normalized=normalized),
        RmaSmooth(source_col=source_col, period=14, normalized=normalized),
        DemaSmooth(source_col=source_col, period=20, normalized=normalized),
        TemaSmooth(source_col=source_col, period=20, normalized=normalized),
        HmaSmooth(source_col=source_col, period=20, normalized=normalized),
        TrimaSmooth(source_col=source_col, period=20, normalized=normalized),
        SwmaSmooth(source_col=source_col, period=4, normalized=normalized),
        SsfSmooth(source_col=source_col, period=10, normalized=normalized),
        FftSmooth(source_col=source_col, period=64, normalized=normalized),
        KamaSmooth(source_col=source_col, period=10, normalized=normalized),
        AlmaSmooth(source_col=source_col, period=9, normalized=normalized),
        JmaSmooth(source_col=source_col, period=7, normalized=normalized),
        VidyaSmooth(source_col=source_col, period=14, normalized=normalized),
        T3Smooth(source_col=source_col, period=5, normalized=normalized),
        ZlmaSmooth(source_col=source_col, period=20, normalized=normalized),
        McGinleySmooth(source_col=source_col, period=20, normalized=normalized),
        FramaSmooth(source_col=source_col, period=20, normalized=normalized),
    ]


def price_transforms_pipe() -> list[Feature]:
    """All price transform indicators.

    Includes: HL2, HLC3, OHLC4, WCP, Midpoint, Midprice, Typical.
    """
    return [
        Hl2Price(),
        Hlc3Price(),
        Ohlc4Price(),
        WcpPrice(),
        MidpointPrice(),
        MidpricePrice(),
        TypicalPrice(),
    ]


def overlap_pipe(
    *,
    source_col: str = "close",
    normalized: bool = False,
) -> list[Feature]:
    """All overlap indicators (smoothers + price transforms)."""
    return [
        *smoothers_pipe(source_col=source_col, normalized=normalized),
        *price_transforms_pipe(),
    ]

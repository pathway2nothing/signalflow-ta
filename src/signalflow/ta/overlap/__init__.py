"""Overlap indicators - smoothers, price transforms, trend followers.

Modules:
    smoothers - Basic moving averages (SMA, EMA, WMA, RMA, HMA, DEMA, TEMA)
    adaptive - Adaptive smoothers (KAMA, ALMA, JMA, VIDYA, T3, ZLMA, McGinley)
    price - Price transforms and midpoints (HL2, HLC3, OHLC4, WCP, Midpoint)
    trend - Trend-following overlays (Supertrend, HiLo, Ichimoku)
"""

from signalflow.ta.overlap.smoothers import (
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
)
from signalflow.ta.overlap.adaptive import (
    KamaSmooth,
    AlmaSmooth,
    JmaSmooth,
    VidyaSmooth,
    T3Smooth,
    ZlmaSmooth,
    McGinleySmooth,
    FramaSmooth,
)
from signalflow.ta.overlap.price import (
    Hl2Price,
    Hlc3Price,
    Ohlc4Price,
    WcpPrice,
    MidpointPrice,
    MidpricePrice,
    TypicalPrice,
)
# from signalflow.ta.overlap.trend import (
#     SupertrendOverlay,
#     HiloOverlay,
#     IchimokuOverlay,
#     ChandelierOverlay,
# )

__all__ = [
    # Basic Smoothers
    "SmaSmooth",
    "EmaSmooth",
    "WmaSmooth",
    "RmaSmooth",
    "DemaSmooth",
    "TemaSmooth",
    "HmaSmooth",
    "TrimaSmooth",
    "SwmaSmooth",
    "SsfSmooth",
    "FftSmooth",
    # Adaptive Smoothers
    "KamaSmooth",
    "AlmaSmooth",
    "JmaSmooth",
    "VidyaSmooth",
    "T3Smooth",
    "ZlmaSmooth",
    "McGinleySmooth",
    "FramaSmooth",
    # Price Transforms
    "Hl2Price",
    "Hlc3Price",
    "Ohlc4Price",
    "WcpPrice",
    "MidpointPrice",
    "MidpricePrice",
    "TypicalPrice",
    # Trend Overlays
    # "SupertrendOverlay",
    # "HiloOverlay",
    # "IchimokuOverlay",
    # "ChandelierOverlay",
]
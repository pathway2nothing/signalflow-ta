"""Overlap indicators - smoothers, price transforms, trend followers.

Modules:
    smoothers - Basic moving averages (SMA, EMA, WMA, RMA, HMA, DEMA, TEMA)
    adaptive - Adaptive smoothers (KAMA, ALMA, JMA, VIDYA, T3, ZLMA, McGinley, Kalman)
    price - Price transforms and midpoints (HL2, HLC3, OHLC4, WCP, Midpoint)
    trend - Trend-following overlays (Supertrend, HiLo, Ichimoku)
"""

from signalflow.ta.overlap.adaptive import (
    AlmaSmooth,
    FramaSmooth,
    JmaSmooth,
    KalmanSmooth,
    KamaSmooth,
    McGinleySmooth,
    T3Smooth,
    VidyaSmooth,
    ZlmaSmooth,
)
from signalflow.ta.overlap.price import (
    Hl2Price,
    Hlc3Price,
    MidpointPrice,
    MidpricePrice,
    Ohlc4Price,
    TypicalPrice,
    WcpPrice,
)
from signalflow.ta.overlap.smoothers import (
    DemaSmooth,
    EmaSmooth,
    FftSmooth,
    HmaSmooth,
    RmaSmooth,
    SmaSmooth,
    SsfSmooth,
    SwmaSmooth,
    TemaSmooth,
    TrimaSmooth,
    WmaSmooth,
)

# from signalflow.ta.overlap.trend import (
#     SupertrendOverlay,
#     HiloOverlay,
#     IchimokuOverlay,
#     ChandelierOverlay,
# )

__all__ = [
    "AlmaSmooth",
    "DemaSmooth",
    "EmaSmooth",
    "FftSmooth",
    "FramaSmooth",
    # Price Transforms
    "Hl2Price",
    "Hlc3Price",
    "HmaSmooth",
    "JmaSmooth",
    "KalmanSmooth",
    # Adaptive Smoothers
    "KamaSmooth",
    "McGinleySmooth",
    "MidpointPrice",
    "MidpricePrice",
    "Ohlc4Price",
    "RmaSmooth",
    # Basic Smoothers
    "SmaSmooth",
    "SsfSmooth",
    "SwmaSmooth",
    "T3Smooth",
    "TemaSmooth",
    "TrimaSmooth",
    "TypicalPrice",
    "VidyaSmooth",
    "WcpPrice",
    "WmaSmooth",
    "ZlmaSmooth",
    # Trend Overlays
    # "SupertrendOverlay",
    # "HiloOverlay",
    # "IchimokuOverlay",
    # "ChandelierOverlay",
]

"""Filter-error features: close minus various smoothers/filters.

All members measure `close - filter(close)` for different filter types.
The error series is a stationary residual capturing what the filter missed.
Useful as direct features (in mean-reversion strategies) or as PID-style control errors.
"""
from signalflow.ta.filters.smoother_errors import (
    AdaptiveEMAError,
    DEMAError,
    HMAError,
    TEMAError,
)
from signalflow.ta.filters.kalman import KalmanResidual
from signalflow.ta.filters.pid import PIDDerivativeTerm, PIDIntegralTerm

__all__ = [
    "AdaptiveEMAError",
    "DEMAError",
    "HMAError",
    "TEMAError",
    "KalmanResidual",
    "PIDDerivativeTerm",
    "PIDIntegralTerm",
]

"""Preset pipes for divergence detectors."""
from __future__ import annotations

from signalflow.feature.base import Feature
from signalflow.ta.divergence import RsiDivergence, MacdDivergence


def divergence_pipe() -> list[Feature]:
    """Divergence detectors: RSI Divergence, MACD Divergence."""
    return [
        RsiDivergence(),
        MacdDivergence(),
    ]

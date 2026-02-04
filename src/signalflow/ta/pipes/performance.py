"""Preset pipes for performance (returns) indicators."""
from __future__ import annotations

from signalflow.feature.base import Feature
from signalflow.ta.performance import LogReturn, PctReturn


def performance_pipe(*, source_col: str = "close") -> list[Feature]:
    """Returns indicators: LogReturn, PctReturn."""
    return [
        LogReturn(source=source_col),
        PctReturn(source=source_col),
    ]

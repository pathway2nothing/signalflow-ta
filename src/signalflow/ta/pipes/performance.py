"""Preset pipes for performance (returns) indicators."""


from signalflow.ta._compat import Feature
from signalflow.ta.performance import LogReturn, PctReturn


def performance_pipe(*, source_col: str = "close") -> list[Feature]:
    """Returns indicators: LogReturn, PctReturn."""
    return [
        LogReturn(source=source_col),
        PctReturn(source=source_col),
    ]

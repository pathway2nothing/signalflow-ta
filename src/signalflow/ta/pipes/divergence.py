"""Preset pipes for divergence detectors."""


from signalflow.ta._compat import Feature
from signalflow.ta.divergence import MacdDivergence, RsiDivergence


def divergence_pipe() -> list[Feature]:
    """Divergence detectors: RSI Divergence, MACD Divergence."""
    return [
        RsiDivergence(),
        MacdDivergence(),
    ]

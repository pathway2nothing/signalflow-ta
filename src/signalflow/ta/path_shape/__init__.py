"""Path-shape descriptors.

Captures geometric properties of the price path: roughness, efficiency,
streaks, entropy, autocorrelation. Distinct from level/momentum/volatility - measures
HOW the price moves through a window, not where it ended up.
"""
from signalflow.ta.path_shape.shape import (
    PathEfficiency,
    PathRoughness,
    PathSimplicity,
    PathTortuosity,
)
from signalflow.ta.path_shape.streaks import (
    LongestStreak,
    MaxConsecutiveGainRun,
    MaxConsecutiveLossRun,
    ReversalCount,
    ZeroCrossingRate,
)
from signalflow.ta.path_shape.entropy import (
    DirectionalEntropy,
    ReturnSignEntropy,
    VolumeEntropy,
)
from signalflow.ta.path_shape.autocorr import (
    ErrorAutoCorrelation,
    ReturnAutocorrShort,
    VolatilityClusterScore,
)

__all__ = [
    "PathEfficiency",
    "PathRoughness",
    "PathSimplicity",
    "PathTortuosity",
    "LongestStreak",
    "MaxConsecutiveGainRun",
    "MaxConsecutiveLossRun",
    "ReversalCount",
    "ZeroCrossingRate",
    "DirectionalEntropy",
    "ReturnSignEntropy",
    "VolumeEntropy",
    "ErrorAutoCorrelation",
    "ReturnAutocorrShort",
    "VolatilityClusterScore",
]

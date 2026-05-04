"""Concrete signal feature implementations for signalflow-ta.

All signal features operate on the signal stream rather than raw
market data. They split into two categories:

- **Unsupervised**: computed from signal history alone.
- **Supervised**: require resolved outcome labels (look-ahead safe).
"""

from signalflow.ta.signal_features.accuracy import (
    FalseSignalRate,
    RollingAccuracy,
    TypeConditionalAccuracy,
)
from signalflow.ta.signal_features.adaptive import (
    AdaptiveConfidence,
    DrawdownSensitivity,
    SignalClusterQuality,
    SignalFragility,
)
from signalflow.ta.signal_features.context import (
    MomentumAlignment,
    RegimeSensitivity,
    VolatilityAdjustedEV,
)
from signalflow.ta.signal_features.cross import (
    CrossPairSpillover,
    SignalCrowding,
    SignalDisagreement,
)
from signalflow.ta.signal_features.frequency import (
    InterSignalDistance,
    SignalFrequency,
)
from signalflow.ta.signal_features.information import (
    BayesianSurpriseRate,
    MutualInformation,
    SignalSurprise,
)
from signalflow.ta.signal_features.outcome import OutcomeStreak
from signalflow.ta.signal_features.performance import (
    InformationCoefficient,
    RollingExpectedValue,
    RollingProfitFactor,
)
from signalflow.ta.signal_features.probability import (
    BayesianPosterior,
    CalibrationError,
    ProbabilityMoments,
)
from signalflow.ta.signal_features.stability import (
    SignalEntropy,
    SignalFlipRate,
    SignalStreak,
    SignalTypeRatio,
)
from signalflow.ta.signal_features.temporal import (
    SignalAlphaDecay,
    SignalLifetime,
    TemporalBias,
)

__all__ = [
    "AdaptiveConfidence",
    "BayesianPosterior",
    "BayesianSurpriseRate",
    "CalibrationError",
    "CrossPairSpillover",
    "DrawdownSensitivity",
    "FalseSignalRate",
    "InformationCoefficient",
    "InterSignalDistance",
    "MomentumAlignment",
    "MutualInformation",
    "OutcomeStreak",
    "ProbabilityMoments",
    "RegimeSensitivity",
    "RollingAccuracy",
    "RollingExpectedValue",
    "RollingProfitFactor",
    "SignalAlphaDecay",
    "SignalClusterQuality",
    "SignalCrowding",
    "SignalDisagreement",
    "SignalEntropy",
    "SignalFlipRate",
    "SignalFragility",
    "SignalFrequency",
    "SignalLifetime",
    "SignalStreak",
    "SignalSurprise",
    "SignalTypeRatio",
    "TemporalBias",
    "TypeConditionalAccuracy",
    "VolatilityAdjustedEV",
]

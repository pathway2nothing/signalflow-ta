# src/signalflow/ta/stat/__init__.py
"""Statistical indicators (auxiliary features)."""

from signalflow.ta.stat.complexity import (
    DfaExponentStat,
    FisherInformationStat,
    LempelZivStat,
    PermutationEntropyStat,
    SampleEntropyStat,
)
from signalflow.ta.stat.control import (
    ARCoefficientStat,
    KalmanInnovationStat,
    LyapunovExponentStat,
    PIDErrorStat,
    PredictionErrorDecompositionStat,
)
from signalflow.ta.stat.cross_sectional import (
    CrossSectionalStat,
)
from signalflow.ta.stat.cycle import (
    BeatFrequencyStat,
    ConstructiveInterferenceStat,
    InstAmplitudeStat,
    InstFrequencyStat,
    InstPhaseStat,
    PhaseAccelerationStat,
    SpectralCentroidStat,
    SpectralEntropyStat,
    StandingWaveRatioStat,
)
from signalflow.ta.stat.dispersion import (
    AadStat,
    CvStat,
    IqrStat,
    MadStat,
    RangeStat,
    RobustZscoreStat,
    StdStat,
    VarianceStat,
    ZscoreStat,
)
from signalflow.ta.stat.distribution import (
    AboveMeanRatioStat,
    EntropyRateStat,
    EntropyStat,
    JarqueBeraStat,
    KurtosisStat,
    MedianStat,
    MinMaxStat,
    ModeDistanceStat,
    PctRankStat,
    QuantileStat,
    SkewStat,
)
from signalflow.ta.stat.dsp import (
    MFCCBandEnergyStat,
    PowerCepstrumStat,
    SpectralBandwidthStat,
    SpectralContrastStat,
    SpectralFlatnessStat,
    SpectralFluxStat,
    SpectralKurtosisStat,
    SpectralRolloffStat,
    SpectralSlopeStat,
    ZeroCrossingRateStat,
)
from signalflow.ta.stat.information import (
    AutoMutualInfoStat,
    JSDivergenceStat,
    KLDivergenceStat,
    RelativeInfoGainStat,
    RenyiEntropyStat,
)
from signalflow.ta.stat.memory import (
    AnomalousDiffusionStat,
    AutocorrStat,
    CorrelationLengthStat,
    DampingRatioStat,
    DiffusionCoeffStat,
    EscapeVelocityStat,
    HurstStat,
    MsdRatioStat,
    NaturalFrequencyStat,
    PlasticStrainStat,
    SpringConstantStat,
    VarianceRatioStat,
)
from signalflow.ta.stat.realized import (
    GarmanKlassVolStat,
    ParkinsonVolStat,
    RealizedVolStat,
    RogersSatchellVolStat,
    YangZhangVolStat,
)
from signalflow.ta.stat.regression import (
    BetaStat,
    CorrelationStat,
    LinRegInterceptStat,
    LinRegResidualStat,
    LinRegSlopeStat,
    RSquaredStat,
)
from signalflow.ta.stat.structure import (
    ReversePointsStat,
    RollingMaxStat,
    RollingMinStat,
    TimeSinceSpikeStat,
    VolatilitySpikeDiffStat,
    VolatilitySpikeStat,
    VolumeSpikeDiffStat,
    VolumeSpikeStat,
)

__all__ = [
    "ARCoefficientStat",
    "AadStat",
    "AboveMeanRatioStat",
    "AnomalousDiffusionStat",
    "AutoMutualInfoStat",
    "AutocorrStat",
    "BeatFrequencyStat",
    "BetaStat",
    # Wave interference & Spectral
    "ConstructiveInterferenceStat",
    "CorrelationLengthStat",
    # Regression
    "CorrelationStat",
    # Cross-Sectional
    "CrossSectionalStat",
    "CvStat",
    "DampingRatioStat",
    "DfaExponentStat",
    # Diffusion
    "DiffusionCoeffStat",
    # Distribution (extended)
    "EntropyRateStat",
    "EntropyStat",
    "EscapeVelocityStat",
    "FisherInformationStat",
    "GarmanKlassVolStat",
    # Memory
    "HurstStat",
    # Cycle analysis
    "InstAmplitudeStat",
    "InstFrequencyStat",
    "InstPhaseStat",
    "IqrStat",
    "JSDivergenceStat",
    "JarqueBeraStat",
    # Information Theory & Information Geometry
    "KLDivergenceStat",
    # Control Theory & Systems Engineering
    "KalmanInnovationStat",
    "KurtosisStat",
    "LempelZivStat",
    "LinRegInterceptStat",
    "LinRegResidualStat",
    "LinRegSlopeStat",
    "LyapunovExponentStat",
    "MFCCBandEnergyStat",
    "MadStat",
    # Distribution
    "MedianStat",
    "MinMaxStat",
    "ModeDistanceStat",
    "MsdRatioStat",
    "NaturalFrequencyStat",
    "PIDErrorStat",
    "ParkinsonVolStat",
    "PctRankStat",
    # Complexity & Information Theory
    "PermutationEntropyStat",
    "PhaseAccelerationStat",
    # Elasticity & Escape
    "PlasticStrainStat",
    "PowerCepstrumStat",
    "PredictionErrorDecompositionStat",
    "QuantileStat",
    "RSquaredStat",
    "RangeStat",
    # Realized Volatility
    "RealizedVolStat",
    "RelativeInfoGainStat",
    "RenyiEntropyStat",
    # Structure & Spikes
    "ReversePointsStat",
    "RobustZscoreStat",
    "RogersSatchellVolStat",
    "RollingMaxStat",
    "RollingMinStat",
    "SampleEntropyStat",
    "SkewStat",
    "SpectralBandwidthStat",
    "SpectralCentroidStat",
    "SpectralContrastStat",
    "SpectralEntropyStat",
    "SpectralFlatnessStat",
    # DSP / Acoustics
    "SpectralFluxStat",
    "SpectralKurtosisStat",
    "SpectralRolloffStat",
    "SpectralSlopeStat",
    # Oscillator dynamics
    "SpringConstantStat",
    "StandingWaveRatioStat",
    "StdStat",
    "TimeSinceSpikeStat",
    "VarianceRatioStat",
    # Dispersion
    "VarianceStat",
    "VolatilitySpikeDiffStat",
    "VolatilitySpikeStat",
    "VolumeSpikeDiffStat",
    "VolumeSpikeStat",
    "YangZhangVolStat",
    "ZeroCrossingRateStat",
    "ZscoreStat",
]

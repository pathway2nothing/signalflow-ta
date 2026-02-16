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
    # Dispersion
    "VarianceStat",
    "StdStat",
    "MadStat",
    "ZscoreStat",
    "CvStat",
    "RangeStat",
    "IqrStat",
    "AadStat",
    "RobustZscoreStat",
    # Distribution
    "MedianStat",
    "QuantileStat",
    "PctRankStat",
    "MinMaxStat",
    "SkewStat",
    "KurtosisStat",
    "EntropyStat",
    "JarqueBeraStat",
    "ModeDistanceStat",
    "AboveMeanRatioStat",
    # Memory
    "HurstStat",
    "AutocorrStat",
    "VarianceRatioStat",
    # Diffusion
    "DiffusionCoeffStat",
    "AnomalousDiffusionStat",
    "MsdRatioStat",
    # Oscillator dynamics
    "SpringConstantStat",
    "DampingRatioStat",
    "NaturalFrequencyStat",
    # Distribution (extended)
    "EntropyRateStat",
    # Elasticity & Escape
    "PlasticStrainStat",
    "EscapeVelocityStat",
    "CorrelationLengthStat",
    # Cycle analysis
    "InstAmplitudeStat",
    "InstPhaseStat",
    "InstFrequencyStat",
    "PhaseAccelerationStat",
    # Wave interference & Spectral
    "ConstructiveInterferenceStat",
    "BeatFrequencyStat",
    "StandingWaveRatioStat",
    "SpectralCentroidStat",
    "SpectralEntropyStat",
    # Regression
    "CorrelationStat",
    "BetaStat",
    "RSquaredStat",
    "LinRegSlopeStat",
    "LinRegInterceptStat",
    "LinRegResidualStat",
    # Realized Volatility
    "RealizedVolStat",
    "ParkinsonVolStat",
    "GarmanKlassVolStat",
    "RogersSatchellVolStat",
    "YangZhangVolStat",
    # Complexity & Information Theory
    "PermutationEntropyStat",
    "SampleEntropyStat",
    "LempelZivStat",
    "FisherInformationStat",
    "DfaExponentStat",
    # Information Theory & Information Geometry
    "KLDivergenceStat",
    "JSDivergenceStat",
    "RenyiEntropyStat",
    "AutoMutualInfoStat",
    "RelativeInfoGainStat",
    # Cross-Sectional
    "CrossSectionalStat",
    # DSP / Acoustics
    "SpectralFluxStat",
    "ZeroCrossingRateStat",
    "SpectralRolloffStat",
    "SpectralFlatnessStat",
    "PowerCepstrumStat",
    "SpectralBandwidthStat",
    "SpectralSlopeStat",
    "SpectralKurtosisStat",
    "SpectralContrastStat",
    "MFCCBandEnergyStat",
    # Control Theory & Systems Engineering
    "KalmanInnovationStat",
    "ARCoefficientStat",
    "LyapunovExponentStat",
    "PIDErrorStat",
    "PredictionErrorDecompositionStat",
    # Structure & Spikes
    "ReversePointsStat",
    "TimeSinceSpikeStat",
    "VolatilitySpikeStat",
    "VolatilitySpikeDiffStat",
    "VolumeSpikeStat",
    "VolumeSpikeDiffStat",
    "RollingMinStat",
    "RollingMaxStat",
]

# src/signalflow/ta/stat/__init__.py
"""Statistical indicators (auxiliary features)."""

from signalflow.ta.stat.dispersion import (
    VarianceStat,
    StdStat,
    MadStat,
    ZscoreStat,
    CvStat,
    RangeStat,
    IqrStat,
    AadStat,
    RobustZscoreStat,
)
from signalflow.ta.stat.distribution import (
    MedianStat,
    QuantileStat,
    PctRankStat,
    MinMaxStat,
    SkewStat,
    KurtosisStat,
    EntropyStat,
    JarqueBeraStat,
    ModeDistanceStat,
    AboveMeanRatioStat,
    EntropyRateStat,
)
from signalflow.ta.stat.memory import (
    HurstStat,
    AutocorrStat,
    VarianceRatioStat,
    DiffusionCoeffStat,
    AnomalousDiffusionStat,
    MsdRatioStat,
    SpringConstantStat,
    DampingRatioStat,
    NaturalFrequencyStat,
    PlasticStrainStat,
    EscapeVelocityStat,
    CorrelationLengthStat,
)
from signalflow.ta.stat.cycle import (
    InstAmplitudeStat,
    InstPhaseStat,
    InstFrequencyStat,
    PhaseAccelerationStat,
    ConstructiveInterferenceStat,
    BeatFrequencyStat,
    StandingWaveRatioStat,
    SpectralCentroidStat,
    SpectralEntropyStat,
)
from signalflow.ta.stat.complexity import (
    PermutationEntropyStat,
    SampleEntropyStat,
    LempelZivStat,
    FisherInformationStat,
    DfaExponentStat,
)
from signalflow.ta.stat.information import (
    KLDivergenceStat,
    JSDivergenceStat,
    RenyiEntropyStat,
    AutoMutualInfoStat,
    RelativeInfoGainStat,
)
from signalflow.ta.stat.cross_sectional import (
    CrossSectionalStat,
)
from signalflow.ta.stat.dsp import (
    SpectralFluxStat,
    ZeroCrossingRateStat,
    SpectralRolloffStat,
    SpectralFlatnessStat,
    PowerCepstrumStat,
    SpectralBandwidthStat,
    SpectralSlopeStat,
    SpectralKurtosisStat,
    SpectralContrastStat,
    MFCCBandEnergyStat,
)
from signalflow.ta.stat.regression import (
    CorrelationStat,
    BetaStat,
    RSquaredStat,
    LinRegSlopeStat,
    LinRegInterceptStat,
    LinRegResidualStat,
)
from signalflow.ta.stat.realized import (
    RealizedVolStat,
    ParkinsonVolStat,
    GarmanKlassVolStat,
    RogersSatchellVolStat,
    YangZhangVolStat,
)
from signalflow.ta.stat.control import (
    KalmanInnovationStat,
    ARCoefficientStat,
    LyapunovExponentStat,
    PIDErrorStat,
    PredictionErrorDecompositionStat,
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
]

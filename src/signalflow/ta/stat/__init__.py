# src/signalflow/ta/stat/__init__.py
"""Statistical indicators (auxiliary features)."""

from signalflow.ta.stat.aftershock import (
    ForeshockBuildupStat,
    GutenbergRichterBValueStat,
    OmoriAftershockRateStat,
    OmoriRateBroadStat,
)
from signalflow.ta.stat.allostatic import (
    AllostaticFastSlowRatioStat,
    AllostaticLoadDirectionalStat,
    AllostaticLoadStat,
    CumZScorePowerLawStat,
    DownsideZScoreEMAStat,
    VolWeightedZScoreEMAStat,
)
from signalflow.ta.stat.asymmetric_stretch import AsymmetricSemiVolStretchStat
from signalflow.ta.stat.changepoint import FocusMaxStatStat
from signalflow.ta.stat.fractal_dim import KatzFractalDimensionStat
from signalflow.ta.stat.brownian_bridge import (
    BBPathRoughnessStat,
    BBTensionDirectionalStat,
    BrownianBridgeTensionStat,
    SwingAmpDisplacementStat,
)
from signalflow.ta.stat.hawkes import (
    HawkesSelfExcitationStat,
    HawkesSignedJumpsStat,
    HawkesVolClimaxIntensityStat,
    MarkedHawkesJumpsStat,
    PowerLawHawkesStat,
)
from signalflow.ta.stat.impact_decay import AlmgrenChrissImpactDecayStat
from signalflow.ta.stat.kl_drift import KullbackLeiblerDriftStat
from signalflow.ta.stat.microstructure_proxies import (
    BidAskSpreadProxyEMAStat,
    FisherInformationReturnsStat,
    KalmanInnovationVolStat,
    OFISignedProxyStat,
    VolWeightedBBWidthStat,
    VwapParkinsonStretchStat,
)
from signalflow.ta.stat.realized_decomposition import (
    JumpTruncatedVarianceRatioStat,
    PreAveragedBipowerVariationStat,
    RealisedBipowerVarianceRatioStat,
    RealisedSemivarianceRatioStat,
    RealizedQuarticityStat,
    RVSemivarianceAsymmetryStat,
)
from signalflow.ta.stat.regression_extensions import (
    LinRegFitQuality,
    LinRegInterceptNormalized,
    LinRegR2,
    LinRegResidualKurtosis,
    LinRegResidualSkew,
    LinRegResidualStd,
    LinRegSlopeAcceleration,
    LinRegSlopeChange,
    LinRegSlopeNormalized,
    LinRegSlopeRatio,
    LinRegSlopeWindow,
    Poly2ResidualStd,
)
from signalflow.ta.stat.distribution_extensions import (
    ReturnKurtosisWindow,
    ReturnSkewWindow,
    ReturnTailRatio,
)
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
    # Regression extensions (sf-profit iter-15/18/20)
    "LinRegFitQuality",
    "LinRegInterceptNormalized",
    "LinRegR2",
    "LinRegResidualKurtosis",
    "LinRegResidualSkew",
    "LinRegResidualStd",
    "LinRegSlopeAcceleration",
    "LinRegSlopeChange",
    "LinRegSlopeNormalized",
    "LinRegSlopeRatio",
    "LinRegSlopeWindow",
    "Poly2ResidualStd",
    # Distribution / tail extensions (sf-profit iter-20)
    "ReturnKurtosisWindow",
    "ReturnSkewWindow",
    "ReturnTailRatio",
    # Cross-disciplinary (sf-profit iter-27/28/29)
    "AllostaticFastSlowRatioStat",
    "AllostaticLoadDirectionalStat",
    "AllostaticLoadStat",
    "AlmgrenChrissImpactDecayStat",
    "AsymmetricSemiVolStretchStat",
    "BBPathRoughnessStat",
    "BBTensionDirectionalStat",
    "BidAskSpreadProxyEMAStat",
    "BrownianBridgeTensionStat",
    "CumZScorePowerLawStat",
    "DownsideZScoreEMAStat",
    "FisherInformationReturnsStat",
    "FocusMaxStatStat",
    "ForeshockBuildupStat",
    "GutenbergRichterBValueStat",
    "HawkesSelfExcitationStat",
    "HawkesSignedJumpsStat",
    "HawkesVolClimaxIntensityStat",
    "JumpTruncatedVarianceRatioStat",
    "KalmanInnovationVolStat",
    "KatzFractalDimensionStat",
    "KullbackLeiblerDriftStat",
    "MarkedHawkesJumpsStat",
    "OFISignedProxyStat",
    "OmoriAftershockRateStat",
    "OmoriRateBroadStat",
    "PowerLawHawkesStat",
    "PreAveragedBipowerVariationStat",
    "RealisedBipowerVarianceRatioStat",
    "RealisedSemivarianceRatioStat",
    "RealizedQuarticityStat",
    "RVSemivarianceAsymmetryStat",
    "SwingAmpDisplacementStat",
    "VolWeightedBBWidthStat",
    "VolWeightedZScoreEMAStat",
    "VwapParkinsonStretchStat",
]

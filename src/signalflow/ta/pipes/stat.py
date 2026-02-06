"""Preset pipes for statistical indicators."""

from __future__ import annotations

from signalflow.feature.base import Feature
from signalflow.ta.stat import (
    # Dispersion
    VarianceStat,
    StdStat,
    MadStat,
    ZscoreStat,
    CvStat,
    RangeStat,
    IqrStat,
    AadStat,
    RobustZscoreStat,
    # Distribution
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
    # Memory & Diffusion
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
    # Cycle
    InstAmplitudeStat,
    InstPhaseStat,
    InstFrequencyStat,
    PhaseAccelerationStat,
    ConstructiveInterferenceStat,
    BeatFrequencyStat,
    StandingWaveRatioStat,
    SpectralCentroidStat,
    SpectralEntropyStat,
    # Regression
    RSquaredStat,
    LinRegSlopeStat,
    LinRegInterceptStat,
    LinRegResidualStat,
    # Realized Volatility
    RealizedVolStat,
    ParkinsonVolStat,
    GarmanKlassVolStat,
    RogersSatchellVolStat,
    YangZhangVolStat,
    # Complexity
    PermutationEntropyStat,
    SampleEntropyStat,
    LempelZivStat,
    FisherInformationStat,
    DfaExponentStat,
    # Information Theory
    KLDivergenceStat,
    JSDivergenceStat,
    RenyiEntropyStat,
    AutoMutualInfoStat,
    RelativeInfoGainStat,
    # DSP / Acoustics
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


def stat_dispersion_pipe(*, source_col: str = "close") -> list[Feature]:
    """Dispersion statistics: Variance, Std, MAD, Z-score, CV, Range, IQR, AAD, Robust Z."""
    return [
        VarianceStat(source_col=source_col, period=30),
        StdStat(source_col=source_col, period=30),
        MadStat(source_col=source_col, period=30),
        ZscoreStat(source_col=source_col, period=30),
        CvStat(source_col=source_col, period=30),
        RangeStat(source_col=source_col, period=30),
        IqrStat(source_col=source_col, period=30),
        AadStat(source_col=source_col, period=30),
        RobustZscoreStat(source_col=source_col, period=30),
    ]


def stat_distribution_pipe(*, source_col: str = "close") -> list[Feature]:
    """Distribution statistics: Median, Quantile, PctRank, MinMax, Skew, Kurtosis, Entropy, etc."""
    return [
        MedianStat(source_col=source_col, period=30),
        QuantileStat(source_col=source_col, period=30),
        PctRankStat(source_col=source_col, period=30),
        MinMaxStat(source_col=source_col, period=30),
        SkewStat(source_col=source_col, period=30),
        KurtosisStat(source_col=source_col, period=30),
        EntropyStat(source_col=source_col, period=10),
        JarqueBeraStat(source_col=source_col, period=30),
        ModeDistanceStat(source_col=source_col, period=30),
        AboveMeanRatioStat(source_col=source_col, period=30),
        EntropyRateStat(source_col=source_col, period=10),
    ]


def stat_memory_pipe(*, source_col: str = "close") -> list[Feature]:
    """Memory, diffusion & oscillator dynamics.

    Includes: Hurst, Autocorrelation, Variance Ratio, Diffusion, MSD,
    Spring Constant, Damping Ratio, Natural Frequency,
    Plastic Strain, Escape Velocity, Correlation Length.
    """
    return [
        HurstStat(source_col=source_col, period=100),
        AutocorrStat(source_col=source_col, period=30),
        VarianceRatioStat(source_col=source_col, period=50),
        DiffusionCoeffStat(source_col=source_col, period=30),
        AnomalousDiffusionStat(source_col=source_col, period=60),
        MsdRatioStat(source_col=source_col, period=60),
        SpringConstantStat(source_col=source_col, period=60),
        DampingRatioStat(source_col=source_col, period=60),
        NaturalFrequencyStat(source_col=source_col, period=60),
        PlasticStrainStat(source_col=source_col, period=60),
        EscapeVelocityStat(source_col=source_col, period=60),
        CorrelationLengthStat(source_col=source_col, period=100),
    ]


def stat_cycle_pipe(*, source_col: str = "close") -> list[Feature]:
    """Cycle analysis & wave interference.

    Includes: Instantaneous Amplitude/Phase/Frequency, Phase Acceleration,
    Constructive Interference, Beat Frequency, Standing Wave Ratio,
    Spectral Centroid, Spectral Entropy.
    """
    return [
        InstAmplitudeStat(source_col=source_col),
        InstPhaseStat(source_col=source_col),
        InstFrequencyStat(source_col=source_col),
        PhaseAccelerationStat(source_col=source_col),
        ConstructiveInterferenceStat(source_col=source_col),
        BeatFrequencyStat(source_col=source_col),
        StandingWaveRatioStat(source_col=source_col),
        SpectralCentroidStat(source_col=source_col),
        SpectralEntropyStat(source_col=source_col),
    ]


def stat_regression_pipe(*, source_col: str = "close") -> list[Feature]:
    """Regression statistics: R-squared, LinReg Slope/Intercept/Residual.

    Note: CorrelationStat and BetaStat require a second column (target_col/benchmark_col)
    and are not included here. Add them manually if needed.
    """
    return [
        RSquaredStat(source_col=source_col, period=30),
        LinRegSlopeStat(source_col=source_col, period=30),
        LinRegInterceptStat(source_col=source_col, period=30),
        LinRegResidualStat(source_col=source_col, period=30),
    ]


def stat_realized_vol_pipe() -> list[Feature]:
    """Realized volatility estimators: Close-to-close, Parkinson, Garman-Klass, Rogers-Satchell, Yang-Zhang.

    These use OHLC columns directly and don't accept source_col.
    """
    return [
        RealizedVolStat(period=20),
        ParkinsonVolStat(period=20),
        GarmanKlassVolStat(period=20),
        RogersSatchellVolStat(period=20),
        YangZhangVolStat(period=20),
    ]


def stat_complexity_pipe(*, source_col: str = "close") -> list[Feature]:
    """Complexity & information theory: Permutation Entropy, Sample Entropy, Lempel-Ziv, Fisher Information, DFA."""
    return [
        PermutationEntropyStat(source_col=source_col),
        SampleEntropyStat(source_col=source_col),
        LempelZivStat(source_col=source_col),
        FisherInformationStat(source_col=source_col),
        DfaExponentStat(source_col=source_col),
    ]


def stat_info_theory_pipe(*, source_col: str = "close") -> list[Feature]:
    """Information geometry: KL Divergence, JS Divergence, Renyi Entropy, Auto MI, Relative Info Gain."""
    return [
        KLDivergenceStat(source_col=source_col),
        JSDivergenceStat(source_col=source_col),
        RenyiEntropyStat(source_col=source_col),
        AutoMutualInfoStat(source_col=source_col),
        RelativeInfoGainStat(source_col=source_col),
    ]


def stat_dsp_pipe(*, source_col: str = "close") -> list[Feature]:
    """DSP / Acoustics: spectral dynamics, shape, and texture (10 indicators)."""
    return [
        SpectralFluxStat(source_col=source_col),
        ZeroCrossingRateStat(source_col=source_col),
        SpectralRolloffStat(source_col=source_col),
        SpectralFlatnessStat(source_col=source_col),
        PowerCepstrumStat(source_col=source_col),
        SpectralBandwidthStat(source_col=source_col),
        SpectralSlopeStat(source_col=source_col),
        SpectralKurtosisStat(source_col=source_col),
        SpectralContrastStat(source_col=source_col),
        MFCCBandEnergyStat(source_col=source_col),
    ]


def stat_pipe(*, source_col: str = "close") -> list[Feature]:
    """All statistical indicators (excluding CrossSectionalStat which is a GlobalFeature)."""
    return [
        *stat_dispersion_pipe(source_col=source_col),
        *stat_distribution_pipe(source_col=source_col),
        *stat_memory_pipe(source_col=source_col),
        *stat_cycle_pipe(source_col=source_col),
        *stat_regression_pipe(source_col=source_col),
        *stat_realized_vol_pipe(),
        *stat_complexity_pipe(source_col=source_col),
        *stat_info_theory_pipe(source_col=source_col),
        *stat_dsp_pipe(source_col=source_col),
    ]

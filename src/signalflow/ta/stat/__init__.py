# src/signalflow/ta/stat/__init__.py
"""Statistical indicators (auxiliary features)."""

from signalflow.ta.stat.dispersion import (
    VarianceStat,
    StdevStat,
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
)
from signalflow.ta.stat.memory import (
    HurstStat,
    AutocorrStat,
    VarianceRatioStat,
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

__all__ = [
    # Dispersion
    "VarianceStat",
    "StdevStat",
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
]
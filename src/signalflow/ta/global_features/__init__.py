"""Global (cross-sectional) features for market-wide indicators."""

from signalflow.ta.global_features.market import (
    MarketIndexFeature,
    MarketRollingMinFeature,
    MarketRsiFeature,
    MarketVolatilityFeature,
    MarketZscoreFeature,
    compute_global_features,
)
from signalflow.ta.global_features.cross_sectional import (
    AvgPairwiseCorrMarket,
    CrossSectionalAdxRank,
    CrossSectionalAtrRank,
    CrossSectionalBeta,
    CrossSectionalDispersion,
    CrossSectionalRangeRank,
    CrossSectionalRetSkew,
    CrossSectionalReturnAccelRank,
    CrossSectionalReturnRank,
    CrossSectionalRsiRank,
    CrossSectionalVolRank,
    DivergenceFromMarketMedian,
    MarketBreadth,
    PairExcessReturn,
    PairLeadLagCorr,
    RelativeStrengthVsMarket,
)

__all__ = [
    "MarketIndexFeature",
    "MarketRollingMinFeature",
    "MarketRsiFeature",
    "MarketVolatilityFeature",
    "MarketZscoreFeature",
    "compute_global_features",
    # Cross-sectional (sf-profit iter-15/16/18/20)
    "AvgPairwiseCorrMarket",
    "CrossSectionalAdxRank",
    "CrossSectionalAtrRank",
    "CrossSectionalBeta",
    "CrossSectionalDispersion",
    "CrossSectionalRangeRank",
    "CrossSectionalRetSkew",
    "CrossSectionalReturnAccelRank",
    "CrossSectionalReturnRank",
    "CrossSectionalRsiRank",
    "CrossSectionalVolRank",
    "DivergenceFromMarketMedian",
    "MarketBreadth",
    "PairExcessReturn",
    "PairLeadLagCorr",
    "RelativeStrengthVsMarket",
]

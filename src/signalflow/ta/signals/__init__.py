"""Signal detectors based on technical analysis indicators."""

from signalflow.ta.signals.adx_regime import (
    AdxDiCrossDetector,
    AdxRegimeRsiDetector,
)
from signalflow.ta.signals.aroon_cross import AroonCrossDetector
from signalflow.ta.signals.bollinger_band import BollingerBreakoutDetector
from signalflow.ta.signals.cci_anomaly import CciAnomalyDetector
from signalflow.ta.signals.cross_pair import CrossPairCorrBollingerDetector
from signalflow.ta.signals.divergence import (
    MacdDivergenceDetector,
    RsiDivergenceDetector,
    RsiDivergenceOffsetDetector,
)
from signalflow.ta.signals.filters import (
    AboveBBUpperFilter,
    BelowBBLowerFilter,
    CciZscoreFilter,
    HighVolatilityFilter,
    LowVolatilityFilter,
    MacdAboveSignalFilter,
    MacdBelowSignalFilter,
    MeanExtensionFilter,
    MeanReversionFilter,
    PriceDowntrendFilter,
    PriceUptrendFilter,
    RsiZscoreFilter,
    SignalFilter,
)
from signalflow.ta.signals.hampel_filter import (
    AdaptiveHampelAnomalyDetector,
    HampelAnomalyDetector,
)
from signalflow.ta.signals.isolation_forest import (
    IsoForestCrossSectionalDetector,
    IsoForestReturnsDetector,
    IsoForestRsiDetector,
)
from signalflow.ta.signals.kalman_filter import KalmanFilterDetector
from signalflow.ta.signals.keltner_channel import (
    KeltnerMacdRsiDetector,
    KeltnerRsiZscoreDetector,
)
from signalflow.ta.signals.market_condition import (
    RsiGlobalVolDetector,
    RsiVsMarketDetector,
    ZscoreRollingMinDetector,
)
from signalflow.ta.signals.mfi import (
    MfiExtremeDetector,
    MfiZscoreReversalDetector,
)
from signalflow.ta.signals.rsi_anomaly import RsiAnomalyDetector
from signalflow.ta.signals.stochastic import (
    StochasticCrossDetector,
    StochasticExtremeZscoreDetector,
)

__all__ = [
    "AboveBBUpperFilter",
    "AdaptiveHampelAnomalyDetector",
    "AdxDiCrossDetector",
    "AdxRegimeRsiDetector",
    "AroonCrossDetector",
    "BelowBBLowerFilter",
    "BollingerBreakoutDetector",
    "CciAnomalyDetector",
    "CciZscoreFilter",
    "CrossPairCorrBollingerDetector",
    "HampelAnomalyDetector",
    "HighVolatilityFilter",
    "IsoForestCrossSectionalDetector",
    "IsoForestReturnsDetector",
    "IsoForestRsiDetector",
    "KalmanFilterDetector",
    "KeltnerMacdRsiDetector",
    "KeltnerRsiZscoreDetector",
    "LowVolatilityFilter",
    "MacdAboveSignalFilter",
    "MacdBelowSignalFilter",
    "MacdDivergenceDetector",
    "MeanExtensionFilter",
    "MeanReversionFilter",
    "MfiExtremeDetector",
    "MfiZscoreReversalDetector",
    "PriceDowntrendFilter",
    "PriceUptrendFilter",
    "RsiAnomalyDetector",
    "RsiDivergenceDetector",
    "RsiDivergenceOffsetDetector",
    "RsiGlobalVolDetector",
    "RsiVsMarketDetector",
    "RsiZscoreFilter",
    "SignalFilter",
    "StochasticCrossDetector",
    "StochasticExtremeZscoreDetector",
    "ZscoreRollingMinDetector",
]

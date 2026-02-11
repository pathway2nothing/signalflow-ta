"""Signal detectors based on technical analysis indicators."""

from signalflow.ta.signals.rsi_anomaly import RsiAnomalyDetector1
from signalflow.ta.signals.aroon_cross import AroonCrossDetector1
from signalflow.ta.signals.cci_anomaly import CciAnomalyDetector1
from signalflow.ta.signals.bollinger_band import BollingerBandDetector1
from signalflow.ta.signals.market_condition import (
    MarketConditionDetector1,
    MarketConditionDetector2,
    MarketConditionDetector3,
)
from signalflow.ta.signals.isolation_forest import (
    IsolationForestDetector1,
    IsolationForestDetector2,
    IsolationForestDetector3,
)
from signalflow.ta.signals.keltner_channel import (
    KeltnerChannelDetector1,
    KeltnerChannelDetector2,
)
from signalflow.ta.signals.hampel_filter import (
    HampelFilterDetector1,
    HampelFilterDetector2,
)
from signalflow.ta.signals.kalman_filter import KalmanFilterDetector1
from signalflow.ta.signals.cross_pair import CrossPairDetector1
from signalflow.ta.signals.stochastic import (
    StochasticDetector1,
    StochasticDetector2,
)
from signalflow.ta.signals.mfi import (
    MfiDetector1,
    MfiDetector2,
)
from signalflow.ta.signals.adx_regime import (
    AdxRegimeDetector1,
    AdxRegimeDetector2,
)
from signalflow.ta.signals.divergence import (
    DivergenceDetector1,
    DivergenceDetector2,
)
from signalflow.ta.signals.filters import (
    SignalFilter,
    PriceUptrendFilter,
    PriceDowntrendFilter,
    LowVolatilityFilter,
    HighVolatilityFilter,
    MeanReversionFilter,
    MeanExtensionFilter,
    RsiZscoreFilter,
    BelowBBLowerFilter,
    AboveBBUpperFilter,
    CciZscoreFilter,
    MacdBelowSignalFilter,
    MacdAboveSignalFilter,
)

__all__ = [
    # Detectors
    "RsiAnomalyDetector1",
    "AroonCrossDetector1",
    "CciAnomalyDetector1",
    "BollingerBandDetector1",
    "MarketConditionDetector1",
    "MarketConditionDetector2",
    "MarketConditionDetector3",
    "IsolationForestDetector1",
    "IsolationForestDetector2",
    "IsolationForestDetector3",
    "KeltnerChannelDetector1",
    "KeltnerChannelDetector2",
    "HampelFilterDetector1",
    "HampelFilterDetector2",
    "KalmanFilterDetector1",
    "CrossPairDetector1",
    "StochasticDetector1",
    "StochasticDetector2",
    "MfiDetector1",
    "MfiDetector2",
    "AdxRegimeDetector1",
    "AdxRegimeDetector2",
    "DivergenceDetector1",
    "DivergenceDetector2",
    # Filters
    "SignalFilter",
    "PriceUptrendFilter",
    "PriceDowntrendFilter",
    "LowVolatilityFilter",
    "HighVolatilityFilter",
    "MeanReversionFilter",
    "MeanExtensionFilter",
    "RsiZscoreFilter",
    "BelowBBLowerFilter",
    "AboveBBUpperFilter",
    "CciZscoreFilter",
    "MacdBelowSignalFilter",
    "MacdAboveSignalFilter",
]

"""Signal detectors based on technical analysis indicators."""

from signalflow.ta.signals.adx_regime import (
    AdxRegimeDetector1,
    AdxRegimeDetector2,
)
from signalflow.ta.signals.aroon_cross import AroonCrossDetector1
from signalflow.ta.signals.bollinger_band import BollingerBandDetector1
from signalflow.ta.signals.cci_anomaly import CciAnomalyDetector1
from signalflow.ta.signals.cross_pair import CrossPairDetector1
from signalflow.ta.signals.divergence import (
    DivergenceDetector1,
    DivergenceDetector2,
    DivergenceDetector3,
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
    HampelFilterDetector1,
    HampelFilterDetector2,
)
from signalflow.ta.signals.isolation_forest import (
    IsolationForestDetector1,
    IsolationForestDetector2,
    IsolationForestDetector3,
)
from signalflow.ta.signals.kalman_filter import KalmanFilterDetector1
from signalflow.ta.signals.keltner_channel import (
    KeltnerChannelDetector1,
    KeltnerChannelDetector2,
)
from signalflow.ta.signals.market_condition import (
    MarketConditionDetector1,
    MarketConditionDetector2,
    MarketConditionDetector3,
)
from signalflow.ta.signals.mfi import (
    MfiDetector1,
    MfiDetector2,
)
from signalflow.ta.signals.rsi_anomaly import RsiAnomalyDetector1
from signalflow.ta.signals.stochastic import (
    StochasticDetector1,
    StochasticDetector2,
)

__all__ = [
    "AboveBBUpperFilter",
    "AdxRegimeDetector1",
    "AdxRegimeDetector2",
    "AroonCrossDetector1",
    "BelowBBLowerFilter",
    "BollingerBandDetector1",
    "CciAnomalyDetector1",
    "CciZscoreFilter",
    "CrossPairDetector1",
    "DivergenceDetector1",
    "DivergenceDetector2",
    "DivergenceDetector3",
    "HampelFilterDetector1",
    "HampelFilterDetector2",
    "HighVolatilityFilter",
    "IsolationForestDetector1",
    "IsolationForestDetector2",
    "IsolationForestDetector3",
    "KalmanFilterDetector1",
    "KeltnerChannelDetector1",
    "KeltnerChannelDetector2",
    "LowVolatilityFilter",
    "MacdAboveSignalFilter",
    "MacdBelowSignalFilter",
    "MarketConditionDetector1",
    "MarketConditionDetector2",
    "MarketConditionDetector3",
    "MeanExtensionFilter",
    "MeanReversionFilter",
    "MfiDetector1",
    "MfiDetector2",
    "PriceDowntrendFilter",
    "PriceUptrendFilter",
    # Detectors
    "RsiAnomalyDetector1",
    "RsiZscoreFilter",
    # Filters
    "SignalFilter",
    "StochasticDetector1",
    "StochasticDetector2",
]

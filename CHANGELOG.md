# Changelog (signalflow-ta)

## [Unreleased]

### Changed (breaking - pre-1.0, no compatibility aliases)

- Detectors carry semantic names and register under `detector/<name>`; the
  numbered classes are gone:

  | Old class (registry) | New class (registry) |
  |---|---|
  | `AdxRegimeDetector1` (`ta/adx_regime_1`) | `AdxDiCrossDetector` (`detector/adx_di_cross`) |
  | `AdxRegimeDetector2` (`ta/adx_regime_2`) | `AdxRegimeRsiDetector` (`detector/adx_regime_rsi`) |
  | `KeltnerChannelDetector1` (`ta/keltner_channel_1`) | `KeltnerRsiZscoreDetector` (`detector/keltner_rsi_zscore`) |
  | `KeltnerChannelDetector2` (`ta/keltner_channel_2`) | `KeltnerMacdRsiDetector` (`detector/keltner_macd_rsi`) |
  | `MfiDetector1` (`ta/mfi_1`) | `MfiExtremeDetector` (`detector/mfi_extreme`) |
  | `MfiDetector2` (`ta/mfi_2`) | `MfiZscoreReversalDetector` (`detector/mfi_zscore_reversal`) |
  | `AroonCrossDetector1` (`ta/aroon_cross_1`) | `AroonCrossDetector` (`detector/aroon_cross`) |
  | `BollingerBandDetector1` (`ta/bollinger_band_1`) | `BollingerBreakoutDetector` (`detector/bollinger_breakout`) |
  | `RsiAnomalyDetector1` (`ta/rsi_anomaly_1`) | `RsiAnomalyDetector` (`detector/rsi_anomaly`) |
  | `DivergenceDetector1` (`ta/divergence_1`) | `RsiDivergenceDetector` (`detector/rsi_divergence`) |
  | `DivergenceDetector2` (`ta/divergence_2`) | `RsiDivergenceOffsetDetector` (`detector/rsi_divergence_offset`) |
  | `DivergenceDetector3` (`ta/divergence_3`) | `MacdDivergenceDetector` (`detector/macd_divergence`) |
  | `StochasticDetector1` (`ta/stochastic_1`) | `StochasticCrossDetector` (`detector/stochastic_cross`) |
  | `StochasticDetector2` (`ta/stochastic_2`) | `StochasticExtremeZscoreDetector` (`detector/stochastic_extreme_zscore`) |
  | `CciAnomalyDetector1` (`ta/cci_anomaly_1`) | `CciAnomalyDetector` (`detector/cci_anomaly`) |
  | `HampelFilterDetector1` (`ta/hampel_filter_1`) | `HampelAnomalyDetector` (`detector/hampel_anomaly`) |
  | `HampelFilterDetector2` (`ta/hampel_filter_2`) | `AdaptiveHampelAnomalyDetector` (`detector/adaptive_hampel_anomaly`) |
  | `KalmanFilterDetector1` (`ta/kalman_filter_1`) | `KalmanFilterDetector` (`detector/kalman_filter`) |
  | `IsolationForestDetector1` (`ta/isolation_forest_1`) | `IsoForestReturnsDetector` (`detector/isoforest_returns`) |
  | `IsolationForestDetector2` (`ta/isolation_forest_2`) | `IsoForestRsiDetector` (`detector/isoforest_rsi`) |
  | `IsolationForestDetector3` (`ta/isolation_forest_3`) | `IsoForestCrossSectionalDetector` (`detector/isoforest_cross_sectional`) |
  | `MarketConditionDetector1` (unregistered) | `RsiGlobalVolDetector` (unregistered) |
  | `MarketConditionDetector2` (unregistered) | `RsiVsMarketDetector` (unregistered) |
  | `MarketConditionDetector3` (unregistered) | `ZscoreRollingMinDetector` (unregistered) |
  | `CrossPairDetector1` (unregistered) | `CrossPairCorrBollingerDetector` (unregistered) |

- Inside detector `Signals` frames the strength column is `score`; `signal` is
  reserved for the discrete RISE/FALL/NONE column the core reads.
- `IsoForest*Detector` classes are flagged `learned = True`: they fit a model on the
  frame they detect on, so a `Flow` warns that their signals are in-sample.
- The compat `Feature`/`SignalDetector` base classes reuse the core `ensure_sorted`
  fast path instead of re-sorting the frame for every transform.
- `stat/skew`, `stat/kurtosis`, `stat/autocorr`, `stat/variance_ratio` and
  `momentum/roc` run on Polars rolling kernels instead of python loops; values are
  unchanged (`tests/test_ported_features.py` pins them to the old implementations).
- `requirements.txt` removed (pyproject is the source); lint config covers `E741`
  (`h, l, c` names) and the plugin `__init__` re-exports; perf tests skip without numba.

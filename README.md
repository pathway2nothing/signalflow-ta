<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/logo-dark.svg" width="120">
  <source media="(prefers-color-scheme: light)" srcset="assets/logo.svg" width="120">
  <img alt="SignalFlow" src="assets/logo.png" width="120">
</picture>

# signalflow-ta

**Technical analysis plugin for SignalFlow - 248 indicator features + 21 signal detectors**

<p>
<a href="https://pypi.org/project/signalflow-ta/"><img src="https://img.shields.io/badge/version-0.8.5-7c3aed" alt="Version"></a>
<a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.12+-3b82f6?logo=python&logoColor=white" alt="Python 3.12+"></a>
<a href="https://signalflow-trading.com"><img src="https://img.shields.io/badge/docs-signalflow--trading.com-7c3aed" alt="Docs"></a>
</p>

</div>

---

Part of the [SignalFlow](https://github.com/pathway2nothing/sf-project) ecosystem.

A plugin that registers 248 technical-analysis indicator features and 21
signal detectors with configurable filters, plus `AutoFeatureNormalizer` for
automatic normalization. Installing it auto-registers every component with the
core `signalflow` registry via entry points - no imports or wiring needed.

## Installation

```bash
pip install signalflow-ta            # pulls signalflow-trading
# or, from the core:
pip install "signalflow-trading[ta]"
```

**Requires:** Python ≥ 3.12, signalflow-trading ≥ 0.8.0. Optional: `signalflow-ta[numba]` for faster kernels.

## Usage

Once installed, every indicator and detector is discoverable from the core
registry (`sf.registry.list(sf.ComponentType.TRANSFORM)`) and usable directly in
a `Flow`. You can also import the classes:

```python
import signalflow.ta as ta

from signalflow.ta.momentum import RsiMom, MacdMom
from signalflow.ta.volatility import BollingerVol
from signalflow.ta.signals import BollingerBreakoutDetector, RsiZscoreFilter

# Use as features
rsi = RsiMom(period=14)
macd = MacdMom(fast=12, slow=26, signal=9)
bb = BollingerVol(period=20, std_dev=2.0)

# Use as detector with filters
detector = BollingerBreakoutDetector(
    period=20, std=2.0, rsi_period=14,
    direction="long",
    filters=[RsiZscoreFilter(threshold=-1.5)]
)
signals = detector.run(raw_data_view)
```

---

## Indicators

### Momentum (18)

| Class | SF Name | Description | Parameters |
|-------|---------|-------------|------------|
| `RsiMom` | `momentum/rsi` | Relative Strength Index | `period`, `normalized` |
| `RocMom` | `momentum/roc` | Rate of Change | `period`, `normalized`, `norm_period` |
| `MomMom` | `momentum/mom` | Momentum (price difference) | `period`, `normalized`, `norm_period` |
| `CmoMom` | `momentum/cmo` | Chande Momentum Oscillator | `period`, `normalized` |
| `StochMom` | `momentum/stoch` | Stochastic Oscillator | `k_period`, `d_period`, `smooth_k` |
| `StochRsiMom` | `momentum/stochrsi` | Stochastic RSI | `rsi_period`, `stoch_period`, `k_period`, `d_period` |
| `WillrMom` | `momentum/willr` | Williams %R | `period` |
| `CciMom` | `momentum/cci` | Commodity Channel Index | `period`, `constant` |
| `UoMom` | `momentum/uo` | Ultimate Oscillator | `fast`, `medium`, `slow` |
| `AoMom` | `momentum/ao` | Awesome Oscillator | `fast`, `slow` |
| `MacdMom` | `momentum/macd` | MACD | `fast`, `slow`, `signal` |
| `PpoMom` | `momentum/ppo` | Percentage Price Oscillator | `fast`, `slow`, `signal` |
| `TsiMom` | `momentum/tsi` | True Strength Index | `fast`, `slow`, `signal` |
| `TrixMom` | `momentum/trix` | Triple Exponential Average ROC | `period`, `signal` |
| `AccelerationMom` | `momentum/acceleration` | Price Acceleration (2nd derivative) | `lag`, `smooth` |
| `JerkMom` | `momentum/jerk` | Price Jerk (3rd derivative) | `lag`, `smooth` |
| `AngularMomentumMom` | `momentum/angular_momentum` | Angular Momentum | `period`, `ma_period` |
| `TorqueMom` | `momentum/torque` | Torque (dL/dt) | `period`, `ma_period`, `torque_lag` |

### Overlap / Smoothing (27)

| Class | SF Name | Description |
|-------|---------|-------------|
| `SmaSmooth` | `smooth/sma` | Simple Moving Average |
| `EmaSmooth` | `smooth/ema` | Exponential Moving Average |
| `WmaSmooth` | `smooth/wma` | Weighted Moving Average |
| `RmaSmooth` | `smooth/rma` | Wilder's Smoothed MA |
| `DemaSmooth` | `smooth/dema` | Double EMA |
| `TemaSmooth` | `smooth/tema` | Triple EMA |
| `HmaSmooth` | `smooth/hma` | Hull Moving Average |
| `TrimaSmooth` | `smooth/trima` | Triangular MA |
| `SwmaSmooth` | `smooth/swma` | Symmetric Weighted MA |
| `SsfSmooth` | `smooth/ssf` | Ehlers Super Smoother |
| `KamaSmooth` | `smooth/kama` | Kaufman Adaptive MA |
| `AlmaSmooth` | `smooth/alma` | Arnaud Legoux MA |
| `JmaSmooth` | `smooth/jma` | Jurik MA |
| `VidyaSmooth` | `smooth/vidya` | Variable Index Dynamic Average |
| `T3Smooth` | `smooth/t3` | Tillson T3 |
| `ZlmaSmooth` | `smooth/zlma` | Zero Lag MA |
| `McGinleySmooth` | `smooth/mcginley` | McGinley Dynamic |
| `FramaSmooth` | `smooth/frama` | Fractal Adaptive MA |
| `KalmanSmooth` | `smooth/kalman` | Adaptive Kalman Filter |
| `LinRegPriceDiff` | `trend/linreg_price_diff` | Price diff from regression |
| `Hl2Price` | `price/hl2` | High-Low Midpoint |
| `Hlc3Price` | `price/hlc3` | Typical Price |
| `Ohlc4Price` | `price/ohlc4` | OHLC Average |
| `WcpPrice` | `price/wcp` | Weighted Close Price |
| `TypicalPrice` | `price/typical` | Configurable Weighted Price |
| `MidpointPrice` | `price/midpoint` | Rolling Midpoint |
| `MidpricePrice` | `price/midprice` | Donchian Channel Midline |

### Volatility (19)

| Class | SF Name | Description |
|-------|---------|-------------|
| `TrueRangeVol` | `volatility/true_range` | True Range |
| `AtrVol` | `volatility/atr` | Average True Range |
| `NatrVol` | `volatility/natr` | Normalized ATR (% of price) |
| `BollingerVol` | `volatility/bollinger` | Bollinger Bands |
| `KeltnerVol` | `volatility/keltner` | Keltner Channels |
| `DonchianVol` | `volatility/donchian` | Donchian Channels |
| `AccBandsVol` | `volatility/accbands` | Acceleration Bands |
| `MassIndexVol` | `volatility/mass_index` | Mass Index |
| `UlcerIndexVol` | `volatility/ulcer_index` | Ulcer Index |
| `RviVol` | `volatility/rvi` | Relative Volatility Index |
| `GapVol` | `volatility/gaps` | Gap Volatility |
| `KineticEnergyVol` | `volatility/kinetic_energy` | Kinetic Energy (½v²) |
| `PotentialEnergyVol` | `volatility/potential_energy` | Potential Energy |
| `TotalEnergyVol` | `volatility/total_energy` | Total Mechanical Energy |
| `EnergyFlowVol` | `volatility/energy_flow` | Energy Flow Rate (dE/dt) |
| `ElasticStrainVol` | `volatility/elastic_strain` | Elastic Strain |
| `TemperatureVol` | `volatility/temperature` | Market Temperature |
| `HeatCapacityVol` | `volatility/heat_capacity` | Heat Capacity |
| `FreeEnergyVol` | `volatility/free_energy` | Helmholtz Free Energy |

### Volume (16)

| Class | SF Name | Description |
|-------|---------|-------------|
| `ObvVolume` | `volume/obv` | On Balance Volume |
| `AdVolume` | `volume/ad` | Accumulation/Distribution |
| `PvtVolume` | `volume/pvt` | Price Volume Trend |
| `NviVolume` | `volume/nvi` | Negative Volume Index |
| `PviVolume` | `volume/pvi` | Positive Volume Index |
| `MfiVolume` | `volume/mfi` | Money Flow Index |
| `CmfVolume` | `volume/cmf` | Chaikin Money Flow |
| `EfiVolume` | `volume/efi` | Elder Force Index |
| `EomVolume` | `volume/eom` | Ease of Movement |
| `KvoVolume` | `volume/kvo` | Klinger Volume Oscillator |
| `MarketForceVolume` | `volume/market_force` | Market Force (F = ma) |
| `ImpulseVolume` | `volume/impulse` | Market Impulse (∑F·dt) |
| `MarketMomentumVolume` | `volume/market_momentum` | Market Momentum (p = mv) |
| `MarketPowerVolume` | `volume/market_power` | Market Power (P = Fv) |
| `MarketCapacitanceVolume` | `volume/market_capacitance` | Market Capacitance |
| `GravitationalPullVolume` | `volume/gravitational_pull` | Gravitational Pull |

### Trend (28)

| Class | SF Name | Description |
|-------|---------|-------------|
| `AdxTrend` | `trend/adx` | Average Directional Index |
| `AroonTrend` | `trend/aroon` | Aroon Indicator |
| `VortexTrend` | `trend/vortex` | Vortex Indicator |
| `VhfTrend` | `trend/vhf` | Vertical Horizontal Filter |
| `ChopTrend` | `trend/chop` | Choppiness Index |
| `ViscosityTrend` | `trend/viscosity` | Viscosity |
| `ReynoldsTrend` | `trend/reynolds` | Reynolds Number |
| `RotationalInertiaTrend` | `trend/rotational_inertia` | Rotational Inertia |
| `MarketImpedanceTrend` | `trend/market_impedance` | Market Impedance |
| `RCTimeConstantTrend` | `trend/rc_time_constant` | RC Time Constant |
| `SNRTrend` | `trend/snr` | Signal-to-Noise Ratio |
| `OrderParameterTrend` | `trend/order_parameter` | Order Parameter |
| `SusceptibilityTrend` | `trend/susceptibility` | Susceptibility |
| `PsarTrend` | `trend/psar` | Parabolic SAR |
| `SupertrendTrend` | `trend/supertrend` | Supertrend |
| `ChandelierTrend` | `trend/chandelier` | Chandelier Stop |
| `HiloTrend` | `trend/hilo` | HiLo Channel |
| `CkspTrend` | `trend/cksp` | Coppock Curve |
| `IchimokuTrend` | `trend/ichimoku` | Ichimoku Kinko Hyo |
| `DpoTrend` | `trend/dpo` | Detrended Price Oscillator |
| `QstickTrend` | `trend/qstick` | Qstick |
| `TtmTrend` | `trend/ttm` | TTM Trend |
| `WilliamsAlligatorRegime` | `trend/alligator_regime` | Williams Alligator |
| `TwoMaRegime` | `trend/two_ma_regime` | Two MA Crossover Regime |
| `SmaDirection` | `trend/sma_direction` | SMA Direction |
| `SmaDiffDirection` | `trend/sma_diff_direction` | SMA Difference Direction |
| `LinRegDirection` | `trend/linreg_direction` | Linear Regression Direction |
| `LinRegDiffDirection` | `trend/linreg_diff_direction` | LinReg Difference Direction |

### Statistics (75+)

**Dispersion:** Variance, Std, MAD, Z-Score, CV, Range, IQR, AAD, Robust Z-Score

**Distribution:** Median, Quantile, Percentile Rank, Min-Max, Skewness, Kurtosis, Entropy, Jarque-Bera, Mode Distance, Above Mean Ratio, Entropy Rate

**Memory:** Hurst Exponent, Autocorrelation, Variance Ratio, Diffusion Coefficient, Anomalous Diffusion, MSD Ratio, Spring Constant, Damping Ratio, Natural Frequency, Plastic Strain, Escape Velocity, Correlation Length

**Cycle:** Inst. Amplitude, Phase, Frequency (Hilbert), Phase Acceleration, Constructive Interference, Beat Frequency, Standing Wave Ratio, Spectral Centroid, Spectral Entropy

**Regression:** Correlation, Beta, R-Squared, LinReg Slope/Intercept/Residual

**Realized Vol:** Realized, Parkinson, Garman-Klass, Rogers-Satchell, Yang-Zhang

**Complexity:** Permutation Entropy, Sample Entropy, Lempel-Ziv, Fisher Information, DFA

**Information Theory:** KL Divergence, JS Divergence, Rényi Entropy, Auto Mutual Info, Relative Info Gain

**DSP:** Spectral Bandwidth, Slope, Kurtosis, Contrast, MFCC Band Energy

**Structure:** Reverse Points, Time Since Spike, Volatility Spike, Volume Spike, Rolling Min/Max

### Performance (2) & Divergence (2)

| Class | SF Name | Description |
|-------|---------|-------------|
| `LogReturn` | `perf/log_ret` | Logarithmic Returns |
| `PctReturn` | `perf/pct_ret` | Percentage Returns |
| `RsiDivergence` | `divergence/rsi` | RSI Divergence (regular & hidden) |
| `MacdDivergence` | `divergence/macd` | MACD Divergence (regular & hidden) |

### Microstructure (14) - NEW

Candle geometry features added from sf-profit feature research.
Strongest single-axis tested - `BodyToRangeRatio` and `WickToBodyRatio` rank in the top-10 of walk-forward IV pool.

| Class | Module | Description |
|-------|--------|-------------|
| `BodyToRangeRatio` | `microstructure/candles` | `\|body\| / range` rolling mean (0=doji, 1=marubozu) |
| `ClosePositionInBar` | `microstructure/candles` | `(close-low) / (high-low) ∈ [0,1]` |
| `HiLoMedianGap` | `microstructure/candles` | `(high+low)/2 - close` rolling |
| `LowerWickPersistence` | `microstructure/candles` | fraction of bars with `lower_wick > \|body\|` |
| `UpperWickPersistence` | `microstructure/candles` | fraction of bars with `upper_wick > \|body\|` |
| `WickAsymmetry` | `microstructure/candles` | `(upper-lower) / (upper+lower) ∈ [-1,+1]` |
| `WickToBodyLower` | `microstructure/candles` | `lower_wick / \|body\|` rolling mean |
| `WickToBodyRatio` | `microstructure/candles` | `(upper+lower) / \|body\|` rolling mean |
| `WickToBodyUpper` | `microstructure/candles` | `upper_wick / \|body\|` rolling mean |
| `HighLowImbalance` | `microstructure/ranges` | signed position in recent (max-min) range |
| `RangeExpansion` | `microstructure/ranges` | `(high-low) / SMA(high-low) - 1` |
| `RangeFragmentation` | `microstructure/ranges` | `sum_abs_returns / (max-min)` over window |
| `RangeNormalizedReturn` | `microstructure/ranges` | `return / prev_bar_range`, rolling mean |
| `SignedRollingRange` | `microstructure/ranges` | signed position by ATR |

### Path-shape (15) - NEW

Geometric properties of price path: roughness, efficiency, streaks, entropy.

| Class | Module | Description |
|-------|--------|-------------|
| `PathRoughness` | `path_shape/shape` | `std(\|return\|) / mean(\|return\|)` - CV of move size |
| `PathEfficiency` | `path_shape/shape` | Kaufman ER: net move / total path |
| `PathTortuosity` | `path_shape/shape` | total path / (max - min) |
| `PathSimplicity` | `path_shape/shape` | `\|net_return\| / sum_abs_returns ∈ [0,1]` |
| `MaxConsecutiveGainRun` | `path_shape/streaks` | longest streak of up-bars in window |
| `MaxConsecutiveLossRun` | `path_shape/streaks` | longest streak of down-bars in window |
| `LongestStreak` | `path_shape/streaks` | max(gain_run, loss_run) per window |
| `ReversalCount` | `path_shape/streaks` | count of local extrema |
| `ZeroCrossingRate` | `path_shape/streaks` | return sign flip rate |
| `ReturnSignEntropy` | `path_shape/entropy` | Shannon entropy of return signs |
| `DirectionalEntropy` | `path_shape/entropy` | entropy of (sign × magnitude_quintile) joint |
| `VolumeEntropy` | `path_shape/entropy` | entropy of volume distribution (10 quantized bins) |
| `ReturnAutocorrShort` | `path_shape/autocorr` | rolling autocorr(returns, lag) |
| `VolatilityClusterScore` | `path_shape/autocorr` | autocorr(\|returns\|) - GARCH proxy |
| `ErrorAutoCorrelation` | `path_shape/autocorr` | rolling autocorr of `close - SMA(period)` |

### Filters (7) - NEW

`close - filter(close)` residuals across various filters. Useful in mean-reversion and PID-style control strategies.

| Class | Module | Description |
|-------|--------|-------------|
| `AdaptiveEMAError` | `filters/smoother_errors` | `close - EMA(period)`. PID-P proxy |
| `DEMAError` | `filters/smoother_errors` | `close - DEMA`. Less lag than EMA |
| `TEMAError` | `filters/smoother_errors` | `close - TEMA`. Even less lag |
| `HMAError` | `filters/smoother_errors` | `close - HMA`. Hull-style smoother residual |
| `KalmanResidual` | `filters/kalman` | 1D Kalman filter innovation (predictive residual) |
| `PIDIntegralTerm` | `filters/pid` | Rolling sum of `(close - SMA)`. PID-I |
| `PIDDerivativeTerm` | `filters/pid` | Derivative of `(close - SMA)`. PID-D |

### Cross-sectional (15) - NEW

Operate on full multi-pair DataFrames. Rank-based variants work robustly on 3+ pairs; distributional variants (skew, breadth) benefit from 5+ pairs.

| Class | Description |
|-------|-------------|
| `CrossSectionalReturnRank` | Rank of pair's return across pairs |
| `CrossSectionalAtrRank` | Rank of pair's ATR across pairs |
| `CrossSectionalAdxRank` | Rank of pair's \|returns\|-sum (ADX proxy) |
| `CrossSectionalRangeRank` | Rank of pair's (high-low) range |
| `CrossSectionalRsiRank` | Rank of pair's RSI proxy |
| `CrossSectionalVolRank` | Rank of pair's realized vol |
| `CrossSectionalReturnAccelRank` | Rank of pair's return acceleration |
| `CrossSectionalBeta` | Rolling β to market median |
| `AvgPairwiseCorrMarket` | Rolling corr(pair, market_median) |
| `PairLeadLagCorr` | Rolling corr(pair_t, market_{t-lag}) - leads/lags |
| `CrossSectionalDispersion` | Mean `\|pair_ret - market_mean\|` across pairs |
| `CrossSectionalRetSkew` | Skewness of returns across pairs (needs 5+) |
| `MarketBreadth` | Fraction of pairs with positive return ∈ [0,1] |
| `RelativeStrengthVsMarket` | z-score of pair vs cross-section |
| `PairExcessReturn` | pair_ret − market_mean_ret |
| `DivergenceFromMarketMedian` | pair_ret − market_median_ret |

### Stat / Volatility / Volume / Momentum / Trend extensions

Each existing module gained new classes from sf-profit feature research. Selected highlights:

**Regression (sf-profit iter-15/18/20)** - in `stat`:
`LinRegSlopeWindow`, `LinRegR2`, `LinRegResidualStd`, `LinRegSlopeChange`, `LinRegSlopeAcceleration`, `Poly2ResidualStd`, `LinRegResidualSkew/Kurtosis`, `LinRegSlopeRatio`, `LinRegSlopeNormalized`, `LinRegInterceptNormalized`, `LinRegFitQuality`.

**Distribution (sf-profit iter-20)** - in `stat`:
`ReturnSkewWindow`, `ReturnKurtosisWindow`, `ReturnTailRatio`.

**Volatility (sf-profit iter-3.1/15/16/20)** - in `volatility`:
`NatrRatio`, `NatrPctRank`, `VolOfVol`, `RealizedVolPctRank`, `RealizedVolRatio`, `ParkinsonZScore`, `ParkinsonAccel`, `ParkinsonVolRatio`, `AltVolDeviation`, `GarmanKlassRatio`, `GarmanKlassPctRank`, `PriceZAtr`.

**Volume×price coupling (sf-profit iter-15/16/18/20)** - in `volume`:
`PriceImpactPerUnit`, `VWAPDeviation`, `SignedVolumeAccumulation`, `AbsReturnVolumeCorr`, `PriceVolumeCorrelation`, `VolumePerRange`, `VolumeImbalance`, `VolumeWeightedReturn`, `VolumeSpike`, `VolumeAcceleration`, `VolumeZScore`, `VolumeMomentumRatio`, `VolPctRankSignedTrend`.

**Momentum (sf-profit feature_research_lib + iter-15/16/18/20)** - in `momentum`:
`MomPosNeg`, `RocSignedLog`, `MacdNorm`, `RsiSpread`, `PriceMomentumConfirmation`, `VolPriceConfirmation`, `TrendPersistence`, `PriceAcceleration`, `MomentumOfMomentum`.

**Trend (sf-profit feature_research_lib + iter-15/16/18)** - in `trend`:
`DiBalance`, `NatrXDiBalance`, `UpDownEntropyAsymmetry`, `EntropyRatio`, `RsiDivPolarity`, `HilbertAmplitudeSlope`.

> **Provenance.** All ~100 new classes were extracted from the sf-profit feature research pipeline (iter-3 → iter-21). They were validated through walk-forward IV scoring on 6 monthly folds and Spearman-correlation dedup at \|corr\| ≥ 0.85. 370 features survived to the final pool. See sf-profit's `docs/feature_catalog.md` for per-feature scores and walk-forward stability.

---

## Signal Detectors (21)

All detectors support configurable `direction` (`"long"`, `"short"`, `"both"`) and optional filters.
Detectors marked *learned* fit a model on the frame they detect on, so their signals are in-sample and a `Flow` warns about it.

| Registry name | Class | What it does |
|---|---|---|
| `detector/adaptive_hampel_anomaly` | `AdaptiveHampelAnomalyDetector` | Adaptive Hampel filter-based anomaly detector |
| `detector/adx_di_cross` | `AdxDiCrossDetector` | ADX trend regime detector with DI crossover |
| `detector/adx_regime_rsi` | `AdxRegimeRsiDetector` | ADX regime detector combining trend/range with RSI |
| `detector/aroon_cross` | `AroonCrossDetector` | Aroon crossover signal detector |
| `detector/bollinger_breakout` | `BollingerBreakoutDetector` | Bollinger Band breakout detector |
| `detector/cci_anomaly` | `CciAnomalyDetector` | CCI statistical anomaly detector |
| `detector/hampel_anomaly` | `HampelAnomalyDetector` | Hampel filter-based anomaly detector |
| `detector/isoforest_cross_sectional` | `IsoForestCrossSectionalDetector` (learned) | Cross-sectional Isolation Forest detector |
| `detector/isoforest_returns` | `IsoForestReturnsDetector` (learned) | Isolation Forest anomaly detector using log returns |
| `detector/isoforest_rsi` | `IsoForestRsiDetector` (learned) | Isolation Forest anomaly detector using RSI |
| `detector/kalman_filter` | `KalmanFilterDetector` | Adaptive Kalman Filter-based signal detector |
| `detector/keltner_macd_rsi` | `KeltnerMacdRsiDetector` | Keltner Channel detector with MACD and RSI conditions |
| `detector/keltner_rsi_zscore` | `KeltnerRsiZscoreDetector` | Keltner Channel detector with RSI z-score condition |
| `detector/macd_divergence` | `MacdDivergenceDetector` | MACD divergence detector |
| `detector/mfi_extreme` | `MfiExtremeDetector` | Money Flow Index extreme zone detector |
| `detector/mfi_zscore_reversal` | `MfiZscoreReversalDetector` | Money Flow Index with z-score and reversal detection |
| `detector/rsi_anomaly` | `RsiAnomalyDetector` | RSI statistical anomaly detector |
| `detector/rsi_divergence` | `RsiDivergenceDetector` | RSI divergence detector |
| `detector/rsi_divergence_offset` | `RsiDivergenceOffsetDetector` | RSI divergence detector with offset subsampling |
| `detector/stochastic_cross` | `StochasticCrossDetector` | Stochastic oscillator crossover detector |
| `detector/stochastic_extreme_zscore` | `StochasticExtremeZscoreDetector` | Stochastic oscillator extreme zone detector with z-score |

### Signal Filters (12)

`PriceUptrendFilter`, `PriceDowntrendFilter`, `LowVolatilityFilter`, `HighVolatilityFilter`, `MeanReversionFilter`, `MeanExtensionFilter`, `RsiZscoreFilter`, `BelowBBLowerFilter`, `AboveBBUpperFilter`, `CciZscoreFilter`, `MacdBelowSignalFilter`, `MacdAboveSignalFilter`

---

## Normalization

Most indicators support `normalized=True`:

- **Bounded** (RSI, Stochastic, Williams %R) → scaled to `[0, 1]` or `[-1, 1]`
- **Unbounded** (MACD, ROC, smoothers) → rolling z-score
- Normalized columns receive a `_norm` suffix

### AutoFeatureNormalizer

```python
from signalflow.ta.auto_norm import AutoFeatureNormalizer

norm = AutoFeatureNormalizer(window=256, warmup=256)
df_normalized = norm.fit_transform(df)
```

Methods: `rolling_robust`, `rolling_z`, `rolling_rank`, `rolling_winsor`, `signed_log1p`

---

**License:** MIT &ensp;·&ensp; Part of [SignalFlow](https://github.com/pathway2nothing/sf-project)

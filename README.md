# signalflow-ta

Technical analysis extension for [signalflow-trading](https://pypi.org/project/signalflow-trading/).

The library provides 166 technical analysis indicators organized into 8 modules: momentum, overlap, volatility, volume, trend, stat, performance, divergence. Each indicator is implemented as a standalone signalflow component class with warmup property support and (where applicable) output value normalization. Additionally, `AutoFeatureNormalizer` is provided for automatic normalization of Polars DataFrames.

## Installation

```bash
pip install signalflow-ta
```

## Requirements

- Python >= 3.12
- signalflow-trading >= 0.3.3
- pandas-ta >= 0.4.67b0

## Usage

```python
import signalflow.ta as ta
```

## Features

### Momentum (18 indicators)

| Class | SF Name | Description | Source | Parameters |
|-------|---------|-------------|--------|------------|
| `RsiMom` | `momentum/rsi` | Relative Strength Index | [core.py](src/signalflow/ta/momentum/core.py) | `["period", "normalized"]` |
| `RocMom` | `momentum/roc` | Rate of Change | [core.py](src/signalflow/ta/momentum/core.py) | `["period", "normalized", "norm_period"]` |
| `MomMom` | `momentum/mom` | Momentum (price difference) | [core.py](src/signalflow/ta/momentum/core.py) | `["period", "normalized", "norm_period"]` |
| `CmoMom` | `momentum/cmo` | Chande Momentum Oscillator | [core.py](src/signalflow/ta/momentum/core.py) | `["period", "normalized"]` |
| `StochMom` | `momentum/stoch` | Stochastic Oscillator | [oscillators.py](src/signalflow/ta/momentum/oscillators.py) | `["k_period", "d_period", "smooth_k", "normalized"]` |
| `StochRsiMom` | `momentum/stochrsi` | Stochastic RSI | [oscillators.py](src/signalflow/ta/momentum/oscillators.py) | `["rsi_period", "stoch_period", "k_period", "d_period", "normalized"]` |
| `WillrMom` | `momentum/willr` | Williams %R | [oscillators.py](src/signalflow/ta/momentum/oscillators.py) | `["period", "normalized"]` |
| `CciMom` | `momentum/cci` | Commodity Channel Index | [oscillators.py](src/signalflow/ta/momentum/oscillators.py) | `["period", "constant", "normalized", "norm_period"]` |
| `UoMom` | `momentum/uo` | Ultimate Oscillator | [oscillators.py](src/signalflow/ta/momentum/oscillators.py) | `["fast", "medium", "slow", "fast_weight", "medium_weight", "slow_weight", "normalized"]` |
| `AoMom` | `momentum/ao` | Awesome Oscillator | [oscillators.py](src/signalflow/ta/momentum/oscillators.py) | `["fast", "slow", "normalized", "norm_period"]` |
| `MacdMom` | `momentum/macd` | Moving Average Convergence Divergence | [macd.py](src/signalflow/ta/momentum/macd.py) | `["fast", "slow", "signal", "normalized", "norm_period"]` |
| `PpoMom` | `momentum/ppo` | Percentage Price Oscillator | [macd.py](src/signalflow/ta/momentum/macd.py) | `["fast", "slow", "signal", "normalized", "norm_period"]` |
| `TsiMom` | `momentum/tsi` | True Strength Index | [macd.py](src/signalflow/ta/momentum/macd.py) | `["fast", "slow", "signal", "normalized", "norm_period"]` |
| `TrixMom` | `momentum/trix` | Triple Exponential Average ROC | [macd.py](src/signalflow/ta/momentum/macd.py) | `["period", "signal", "normalized", "norm_period"]` |
| `AccelerationMom` | `momentum/acceleration` | Price Acceleration (2nd derivative of log-returns) | [kinematics.py](src/signalflow/ta/momentum/kinematics.py) | `["source_col", "lag", "smooth", "normalized", "norm_period"]` |
| `JerkMom` | `momentum/jerk` | Price Jerk (3rd derivative of log-returns) | [kinematics.py](src/signalflow/ta/momentum/kinematics.py) | `["source_col", "lag", "smooth", "normalized", "norm_period"]` |
| `AngularMomentumMom` | `momentum/angular_momentum` | Angular Momentum (L = displacement x velocity) | [kinematics.py](src/signalflow/ta/momentum/kinematics.py) | `["source_col", "period", "ma_period", "normalized", "norm_period"]` |
| `TorqueMom` | `momentum/torque` | Torque (rate of change of angular momentum) | [kinematics.py](src/signalflow/ta/momentum/kinematics.py) | `["source_col", "period", "ma_period", "torque_lag", "normalized", "norm_period"]` |

### Overlap / Smoothing (25 indicators)

| Class | SF Name | Description | Source | Parameters |
|-------|---------|-------------|--------|------------|
| `SmaSmooth` | `smooth/sma` | Simple Moving Average | [smoothers.py](src/signalflow/ta/overlap/smoothers.py) | `["source_col", "period", "normalized"]` |
| `EmaSmooth` | `smooth/ema` | Exponential Moving Average | [smoothers.py](src/signalflow/ta/overlap/smoothers.py) | `["source_col", "period", "normalized"]` |
| `WmaSmooth` | `smooth/wma` | Weighted Moving Average | [smoothers.py](src/signalflow/ta/overlap/smoothers.py) | `["source_col", "period", "normalized"]` |
| `RmaSmooth` | `smooth/rma` | Wilder's Smoothed Moving Average | [smoothers.py](src/signalflow/ta/overlap/smoothers.py) | `["source_col", "period", "normalized"]` |
| `DemaSmooth` | `smooth/dema` | Double Exponential Moving Average | [smoothers.py](src/signalflow/ta/overlap/smoothers.py) | `["source_col", "period", "normalized"]` |
| `TemaSmooth` | `smooth/tema` | Triple Exponential Moving Average | [smoothers.py](src/signalflow/ta/overlap/smoothers.py) | `["source_col", "period", "normalized"]` |
| `HmaSmooth` | `smooth/hma` | Hull Moving Average | [smoothers.py](src/signalflow/ta/overlap/smoothers.py) | `["source_col", "period", "normalized"]` |
| `TrimaSmooth` | `smooth/trima` | Triangular Moving Average | [smoothers.py](src/signalflow/ta/overlap/smoothers.py) | `["source_col", "period", "normalized"]` |
| `SwmaSmooth` | `smooth/swma` | Symmetric Weighted Moving Average | [smoothers.py](src/signalflow/ta/overlap/smoothers.py) | `["source_col", "period", "normalized"]` |
| `SsfSmooth` | `smooth/ssf` | Ehlers Super Smoother Filter | [smoothers.py](src/signalflow/ta/overlap/smoothers.py) | `["source_col", "period", "poles", "normalized"]` |
| `KamaSmooth` | `smooth/kama` | Kaufman Adaptive Moving Average | [adaptive.py](src/signalflow/ta/overlap/adaptive.py) | `["source_col", "period", "fast", "slow", "normalized"]` |
| `AlmaSmooth` | `smooth/alma` | Arnaud Legoux Moving Average | [adaptive.py](src/signalflow/ta/overlap/adaptive.py) | `["source_col", "period", "offset", "sigma", "normalized"]` |
| `JmaSmooth` | `smooth/jma` | Jurik Moving Average | [adaptive.py](src/signalflow/ta/overlap/adaptive.py) | `["source_col", "period", "phase", "normalized"]` |
| `VidyaSmooth` | `smooth/vidya` | Variable Index Dynamic Average | [adaptive.py](src/signalflow/ta/overlap/adaptive.py) | `["source_col", "period", "normalized"]` |
| `T3Smooth` | `smooth/t3` | Tillson T3 | [adaptive.py](src/signalflow/ta/overlap/adaptive.py) | `["source_col", "period", "vfactor", "normalized"]` |
| `ZlmaSmooth` | `smooth/zlma` | Zero Lag Moving Average | [adaptive.py](src/signalflow/ta/overlap/adaptive.py) | `["source_col", "period", "ma_type", "normalized"]` |
| `McGinleySmooth` | `smooth/mcginley` | McGinley Dynamic | [adaptive.py](src/signalflow/ta/overlap/adaptive.py) | `["source_col", "period", "k", "normalized"]` |
| `FramaSmooth` | `smooth/frama` | Fractal Adaptive Moving Average | [adaptive.py](src/signalflow/ta/overlap/adaptive.py) | `["source_col", "period", "normalized"]` |
| `Hl2Price` | `price/hl2` | High-Low Midpoint | [price.py](src/signalflow/ta/overlap/price.py) | `[]` |
| `Hlc3Price` | `price/hlc3` | Typical Price (HLC/3) | [price.py](src/signalflow/ta/overlap/price.py) | `[]` |
| `Ohlc4Price` | `price/ohlc4` | OHLC Average | [price.py](src/signalflow/ta/overlap/price.py) | `[]` |
| `WcpPrice` | `price/wcp` | Weighted Close Price | [price.py](src/signalflow/ta/overlap/price.py) | `[]` |
| `TypicalPrice` | `price/typical` | Configurable Weighted Price | [price.py](src/signalflow/ta/overlap/price.py) | `["weight_high", "weight_low", "weight_close"]` |
| `MidpointPrice` | `price/midpoint` | Rolling Midpoint | [price.py](src/signalflow/ta/overlap/price.py) | `["source_col", "period"]` |
| `MidpricePrice` | `price/midprice` | Donchian Channel Midline | [price.py](src/signalflow/ta/overlap/price.py) | `["period"]` |

### Volatility (19 indicators)

| Class | SF Name | Description | Source | Parameters |
|-------|---------|-------------|--------|------------|
| `TrueRangeVol` | `volatility/true_range` | True Range | [range.py](src/signalflow/ta/volatility/range.py) | `[]` |
| `AtrVol` | `volatility/atr` | Average True Range | [range.py](src/signalflow/ta/volatility/range.py) | `["period", "ma_type"]` |
| `NatrVol` | `volatility/natr` | Normalized ATR (% of price) | [range.py](src/signalflow/ta/volatility/range.py) | `["period", "ma_type"]` |
| `BollingerVol` | `volatility/bollinger` | Bollinger Bands | [bands.py](src/signalflow/ta/volatility/bands.py) | `["period", "std_dev", "ma_type", "normalized", "norm_period"]` |
| `KeltnerVol` | `volatility/keltner` | Keltner Channels | [bands.py](src/signalflow/ta/volatility/bands.py) | `["period", "multiplier", "ma_type", "use_true_range", "normalized", "norm_period"]` |
| `DonchianVol` | `volatility/donchian` | Donchian Channels | [bands.py](src/signalflow/ta/volatility/bands.py) | `["period", "normalized", "norm_period"]` |
| `AccBandsVol` | `volatility/accbands` | Acceleration Bands | [bands.py](src/signalflow/ta/volatility/bands.py) | `["period", "normalized", "norm_period"]` |
| `MassIndexVol` | `volatility/mass_index` | Mass Index | [measures.py](src/signalflow/ta/volatility/measures.py) | `["fast", "slow"]` |
| `UlcerIndexVol` | `volatility/ulcer_index` | Ulcer Index (downside volatility) | [measures.py](src/signalflow/ta/volatility/measures.py) | `["period"]` |
| `RviVol` | `volatility/rvi` | Relative Volatility Index | [measures.py](src/signalflow/ta/volatility/measures.py) | `["period", "std_period"]` |
| `GapVol` | `volatility/gaps` | Gap Volatility | [gaps.py](src/signalflow/ta/volatility/gaps.py) | `["period", "normalized", "norm_period"]` |
| `KineticEnergyVol` | `volatility/kinetic_energy` | Kinetic Energy (KE = 1/2 v^2) | [energy.py](src/signalflow/ta/volatility/energy.py) | `["period", "normalized", "norm_period"]` |
| `PotentialEnergyVol` | `volatility/potential_energy` | Potential Energy (displacement from MA) | [energy.py](src/signalflow/ta/volatility/energy.py) | `["period", "ma_period", "normalized", "norm_period"]` |
| `TotalEnergyVol` | `volatility/total_energy` | Total Mechanical Energy (KE + PE) | [energy.py](src/signalflow/ta/volatility/energy.py) | `["period", "ma_period", "normalized", "norm_period"]` |
| `EnergyFlowVol` | `volatility/energy_flow` | Energy Flow Rate (dE/dt) | [energy.py](src/signalflow/ta/volatility/energy.py) | `["period", "ma_period", "flow_lag", "normalized", "norm_period"]` |
| `ElasticStrainVol` | `volatility/elastic_strain` | Elastic Strain (relative displacement from equilibrium) | [energy.py](src/signalflow/ta/volatility/energy.py) | `["period", "ma_period", "normalized", "norm_period"]` |
| `TemperatureVol` | `volatility/temperature` | Market Temperature (kinetic energy per DOF) | [energy.py](src/signalflow/ta/volatility/energy.py) | `["period", "normalized", "norm_period"]` |
| `HeatCapacityVol` | `volatility/heat_capacity` | Heat Capacity (resistance to temperature change) | [energy.py](src/signalflow/ta/volatility/energy.py) | `["period", "lag", "normalized", "norm_period"]` |
| `FreeEnergyVol` | `volatility/free_energy` | Helmholtz Free Energy (E - T*S) | [energy.py](src/signalflow/ta/volatility/energy.py) | `["period", "entropy_bins", "normalized", "norm_period"]` |

### Volume (16 indicators)

| Class | SF Name | Description | Source | Parameters |
|-------|---------|-------------|--------|------------|
| `ObvVolume` | `volume/obv` | On Balance Volume | [cumulative.py](src/signalflow/ta/volume/cumulative.py) | `["period", "normalized", "norm_period"]` |
| `AdVolume` | `volume/ad` | Accumulation/Distribution Line | [cumulative.py](src/signalflow/ta/volume/cumulative.py) | `["period", "normalized", "norm_period"]` |
| `PvtVolume` | `volume/pvt` | Price Volume Trend | [cumulative.py](src/signalflow/ta/volume/cumulative.py) | `["period", "normalized", "norm_period"]` |
| `NviVolume` | `volume/nvi` | Negative Volume Index | [cumulative.py](src/signalflow/ta/volume/cumulative.py) | `["period", "normalized", "norm_period"]` |
| `PviVolume` | `volume/pvi` | Positive Volume Index | [cumulative.py](src/signalflow/ta/volume/cumulative.py) | `["period", "normalized", "norm_period"]` |
| `MfiVolume` | `volume/mfi` | Money Flow Index | [oscillators.py](src/signalflow/ta/volume/oscillators.py) | `["period", "normalized"]` |
| `CmfVolume` | `volume/cmf` | Chaikin Money Flow | [oscillators.py](src/signalflow/ta/volume/oscillators.py) | `["period", "normalized", "norm_period"]` |
| `EfiVolume` | `volume/efi` | Elder Force Index | [oscillators.py](src/signalflow/ta/volume/oscillators.py) | `["period", "normalized", "norm_period"]` |
| `EomVolume` | `volume/eom` | Ease of Movement | [oscillators.py](src/signalflow/ta/volume/oscillators.py) | `["period", "scale", "normalized", "norm_period"]` |
| `KvoVolume` | `volume/kvo` | Klinger Volume Oscillator | [oscillators.py](src/signalflow/ta/volume/oscillators.py) | `["fast", "slow", "signal", "normalized", "norm_period"]` |
| `MarketForceVolume` | `volume/market_force` | Market Force (F = m * a) | [dynamics.py](src/signalflow/ta/volume/dynamics.py) | `["period", "normalized", "norm_period"]` |
| `ImpulseVolume` | `volume/impulse` | Market Impulse (J = sum F * dt) | [dynamics.py](src/signalflow/ta/volume/dynamics.py) | `["period", "normalized", "norm_period"]` |
| `MarketMomentumVolume` | `volume/market_momentum` | Market Momentum (p = m * v) | [dynamics.py](src/signalflow/ta/volume/dynamics.py) | `["period", "normalized", "norm_period"]` |
| `MarketPowerVolume` | `volume/market_power` | Market Power (P = F * v) | [dynamics.py](src/signalflow/ta/volume/dynamics.py) | `["period", "normalized", "norm_period"]` |
| `MarketCapacitanceVolume` | `volume/market_capacitance` | Market Capacitance (volume absorbed per unit price change) | [dynamics.py](src/signalflow/ta/volume/dynamics.py) | `["period", "normalized", "norm_period"]` |
| `GravitationalPullVolume` | `volume/gravitational_pull` | Gravitational Pull (volume-weighted attraction) | [dynamics.py](src/signalflow/ta/volume/dynamics.py) | `["period", "normalized", "norm_period"]` |

### Trend (22 indicators)

| Class | SF Name | Description | Source | Parameters |
|-------|---------|-------------|--------|------------|
| `AdxTrend` | `trend/adx` | Average Directional Index | [strength.py](src/signalflow/ta/trend/strength.py) | `["period", "normalized"]` |
| `AroonTrend` | `trend/aroon` | Aroon Indicator | [strength.py](src/signalflow/ta/trend/strength.py) | `["period", "normalized"]` |
| `VortexTrend` | `trend/vortex` | Vortex Indicator | [strength.py](src/signalflow/ta/trend/strength.py) | `["period", "normalized"]` |
| `VhfTrend` | `trend/vhf` | Vertical Horizontal Filter | [strength.py](src/signalflow/ta/trend/strength.py) | `["period", "normalized"]` |
| `ChopTrend` | `trend/chop` | Choppiness Index | [strength.py](src/signalflow/ta/trend/strength.py) | `["period", "normalized"]` |
| `ViscosityTrend` | `trend/viscosity` | Viscosity (resistance to velocity change) | [physics.py](src/signalflow/ta/trend/physics.py) | `["period", "normalized", "norm_period"]` |
| `ReynoldsTrend` | `trend/reynolds` | Reynolds Number (laminar vs turbulent regime) | [physics.py](src/signalflow/ta/trend/physics.py) | `["period", "normalized", "norm_period"]` |
| `RotationalInertiaTrend` | `trend/rotational_inertia` | Rotational Inertia (resistance to trend change) | [physics.py](src/signalflow/ta/trend/physics.py) | `["period", "normalized", "norm_period"]` |
| `MarketImpedanceTrend` | `trend/market_impedance` | Market Impedance (Z = V/I analogy) | [physics.py](src/signalflow/ta/trend/physics.py) | `["period", "normalized", "norm_period"]` |
| `RCTimeConstantTrend` | `trend/rc_time_constant` | RC Time Constant (circuit analogue) | [physics.py](src/signalflow/ta/trend/physics.py) | `["period", "normalized", "norm_period"]` |
| `SNRTrend` | `trend/snr` | Signal-to-Noise Ratio | [physics.py](src/signalflow/ta/trend/physics.py) | `["period", "normalized", "norm_period"]` |
| `OrderParameterTrend` | `trend/order_parameter` | Order Parameter (phase transition) | [physics.py](src/signalflow/ta/trend/physics.py) | `["period", "normalized", "norm_period"]` |
| `SusceptibilityTrend` | `trend/susceptibility` | Susceptibility (response to changes) | [physics.py](src/signalflow/ta/trend/physics.py) | `["period", "normalized", "norm_period"]` |
| `PsarTrend` | `trend/psar` | Parabolic SAR | [stops.py](src/signalflow/ta/trend/stops.py) | `["iaf", "maxaf"]` |
| `SupertrendTrend` | `trend/supertrend` | Supertrend | [stops.py](src/signalflow/ta/trend/stops.py) | `["period", "multiplier", "normalized", "norm_period"]` |
| `ChandelierTrend` | `trend/chandelier` | Chandelier Stop | [stops.py](src/signalflow/ta/trend/stops.py) | `["period", "atr_period", "multiplier"]` |
| `HiloTrend` | `trend/hilo` | HiLo Channel | [stops.py](src/signalflow/ta/trend/stops.py) | `["period", "normalized", "norm_period"]` |
| `CkspTrend` | `trend/cksp` | Coppock Curve | [stops.py](src/signalflow/ta/trend/stops.py) | `["roc_period", "ema_period", "normalized", "norm_period"]` |
| `IchimokuTrend` | `trend/ichimoku` | Ichimoku Kinko Hyo | [detection.py](src/signalflow/ta/trend/detection.py) | `["tenkan", "kijun", "senkou", "normalized", "norm_period"]` |
| `DpoTrend` | `trend/dpo` | Detrended Price Oscillator | [detection.py](src/signalflow/ta/trend/detection.py) | `["period", "normalized", "norm_period"]` |
| `QstickTrend` | `trend/qstick` | Qstick | [detection.py](src/signalflow/ta/trend/detection.py) | `["period", "normalized", "norm_period"]` |
| `TtmTrend` | `trend/ttm` | TTM Trend | [detection.py](src/signalflow/ta/trend/detection.py) | `["length", "normalized", "norm_period"]` |

### Statistics (62 indicators)

| Class | SF Name | Description | Source | Parameters |
|-------|---------|-------------|--------|------------|
| `VarianceStat` | `stat/variance` | Rolling Variance | [dispersion.py](src/signalflow/ta/stat/dispersion.py) | `["source_col", "period", "ddof"]` |
| `StdStat` | `stat/std` | Rolling Standard Deviation | [dispersion.py](src/signalflow/ta/stat/dispersion.py) | `["source_col", "period", "ddof"]` |
| `MadStat` | `stat/mad` | Mean Absolute Deviation | [dispersion.py](src/signalflow/ta/stat/dispersion.py) | `["source_col", "period"]` |
| `ZscoreStat` | `stat/zscore` | Z-Score | [dispersion.py](src/signalflow/ta/stat/dispersion.py) | `["source_col", "period"]` |
| `CvStat` | `stat/cv` | Coefficient of Variation | [dispersion.py](src/signalflow/ta/stat/dispersion.py) | `["source_col", "period"]` |
| `RangeStat` | `stat/range` | Range (max - min) | [dispersion.py](src/signalflow/ta/stat/dispersion.py) | `["source_col", "period"]` |
| `IqrStat` | `stat/iqr` | Interquartile Range | [dispersion.py](src/signalflow/ta/stat/dispersion.py) | `["source_col", "period"]` |
| `AadStat` | `stat/aad` | Average Absolute Deviation | [dispersion.py](src/signalflow/ta/stat/dispersion.py) | `["source_col", "period"]` |
| `RobustZscoreStat` | `stat/robust_zscore` | Robust Z-Score (via MAD) | [dispersion.py](src/signalflow/ta/stat/dispersion.py) | `["source_col", "period"]` |
| `MedianStat` | `stat/median` | Rolling Median | [distribution.py](src/signalflow/ta/stat/distribution.py) | `["source_col", "period"]` |
| `QuantileStat` | `stat/quantile` | Quantile | [distribution.py](src/signalflow/ta/stat/distribution.py) | `["source_col", "period", "quantile"]` |
| `PctRankStat` | `stat/pct_rank` | Percentile Rank | [distribution.py](src/signalflow/ta/stat/distribution.py) | `["source_col", "period"]` |
| `MinMaxStat` | `stat/minmax` | Min-Max Scaler | [distribution.py](src/signalflow/ta/stat/distribution.py) | `["source_col", "period"]` |
| `SkewStat` | `stat/skew` | Skewness | [distribution.py](src/signalflow/ta/stat/distribution.py) | `["source_col", "period"]` |
| `KurtosisStat` | `stat/kurtosis` | Kurtosis | [distribution.py](src/signalflow/ta/stat/distribution.py) | `["source_col", "period"]` |
| `EntropyStat` | `stat/entropy` | Entropy | [distribution.py](src/signalflow/ta/stat/distribution.py) | `["source_col", "period", "bins"]` |
| `JarqueBeraStat` | `stat/jarque_bera` | Jarque-Bera Test | [distribution.py](src/signalflow/ta/stat/distribution.py) | `["source_col", "period"]` |
| `ModeDistanceStat` | `stat/mode_distance` | Distance to Mode | [distribution.py](src/signalflow/ta/stat/distribution.py) | `["source_col", "period"]` |
| `AboveMeanRatioStat` | `stat/above_mean_ratio` | Ratio of Values Above Mean | [distribution.py](src/signalflow/ta/stat/distribution.py) | `["source_col", "period"]` |
| `EntropyRateStat` | `stat/entropy_rate` | Entropy Rate (speed of information change) | [distribution.py](src/signalflow/ta/stat/distribution.py) | `["source_col", "period", "lag", "base", "normalized", "norm_period"]` |
| `HurstStat` | `stat/hurst` | Hurst Exponent | [memory.py](src/signalflow/ta/stat/memory.py) | `["source_col", "period", "min_lag"]` |
| `AutocorrStat` | `stat/autocorr` | Autocorrelation | [memory.py](src/signalflow/ta/stat/memory.py) | `["source_col", "period", "lag"]` |
| `VarianceRatioStat` | `stat/variance_ratio` | Variance Ratio | [memory.py](src/signalflow/ta/stat/memory.py) | `["source_col", "period", "lag"]` |
| `DiffusionCoeffStat` | `stat/diffusion_coeff` | Diffusion Coefficient (D = Var / 2dt) | [memory.py](src/signalflow/ta/stat/memory.py) | `["source_col", "period", "normalized", "norm_period"]` |
| `AnomalousDiffusionStat` | `stat/anomalous_diffusion` | Anomalous Diffusion Exponent | [memory.py](src/signalflow/ta/stat/memory.py) | `["source_col", "period", "tau_short", "tau_long", "normalized", "norm_period"]` |
| `MsdRatioStat` | `stat/msd_ratio` | Mean Squared Displacement Ratio | [memory.py](src/signalflow/ta/stat/memory.py) | `["source_col", "period", "tau", "normalized", "norm_period"]` |
| `SpringConstantStat` | `stat/spring_constant` | Spring Constant (mean-reversion strength) | [memory.py](src/signalflow/ta/stat/memory.py) | `["source_col", "period", "ma_period", "normalized", "norm_period"]` |
| `DampingRatioStat` | `stat/damping_ratio` | Damping Ratio (oscillation decay) | [memory.py](src/signalflow/ta/stat/memory.py) | `["source_col", "period", "ma_period", "normalized", "norm_period"]` |
| `NaturalFrequencyStat` | `stat/natural_frequency` | Natural Frequency (dominant oscillation) | [memory.py](src/signalflow/ta/stat/memory.py) | `["source_col", "period", "ma_period", "normalized", "norm_period"]` |
| `PlasticStrainStat` | `stat/plastic_strain` | Plastic Strain (non-reversible deformation ratio) | [memory.py](src/signalflow/ta/stat/memory.py) | `["source_col", "period", "ma_period", "normalized", "norm_period"]` |
| `EscapeVelocityStat` | `stat/escape_velocity` | Escape Velocity (velocity to break from MA) | [memory.py](src/signalflow/ta/stat/memory.py) | `["source_col", "period", "ma_period", "normalized", "norm_period"]` |
| `CorrelationLengthStat` | `stat/correlation_length` | Correlation Length (autocorrelation zero-crossing) | [memory.py](src/signalflow/ta/stat/memory.py) | `["source_col", "period", "max_lag", "normalized", "norm_period"]` |
| `InstAmplitudeStat` | `stat/inst_amplitude` | Instantaneous Amplitude (Hilbert transform) | [cycle.py](src/signalflow/ta/stat/cycle.py) | `["source_col", "period", "normalized", "norm_period"]` |
| `InstPhaseStat` | `stat/inst_phase` | Instantaneous Phase (Hilbert transform) | [cycle.py](src/signalflow/ta/stat/cycle.py) | `["source_col", "period", "normalized"]` |
| `InstFrequencyStat` | `stat/inst_frequency` | Instantaneous Frequency (Hilbert transform) | [cycle.py](src/signalflow/ta/stat/cycle.py) | `["source_col", "period", "normalized", "norm_period"]` |
| `PhaseAccelerationStat` | `stat/phase_acceleration` | Phase Acceleration (2nd derivative of phase) | [cycle.py](src/signalflow/ta/stat/cycle.py) | `["source_col", "period", "normalized", "norm_period"]` |
| `ConstructiveInterferenceStat` | `stat/constructive_interference` | Constructive Interference (phase-aligned amplitude) | [cycle.py](src/signalflow/ta/stat/cycle.py) | `["source_col", "fast_period", "slow_period", "smooth", "normalized", "norm_period"]` |
| `BeatFrequencyStat` | `stat/beat_frequency` | Beat Frequency (two-oscillator difference) | [cycle.py](src/signalflow/ta/stat/cycle.py) | `["source_col", "fast_period", "slow_period", "normalized", "norm_period"]` |
| `StandingWaveRatioStat` | `stat/standing_wave_ratio` | Standing Wave Ratio (market resonance) | [cycle.py](src/signalflow/ta/stat/cycle.py) | `["source_col", "period", "swr_window", "normalized", "norm_period"]` |
| `SpectralCentroidStat` | `stat/spectral_centroid` | Spectral Centroid (frequency center of mass) | [cycle.py](src/signalflow/ta/stat/cycle.py) | `["source_col", "period", "normalized", "norm_period"]` |
| `SpectralEntropyStat` | `stat/spectral_entropy` | Spectral Entropy (frequency distribution disorder) | [cycle.py](src/signalflow/ta/stat/cycle.py) | `["source_col", "period", "normalized"]` |
| `CorrelationStat` | `stat/correlation` | Correlation | [regression.py](src/signalflow/ta/stat/regression.py) | `["col1", "col2", "period"]` |
| `BetaStat` | `stat/beta` | Beta Coefficient | [regression.py](src/signalflow/ta/stat/regression.py) | `["col1", "col2", "period"]` |
| `RSquaredStat` | `stat/rsquared` | R-Squared | [regression.py](src/signalflow/ta/stat/regression.py) | `["col1", "col2", "period"]` |
| `LinRegSlopeStat` | `stat/linreg_slope` | Linear Regression Slope | [regression.py](src/signalflow/ta/stat/regression.py) | `["source_col", "period"]` |
| `LinRegInterceptStat` | `stat/linreg_intercept` | Linear Regression Intercept | [regression.py](src/signalflow/ta/stat/regression.py) | `["source_col", "period"]` |
| `LinRegResidualStat` | `stat/linreg_residual` | Linear Regression Residual | [regression.py](src/signalflow/ta/stat/regression.py) | `["source_col", "period"]` |
| `RealizedVolStat` | `stat/realized_vol` | Realized Volatility | [realized.py](src/signalflow/ta/stat/realized.py) | `["source_col", "period"]` |
| `ParkinsonVolStat` | `stat/parkinson_vol` | Parkinson Volatility | [realized.py](src/signalflow/ta/stat/realized.py) | `["period"]` |
| `GarmanKlassVolStat` | `stat/garman_klass_vol` | Garman-Klass Volatility | [realized.py](src/signalflow/ta/stat/realized.py) | `["period"]` |
| `RogersSatchellVolStat` | `stat/rogers_satchell_vol` | Rogers-Satchell Volatility | [realized.py](src/signalflow/ta/stat/realized.py) | `["period"]` |
| `YangZhangVolStat` | `stat/yang_zhang_vol` | Yang-Zhang Volatility | [realized.py](src/signalflow/ta/stat/realized.py) | `["period"]` |
| `PermutationEntropyStat` | `stat/permutation_entropy` | Permutation Entropy (ordinal pattern complexity) | [complexity.py](src/signalflow/ta/stat/complexity.py) | `["source_col", "period", "m", "normalized", "norm_period"]` |
| `SampleEntropyStat` | `stat/sample_entropy` | Sample Entropy (regularity / self-similarity) | [complexity.py](src/signalflow/ta/stat/complexity.py) | `["source_col", "period", "m", "r_mult", "normalized", "norm_period"]` |
| `LempelZivStat` | `stat/lempel_ziv` | Lempel-Ziv Complexity (compressibility) | [complexity.py](src/signalflow/ta/stat/complexity.py) | `["source_col", "period", "normalized", "norm_period"]` |
| `FisherInformationStat` | `stat/fisher_information` | Fisher Information (distribution sharpness) | [complexity.py](src/signalflow/ta/stat/complexity.py) | `["source_col", "period", "bins", "normalized", "norm_period"]` |
| `DfaExponentStat` | `stat/dfa` | DFA Exponent (detrended long-range correlations) | [complexity.py](src/signalflow/ta/stat/complexity.py) | `["source_col", "period", "min_box", "normalized", "norm_period"]` |
| `KLDivergenceStat` | `stat/kl_divergence` | KL Divergence (regime deviation from baseline) | [information.py](src/signalflow/ta/stat/information.py) | `["source_col", "period", "short_period", "bins", "normalized", "norm_period"]` |
| `JSDivergenceStat` | `stat/js_divergence` | Jensen-Shannon Divergence (symmetric regime change) | [information.py](src/signalflow/ta/stat/information.py) | `["source_col", "period", "short_period", "bins", "normalized", "norm_period"]` |
| `RenyiEntropyStat` | `stat/renyi_entropy` | Rényi Entropy (generalized tail-sensitive entropy) | [information.py](src/signalflow/ta/stat/information.py) | `["source_col", "period", "alpha", "bins", "normalized", "norm_period"]` |
| `AutoMutualInfoStat` | `stat/auto_mutual_info` | Auto Mutual Information (nonlinear temporal dependency) | [information.py](src/signalflow/ta/stat/information.py) | `["source_col", "period", "lag", "bins", "normalized", "norm_period"]` |
| `RelativeInfoGainStat` | `stat/info_gain` | Relative Information Gain (distributional change rate) | [information.py](src/signalflow/ta/stat/information.py) | `["source_col", "period", "bins", "normalized", "norm_period"]` |

### Performance (2 indicators)

| Class | SF Name | Description | Source | Parameters |
|-------|---------|-------------|--------|------------|
| `LogReturn` | `perf/log_ret` | Logarithmic Returns | [returns.py](src/signalflow/ta/performance/returns.py) | `["source", "period"]` |
| `PctReturn` | `perf/pct_ret` | Percentage Returns | [returns.py](src/signalflow/ta/performance/returns.py) | `["source", "period"]` |

### Divergence (2 indicators)

| Class | SF Name | Description | Source | Parameters |
|-------|---------|-------------|--------|------------|
| `RsiDivergence` | `divergence/rsi` | RSI Divergence Detector (regular & hidden) | [rsi_div.py](src/signalflow/ta/divergence/rsi_div.py) | `["rsi_period", "rsi_overbought", "rsi_oversold", "pivot_window", "min_pivot_distance", "lookback", "min_divergence_magnitude"]` |
| `MacdDivergence` | `divergence/macd` | MACD Divergence Detector (regular & hidden) | [macd_div.py](src/signalflow/ta/divergence/macd_div.py) | `["fast", "slow", "signal", "pivot_window", "min_pivot_distance", "lookback", "min_divergence_magnitude"]` |

## Normalization

Most indicators support a `normalized` parameter. When `normalized=True`, output values are transformed to relative scales:

- **Bounded indicators** (RSI, Stochastic, Williams %R): scaled to `[0, 1]` or `[-1, 1]`
- **Unbounded indicators** (MACD, ROC, smoothers): normalized via rolling z-score
- Normalized columns receive a `_norm` suffix

## AutoFeatureNormalizer

`AutoFeatureNormalizer` is an automatic normalizer for Polars DataFrames. It analyzes the statistical properties of each feature and selects the optimal normalization method.

```python
from signalflow.ta.auto_norm import AutoFeatureNormalizer

norm = AutoFeatureNormalizer(window=256, warmup=256)
artifact = norm.fit(df)
df_normalized = norm.transform(df)
# or in one step:
df_normalized = norm.fit_transform(df)
```

Normalization methods:

- `rolling_robust` -- `(x - median) / IQR`
- `rolling_z` -- `(x - mean) / std`
- `rolling_rank` -- percentile rank within a window
- `rolling_winsor` -- quantile clipping (outlier resistance)
- `signed_log1p` -- `sign(x) * log1p(|x|)` (for skewed distributions)

The `fit()` artifact can be saved/loaded via `save()`/`load()` for reproducibility.

## License

See [signalflow-trading](https://pypi.org/project/signalflow-trading/) for license details.

<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../logo-dark.svg" width="120">
  <source media="(prefers-color-scheme: light)" srcset="../logo.svg" width="120">
  <img alt="SignalFlow" src="../logo.png" width="120">
</picture>

# signalflow-ta

**Technical analysis extension for SignalFlow — 189 indicators + 24 signal detectors**

<p>
<a href="https://pypi.org/project/signalflow-ta/"><img src="https://img.shields.io/badge/version-0.6.0-7c3aed" alt="Version"></a>
<a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.12+-3b82f6?logo=python&logoColor=white" alt="Python 3.12+"></a>
<a href="https://signalflow-trading.com"><img src="https://img.shields.io/badge/docs-signalflow--trading.com-7c3aed" alt="Docs"></a>
</p>

</div>

---

Part of the [SignalFlow](https://github.com/pathway2nothing/sf-project) ecosystem.

189 technical analysis indicators organized into 8 modules, 24 signal detectors with configurable filters, and `AutoFeatureNormalizer` for automatic normalization.

## Installation

```bash
pip install signalflow-ta
```

**Requires:** Python ≥ 3.12, signalflow-trading ≥ 0.5.0, pandas-ta ≥ 0.4.67b0

## Usage

```python
import signalflow.ta as ta

from signalflow.ta.momentum import RsiMom, MacdMom
from signalflow.ta.volatility import BollingerVol
from signalflow.ta.signals import BollingerBandDetector1, RsiZscoreFilter

# Use as features
rsi = RsiMom(period=14)
macd = MacdMom(fast=12, slow=26, signal=9)
bb = BollingerVol(period=20, std_dev=2.0)

# Use as detector with filters
detector = BollingerBandDetector1(
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

---

## Signal Detectors (24)

All detectors support configurable `direction` (`"long"`, `"short"`, `"both"`) and optional filters.

| Category | Detectors |
|----------|-----------|
| **Momentum** | RSI Anomaly, CCI Anomaly, Stochastic ×2 |
| **Volume** | MFI Oversold/Overbought, MFI Z-Score |
| **Trend** | Aroon Cross, ADX Regime ×2 |
| **Volatility** | Bollinger Breakout, Keltner + RSI, Keltner + MACD |
| **Divergence** | Price/RSI, Price/MACD |
| **Filter-based** | Hampel ×2, Adaptive Kalman |
| **Market Condition** | Volatility Regime, Cross-Sectional, Market Breadth |
| **ML-based** | Isolation Forest (Returns, RSI, Cross-Sectional) |
| **Cross-pair** | Correlation + Bollinger |

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

"""
SignalFlow-TA Indicator Registry

Automatically discovers and configures all Feature-based indicators
from signalflow.ta for comprehensive testing.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Type, Any, get_type_hints
import importlib

TEST_PARAMS_LIMIT = 1


@dataclass
class IndicatorConfig:
    """Configuration for testing an indicator."""

    cls: Type
    name: str
    category: str
    params: dict = field(default_factory=dict)
    requires: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    bounded: tuple[float, float] | None = None
    warmup: int = 20
    variation_idx: int | None = None  # Index in test_params list

    @property
    def test_id(self) -> str:
        """Generate test ID for pytest parametrization.

        Avoids '/' and nested '[]' which conflict with pytest node ID parsing
        and break VSCode test discovery.
        """
        base = f"{self.category}-{self.name}"
        if self.variation_idx is not None:
            # Create readable param string: period=7,std=2.0
            param_str = ",".join(
                f"{k}={v}"
                for k, v in self.params.items()
                if k not in ("source_col", "pair_col", "ts_col")
            )
            return f"{base}({param_str})"
        return base


def get_default_params(cls: Type) -> dict:
    """Extract default parameters from indicator class."""
    params = {}

    # Get from dataclass fields
    if hasattr(cls, "__dataclass_fields__"):
        for name, field_info in cls.__dataclass_fields__.items():
            # Skip inherited fields from base classes, test metadata, and private fields
            if name in (
                "pair_col",
                "ts_col",
                "offset_col",
                "offset_window",
                "compute_last_offset",
                "use_resample",
                "resample_mode",
                "resample_prefix",
                "raw_data_type",
                "component_type",
                "keep_input_columns",
                "requires",
                "outputs",
                "test_params",
            ):  # ClassVar for test parameter variations
                continue
            if name.startswith("_"):
                continue

            # Get default value
            import dataclasses as _dc

            if field_info.default is not _dc.MISSING:
                if field_info.default is not None and not callable(field_info.default):
                    params[name] = field_info.default
            elif field_info.default_factory is not _dc.MISSING:
                try:
                    params[name] = field_info.default_factory()
                except:
                    pass

    return params


def get_requires(cls: Type) -> list[str]:
    """Extract required columns from indicator class."""
    # Check class attribute
    if hasattr(cls, "requires"):
        req = cls.requires
        if isinstance(req, (list, tuple)):
            # Handle template strings like "{source_col}"
            return [r for r in req if not r.startswith("{")]

    # Default based on category
    return ["close"]


def get_outputs(cls: Type, params: dict) -> list[str]:
    """Extract output column names from indicator class."""
    if hasattr(cls, "outputs"):
        outputs = cls.outputs
        if isinstance(outputs, (list, tuple)):
            # Substitute params into template strings
            result = []
            for out in outputs:
                if "{" in out:
                    try:
                        result.append(out.format(**params))
                    except KeyError:
                        result.append(out.split("_")[0])
                else:
                    result.append(out)
            return result

    # Generate from class name
    name = cls.__name__.lower()
    for suffix in (
        "mom",
        "smooth",
        "price",
        "vol",
        "stat",
        "trend",
        "overlay",
        "volume",
    ):
        name = name.replace(suffix, "")

    period = params.get("period", params.get("length", ""))
    if period:
        return [f"{name}_{period}"]
    return [name]


def get_bounds(cls: Type, category: str) -> tuple[float, float] | None:
    """Determine value bounds based on indicator type."""
    name = cls.__name__.lower()

    # Explicitly unbounded indicators (percentage change can exceed 100%)
    if name in ["rocmom", "mommom"]:
        return None

    # RSI: 0-100
    if name in ["rsimom", "stochmom", "stochrsimom", "mfimom", "mfivolume"]:
        return (0, 100)

    # Williams %R: -100 to 0
    if "willr" in name:
        return (-100, 0)

    # CMO: -100 to 100
    if "cmo" in name:
        return (-100, 100)

    # ADX: 0-100
    if name == "adxtrend":
        return (0, 100)

    # Aroon: 0-100 for up/down
    if "aroon" in name:
        return (0, 100)

    # Choppiness: 0-100
    if "chop" in name:
        return (0, 100)

    # UO (Ultimate Oscillator): 0-100
    if name == "uomom":
        return (0, 100)

    # TSI: -100 to 100
    if name == "tsimom":
        return (-100, 100)

    # Hurst: 0-1
    if "hurst" in name:
        return (0, 1)

    # Correlation/Autocorr: -1 to 1
    if name in ["correlationstat", "autocorrstat"]:
        return (-1, 1)

    # R-squared: 0-1
    if name == "rsquaredstat":
        return (0, 1)

    # MinMax normalization: 0-1
    if name == "minmaxstat":
        return (0, 1)

    # AboveMeanRatio: 0-1
    if name == "abovemeanratiostat":
        return (0, 1)

    # CMF (Chaikin Money Flow): -1 to 1
    if name == "cmfvolume":
        return (-1, 1)

    # Positive-only volatility measures
    if name in ["truerangevol", "atrvol", "natrvol", "ulcerindexvol"]:
        return (0, float("inf"))

    # Positive-only stats
    if name in [
        "variancestat",
        "stdevstat",
        "madstat",
        "cvstat",
        "rangestat",
        "iqrstat",
        "aadstat",
        "entropystat",
        "jarqueberastat",
        "realizedvolstat",
        "parkinsonvolstat",
        "garmanklassvolstat",
        "rogerssatchellvolstat",
        "yangzhangvolstat",
    ]:
        return (0, float("inf"))

    # Most indicators are unbounded
    return None


def get_warmup(cls: Type, params: dict) -> int:
    """Estimate warmup period based on parameters.

    Handles various indicator parameter patterns:
    - Single period: period, length
    - Multiple periods: slow, fast, signal, rsi_period, stoch_period
    - Composite indicators need sum of dependent periods
    """
    warmup = 2000

    # StochRSI-like: needs RSI warmup + Stoch warmup
    if "rsi_period" in params:
        warmup += int(params["rsi_period"])
    if "stoch_period" in params:
        warmup += int(params["stoch_period"])
    if "k_period" in params:
        warmup += int(params["k_period"])
    if "d_period" in params:
        warmup += int(params["d_period"])

    # MACD-like: slow period dominates
    if "slow" in params:
        warmup = max(warmup, int(params["slow"]))
    if "signal" in params:
        warmup += int(params["signal"])

    # TSI/TRIX: multiple smoothing passes
    if "fast" in params and "slow" in params:
        warmup = max(warmup, int(params["slow"]) + int(params["fast"]))

    # Single period indicators
    if "period" in params:
        warmup = max(warmup, int(params["period"]))
    if "length" in params:
        warmup = max(warmup, int(params["length"]))
    if "long_period" in params:
        warmup = max(warmup, int(params["long_period"]))

    # Minimum warmup
    if warmup == 0:
        warmup = 20

    # Add buffer
    return warmup + 10


def discover_indicators() -> list[IndicatorConfig]:
    """
    Discover all Feature-based indicators from signalflow.ta.

    If an indicator class has `test_params` attribute (ClassVar[list[dict]]),
    creates multiple configs - one for each parameter set.

    Example indicator with test params:

        from typing import ClassVar

        @dataclass
        class RsiMom(Feature):
            period: int = 14

            # Parameter sets for testing
            test_params: ClassVar[list[dict]] = [
                {"period": 7},
                {"period": 14},
                {"period": 21},
            ]

    This will generate 3 test configs with IDs:
    - "momentum/RsiMom[period=7]"
    - "momentum/RsiMom[period=14]"
    - "momentum/RsiMom[period=21]"

    Returns:
        List of IndicatorConfig for all discovered indicators
    """
    configs = []

    try:
        import signalflow.ta as ta
        from signalflow.feature.base import Feature, GlobalFeature
    except ImportError as e:
        print(f"Warning: Could not import signalflow.ta: {e}")
        return configs

    # Get all exported names
    all_exports = getattr(ta, "__all__", dir(ta))

    # Category mapping based on naming convention
    category_map = {
        "Mom": "momentum",
        "Smooth": "overlap",
        "Price": "overlap",
        "Overlay": "overlap",
        "Vol": "volatility",
        "Stat": "stat",
        "Trend": "trend",
        "Volume": "volume",
        "Return": "performance",
    }

    for name in all_exports:
        try:
            cls = getattr(ta, name)

            # Check if it's a class that inherits from Feature
            if not inspect.isclass(cls):
                continue

            # Check inheritance
            if not issubclass(cls, Feature) or cls is Feature:
                continue

            # Skip GlobalFeature subclasses (they use compute(), not compute_pair())
            if issubclass(cls, GlobalFeature) and cls is not GlobalFeature:
                continue

            # Determine category
            category = "other"
            for suffix, cat in category_map.items():
                if suffix in name:
                    category = cat
                    break

            # Check for test_params attribute (ClassVar[list[dict]])
            test_params = getattr(cls, "test_params", None)
            test_params = test_params if test_params else [get_default_params(cls)]
            if test_params and isinstance(test_params, list) and len(test_params) > 0:
                # Create config for each parameter set
                for idx, param_set in enumerate(test_params):
                    if not isinstance(param_set, dict):
                        continue

                    # Merge default params with test params
                    base_params = get_default_params(cls)
                    params = {**base_params, **param_set}

                    requires = get_requires(cls)
                    outputs = get_outputs(cls, params)
                    bounds = get_bounds(cls, category)
                    warmup = get_warmup(cls, params)

                    config = IndicatorConfig(
                        cls=cls,
                        name=name,
                        category=category,
                        params=params,
                        requires=requires,
                        outputs=outputs,
                        bounded=bounds,
                        warmup=warmup,
                        variation_idx=idx,
                    )
                    configs.append(config)
            else:
                # No test_params - use default params
                params = get_default_params(cls)
                requires = get_requires(cls)
                outputs = get_outputs(cls, params)
                bounds = get_bounds(cls, category)
                warmup = get_warmup(cls, params)

                config = IndicatorConfig(
                    cls=cls,
                    name=name,
                    category=category,
                    params=params,
                    requires=requires,
                    outputs=outputs,
                    bounded=bounds,
                    warmup=warmup,
                )
                configs.append(config)

        except Exception as e:
            print(f"Warning: Could not process {name}: {e}")
            continue

    return configs


# =============================================================================
# Manual Configuration for All Known Indicators
# =============================================================================


def get_all_indicator_configs() -> list[IndicatorConfig]:
    """
    Get configurations for all known signalflow.ta indicators.

    This provides explicit configuration when auto-discovery isn't available.
    """

    configs = []

    try:
        from signalflow.ta import (
            # Momentum
            RsiMom,
            RocMom,
            MomMom,
            CmoMom,
            StochMom,
            StochRsiMom,
            WillrMom,
            CciMom,
            UoMom,
            AoMom,
            MacdMom,
            PpoMom,
            TsiMom,
            TrixMom,
            # Overlap - Smoothers
            SmaSmooth,
            EmaSmooth,
            WmaSmooth,
            RmaSmooth,
            DemaSmooth,
            TemaSmooth,
            HmaSmooth,
            TrimaSmooth,
            SwmaSmooth,
            SsfSmooth,
            # Overlap - Adaptive
            KamaSmooth,
            AlmaSmooth,
            JmaSmooth,
            VidyaSmooth,
            T3Smooth,
            ZlmaSmooth,
            McGinleySmooth,
            FramaSmooth,
            # Overlap - Price
            Hl2Price,
            Hlc3Price,
            Ohlc4Price,
            WcpPrice,
            MidpointPrice,
            MidpricePrice,
            TypicalPrice,
            # Overlap - Trend
            SupertrendOverlay,
            HiloOverlay,
            IchimokuOverlay,
            ChandelierOverlay,
            # Performance
            LogReturn,
            PctReturn,
            # Stat - Dispersion
            VarianceStat,
            StdevStat,
            MadStat,
            ZscoreStat,
            CvStat,
            RangeStat,
            IqrStat,
            AadStat,
            RobustZscoreStat,
            # Stat - Distribution
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
            # Stat - Memory
            HurstStat,
            AutocorrStat,
            VarianceRatioStat,
            # Stat - Regression
            CorrelationStat,
            BetaStat,
            RSquaredStat,
            LinRegSlopeStat,
            LinRegInterceptStat,
            LinRegResidualStat,
            # Stat - Volatility
            RealizedVolStat,
            ParkinsonVolStat,
            GarmanKlassVolStat,
            RogersSatchellVolStat,
            YangZhangVolStat,
            # Trend
            AdxTrend,
            AroonTrend,
            VortexTrend,
            VhfTrend,
            ChopTrend,
            PsarTrend,
            SupertrendTrend,
            ChandelierTrend,
            HiloTrend,
            CkspTrend,
            IchimokuTrend,
            DpoTrend,
            QstickTrend,
            TtmTrend,
            # Volatility
            TrueRangeVol,
            AtrVol,
            NatrVol,
            BollingerVol,
            KeltnerVol,
            DonchianVol,
            AccBandsVol,
            MassIndexVol,
            UlcerIndexVol,
            RviVol,
            # Volume
            ObvVolume,
            AdVolume,
            PvtVolume,
            NviVolume,
            PviVolume,
            MfiVolume,
            CmfVolume,
            EfiVolume,
            EomVolume,
            KvoVolume,
            # Gap
            GapVol,
        )
    except ImportError as e:
        print(f"Could not import signalflow.ta indicators: {e}")
        return []

    # ==========================================================================
    # MOMENTUM INDICATORS
    # ==========================================================================

    configs.extend(
        [
            IndicatorConfig(
                cls=RsiMom,
                name="RsiMom",
                category="momentum",
                params={"period": 14},
                requires=["close"],
                outputs=["rsi_14"],
                bounded=(0, 100),
                warmup=15,
            ),
            IndicatorConfig(
                cls=RocMom,
                name="RocMom",
                category="momentum",
                params={"period": 10},
                requires=["close"],
                outputs=["roc_10"],
                bounded=None,  # ROC is unbounded - can be >100% if price doubles
                warmup=11,
            ),
            IndicatorConfig(
                cls=MomMom,
                name="MomMom",
                category="momentum",
                params={"period": 10},
                requires=["close"],
                outputs=["mom_10"],
                bounded=None,
                warmup=11,
            ),
            IndicatorConfig(
                cls=CmoMom,
                name="CmoMom",
                category="momentum",
                params={"period": 14},
                requires=["close"],
                outputs=["cmo_14"],
                bounded=(-100, 100),
                warmup=15,
            ),
            IndicatorConfig(
                cls=StochMom,
                name="StochMom",
                category="momentum",
                params={"k_period": 14, "d_period": 3},
                requires=["high", "low", "close"],
                outputs=["stoch_k_14", "stoch_d_3"],
                bounded=(0, 100),
                warmup=17,
            ),
            IndicatorConfig(
                cls=StochRsiMom,
                name="StochRsiMom",
                category="momentum",
                params={
                    "rsi_period": 14,
                    "stoch_period": 14,
                    "k_period": 3,
                    "d_period": 3,
                },
                requires=["close"],
                outputs=["stochrsi_k", "stochrsi_d"],
                bounded=(0, 100),
                warmup=35,
            ),
            IndicatorConfig(
                cls=WillrMom,
                name="WillrMom",
                category="momentum",
                params={"period": 14},
                requires=["high", "low", "close"],
                outputs=["willr_14"],
                bounded=(-100, 0),
                warmup=14,
            ),
            IndicatorConfig(
                cls=CciMom,
                name="CciMom",
                category="momentum",
                params={"period": 20},
                requires=["high", "low", "close"],
                outputs=["cci_20"],
                bounded=None,  # Typically ±200 but unbounded
                warmup=20,
            ),
            IndicatorConfig(
                cls=UoMom,
                name="UoMom",
                category="momentum",
                params={"fast": 7, "medium": 14, "slow": 28},
                requires=["high", "low", "close"],
                outputs=["uo_7_14_28"],
                bounded=(0, 100),
                warmup=30,
            ),
            IndicatorConfig(
                cls=AoMom,
                name="AoMom",
                category="momentum",
                params={"fast": 5, "slow": 34},
                requires=["high", "low"],
                outputs=["ao_5_34"],
                bounded=None,
                warmup=35,
            ),
            IndicatorConfig(
                cls=MacdMom,
                name="MacdMom",
                category="momentum",
                params={"fast": 12, "slow": 26, "signal": 9},
                requires=["close"],
                outputs=["macd_12_26", "macd_signal_9", "macd_hist_12_26"],
                bounded=None,
                warmup=35,
            ),
            IndicatorConfig(
                cls=PpoMom,
                name="PpoMom",
                category="momentum",
                params={"fast": 12, "slow": 26, "signal": 9},
                requires=["close"],
                outputs=["ppo_12_26", "ppo_signal_9", "ppo_hist_12_26"],
                bounded=None,
                warmup=35,
            ),
            IndicatorConfig(
                cls=TsiMom,
                name="TsiMom",
                category="momentum",
                params={"fast": 13, "slow": 25},
                requires=["close"],
                outputs=["tsi_13_25"],
                bounded=(-100, 100),
                warmup=40,
            ),
            IndicatorConfig(
                cls=TrixMom,
                name="TrixMom",
                category="momentum",
                params={"period": 15},
                requires=["close"],
                outputs=["trix_15"],
                bounded=None,
                warmup=50,
            ),
        ]
    )

    # ==========================================================================
    # OVERLAP - SMOOTHERS
    # ==========================================================================

    configs.extend(
        [
            IndicatorConfig(
                cls=SmaSmooth,
                name="SmaSmooth",
                category="overlap",
                params={"period": 20, "source_col": "close"},
                requires=["close"],
                outputs=["sma_20"],
                bounded=None,
                warmup=20,
            ),
            IndicatorConfig(
                cls=EmaSmooth,
                name="EmaSmooth",
                category="overlap",
                params={"period": 20, "source_col": "close"},
                requires=["close"],
                outputs=["ema_20"],
                bounded=None,
                warmup=20,
            ),
            IndicatorConfig(
                cls=WmaSmooth,
                name="WmaSmooth",
                category="overlap",
                params={"period": 20, "source_col": "close"},
                requires=["close"],
                outputs=["wma_20"],
                bounded=None,
                warmup=20,
            ),
            IndicatorConfig(
                cls=RmaSmooth,
                name="RmaSmooth",
                category="overlap",
                params={"period": 20, "source_col": "close"},
                requires=["close"],
                outputs=["rma_20"],
                bounded=None,
                warmup=20,
            ),
            IndicatorConfig(
                cls=DemaSmooth,
                name="DemaSmooth",
                category="overlap",
                params={"period": 20, "source_col": "close"},
                requires=["close"],
                outputs=["dema_20"],
                bounded=None,
                warmup=40,
            ),
            IndicatorConfig(
                cls=TemaSmooth,
                name="TemaSmooth",
                category="overlap",
                params={"period": 20, "source_col": "close"},
                requires=["close"],
                outputs=["tema_20"],
                bounded=None,
                warmup=60,
            ),
            IndicatorConfig(
                cls=HmaSmooth,
                name="HmaSmooth",
                category="overlap",
                params={"period": 20, "source_col": "close"},
                requires=["close"],
                outputs=["hma_20"],
                bounded=None,
                warmup=25,
            ),
            IndicatorConfig(
                cls=TrimaSmooth,
                name="TrimaSmooth",
                category="overlap",
                params={"period": 20, "source_col": "close"},
                requires=["close"],
                outputs=["trima_20"],
                bounded=None,
                warmup=20,
            ),
            IndicatorConfig(
                cls=SwmaSmooth,
                name="SwmaSmooth",
                category="overlap",
                params={"source_col": "close"},
                requires=["close"],
                outputs=["swma"],
                bounded=None,
                warmup=5,
            ),
            IndicatorConfig(
                cls=SsfSmooth,
                name="SsfSmooth",
                category="overlap",
                params={"period": 20, "poles": 2, "source_col": "close"},
                requires=["close"],
                outputs=["ssf_20"],
                bounded=None,
                warmup=25,
            ),
        ]
    )

    # ==========================================================================
    # OVERLAP - ADAPTIVE SMOOTHERS
    # ==========================================================================

    configs.extend(
        [
            IndicatorConfig(
                cls=KamaSmooth,
                name="KamaSmooth",
                category="overlap",
                params={"period": 10, "fast": 2, "slow": 30, "source_col": "close"},
                requires=["close"],
                outputs=["kama_10"],
                bounded=None,
                warmup=35,
            ),
            IndicatorConfig(
                cls=AlmaSmooth,
                name="AlmaSmooth",
                category="overlap",
                params={
                    "period": 20,
                    "offset": 0.85,
                    "sigma": 6,
                    "source_col": "close",
                },
                requires=["close"],
                outputs=["alma_20"],
                bounded=None,
                warmup=20,
            ),
            IndicatorConfig(
                cls=JmaSmooth,
                name="JmaSmooth",
                category="overlap",
                params={"period": 7, "phase": 0, "source_col": "close"},
                requires=["close"],
                outputs=["jma_7"],
                bounded=None,
                warmup=15,
            ),
            IndicatorConfig(
                cls=VidyaSmooth,
                name="VidyaSmooth",
                category="overlap",
                params={"period": 14, "source_col": "close"},
                requires=["close"],
                outputs=["vidya_14"],
                bounded=None,
                warmup=20,
            ),
            IndicatorConfig(
                cls=T3Smooth,
                name="T3Smooth",
                category="overlap",
                params={"period": 5, "vfactor": 0.7, "source_col": "close"},
                requires=["close"],
                outputs=["t3_5"],
                bounded=None,
                warmup=35,
            ),
            IndicatorConfig(
                cls=ZlmaSmooth,
                name="ZlmaSmooth",
                category="overlap",
                params={"period": 20, "source_col": "close"},
                requires=["close"],
                outputs=["zlma_20"],
                bounded=None,
                warmup=25,
            ),
            IndicatorConfig(
                cls=McGinleySmooth,
                name="McGinleySmooth",
                category="overlap",
                params={"period": 14, "source_col": "close"},
                requires=["close"],
                outputs=["mcginley_14"],
                bounded=None,
                warmup=15,
            ),
            IndicatorConfig(
                cls=FramaSmooth,
                name="FramaSmooth",
                category="overlap",
                params={"period": 16, "source_col": "close"},
                requires=["close"],
                outputs=["frama_16"],
                bounded=None,
                warmup=20,
            ),
        ]
    )

    # ==========================================================================
    # OVERLAP - PRICE TRANSFORMS
    # ==========================================================================

    configs.extend(
        [
            IndicatorConfig(
                cls=Hl2Price,
                name="Hl2Price",
                category="overlap",
                params={},
                requires=["high", "low"],
                outputs=["hl2"],
                bounded=None,
                warmup=1,
            ),
            IndicatorConfig(
                cls=Hlc3Price,
                name="Hlc3Price",
                category="overlap",
                params={},
                requires=["high", "low", "close"],
                outputs=["hlc3"],
                bounded=None,
                warmup=1,
            ),
            IndicatorConfig(
                cls=Ohlc4Price,
                name="Ohlc4Price",
                category="overlap",
                params={},
                requires=["open", "high", "low", "close"],
                outputs=["ohlc4"],
                bounded=None,
                warmup=1,
            ),
            IndicatorConfig(
                cls=WcpPrice,
                name="WcpPrice",
                category="overlap",
                params={},
                requires=["high", "low", "close"],
                outputs=["wcp"],
                bounded=None,
                warmup=1,
            ),
            IndicatorConfig(
                cls=MidpointPrice,
                name="MidpointPrice",
                category="overlap",
                params={"period": 14, "source_col": "close"},
                requires=["close"],
                outputs=["midpoint_14"],
                bounded=None,
                warmup=14,
            ),
            IndicatorConfig(
                cls=MidpricePrice,
                name="MidpricePrice",
                category="overlap",
                params={"period": 14},
                requires=["high", "low"],
                outputs=["midprice_14"],
                bounded=None,
                warmup=14,
            ),
            IndicatorConfig(
                cls=TypicalPrice,
                name="TypicalPrice",
                category="overlap",
                params={},
                requires=["high", "low", "close"],
                outputs=["typical"],
                bounded=None,
                warmup=1,
            ),
        ]
    )

    # ==========================================================================
    # OVERLAP - TREND OVERLAYS
    # ==========================================================================

    configs.extend(
        [
            IndicatorConfig(
                cls=SupertrendOverlay,
                name="SupertrendOverlay",
                category="overlap",
                params={"period": 10, "multiplier": 3.0},
                requires=["high", "low", "close"],
                outputs=["supertrend_10", "supertrend_direction_10"],
                bounded=None,
                warmup=15,
            ),
            IndicatorConfig(
                cls=HiloOverlay,
                name="HiloOverlay",
                category="overlap",
                params={"period": 14},
                requires=["high", "low", "close"],
                outputs=["hilo_14"],
                bounded=None,
                warmup=15,
            ),
            IndicatorConfig(
                cls=IchimokuOverlay,
                name="IchimokuOverlay",
                category="overlap",
                params={"tenkan": 9, "kijun": 26, "senkou": 52},
                requires=["high", "low", "close"],
                outputs=[
                    "ichimoku_tenkan",
                    "ichimoku_kijun",
                    "ichimoku_senkou_a",
                    "ichimoku_senkou_b",
                    "ichimoku_chikou",
                ],
                bounded=None,
                warmup=55,
            ),
            IndicatorConfig(
                cls=ChandelierOverlay,
                name="ChandelierOverlay",
                category="overlap",
                params={"period": 22, "multiplier": 3.0},
                requires=["high", "low", "close"],
                outputs=["chandelier_long_22", "chandelier_short_22"],
                bounded=None,
                warmup=25,
            ),
        ]
    )

    # ==========================================================================
    # PERFORMANCE
    # ==========================================================================

    configs.extend(
        [
            IndicatorConfig(
                cls=LogReturn,
                name="LogReturn",
                category="performance",
                params={"period": 1, "source_col": "close"},
                requires=["close"],
                outputs=["log_return_1"],
                bounded=None,
                warmup=2,
            ),
            IndicatorConfig(
                cls=PctReturn,
                name="PctReturn",
                category="performance",
                params={"period": 1, "source_col": "close"},
                requires=["close"],
                outputs=["pct_return_1"],
                bounded=None,
                warmup=2,
            ),
        ]
    )

    # ==========================================================================
    # STAT - DISPERSION
    # ==========================================================================

    configs.extend(
        [
            IndicatorConfig(
                cls=VarianceStat,
                name="VarianceStat",
                category="stat",
                params={"period": 30, "source_col": "close"},
                requires=["close"],
                outputs=["close_var_30"],
                bounded=(0, float("inf")),
                warmup=30,
            ),
            IndicatorConfig(
                cls=StdevStat,
                name="StdevStat",
                category="stat",
                params={"period": 30, "source_col": "close"},
                requires=["close"],
                outputs=["close_std_30"],
                bounded=(0, float("inf")),
                warmup=30,
            ),
            IndicatorConfig(
                cls=MadStat,
                name="MadStat",
                category="stat",
                params={"period": 30, "source_col": "close"},
                requires=["close"],
                outputs=["close_mad_30"],
                bounded=(0, float("inf")),
                warmup=30,
            ),
            IndicatorConfig(
                cls=ZscoreStat,
                name="ZscoreStat",
                category="stat",
                params={"period": 30, "source_col": "close"},
                requires=["close"],
                outputs=["close_zscore_30"],
                bounded=None,
                warmup=30,
            ),
            IndicatorConfig(
                cls=CvStat,
                name="CvStat",
                category="stat",
                params={"period": 30, "source_col": "close"},
                requires=["close"],
                outputs=["close_cv_30"],
                bounded=(0, float("inf")),
                warmup=30,
            ),
            IndicatorConfig(
                cls=RangeStat,
                name="RangeStat",
                category="stat",
                params={"period": 30, "source_col": "close"},
                requires=["close"],
                outputs=["close_range_30"],
                bounded=(0, float("inf")),
                warmup=30,
            ),
            IndicatorConfig(
                cls=IqrStat,
                name="IqrStat",
                category="stat",
                params={"period": 30, "source_col": "close"},
                requires=["close"],
                outputs=["close_iqr_30"],
                bounded=(0, float("inf")),
                warmup=30,
            ),
            IndicatorConfig(
                cls=AadStat,
                name="AadStat",
                category="stat",
                params={"period": 30, "source_col": "close"},
                requires=["close"],
                outputs=["close_aad_30"],
                bounded=(0, float("inf")),
                warmup=30,
            ),
            IndicatorConfig(
                cls=RobustZscoreStat,
                name="RobustZscoreStat",
                category="stat",
                params={"period": 30, "source_col": "close"},
                requires=["close"],
                outputs=["close_robustz_30"],
                bounded=None,
                warmup=30,
            ),
        ]
    )

    # ==========================================================================
    # STAT - DISTRIBUTION
    # ==========================================================================

    configs.extend(
        [
            IndicatorConfig(
                cls=MedianStat,
                name="MedianStat",
                category="stat",
                params={"period": 30, "source_col": "close"},
                requires=["close"],
                outputs=["close_median_30"],
                bounded=None,
                warmup=30,
            ),
            IndicatorConfig(
                cls=QuantileStat,
                name="QuantileStat",
                category="stat",
                params={"period": 30, "quantile": 0.5, "source_col": "close"},
                requires=["close"],
                outputs=["close_q50_30"],
                bounded=None,
                warmup=30,
            ),
            IndicatorConfig(
                cls=PctRankStat,
                name="PctRankStat",
                category="stat",
                params={"period": 30, "source_col": "close"},
                requires=["close"],
                outputs=["close_pctrank_30"],
                bounded=(0, 100),
                warmup=30,
            ),
            IndicatorConfig(
                cls=MinMaxStat,
                name="MinMaxStat",
                category="stat",
                params={"period": 30, "source_col": "close"},
                requires=["close"],
                outputs=["close_minmax_30"],
                bounded=(0, 1),
                warmup=30,
            ),
            IndicatorConfig(
                cls=SkewStat,
                name="SkewStat",
                category="stat",
                params={"period": 30, "source_col": "close"},
                requires=["close"],
                outputs=["close_skew_30"],
                bounded=None,
                warmup=30,
            ),
            IndicatorConfig(
                cls=KurtosisStat,
                name="KurtosisStat",
                category="stat",
                params={"period": 30, "source_col": "close"},
                requires=["close"],
                outputs=["close_kurt_30"],
                bounded=None,
                warmup=30,
            ),
            IndicatorConfig(
                cls=EntropyStat,
                name="EntropyStat",
                category="stat",
                params={"period": 10, "source_col": "close"},
                requires=["close"],
                outputs=["close_entropy_10"],
                bounded=(0, float("inf")),
                warmup=10,
            ),
            IndicatorConfig(
                cls=JarqueBeraStat,
                name="JarqueBeraStat",
                category="stat",
                params={"period": 30, "source_col": "close"},
                requires=["close"],
                outputs=["close_jb_30"],
                bounded=(0, float("inf")),
                warmup=30,
            ),
            IndicatorConfig(
                cls=ModeDistanceStat,
                name="ModeDistanceStat",
                category="stat",
                params={"period": 30, "source_col": "close"},
                requires=["close"],
                outputs=["close_modedist_30"],
                bounded=None,
                warmup=30,
            ),
            IndicatorConfig(
                cls=AboveMeanRatioStat,
                name="AboveMeanRatioStat",
                category="stat",
                params={"period": 30, "source_col": "close"},
                requires=["close"],
                outputs=["close_abovemean_30"],
                bounded=(0, 1),
                warmup=30,
            ),
        ]
    )

    # ==========================================================================
    # STAT - MEMORY
    # ==========================================================================

    configs.extend(
        [
            IndicatorConfig(
                cls=HurstStat,
                name="HurstStat",
                category="stat",
                params={"period": 100, "source_col": "close"},
                requires=["close"],
                outputs=["close_hurst_100"],
                bounded=(0, 1),
                warmup=100,
            ),
            IndicatorConfig(
                cls=AutocorrStat,
                name="AutocorrStat",
                category="stat",
                params={"period": 30, "lag": 1, "source_col": "close"},
                requires=["close"],
                outputs=["close_autocorr_30_1"],
                bounded=(-1, 1),
                warmup=30,
            ),
            IndicatorConfig(
                cls=VarianceRatioStat,
                name="VarianceRatioStat",
                category="stat",
                params={"period": 30, "lag": 2, "source_col": "close"},
                requires=["close"],
                outputs=["close_varratio_30_2"],
                bounded=None,
                warmup=30,
            ),
        ]
    )

    # ==========================================================================
    # STAT - REGRESSION
    # ==========================================================================

    configs.extend(
        [
            IndicatorConfig(
                cls=CorrelationStat,
                name="CorrelationStat",
                category="stat",
                params={"period": 30, "source_col": "close", "target_col": "volume"},
                requires=["close", "volume"],
                outputs=["close_volume_corr_30"],
                bounded=(-1, 1),
                warmup=30,
            ),
            IndicatorConfig(
                cls=BetaStat,
                name="BetaStat",
                category="stat",
                params={"period": 30, "source_col": "close", "benchmark_col": "volume"},
                requires=["close", "volume"],
                outputs=["close_volume_beta_30"],
                bounded=None,
                warmup=30,
            ),
            IndicatorConfig(
                cls=RSquaredStat,
                name="RSquaredStat",
                category="stat",
                params={"period": 30, "source_col": "close"},
                requires=["close"],
                outputs=["close_rsq_30"],
                bounded=(0, 1),
                warmup=30,
            ),
            IndicatorConfig(
                cls=LinRegSlopeStat,
                name="LinRegSlopeStat",
                category="stat",
                params={"period": 30, "source_col": "close"},
                requires=["close"],
                outputs=["close_linreg_slope_30"],
                bounded=None,
                warmup=30,
            ),
            IndicatorConfig(
                cls=LinRegInterceptStat,
                name="LinRegInterceptStat",
                category="stat",
                params={"period": 30, "source_col": "close"},
                requires=["close"],
                outputs=["close_linreg_intercept_30"],
                bounded=None,
                warmup=30,
            ),
            IndicatorConfig(
                cls=LinRegResidualStat,
                name="LinRegResidualStat",
                category="stat",
                params={"period": 30, "source_col": "close"},
                requires=["close"],
                outputs=["close_linreg_residual_30"],
                bounded=None,
                warmup=30,
            ),
        ]
    )

    # ==========================================================================
    # STAT - VOLATILITY
    # ==========================================================================

    configs.extend(
        [
            IndicatorConfig(
                cls=RealizedVolStat,
                name="RealizedVolStat",
                category="stat",
                params={"period": 30, "source_col": "close"},
                requires=["close"],
                outputs=["close_realvol_30"],
                bounded=(0, float("inf")),
                warmup=30,
            ),
            IndicatorConfig(
                cls=ParkinsonVolStat,
                name="ParkinsonVolStat",
                category="stat",
                params={"period": 30},
                requires=["high", "low"],
                outputs=["parkinson_30"],
                bounded=(0, float("inf")),
                warmup=30,
            ),
            IndicatorConfig(
                cls=GarmanKlassVolStat,
                name="GarmanKlassVolStat",
                category="stat",
                params={"period": 30},
                requires=["open", "high", "low", "close"],
                outputs=["garmanklass_30"],
                bounded=(0, float("inf")),
                warmup=30,
            ),
            IndicatorConfig(
                cls=RogersSatchellVolStat,
                name="RogersSatchellVolStat",
                category="stat",
                params={"period": 30},
                requires=["open", "high", "low", "close"],
                outputs=["rogerssatchell_30"],
                bounded=(0, float("inf")),
                warmup=30,
            ),
            IndicatorConfig(
                cls=YangZhangVolStat,
                name="YangZhangVolStat",
                category="stat",
                params={"period": 30},
                requires=["open", "high", "low", "close"],
                outputs=["yangzhang_30"],
                bounded=(0, float("inf")),
                warmup=30,
            ),
        ]
    )

    # ==========================================================================
    # TREND
    # ==========================================================================

    configs.extend(
        [
            IndicatorConfig(
                cls=AdxTrend,
                name="AdxTrend",
                category="trend",
                params={"period": 14},
                requires=["high", "low", "close"],
                outputs=["adx_14", "plus_di_14", "minus_di_14"],
                bounded=(0, 100),
                warmup=30,
            ),
            IndicatorConfig(
                cls=AroonTrend,
                name="AroonTrend",
                category="trend",
                params={"period": 25},
                requires=["high", "low"],
                outputs=["aroon_up_25", "aroon_down_25", "aroon_osc_25"],
                bounded=(0, 100),  # for up/down; osc is -100 to 100
                warmup=26,
            ),
            IndicatorConfig(
                cls=VortexTrend,
                name="VortexTrend",
                category="trend",
                params={"period": 14},
                requires=["high", "low", "close"],
                outputs=["vortex_plus_14", "vortex_minus_14"],
                bounded=None,
                warmup=15,
            ),
            IndicatorConfig(
                cls=VhfTrend,
                name="VhfTrend",
                category="trend",
                params={"period": 28},
                requires=["close"],
                outputs=["vhf_28"],
                bounded=(0, float("inf")),
                warmup=28,
            ),
            IndicatorConfig(
                cls=ChopTrend,
                name="ChopTrend",
                category="trend",
                params={"period": 14},
                requires=["high", "low", "close"],
                outputs=["chop_14"],
                bounded=(0, 100),
                warmup=15,
            ),
            IndicatorConfig(
                cls=PsarTrend,
                name="PsarTrend",
                category="trend",
                params={"af_start": 0.02, "af_max": 0.2},
                requires=["high", "low", "close"],
                outputs=["psar", "psar_direction"],
                bounded=None,
                warmup=5,
            ),
            IndicatorConfig(
                cls=SupertrendTrend,
                name="SupertrendTrend",
                category="trend",
                params={"period": 10, "multiplier": 3.0},
                requires=["high", "low", "close"],
                outputs=["supertrend_10", "supertrend_direction_10"],
                bounded=None,
                warmup=15,
            ),
            IndicatorConfig(
                cls=ChandelierTrend,
                name="ChandelierTrend",
                category="trend",
                params={"period": 22, "multiplier": 3.0},
                requires=["high", "low", "close"],
                outputs=["chandelier_long_22", "chandelier_short_22"],
                bounded=None,
                warmup=25,
            ),
            IndicatorConfig(
                cls=HiloTrend,
                name="HiloTrend",
                category="trend",
                params={"period": 14},
                requires=["high", "low", "close"],
                outputs=["hilo_14"],
                bounded=None,
                warmup=15,
            ),
            IndicatorConfig(
                cls=CkspTrend,
                name="CkspTrend",
                category="trend",
                params={"p": 10, "x": 1, "q": 9},
                requires=["high", "low", "close"],
                outputs=["cksp_long", "cksp_short"],
                bounded=None,
                warmup=20,
            ),
            IndicatorConfig(
                cls=IchimokuTrend,
                name="IchimokuTrend",
                category="trend",
                params={"tenkan": 9, "kijun": 26, "senkou": 52},
                requires=["high", "low", "close"],
                outputs=[
                    "ichimoku_tenkan",
                    "ichimoku_kijun",
                    "ichimoku_senkou_a",
                    "ichimoku_senkou_b",
                ],
                bounded=None,
                warmup=55,
            ),
            IndicatorConfig(
                cls=DpoTrend,
                name="DpoTrend",
                category="trend",
                params={"period": 20},
                requires=["close"],
                outputs=["dpo_20"],
                bounded=None,
                warmup=25,
            ),
            IndicatorConfig(
                cls=QstickTrend,
                name="QstickTrend",
                category="trend",
                params={"period": 14},
                requires=["open", "close"],
                outputs=["qstick_14"],
                bounded=None,
                warmup=14,
            ),
            IndicatorConfig(
                cls=TtmTrend,
                name="TtmTrend",
                category="trend",
                params={"period": 20},
                requires=["high", "low", "close"],
                outputs=["ttm_squeeze_20", "ttm_momentum_20"],
                bounded=None,
                warmup=25,
            ),
        ]
    )

    # ==========================================================================
    # VOLATILITY
    # ==========================================================================

    configs.extend(
        [
            IndicatorConfig(
                cls=TrueRangeVol,
                name="TrueRangeVol",
                category="volatility",
                params={},
                requires=["high", "low", "close"],
                outputs=["true_range"],
                bounded=(0, float("inf")),
                warmup=2,
            ),
            IndicatorConfig(
                cls=AtrVol,
                name="AtrVol",
                category="volatility",
                params={"period": 14},
                requires=["high", "low", "close"],
                outputs=["atr_14"],
                bounded=(0, float("inf")),
                warmup=15,
            ),
            IndicatorConfig(
                cls=NatrVol,
                name="NatrVol",
                category="volatility",
                params={"period": 14},
                requires=["high", "low", "close"],
                outputs=["natr_14"],
                bounded=(0, float("inf")),
                warmup=15,
            ),
            IndicatorConfig(
                cls=BollingerVol,
                name="BollingerVol",
                category="volatility",
                params={"period": 20, "std_dev": 2.0},
                requires=["close"],
                outputs=[
                    "bb_upper_20",
                    "bb_mid_20",
                    "bb_lower_20",
                    "bb_width_20",
                    "bb_pct_20",
                ],
                bounded=None,
                warmup=20,
            ),
            IndicatorConfig(
                cls=KeltnerVol,
                name="KeltnerVol",
                category="volatility",
                params={"period": 20, "atr_period": 10, "multiplier": 2.0},
                requires=["high", "low", "close"],
                outputs=["kc_upper_20", "kc_mid_20", "kc_lower_20"],
                bounded=None,
                warmup=25,
            ),
            IndicatorConfig(
                cls=DonchianVol,
                name="DonchianVol",
                category="volatility",
                params={"period": 20},
                requires=["high", "low"],
                outputs=["dc_upper_20", "dc_mid_20", "dc_lower_20"],
                bounded=None,
                warmup=20,
            ),
            IndicatorConfig(
                cls=AccBandsVol,
                name="AccBandsVol",
                category="volatility",
                params={"period": 20},
                requires=["high", "low", "close"],
                outputs=["accbands_upper_20", "accbands_mid_20", "accbands_lower_20"],
                bounded=None,
                warmup=20,
            ),
            IndicatorConfig(
                cls=MassIndexVol,
                name="MassIndexVol",
                category="volatility",
                params={"fast": 9, "slow": 25},
                requires=["high", "low"],
                outputs=["mass_index_9_25"],
                bounded=(0, float("inf")),
                warmup=35,
            ),
            IndicatorConfig(
                cls=UlcerIndexVol,
                name="UlcerIndexVol",
                category="volatility",
                params={"period": 14},
                requires=["close"],
                outputs=["ulcer_14"],
                bounded=(0, float("inf")),
                warmup=15,
            ),
            IndicatorConfig(
                cls=RviVol,
                name="RviVol",
                category="volatility",
                params={"period": 14},
                requires=["high", "low", "close"],
                outputs=["rvi_14"],
                bounded=(0, float("inf")),
                warmup=20,
            ),
        ]
    )

    # ==========================================================================
    # VOLUME
    # ==========================================================================

    configs.extend(
        [
            IndicatorConfig(
                cls=ObvVolume,
                name="ObvVolume",
                category="volume",
                params={},
                requires=["close", "volume"],
                outputs=["obv"],
                bounded=None,
                warmup=2,
            ),
            IndicatorConfig(
                cls=AdVolume,
                name="AdVolume",
                category="volume",
                params={},
                requires=["high", "low", "close", "volume"],
                outputs=["ad"],
                bounded=None,
                warmup=2,
            ),
            IndicatorConfig(
                cls=PvtVolume,
                name="PvtVolume",
                category="volume",
                params={},
                requires=["close", "volume"],
                outputs=["pvt"],
                bounded=None,
                warmup=2,
            ),
            IndicatorConfig(
                cls=NviVolume,
                name="NviVolume",
                category="volume",
                params={},
                requires=["close", "volume"],
                outputs=["nvi"],
                bounded=None,
                warmup=2,
            ),
            IndicatorConfig(
                cls=PviVolume,
                name="PviVolume",
                category="volume",
                params={},
                requires=["close", "volume"],
                outputs=["pvi"],
                bounded=None,
                warmup=2,
            ),
            IndicatorConfig(
                cls=MfiVolume,
                name="MfiVolume",
                category="volume",
                params={"period": 14},
                requires=["high", "low", "close", "volume"],
                outputs=["mfi_14"],
                bounded=(0, 100),
                warmup=15,
            ),
            IndicatorConfig(
                cls=CmfVolume,
                name="CmfVolume",
                category="volume",
                params={"period": 20},
                requires=["high", "low", "close", "volume"],
                outputs=["cmf_20"],
                bounded=(-1, 1),
                warmup=20,
            ),
            IndicatorConfig(
                cls=EfiVolume,
                name="EfiVolume",
                category="volume",
                params={"period": 13},
                requires=["close", "volume"],
                outputs=["efi_13"],
                bounded=None,
                warmup=14,
            ),
            IndicatorConfig(
                cls=EomVolume,
                name="EomVolume",
                category="volume",
                params={"period": 14, "divisor": 10000},
                requires=["high", "low", "volume"],
                outputs=["eom_14"],
                bounded=None,
                warmup=15,
            ),
            IndicatorConfig(
                cls=KvoVolume,
                name="KvoVolume",
                category="volume",
                params={"fast": 34, "slow": 55, "signal": 13},
                requires=["high", "low", "close", "volume"],
                outputs=["kvo_34_55", "kvo_signal_13"],
                bounded=None,
                warmup=60,
            ),
            IndicatorConfig(
                cls=GapVol,
                name="GapVol",
                category="volatility",
                params={"min_gap_pct": 0.0},
                requires=["open", "high", "low", "close"],
                outputs=[
                    "gap_val",
                    "gap_pct",
                    "gap_fill_pct",
                    "gap_run_ratio",
                    "gap_range_ratio",
                    "is_gap_up",
                    "is_gap_down",
                ],
                bounded=None,
                warmup=1,
            ),
        ]
    )

    return configs


# =============================================================================
# Export
# =============================================================================


def _load_indicator_configs() -> list[IndicatorConfig]:
    """Load indicator configs with detailed error reporting."""

    # Indicators that require non-standard columns (exclude from auto-testing)
    EXCLUDED_INDICATORS = {
        "BetaStat",  # requires benchmark_col which is not in standard OHLCV
    }

    # Try dynamic discovery first
    try:
        configs = discover_indicators()
        if configs:
            # Filter out excluded indicators
            configs = [c for c in configs if c.name not in EXCLUDED_INDICATORS]
            print(
                f"[indicator_registry] Auto-discovered {len(configs)} indicators from signalflow.ta"
            )
            return configs
    except ImportError as e:
        print(f"[indicator_registry] signalflow.ta not available: {e}")
    except Exception as e:
        print(f"[indicator_registry] Auto-discovery failed: {e}")

    # Try manual configuration
    try:
        configs = get_all_indicator_configs()
        if configs:
            # Filter out excluded indicators
            configs = [c for c in configs if c.name not in EXCLUDED_INDICATORS]
            print(
                f"[indicator_registry] Loaded {len(configs)} indicator configs manually"
            )
            return configs
    except ImportError as e:
        print(
            f"[indicator_registry] Manual config failed - signalflow.ta not installed: {e}"
        )
    except Exception as e:
        print(f"[indicator_registry] Manual config failed: {e}")

    # Return empty list if nothing works
    print(
        "[indicator_registry] WARNING: No indicators available. Install signalflow.ta to run indicator tests."
    )
    print("[indicator_registry] Core framework tests in test_core.py will still run.")
    return []


# Load configs at module import time
INDICATOR_CONFIGS = _load_indicator_configs()


def get_configs_by_category(category: str) -> list[IndicatorConfig]:
    """Get indicator configs filtered by category."""
    return [c for c in INDICATOR_CONFIGS if c.category == category]


def get_indicator_ids() -> list[str]:
    """Get test IDs for pytest parametrization.

    Returns IDs like:
    - "momentum/RsiMom" (no test_params)
    - "momentum/RsiMom[period=7]" (with test_params)
    - "momentum/RsiMom[period=14,smoothing=ema]" (multiple params)
    """
    if not INDICATOR_CONFIGS:
        return []  # Empty list - tests will be skipped via pytestmark
    return [c.test_id for c in INDICATOR_CONFIGS]


# =============================================================================
# Test Configuration Filtering
# =============================================================================


def filter_configs_by_options(
    configs: list[IndicatorConfig], pytest_config=None
) -> tuple[list, list]:
    """
    Filter indicator configs based on pytest command-line options.

    Parameters
    ----------
    configs : list[IndicatorConfig]
        All available indicator configurations
    pytest_config : pytest.Config, optional
        Pytest config object to access options

    Returns
    -------
    filtered_configs : list[IndicatorConfig]
        Filtered configs
    filtered_ids : list[str]
        IDs for the filtered configs
    """
    if not configs:
        return [None], ["no_indicators"]

    filtered = list(configs)  # Make a copy

    # Apply feature group filtering if specified
    if pytest_config and hasattr(pytest_config, "test_feature_groups"):
        feature_groups = pytest_config.test_feature_groups
        if feature_groups:
            allowed_groups = {g.strip().lower() for g in feature_groups.split(",")}
            filtered = [c for c in filtered if c.category.lower() in allowed_groups]

    # Apply max params limit if specified
    if pytest_config and hasattr(pytest_config, "test_max_params"):
        max_params = pytest_config.test_max_params
        if max_params is not None and max_params > 0:
            # Group configs by indicator name (without params)
            from collections import defaultdict

            by_indicator = defaultdict(list)
            for config in filtered:
                # Extract base name (e.g., "momentum/RsiMom" from "momentum/RsiMom[...]")
                base_name = config.name.split("[")[0]
                by_indicator[base_name].append(config)

            # Keep only first N param sets for each indicator
            filtered = []
            for base_name, param_configs in sorted(by_indicator.items()):
                filtered.extend(param_configs[:max_params])

    if not filtered:
        return [None], ["no_matching_indicators"]

    return filtered, [c.test_id for c in filtered]


# Parametrization helpers for test_indicators.py
_CONFIGS_FOR_PARAM = INDICATOR_CONFIGS if INDICATOR_CONFIGS else [None]
_IDS_FOR_PARAM = get_indicator_ids() if INDICATOR_CONFIGS else ["no_indicators"]


def print_available_indicators():
    """Print summary of available indicators for debugging."""
    if not INDICATOR_CONFIGS:
        print("No indicators loaded.")
        return

    by_category = {}
    for c in INDICATOR_CONFIGS:
        by_category.setdefault(c.category, []).append(c)

    print(f"\nAvailable indicators ({len(INDICATOR_CONFIGS)} total configs):")
    for cat, configs in sorted(by_category.items()):
        print(f"  {cat}: {len(configs)} configs")
        for config in sorted(configs, key=lambda x: x.test_id):
            params_info = ", ".join(
                f"{k}={v}"
                for k, v in config.params.items()
                if k not in ("source_col", "pair_col", "ts_col")
            )
            variation_marker = (
                f" #{config.variation_idx}" if config.variation_idx is not None else ""
            )
            print(f"    - {config.name}{variation_marker}: {params_info}")

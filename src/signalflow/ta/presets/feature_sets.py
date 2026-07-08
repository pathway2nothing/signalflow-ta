"""Curated stationary feature-set preset.

``STATIONARY_CORE`` is a lean, scale-free set of oscillators, normalized
volatility, returns, and z-scores with no price levels, suited to model
training under walk-forward where levels leak.
"""

from signalflow.enums import ComponentType
from signalflow.registry import registry
from signalflow.transform.pipe import FeaturePipe

STATIONARY_CORE: tuple[str, ...] = (
    "momentum/rsi",
    "momentum/ppo",
    "momentum/roc",
    "momentum/cci",
    "momentum/cmo",
    "momentum/stoch",
    "momentum/stochrsi",
    "momentum/willr",
    "momentum/uo",
    "momentum/tsi",
    "momentum/trix",
    "trend/adx",
    "trend/aroon",
    "trend/chop",
    "trend/vhf",
    "trend/vortex",
    "volatility/atr_percent",
    "volatility/natr",
    "volatility/historical_vol",
    "volatility/rvi",
    "volatility/ulcer_index",
    "volatility/mass_index",
    "volume/cmf",
    "volume/mfi",
    "volume/kvo",
    "stat/zscore",
    "stat/robust_zscore",
    "stat/skew",
    "stat/kurtosis",
    "stat/realized_vol",
    "stat/parkinson_vol",
    "stat/garman_klass_vol",
    "stat/pctrank",
    "stat/cv",
    "stat/variance_ratio",
    "stat/autocorr",
    "perf/log_ret",
)


def stationary_core_pipe(**overrides: dict) -> FeaturePipe:
    """Build a ``FeaturePipe`` of the ``STATIONARY_CORE`` features.

    Override per-feature parameters by passing dicts keyed on registry name,
    e.g. ``stationary_core_pipe(**{"momentum/rsi": {"period": 21}})``.
    """
    transforms = [registry.create(ComponentType.TRANSFORM, name, **overrides.get(name, {})) for name in STATIONARY_CORE]
    return FeaturePipe(*transforms)

"""Preset pipeline factories for composing FeaturePipeline from pre-configured groups."""


from signalflow.ta._compat import Feature
from signalflow.ta.pipes.divergence import divergence_pipe
from signalflow.ta.pipes.momentum import (
    momentum_core_pipe,
    momentum_kinematics_pipe,
    momentum_macd_pipe,
    momentum_oscillators_pipe,
    momentum_pipe,
)
from signalflow.ta.pipes.overlap import (
    overlap_pipe,
    price_transforms_pipe,
    smoothers_pipe,
)
from signalflow.ta.pipes.performance import performance_pipe
from signalflow.ta.pipes.stat import (
    stat_complexity_pipe,
    stat_cycle_pipe,
    stat_dispersion_pipe,
    stat_distribution_pipe,
    stat_dsp_pipe,
    stat_info_theory_pipe,
    stat_memory_pipe,
    stat_pipe,
    stat_realized_vol_pipe,
    stat_regression_pipe,
)
from signalflow.ta.pipes.trend import (
    trend_detection_pipe,
    trend_pipe,
    trend_stops_pipe,
    trend_strength_pipe,
)
from signalflow.ta.pipes.volatility import (
    volatility_bands_pipe,
    volatility_energy_pipe,
    volatility_measures_pipe,
    volatility_pipe,
    volatility_range_pipe,
)
from signalflow.ta.pipes.volume import (
    volume_cumulative_pipe,
    volume_dynamics_pipe,
    volume_oscillators_pipe,
    volume_pipe,
)


def all_ta_pipe(
    *,
    source_col: str = "close",
    normalized: bool = False,
) -> list[Feature]:
    """All technical analysis indicators with default parameters.

    Composes all module-level pipes into a single feature list.
    Does NOT include CrossSectionalStat (GlobalFeature) - add it separately if needed.
    """
    return [
        *overlap_pipe(source_col=source_col, normalized=normalized),
        *momentum_pipe(source_col=source_col, normalized=normalized),
        *volatility_pipe(normalized=normalized),
        *volume_pipe(normalized=normalized),
        *trend_pipe(normalized=normalized),
        *stat_pipe(source_col=source_col),
        *performance_pipe(source_col=source_col),
        *divergence_pipe(),
    ]


__all__ = [
    "all_ta_pipe",
    "divergence_pipe",
    "momentum_core_pipe",
    "momentum_kinematics_pipe",
    "momentum_macd_pipe",
    "momentum_oscillators_pipe",
    "momentum_pipe",
    "overlap_pipe",
    "performance_pipe",
    "price_transforms_pipe",
    "smoothers_pipe",
    "stat_complexity_pipe",
    "stat_cycle_pipe",
    "stat_dispersion_pipe",
    "stat_distribution_pipe",
    "stat_dsp_pipe",
    "stat_info_theory_pipe",
    "stat_memory_pipe",
    "stat_pipe",
    "stat_realized_vol_pipe",
    "stat_regression_pipe",
    "trend_detection_pipe",
    "trend_pipe",
    "trend_stops_pipe",
    "trend_strength_pipe",
    "volatility_bands_pipe",
    "volatility_energy_pipe",
    "volatility_measures_pipe",
    "volatility_pipe",
    "volatility_range_pipe",
    "volume_cumulative_pipe",
    "volume_dynamics_pipe",
    "volume_oscillators_pipe",
    "volume_pipe",
]

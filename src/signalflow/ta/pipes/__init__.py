"""Preset pipeline factories for composing FeaturePipeline from pre-configured groups.

Usage::

    from signalflow.feature import FeaturePipeline
    from signalflow.ta.pipes import smoothers_pipe, momentum_core_pipe, all_ta_pipe

    # Compose a custom pipeline from subgroups
    pipeline = FeaturePipeline(features=[
        *smoothers_pipe(source_col="close", normalized=True),
        *momentum_core_pipe(normalized=True),
        *volatility_range_pipe(),
    ])

    # Or use everything at once
    pipeline = FeaturePipeline(features=all_ta_pipe())
"""

from __future__ import annotations

from signalflow.feature.base import Feature

# Overlap
from signalflow.ta.pipes.overlap import (
    smoothers_pipe,
    price_transforms_pipe,
    overlap_pipe,
)

# Momentum
from signalflow.ta.pipes.momentum import (
    momentum_core_pipe,
    momentum_oscillators_pipe,
    momentum_macd_pipe,
    momentum_kinematics_pipe,
    momentum_pipe,
)

# Volatility
from signalflow.ta.pipes.volatility import (
    volatility_range_pipe,
    volatility_bands_pipe,
    volatility_measures_pipe,
    volatility_energy_pipe,
    volatility_pipe,
)

# Volume
from signalflow.ta.pipes.volume import (
    volume_cumulative_pipe,
    volume_oscillators_pipe,
    volume_dynamics_pipe,
    volume_pipe,
)

# Trend
from signalflow.ta.pipes.trend import (
    trend_strength_pipe,
    trend_stops_pipe,
    trend_detection_pipe,
    trend_pipe,
)

# Stat
from signalflow.ta.pipes.stat import (
    stat_dispersion_pipe,
    stat_distribution_pipe,
    stat_memory_pipe,
    stat_cycle_pipe,
    stat_regression_pipe,
    stat_realized_vol_pipe,
    stat_complexity_pipe,
    stat_info_theory_pipe,
    stat_dsp_pipe,
    stat_pipe,
)

# Performance
from signalflow.ta.pipes.performance import performance_pipe

# Divergence
from signalflow.ta.pipes.divergence import divergence_pipe


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
    # Subgroup pipes
    "smoothers_pipe",
    "price_transforms_pipe",
    "momentum_core_pipe",
    "momentum_oscillators_pipe",
    "momentum_macd_pipe",
    "momentum_kinematics_pipe",
    "volatility_range_pipe",
    "volatility_bands_pipe",
    "volatility_measures_pipe",
    "volatility_energy_pipe",
    "volume_cumulative_pipe",
    "volume_oscillators_pipe",
    "volume_dynamics_pipe",
    "trend_strength_pipe",
    "trend_stops_pipe",
    "trend_detection_pipe",
    "stat_dispersion_pipe",
    "stat_distribution_pipe",
    "stat_memory_pipe",
    "stat_cycle_pipe",
    "stat_regression_pipe",
    "stat_realized_vol_pipe",
    "stat_complexity_pipe",
    "stat_info_theory_pipe",
    "stat_dsp_pipe",
    "performance_pipe",
    "divergence_pipe",
    # Module-level aggregates
    "overlap_pipe",
    "momentum_pipe",
    "volatility_pipe",
    "volume_pipe",
    "trend_pipe",
    "stat_pipe",
    # Top-level
    "all_ta_pipe",
]

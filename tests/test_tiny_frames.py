"""Smoke test: every ta component survives tiny frames and a full V5 frame.

Numba kernels run with bounds checking disabled. A frame shorter than a kernel's
period must not read or write out of bounds. This enumerates the whole ta registry
and computes each component on 1-, 3-, and 5-row OHLCV frames, asserting no
exception and no process crash. All-NaN output is acceptable.

It also computes every registered component on a 500-row V5 frame (time column
``ts``) to catch components that hardcode ``timestamp`` or return reduced frames.
"""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

import signalflow.ta  # noqa: F401
from signalflow.enums import ComponentType
from signalflow.registry import registry

registry.autodiscover()


NON_FRAME_SKIPS = {
    "stat/beta": "needs a precomputed 'benchmark' input column",
    "stat/time_since_spike": "needs a precomputed 'spike' input column",
}

TINY_SIZES = (1, 3, 5)
V5_FRAME_ROWS = 500


def _ta_component_names() -> list[str]:
    names = []
    for name in registry.list(ComponentType.TRANSFORM):
        info = registry.get_info(ComponentType.TRANSFORM, name)
        if info.module.startswith("signalflow.ta"):
            names.append(name)
    return names


def _make_ohlcv(n: int) -> pl.DataFrame:
    """Synthetic OHLCV frame with valid bar geometry and positive prices."""
    rng = np.random.default_rng(0)
    start = datetime(2024, 1, 1)
    ts = [start + timedelta(minutes=i) for i in range(n)]
    close = 100.0 + np.cumsum(rng.normal(0, 1.0, n))
    openp = np.empty(n)
    openp[0] = close[0]
    if n > 1:
        openp[1:] = close[:-1]
    high = np.maximum(openp, close) + np.abs(rng.normal(0, 0.5, n))
    low = np.minimum(openp, close) - np.abs(rng.normal(0, 0.5, n))
    vol = np.abs(rng.normal(1000.0, 100.0, n)) + 1.0
    return pl.DataFrame(
        {
            "pair": ["BTCUSDT"] * n,
            "ts": ts,
            "open": openp,
            "high": high,
            "low": low,
            "close": close,
            "volume": vol,
        }
    )


TINY_FRAMES = {n: _make_ohlcv(n) for n in TINY_SIZES}
V5_FRAME = _make_ohlcv(V5_FRAME_ROWS)

TA_COMPONENT_NAMES = _ta_component_names()


@pytest.mark.parametrize("n_rows", TINY_SIZES)
@pytest.mark.parametrize("name", TA_COMPONENT_NAMES)
def test_tiny_frame_does_not_crash(name: str, n_rows: int) -> None:
    if name in NON_FRAME_SKIPS:
        pytest.skip(NON_FRAME_SKIPS[name])

    component = registry.create(ComponentType.TRANSFORM, name)
    result = component.compute(TINY_FRAMES[n_rows].clone())

    assert isinstance(result, pl.DataFrame)


@pytest.mark.parametrize("name", TA_COMPONENT_NAMES)
def test_v5_frame_computes(name: str) -> None:
    """Every registered ta component computes on a 500-row V5 frame (ts column)."""
    if name in NON_FRAME_SKIPS:
        pytest.skip(NON_FRAME_SKIPS[name])

    component = registry.create(ComponentType.TRANSFORM, name)
    result = component.compute(V5_FRAME.clone())

    assert isinstance(result, pl.DataFrame)
    assert result.height == V5_FRAME.height


def test_registry_enumeration_is_non_trivial() -> None:
    """Guard against the enumeration silently collapsing to zero components."""
    assert len(TA_COMPONENT_NAMES) > 200


def _make_multi_pair(n: int, pairs: tuple[str, ...]) -> pl.DataFrame:
    return pl.concat([_make_ohlcv(n).with_columns(pl.lit(p).alias("pair")) for p in pairs])


def test_global_features_append_outputs() -> None:
    """market.py / cross_sectional.py join outputs onto the input frame (append contract)."""
    from signalflow.ta.global_features import (
        CrossSectionalReturnRank,
        MarketBreadth,
        MarketIndexFeature,
        MarketRsiFeature,
        MarketVolatilityFeature,
        MarketZscoreFeature,
    )

    df = _make_multi_pair(500, ("BTCUSDT", "ETHUSDT", "BNBUSDT"))
    for feature in (
        MarketVolatilityFeature(),
        MarketIndexFeature(),
        MarketRsiFeature(),
        MarketZscoreFeature(),
        CrossSectionalReturnRank(),
        MarketBreadth(),
    ):
        result = feature.compute(df.clone())
        assert result.height == df.height
        assert set(df.columns) <= set(result.columns)
        assert set(feature.outputs) <= set(result.columns)


def test_context_detectors_deregistered() -> None:
    """The context-only detectors are no longer registered under V5."""
    for name in (
        "ta/cross_pair_1",
        "ta/market_condition_1",
        "ta/market_condition_2",
        "ta/market_condition_3",
    ):
        assert name not in TA_COMPONENT_NAMES

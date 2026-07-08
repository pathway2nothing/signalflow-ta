"""Every STATIONARY_CORE feature resolves and computes on synthetic data."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

import signalflow.ta  # noqa: F401
from signalflow.enums import ComponentType
from signalflow.registry import registry
from signalflow.ta.presets import STATIONARY_CORE, stationary_core_pipe

registry.autodiscover()


def _make_ohlcv(n: int) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    start = datetime(2024, 1, 1)
    ts = [start + timedelta(minutes=i) for i in range(n)]
    close = 100.0 + np.cumsum(rng.normal(0, 1.0, n))
    openp = np.empty(n)
    openp[0] = close[0]
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


FRAME = _make_ohlcv(1000)


@pytest.mark.parametrize("name", STATIONARY_CORE)
def test_stationary_core_feature_computes(name: str) -> None:
    component = registry.create(ComponentType.TRANSFORM, name)
    result = component.compute(FRAME.clone())
    assert isinstance(result, pl.DataFrame)
    for col in component.outputs:
        assert col in result.columns
    assert any(result.get_column(col).drop_nulls().len() > 0 for col in component.outputs)


def test_stationary_core_pipe_builds_and_computes() -> None:
    pipe = stationary_core_pipe()
    assert len(pipe.transforms) == len(STATIONARY_CORE)
    result = pipe.compute(FRAME.clone())
    assert isinstance(result, pl.DataFrame)
    for col in pipe.outputs:
        assert col in result.columns


def test_stationary_core_pipe_accepts_overrides() -> None:
    pipe = stationary_core_pipe(**{"momentum/rsi": {"period": 21}})
    rsi = next(t for t in pipe.transforms if t.name == "momentum/rsi")
    assert rsi.period == 21

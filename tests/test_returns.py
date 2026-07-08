"""Returns features and their isolation-forest consumers are computable under V5."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

import signalflow.ta  # noqa: F401
from signalflow.enums import ComponentType
from signalflow.registry import registry
from signalflow.ta.performance import LogReturn, PctReturn

registry.autodiscover()


def _ohlcv(n: int) -> pl.DataFrame:
    rng = np.random.default_rng(1)
    ts = [datetime(2024, 1, 1) + timedelta(minutes=i) for i in range(n)]
    close = np.abs(100.0 + np.cumsum(rng.normal(0, 1.0, n))) + 5.0
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


def test_log_return_instantiates_and_declares_outputs() -> None:
    lr = LogReturn(period=5)
    assert lr.outputs == ["logret_5_close"]


def test_pct_return_instantiates_and_declares_outputs() -> None:
    pr = PctReturn(period=3)
    assert pr.outputs == ["pct_ret_3_close"]


def test_log_return_values_match_definition() -> None:
    df = _ohlcv(500)
    lr = LogReturn(period=5)
    out = lr.compute(df.clone())
    col = lr.outputs[0]
    assert col in out.columns
    close = out["close"].to_numpy()
    got = out[col].to_numpy()
    for t in (10, 123, 400):
        assert np.isclose(got[t], np.log(close[t] / close[t - 5]))
    assert np.all(np.isnan(got[:5]))


def test_pct_return_values_match_definition() -> None:
    df = _ohlcv(500)
    pr = PctReturn(period=3)
    out = pr.compute(df.clone())
    col = pr.outputs[0]
    assert col in out.columns
    close = out["close"].to_numpy()
    got = out[col].to_numpy()
    for t in (10, 123, 400):
        assert np.isclose(got[t], close[t] / close[t - 3] - 1.0)


@pytest.mark.parametrize("name", ["ta/isolation_forest_1", "ta/isolation_forest_3"])
def test_isolation_forest_features_compute(name: str) -> None:
    df = _ohlcv(500)
    detector = registry.create(ComponentType.TRANSFORM, name)
    result = detector.compute(df.clone())
    for col in detector.outputs:
        assert col in result.columns
    assert result.height == df.height

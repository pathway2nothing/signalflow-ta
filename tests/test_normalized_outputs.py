"""SPEC-031: declared ``outputs`` match produced columns under normalization.

Under ``normalized=True`` features append ``_norm``-suffixed columns via their own
``_get_output_name`` writers. The compat shim's ``outputs`` property must declare
those exact names, otherwise ``FeaturePipeline.outputs`` and schema-driven consumers lie.
"""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

import signalflow.ta  # noqa: F401
from signalflow.enums import ComponentType
from signalflow.registry import registry

registry.autodiscover()


def _make_frame(n: int) -> pl.DataFrame:
    """Single-pair OHLCV frame long enough to exercise normalization windows."""
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


FRAME = _make_frame(400)
BASE_COLS = set(FRAME.columns)


def _normalizing_feature_names() -> list[str]:
    """Registered ta features whose declared outputs change under normalization."""
    names = []
    for name in registry.list(ComponentType.TRANSFORM):
        info = registry.get_info(ComponentType.TRANSFORM, name)
        if not info.module.startswith("signalflow.ta"):
            continue
        try:
            plain = registry.create(ComponentType.TRANSFORM, name, normalized=False)
            norm = registry.create(ComponentType.TRANSFORM, name, normalized=True)
        except Exception:
            continue
        if not getattr(norm, "normalized", False):
            continue
        try:
            if set(plain.outputs) != set(norm.outputs):
                names.append(name)
        except Exception:
            continue
    return names


NORMALIZING_FEATURES = _normalizing_feature_names()


def test_normalizing_sample_is_non_trivial() -> None:
    """The normalization contract must cover a broad set, not a handful."""
    assert len(NORMALIZING_FEATURES) >= 20


@pytest.mark.parametrize("name", NORMALIZING_FEATURES)
def test_normalized_outputs_match_produced_columns(name: str) -> None:
    feature = registry.create(ComponentType.TRANSFORM, name, normalized=True)
    produced = set(feature.compute(FRAME.clone()).columns)
    declared = set(feature.outputs)

    assert declared <= produced

    appended = produced - BASE_COLS
    assert declared == appended


@pytest.mark.parametrize(
    ("name", "expected_plain", "expected_norm"),
    [
        ("volatility/mass_index", {"massi_9_25"}, {"massi_9_25_norm"}),
        ("momentum/cci", {"cci_20"}, {"cci_20_norm"}),
        (
            "momentum/macd",
            {"macd_12_26", "macd_signal_9", "macd_hist_12_26"},
            {"macd_12_26_norm", "macd_signal_9_norm", "macd_hist_12_26_norm"},
        ),
    ],
)
def test_representative_features(name: str, expected_plain: set, expected_norm: set) -> None:
    """Volatility, momentum, and multi-output features declare exact column names."""
    plain = registry.create(ComponentType.TRANSFORM, name, normalized=False)
    plain_cols = set(plain.compute(FRAME.clone()).columns) - BASE_COLS
    assert set(plain.outputs) == expected_plain == plain_cols

    norm = registry.create(ComponentType.TRANSFORM, name, normalized=True)
    norm_cols = set(norm.compute(FRAME.clone()).columns) - BASE_COLS
    assert set(norm.outputs) == expected_norm == norm_cols

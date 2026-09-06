"""Compat-shim plugin tests.

Verifies that the ta plugin imports cleanly on the core, registers its
transforms/detectors into the registry, and that representative features and
detectors compute correctly on a real Dataset frame. The namespace bridge
that makes ``import signalflow.ta`` resolve lives in ``tests/conftest.py``.
"""


import warnings

import polars as pl
import pytest

warnings.filterwarnings("ignore", message="X does not have valid feature names")

import signalflow as sf
import signalflow.ta as ta
from signalflow import ComponentType, registry


@pytest.fixture(scope="module")
def frame() -> pl.DataFrame:
    ds = sf.data(
        "synthetic",
        pairs=["BTCUSDT", "ETHUSDT"],
        start="2023-01-01",
        end="2023-02-01",
        interval="1h",
    )
    return ds.frame


@pytest.fixture(scope="module")
def dataset():
    return sf.data(
        "synthetic",
        pairs=["BTCUSDT", "ETHUSDT"],
        start="2023-01-01",
        end="2023-03-01",
        interval="1h",
    )


def test_import_succeeds():
    assert ta is not None


def test_registry_has_many_transforms():
    transforms = registry.list(ComponentType.TRANSFORM)
    assert len(transforms) > 50, f"only {len(transforms)} transforms registered"


FEATURES = [
    ("momentum", lambda: ta.RsiMom(period=14)),
    ("overlap", lambda: ta.EmaSmooth(period=20)),
    ("volatility", lambda: ta.AtrVol(period=14)),
    ("volume", lambda: ta.ObvVolume()),
    ("microstructure", lambda: ta.WickToBodyRatio()),
    ("path_shape", lambda: ta.PathEfficiency()),
    ("stat", lambda: ta.StdStat(period=20)),
    ("trend", lambda: ta.AdxTrend(period=14)),
]


@pytest.mark.parametrize("family,factory", FEATURES, ids=[f[0] for f in FEATURES])
def test_feature_compute(family, factory, frame):
    feat = factory()
    out = feat.compute(frame)
    cols = feat.output_cols()
    assert cols, f"{family}: no declared output_cols()"
    missing = [c for c in cols if c not in out.columns]
    assert not missing, f"{family}: missing {missing}, have {out.columns}"
    assert out.height == frame.height, f"{family}: row count changed {out.height} != {frame.height}"


DETECTORS = [
    "AdxDiCrossDetector",
    "BollingerBreakoutDetector",
    "CciAnomalyDetector",
]


@pytest.mark.parametrize("name", DETECTORS)
def test_detector_compute(name, frame):
    cls = getattr(ta.signals, name)
    det = cls()
    out = det.compute(frame)
    assert "signal" in out.columns
    assert out.height == frame.height


def test_detector_signal_values(frame):
    det = ta.signals.AdxDiCrossDetector()
    out = det.compute(frame)
    vals = set(out["signal"].unique().to_list())
    assert vals <= {"rise", "fall", "none"}, f"unexpected signal values {vals}"


def test_feature_pipe_and_forecast_model(dataset):
    pipe = sf.FeaturePipeline(
        ta.RsiMom(period=14),
        ta.EmaSmooth(period=20),
        ta.AtrVol(period=14),
    )
    assert pipe.outputs
    model = sf.ForecastModel(target=sf.FixedHorizon(12), features=pipe, cv=sf.KFold(3))
    model.fit(dataset)
    pred = model.predict(dataset)
    assert "p_rise" in pred.columns
    assert pred.height > 0

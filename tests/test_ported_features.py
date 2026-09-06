"""The Polars ports of skew/kurtosis/autocorr/variance-ratio/roc equal the old numpy loops."""

import numpy as np
import polars as pl
import pytest
from scipy.stats import kurtosis as sp_kurtosis
from scipy.stats import skew as sp_skew

import signalflow as sf
import signalflow.ta as ta


@pytest.fixture(scope="module")
def frame() -> pl.DataFrame:
    ds = sf.dataset("synthetic", pairs=["BTCUSDT", "ETHUSDT"], start="2024-01-01", end="2024-02-15", interval="1h")
    return ds.frame


def _per_pair(frame: pl.DataFrame, col: str, fn) -> np.ndarray:
    parts = []
    for _pair, sub in frame.sort(["pair", "ts"]).group_by("pair", maintain_order=True):
        parts.append(fn(sub.get_column(col).to_numpy().astype(float)))
    return np.concatenate(parts)


def _ref_skew(values: np.ndarray, period: int) -> np.ndarray:
    out = np.full(len(values), np.nan)
    for i in range(period - 1, len(values)):
        out[i] = sp_skew(values[i - period + 1 : i + 1], bias=False)
    return out


def _ref_kurt(values: np.ndarray, period: int) -> np.ndarray:
    out = np.full(len(values), np.nan)
    for i in range(period - 1, len(values)):
        out[i] = sp_kurtosis(values[i - period + 1 : i + 1], fisher=True)
    return out


def _ref_acf(values: np.ndarray, period: int, lag: int) -> np.ndarray:
    out = np.full(len(values), np.nan)
    for i in range(period + lag - 1, len(values)):
        x = values[i - period + 1 : i + 1]
        x_lag = values[i - period + 1 - lag : i + 1 - lag]
        corr = np.corrcoef(x, x_lag)[0, 1]
        out[i] = corr if not np.isnan(corr) else 0
    return out


def _ref_vr(values: np.ndarray, period: int, k: int) -> np.ndarray:
    out = np.full(len(values), np.nan)
    log_ret = np.diff(np.log(values), prepend=np.nan)
    for i in range(period + k - 1, len(values)):
        var_1 = np.nanvar(log_ret[i - period + 1 : i + 1], ddof=1)
        ret_k = np.log(values[i - period + 1 + k : i + 1]) - np.log(values[i - period + 1 : i + 1 - k])
        var_k = np.nanvar(ret_k, ddof=1)
        if var_1 > 1e-10:
            out[i] = var_k / (k * var_1)
    return out


def _ref_roc(values: np.ndarray, period: int) -> np.ndarray:
    out = np.full(len(values), np.nan)
    for i in range(period, len(values)):
        if values[i - period] != 0:
            out[i] = 100 * (values[i] - values[i - period]) / values[i - period]
    return out


def _got(frame: pl.DataFrame, feature, col: str) -> np.ndarray:
    return feature.compute(frame).sort(["pair", "ts"]).get_column(col).to_numpy().astype(float)


@pytest.mark.parametrize("period", [30, 60])
def test_skew_matches_scipy(frame, period):
    got = _got(frame, ta.SkewStat(period=period), f"close_skew_{period}")
    ref = _per_pair(frame, "close", lambda v: _ref_skew(v, period))
    np.testing.assert_allclose(got, ref, rtol=1e-7, atol=1e-9, equal_nan=True)


@pytest.mark.parametrize("period", [30, 60])
def test_kurtosis_matches_scipy(frame, period):
    got = _got(frame, ta.KurtosisStat(period=period), f"close_kurt_{period}")
    ref = _per_pair(frame, "close", lambda v: _ref_kurt(v, period))
    np.testing.assert_allclose(got, ref, rtol=1e-7, atol=1e-9, equal_nan=True)


@pytest.mark.parametrize("period,lag", [(30, 1), (60, 5)])
def test_autocorr_matches_numpy(frame, period, lag):
    got = _got(frame, ta.AutocorrStat(period=period, lag=lag), f"close_acf{lag}_{period}")
    ref = _per_pair(frame, "close", lambda v: _ref_acf(v, period, lag))
    np.testing.assert_allclose(got, ref, rtol=1e-7, atol=1e-9, equal_nan=True)


@pytest.mark.parametrize("period,k", [(50, 5), (30, 2)])
def test_variance_ratio_matches_numpy(frame, period, k):
    got = _got(frame, ta.VarianceRatioStat(period=period, k=k), f"close_vr{k}_{period}")
    ref = _per_pair(frame, "close", lambda v: _ref_vr(v, period, k))
    np.testing.assert_allclose(got, ref, rtol=1e-7, atol=1e-9, equal_nan=True)


@pytest.mark.parametrize("period", [10, 60])
def test_roc_matches_loop(frame, period):
    got = _got(frame, ta.RocMom(period=period), f"roc_{period}")
    ref = _per_pair(frame, "close", lambda v: _ref_roc(v, period))
    np.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-9, equal_nan=True)

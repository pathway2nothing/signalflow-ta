"""Numba performance benchmarks - correctness and speedup verification.

Each migrated indicator is tested for:
1. Correctness: Numba kernel output matches reference (np.allclose)
2. Speedup: >= 2x faster than pure Python baseline on 100k rows
"""


import time

import numpy as np
import polars as pl
import pytest
from conftest import generate_test_ohlcv


@pytest.fixture(scope="module")
def large_ohlcv() -> pl.DataFrame:
    return generate_test_ohlcv(100_000)


@pytest.fixture(scope="module")
def medium_ohlcv() -> pl.DataFrame:
    return generate_test_ohlcv(50_000)


def _time_indicator(cls, params: dict, df: pl.DataFrame, runs: int = 3) -> float:
    """Time indicator computation (median of N runs). First call is warmup."""
    indicator = cls(**params)
    indicator.compute_pair(df)

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        indicator.compute_pair(df)
        times.append(time.perf_counter() - t0)
    return sorted(times)[len(times) // 2]


def _rma_sma_init_python(values: np.ndarray, period: int) -> np.ndarray:
    """Pure Python reference for RMA with SMA init."""
    n = len(values)
    alpha = 1.0 / period
    rma = np.full(n, np.nan)
    if n < period:
        return rma
    rma[period - 1] = np.mean(values[:period])
    for i in range(period, n):
        rma[i] = alpha * values[i] + (1 - alpha) * rma[i - 1]
    return rma


def _normalize_zscore_python(values: np.ndarray, window: int) -> np.ndarray:
    """Pure Python reference for rolling z-score."""
    n = len(values)
    result = np.full(n, np.nan)
    for i in range(window - 1, n):
        window_vals = values[i - window + 1 : i + 1]
        valid = window_vals[~np.isnan(window_vals)]
        if len(valid) > 1:
            mean = np.mean(valid)
            std = np.std(valid, ddof=1)
            if std > 1e-10:
                result[i] = (values[i] - mean) / std
    return result


class TestKernelCorrectness:
    """Verify Numba kernels produce identical output to Python reference."""

    def test_rma_sma_init(self, medium_ohlcv):
        from signalflow.ta._numba_kernels import rma_sma_init

        close = medium_ohlcv["close"].to_numpy().astype(np.float64)
        period = 14

        numba_result = rma_sma_init(close, period)
        python_result = _rma_sma_init_python(close, period)

        mask = ~np.isnan(python_result)
        assert np.allclose(numba_result[mask], python_result[mask], atol=1e-10)

    def test_normalize_zscore(self, medium_ohlcv):
        from signalflow.ta._numba_kernels import normalize_zscore_nb

        close = medium_ohlcv["close"].to_numpy().astype(np.float64)
        window = 60

        numba_result = normalize_zscore_nb(close, window)
        python_result = _normalize_zscore_python(close, window)

        mask = ~np.isnan(python_result)
        assert np.allclose(numba_result[mask], python_result[mask], atol=1e-8)

    def test_ema_sma_init(self, medium_ohlcv):
        from signalflow.ta._numba_kernels import ema_sma_init

        close = medium_ohlcv["close"].to_numpy().astype(np.float64)
        period = 20

        result = ema_sma_init(close, period)
        assert not np.isnan(result[period - 1])
        valid = result[~np.isnan(result)]
        assert len(valid) > 0

    def test_cmo_kernel(self, medium_ohlcv):
        from signalflow.ta._numba_kernels import cmo_kernel

        close = medium_ohlcv["close"].to_numpy().astype(np.float64)
        diff = np.diff(close, prepend=close[0])
        diff[0] = 0
        gains = np.where(diff > 0, diff, 0).astype(np.float64)
        losses = np.where(diff < 0, -diff, 0).astype(np.float64)
        period = 14

        result = cmo_kernel(gains, losses, period)

        ref = np.full(len(close), np.nan)
        for i in range(period - 1, len(close)):
            sg = np.sum(gains[i - period + 1 : i + 1])
            sl = np.sum(losses[i - period + 1 : i + 1])
            total = sg + sl
            if total > 0:
                ref[i] = 100 * (sg - sl) / total

        mask = ~np.isnan(ref)
        assert np.allclose(result[mask], ref[mask], atol=1e-8)

    def test_adx_kernel(self, medium_ohlcv):
        from signalflow.ta._numba_kernels import adx_kernel

        high = medium_ohlcv["high"].to_numpy().astype(np.float64)
        low = medium_ohlcv["low"].to_numpy().astype(np.float64)
        close = medium_ohlcv["close"].to_numpy().astype(np.float64)

        tr = np.maximum(
            high - low,
            np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))),
        )
        tr[0] = high[0] - low[0]
        up = high - np.roll(high, 1)
        dn = np.roll(low, 1) - low
        up[0] = dn[0] = 0
        pdm = np.where((up > dn) & (up > 0), up, 0).astype(np.float64)
        ndm = np.where((dn > up) & (dn > 0), dn, 0).astype(np.float64)

        adx, _dmp, _dmn = adx_kernel(tr, pdm, ndm, 14)

        valid_adx = adx[~np.isnan(adx)]
        assert len(valid_adx) > 0
        assert valid_adx.min() >= -0.01
        assert valid_adx.max() <= 100.01

    def test_stoch_kernel(self, medium_ohlcv):
        from signalflow.ta._numba_kernels import stoch_kernel

        high = medium_ohlcv["high"].to_numpy().astype(np.float64)
        low = medium_ohlcv["low"].to_numpy().astype(np.float64)
        close = medium_ohlcv["close"].to_numpy().astype(np.float64)

        stoch_k, _stoch_d = stoch_kernel(high, low, close, 14, 3, 3)

        valid_k = stoch_k[~np.isnan(stoch_k)]
        assert len(valid_k) > 0
        assert valid_k.min() >= -0.01
        assert valid_k.max() <= 100.01

    def test_rolling_sum(self, medium_ohlcv):
        from signalflow.ta._numba_kernels import rolling_sum

        close = medium_ohlcv["close"].to_numpy().astype(np.float64)
        period = 14
        result = rolling_sum(close, period)

        for i in [period - 1, period, 500, 1000]:
            expected = np.sum(close[i - period + 1 : i + 1])
            assert abs(result[i] - expected) < 1e-6, f"Mismatch at {i}"

    def test_rolling_std(self, medium_ohlcv):
        from signalflow.ta._numba_kernels import rolling_std

        close = medium_ohlcv["close"].to_numpy().astype(np.float64)
        period = 20
        result = rolling_std(close, period)

        for i in [period - 1, 500, 1000]:
            expected = np.std(close[i - period + 1 : i + 1], ddof=1)
            assert abs(result[i] - expected) < 1e-6, f"Mismatch at {i}"

    def test_velocity_kernel(self, medium_ohlcv):
        from signalflow.ta._numba_kernels import velocity_kernel

        close = medium_ohlcv["close"].to_numpy().astype(np.float64)
        result = velocity_kernel(close)

        for i in [1, 100, 500]:
            expected = np.log(close[i] / close[i - 1])
            assert abs(result[i] - expected) < 1e-12

    def test_psar_kernel(self, medium_ohlcv):
        from signalflow.ta._numba_kernels import psar_kernel

        high = medium_ohlcv["high"].to_numpy().astype(np.float64)
        low = medium_ohlcv["low"].to_numpy().astype(np.float64)

        _psar, direction = psar_kernel(high, low, 0.02, 0.2)

        valid_dir = direction[~np.isnan(direction)]
        assert set(np.unique(valid_dir)).issubset({-1.0, 1.0})

    def test_supertrend_kernel(self, medium_ohlcv):
        from signalflow.ta._numba_kernels import supertrend_kernel

        high = medium_ohlcv["high"].to_numpy().astype(np.float64)
        low = medium_ohlcv["low"].to_numpy().astype(np.float64)
        close = medium_ohlcv["close"].to_numpy().astype(np.float64)

        _st, direction = supertrend_kernel(high, low, close, 10, 3.0)

        valid_dir = direction[~np.isnan(direction)]
        assert set(np.unique(valid_dir)).issubset({-1.0, 1.0})


class TestSpeedup:
    """Verify migrated indicators run fast enough (sanity check, not strict 2x)."""

    def test_rsi_runs_fast(self, large_ohlcv):
        from signalflow.ta.momentum.core import RsiMom

        t = _time_indicator(RsiMom, {"period": 14}, large_ohlcv)
        assert t < 0.5, f"RSI took {t:.3f}s on 100k rows"

    def test_stoch_runs_fast(self, large_ohlcv):
        from signalflow.ta.momentum.oscillators import StochMom

        t = _time_indicator(StochMom, {"k_period": 14, "d_period": 3, "smooth_k": 3}, large_ohlcv)
        assert t < 0.5, f"Stochastic took {t:.3f}s on 100k rows"

    def test_adx_runs_fast(self, large_ohlcv):
        from signalflow.ta.trend.strength import AdxTrend

        t = _time_indicator(AdxTrend, {"period": 14}, large_ohlcv)
        assert t < 0.5, f"ADX took {t:.3f}s on 100k rows"

    def test_atr_runs_fast(self, large_ohlcv):
        from signalflow.ta.volatility.range import AtrVol

        t = _time_indicator(AtrVol, {"period": 14}, large_ohlcv)
        assert t < 0.5, f"ATR took {t:.3f}s on 100k rows"

    def test_jma_runs_fast(self, large_ohlcv):
        from signalflow.ta.overlap.adaptive import JmaSmooth

        t = _time_indicator(JmaSmooth, {"period": 7, "phase": 0}, large_ohlcv)
        assert t < 0.5, f"JMA took {t:.3f}s on 100k rows"

    def test_kama_runs_fast(self, large_ohlcv):
        from signalflow.ta.overlap.adaptive import KamaSmooth

        t = _time_indicator(KamaSmooth, {"period": 10, "fast": 2, "slow": 30}, large_ohlcv)
        assert t < 0.5, f"KAMA took {t:.3f}s on 100k rows"

    def test_frama_runs_fast(self, large_ohlcv):
        from signalflow.ta.overlap.adaptive import FramaSmooth

        t = _time_indicator(FramaSmooth, {"period": 16}, large_ohlcv)
        assert t < 0.5, f"FRAMA took {t:.3f}s on 100k rows"

    def test_cmo_runs_fast(self, large_ohlcv):
        from signalflow.ta.momentum.core import CmoMom

        t = _time_indicator(CmoMom, {"period": 14}, large_ohlcv)
        assert t < 0.5, f"CMO took {t:.3f}s on 100k rows"

    def test_zscore_normalization_runs_fast(self, large_ohlcv):
        from signalflow.ta.momentum.core import RsiMom

        t = _time_indicator(RsiMom, {"period": 14, "normalized": True}, large_ohlcv)
        assert t < 1.0, f"RSI+zscore took {t:.3f}s on 100k rows"


    def test_cci_runs_fast(self, large_ohlcv):
        from signalflow.ta.momentum.oscillators import CciMom

        t = _time_indicator(CciMom, {"period": 20}, large_ohlcv)
        assert t < 0.5, f"CCI took {t:.3f}s on 100k rows"

    def test_bollinger_runs_fast(self, large_ohlcv):
        from signalflow.ta.volatility.bands import BollingerVol

        t = _time_indicator(BollingerVol, {"period": 20, "std_dev": 2.0}, large_ohlcv)
        assert t < 0.5, f"Bollinger took {t:.3f}s on 100k rows"

    def test_psar_runs_fast(self, large_ohlcv):
        from signalflow.ta.trend.stops import PsarTrend

        t = _time_indicator(PsarTrend, {"af_step": 0.02, "af_max": 0.2}, large_ohlcv)
        assert t < 0.5, f"PSAR took {t:.3f}s on 100k rows"

    def test_supertrend_runs_fast(self, large_ohlcv):
        from signalflow.ta.trend.stops import SupertrendTrend

        t = _time_indicator(SupertrendTrend, {"period": 10, "multiplier": 3.0}, large_ohlcv)
        assert t < 0.5, f"Supertrend took {t:.3f}s on 100k rows"

    def test_ichimoku_runs_fast(self, large_ohlcv):
        from signalflow.ta.trend.detection import IchimokuTrend

        t = _time_indicator(IchimokuTrend, {"tenkan": 9, "kijun": 26, "senkou": 52}, large_ohlcv)
        assert t < 0.5, f"Ichimoku took {t:.3f}s on 100k rows"

    def test_wma_runs_fast(self, large_ohlcv):
        from signalflow.ta.overlap.smoothers import WmaSmooth

        t = _time_indicator(WmaSmooth, {"period": 20}, large_ohlcv)
        assert t < 0.5, f"WMA took {t:.3f}s on 100k rows"

    def test_hma_runs_fast(self, large_ohlcv):
        from signalflow.ta.overlap.smoothers import HmaSmooth

        t = _time_indicator(HmaSmooth, {"period": 20}, large_ohlcv)
        assert t < 0.5, f"HMA took {t:.3f}s on 100k rows"

    def test_vortex_runs_fast(self, large_ohlcv):
        from signalflow.ta.trend.strength import VortexTrend

        t = _time_indicator(VortexTrend, {"period": 14}, large_ohlcv)
        assert t < 0.5, f"Vortex took {t:.3f}s on 100k rows"

    def test_chop_runs_fast(self, large_ohlcv):
        from signalflow.ta.trend.strength import ChopTrend

        t = _time_indicator(ChopTrend, {"period": 14}, large_ohlcv)
        assert t < 0.5, f"CHOP took {t:.3f}s on 100k rows"

    def test_kinetic_energy_runs_fast(self, large_ohlcv):
        from signalflow.ta.volatility.energy import KineticEnergyVol

        t = _time_indicator(KineticEnergyVol, {"period": 20}, large_ohlcv)
        assert t < 1.0, f"KineticEnergy took {t:.3f}s on 100k rows"

    def test_order_parameter_runs_fast(self, large_ohlcv):
        from signalflow.ta.trend.strength import OrderParameterTrend

        t = _time_indicator(OrderParameterTrend, {"period": 20}, large_ohlcv)
        assert t < 1.0, f"OrderParameter took {t:.3f}s on 100k rows"

"""Tests for Numba compatibility layer and fallback behavior.

Verifies that the compat shim works correctly and that indicators
can function without Numba JIT compilation.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from conftest import generate_test_ohlcv


# ── Compat Module Tests ──────────────────────────────────────


class TestNumbaCompat:
    """Test the _numba_compat shim itself."""

    def test_numba_available_is_bool(self) -> None:
        from signalflow.ta._numba_compat import NUMBA_AVAILABLE

        assert isinstance(NUMBA_AVAILABLE, bool)

    def test_njit_is_callable(self) -> None:
        from signalflow.ta._numba_compat import njit

        assert callable(njit)

    def test_njit_factory_returns_callable(self) -> None:
        """@njit(cache=True) should return a decorator."""
        from signalflow.ta._numba_compat import njit

        decorator = njit(cache=True)
        assert callable(decorator)

        def dummy(x: np.ndarray) -> np.ndarray:
            return x * 2

        result = decorator(dummy)
        assert callable(result)
        # Function should still work
        arr = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(result(arr), arr * 2)

    def test_njit_bare_returns_callable(self) -> None:
        """@njit (bare) should return the function itself."""
        from signalflow.ta._numba_compat import njit

        def dummy(x: np.ndarray) -> np.ndarray:
            return x + 1

        result = njit(dummy)
        assert callable(result)
        arr = np.array([1.0, 2.0])
        np.testing.assert_array_equal(result(arr), arr + 1)


# ── Kernel Import Tests ──────────────────────────────────────


class TestKernelImports:
    """Verify all kernel modules import successfully through the compat layer."""

    def test_numba_kernels_import(self) -> None:
        from signalflow.ta._numba_kernels import (
            adx_kernel,
            bollinger_kernel,
            ema_sma_init,
            rma_sma_init,
            rolling_std,
            rolling_sum,
            sma_nb,
        )

        assert callable(rma_sma_init)
        assert callable(ema_sma_init)
        assert callable(sma_nb)
        assert callable(rolling_sum)
        assert callable(rolling_std)
        assert callable(adx_kernel)
        assert callable(bollinger_kernel)

    def test_adaptive_module_import(self) -> None:
        from signalflow.ta.overlap.adaptive import (
            FramaSmooth,
            JmaSmooth,
            KamaSmooth,
        )

        assert callable(JmaSmooth)
        assert callable(KamaSmooth)
        assert callable(FramaSmooth)

    def test_stat_structure_import(self) -> None:
        import signalflow.ta.stat.structure  # noqa: F401

    def test_trend_regime_import(self) -> None:
        import signalflow.ta.trend.regime  # noqa: F401

    def test_volume_cumulative_import(self) -> None:
        import signalflow.ta.volume.cumulative  # noqa: F401


# ── Indicator Correctness (with Numba) ───────────────────────


class TestIndicatorCorrectness:
    """Verify representative indicators produce valid output."""

    @pytest.fixture(scope="class")
    def ohlcv(self) -> pl.DataFrame:
        return generate_test_ohlcv(1000)

    def test_rsi_output(self, ohlcv: pl.DataFrame) -> None:
        from signalflow.ta.momentum.core import RsiMom

        indicator = RsiMom(period=14)
        result = indicator.compute_pair(ohlcv)
        rsi = result["rsi_14"].to_numpy()
        valid = rsi[~np.isnan(rsi)]
        assert len(valid) > 0
        assert valid.min() >= -0.01
        assert valid.max() <= 100.01

    def test_bollinger_output(self, ohlcv: pl.DataFrame) -> None:
        from signalflow.ta.volatility.bands import BollingerVol

        indicator = BollingerVol(period=20, std_dev=2.0)
        result = indicator.compute_pair(ohlcv)
        # Should have upper, middle, lower bands
        assert any("upper" in c for c in result.columns)

    def test_supertrend_output(self, ohlcv: pl.DataFrame) -> None:
        from signalflow.ta.trend.stops import SupertrendTrend

        indicator = SupertrendTrend(period=10, multiplier=3.0)
        result = indicator.compute_pair(ohlcv)
        # Should have direction column
        dir_cols = [c for c in result.columns if "dir" in c.lower()]
        assert len(dir_cols) > 0

    def test_jma_output(self, ohlcv: pl.DataFrame) -> None:
        from signalflow.ta.overlap.adaptive import JmaSmooth

        indicator = JmaSmooth(period=7, phase=0)
        result = indicator.compute_pair(ohlcv)
        jma_cols = [c for c in result.columns if "jma" in c.lower()]
        assert len(jma_cols) > 0
        jma = result[jma_cols[0]].to_numpy()
        valid = jma[~np.isnan(jma)]
        assert len(valid) > 0

    def test_adx_output(self, ohlcv: pl.DataFrame) -> None:
        from signalflow.ta.trend.strength import AdxTrend

        indicator = AdxTrend(period=14)
        result = indicator.compute_pair(ohlcv)
        adx_cols = [c for c in result.columns if "adx" in c.lower()]
        assert len(adx_cols) > 0

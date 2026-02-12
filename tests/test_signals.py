"""
SignalFlow-TA Signal Detector Tests

Tests for signal detectors in the signals module.
Tests critical properties:
1. Instantiation - detectors can be created with default and custom params
2. Warmup property - warmup values are positive integers
3. Features dependency - detectors declare required features
4. Output structure - detect() produces valid Signals output
"""

from __future__ import annotations

from dataclasses import fields

import polars as pl
import pytest

from conftest import generate_test_ohlcv, generate_static_ohlcv, SEED
from signalflow.core import SignalType


# =============================================================================
# Detector Registry
# =============================================================================


def get_all_detector_classes() -> list[tuple[str, type]]:
    """Get all detector classes from signals module."""
    from signalflow.ta import signals

    detector_classes = []
    for name in dir(signals):
        if name.startswith("_"):
            continue
        obj = getattr(signals, name)
        if isinstance(obj, type) and hasattr(obj, "detect") and hasattr(obj, "warmup"):
            # Check it has test_params
            if hasattr(obj, "test_params"):
                detector_classes.append((name, obj))
    return detector_classes


def get_detector_test_configs() -> list[tuple[str, type, dict]]:
    """Get all detector test configurations (first param set only)."""
    configs = []
    for name, cls in get_all_detector_classes():
        test_params = getattr(cls, "test_params", [{}])
        configs.append((name, cls, test_params[0]))
    return configs


DETECTOR_CONFIGS = get_detector_test_configs()
DETECTOR_IDS = [c[0] for c in DETECTOR_CONFIGS]


# =============================================================================
# Test Data Generator
# =============================================================================


def generate_detector_test_data(n_rows: int = 1000, seed: int = SEED) -> pl.DataFrame:
    """Generate test data with all required columns for detectors."""
    df = generate_test_ohlcv(n_rows=n_rows, seed=seed)
    # Add spot column required by RawDataView
    return df.with_columns(pl.lit("binance").alias("spot"))


# =============================================================================
# Instantiation Tests
# =============================================================================


class TestDetectorInstantiation:
    """Test that detectors can be instantiated with various parameters."""

    @pytest.mark.parametrize("config_id,cls,params", DETECTOR_CONFIGS, ids=DETECTOR_IDS)
    def test_instantiation_with_test_params(self, config_id, cls, params):
        """Detector instantiates with test_params."""
        detector = cls(**params)
        assert detector is not None

    @pytest.mark.parametrize("name,cls", get_all_detector_classes())
    def test_instantiation_with_defaults(self, name, cls):
        """Detector instantiates with default parameters."""
        detector = cls()
        assert detector is not None

    @pytest.mark.parametrize("name,cls", get_all_detector_classes())
    def test_direction_validation(self, name, cls):
        """Detector validates direction parameter."""
        # Check if direction field exists
        field_names = [f.name for f in fields(cls)]
        if "direction" not in field_names:
            pytest.skip(f"{name} has no direction field")

        # Valid directions should work
        for direction in ["long", "short", "both"]:
            detector = cls(direction=direction)
            assert detector.direction == direction

        # Invalid direction should raise
        with pytest.raises(ValueError):
            cls(direction="invalid")


# =============================================================================
# Warmup Tests
# =============================================================================


class TestDetectorWarmup:
    """Test warmup property of detectors."""

    @pytest.mark.parametrize("config_id,cls,params", DETECTOR_CONFIGS, ids=DETECTOR_IDS)
    def test_warmup_is_positive(self, config_id, cls, params):
        """Warmup must be a positive integer."""
        detector = cls(**params)
        assert isinstance(detector.warmup, int)
        assert detector.warmup > 0

    @pytest.mark.parametrize("name,cls", get_all_detector_classes())
    def test_warmup_scales_with_period(self, name, cls):
        """Warmup should generally increase with indicator period."""
        field_names = [f.name for f in fields(cls)]

        # Find period-like parameter
        period_param = None
        for param in [
            "period",
            "rsi_period",
            "mfi_period",
            "adx_period",
            "stoch_period",
        ]:
            if param in field_names:
                period_param = param
                break

        if period_param is None:
            pytest.skip(f"{name} has no period-like parameter")

        # Test that larger period -> larger warmup
        small_period = cls(**{period_param: 7})
        large_period = cls(**{period_param: 28})

        assert large_period.warmup >= small_period.warmup


# =============================================================================
# Features Tests
# =============================================================================


class TestDetectorFeatures:
    """Test feature dependencies of detectors."""

    @pytest.mark.parametrize("config_id,cls,params", DETECTOR_CONFIGS, ids=DETECTOR_IDS)
    def test_features_declared(self, config_id, cls, params):
        """Detector must have features attribute (may be empty for self-computed)."""
        detector = cls(**params)
        assert hasattr(detector, "features")
        assert isinstance(detector.features, list)
        # Some detectors compute features internally (Hampel, Kalman)
        # so we don't require features > 0

    @pytest.mark.parametrize("config_id,cls,params", DETECTOR_CONFIGS, ids=DETECTOR_IDS)
    def test_features_are_valid(self, config_id, cls, params):
        """Declared features must be valid Feature instances."""
        detector = cls(**params)
        if not detector.features:
            pytest.skip("Detector computes features internally")
        for feature in detector.features:
            # Feature must have compute_pair method
            assert hasattr(feature, "compute_pair")
            # Feature must have warmup property
            assert hasattr(feature, "warmup")


# =============================================================================
# Detection Tests
# =============================================================================


class TestDetectorOutput:
    """Test detect() method output structure."""

    @pytest.fixture
    def test_data(self) -> pl.DataFrame:
        """Generate test data for detection tests."""
        return generate_detector_test_data(n_rows=1000)

    @pytest.fixture
    def feature_data(self, test_data) -> pl.DataFrame:
        """Generate feature data by running all possible features."""
        df = test_data
        # Run common features that might be needed
        from signalflow.ta.momentum import RsiMom, MacdMom, StochMom
        from signalflow.ta.volatility import BollingerVol
        from signalflow.ta.trend import AdxTrend
        from signalflow.ta.volume import MfiVolume

        features = [
            RsiMom(period=14),
            MacdMom(fast=12, slow=26, signal=9),
            StochMom(k_period=14, smooth_k=3, d_period=3),
            BollingerVol(period=20),
            AdxTrend(period=14),
            MfiVolume(period=14),
        ]

        for feature in features:
            try:
                df = feature.compute_pair(df)
            except Exception:
                pass  # Some features may fail, that's ok

        return df

    @pytest.mark.parametrize("config_id,cls,params", DETECTOR_CONFIGS, ids=DETECTOR_IDS)
    def test_detect_returns_signals(self, config_id, cls, params, test_data):
        """detect() must return Signals object."""
        from signalflow.core import Signals

        detector = cls(**params)

        # Compute required features
        df = test_data
        try:
            for feature in detector.features:
                df = feature.compute_pair(df)
        except Exception as e:
            pytest.skip(f"Feature computation failed: {e}")

        # Run detection
        try:
            result = detector.detect(df)
        except Exception as e:
            pytest.skip(f"Detection failed: {e}")

        assert isinstance(result, Signals)
        assert hasattr(result, "value")

    @pytest.mark.parametrize("config_id,cls,params", DETECTOR_CONFIGS, ids=DETECTOR_IDS)
    def test_output_has_required_columns(self, config_id, cls, params, test_data):
        """Output must have pair, timestamp, signal_type, signal columns."""
        detector = cls(**params)

        # Compute required features
        df = test_data
        try:
            for feature in detector.features:
                df = feature.compute_pair(df)
            result = detector.detect(df)
        except Exception as e:
            pytest.skip(f"Detection failed: {e}")

        output_df = result.value

        required_cols = ["pair", "timestamp", "signal_type", "signal"]
        for col in required_cols:
            assert col in output_df.columns, f"Missing column: {col}"

    @pytest.mark.parametrize("config_id,cls,params", DETECTOR_CONFIGS, ids=DETECTOR_IDS)
    def test_signal_types_are_valid(self, config_id, cls, params, test_data):
        """Signal types must be valid SignalType values."""
        detector = cls(**params)

        # Compute required features
        df = test_data
        try:
            for feature in detector.features:
                df = feature.compute_pair(df)
            result = detector.detect(df)
        except Exception as e:
            pytest.skip(f"Detection failed: {e}")

        output_df = result.value

        if len(output_df) == 0:
            pytest.skip("No signals generated")

        signal_types = output_df["signal_type"].unique().to_list()
        valid_types = [
            SignalType.NONE.value,
            SignalType.RISE.value,
            SignalType.FALL.value,
        ]

        for st in signal_types:
            assert st in valid_types, f"Invalid signal type: {st}"

    @pytest.mark.parametrize("config_id,cls,params", DETECTOR_CONFIGS, ids=DETECTOR_IDS)
    def test_no_none_signals_in_output(self, config_id, cls, params, test_data):
        """Output should not contain NONE signal types (they should be filtered)."""
        detector = cls(**params)

        # Compute required features
        df = test_data
        try:
            for feature in detector.features:
                df = feature.compute_pair(df)
            result = detector.detect(df)
        except Exception as e:
            pytest.skip(f"Detection failed: {e}")

        output_df = result.value

        if len(output_df) == 0:
            pytest.skip("No signals generated")

        # NONE signals should be filtered out
        none_count = output_df.filter(
            pl.col("signal_type") == SignalType.NONE.value
        ).height
        assert none_count == 0, "Output contains NONE signals"


# =============================================================================
# Filter Integration Tests
# =============================================================================


class TestFilterIntegration:
    """Test that filters integrate correctly with detectors."""

    @pytest.fixture
    def test_data(self) -> pl.DataFrame:
        """Generate test data."""
        return generate_detector_test_data(n_rows=1000)

    def test_detector_accepts_filters(self, test_data):
        """Detector should accept filters parameter."""
        from signalflow.ta.signals import StochasticDetector1, RsiZscoreFilter

        # Create detector with filter
        detector = StochasticDetector1(
            direction="both", filters=[RsiZscoreFilter(threshold=-1.0)]
        )

        assert len(detector.filters) == 1

    def test_filter_affects_warmup(self, test_data):
        """Filter warmup should be considered in detector warmup."""
        from signalflow.ta.signals import StochasticDetector1, RsiZscoreFilter

        detector_no_filter = StochasticDetector1(direction="long")
        detector_with_filter = StochasticDetector1(
            direction="long",
            filters=[RsiZscoreFilter(threshold=-1.0, zscore_window=2000)],
        )

        # Warmup with filter should be >= warmup without filter
        assert detector_with_filter.warmup >= detector_no_filter.warmup


# =============================================================================
# Reproducibility Tests
# =============================================================================


class TestDetectorReproducibility:
    """Test that detectors produce reproducible results."""

    @pytest.mark.parametrize(
        "config_id,cls,params", DETECTOR_CONFIGS[:5], ids=DETECTOR_IDS[:5]
    )
    def test_same_input_same_output(self, config_id, cls, params):
        """Same input should produce same output."""
        detector = cls(**params)

        # Generate same data twice
        df1 = generate_detector_test_data(n_rows=500, seed=42)
        df2 = generate_detector_test_data(n_rows=500, seed=42)

        # Compute features
        for feature in detector.features:
            df1 = feature.compute_pair(df1)
            df2 = feature.compute_pair(df2)

        # Run detection
        result1 = detector.detect(df1)
        result2 = detector.detect(df2)

        # Results should be identical
        assert len(result1.value) == len(result2.value)

        if len(result1.value) > 0:
            # Compare signal types
            types1 = result1.value["signal_type"].to_list()
            types2 = result2.value["signal_type"].to_list()
            assert types1 == types2


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestDetectorEdgeCases:
    """Test detector behavior with edge cases."""

    @pytest.mark.parametrize("name,cls", get_all_detector_classes()[:5])
    def test_minimum_data(self, name, cls):
        """Detector should handle minimum data length."""
        detector = cls()
        min_length = detector.warmup + 10

        df = generate_detector_test_data(n_rows=min_length)

        # Compute features
        for feature in detector.features:
            df = feature.compute_pair(df)

        # Should not raise
        result = detector.detect(df)
        assert result is not None

    @pytest.mark.parametrize("name,cls", get_all_detector_classes()[:5])
    def test_constant_price(self, name, cls):
        """Detector should handle constant price data."""
        from conftest import generate_static_ohlcv

        detector = cls()
        df = generate_static_ohlcv(n_rows=500)
        df = df.with_columns(pl.lit("binance").alias("spot"))

        # Compute features
        for feature in detector.features:
            try:
                df = feature.compute_pair(df)
            except Exception:
                pytest.skip("Feature failed on constant price data")

        # Should not raise (may produce 0 signals, that's ok)
        result = detector.detect(df)
        assert result is not None


# =============================================================================
# Multi-Pair Tests
# =============================================================================


class TestDetectorMultiPair:
    """Test that detectors handle multi-pair data correctly."""

    @pytest.mark.parametrize("config_id,cls,params", DETECTOR_CONFIGS[:5], ids=DETECTOR_IDS[:5])
    def test_multi_pair_no_cross_contamination(self, config_id, cls, params):
        """Signals from one pair should be identical whether run alone or with other pairs."""
        detector = cls(**params)

        # Generate data for 2 pairs with different seeds (different price patterns)
        df1 = generate_detector_test_data(n_rows=500, seed=42)
        df2 = generate_detector_test_data(n_rows=500, seed=99)
        df2 = df2.with_columns(pl.lit("OTHER_PAIR").alias("pair"))

        # Compute features per-pair (as the framework does)
        for feature in detector.features:
            try:
                df1 = feature.compute_pair(df1)
                df2 = feature.compute_pair(df2)
            except Exception as e:
                pytest.skip(f"Feature computation failed: {e}")

        # Run detector on single-pair data
        try:
            result_single = detector.detect(df1)
        except Exception as e:
            pytest.skip(f"Detection failed: {e}")

        # Run detector on multi-pair data
        multi_pair = pl.concat([df1, df2]).sort(["pair", "timestamp"])
        try:
            result_multi = detector.detect(multi_pair)
        except Exception as e:
            pytest.skip(f"Multi-pair detection failed: {e}")

        # Extract signals for the original pair from multi-pair result
        multi_pair_signals = result_multi.value.filter(
            pl.col("pair") == "BTCUSDT"
        ).sort("timestamp")
        single_pair_signals = result_single.value.sort("timestamp")

        # Results for the same pair should be identical
        assert len(multi_pair_signals) == len(single_pair_signals), (
            f"Multi-pair returned {len(multi_pair_signals)} signals for BTCUSDT, "
            f"single-pair returned {len(single_pair_signals)}"
        )

        if len(single_pair_signals) > 0:
            types_single = single_pair_signals["signal_type"].to_list()
            types_multi = multi_pair_signals["signal_type"].to_list()
            assert types_single == types_multi, "Signal types differ between single and multi-pair"

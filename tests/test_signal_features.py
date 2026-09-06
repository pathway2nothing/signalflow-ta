"""Tests for sf-ta signal feature implementations."""

from __future__ import annotations

import pytest

pytest.importorskip("signalflow.core", reason="pre-V5 test module: written against the old signalflow.core API")

from datetime import datetime, timedelta
from typing import Any

import polars as pl
import pytest

from signalflow.core.enums import SfComponentType
from signalflow.signal_feature.base import SignalFeature

# =============================================================================
# Fixtures
# =============================================================================


def _make_signals(n: int = 30, pairs: int = 1) -> pl.DataFrame:
    """Build synthetic signals with alternating rise/fall pattern."""
    base = datetime(2024, 1, 1)
    rows: list[dict[str, Any]] = []
    for p in range(pairs):
        pair = f"PAIR{p}"
        for i in range(n):
            rows.append(
                {
                    "pair": pair,
                    "timestamp": base + timedelta(hours=i),
                    "signal_type": "rise" if i % 3 != 0 else "fall",
                    "signal": 1 if i % 3 != 0 else -1,
                    "probability": 0.5 + (i % 5) * 0.1,
                }
            )
    return pl.DataFrame(rows)


def _make_labels(
    signals: pl.DataFrame,
    *,
    accuracy: float = 0.7,
    with_t_hit: bool = True,
    hit_delay_hours: int = 0,
    with_ret: bool = False,
) -> pl.DataFrame:
    """Build labels aligned to signals."""
    rows: list[dict[str, Any]] = []
    for i, row in enumerate(signals.to_dicts()):
        correct = (i % 10) < int(accuracy * 10)
        label = (
            row["signal_type"]
            if correct
            else ("fall" if row["signal_type"] == "rise" else "rise")
        )
        entry: dict[str, Any] = {
            "pair": row["pair"],
            "timestamp": row["timestamp"],
            "label": label,
        }
        if with_t_hit:
            entry["t_hit"] = row["timestamp"] + timedelta(hours=hit_delay_hours)
        if with_ret:
            # Raw market return: positive when market rose, negative when fell.
            # Correct rise → +0.02, correct fall → -0.02
            # Wrong rise (actual fall) → -0.015, wrong fall (actual rise) → +0.015
            if label == "rise":
                entry["ret"] = 0.02 if correct else 0.015
            else:
                entry["ret"] = -0.02 if correct else -0.015
        rows.append(entry)
    return pl.DataFrame(rows)


# =============================================================================
# Frequency features
# =============================================================================


class TestSignalFrequency:
    def test_basic(self) -> None:
        from signalflow.ta.signal_features.frequency import SignalFrequency

        signals = _make_signals(20)
        feat = SignalFrequency(window=5)
        result = feat(signals)

        assert result.height == signals.height
        assert "signal_freq_5" in result.columns
        assert result["signal_freq_5"].null_count() == 0

    def test_multi_pair(self) -> None:
        from signalflow.ta.signal_features.frequency import SignalFrequency

        signals = _make_signals(20, pairs=3)
        feat = SignalFrequency(window=10)
        result = feat(signals)

        assert result.height == signals.height
        for pair in ["PAIR0", "PAIR1", "PAIR2"]:
            subset = result.filter(pl.col("pair") == pair)
            assert subset.height == 20

    def test_component_type(self) -> None:
        from signalflow.ta.signal_features.frequency import SignalFrequency

        feat = SignalFrequency()
        assert feat.component_type == SfComponentType.SIGNAL_FEATURE
        assert not feat.requires_labels

    def test_warmup(self) -> None:
        from signalflow.ta.signal_features.frequency import SignalFrequency

        feat = SignalFrequency(window=25)
        assert feat.warmup == 25


class TestInterSignalDistance:
    def test_basic(self) -> None:
        from signalflow.ta.signal_features.frequency import InterSignalDistance

        signals = _make_signals(30)
        feat = InterSignalDistance(zscore_window=10)
        result = feat(signals)

        assert result.height == signals.height
        assert "isd_bars" in result.columns
        assert "isd_zscore_10" in result.columns

    def test_first_row_null(self) -> None:
        from signalflow.ta.signal_features.frequency import InterSignalDistance

        signals = _make_signals(10)
        feat = InterSignalDistance()
        result = feat(signals)

        # First row has no previous signal → isd_bars should be null
        first = result.sort("timestamp").row(0, named=True)
        assert first["isd_bars"] is None

    def test_all_isd_positive(self) -> None:
        from signalflow.ta.signal_features.frequency import InterSignalDistance

        signals = _make_signals(30)
        feat = InterSignalDistance()
        result = feat(signals)

        non_null = result["isd_bars"].drop_nulls()
        assert (non_null > 0).all()


# =============================================================================
# Stability features
# =============================================================================


class TestSignalFlipRate:
    def test_basic(self) -> None:
        from signalflow.ta.signal_features.stability import SignalFlipRate

        signals = _make_signals(20)
        feat = SignalFlipRate(window=10)
        result = feat(signals)

        assert result.height == signals.height
        assert "flip_rate_10" in result.columns

    def test_range_zero_one(self) -> None:
        from signalflow.ta.signal_features.stability import SignalFlipRate

        signals = _make_signals(50)
        feat = SignalFlipRate(window=20)
        result = feat(signals)

        non_null = result["flip_rate_20"].drop_nulls()
        assert (non_null >= 0).all()
        assert (non_null <= 1).all()


class TestSignalStreak:
    def test_basic(self) -> None:
        from signalflow.ta.signal_features.stability import SignalStreak

        signals = _make_signals(20)
        feat = SignalStreak()
        result = feat(signals)

        assert result.height == signals.height
        assert "streak_len" in result.columns
        assert "streak_dir" in result.columns

    def test_streak_positive(self) -> None:
        from signalflow.ta.signal_features.stability import SignalStreak

        signals = _make_signals(20)
        feat = SignalStreak()
        result = feat(signals)

        assert (result["streak_len"] > 0).all()

    def test_direction_encoding(self) -> None:
        from signalflow.ta.signal_features.stability import SignalStreak

        signals = _make_signals(20)
        feat = SignalStreak()
        result = feat(signals)

        dirs = result["streak_dir"].unique().sort().to_list()
        assert all(d in (-1, 0, 1) for d in dirs)


class TestSignalEntropy:
    def test_basic(self) -> None:
        from signalflow.ta.signal_features.stability import SignalEntropy

        signals = _make_signals(30)
        feat = SignalEntropy(window=10)
        result = feat(signals)

        assert result.height == signals.height
        assert "signal_entropy_10" in result.columns

    def test_range_zero_one(self) -> None:
        from signalflow.ta.signal_features.stability import SignalEntropy

        signals = _make_signals(50)
        feat = SignalEntropy(window=20)
        result = feat(signals)

        non_null = result["signal_entropy_20"].drop_nulls()
        assert (non_null >= -0.01).all()  # Small tolerance for float precision
        assert (non_null <= 1.01).all()

    def test_pure_single_type_low_entropy(self) -> None:
        """All same signal_type → entropy should approach 0."""
        from signalflow.ta.signal_features.stability import SignalEntropy

        base = datetime(2024, 1, 1)
        signals = pl.DataFrame(
            {
                "pair": ["P0"] * 20,
                "timestamp": [base + timedelta(hours=i) for i in range(20)],
                "signal_type": ["rise"] * 20,
                "signal": [1] * 20,
                "probability": [0.8] * 20,
            }
        )
        feat = SignalEntropy(window=10)
        result = feat(signals)

        last = result.sort("timestamp").row(-1, named=True)
        assert last["signal_entropy_10"] < 0.01


class TestSignalTypeRatio:
    def test_basic(self) -> None:
        from signalflow.ta.signal_features.stability import SignalTypeRatio

        signals = _make_signals(30)
        feat = SignalTypeRatio(window=10)
        result = feat(signals)

        assert result.height == signals.height
        assert "rise_ratio_10" in result.columns
        assert "fall_ratio_10" in result.columns

    def test_ratios_sum_to_one(self) -> None:
        from signalflow.ta.signal_features.stability import SignalTypeRatio

        signals = _make_signals(30)
        feat = SignalTypeRatio(window=10)
        result = feat(signals)

        sums = (result["rise_ratio_10"] + result["fall_ratio_10"]).drop_nulls()
        for s in sums.to_list():
            assert abs(s - 1.0) < 0.01


# =============================================================================
# Probability features
# =============================================================================


class TestProbabilityMoments:
    def test_basic(self) -> None:
        from signalflow.ta.signal_features.probability import ProbabilityMoments

        signals = _make_signals(30)
        feat = ProbabilityMoments(window=10)
        result = feat(signals)

        assert result.height == signals.height
        assert "prob_mean_10" in result.columns
        assert "prob_std_10" in result.columns
        assert "prob_slope_10" in result.columns

    def test_no_probability_column(self) -> None:
        from signalflow.ta.signal_features.probability import ProbabilityMoments

        base = datetime(2024, 1, 1)
        signals = pl.DataFrame(
            {
                "pair": ["P0"] * 10,
                "timestamp": [base + timedelta(hours=i) for i in range(10)],
                "signal_type": ["rise"] * 10,
                "signal": [1] * 10,
            }
        )
        feat = ProbabilityMoments(window=5)
        result = feat(signals)

        # Should return nulls gracefully
        assert result.height == 10
        assert result["prob_mean_5"].null_count() == 10

    def test_mean_in_range(self) -> None:
        from signalflow.ta.signal_features.probability import ProbabilityMoments

        signals = _make_signals(30)
        feat = ProbabilityMoments(window=10)
        result = feat(signals)

        means = result["prob_mean_10"].drop_nulls()
        assert (means >= 0).all()
        assert (means <= 1).all()


class TestCalibrationError:
    def test_basic(self) -> None:
        from signalflow.ta.signal_features.probability import CalibrationError

        signals = _make_signals(30)
        labels = _make_labels(signals, accuracy=0.7)
        feat = CalibrationError(window=10)
        result = feat(signals, labels=labels)

        assert result.height == signals.height
        assert "calibration_err_10" in result.columns

    def test_non_negative(self) -> None:
        from signalflow.ta.signal_features.probability import CalibrationError

        signals = _make_signals(30)
        labels = _make_labels(signals, accuracy=0.7)
        feat = CalibrationError(window=10)
        result = feat(signals, labels=labels)

        non_null = result["calibration_err_10"].drop_nulls()
        assert (non_null >= 0).all()

    def test_requires_labels(self) -> None:
        from signalflow.ta.signal_features.probability import CalibrationError

        signals = _make_signals(10)
        feat = CalibrationError()
        with pytest.raises(ValueError, match="requires labels"):
            feat(signals, labels=None)


class TestBayesianPosterior:
    def test_basic(self) -> None:
        from signalflow.ta.signal_features.probability import BayesianPosterior

        signals = _make_signals(30)
        labels = _make_labels(signals, accuracy=0.7)
        feat = BayesianPosterior(window=10)
        result = feat(signals, labels=labels)

        assert result.height == signals.height
        assert "bayesian_prob" in result.columns

    def test_range_zero_one(self) -> None:
        from signalflow.ta.signal_features.probability import BayesianPosterior

        signals = _make_signals(30)
        labels = _make_labels(signals, accuracy=0.7)
        feat = BayesianPosterior(window=10)
        result = feat(signals, labels=labels)

        non_null = result["bayesian_prob"].drop_nulls()
        assert (non_null >= 0).all()
        assert (non_null <= 1).all()


# =============================================================================
# Accuracy features
# =============================================================================


class TestRollingAccuracy:
    def test_basic(self) -> None:
        from signalflow.ta.signal_features.accuracy import RollingAccuracy

        signals = _make_signals(30)
        labels = _make_labels(signals, accuracy=0.7)
        feat = RollingAccuracy(window=10)
        result = feat(signals, labels=labels)

        assert result.height == signals.height
        assert "rolling_acc_10" in result.columns

    def test_range_zero_one(self) -> None:
        from signalflow.ta.signal_features.accuracy import RollingAccuracy

        signals = _make_signals(30)
        labels = _make_labels(signals, accuracy=0.7)
        feat = RollingAccuracy(window=10)
        result = feat(signals, labels=labels)

        non_null = result["rolling_acc_10"].drop_nulls()
        assert (non_null >= 0).all()
        assert (non_null <= 1).all()

    def test_perfect_accuracy(self) -> None:
        from signalflow.ta.signal_features.accuracy import RollingAccuracy

        signals = _make_signals(20)
        labels = _make_labels(signals, accuracy=1.0)
        feat = RollingAccuracy(window=10)
        result = feat(signals, labels=labels)

        non_null = result["rolling_acc_10"].drop_nulls()
        # Perfect accuracy → all 1.0
        for v in non_null.to_list():
            assert abs(v - 1.0) < 0.01

    def test_look_ahead_prevention(self) -> None:
        """Supervised features must not use future labels."""
        from signalflow.ta.signal_features.accuracy import RollingAccuracy

        base = datetime(2024, 1, 1)
        n = 20
        signals = pl.DataFrame(
            {
                "pair": ["P0"] * n,
                "timestamp": [base + timedelta(hours=i) for i in range(n)],
                "signal_type": ["rise"] * n,
                "signal": [1] * n,
                "probability": [0.8] * n,
            }
        )
        # First 10: correct, resolve immediately
        # Last 10: wrong, resolve 100h later
        label_rows = []
        for i in range(n):
            ts = base + timedelta(hours=i)
            if i < 10:
                label_rows.append(
                    {"pair": "P0", "timestamp": ts, "label": "rise", "t_hit": ts}
                )
            else:
                label_rows.append(
                    {"pair": "P0", "timestamp": ts, "label": "fall", "t_hit": ts + timedelta(hours=100)}
                )
        labels = pl.DataFrame(label_rows)

        feat = RollingAccuracy(window=5)
        result = feat(signals, labels=labels)

        # At hour 12, only labels 0-9 should be visible → all correct → ~1.0
        row_12 = result.sort("timestamp").row(12, named=True)
        assert row_12["rolling_acc_5"] is not None
        assert row_12["rolling_acc_5"] > 0.9


class TestTypeConditionalAccuracy:
    def test_basic(self) -> None:
        from signalflow.ta.signal_features.accuracy import TypeConditionalAccuracy

        signals = _make_signals(30)
        labels = _make_labels(signals, accuracy=0.7)
        feat = TypeConditionalAccuracy(window=10)
        result = feat(signals, labels=labels)

        assert result.height == signals.height
        assert "cond_acc_rise_10" in result.columns
        assert "cond_acc_fall_10" in result.columns


class TestFalseSignalRate:
    def test_basic(self) -> None:
        from signalflow.ta.signal_features.accuracy import FalseSignalRate

        signals = _make_signals(30)
        labels = _make_labels(signals, accuracy=0.7)
        feat = FalseSignalRate(window=10)
        result = feat(signals, labels=labels)

        assert result.height == signals.height
        assert "fpr_10" in result.columns
        assert "fnr_10" in result.columns

    def test_range_zero_one(self) -> None:
        from signalflow.ta.signal_features.accuracy import FalseSignalRate

        signals = _make_signals(30)
        labels = _make_labels(signals, accuracy=0.7)
        feat = FalseSignalRate(window=10)
        result = feat(signals, labels=labels)

        for col_name in ["fpr_10", "fnr_10"]:
            non_null = result[col_name].drop_nulls()
            assert (non_null >= 0).all()
            assert (non_null <= 1).all()


# =============================================================================
# Performance features
# =============================================================================


class TestRollingExpectedValue:
    def test_basic(self) -> None:
        from signalflow.ta.signal_features.performance import RollingExpectedValue

        signals = _make_signals(30)
        labels = _make_labels(signals, accuracy=0.7, with_ret=True)
        feat = RollingExpectedValue(window=10)
        result = feat(signals, labels=labels)

        assert result.height == signals.height
        assert "rolling_ev_10" in result.columns

    def test_without_ret_column(self) -> None:
        """Falls back to hit/miss ±1 proxy."""
        from signalflow.ta.signal_features.performance import RollingExpectedValue

        signals = _make_signals(30)
        labels = _make_labels(signals, accuracy=0.7, with_ret=False)
        feat = RollingExpectedValue(window=10)
        result = feat(signals, labels=labels)

        assert result.height == signals.height
        assert "rolling_ev_10" in result.columns

    def test_positive_ev_high_accuracy(self) -> None:
        from signalflow.ta.signal_features.performance import RollingExpectedValue

        signals = _make_signals(30)
        labels = _make_labels(signals, accuracy=1.0, with_ret=True)
        feat = RollingExpectedValue(window=10)
        result = feat(signals, labels=labels)

        non_null = result["rolling_ev_10"].drop_nulls()
        # Perfect accuracy with positive returns → EV should be positive
        assert (non_null > 0).all()


class TestRollingProfitFactor:
    def test_basic(self) -> None:
        from signalflow.ta.signal_features.performance import RollingProfitFactor

        signals = _make_signals(30)
        labels = _make_labels(signals, accuracy=0.7, with_ret=True)
        feat = RollingProfitFactor(window=10)
        result = feat(signals, labels=labels)

        assert result.height == signals.height
        assert "rolling_pf_10" in result.columns

    def test_perfect_accuracy_high_pf(self) -> None:
        from signalflow.ta.signal_features.performance import RollingProfitFactor

        signals = _make_signals(30)
        labels = _make_labels(signals, accuracy=1.0, with_ret=True)
        feat = RollingProfitFactor(window=10)
        result = feat(signals, labels=labels)

        non_null = result["rolling_pf_10"].drop_nulls()
        # All wins, no losses → PF should be null (no losses denominator)
        # or very high if there's any small loss
        assert len(non_null) == 0 or (non_null > 1).all()


class TestInformationCoefficient:
    def test_basic(self) -> None:
        from signalflow.ta.signal_features.performance import InformationCoefficient

        signals = _make_signals(30)
        labels = _make_labels(signals, accuracy=0.7, with_ret=True)
        feat = InformationCoefficient(window=10)
        result = feat(signals, labels=labels)

        assert result.height == signals.height
        assert "rolling_ic_10" in result.columns

    def test_without_signal_or_ret(self) -> None:
        """Without signal or ret column → all nulls."""
        from signalflow.ta.signal_features.performance import InformationCoefficient

        base = datetime(2024, 1, 1)
        signals = pl.DataFrame(
            {
                "pair": ["P0"] * 10,
                "timestamp": [base + timedelta(hours=i) for i in range(10)],
                "signal_type": ["rise"] * 10,
                "probability": [0.8] * 10,
            }
        )
        labels = _make_labels(signals, accuracy=0.7)
        feat = InformationCoefficient(window=5)
        result = feat(signals, labels=labels)

        assert result.height == 10
        assert result["rolling_ic_5"].null_count() == 10

    def test_range(self) -> None:
        from signalflow.ta.signal_features.performance import InformationCoefficient

        signals = _make_signals(50)
        labels = _make_labels(signals, accuracy=0.7, with_ret=True)
        feat = InformationCoefficient(window=20)
        result = feat(signals, labels=labels)

        non_null = result["rolling_ic_20"].drop_nulls()
        # Correlation ranges from -1 to 1
        assert (non_null >= -1.01).all()
        assert (non_null <= 1.01).all()


# =============================================================================
# Outcome features
# =============================================================================


class TestOutcomeStreak:
    def test_basic(self) -> None:
        from signalflow.ta.signal_features.outcome import OutcomeStreak

        signals = _make_signals(30)
        labels = _make_labels(signals, accuracy=0.7)
        feat = OutcomeStreak(window=10)
        result = feat(signals, labels=labels)

        assert result.height == signals.height
        assert "outcome_streak" in result.columns
        assert "outcome_autocorr_10" in result.columns

    def test_streak_sign(self) -> None:
        """Win streaks are positive, loss streaks negative."""
        from signalflow.ta.signal_features.outcome import OutcomeStreak

        signals = _make_signals(30)
        labels = _make_labels(signals, accuracy=0.7)
        feat = OutcomeStreak(window=10)
        result = feat(signals, labels=labels)

        non_null = result["outcome_streak"].drop_nulls()
        # Both positive and negative streaks should exist with 70% accuracy
        vals = non_null.to_list()
        has_positive = any(v > 0 for v in vals)
        has_negative = any(v < 0 for v in vals)
        assert has_positive
        assert has_negative


# =============================================================================
# Integration: import from package
# =============================================================================


class TestPackageImports:
    def test_import_all(self) -> None:
        from signalflow.ta.signal_features import (
            BayesianPosterior,
            CalibrationError,
            FalseSignalRate,
            InformationCoefficient,
            InterSignalDistance,
            OutcomeStreak,
            ProbabilityMoments,
            RollingAccuracy,
            RollingExpectedValue,
            RollingProfitFactor,
            SignalEntropy,
            SignalFlipRate,
            SignalFrequency,
            SignalStreak,
            SignalTypeRatio,
            TypeConditionalAccuracy,
        )

        all_classes = [
            BayesianPosterior,
            CalibrationError,
            FalseSignalRate,
            InformationCoefficient,
            InterSignalDistance,
            OutcomeStreak,
            ProbabilityMoments,
            RollingAccuracy,
            RollingExpectedValue,
            RollingProfitFactor,
            SignalEntropy,
            SignalFlipRate,
            SignalFrequency,
            SignalStreak,
            SignalTypeRatio,
            TypeConditionalAccuracy,
        ]

        for cls in all_classes:
            assert issubclass(cls, SignalFeature)

    def test_all_have_correct_component_type(self) -> None:
        from signalflow.ta.signal_features import (
            BayesianPosterior,
            CalibrationError,
            FalseSignalRate,
            InformationCoefficient,
            InterSignalDistance,
            OutcomeStreak,
            ProbabilityMoments,
            RollingAccuracy,
            RollingExpectedValue,
            RollingProfitFactor,
            SignalEntropy,
            SignalFlipRate,
            SignalFrequency,
            SignalStreak,
            SignalTypeRatio,
            TypeConditionalAccuracy,
        )

        supervised = [
            BayesianPosterior,
            CalibrationError,
            FalseSignalRate,
            InformationCoefficient,
            OutcomeStreak,
            RollingAccuracy,
            RollingExpectedValue,
            RollingProfitFactor,
            TypeConditionalAccuracy,
        ]
        unsupervised = [
            InterSignalDistance,
            ProbabilityMoments,
            SignalEntropy,
            SignalFlipRate,
            SignalFrequency,
            SignalStreak,
            SignalTypeRatio,
        ]

        for cls in supervised:
            assert cls.requires_labels is True, f"{cls.__name__} should require labels"
            assert cls.component_type == SfComponentType.SIGNAL_FEATURE

        for cls in unsupervised:
            assert cls.requires_labels is False, f"{cls.__name__} should not require labels"
            assert cls.component_type == SfComponentType.SIGNAL_FEATURE

    def test_import_from_ta_root(self) -> None:
        """All signal features should be importable from signalflow.ta."""
        from signalflow.ta import (
            BayesianPosterior,
            CalibrationError,
            FalseSignalRate,
            InformationCoefficient,
            InterSignalDistance,
            OutcomeStreak,
            ProbabilityMoments,
            RollingAccuracy,
            RollingExpectedValue,
            RollingProfitFactor,
            SignalEntropy,
            SignalFlipRate,
            SignalFrequency,
            SignalStreak,
            SignalTypeRatio,
            TypeConditionalAccuracy,
        )

        assert BayesianPosterior is not None
        assert CalibrationError is not None
        assert FalseSignalRate is not None
        assert InformationCoefficient is not None
        assert InterSignalDistance is not None
        assert OutcomeStreak is not None
        assert ProbabilityMoments is not None
        assert RollingAccuracy is not None
        assert RollingExpectedValue is not None
        assert RollingProfitFactor is not None
        assert SignalEntropy is not None
        assert SignalFlipRate is not None
        assert SignalFrequency is not None
        assert SignalStreak is not None
        assert SignalTypeRatio is not None
        assert TypeConditionalAccuracy is not None

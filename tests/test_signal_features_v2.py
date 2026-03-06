"""Tests for sf-ta signal features v2 (context, temporal, cross, information, adaptive)."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import polars as pl

from signalflow.core.enums import SfComponentType
from signalflow.signal_feature.base import SignalFeature

# =============================================================================
# Fixtures
# =============================================================================


def _make_signals(n: int = 30, pairs: int = 1) -> pl.DataFrame:
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
    with_ret: bool = False,
) -> pl.DataFrame:
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
            "t_hit": row["timestamp"],  # immediate resolution
        }
        if with_ret:
            if label == "rise":
                entry["ret"] = 0.02 if correct else 0.015
            else:
                entry["ret"] = -0.02 if correct else -0.015
        rows.append(entry)
    return pl.DataFrame(rows)


def _make_ohlcv(n: int = 30, pairs: int = 1) -> pl.DataFrame:
    """Synthetic OHLCV aligned with signals."""
    base = datetime(2024, 1, 1)
    rows: list[dict[str, Any]] = []
    for p in range(pairs):
        pair = f"PAIR{p}"
        price = 100.0
        for i in range(n):
            change = 0.5 * (1 if i % 4 < 2 else -1)
            price += change
            rows.append(
                {
                    "pair": pair,
                    "timestamp": base + timedelta(hours=i),
                    "open": price - 0.1,
                    "high": price + 0.5,
                    "low": price - 0.5,
                    "close": price,
                    "volume": 1000.0,
                }
            )
    return pl.DataFrame(rows)


# =============================================================================
# Context features
# =============================================================================


class TestRegimeSensitivity:
    def test_basic_with_context(self) -> None:
        from signalflow.ta.signal_features.context import RegimeSensitivity

        signals = _make_signals(50)
        labels = _make_labels(signals)
        ohlcv = _make_ohlcv(50)
        feat = RegimeSensitivity(window=20, vol_window=10)
        result = feat(signals, labels=labels, context={"ohlcv": ohlcv})

        assert result.height == signals.height
        assert "acc_high_vol_20" in result.columns
        assert "acc_low_vol_20" in result.columns
        assert "regime_spread_20" in result.columns

    def test_no_context_returns_nulls(self) -> None:
        from signalflow.ta.signal_features.context import RegimeSensitivity

        signals = _make_signals(20)
        labels = _make_labels(signals)
        feat = RegimeSensitivity(window=10)
        result = feat(signals, labels=labels, context=None)

        assert result.height == signals.height
        assert result["acc_high_vol_10"].null_count() == 20

    def test_component_type(self) -> None:
        from signalflow.ta.signal_features.context import RegimeSensitivity

        feat = RegimeSensitivity()
        assert feat.component_type == SfComponentType.SIGNAL_FEATURE
        assert feat.requires_labels


class TestVolatilityAdjustedEV:
    def test_basic(self) -> None:
        from signalflow.ta.signal_features.context import VolatilityAdjustedEV

        signals = _make_signals(50)
        labels = _make_labels(signals, with_ret=True)
        ohlcv = _make_ohlcv(50)
        feat = VolatilityAdjustedEV(window=20, vol_window=10)
        result = feat(signals, labels=labels, context={"ohlcv": ohlcv})

        assert result.height == signals.height
        assert "vol_adj_ev_20" in result.columns

    def test_no_context(self) -> None:
        from signalflow.ta.signal_features.context import VolatilityAdjustedEV

        signals = _make_signals(20)
        labels = _make_labels(signals)
        feat = VolatilityAdjustedEV(window=10)
        result = feat(signals, labels=labels)

        assert result["vol_adj_ev_10"].null_count() == 20


class TestMomentumAlignment:
    def test_basic(self) -> None:
        from signalflow.ta.signal_features.context import MomentumAlignment

        signals = _make_signals(50)
        labels = _make_labels(signals)
        ohlcv = _make_ohlcv(50)
        feat = MomentumAlignment(window=20, mom_window=5)
        result = feat(signals, labels=labels, context={"ohlcv": ohlcv})

        assert result.height == signals.height
        assert "trend_aligned_acc_20" in result.columns
        assert "trend_counter_acc_20" in result.columns


# =============================================================================
# Temporal features
# =============================================================================


class TestTemporalBias:
    def test_basic(self) -> None:
        from signalflow.ta.signal_features.temporal import TemporalBias

        signals = _make_signals(50)
        labels = _make_labels(signals)
        feat = TemporalBias(window=20)
        result = feat(signals, labels=labels)

        assert result.height == signals.height
        assert "hour_acc_bias" in result.columns
        assert "weekday_acc_bias" in result.columns


class TestSignalAlphaDecay:
    def test_basic(self) -> None:
        from signalflow.ta.signal_features.temporal import SignalAlphaDecay

        signals = _make_signals(50)
        labels = _make_labels(signals)
        ohlcv = _make_ohlcv(50)
        feat = SignalAlphaDecay(window=20, near_horizon=1, far_horizon=5)
        result = feat(signals, labels=labels, context={"ohlcv": ohlcv})

        assert result.height == signals.height
        assert "alpha_halflife_20" in result.columns
        assert "alpha_decay_rate" in result.columns

    def test_no_context(self) -> None:
        from signalflow.ta.signal_features.temporal import SignalAlphaDecay

        signals = _make_signals(20)
        labels = _make_labels(signals)
        feat = SignalAlphaDecay(window=10)
        result = feat(signals, labels=labels)

        assert result["alpha_halflife_10"].null_count() == 20


class TestSignalLifetime:
    def test_basic(self) -> None:
        from signalflow.ta.signal_features.temporal import SignalLifetime

        signals = _make_signals(30)
        feat = SignalLifetime(window=10)
        result = feat(signals)

        assert result.height == signals.height
        assert "signal_lifetime_mean_10" in result.columns
        assert "signal_lifetime_std_10" in result.columns

    def test_unsupervised(self) -> None:
        from signalflow.ta.signal_features.temporal import SignalLifetime

        feat = SignalLifetime()
        assert not feat.requires_labels


# =============================================================================
# Cross features
# =============================================================================


class TestSignalCrowding:
    def test_basic_multi_pair(self) -> None:
        from signalflow.ta.signal_features.cross import SignalCrowding

        signals = _make_signals(30, pairs=5)
        feat = SignalCrowding(zscore_window=10)
        result = feat(signals)

        assert result.height == signals.height
        assert "crowding_ratio" in result.columns
        assert "crowding_zscore_10" in result.columns

    def test_range(self) -> None:
        from signalflow.ta.signal_features.cross import SignalCrowding

        signals = _make_signals(30, pairs=5)
        feat = SignalCrowding(zscore_window=10)
        result = feat(signals)

        ratios = result["crowding_ratio"].drop_nulls()
        assert (ratios >= 0).all()
        assert (ratios <= 1).all()

    def test_unsupervised(self) -> None:
        from signalflow.ta.signal_features.cross import SignalCrowding

        feat = SignalCrowding()
        assert not feat.requires_labels


class TestCrossPairSpillover:
    def test_basic(self) -> None:
        from signalflow.ta.signal_features.cross import CrossPairSpillover

        signals = _make_signals(50, pairs=3)
        labels = _make_labels(signals)
        ohlcv = _make_ohlcv(50, pairs=3)
        feat = CrossPairSpillover(window=20)
        result = feat(signals, labels=labels, context={"ohlcv": ohlcv})

        assert result.height == signals.height
        assert "spillover_ic_20" in result.columns

    def test_no_context(self) -> None:
        from signalflow.ta.signal_features.cross import CrossPairSpillover

        signals = _make_signals(20)
        labels = _make_labels(signals)
        feat = CrossPairSpillover(window=10)
        result = feat(signals, labels=labels)

        assert result["spillover_ic_10"].null_count() == 20


class TestSignalDisagreement:
    def test_basic(self) -> None:
        from signalflow.ta.signal_features.cross import SignalDisagreement

        signals = _make_signals(30)
        labels = _make_labels(signals)

        # Create multi-detector signals
        base = datetime(2024, 1, 1)
        all_sigs_rows: list[dict[str, Any]] = []
        for i in range(30):
            ts = base + timedelta(hours=i)
            for det in ["det_A", "det_B", "det_C"]:
                all_sigs_rows.append(
                    {
                        "pair": "PAIR0",
                        "timestamp": ts,
                        "signal_type": "rise" if (i + hash(det)) % 2 == 0 else "fall",
                        "detector": det,
                    }
                )
        all_signals = pl.DataFrame(all_sigs_rows)

        feat = SignalDisagreement(window=10)
        result = feat(signals, labels=labels, context={"all_signals": all_signals})

        assert result.height == signals.height
        assert "agreement_ratio" in result.columns
        assert "disagree_acc_10" in result.columns

    def test_no_context(self) -> None:
        from signalflow.ta.signal_features.cross import SignalDisagreement

        signals = _make_signals(20)
        labels = _make_labels(signals)
        feat = SignalDisagreement(window=10)
        result = feat(signals, labels=labels)

        assert result["agreement_ratio"].null_count() == 20


# =============================================================================
# Information features
# =============================================================================


class TestSignalSurprise:
    def test_basic(self) -> None:
        from signalflow.ta.signal_features.information import SignalSurprise

        signals = _make_signals(30)
        feat = SignalSurprise(window=10)
        result = feat(signals)

        assert result.height == signals.height
        assert "signal_surprise_10" in result.columns

    def test_non_negative(self) -> None:
        from signalflow.ta.signal_features.information import SignalSurprise

        signals = _make_signals(30)
        feat = SignalSurprise(window=10)
        result = feat(signals)

        # -log(p) should be >= 0 for p in [0,1]
        non_null = result["signal_surprise_10"].drop_nulls()
        assert (non_null >= -0.01).all()

    def test_unsupervised(self) -> None:
        from signalflow.ta.signal_features.information import SignalSurprise

        feat = SignalSurprise()
        assert not feat.requires_labels


class TestMutualInformation:
    def test_basic(self) -> None:
        from signalflow.ta.signal_features.information import MutualInformation

        signals = _make_signals(50)
        labels = _make_labels(signals, with_ret=True)
        feat = MutualInformation(window=20)
        result = feat(signals, labels=labels)

        assert result.height == signals.height
        assert "mi_signal_ret_20" in result.columns

    def test_no_signal_or_ret(self) -> None:
        from signalflow.ta.signal_features.information import MutualInformation

        base = datetime(2024, 1, 1)
        signals = pl.DataFrame(
            {
                "pair": ["P0"] * 10,
                "timestamp": [base + timedelta(hours=i) for i in range(10)],
                "signal_type": ["rise"] * 10,
                "probability": [0.8] * 10,
            }
        )
        labels = _make_labels(signals)
        feat = MutualInformation(window=5)
        result = feat(signals, labels=labels)

        assert result["mi_signal_ret_5"].null_count() == 10

    def test_non_negative(self) -> None:
        from signalflow.ta.signal_features.information import MutualInformation

        signals = _make_signals(50)
        labels = _make_labels(signals, with_ret=True)
        feat = MutualInformation(window=20)
        result = feat(signals, labels=labels)

        non_null = result["mi_signal_ret_20"].drop_nulls()
        # MI is always >= 0
        assert (non_null >= -0.01).all()


class TestBayesianSurpriseRate:
    def test_basic(self) -> None:
        from signalflow.ta.signal_features.information import BayesianSurpriseRate

        signals = _make_signals(30)
        labels = _make_labels(signals)
        feat = BayesianSurpriseRate(window=10)
        result = feat(signals, labels=labels)

        assert result.height == signals.height
        assert "bayes_surprise_10" in result.columns

    def test_non_negative(self) -> None:
        from signalflow.ta.signal_features.information import BayesianSurpriseRate

        signals = _make_signals(30)
        labels = _make_labels(signals)
        feat = BayesianSurpriseRate(window=10)
        result = feat(signals, labels=labels)

        non_null = result["bayes_surprise_10"].drop_nulls()
        # Absolute delta is always >= 0
        assert (non_null >= -0.01).all()


# =============================================================================
# Adaptive features
# =============================================================================


class TestSignalClusterQuality:
    def test_basic(self) -> None:
        from signalflow.ta.signal_features.adaptive import SignalClusterQuality

        signals = _make_signals(50)
        labels = _make_labels(signals)
        feat = SignalClusterQuality(window=20, cluster_gap=3, min_cluster=2)
        result = feat(signals, labels=labels)

        assert result.height == signals.height
        assert "cluster_acc_20" in result.columns
        assert "isolated_acc_20" in result.columns
        assert "cluster_ratio" in result.columns

    def test_ratio_range(self) -> None:
        from signalflow.ta.signal_features.adaptive import SignalClusterQuality

        signals = _make_signals(50)
        labels = _make_labels(signals)
        feat = SignalClusterQuality(window=20, cluster_gap=3, min_cluster=2)
        result = feat(signals, labels=labels)

        ratios = result["cluster_ratio"].drop_nulls()
        assert (ratios >= 0).all()
        assert (ratios <= 1).all()


class TestDrawdownSensitivity:
    def test_basic(self) -> None:
        from signalflow.ta.signal_features.adaptive import DrawdownSensitivity

        signals = _make_signals(50)
        labels = _make_labels(signals)
        ohlcv = _make_ohlcv(50)
        feat = DrawdownSensitivity(window=20, dd_threshold=0.01)
        result = feat(signals, labels=labels, context={"ohlcv": ohlcv})

        assert result.height == signals.height
        assert "dd_acc_20" in result.columns
        assert "normal_acc_20" in result.columns
        assert "dd_sensitivity" in result.columns

    def test_no_context(self) -> None:
        from signalflow.ta.signal_features.adaptive import DrawdownSensitivity

        signals = _make_signals(20)
        labels = _make_labels(signals)
        feat = DrawdownSensitivity(window=10)
        result = feat(signals, labels=labels)

        assert result["dd_acc_10"].null_count() == 20


class TestAdaptiveConfidence:
    def test_basic(self) -> None:
        from signalflow.ta.signal_features.adaptive import AdaptiveConfidence

        signals = _make_signals(30)
        labels = _make_labels(signals)
        feat = AdaptiveConfidence(span=10)
        result = feat(signals, labels=labels)

        assert result.height == signals.height
        assert "adaptive_conf" in result.columns

    def test_range(self) -> None:
        from signalflow.ta.signal_features.adaptive import AdaptiveConfidence

        signals = _make_signals(30)
        labels = _make_labels(signals)
        feat = AdaptiveConfidence(span=10)
        result = feat(signals, labels=labels)

        non_null = result["adaptive_conf"].drop_nulls()
        assert (non_null >= 0).all()
        assert (non_null <= 1).all()


class TestSignalFragility:
    def test_basic(self) -> None:
        from signalflow.ta.signal_features.adaptive import SignalFragility

        signals = _make_signals(60)
        labels = _make_labels(signals)
        feat = SignalFragility(window=10)
        result = feat(signals, labels=labels)

        assert result.height == signals.height
        assert "fragility_score" in result.columns

    def test_non_negative(self) -> None:
        from signalflow.ta.signal_features.adaptive import SignalFragility

        signals = _make_signals(60)
        labels = _make_labels(signals)
        feat = SignalFragility(window=10)
        result = feat(signals, labels=labels)

        non_null = result["fragility_score"].drop_nulls()
        # max - min is always >= 0
        assert (non_null >= -0.01).all()


# =============================================================================
# Integration: all new classes importable from package
# =============================================================================


class TestV2PackageImports:
    def test_import_all_new(self) -> None:
        from signalflow.ta.signal_features import (
            AdaptiveConfidence,
            BayesianSurpriseRate,
            CrossPairSpillover,
            DrawdownSensitivity,
            MomentumAlignment,
            MutualInformation,
            RegimeSensitivity,
            SignalAlphaDecay,
            SignalClusterQuality,
            SignalCrowding,
            SignalDisagreement,
            SignalLifetime,
            SignalSurprise,
            TemporalBias,
            VolatilityAdjustedEV,
        )

        all_new = [
            AdaptiveConfidence,
            BayesianSurpriseRate,
            CrossPairSpillover,
            DrawdownSensitivity,
            MomentumAlignment,
            MutualInformation,
            RegimeSensitivity,
            SignalAlphaDecay,
            SignalClusterQuality,
            SignalCrowding,
            SignalDisagreement,
            SignalLifetime,
            SignalSurprise,
            TemporalBias,
            VolatilityAdjustedEV,
        ]

        for cls in all_new:
            assert issubclass(cls, SignalFeature)
            assert cls.component_type == SfComponentType.SIGNAL_FEATURE

    def test_import_from_ta_root(self) -> None:
        from signalflow.ta import (
            AdaptiveConfidence,
            BayesianSurpriseRate,
            CrossPairSpillover,
            DrawdownSensitivity,
            MomentumAlignment,
            MutualInformation,
            RegimeSensitivity,
            SignalAlphaDecay,
            SignalClusterQuality,
            SignalCrowding,
            SignalDisagreement,
            SignalFragility,
            SignalLifetime,
            SignalSurprise,
            TemporalBias,
            VolatilityAdjustedEV,
        )

        for cls in [
            AdaptiveConfidence, BayesianSurpriseRate, CrossPairSpillover,
            DrawdownSensitivity, MomentumAlignment, MutualInformation,
            RegimeSensitivity, SignalAlphaDecay, SignalClusterQuality,
            SignalCrowding, SignalDisagreement, SignalFragility,
            SignalLifetime, SignalSurprise, TemporalBias,
            VolatilityAdjustedEV,
        ]:
            assert cls is not None

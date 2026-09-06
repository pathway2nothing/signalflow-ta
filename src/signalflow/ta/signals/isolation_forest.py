"""Isolation Forest-based anomaly detection signals."""

from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
import polars as pl
from sklearn.ensemble import IsolationForest

from signalflow.ta._compat import SignalDetector, Signals, SignalType, detector
from signalflow.ta.momentum import RsiMom
from signalflow.ta.performance import LogReturn
from signalflow.ta.signals.filters import SignalFilter


@dataclass
@detector("detector/isoforest_returns")
class IsoForestReturnsDetector(SignalDetector):
    """Isolation Forest anomaly detector using log returns.

    Detects anomalies in log return distribution using Isolation Forest.
    Anomalies (extreme negative returns) are potential long signals.

    Signal logic:
        - Computes rolling log returns
        - Trains Isolation Forest on rolling window
        - Anomaly with negative return = LONG signal (oversold)
        - Anomaly with positive return = SHORT signal (overbought)
    """

    return_periods: list[int] = field(default_factory=lambda: [1, 5, 15, 60])
    window: int = 1440
    contamination: float = 0.01
    n_estimators: int = 100
    anomaly_threshold: float = -0.5
    direction: str = "long"
    learned: ClassVar[bool] = True
    filters: list[SignalFilter] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.direction not in ("long", "short", "both"):
            raise ValueError(f"direction must be 'long', 'short', or 'both', got {self.direction}")

        self.features = [LogReturn(period=p) for p in self.return_periods]

    def detect(self, features: pl.DataFrame, context: dict[str, Any] | None = None) -> Signals:
        """Generate signals based on Isolation Forest anomaly detection."""
        pairs = features[self.pair_col].unique().sort().to_list()
        if len(pairs) > 1:
            results = []
            for pair in pairs:
                pair_df = features.filter(pl.col(self.pair_col) == pair)
                sig = self._detect_single(pair_df, context)
                if len(sig.value) > 0:
                    results.append(sig.value)
            if results:
                return Signals(pl.concat(results))
            return Signals(
                features.head(0).select(
                    [
                        self.pair_col,
                        self.ts_col,
                        pl.lit(0).alias("signal_type"),
                        pl.lit(0.0).alias("score"),
                    ]
                )
            )
        return self._detect_single(features, context)

    def _detect_single(self, features: pl.DataFrame, context: dict[str, Any] | None = None) -> Signals:
        n = len(features)

        return_cols = [f"logret_{p}_close" for p in self.return_periods]
        feature_matrix = np.column_stack([features[col].to_numpy() for col in return_cols])

        primary_return = features[return_cols[0]].to_numpy()

        anomaly_scores = np.full(n, np.nan)
        signal_type = np.full(n, SignalType.NONE.value)

        for i in range(self.window, n):
            window_data = feature_matrix[i - self.window : i]

            valid_mask = ~np.isnan(window_data).any(axis=1)
            valid_data = window_data[valid_mask]

            if len(valid_data) < 50:
                continue

            model = IsolationForest(
                n_estimators=self.n_estimators,
                contamination=self.contamination,
                random_state=42,
            )
            model.fit(valid_data)

            current_point = feature_matrix[i : i + 1]
            if not np.isnan(current_point).any():
                score = model.score_samples(current_point)[0]
                anomaly_scores[i] = score

                if score < self.anomaly_threshold:
                    if self.direction in ("long", "both") and primary_return[i] < 0:
                        signal_type[i] = SignalType.RISE.value
                    elif self.direction in ("short", "both") and primary_return[i] > 0:
                        signal_type[i] = SignalType.FALL.value

        out = features.select(
            [
                self.pair_col,
                self.ts_col,
                pl.Series(name="signal_type", values=signal_type),
                pl.Series(name="score", values=anomaly_scores),
            ]
        )

        if self.filters:
            combined_mask = np.ones(len(out), dtype=bool)
            for flt in self.filters:
                filter_mask = flt.apply(features).to_numpy()
                combined_mask = combined_mask & filter_mask

            out = out.with_columns(
                pl.when(pl.Series(values=combined_mask))
                .then(pl.col("signal_type"))
                .otherwise(pl.lit(SignalType.NONE.value))
                .alias("signal_type")
            )

        out = out.filter(pl.col("signal_type") != SignalType.NONE.value)

        return Signals(out)

    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable output."""
        base_warmup = self.window + max(self.return_periods)
        filter_warmup = max((f.warmup for f in self.filters), default=0)
        return max(base_warmup, filter_warmup)

    test_params: ClassVar[list[dict]] = [
        {"return_periods": [1, 5], "window": 500, "direction": "long"},
    ]


@dataclass
@detector("detector/isoforest_rsi")
class IsoForestRsiDetector(SignalDetector):
    """Isolation Forest anomaly detector using RSI.

    Detects anomalies in RSI distribution using Isolation Forest.
    Extreme RSI values are potential entry signals.

    Signal logic:
        - Computes RSI for multiple periods
        - Trains Isolation Forest on RSI feature space
        - Anomaly with low RSI = LONG signal (oversold)
        - Anomaly with high RSI = SHORT signal (overbought)
    """

    rsi_periods: list[int] = field(default_factory=lambda: [6, 14, 30])
    window: int = 1440
    contamination: float = 0.01
    n_estimators: int = 100
    anomaly_threshold: float = -0.5
    rsi_long_threshold: float = 30.0
    rsi_short_threshold: float = 70.0
    direction: str = "long"
    learned: ClassVar[bool] = True
    filters: list[SignalFilter] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.direction not in ("long", "short", "both"):
            raise ValueError(f"direction must be 'long', 'short', or 'both', got {self.direction}")

        self.features = [RsiMom(period=p) for p in self.rsi_periods]

    def detect(self, features: pl.DataFrame, context: dict[str, Any] | None = None) -> Signals:
        """Generate signals based on RSI anomaly detection."""
        pairs = features[self.pair_col].unique().sort().to_list()
        if len(pairs) > 1:
            results = []
            for pair in pairs:
                pair_df = features.filter(pl.col(self.pair_col) == pair)
                sig = self._detect_single(pair_df, context)
                if len(sig.value) > 0:
                    results.append(sig.value)
            if results:
                return Signals(pl.concat(results))
            return Signals(
                features.head(0).select(
                    [
                        self.pair_col,
                        self.ts_col,
                        pl.lit(0).alias("signal_type"),
                        pl.lit(0.0).alias("score"),
                    ]
                )
            )
        return self._detect_single(features, context)

    def _detect_single(self, features: pl.DataFrame, context: dict[str, Any] | None = None) -> Signals:
        n = len(features)

        rsi_cols = [f"rsi_{p}" for p in self.rsi_periods]
        feature_matrix = np.column_stack([features[col].to_numpy() for col in rsi_cols])

        primary_rsi = features[rsi_cols[0]].to_numpy()

        anomaly_scores = np.full(n, np.nan)
        signal_type = np.full(n, SignalType.NONE.value)

        for i in range(self.window, n):
            window_data = feature_matrix[i - self.window : i]

            valid_mask = ~np.isnan(window_data).any(axis=1)
            valid_data = window_data[valid_mask]

            if len(valid_data) < 50:
                continue

            model = IsolationForest(
                n_estimators=self.n_estimators,
                contamination=self.contamination,
                random_state=42,
            )
            model.fit(valid_data)

            current_point = feature_matrix[i : i + 1]
            if not np.isnan(current_point).any():
                score = model.score_samples(current_point)[0]
                anomaly_scores[i] = score

                if score < self.anomaly_threshold:
                    if self.direction in ("long", "both") and primary_rsi[i] < self.rsi_long_threshold:
                        signal_type[i] = SignalType.RISE.value
                    elif self.direction in ("short", "both") and primary_rsi[i] > self.rsi_short_threshold:
                        signal_type[i] = SignalType.FALL.value

        out = features.select(
            [
                self.pair_col,
                self.ts_col,
                pl.Series(name="signal_type", values=signal_type),
                pl.Series(name="score", values=anomaly_scores),
            ]
        )

        if self.filters:
            combined_mask = np.ones(len(out), dtype=bool)
            for flt in self.filters:
                filter_mask = flt.apply(features).to_numpy()
                combined_mask = combined_mask & filter_mask

            out = out.with_columns(
                pl.when(pl.Series(values=combined_mask))
                .then(pl.col("signal_type"))
                .otherwise(pl.lit(SignalType.NONE.value))
                .alias("signal_type")
            )

        out = out.filter(pl.col("signal_type") != SignalType.NONE.value)

        return Signals(out)

    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable output."""
        base_warmup = self.window + max(self.rsi_periods) * 10
        filter_warmup = max((f.warmup for f in self.filters), default=0)
        return max(base_warmup, filter_warmup)

    test_params: ClassVar[list[dict]] = [
        {"rsi_periods": [6, 14], "window": 500, "direction": "long"},
    ]


@dataclass
@detector("detector/isoforest_cross_sectional")
class IsoForestCrossSectionalDetector(SignalDetector):
    """Cross-sectional Isolation Forest detector.

    Detects anomalies using both time-series and cross-sectional features.
    Combines log returns, RSI, and volatility in a multi-dimensional space.

    Signal logic:
        - Computes multiple features (returns, RSI, volatility)
        - Trains Isolation Forest on combined feature space
        - Uses global features for market context
        - Anomaly in oversold conditions = LONG signal
        - Anomaly in overbought conditions = SHORT signal

    Requires global features in context:
        - market_volatility: mean volatility across market
        - market_rsi: RSI of market index (optional)
    """

    return_periods: list[int] = field(default_factory=lambda: [1, 5, 15])
    rsi_period: int = 14
    volatility_window: int = 60
    window: int = 1440
    contamination: float = 0.01
    n_estimators: int = 100
    anomaly_threshold: float = -0.5
    rsi_long_threshold: float = 30.0
    rsi_short_threshold: float = 70.0
    direction: str = "long"
    learned: ClassVar[bool] = True
    filters: list[SignalFilter] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.direction not in ("long", "short", "both"):
            raise ValueError(f"direction must be 'long', 'short', or 'both', got {self.direction}")

        self.rsi_col = f"rsi_{self.rsi_period}"
        self.features = [
            *[LogReturn(period=p) for p in self.return_periods],
            RsiMom(period=self.rsi_period),
        ]

    def detect(self, features: pl.DataFrame, context: dict[str, Any] | None = None) -> Signals:
        """Generate signals based on cross-sectional anomaly detection."""
        pairs = features[self.pair_col].unique().sort().to_list()
        if len(pairs) > 1:
            results = []
            for pair in pairs:
                pair_df = features.filter(pl.col(self.pair_col) == pair)
                sig = self._detect_single(pair_df, context)
                if len(sig.value) > 0:
                    results.append(sig.value)
            if results:
                return Signals(pl.concat(results))
            return Signals(
                features.head(0).select(
                    [
                        self.pair_col,
                        self.ts_col,
                        pl.lit(0).alias("signal_type"),
                        pl.lit(0.0).alias("score"),
                    ]
                )
            )
        return self._detect_single(features, context)

    def _detect_single(self, features: pl.DataFrame, context: dict[str, Any] | None = None) -> Signals:
        n = len(features)
        close = features["close"].to_numpy()
        high = features["high"].to_numpy()
        low = features["low"].to_numpy()

        volatility = np.full(n, np.nan)
        for i in range(self.volatility_window - 1, n):
            window_high = high[i - self.volatility_window + 1 : i + 1]
            window_low = low[i - self.volatility_window + 1 : i + 1]
            window_close = close[i - self.volatility_window + 1 : i + 1]
            price = (window_high + window_low + window_close) / 3
            returns = np.diff(price) / (price[:-1] + 1e-10)
            volatility[i] = np.std(returns, ddof=1) if len(returns) > 1 else np.nan

        return_cols = [f"logret_{p}_close" for p in self.return_periods]
        feature_list = [features[col].to_numpy() for col in return_cols]
        feature_list.append(features[self.rsi_col].to_numpy())
        feature_list.append(volatility)

        if context and "global_features" in context:
            global_feats = context["global_features"]
            df = features.join(global_feats, on=self.ts_col, how="left")

            if "market_volatility" in df.columns:
                feature_list.append(df["market_volatility"].to_numpy())
            if "market_rsi" in df.columns:
                feature_list.append(df["market_rsi"].to_numpy())

        feature_matrix = np.column_stack(feature_list)
        rsi = features[self.rsi_col].to_numpy()

        anomaly_scores = np.full(n, np.nan)
        signal_type = np.full(n, SignalType.NONE.value)

        for i in range(self.window, n):
            window_data = feature_matrix[i - self.window : i]

            valid_mask = ~np.isnan(window_data).any(axis=1)
            valid_data = window_data[valid_mask]

            if len(valid_data) < 50:
                continue

            model = IsolationForest(
                n_estimators=self.n_estimators,
                contamination=self.contamination,
                random_state=42,
            )
            model.fit(valid_data)

            current_point = feature_matrix[i : i + 1]
            if not np.isnan(current_point).any():
                score = model.score_samples(current_point)[0]
                anomaly_scores[i] = score

                if score < self.anomaly_threshold:
                    if self.direction in ("long", "both") and rsi[i] < self.rsi_long_threshold:
                        signal_type[i] = SignalType.RISE.value
                    elif self.direction in ("short", "both") and rsi[i] > self.rsi_short_threshold:
                        signal_type[i] = SignalType.FALL.value

        out = features.select(
            [
                self.pair_col,
                self.ts_col,
                pl.Series(name="signal_type", values=signal_type),
                pl.Series(name="score", values=anomaly_scores),
            ]
        )

        if self.filters:
            combined_mask = np.ones(len(out), dtype=bool)
            for flt in self.filters:
                filter_mask = flt.apply(features).to_numpy()
                combined_mask = combined_mask & filter_mask

            out = out.with_columns(
                pl.when(pl.Series(values=combined_mask))
                .then(pl.col("signal_type"))
                .otherwise(pl.lit(SignalType.NONE.value))
                .alias("signal_type")
            )

        out = out.filter(pl.col("signal_type") != SignalType.NONE.value)

        return Signals(out)

    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable output."""
        base_warmup = max(
            self.window,
            self.rsi_period * 10,
            self.volatility_window,
            max(self.return_periods),
        )
        filter_warmup = max((f.warmup for f in self.filters), default=0)
        return max(base_warmup, filter_warmup)

    test_params: ClassVar[list[dict]] = [
        {
            "return_periods": [1, 5],
            "rsi_period": 14,
            "window": 500,
            "direction": "long",
        },
    ]

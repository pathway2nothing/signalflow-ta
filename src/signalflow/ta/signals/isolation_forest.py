"""Isolation Forest-based anomaly detection signals."""

from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
import polars as pl
from sklearn.ensemble import IsolationForest

from signalflow import sf_component
from signalflow.core import Signals, SignalType
from signalflow.detector import SignalDetector
from signalflow.ta.momentum import RsiMom
from signalflow.ta.performance import LogReturn
from signalflow.ta.signals.filters import SignalFilter


from signalflow.ta.signals._utils import _rma_sma_init  # noqa: F401


@dataclass
@sf_component(name="ta/isolation_forest_1")
class IsolationForestDetector1(SignalDetector):
    """Isolation Forest anomaly detector using log returns.

    Detects anomalies in log return distribution using Isolation Forest.
    Anomalies (extreme negative returns) are potential long signals.

    Signal logic:
        - Computes rolling log returns
        - Trains Isolation Forest on rolling window
        - Anomaly with negative return = LONG signal (oversold)
        - Anomaly with positive return = SHORT signal (overbought)

    Attributes:
        return_periods: List of return periods to compute.
        window: Rolling window for training Isolation Forest.
        contamination: Expected proportion of outliers.
        n_estimators: Number of trees in the forest.
        anomaly_threshold: Threshold for anomaly score (default -0.5).
        direction: Signal direction - "long", "short", or "both".
        filters: List of SignalFilter instances to apply.

    Example:
        ```python
        from signalflow.ta.signals import IsolationForestDetector1

        detector = IsolationForestDetector1(
            return_periods=[1, 5, 15, 60],
            window=1440,
            contamination=0.01,
            direction="long"
        )
        signals = detector.run(raw_data_view)
        ```
    """

    return_periods: list[int] = field(default_factory=lambda: [1, 5, 15, 60])
    window: int = 1440
    contamination: float = 0.01
    n_estimators: int = 100
    anomaly_threshold: float = -0.5
    direction: str = "long"
    filters: list[SignalFilter] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.direction not in ("long", "short", "both"):
            raise ValueError(
                f"direction must be 'long', 'short', or 'both', got {self.direction}"
            )

        self.features = [LogReturn(period=p) for p in self.return_periods]

    def detect(
        self, features: pl.DataFrame, context: dict[str, Any] | None = None
    ) -> Signals:
        """Generate signals based on Isolation Forest anomaly detection.

        Args:
            features: DataFrame with computed log returns.
            context: Optional context dictionary.

        Returns:
            Signals container with detected anomaly signals.
        """
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
                        pl.lit(0.0).alias("signal"),
                    ]
                )
            )
        return self._detect_single(features, context)

    def _detect_single(
        self, features: pl.DataFrame, context: dict[str, Any] | None = None
    ) -> Signals:
        n = len(features)

        # Build feature matrix from log returns
        return_cols = [f"logret_{p}_close" for p in self.return_periods]
        feature_matrix = np.column_stack(
            [features[col].to_numpy() for col in return_cols]
        )

        # Primary return for direction determination
        primary_return = features[return_cols[0]].to_numpy()

        # Initialize output arrays
        anomaly_scores = np.full(n, np.nan)
        signal_type = np.full(n, SignalType.NONE.value)

        # Sliding window Isolation Forest
        for i in range(self.window, n):
            window_data = feature_matrix[i - self.window : i]

            # Remove NaN rows
            valid_mask = ~np.isnan(window_data).any(axis=1)
            valid_data = window_data[valid_mask]

            if len(valid_data) < 50:
                continue

            # Train Isolation Forest
            model = IsolationForest(
                n_estimators=self.n_estimators,
                contamination=self.contamination,
                random_state=42,
            )
            model.fit(valid_data)

            # Score current point
            current_point = feature_matrix[i : i + 1]
            if not np.isnan(current_point).any():
                score = model.score_samples(current_point)[0]
                anomaly_scores[i] = score

                # Generate signal if anomaly
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
                pl.Series(name="signal", values=anomaly_scores),
            ]
        )

        # Apply filters
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
@sf_component(name="ta/isolation_forest_2")
class IsolationForestDetector2(SignalDetector):
    """Isolation Forest anomaly detector using RSI.

    Detects anomalies in RSI distribution using Isolation Forest.
    Extreme RSI values are potential entry signals.

    Signal logic:
        - Computes RSI for multiple periods
        - Trains Isolation Forest on RSI feature space
        - Anomaly with low RSI = LONG signal (oversold)
        - Anomaly with high RSI = SHORT signal (overbought)

    Attributes:
        rsi_periods: List of RSI periods to compute.
        window: Rolling window for training Isolation Forest.
        contamination: Expected proportion of outliers.
        n_estimators: Number of trees in the forest.
        anomaly_threshold: Threshold for anomaly score.
        rsi_long_threshold: RSI threshold for long signals.
        rsi_short_threshold: RSI threshold for short signals.
        direction: Signal direction - "long", "short", or "both".
        filters: List of SignalFilter instances to apply.

    Example:
        ```python
        from signalflow.ta.signals import IsolationForestDetector2

        detector = IsolationForestDetector2(
            rsi_periods=[6, 14, 30],
            window=1440,
            contamination=0.01,
            direction="long"
        )
        signals = detector.run(raw_data_view)
        ```
    """

    rsi_periods: list[int] = field(default_factory=lambda: [6, 14, 30])
    window: int = 1440
    contamination: float = 0.01
    n_estimators: int = 100
    anomaly_threshold: float = -0.5
    rsi_long_threshold: float = 30.0
    rsi_short_threshold: float = 70.0
    direction: str = "long"
    filters: list[SignalFilter] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.direction not in ("long", "short", "both"):
            raise ValueError(
                f"direction must be 'long', 'short', or 'both', got {self.direction}"
            )

        self.features = [RsiMom(period=p) for p in self.rsi_periods]

    def detect(
        self, features: pl.DataFrame, context: dict[str, Any] | None = None
    ) -> Signals:
        """Generate signals based on RSI anomaly detection.

        Args:
            features: DataFrame with computed RSI values.
            context: Optional context dictionary.

        Returns:
            Signals container with detected anomaly signals.
        """
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
                        pl.lit(0.0).alias("signal"),
                    ]
                )
            )
        return self._detect_single(features, context)

    def _detect_single(
        self, features: pl.DataFrame, context: dict[str, Any] | None = None
    ) -> Signals:
        n = len(features)

        # Build feature matrix from RSI values
        rsi_cols = [f"rsi_{p}" for p in self.rsi_periods]
        feature_matrix = np.column_stack([features[col].to_numpy() for col in rsi_cols])

        # Primary RSI for threshold checks
        primary_rsi = features[rsi_cols[0]].to_numpy()

        # Initialize output arrays
        anomaly_scores = np.full(n, np.nan)
        signal_type = np.full(n, SignalType.NONE.value)

        # Sliding window Isolation Forest
        for i in range(self.window, n):
            window_data = feature_matrix[i - self.window : i]

            # Remove NaN rows
            valid_mask = ~np.isnan(window_data).any(axis=1)
            valid_data = window_data[valid_mask]

            if len(valid_data) < 50:
                continue

            # Train Isolation Forest
            model = IsolationForest(
                n_estimators=self.n_estimators,
                contamination=self.contamination,
                random_state=42,
            )
            model.fit(valid_data)

            # Score current point
            current_point = feature_matrix[i : i + 1]
            if not np.isnan(current_point).any():
                score = model.score_samples(current_point)[0]
                anomaly_scores[i] = score

                # Generate signal if anomaly with extreme RSI
                if score < self.anomaly_threshold:
                    if (
                        self.direction in ("long", "both")
                        and primary_rsi[i] < self.rsi_long_threshold
                    ):
                        signal_type[i] = SignalType.RISE.value
                    elif (
                        self.direction in ("short", "both")
                        and primary_rsi[i] > self.rsi_short_threshold
                    ):
                        signal_type[i] = SignalType.FALL.value

        out = features.select(
            [
                self.pair_col,
                self.ts_col,
                pl.Series(name="signal_type", values=signal_type),
                pl.Series(name="signal", values=anomaly_scores),
            ]
        )

        # Apply filters
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
@sf_component(name="ta/isolation_forest_3")
class IsolationForestDetector3(SignalDetector):
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

    Attributes:
        return_periods: List of return periods.
        rsi_period: RSI calculation period.
        volatility_window: Window for volatility calculation.
        window: Rolling window for Isolation Forest.
        contamination: Expected proportion of outliers.
        n_estimators: Number of trees in the forest.
        anomaly_threshold: Threshold for anomaly score.
        direction: Signal direction.
        filters: List of SignalFilter instances.

    Example:
        ```python
        from signalflow.ta.signals import IsolationForestDetector3
        from signalflow.ta.signals import compute_global_features, MarketVolatilityFeature

        global_feats = compute_global_features(
            all_pairs_df,
            features=[MarketVolatilityFeature()]
        )

        detector = IsolationForestDetector3(
            return_periods=[1, 5, 15],
            rsi_period=14,
            window=1440
        )
        signals = detector.run(raw_data_view, context={"global_features": global_feats})
        ```
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
    filters: list[SignalFilter] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.direction not in ("long", "short", "both"):
            raise ValueError(
                f"direction must be 'long', 'short', or 'both', got {self.direction}"
            )

        self.rsi_col = f"rsi_{self.rsi_period}"
        self.features = [
            *[LogReturn(period=p) for p in self.return_periods],
            RsiMom(period=self.rsi_period),
        ]

    def detect(
        self, features: pl.DataFrame, context: dict[str, Any] | None = None
    ) -> Signals:
        """Generate signals based on cross-sectional anomaly detection.

        Args:
            features: DataFrame with computed features.
            context: May contain "global_features" DataFrame.

        Returns:
            Signals container with detected anomaly signals.
        """
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
                        pl.lit(0.0).alias("signal"),
                    ]
                )
            )
        return self._detect_single(features, context)

    def _detect_single(
        self, features: pl.DataFrame, context: dict[str, Any] | None = None
    ) -> Signals:
        n = len(features)
        close = features["close"].to_numpy()
        high = features["high"].to_numpy()
        low = features["low"].to_numpy()

        # Compute volatility
        volatility = np.full(n, np.nan)
        for i in range(self.volatility_window - 1, n):
            window_high = high[i - self.volatility_window + 1 : i + 1]
            window_low = low[i - self.volatility_window + 1 : i + 1]
            window_close = close[i - self.volatility_window + 1 : i + 1]
            price = (window_high + window_low + window_close) / 3
            returns = np.diff(price) / (price[:-1] + 1e-10)
            volatility[i] = np.std(returns, ddof=1) if len(returns) > 1 else np.nan

        # Build feature matrix
        return_cols = [f"logret_{p}_close" for p in self.return_periods]
        feature_list = [features[col].to_numpy() for col in return_cols]
        feature_list.append(features[self.rsi_col].to_numpy())
        feature_list.append(volatility)

        # Add global features if available
        if context and "global_features" in context:
            global_feats = context["global_features"]
            df = features.join(global_feats, on=self.ts_col, how="left")

            if "market_volatility" in df.columns:
                feature_list.append(df["market_volatility"].to_numpy())
            if "market_rsi" in df.columns:
                feature_list.append(df["market_rsi"].to_numpy())

        feature_matrix = np.column_stack(feature_list)
        rsi = features[self.rsi_col].to_numpy()

        # Initialize output arrays
        anomaly_scores = np.full(n, np.nan)
        signal_type = np.full(n, SignalType.NONE.value)

        # Sliding window Isolation Forest
        for i in range(self.window, n):
            window_data = feature_matrix[i - self.window : i]

            # Remove NaN rows
            valid_mask = ~np.isnan(window_data).any(axis=1)
            valid_data = window_data[valid_mask]

            if len(valid_data) < 50:
                continue

            # Train Isolation Forest
            model = IsolationForest(
                n_estimators=self.n_estimators,
                contamination=self.contamination,
                random_state=42,
            )
            model.fit(valid_data)

            # Score current point
            current_point = feature_matrix[i : i + 1]
            if not np.isnan(current_point).any():
                score = model.score_samples(current_point)[0]
                anomaly_scores[i] = score

                # Generate signal if anomaly
                if score < self.anomaly_threshold:
                    if (
                        self.direction in ("long", "both")
                        and rsi[i] < self.rsi_long_threshold
                    ):
                        signal_type[i] = SignalType.RISE.value
                    elif (
                        self.direction in ("short", "both")
                        and rsi[i] > self.rsi_short_threshold
                    ):
                        signal_type[i] = SignalType.FALL.value

        out = features.select(
            [
                self.pair_col,
                self.ts_col,
                pl.Series(name="signal_type", values=signal_type),
                pl.Series(name="signal", values=anomaly_scores),
            ]
        )

        # Apply filters
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

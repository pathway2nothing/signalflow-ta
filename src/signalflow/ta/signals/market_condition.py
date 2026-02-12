"""Market condition-based signal detectors using global features."""

from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
import polars as pl

from signalflow import sf_component
from signalflow.core import Signals, SignalType, SignalCategory
from signalflow.detector import SignalDetector
from signalflow.ta.momentum import RsiMom
from signalflow.ta._normalization import normalize_zscore
from signalflow.ta.signals.filters import SignalFilter


@dataclass
@sf_component(name="ta/market_condition_1")
class MarketConditionDetector1(SignalDetector):
    """Market condition detector using RSI and global volatility.

    Generates signals when:
    - RSI is below threshold (oversold)
    - Market volatility conditions are met (high vol regime)

    Requires global features in context:
    - market_volatility: mean volatility across market
    - market_volatility_std: std of volatility across market

    Attributes:
        rsi_period: RSI calculation period.
        rsi_threshold: RSI threshold for entry (default 20).
        vol_threshold: Market volatility threshold.
        vol_std_threshold: Market volatility std threshold.
        direction: Signal direction - "long", "short", or "both".
        filters: List of SignalFilter instances to apply.

    Example:
        ```python
        from signalflow.ta.signals import MarketConditionDetector1
        from signalflow.ta.signals import compute_global_features, MarketVolatilityFeature

        # Compute global features first
        global_feats = compute_global_features(
            all_pairs_df,
            features=[MarketVolatilityFeature()]
        )

        # Run detector with global features in context
        detector = MarketConditionDetector1(rsi_period=6, rsi_threshold=20)
        signals = detector.run(raw_data_view, context={"global_features": global_feats})
        ```
    """

    signal_category = SignalCategory.MARKET_WIDE

    rsi_period: int = 6
    rsi_threshold: float = 20.0
    vol_threshold: float = 0.01
    vol_std_threshold: float = 0.009
    direction: str = "long"
    filters: list[SignalFilter] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.direction not in ("long", "short", "both"):
            raise ValueError(
                f"direction must be 'long', 'short', or 'both', got {self.direction}"
            )

        self.rsi_col = f"rsi_{self.rsi_period}"
        self.features = [RsiMom(period=self.rsi_period)]

    def detect(
        self, features: pl.DataFrame, context: dict[str, Any] | None = None
    ) -> Signals:
        """Generate signals based on RSI and market volatility conditions.

        Args:
            features: DataFrame with computed RSI values.
            context: Must contain "global_features" DataFrame with
                     market_volatility and market_volatility_std columns.

        Returns:
            Signals container with detected signals.
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
        # Get global features from context
        if context is None or "global_features" not in context:
            raise ValueError(
                "MarketConditionDetector1 requires 'global_features' in context"
            )

        global_feats = context["global_features"]

        # Join global features to pair data
        df = features.join(global_feats, on=self.ts_col, how="left")

        # Check required columns
        required = ["market_volatility", "market_volatility_std"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing global feature columns: {missing}")

        rsi = df[self.rsi_col].to_numpy()
        market_vol = df["market_volatility"].to_numpy()
        market_vol_std = df["market_volatility_std"].to_numpy()

        # Signal conditions
        rsi_oversold = rsi < self.rsi_threshold
        rsi_overbought = rsi > (100 - self.rsi_threshold)
        vol_condition = (market_vol_std > self.vol_std_threshold) & (
            market_vol > self.vol_threshold
        )

        # Build signal type array
        signal_type = np.full(len(df), SignalType.NONE.value)

        if self.direction in ("long", "both"):
            long_signal = rsi_oversold & vol_condition
            signal_type = np.where(long_signal, SignalType.RISE.value, signal_type)

        if self.direction in ("short", "both"):
            short_signal = rsi_overbought & vol_condition
            signal_type = np.where(short_signal, SignalType.FALL.value, signal_type)

        out = df.select(
            [
                self.pair_col,
                self.ts_col,
                pl.Series(name="signal_type", values=signal_type),
                pl.col(self.rsi_col).alias("signal"),
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
        base_warmup = self.rsi_period * 10
        filter_warmup = max((f.warmup for f in self.filters), default=0)
        return max(base_warmup, filter_warmup)

    test_params: ClassVar[list[dict]] = [
        {"rsi_period": 6, "rsi_threshold": 20.0, "direction": "long"},
        {"rsi_period": 14, "rsi_threshold": 30.0, "direction": "long"},
    ]


@dataclass
@sf_component(name="ta/market_condition_2")
class MarketConditionDetector2(SignalDetector):
    """Market condition detector with RSI comparison to market RSI.

    Generates signals when:
    - Asset RSI is below threshold
    - Asset RSI is below market RSI * ratio (relative weakness)
    - Market volatility conditions are met

    Requires global features in context:
    - market_rsi: RSI of market index
    - market_volatility: mean volatility across market
    - market_volatility_std: std of volatility across market

    Attributes:
        rsi_period: RSI calculation period.
        rsi_threshold: RSI threshold for entry.
        rsi_ratio: Asset RSI must be below market_rsi * ratio.
        vol_threshold: Market volatility threshold.
        vol_std_threshold: Market volatility std threshold.
        sum_vol_threshold: Combined vol + vol_std threshold.
        direction: Signal direction.
        filters: List of SignalFilter instances.
    """

    signal_category = SignalCategory.MARKET_WIDE

    rsi_period: int = 65
    rsi_smoothing: int = 5
    rsi_threshold: float = 40.0
    rsi_ratio: float = 0.8
    vol_threshold: float = 0.06
    vol_std_threshold: float = 0.1
    sum_vol_threshold: float = 0.024
    direction: str = "long"
    filters: list[SignalFilter] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.direction not in ("long", "short", "both"):
            raise ValueError(
                f"direction must be 'long', 'short', or 'both', got {self.direction}"
            )

        self.rsi_col = f"rsi_{self.rsi_period}"
        self.features = [RsiMom(period=self.rsi_period)]

    def detect(
        self, features: pl.DataFrame, context: dict[str, Any] | None = None
    ) -> Signals:
        """Generate signals based on RSI relative to market."""
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
        if context is None or "global_features" not in context:
            raise ValueError(
                "MarketConditionDetector2 requires 'global_features' in context"
            )

        global_feats = context["global_features"]
        df = features.join(global_feats, on=self.ts_col, how="left")

        required = ["market_rsi", "market_volatility", "market_volatility_std"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing global feature columns: {missing}")

        # Apply smoothing to RSI
        rsi_raw = df[self.rsi_col].to_numpy()
        n = len(rsi_raw)
        rsi = np.full(n, np.nan)
        for i in range(self.rsi_smoothing - 1, n):
            window = rsi_raw[i - self.rsi_smoothing + 1 : i + 1]
            valid = window[~np.isnan(window)]
            if len(valid) > 0:
                rsi[i] = np.mean(valid)

        market_rsi = df["market_rsi"].to_numpy()
        market_vol = df["market_volatility"].to_numpy()
        market_vol_std = df["market_volatility_std"].to_numpy()

        # Signal conditions
        rsi_below_threshold = rsi < self.rsi_threshold
        rsi_below_market = rsi < (market_rsi * self.rsi_ratio)
        vol_condition = (
            (market_vol_std > self.vol_std_threshold)
            & (market_vol > self.vol_threshold)
        ) | ((market_vol + market_vol_std) > self.sum_vol_threshold)

        signal_type = np.full(n, SignalType.NONE.value)

        if self.direction in ("long", "both"):
            long_signal = rsi_below_threshold & rsi_below_market & vol_condition
            signal_type = np.where(long_signal, SignalType.RISE.value, signal_type)

        if self.direction in ("short", "both"):
            rsi_above_threshold = rsi > (100 - self.rsi_threshold)
            rsi_above_market = rsi > (market_rsi * (2 - self.rsi_ratio))
            short_signal = rsi_above_threshold & rsi_above_market & vol_condition
            signal_type = np.where(short_signal, SignalType.FALL.value, signal_type)

        out = df.select(
            [
                self.pair_col,
                self.ts_col,
                pl.Series(name="signal_type", values=signal_type),
                pl.Series(name="signal", values=rsi),
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
        base_warmup = self.rsi_period * 10 + self.rsi_smoothing
        filter_warmup = max((f.warmup for f in self.filters), default=0)
        return max(base_warmup, filter_warmup)

    test_params: ClassVar[list[dict]] = [
        {"rsi_period": 65, "rsi_threshold": 40.0, "direction": "long"},
    ]


@dataclass
@sf_component(name="ta/market_condition_3")
class MarketConditionDetector3(SignalDetector):
    """Advanced market condition detector with z-score and rolling min.

    Generates signals when:
    - Base conditions (RSI + vol) are met, OR
    - Asset z-score is extremely low, OR
    - Market z-score is extremely low (forced entry)
    - AND price is at rolling minimum

    Requires global features in context:
    - market_rsi, market_volatility, market_volatility_std, market_zscore
    """

    signal_category = SignalCategory.MARKET_WIDE

    rsi_period: int = 50
    rsi_smoothing: int = 5
    rsi_threshold: float = 20.0
    rsi_ratio: float = 0.3
    vol_threshold: float = 0.04
    vol_std_threshold: float = 0.1
    sum_vol_threshold: float = 0.055
    zscore_window: int = 3700
    zscore_threshold: float = -4.0
    market_zscore_threshold: float = 0.0
    forced_zscore: float = -5.0
    forced_vol_multiplier: float = 2.0
    rolling_min_window: int = 2600
    direction: str = "long"
    filters: list[SignalFilter] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.direction not in ("long", "short", "both"):
            raise ValueError(
                f"direction must be 'long', 'short', or 'both', got {self.direction}"
            )

        self.rsi_col = f"rsi_{self.rsi_period}"
        self.features = [RsiMom(period=self.rsi_period)]

    def detect(
        self, features: pl.DataFrame, context: dict[str, Any] | None = None
    ) -> Signals:
        """Generate signals with multiple condition pathways."""
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
        if context is None or "global_features" not in context:
            raise ValueError(
                "MarketConditionDetector3 requires 'global_features' in context"
            )

        global_feats = context["global_features"]
        df = features.join(global_feats, on=self.ts_col, how="left")

        required = [
            "market_rsi",
            "market_volatility",
            "market_volatility_std",
            "market_zscore",
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing global feature columns: {missing}")

        close = df["close"].to_numpy()
        n = len(close)

        # Compute smoothed RSI
        rsi_raw = df[self.rsi_col].to_numpy()
        rsi = np.full(n, np.nan)
        for i in range(self.rsi_smoothing - 1, n):
            window = rsi_raw[i - self.rsi_smoothing + 1 : i + 1]
            valid = window[~np.isnan(window)]
            if len(valid) > 0:
                rsi[i] = np.mean(valid)

        # Compute asset z-score
        zscore = normalize_zscore(close, window=self.zscore_window)

        # Compute rolling min
        rolling_min = np.full(n, np.nan)
        for i in range(self.rolling_min_window - 1, n):
            rolling_min[i] = np.min(close[i - self.rolling_min_window + 1 : i + 1])

        market_rsi = df["market_rsi"].to_numpy()
        market_vol = df["market_volatility"].to_numpy()
        market_vol_std = df["market_volatility_std"].to_numpy()
        market_zscore = df["market_zscore"].to_numpy()

        # Condition 1: RSI + vol based
        rsi_cond = (rsi < self.rsi_threshold) & (rsi < market_rsi * self.rsi_ratio)
        vol_cond = (
            (market_vol_std > self.vol_std_threshold)
            & (market_vol > self.vol_threshold)
        ) | ((market_vol + market_vol_std) > self.sum_vol_threshold)
        base_signal = rsi_cond & vol_cond

        # Condition 2: Z-score based
        zscore_signal = (zscore < self.zscore_threshold) & (
            market_zscore > self.market_zscore_threshold
        )

        # Condition 3: Forced entry on extreme market conditions
        forced_signal = (market_zscore < self.forced_zscore) & (
            zscore < self.forced_zscore
        )

        # Condition 4: Extreme volatility
        extreme_vol = (market_vol + market_vol_std) > (
            self.sum_vol_threshold * self.forced_vol_multiplier
        )

        # Combine conditions
        main_signal = base_signal | zscore_signal | forced_signal | extreme_vol

        # Must be at rolling minimum
        at_min = close <= rolling_min
        final_signal = main_signal & at_min

        signal_type = np.full(n, SignalType.NONE.value)

        if self.direction in ("long", "both"):
            signal_type = np.where(final_signal, SignalType.RISE.value, signal_type)

        out = df.select(
            [
                self.pair_col,
                self.ts_col,
                pl.Series(name="signal_type", values=signal_type),
                pl.Series(name="signal", values=zscore),
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
        base_warmup = max(
            self.rsi_period * 10,
            self.zscore_window,
            self.rolling_min_window,
        )
        filter_warmup = max((f.warmup for f in self.filters), default=0)
        return max(base_warmup, filter_warmup)

    test_params: ClassVar[list[dict]] = [
        {"rsi_period": 50, "zscore_window": 1440, "rolling_min_window": 720},
    ]

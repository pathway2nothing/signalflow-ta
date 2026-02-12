"""Cross-pair correlation signal detectors."""

from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
import polars as pl

from signalflow import sf_component
from signalflow.core import Signals, SignalType, SignalCategory
from signalflow.detector import SignalDetector
from signalflow.ta.volatility import BollingerVol
from signalflow.ta.signals.filters import SignalFilter


@dataclass
@sf_component(name="ta/cross_pair_1")
class CrossPairDetector1(SignalDetector):
    """Cross-pair correlation detector with Bollinger Bands.

    Generates signals based on cross-pair z-score correlation and BB position.

    Signal logic:
        - Computes z-score of reference pair returns (e.g., BTC_USDT)
        - Computes z-score of target pair USDT returns
        - Multiplies z-scores for correlation signal
        - Combines with Bollinger Band signal (price below lower band)
        - LONG: BB signal AND correlation z-score below threshold

    Requires global features in context:
        - reference_returns: DataFrame with timestamp and reference pair returns
        - target_usdt_returns: DataFrame with timestamp and target USDT pair returns

    Attributes:
        bb_period: Bollinger Band period.
        bb_std: Bollinger Band standard deviation multiplier.
        zscore_window: Window for z-score normalization.
        zscore_threshold: Z-score threshold for signal.
        direction: Signal direction.
        filters: List of SignalFilter instances.

    Example:
        ```python
        from signalflow.ta.signals import CrossPairDetector1

        # Prepare context with reference pair data
        context = {
            "reference_returns": btc_usdt_returns_df,
            "target_usdt_returns": target_usdt_returns_df,
        }

        detector = CrossPairDetector1(
            bb_period=2880,
            zscore_window=2880,
            zscore_threshold=10,
            direction="long"
        )
        signals = detector.run(raw_data_view, context=context)
        ```
    """

    signal_category = SignalCategory.MARKET_WIDE

    bb_period: int = 2880
    bb_std: float = 1.0
    zscore_window: int = 2880
    zscore_threshold: float = 10.0
    direction: str = "long"
    filters: list[SignalFilter] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.direction not in ("long", "short", "both"):
            raise ValueError(
                f"direction must be 'long', 'short', or 'both', got {self.direction}"
            )

        self.bb_lower_col = f"bb_lower_{self.bb_period}"
        self.features = [BollingerVol(period=self.bb_period, std_dev=self.bb_std)]

    def _compute_zscore(self, values: np.ndarray) -> np.ndarray:
        """Compute rolling z-score."""
        n = len(values)
        zscore = np.full(n, np.nan)

        for i in range(self.zscore_window - 1, n):
            window_vals = values[i - self.zscore_window + 1 : i + 1]
            valid = window_vals[~np.isnan(window_vals)]
            if len(valid) > 1:
                mean = np.mean(valid)
                std = np.std(valid, ddof=1)
                if std > 1e-10:
                    zscore[i] = (values[i] - mean) / std

        return zscore

    def detect(
        self, features: pl.DataFrame, context: dict[str, Any] | None = None
    ) -> Signals:
        """Generate signals based on cross-pair correlation.

        Args:
            features: DataFrame with computed BB values.
            context: Must contain reference and target USDT returns.

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
        close = features["close"].to_numpy()
        bb_lower = features[self.bb_lower_col].to_numpy()
        n = len(close)

        # BB signal: price below lower band
        bb_signal = close < bb_lower
        bb_diff = bb_lower - close

        # Get cross-pair data from context
        if context is None:
            raise ValueError("CrossPairDetector1 requires context with cross-pair data")

        # Initialize correlation signal
        correlation_signal = np.ones(n, dtype=bool)

        if "reference_returns" in context and "target_usdt_returns" in context:
            ref_returns = context["reference_returns"]
            target_returns = context["target_usdt_returns"]

            # Join to features data
            df = features.join(ref_returns, on=self.ts_col, how="left")
            df = df.join(target_returns, on=self.ts_col, how="left")

            # Get return columns (assume named 'returns' or similar)
            ref_col = [
                c for c in df.columns if "ref" in c.lower() or "btc" in c.lower()
            ]
            target_col = [
                c for c in df.columns if "target" in c.lower() or "usdt" in c.lower()
            ]

            if ref_col and target_col:
                ref_vals = df[ref_col[0]].to_numpy()
                target_vals = df[target_col[0]].to_numpy()

                # Compute z-scores
                ref_zscore = self._compute_zscore(ref_vals)
                target_zscore = self._compute_zscore(target_vals)

                # Multiply z-scores for correlation
                zscore_mult = ref_zscore * target_zscore

                # Signal condition
                correlation_signal = zscore_mult < self.zscore_threshold

        elif "zscore_mult" in context:
            # Alternative: pre-computed correlation z-score
            zscore_mult = context["zscore_mult"]
            if isinstance(zscore_mult, pl.Series):
                zscore_mult = zscore_mult.to_numpy()
            correlation_signal = zscore_mult < self.zscore_threshold

        # Combine signals
        long_signal = bb_signal & correlation_signal
        short_signal = (
            close > features[f"bb_upper_{self.bb_period}"].to_numpy()
        ) & ~correlation_signal

        # Build signal type array
        signal_type = np.full(n, SignalType.NONE.value)

        if self.direction in ("long", "both"):
            signal_type = np.where(long_signal, SignalType.RISE.value, signal_type)

        if self.direction in ("short", "both"):
            signal_type = np.where(short_signal, SignalType.FALL.value, signal_type)

        out = features.select(
            [
                self.pair_col,
                self.ts_col,
                pl.Series(name="signal_type", values=signal_type),
                pl.Series(name="signal", values=bb_diff),
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
        base_warmup = max(self.bb_period, self.zscore_window) * 3
        filter_warmup = max((f.warmup for f in self.filters), default=0)
        return max(base_warmup, filter_warmup)

    test_params: ClassVar[list[dict]] = [
        {"bb_period": 720, "zscore_window": 720, "direction": "long"},
    ]

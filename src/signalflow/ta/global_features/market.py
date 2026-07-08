"""Market-wide global features computed across all pairs."""

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import polars as pl

from signalflow.ta._compat import GlobalFeature


def _rma_sma_init(values: np.ndarray, period: int) -> np.ndarray:
    """RMA with SMA initialization."""
    n = len(values)
    alpha = 1 / period
    rma = np.full(n, np.nan)

    if n < period:
        return rma

    rma[period - 1] = np.mean(values[:period])

    for i in range(period, n):
        rma[i] = alpha * values[i] + (1 - alpha) * rma[i - 1]

    return rma


@dataclass
class MarketVolatilityFeature(GlobalFeature):
    """Market-wide volatility statistics.

    Computes mean and std of volatility across all pairs.

    Outputs:
        - market_volatility: mean((high - low) / open) across pairs
        - market_volatility_std: std of volatility across pairs
    """

    outputs: ClassVar[list[str]] = ["market_volatility", "market_volatility_std"]

    def compute(self, df: pl.DataFrame, context: dict[str, Any] | None = None) -> pl.DataFrame:
        tmp = df.with_columns(((pl.col("high") - pl.col("low")) / (pl.col("open") + 1e-10)).alias("_volatility"))

        agg = tmp.group_by(self.ts_col).agg(
            [
                pl.col("_volatility").mean().alias("market_volatility"),
                pl.col("_volatility").std().alias("market_volatility_std"),
            ]
        )

        return df.join(agg, on=self.ts_col, how="left")

    @property
    def warmup(self) -> int:
        return 1


@dataclass
class MarketIndexFeature(GlobalFeature):
    """Market index (mean close across all pairs).

    Outputs:
        - market_close: mean close price across all pairs
    """

    outputs: ClassVar[list[str]] = ["market_close"]

    def compute(self, df: pl.DataFrame, context: dict[str, Any] | None = None) -> pl.DataFrame:
        agg = df.group_by(self.ts_col).agg(
            [
                pl.col("close").mean().alias("market_close"),
            ]
        )

        return df.join(agg, on=self.ts_col, how="left")

    @property
    def warmup(self) -> int:
        return 1


@dataclass
class MarketRsiFeature(GlobalFeature):
    """Market RSI computed from market index.

    Computes RSI of the mean close price across all pairs.

    Outputs:
        - market_rsi: RSI of market index
    """

    period: int = 14
    smoothing: int = 5

    outputs: ClassVar[list[str]] = ["market_rsi"]

    def compute(self, df: pl.DataFrame, context: dict[str, Any] | None = None) -> pl.DataFrame:
        market = df.group_by(self.ts_col).agg([pl.col("close").mean().alias("market_close")]).sort(self.ts_col)

        close = market["market_close"].to_numpy()
        n = len(close)

        diff = np.diff(close, prepend=close[0])
        diff[0] = 0

        gains = np.where(diff > 0, diff, 0)
        losses = np.where(diff < 0, -diff, 0)

        avg_gain = _rma_sma_init(gains, self.period)
        avg_loss = _rma_sma_init(losses, self.period)

        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))

        if self.smoothing > 1:
            smoothed_rsi = np.full(n, np.nan)
            for i in range(self.smoothing - 1, n):
                window = rsi[i - self.smoothing + 1 : i + 1]
                valid = window[~np.isnan(window)]
                if len(valid) > 0:
                    smoothed_rsi[i] = np.mean(valid)
            rsi = smoothed_rsi

        result = market.with_columns(pl.Series(name="market_rsi", values=rsi))

        return df.join(result.select([self.ts_col, "market_rsi"]), on=self.ts_col, how="left")

    @property
    def warmup(self) -> int:
        return self.period * 10 + self.smoothing


@dataclass
class MarketZscoreFeature(GlobalFeature):
    """Market z-score computed from market index.

    Computes rolling z-score of the mean close price across all pairs.

    Outputs:
        - market_zscore: z-score of market index
    """

    window: int = 1440

    outputs: ClassVar[list[str]] = ["market_zscore"]

    def compute(self, df: pl.DataFrame, context: dict[str, Any] | None = None) -> pl.DataFrame:
        market = df.group_by(self.ts_col).agg([pl.col("close").mean().alias("market_close")]).sort(self.ts_col)

        close = market["market_close"].to_numpy()
        n = len(close)

        zscore = np.full(n, np.nan)
        for i in range(self.window - 1, n):
            window_vals = close[i - self.window + 1 : i + 1]
            mean = np.mean(window_vals)
            std = np.std(window_vals, ddof=1)
            if std > 1e-10:
                zscore[i] = (close[i] - mean) / std

        result = market.with_columns(pl.Series(name="market_zscore", values=zscore))

        return df.join(result.select([self.ts_col, "market_zscore"]), on=self.ts_col, how="left")

    @property
    def warmup(self) -> int:
        return self.window


@dataclass
class MarketRollingMinFeature(GlobalFeature):
    """Check if current price is at rolling minimum.

    Useful for detecting local lows.

    Outputs:
        - at_rolling_min: boolean, True if close <= rolling min
    """

    window: int = 2600

    outputs: ClassVar[list[str]] = ["at_rolling_min"]

    def compute(self, df: pl.DataFrame, context: dict[str, Any] | None = None) -> pl.DataFrame:
        close = df["close"].to_numpy()
        n = len(close)

        rolling_min = np.full(n, np.nan)
        for i in range(self.window - 1, n):
            rolling_min[i] = np.min(close[i - self.window + 1 : i + 1])

        at_min = close <= rolling_min

        return df.with_columns(pl.Series(name="at_rolling_min", values=at_min))

    @property
    def warmup(self) -> int:
        return self.window


def compute_global_features(df: pl.DataFrame, features: list[GlobalFeature]) -> pl.DataFrame:
    """Compute multiple global features and join them."""
    if not features:
        ts_col = features[0].ts_col if features else "ts"
        return df.select(ts_col).unique().sort(ts_col)

    result = None
    ts_col = features[0].ts_col

    for feat in features:
        feat_df = feat.compute(df)
        result = feat_df if result is None else result.join(feat_df, on=ts_col, how="outer")

    assert result is not None
    return result.sort(ts_col)

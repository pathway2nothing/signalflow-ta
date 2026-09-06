"""Path streak / reversal features."""
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import polars as pl

from signalflow.ta._compat import Feature


@dataclass
class ReversalCount(Feature):
    """Count of local extrema in window: sign(diff) flips."""
    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["reversal_count_{window}"]
    window: int = 240

    def compute_pair(self, df):
        out = f"reversal_count_{self.window}"
        d = pl.col("close").diff()
        sign_change = (d.sign() != d.shift(1).sign()).cast(pl.Float32)
        return df.with_columns(sign_change.rolling_sum(self.window).alias(out))


@dataclass
class ZeroCrossingRate(Feature):
    """Count of return sign flips per window, normalized."""
    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["zcr_{window}"]
    window: int = 240

    def compute_pair(self, df):
        out = f"zcr_{self.window}"
        c = pl.col("close")
        ret = c.diff()
        cross = ret.sign().diff().abs()
        return df.with_columns((cross.rolling_sum(self.window) / (2 * self.window)).alias(out))


@dataclass
class MaxConsecutiveGainRun(Feature):
    """Longest streak of consecutive up-bars (close > open) in window."""
    requires: ClassVar[list[str]] = ["open", "close"]
    outputs: ClassVar[list[str]] = ["max_gain_run_{window}"]
    window: int = 240

    def compute_pair(self, df):
        out = f"max_gain_run_{self.window}"
        c = df.get_column("close").to_numpy()
        o = df.get_column("open").to_numpy()
        up = (c > o).astype(np.int32)
        n = len(c)
        if n < self.window:
            return df.with_columns(pl.Series(out, np.full(n, np.nan)))
        from numpy.lib.stride_tricks import sliding_window_view as swv
        windows = swv(up, window_shape=self.window)
        runs = np.zeros(len(windows), dtype=np.int32)
        for i, w in enumerate(windows):
            cur = 0
            mx = 0
            for x in w:
                if x:
                    cur += 1
                    mx = max(mx, cur)
                else:
                    cur = 0
            runs[i] = mx
        full = np.full(n, np.nan)
        full[self.window - 1:] = runs
        return df.with_columns(pl.Series(out, full))


@dataclass
class MaxConsecutiveLossRun(Feature):
    """Longest streak of consecutive down-bars (close < open) in window."""
    requires: ClassVar[list[str]] = ["open", "close"]
    outputs: ClassVar[list[str]] = ["max_loss_run_{window}"]
    window: int = 240

    def compute_pair(self, df):
        out = f"max_loss_run_{self.window}"
        c = df.get_column("close").to_numpy()
        o = df.get_column("open").to_numpy()
        down = (c < o).astype(np.int32)
        n = len(c)
        if n < self.window:
            return df.with_columns(pl.Series(out, np.full(n, np.nan)))
        from numpy.lib.stride_tricks import sliding_window_view as swv
        windows = swv(down, window_shape=self.window)
        runs = np.zeros(len(windows), dtype=np.int32)
        for i, w in enumerate(windows):
            cur = 0
            mx = 0
            for x in w:
                if x:
                    cur += 1
                    mx = max(mx, cur)
                else:
                    cur = 0
            runs[i] = mx
        full = np.full(n, np.nan)
        full[self.window - 1:] = runs
        return df.with_columns(pl.Series(out, full))


@dataclass
class LongestStreak(Feature):
    """max(longest_gain_run, longest_loss_run) per window - directional persistence."""
    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["longest_streak_{window}"]
    window: int = 240

    def compute_pair(self, df):
        out = f"longest_streak_{self.window}"
        c = df.get_column("close").to_numpy()
        n = len(c)
        if n < self.window:
            return df.with_columns(pl.Series(out, np.full(n, np.nan, dtype=np.float32)))
        sign = np.sign(np.diff(c, prepend=c[0]))
        from numpy.lib.stride_tricks import sliding_window_view as swv
        windows = swv(sign, window_shape=self.window)
        out_arr = np.full(n, np.nan, dtype=np.float32)
        for i in range(len(windows)):
            w = windows[i]
            longest_pos = longest_neg = cur_pos = cur_neg = 0
            for s in w:
                if s > 0:
                    cur_pos += 1
                    cur_neg = 0
                    if cur_pos > longest_pos:
                        longest_pos = cur_pos
                elif s < 0:
                    cur_neg += 1
                    cur_pos = 0
                    if cur_neg > longest_neg:
                        longest_neg = cur_neg
                else:
                    cur_pos = cur_neg = 0
            out_arr[self.window - 1 + i] = max(longest_pos, longest_neg)
        return df.with_columns(pl.Series(out, out_arr))

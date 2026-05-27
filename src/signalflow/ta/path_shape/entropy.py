"""Information-theoretic path entropy features."""
from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import polars as pl

from signalflow.feature.base import Feature


@dataclass
class ReturnSignEntropy(Feature):
    """Shannon entropy of return sign sequence {-1,0,+1} over window. Path predictability."""
    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["sign_entropy_{window}"]
    window: int = 240

    def compute_pair(self, df):
        out = f"sign_entropy_{self.window}"
        c = df.get_column("close").to_numpy()
        ret = np.diff(c, prepend=np.nan)
        signs = np.sign(ret)
        n = len(c)
        if n < self.window:
            return df.with_columns(pl.Series(out, np.full(n, np.nan, dtype=np.float32)))
        from numpy.lib.stride_tricks import sliding_window_view as swv
        ent = np.full(n, np.nan, dtype=np.float32)
        coded = np.where(signs > 0, 2, np.where(signs < 0, 1, 0)).astype(np.int8)
        windows = swv(coded, window_shape=self.window)
        for i in range(len(windows)):
            w = windows[i]
            counts = np.bincount(w, minlength=3) / self.window
            counts = counts[counts > 0]
            ent[self.window - 1 + i] = -(counts * np.log2(counts)).sum()
        return df.with_columns(pl.Series(out, ent))


@dataclass
class DirectionalEntropy(Feature):
    """Shannon entropy of joint (return_sign × magnitude_quintile) distribution.

    Captures path randomness scaled by move size.
    """
    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["dir_entropy_{window}"]
    window: int = 480

    def compute_pair(self, df):
        out = f"dir_entropy_{self.window}"
        c = df.get_column("close").to_numpy().astype(np.float64)
        n = len(c)
        if n < self.window:
            return df.with_columns(pl.Series(out, np.full(n, np.nan, dtype=np.float32)))
        ret = np.diff(c, prepend=c[0])
        absret = np.abs(ret)
        from numpy.lib.stride_tricks import sliding_window_view as swv
        rwins = swv(ret, window_shape=self.window)
        awins = swv(absret, window_shape=self.window)
        out_arr = np.full(n, np.nan, dtype=np.float32)
        for i in range(len(rwins)):
            r, a = rwins[i], awins[i]
            qs = np.quantile(a, [0.2, 0.4, 0.6, 0.8])
            mag_bin = np.digitize(a, qs)  # 0..4
            sign_bin = np.where(r > 0, 1, np.where(r < 0, 2, 0))
            joint = sign_bin * 5 + mag_bin
            counts = np.bincount(joint, minlength=15) / self.window
            counts = counts[counts > 0]
            out_arr[self.window - 1 + i] = -(counts * np.log2(counts)).sum()
        return df.with_columns(pl.Series(out, out_arr))


@dataclass
class VolumeEntropy(Feature):
    """Shannon entropy of volume distribution (10 quantized bins) over window."""
    requires: ClassVar[list[str]] = ["volume"]
    outputs: ClassVar[list[str]] = ["vol_entropy_{window}"]
    window: int = 480

    def compute_pair(self, df):
        out = f"vol_entropy_{self.window}"
        v = df.get_column("volume").to_numpy().astype(np.float64)
        n = len(v)
        if n < self.window:
            return df.with_columns(pl.Series(out, np.full(n, np.nan, dtype=np.float32)))
        from numpy.lib.stride_tricks import sliding_window_view as swv
        windows = swv(v, window_shape=self.window)
        ent = np.full(n, np.nan, dtype=np.float32)
        n_bins = 10
        for i in range(len(windows)):
            w = windows[i]
            lo, hi = w.min(), w.max()
            if hi - lo < 1e-9:
                ent[self.window - 1 + i] = 0.0
                continue
            bins = np.clip(((w - lo) / (hi - lo) * n_bins).astype(np.int32), 0, n_bins - 1)
            counts = np.bincount(bins, minlength=n_bins) / self.window
            counts = counts[counts > 0]
            ent[self.window - 1 + i] = -(counts * np.log2(counts)).sum()
        return df.with_columns(pl.Series(out, ent))

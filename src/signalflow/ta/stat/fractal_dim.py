"""Path fractal dimension estimators (Katz / Higuchi families).

Quantify multi-scale roughness of price paths. Values near 1.0 indicate
smooth directional trends; values near 2.0 indicate space-filling chaotic
walks.

Reference: Katz, M. J. (1988). Fractals and the analysis of waveforms.
Computers in Biology and Medicine 18(3):145-156.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import polars as pl
from numpy.lib.stride_tricks import sliding_window_view

from signalflow.core import feature
from signalflow.feature.base import Feature


@dataclass
@feature("stat/katz_fractal_dimension")
class KatzFractalDimensionStat(Feature):
    """Katz fractal dimension: log(N)/(log(N) + log(diameter/path_length)).

    Iter-29 stability: top mean MI_normalised = 0.107 across 6 stable triples.
    """

    period: int = 30
    source_col: str = "close"

    requires: ClassVar[list[str]] = ["close"]
    outputs: ClassVar[list[str]] = ["f021_katz_{period}"]
    test_params: ClassVar[list[dict]] = [
        {"period": 30},
        {"period": 120},
        {"period": 480},
    ]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        out_col = f"f021_katz_{self.period}"
        c = df[self.source_col].to_numpy().astype(np.float64)
        n = len(c); w = int(self.period)
        if n < w:
            return df.with_columns(pl.lit(np.nan).alias(out_col))
        diffs = np.abs(np.diff(c))
        wins_d = sliding_window_view(diffs, w - 1)
        L = wins_d.sum(axis=1)
        wins = sliding_window_view(c, w)
        # diameter = max distance from first point
        d = np.abs(wins - wins[:, 0:1]).max(axis=1)
        ratio = d / np.maximum(L, 1e-12)
        katz = np.log(w) / (np.log(w) + np.log(np.maximum(ratio, 1e-12)))
        pad = np.full(w - 1, np.nan, dtype=np.float64)
        return df.with_columns(pl.Series(out_col, np.concatenate([pad, katz]), dtype=pl.Float64))

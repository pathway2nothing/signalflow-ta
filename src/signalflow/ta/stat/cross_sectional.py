# src/signalflow/ta/stat/cross_sectional.py
"""Cross-sectional statistics — universal global features.

Computes statistics for any pre-computed column across all pairs at each
timestamp. No dependency on specific indicators — just pass a column name.

Example:
    >>> cs = CrossSectionalStat(col="rsi_14", stats=["rank", "zscore", "mean"])
    >>> df = cs.compute(df)
    # Adds: rsi_14_cs_rank, rsi_14_cs_zscore, rsi_14_cs_mean
"""

from dataclasses import dataclass, field
from typing import ClassVar, Any

import polars as pl

from signalflow import sf_component
from signalflow.feature.base import GlobalFeature


@dataclass
@sf_component(name="stat/cross_sectional")
class CrossSectionalStat(GlobalFeature):
    """Universal cross-sectional statistics for any column.

    Given a column already present in the DataFrame (e.g. ``rsi_14``,
    ``atr_20``, ``close``), computes statistics **across all pairs** at
    each timestamp and attaches the results back per row.

    Supported stats:
        rank      — percentile rank [0, 1] among all pairs at time t.
        zscore    — (value - cross_mean) / cross_std.
        mean      — cross-sectional mean (broadcast to every pair).
        std       — cross-sectional std (market dispersion).
        median    — cross-sectional median (broadcast).
        min       — cross-sectional minimum (broadcast).
        max       — cross-sectional maximum (broadcast).
        diff      — value - cross_mean (pair deviation from market).

    All computations use native Polars ``over()`` window expressions,
    so performance scales well regardless of pair count.

    Args:
        col: Name of the column to compute stats for. Must exist in the
             input DataFrame.
        stats: Which statistics to produce.
               Default: ``["rank", "zscore", "mean"]``.
        prefix: Optional prefix for output column names.
                Default: ``""`` (columns named ``{col}_cs_{stat}``).
    """

    col: str = "close"
    stats: list[str] = field(default_factory=lambda: ["rank", "zscore", "mean"])
    prefix: str = ""

    requires: ClassVar[list[str]] = []
    outputs: ClassVar[list[str]] = []

    _SUPPORTED_STATS: ClassVar[set[str]] = {
        "rank",
        "zscore",
        "mean",
        "std",
        "median",
        "min",
        "max",
        "diff",
    }

    def __post_init__(self):
        unknown = set(self.stats) - self._SUPPORTED_STATS
        if unknown:
            raise ValueError(
                f"Unknown stats: {unknown}. Supported: {sorted(self._SUPPORTED_STATS)}"
            )
        if not self.stats:
            raise ValueError("stats must contain at least one entry")

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def _out_name(self, stat: str) -> str:
        return f"{self.prefix}{self.col}_cs_{stat}"

    def output_cols(self, prefix: str = "") -> list[str]:
        return [f"{prefix}{self._out_name(s)}" for s in self.stats]

    def required_cols(self) -> list[str]:
        return [self.col]

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def compute(
        self, df: pl.DataFrame, context: dict[str, Any] | None = None
    ) -> pl.DataFrame:
        """Compute cross-sectional statistics across all pairs."""
        if self.col not in df.columns:
            raise ValueError(
                f"Column '{self.col}' not found in DataFrame. Available: {df.columns}"
            )

        src = pl.col(self.col)
        ts = self.ts_col
        exprs: list[pl.Expr] = []

        for stat in self.stats:
            name = self._out_name(stat)

            if stat == "rank":
                exprs.append((src.rank().over(ts) / src.count().over(ts)).alias(name))

            elif stat == "zscore":
                exprs.append(
                    ((src - src.mean().over(ts)) / src.std().over(ts)).alias(name)
                )

            elif stat == "mean":
                exprs.append(src.mean().over(ts).alias(name))

            elif stat == "std":
                exprs.append(src.std().over(ts).alias(name))

            elif stat == "median":
                exprs.append(src.median().over(ts).alias(name))

            elif stat == "min":
                exprs.append(src.min().over(ts).alias(name))

            elif stat == "max":
                exprs.append(src.max().over(ts).alias(name))

            elif stat == "diff":
                exprs.append((src - src.mean().over(ts)).alias(name))

        return df.with_columns(exprs)

    @property
    def warmup(self) -> int:
        return 0

    test_params: ClassVar[list[dict]] = [
        {"col": "close", "stats": ["rank", "zscore", "mean"]},
        {"col": "close", "stats": ["std", "median", "min", "max", "diff"]},
        {"col": "close", "stats": ["rank"]},
    ]

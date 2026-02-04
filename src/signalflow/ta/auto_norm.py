from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
import json
import math

import polars as pl


@dataclass
class AutoFeatureNormalizer:
    """
    One-class auto normalizer for time-series features (Polars).

    Input DF columns: [timestamp, pair, feature_...]
    Assumptions:
      - timestamp strictly increasing within each pair
      - no missing timestamps per pair
    Guarantees (after warmup):
      - result at time t depends only on last W samples (start-invariant)
      - full reproducibility via fit() artifact
    """

    ts_col: str = "timestamp"
    pair_col: str = "pair"

    window: int = 256
    warmup: int = 256

    eps: float = 1e-8
    skew_hi: float = 1.25          # skewness threshold to consider log transform
    outlier_ratio_hi: float = 25.0 # max(|x|)/p95(|x|) threshold for winsor
    scale_cv_hi: float = 0.80      # cross-pair scale dispersion threshold
    near_zero_mean: float = 0.15   # |mean| / (std+eps) small => centered-ish
    min_unique: int = 10           # ignore near-constant features

    winsor_low_q: float = 0.01
    winsor_high_q: float = 0.99

    keep_original: bool = False    # if False -> only normalized features
    always_include: Tuple[str, ...] = ("robust",)  # base channels for each feature: "robust"|"z"|"rank"|"none"
    allow_multi: bool = True       # allow multiple channels per feature

    artifact: Optional[Dict[str, Any]] = None

    def fit(self, df: pl.DataFrame) -> Dict[str, Any]:
        df = self._ensure_sorted(df)
        feature_cols = self._feature_cols(df)

        stats = self._compute_feature_stats(df, feature_cols)

        plan: Dict[str, Dict[str, Any]] = {}
        for col in feature_cols:
            col_stats = stats.get(col, {})
            plan[col] = self._decide_plan_for_col(col_stats)

        artifact = {
            "version": 1,
            "config": self._config_dict(),
            "feature_cols": feature_cols,
            "plan": plan,

        }
        self.artifact = artifact
        return artifact

    def transform(self, df: pl.DataFrame, artifact: Optional[Dict[str, Any]] = None) -> pl.DataFrame:
        art = artifact or self.artifact
        if art is None:
            raise ValueError("No artifact. Call fit() or pass artifact to transform().")

        df = self._ensure_sorted(df)
        feature_cols: List[str] = art["feature_cols"]
        plan: Dict[str, Dict[str, Any]] = art["plan"]

        exprs: List[pl.Expr] = []
        for col in feature_cols:
            p = plan.get(col, {"methods": []})
            methods: List[str] = p.get("methods", [])
            if not methods:
                methods = ["rolling_robust"]

            for m in methods:
                exprs.append(self._expr_for_method(col, m).alias(f"{col}__{m}"))

        base_cols = [self.ts_col, self.pair_col]
        if self.keep_original:
            return df.with_columns(exprs)
        return df.select(base_cols + exprs)

    def save(self, path: str, artifact: Optional[Dict[str, Any]] = None) -> None:
        art = artifact or self.artifact
        if art is None:
            raise ValueError("No artifact to save. Call fit() first.")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(art, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load(path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


    def _decide_plan_for_col(self, s: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decide which normalization methods to add for a feature.

        Methods (all strictly-causal rolling, start-invariant after warmup):
          - signed_log1p: sign(x)*log1p(|x|)
          - rolling_winsor: rolling quantile clip
          - rolling_robust: (x - rolling_median)/rolling_iqr
          - rolling_z: (x - rolling_mean)/rolling_std
          - rolling_rank: rolling percentile rank in window
        """
        methods: List[str] = []

        if s.get("n_unique_min", 0) < self.min_unique:
            return {"methods": ["rolling_robust"]}

        skew = abs(float(s.get("skew_median", 0.0)))
        mean_over_std = abs(float(s.get("mean_over_std_median", 999.0)))
        out_ratio = float(s.get("outlier_ratio_median", 0.0))
        scale_cv = float(s.get("scale_cv", 0.0))

        if out_ratio >= self.outlier_ratio_hi:
            methods.append("rolling_winsor")

        if skew >= self.skew_hi:
            methods.append("signed_log1p")

     
        if scale_cv >= self.scale_cv_hi:
            methods.append("rolling_robust")
            if self.allow_multi:
                methods.append("rolling_rank")
        else:
            if mean_over_std <= self.near_zero_mean and skew < self.skew_hi:
                methods.append("rolling_z")
            else:
                methods.append("rolling_robust")
                if self.allow_multi and skew < (self.skew_hi * 1.5):
                    methods.append("rolling_z")

        base = []
        for b in self.always_include:
            if b == "robust":
                base.append("rolling_robust")
            elif b == "z":
                base.append("rolling_z")
            elif b == "rank":
                base.append("rolling_rank")
        for b in base:
            if b not in methods:
                methods.insert(0, b)

        seen = set()
        methods = [m for m in methods if not (m in seen or seen.add(m))]
        return {"methods": methods}


    def _expr_for_method(self, col: str, method: str) -> pl.Expr:
        x = pl.col(col)

        if method == "signed_log1p":
            return pl.when(x.is_null()).then(None).otherwise(
                pl.when(x >= 0)
                .then((x + pl.lit(0.0)).abs().log1p())
                .otherwise(-(x.abs().log1p()))
            )

        if method == "rolling_winsor":
            lo = x.rolling_quantile(
                quantile=self.winsor_low_q,
                window_size=self.window,
                min_periods=self.warmup,
                interpolation="nearest",
            ).over(self.pair_col)
            hi = x.rolling_quantile(
                quantile=self.winsor_high_q,
                window_size=self.window,
                min_periods=self.warmup,
                interpolation="nearest",
            ).over(self.pair_col)
            return x.clip(lo, hi)

        if method == "rolling_robust":
            med = x.rolling_median(self.window, min_periods=self.warmup).over(self.pair_col)
            q1 = x.rolling_quantile(0.25, self.window, min_periods=self.warmup, interpolation="nearest").over(self.pair_col)
            q3 = x.rolling_quantile(0.75, self.window, min_periods=self.warmup, interpolation="nearest").over(self.pair_col)
            iqr = (q3 - q1)
            return (x - med) / (iqr.abs() + pl.lit(self.eps))

        if method == "rolling_z":
            mu = x.rolling_mean(self.window, min_periods=self.warmup).over(self.pair_col)
            sd = x.rolling_std(self.window, min_periods=self.warmup, ddof=1).over(self.pair_col)
            return (x - mu) / (sd.abs() + pl.lit(self.eps))

        if method == "rolling_rank":
            mn = x.rolling_min(self.window, min_periods=self.warmup).over(self.pair_col)
            mx = x.rolling_max(self.window, min_periods=self.warmup).over(self.pair_col)
            return (x - mn) / ((mx - mn).abs() + pl.lit(self.eps))

        raise ValueError(f"Unknown method: {method}")


    def _compute_feature_stats(self, df: pl.DataFrame, feature_cols: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Compute robust-enough per-feature stats to drive decisions.

        We compute stats per (pair, feature) and then summarize across pairs:
          - skew_median
          - mean_over_std_median
          - outlier_ratio_median: max(|x|) / p95(|x|)
          - n_unique_min
          - scale_cv: CV of per-pair p95(|x|) across pairs (scale mismatch)
        """
        aggs = []
        for c in feature_cols:
            absx = pl.col(c).abs()
            aggs.extend([
                pl.col(c).mean().alias(f"{c}__mean"),
                pl.col(c).std(ddof=1).alias(f"{c}__std"),
                pl.col(c).skew().alias(f"{c}__skew"),
                absx.max().alias(f"{c}__absmax"),
                absx.quantile(0.95, interpolation="nearest").alias(f"{c}__absq95"),
                pl.col(c).n_unique().alias(f"{c}__nuniq"),
            ])

        per_pair = df.group_by(self.pair_col).agg(aggs)

        out: Dict[str, Dict[str, Any]] = {}
        pairs_n = per_pair.height

        for c in feature_cols:
            mean_col = pl.col(f"{c}__mean")
            std_col = pl.col(f"{c}__std")
            skew_col = pl.col(f"{c}__skew")
            absmax_col = pl.col(f"{c}__absmax")
            absq95_col = pl.col(f"{c}__absq95")
            nuniq_col = pl.col(f"{c}__nuniq")

            tmp = per_pair.select([
                (mean_col / (std_col.abs() + pl.lit(self.eps))).abs().alias("mean_over_std"),
                skew_col.abs().alias("skew_abs"),
                (absmax_col / (absq95_col + pl.lit(self.eps))).alias("outlier_ratio"),
                absq95_col.alias("scale_proxy"),
                nuniq_col.alias("nuniq"),
            ])

            res = tmp.select([
                pl.col("mean_over_std").median().alias("mean_over_std_median"),
                pl.col("skew_abs").median().alias("skew_median"),
                pl.col("outlier_ratio").median().alias("outlier_ratio_median"),
                pl.col("nuniq").min().alias("n_unique_min"),
                (pl.col("scale_proxy").std(ddof=1) / (pl.col("scale_proxy").mean() + pl.lit(self.eps))).alias("scale_cv"),
            ]).to_dicts()[0]

            for k, v in list(res.items()):
                if v is None:
                    res[k] = 0.0

            out[c] = res

        return out


    def _config_dict(self) -> Dict[str, Any]:
        return {
            "ts_col": self.ts_col,
            "pair_col": self.pair_col,
            "window": self.window,
            "warmup": self.warmup,
            "eps": self.eps,
            "skew_hi": self.skew_hi,
            "outlier_ratio_hi": self.outlier_ratio_hi,
            "scale_cv_hi": self.scale_cv_hi,
            "near_zero_mean": self.near_zero_mean,
            "min_unique": self.min_unique,
            "winsor_low_q": self.winsor_low_q,
            "winsor_high_q": self.winsor_high_q,
            "keep_original": self.keep_original,
            "always_include": list(self.always_include),
            "allow_multi": self.allow_multi,
        }

    def _feature_cols(self, df: pl.DataFrame) -> List[str]:
        meta = {self.ts_col, self.pair_col}
        return [c for c in df.columns if c not in meta]

    def _ensure_sorted(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.sort([self.pair_col, self.ts_col])

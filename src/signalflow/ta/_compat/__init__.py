"""Compatibility shim mapping the OLD signalflow core API onto the core.

The 130 ``signalflow.ta`` indicator / detector / signal-feature files were written
against the old core (``signalflow.core``, ``signalflow.feature``,
``signalflow.detector``, ``signalflow.signal_feature``). The rewrite deleted those
modules and unified everything onto a single
:class:`signalflow.transform.base.Transform` contract.

This module reproduces the OLD public surface the ta files import - without
rewriting a single indicator - by subclassing the ``Transform`` base and replaying
the OLD ``compute_pair`` / ``detect`` / ``compute(signals, labels)`` semantics.

Exported names (the 11 the sweep redirects to here):

* ``Feature``            - per-pair indicator base (OLD ``feature.base.Feature``)
* ``GlobalFeature``      - cross-pair feature base
* ``FeaturePipeline``    - ordered feature composition (the core class, re-exported)
* ``SignalDetector``     - OLD detector surface mapped onto the ``signal`` column
* ``SignalFeature``      - meta-features over signal history
* ``feature`` / ``detector`` - registration decorators (re-export of core decorators)
* ``Signals``            - immutable long-format signal container
* ``SignalType``         - alias of ``Signal`` (RISE / FALL / NONE)
* ``SignalCategory``     - core enum (re-export)

Key adaptations:

* Datasets key time on ``ts`` (not ``timestamp``); the shim defaults
  ``ts_col="ts"``.
* The OLD ``Feature.outputs`` was a *class-level list of templates*; the core makes
  ``outputs`` an abstract *property*. The :class:`FeatureMeta` metaclass relocates
  any class-level ``outputs`` list to ``_output_templates`` so the property
  (returning :meth:`Feature.output_cols`) is not shadowed.
* OLD detectors emit a filtered long-format ``Signals`` frame keyed on
  ``(pair, ts, signal_type)``; the core wants a ``signal`` column appended to *every*
  row. :class:`SignalDetector.compute` runs the OLD ``detect`` then left-joins the
  ``signal_type`` back onto the full frame as the ``signal`` column.
"""


import contextlib
from abc import ABCMeta
from dataclasses import dataclass, field
from typing import Any, ClassVar

import polars as pl

from signalflow.decorators import detector as _v5_detector
from signalflow.decorators import feature as _v5_feature
from signalflow.detector import SignalDetector as _V5SignalDetector
from signalflow.enums import SIGNAL_COL, Signal, SignalCategory
from signalflow.transform import FeaturePipeline
from signalflow.transform.base import Transform, ensure_sorted

__all__ = [
    "Feature",
    "FeaturePipeline",
    "GlobalFeature",
    "SignalCategory",
    "SignalDetector",
    "SignalFeature",
    "SignalType",
    "Signals",
    "detector",
    "feature",
]

SignalType = Signal
"""OLD name for the discrete-signal enum (RISE / FALL / NONE)."""

_ACTIVE = {Signal.RISE.value, Signal.FALL.value}


def _sorted(df: pl.DataFrame, group_col: str, ts_col: str) -> pl.DataFrame:
    """(group, ts)-ordered frame without a copy when it already is (the core fast path)."""
    if (group_col, ts_col) == ("pair", "ts"):
        return ensure_sorted(df)
    return df.sort([group_col, ts_col])


def _tolerant(v5_decorator):
    """Wrap a core decorator so OLD-style extra kwargs (e.g. ``override=``) are accepted and ignored."""

    def factory(name: str, *_args: Any, **_kwargs: Any):
        return v5_decorator(name)

    return factory


feature = _tolerant(_v5_feature)
detector = _tolerant(_v5_detector)


class FeatureMeta(ABCMeta):
    """Relocate a class-level ``outputs`` list to ``_output_templates``.

    The OLD ``Feature`` declared ``outputs: ClassVar[list[str]]`` of templates.
    The ``Transform`` base exposes ``outputs`` as an abstract *property*. If a subclass
    sets ``outputs = [...]`` it would shadow the property and break the registry's
    ``outputs`` access. This metaclass moves any class-level list-valued ``outputs``
    into ``_output_templates`` at subclass-creation time, leaving the property
    intact.
    """

    def __new__(mcs, name, bases, ns, **kw):
        out = ns.get("outputs")
        if isinstance(out, (list, tuple)):
            ns["_output_templates"] = list(out)
            del ns["outputs"]
        return super().__new__(mcs, name, bases, ns, **kw)


@dataclass
class Feature(Transform, metaclass=FeatureMeta):
    """Per-pair feature base reproducing the OLD ``feature.base.Feature`` contract.

    Subclasses implement :meth:`compute_pair` and declare ``outputs`` (a list of
    templates such as ``["rsi_{period}"]``) and ``requires``. The base groups by
    pair, sorts by ``[group_col, ts_col]`` and maps ``compute_pair`` over groups.
    """

    requires: ClassVar[list[str]] = []
    test_params: ClassVar[list[dict]] = []
    is_recursive: ClassVar[bool] = False
    warmup_invariant: ClassVar[bool] = True
    component_type: ClassVar[str] = "feature"

    _output_templates: ClassVar[list[str]] = []

    group_col: str = "pair"
    ts_col: str = "ts"
    normalized: bool = False
    norm_period: int | None = None

    @property
    def outputs(self) -> list[str]:
        """Concrete output column names, with the normalization suffix applied."""
        if getattr(self, "normalized", False):
            resolved = self._normalized_output_cols()
            if resolved is not None:
                return resolved
        return self.output_cols()

    def _normalized_output_cols(self) -> "list[str] | None":
        """Output names under ``normalized=True``, matching each feature's own writer.

        Features that expose a name resolver (``_get_output_name(s)``) route naming
        through it, appending ``_norm`` per column while leaving already-bounded
        columns unsuffixed; delegate so declared names equal the columns written.
        Features without a resolver that still declare their own ``normalized`` field
        append ``_norm`` inline to every output. Features that merely inherit the base
        ``normalized`` flag ignore it and keep plain names, signalled by ``None``.
        """
        if hasattr(type(self), "_get_output_names"):
            return list(self._get_output_names())
        if hasattr(type(self), "_get_output_name"):
            return [self._get_output_name()]
        if self._declares_normalized():
            return [f"{name}_norm" for name in self.output_cols()]
        return None

    def _declares_normalized(self) -> bool:
        """True when a subclass (not the base) declares its own ``normalized`` field."""
        for klass in type(self).__mro__:
            if klass is Feature:
                return False
            if "normalized" in getattr(klass, "__annotations__", {}):
                return True
        return False

    @property
    def warmup(self) -> int:
        return 0

    def compute(self, df: pl.DataFrame, context: dict[str, Any] | None = None) -> pl.DataFrame:
        """Compute the feature for all pairs (OLD contract)."""
        sorted_df = _sorted(df, self.group_col, self.ts_col)
        return sorted_df.group_by(self.group_col, maintain_order=True).map_groups(self.compute_pair)

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute the feature for a single pair. Subclasses override."""
        raise NotImplementedError(f"{type(self).__name__} must implement compute_pair()")

    def output_cols(self, prefix: str = "") -> list[str]:
        """Concrete output column names with template substitution."""
        return [f"{prefix}{tpl.format(**self.__dict__)}" for tpl in self._output_templates]

    def required_cols(self) -> list[str]:
        """Concrete required column names with template substitution."""
        return [tpl.format(**self.__dict__) if "{" in tpl else tpl for tpl in self.requires]

    def assert_reproducible(self) -> None:
        """Mirror the OLD warmup-reproducibility contract."""
        if self.is_recursive and not self.warmup_invariant:
            raise RuntimeError(
                f"{type(self).__name__} is recursive and not warmup-invariant; "
                "live values will diverge from backtest."
            )


@dataclass
class GlobalFeature(Feature):
    """Cross-pair feature base reproducing the OLD ``GlobalFeature`` contract.

    Global features override :meth:`compute` directly (operating across all pairs
    at once), typically grouping by ``ts_col``. The OLD multi-source ``RawData``
    accessor (``get_source_data`` / ``iter_sources``) is gone in the rewrite; the few
    features that need it are guarded at their own import site (see
    ``ta/global_features``).
    """

    sources: list[str] | None = field(default=None)

    def compute(self, df: pl.DataFrame, context: dict[str, Any] | None = None) -> pl.DataFrame:
        """Subclasses override with cross-pair aggregation logic."""
        raise NotImplementedError(f"{type(self).__name__} must implement compute()")


@dataclass(frozen=True)
class Signals:
    """Immutable container for detector output (OLD ``core.Signals`` surface).

    Wraps a Polars DataFrame with at least ``(pair, ts, signal_type)`` columns.
    The ta detectors construct ``Signals(df)`` and read back ``.value``; the
    shim's :class:`SignalDetector` consumes ``.value`` to project the ``signal``
    column. Only the attributes/methods the ta files actually touch are provided.
    """

    value: pl.DataFrame

    def __len__(self) -> int:
        return self.value.height

    def __iter__(self):
        return iter((self.value,))

    @property
    def df(self) -> pl.DataFrame:
        return self.value

    def apply(self, transform) -> "Signals":
        return Signals(transform(self.value))

    def pipe(self, *transforms) -> "Signals":
        s = self
        for t in transforms:
            s = s.apply(t)
        return s


@dataclass
class SignalDetector(_V5SignalDetector):
    """OLD detector surface on top of the core ``SignalDetector``.

    OLD detectors:
      * declare ``self.features`` (a Feature / list / pipeline) in ``__post_init__``,
      * implement ``detect(features, context=None) -> Signals`` where the returned
        frame is *filtered* to active-signal rows keyed on ``(pair, ts, signal_type)``.

    The core wants ``detect(df) -> df`` appending a ``signal`` column to **every** row.
    This shim:
      1. computes ``self.features`` onto the frame (``preprocess``),
      2. runs the OLD ``detect`` to get the filtered ``Signals``,
      3. left-joins ``signal_type`` back onto the full frame as ``signal``,
         filling unmatched rows with ``NONE``.
    """

    pair_col: str = "pair"
    ts_col: str = "ts"
    signal_category: SignalCategory = SignalCategory.PRICE_DIRECTION
    require_probability: bool = False
    keep_only_latest_per_pair: bool = False

    features: Any = None

    @property
    def outputs(self) -> list[str]:
        return [SIGNAL_COL]

    @property
    def warmup(self) -> int:
        return 0

    def compute(self, df: pl.DataFrame, context: dict[str, Any] | None = None) -> pl.DataFrame:
        return self.detect_v5(_sorted(df, self.pair_col, self.ts_col), context=context)

    def detect_v5(self, df: pl.DataFrame, context: dict[str, Any] | None = None) -> pl.DataFrame:
        """Bridge: run OLD ``detect`` and project onto the ``signal`` column."""
        feats = self.preprocess(df, context=context)
        result = self.detect(feats, context=context)

        signals_df = result.value if isinstance(result, Signals) else result

        key = [self.pair_col, self.ts_col]
        src_col = "signal_type" if "signal_type" in signals_df.columns else SIGNAL_COL
        sig = signals_df.select(
            [*key, pl.col(src_col).cast(pl.Utf8).alias(SIGNAL_COL)]
        )
        sig = sig.filter(pl.col(SIGNAL_COL).is_in(list(_ACTIVE)))

        sig = sig.unique(subset=key, keep="last")

        out = df.join(sig, on=key, how="left")
        out = out.with_columns(
            pl.col(SIGNAL_COL).fill_null(Signal.NONE.value).alias(SIGNAL_COL)
        )
        return out

    def preprocess(self, df: pl.DataFrame, context: dict[str, Any] | None = None) -> pl.DataFrame:
        """Compute ``self.features`` onto the frame (OLD preprocess)."""
        out = _sorted(df, self.pair_col, self.ts_col)
        feats = self.features
        if feats is None:
            return out
        if isinstance(feats, (list, tuple)):
            for f in feats:
                out = f.compute(out)
        else:
            out = feats.compute(out)
        return out

    def detect(self, features: pl.DataFrame, context: dict[str, Any] | None = None):
        """Generate signals from features (subclasses override, returns Signals)."""
        raise NotImplementedError(f"{type(self).__name__} must implement detect()")


@dataclass
class SignalFeature(Transform):
    """Meta-feature over signal history (OLD ``signal_feature.base.SignalFeature``).

    Subclasses implement ``compute(signals, labels=None, context=None)`` returning a
    frame keyed on ``(pair, ts)`` with only the produced feature columns. These
    operate on the *signal* stream rather than raw OHLCV, so the ``compute(df)``
    just forwards ``df`` as the ``signals`` argument.
    """

    component_type: ClassVar[str] = "signal_feature"
    requires_labels: ClassVar[bool] = False
    _output_templates: ClassVar[list[str]] = []

    group_col: str = "pair"
    ts_col: str = "ts"
    label_resolve_col: str | None = "t_hit"
    label_delay: int | None = None

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        out = cls.__dict__.get("outputs")
        if isinstance(out, (list, tuple)):
            cls._output_templates = list(out)
            with contextlib.suppress(AttributeError):
                delattr(cls, "outputs")

    @property
    def outputs(self) -> list[str]:
        return self.output_cols()

    @property
    def warmup(self) -> int:
        return 0

    def output_cols(self) -> list[str]:
        return [tpl.format(**self.__dict__) for tpl in self._output_templates]

    def compute(
        self,
        signals: pl.DataFrame,
        labels: pl.DataFrame | None = None,
        context: dict[str, Any] | None = None,
    ) -> pl.DataFrame:
        raise NotImplementedError(f"{type(self).__name__} must implement compute()")

    def __call__(
        self,
        signals: pl.DataFrame,
        labels: pl.DataFrame | None = None,
        context: dict[str, Any] | None = None,
    ) -> pl.DataFrame:
        return self.compute(signals=signals, labels=labels, context=context)

    def prepare_labels(self, signals: pl.DataFrame, labels: pl.DataFrame) -> pl.DataFrame:
        key = [self.group_col, self.ts_col]
        label_cols = [c for c in labels.columns if c not in key]
        merged = signals.join(labels.select([*key, *label_cols]), on=key, how="left")
        if self.label_resolve_col is not None and self.label_resolve_col in merged.columns:
            merged = merged.with_columns(pl.col(self.label_resolve_col).alias("_resolved_at"))
        elif self.label_delay is not None:
            merged = merged.sort(key).with_columns(
                pl.col(self.ts_col).shift(-self.label_delay).over(self.group_col).alias("_resolved_at")
            )
        else:
            merged = merged.with_columns(pl.col(self.ts_col).alias("_resolved_at"))
        return merged

    def mask_unresolved(self, df: pl.DataFrame, label_col: str = "label") -> pl.DataFrame:
        if "_resolved_at" not in df.columns:
            raise ValueError("DataFrame missing '_resolved_at'. Call prepare_labels() first.")
        is_resolved = pl.col("_resolved_at") <= pl.col(self.ts_col)
        mask_cols = [label_col]
        if self.label_resolve_col and self.label_resolve_col in df.columns:
            mask_cols.append(self.label_resolve_col)
        exprs = [
            pl.when(is_resolved).then(pl.col(c)).otherwise(pl.lit(None)).alias(c)
            for c in mask_cols
            if c in df.columns
        ]
        return df.with_columns(exprs)

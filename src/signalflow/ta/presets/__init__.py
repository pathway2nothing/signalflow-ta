"""Strategy presets - ready-to-use FlowBuilder configurations."""


from typing import TYPE_CHECKING

from signalflow.ta.presets.catalog import PRESET_CATALOG, PresetInfo

if TYPE_CHECKING:
    from signalflow.api.flow import FlowBuilder

__all__ = ["PresetInfo", "get_preset", "list_presets", "load_preset"]


def list_presets(
    *,
    difficulty: str | None = None,
    tag: str | None = None,
) -> list[PresetInfo]:
    """Return all available presets, optionally filtered."""
    difficulty_order = {"beginner": 0, "intermediate": 1, "advanced": 2}
    results = list(PRESET_CATALOG.values())

    if difficulty is not None:
        results = [p for p in results if p.difficulty == difficulty]

    if tag is not None:
        tag_lower = tag.lower()
        results = [p for p in results if any(tag_lower in t.lower() for t in p.tags)]

    results.sort(key=lambda p: (difficulty_order.get(p.difficulty, 9), p.name))
    return results


def get_preset(name: str) -> PresetInfo:
    """Get preset metadata by name."""
    if name not in PRESET_CATALOG:
        available = ", ".join(sorted(PRESET_CATALOG.keys()))
        msg = f"Unknown preset {name!r}. Available: {available}"
        raise KeyError(msg)
    return PRESET_CATALOG[name]


def load_preset(name: str) -> "FlowBuilder":
    """Create a pre-configured :class:`FlowBuilder` from a preset.

    The returned builder has detector, entry, exit, capital, and fee
    already configured.  You still need to call ``.data(...)`` before
    ``.run()``.
    """
    from signalflow.api.flow import FlowBuilder

    preset = get_preset(name)

    builder = FlowBuilder(strategy_id=f"preset/{preset.name}")
    builder = builder.detector(preset.detector, **preset.detector_params)
    builder = builder.entry(**preset.entry_params)
    builder = builder.exit(**preset.exit_params)
    builder = builder.capital(preset.capital)
    builder = builder.fee(preset.fee)

    return builder

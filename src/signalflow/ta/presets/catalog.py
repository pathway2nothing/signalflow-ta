"""Strategy preset catalog - ready-to-use strategy configurations.

Each preset is a dict describing a complete FlowBuilder configuration
with sf-ta detectors, entry/exit rules, and recommended parameters.
"""


from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PresetInfo:
    """Metadata for a strategy preset."""

    name: str
    display_name: str
    description: str
    difficulty: str
    tags: tuple[str, ...]
    detector: str
    detector_params: dict[str, Any] = field(default_factory=dict)
    features: tuple[str, ...] = ()
    entry_params: dict[str, Any] = field(default_factory=dict)
    exit_params: dict[str, Any] = field(default_factory=dict)
    capital: float = 10_000.0
    fee: float = 0.001
    notes: str = ""


PRESET_CATALOG: dict[str, PresetInfo] = {
    "grid": PresetInfo(
        name="grid",
        display_name="Grid Trading",
        description=(
            "Places buy and sell orders at regular price intervals. "
            "Profits from range-bound markets by capturing small moves. "
            "Best suited for sideways/consolidating markets."
        ),
        difficulty="beginner",
        tags=("grid", "range-bound", "passive", "low-risk"),
        detector="ta/bollinger_band_1",
        detector_params={"period": 20, "direction": "both"},
        entry_params={"size_pct": 0.05, "max_positions": 5},
        exit_params={"tp": 0.015, "sl": 0.01},
        notes="Works best on 15m-1h timeframes for major pairs.",
    ),
    "momentum": PresetInfo(
        name="momentum",
        display_name="Momentum Crossover",
        description=(
            "Classic SMA crossover strategy with RSI confirmation. "
            "Goes long when fast SMA crosses above slow SMA and RSI is not overbought. "
            "Simple and well-understood approach to trend following."
        ),
        difficulty="beginner",
        tags=("momentum", "trend", "SMA", "crossover"),
        detector="ta/aroon_cross_1",
        detector_params={"period": 25, "direction": "long"},
        features=("momentum/rsi", "momentum/macd"),
        entry_params={"size_pct": 0.1, "max_positions": 3},
        exit_params={"tp": 0.03, "sl": 0.015, "trailing": True},
        notes="Start with 1h timeframe on BTC or ETH for stable signals.",
    ),
    "mean_reversion": PresetInfo(
        name="mean_reversion",
        display_name="Mean Reversion",
        description=(
            "Bollinger Band reversion strategy - enters on extreme deviations "
            "from the moving average and exits when price reverts to the mean. "
            "Higher win rate but smaller gains per trade."
        ),
        difficulty="intermediate",
        tags=("mean-reversion", "bollinger", "statistical", "range"),
        detector="ta/bollinger_band_1",
        detector_params={"period": 40, "direction": "both"},
        features=("momentum/rsi", "volatility/bollinger"),
        entry_params={"size_pct": 0.08, "max_positions": 4},
        exit_params={"tp": 0.02, "sl": 0.012},
        notes="Works well on 4h timeframe in stable markets. Avoid during breakouts.",
    ),
    "trend_following": PresetInfo(
        name="trend_following",
        display_name="Trend Following",
        description=(
            "ADX-confirmed trend strategy with Keltner Channel breakout. "
            "Only trades when ADX indicates a strong trend (>25). "
            "Uses trailing stop to ride trends as far as possible."
        ),
        difficulty="intermediate",
        tags=("trend", "ADX", "keltner", "breakout"),
        detector="ta/keltner_channel_1",
        detector_params={"period": 20, "direction": "both"},
        features=("trend/adx", "volatility/atr"),
        entry_params={"size_pct": 0.1, "max_positions": 2},
        exit_params={"tp": 0.05, "sl": 0.02, "trailing": True},
        notes="Best on 1h-4h. Larger stops needed for volatile assets.",
    ),
    "scalper": PresetInfo(
        name="scalper",
        display_name="Scalper",
        description=(
            "High-frequency RSI + Stochastic scalping strategy. "
            "Takes quick entries on short-term oversold/overbought conditions. "
            "Requires low-latency execution and tight spreads."
        ),
        difficulty="advanced",
        tags=("scalping", "high-frequency", "RSI", "stochastic"),
        detector="ta/stochastic_1",
        detector_params={"k_period": 14, "d_period": 3, "direction": "both"},
        features=("momentum/rsi", "momentum/stoch"),
        entry_params={"size_pct": 0.15, "max_positions": 1},
        exit_params={"tp": 0.008, "sl": 0.005},
        fee=0.0004,
        notes="Use 1m-5m timeframes. Requires maker-fee exchange tier.",
    ),
    "ml_ensemble": PresetInfo(
        name="ml_ensemble",
        display_name="ML Ensemble",
        description=(
            "Meta-labeling pipeline with signal features and ML validator. "
            "Detector generates candidate signals, validator filters with "
            "iTransformer model trained on signal meta-features."
        ),
        difficulty="advanced",
        tags=("ML", "meta-labeling", "iTransformer", "ensemble"),
        detector="ta/aroon_cross_1",
        detector_params={"period": 25, "direction": "long"},
        features=("momentum/rsi", "momentum/macd", "volatility/atr", "trend/adx"),
        entry_params={"size_pct": 0.1, "max_positions": 3},
        exit_params={"tp": 0.03, "sl": 0.015, "trailing": True},
        capital=10_000.0,
        notes="Requires sf-nn for iTransformer. Use .signal_features().validator('iTransformer').",
    ),
}

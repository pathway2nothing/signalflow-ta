"""Volume-based oscillators."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl

from signalflow import sf_component
from signalflow.feature.base import Feature
from typing import ClassVar


@dataclass
@sf_component(name="volume/mfi")
class MfiVolume(Feature):
    """Money Flow Index (MFI).

    Volume-weighted RSI using typical price.

    TP = (High + Low + Close) / 3
    Raw Money Flow = TP * Volume
    MFI = 100 * Pos_MF / (Pos_MF + Neg_MF)

    Bounded 0-100, similar interpretation to RSI:
    - MFI > 80: overbought
    - MFI < 20: oversold
    - Divergence signals potential reversals

    Reference: Gene Quong & Avrum Soudack
    https://www.investopedia.com/terms/m/mfi.asp
    """

    period: int = 14
    normalized: bool = False

    requires = ["high", "low", "close", "volume"]
    outputs = ["mfi_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        close = df["close"].to_numpy()
        volume = df["volume"].to_numpy()
        n = len(close)

        tp = (high + low + close) / 3
        rmf = tp * volume
        tp_diff = np.diff(tp, prepend=tp[0])
        pos_mf = np.where(tp_diff > 0, rmf, 0)
        neg_mf = np.where(tp_diff < 0, rmf, 0)

        mfi = np.full(n, np.nan)

        for i in range(self.period - 1, n):
            pos_sum = np.sum(pos_mf[i - self.period + 1 : i + 1])
            neg_sum = np.sum(neg_mf[i - self.period + 1 : i + 1])
            total = pos_sum + neg_sum

            if total > 0:
                mfi[i] = 100 * pos_sum / total

        # Normalization: [0, 100] → [0, 1]
        if self.normalized:
            mfi = mfi / 100

        col_name = self._get_output_name()
        return df.with_columns(pl.Series(name=col_name, values=mfi))

    def _get_output_name(self) -> str:
        """Generate output column name with normalization suffix."""
        suffix = "_norm" if self.normalized else ""
        return f"mfi_{self.period}{suffix}"

    test_params: ClassVar[list[dict]] = [
        {"period": 14},
        {"period": 14, "normalized": True},
        {"period": 30},
        {"period": 60},
    ]

    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable, reproducible output."""
        return self.period * 5


@dataclass
@sf_component(name="volume/cmf")
class CmfVolume(Feature):
    """Chaikin Money Flow (CMF).

    Measures buying/selling pressure over a period.

    CLV = ((Close - Low) - (High - Close)) / (High - Low)
    CMF = SUM(CLV * Volume, period) / SUM(Volume, period)

    Bounded approximately -1 to +1:
    - CMF > 0: buying pressure (accumulation)
    - CMF < 0: selling pressure (distribution)
    - CMF > 0.25: strong buying
    - CMF < -0.25: strong selling

    Reference: Marc Chaikin
    https://school.stockcharts.com/doku.php?id=technical_indicators:chaikin_money_flow_cmf
    """

    period: int = 20
    normalized: bool = False
    norm_period: int | None = None

    requires = ["high", "low", "close", "volume"]
    outputs = ["cmf_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        close = df["close"].to_numpy()
        volume = df["volume"].to_numpy()
        n = len(close)

        hl_range = high - low
        clv = np.where(hl_range > 0, ((close - low) - (high - close)) / hl_range, 0)

        mfv = clv * volume
        cmf = np.full(n, np.nan)

        for i in range(self.period - 1, n):
            mfv_sum = np.sum(mfv[i - self.period + 1 : i + 1])
            vol_sum = np.sum(volume[i - self.period + 1 : i + 1])

            if vol_sum > 0:
                cmf[i] = mfv_sum / vol_sum

        # Normalization: z-score for unbounded oscillator
        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window

            norm_window = self.norm_period or get_norm_window(self.period)
            cmf = normalize_zscore(cmf, window=norm_window)

        col_name = self._get_output_name()
        return df.with_columns(pl.Series(name=col_name, values=cmf))

    def _get_output_name(self) -> str:
        """Generate output column name with normalization suffix."""
        suffix = "_norm" if self.normalized else ""
        return f"cmf_{self.period}{suffix}"

    test_params: ClassVar[list[dict]] = [
        {"period": 20},
        {"period": 20, "normalized": True},
        {"period": 30},
        {"period": 60},
    ]

    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable, reproducible output."""
        base_warmup = self.period * 5
        if self.normalized:
            from signalflow.ta._normalization import get_norm_window

            norm_window = self.norm_period or get_norm_window(self.period)
            return base_warmup + norm_window
        return base_warmup


@dataclass
@sf_component(name="volume/efi")
class EfiVolume(Feature):
    """Elder's Force Index (EFI).

    Measures the force behind price movements.

    Force = (Close - Prev_Close) * Volume
    EFI = EMA(Force, period)

    Unbounded oscillator:
    - Positive: buying force
    - Negative: selling force
    - Magnitude indicates strength
    - Zero crossings signal changes

    Reference: Alexander Elder, "Trading for a Living"
    https://www.investopedia.com/terms/f/force-index.asp
    """

    period: int = 13
    normalized: bool = False
    norm_period: int | None = None

    requires = ["close", "volume"]
    outputs = ["efi_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        close = df["close"].to_numpy()
        volume = df["volume"].to_numpy()
        n = len(close)

        force = np.diff(close, prepend=np.nan) * volume
        force[0] = 0

        alpha = 2 / (self.period + 1)
        efi = np.full(n, np.nan)

        if n >= self.period:
            # Initialize with SMA for reproducibility
            efi[self.period - 1] = np.mean(force[: self.period])

        for i in range(self.period, n):
            efi[i] = alpha * force[i] + (1 - alpha) * efi[i - 1]

        # Normalization: z-score for unbounded oscillator
        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window

            norm_window = self.norm_period or get_norm_window(self.period)
            efi = normalize_zscore(efi, window=norm_window)

        col_name = self._get_output_name()
        return df.with_columns(pl.Series(name=col_name, values=efi))

    def _get_output_name(self) -> str:
        """Generate output column name with normalization suffix."""
        suffix = "_norm" if self.normalized else ""
        return f"efi_{self.period}{suffix}"

    test_params: ClassVar[list[dict]] = [
        {"period": 13},
        {"period": 13, "normalized": True},
        {"period": 30},
        {"period": 60},
    ]

    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable, reproducible output."""
        base_warmup = self.period * 5
        if self.normalized:
            from signalflow.ta._normalization import get_norm_window

            norm_window = self.norm_period or get_norm_window(self.period)
            return base_warmup + norm_window
        return base_warmup


@dataclass
@sf_component(name="volume/eom")
class EomVolume(Feature):
    """Ease of Movement (EMV/EOM).

    Relates price change to volume.

    Distance = ((High + Low)/2) - ((Prev_High + Prev_Low)/2)
    Box_Ratio = (Volume / divisor) / (High - Low)
    EMV = Distance / Box_Ratio
    EOM = SMA(EMV, period)

    Interpretation:
    - Positive: price moving up on low volume (easy movement)
    - Negative: price moving down on low volume
    - Near zero: consolidation or high volume moves

    Reference: Richard Arms
    https://school.stockcharts.com/doku.php?id=technical_indicators:ease_of_movement_emv
    """

    period: int = 14
    divisor: float = 100_000_000
    normalized: bool = False
    norm_period: int | None = None

    requires = ["high", "low", "volume"]
    outputs = ["eom_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        volume = df["volume"].to_numpy()
        n = len(high)

        hl2 = (high + low) / 2
        prev_hl2 = np.roll(hl2, 1)
        prev_hl2[0] = hl2[0]

        distance = hl2 - prev_hl2

        hl_range = high - low
        box_ratio = (volume / self.divisor) / (hl_range + 1e-10)

        emv = distance / (box_ratio + 1e-10)
        emv[0] = 0

        eom = np.full(n, np.nan)
        for i in range(self.period - 1, n):
            eom[i] = np.mean(emv[i - self.period + 1 : i + 1])

        # Normalization: z-score for unbounded oscillator
        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window

            norm_window = self.norm_period or get_norm_window(self.period)
            eom = normalize_zscore(eom, window=norm_window)

        col_name = self._get_output_name()
        return df.with_columns(pl.Series(name=col_name, values=eom))

    def _get_output_name(self) -> str:
        """Generate output column name with normalization suffix."""
        suffix = "_norm" if self.normalized else ""
        return f"eom_{self.period}{suffix}"

    test_params: ClassVar[list[dict]] = [
        {"period": 14, "divisor": 100_000_000},
        {"period": 14, "divisor": 100_000_000, "normalized": True},
        {"period": 30, "divisor": 100_000_000},
        {"period": 60, "divisor": 1_000_000},
    ]

    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable, reproducible output."""
        base_warmup = self.period * 5
        if self.normalized:
            from signalflow.ta._normalization import get_norm_window

            norm_window = self.norm_period or get_norm_window(self.period)
            return base_warmup + norm_window
        return base_warmup


@dataclass
@sf_component(name="volume/kvo")
class KvoVolume(Feature):
    """Klinger Volume Oscillator (KVO).

    Measures long-term money flow with trend detection.

    Trend = sign(HLC3 - Prev_HLC3)
    dm = High - Low
    cm = cumulative dm when trend unchanged, else resets
    VF = Volume * |2*(dm/cm) - 1| * Trend * 100
    KVO = EMA(VF, fast) - EMA(VF, slow)
    Signal = EMA(KVO, signal)

    Outputs:
    - kvo: oscillator line
    - kvo_signal: signal line

    Interpretation:
    - KVO above signal: bullish
    - KVO below signal: bearish
    - Zero crossings: trend confirmation

    Reference: Stephen Klinger
    https://www.investopedia.com/terms/k/klingeroscillator.asp
    """

    fast: int = 34
    slow: int = 55
    signal: int = 13
    normalized: bool = False
    norm_period: int | None = None

    requires = ["high", "low", "close", "volume"]
    outputs = ["kvo_{fast}_{slow}", "kvo_signal_{signal}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        close = df["close"].to_numpy()
        volume = df["volume"].to_numpy()
        n = len(close)

        hlc3 = (high + low + close) / 3

        hlc3_diff = np.diff(hlc3, prepend=np.nan)
        trend = np.sign(hlc3_diff)
        trend[0] = 0  # No trend on first bar

        sv = volume * trend

        alpha_fast = 2 / (self.fast + 1)
        alpha_slow = 2 / (self.slow + 1)
        alpha_sig = 2 / (self.signal + 1)

        ema_fast = np.full(n, np.nan)
        ema_slow = np.full(n, np.nan)

        # Initialize with SMA for reproducibility
        if n >= self.fast:
            ema_fast[self.fast - 1] = np.mean(sv[: self.fast])
        if n >= self.slow:
            ema_slow[self.slow - 1] = np.mean(sv[: self.slow])

        for i in range(self.fast, n):
            ema_fast[i] = alpha_fast * sv[i] + (1 - alpha_fast) * ema_fast[i - 1]
        for i in range(self.slow, n):
            ema_slow[i] = alpha_slow * sv[i] + (1 - alpha_slow) * ema_slow[i - 1]

        kvo = ema_fast - ema_slow

        kvo_signal = np.full(n, np.nan)

        # Initialize signal line with SMA for reproducibility
        # Both EMAs are valid starting from slow-1 (since slow >= fast)
        kvo_start = self.slow - 1

        if n >= kvo_start + self.signal:
            # Initialize signal with SMA of first signal_period KVO values
            init_idx = kvo_start + self.signal - 1
            kvo_signal[init_idx] = np.mean(kvo[kvo_start : kvo_start + self.signal])

        # Continue with EMA smoothing
        for i in range(kvo_start + self.signal, n):
            kvo_signal[i] = alpha_sig * kvo[i] + (1 - alpha_sig) * kvo_signal[i - 1]

        # Normalization: z-score for unbounded oscillator
        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window

            norm_window = self.norm_period or get_norm_window(self.slow)
            kvo = normalize_zscore(kvo, window=norm_window)
            kvo_signal = normalize_zscore(kvo_signal, window=norm_window)

        col_kvo, col_signal = self._get_output_names()
        return df.with_columns(
            [
                pl.Series(name=col_kvo, values=kvo),
                pl.Series(name=col_signal, values=kvo_signal),
            ]
        )

    def _get_output_names(self) -> tuple[str, str]:
        """Generate output column names with normalization suffix."""
        suffix = "_norm" if self.normalized else ""
        return (
            f"kvo_{self.fast}_{self.slow}{suffix}",
            f"kvo_signal_{self.signal}{suffix}",
        )

    test_params: ClassVar[list[dict]] = [
        {"fast": 34, "slow": 55, "signal": 13},
        {"fast": 34, "slow": 55, "signal": 13, "normalized": True},
        {"fast": 20, "slow": 40, "signal": 10},
        {"fast": 50, "slow": 80, "signal": 20},
    ]

    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable, reproducible output."""
        base_warmup = (self.slow + self.signal) * 5
        if self.normalized:
            from signalflow.ta._normalization import get_norm_window

            norm_window = self.norm_period or get_norm_window(self.slow)
            return base_warmup + norm_window
        return base_warmup


@dataclass
@sf_component(name="volume/vwap")
class VwapVolume(Feature):
    """Volume Weighted Average Price (VWAP).

    Average price weighted by volume.

    VWAP = cumsum(TP * Volume) / cumsum(Volume)

    Note: This is cumulative VWAP (session-based).
    For intraday, reset should occur at session start.

    Interpretation:
    - Price above VWAP: bullish
    - Price below VWAP: bearish
    - VWAP acts as support/resistance

    Reference: https://www.investopedia.com/terms/v/vwap.asp
    """

    normalized: bool = False
    norm_period: int | None = None

    requires = ["high", "low", "close", "volume"]
    outputs = ["vwap"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        close = df["close"].to_numpy()
        volume = df["volume"].to_numpy()

        tp = (high + low + close) / 3

        cum_tp_vol = np.cumsum(tp * volume)
        cum_vol = np.cumsum(volume)

        vwap = cum_tp_vol / (cum_vol + 1e-10)

        # Normalization: z-score for unbounded price-like indicator
        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window

            norm_window = self.norm_period or get_norm_window(20)
            vwap = normalize_zscore(vwap, window=norm_window)

        col_name = self._get_output_name()
        return df.with_columns(pl.Series(name=col_name, values=vwap))

    def _get_output_name(self) -> str:
        """Generate output column name with normalization suffix."""
        suffix = "_norm" if self.normalized else ""
        return f"vwap{suffix}"

    test_params: ClassVar[list[dict]] = [
        {},
        {"normalized": True},
    ]

    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable, reproducible output."""
        base_warmup = 100  # Cumulative indicator
        if self.normalized:
            from signalflow.ta._normalization import get_norm_window

            norm_window = self.norm_period or get_norm_window(20)
            return base_warmup + norm_window
        return base_warmup


@dataclass
@sf_component(name="volume/vwap_bands")
class VwapBandsVolume(Feature):
    """VWAP with Standard Deviation Bands.

    VWAP with rolling standard deviation bands.

    Outputs:
    - vwap: volume weighted average price
    - vwap_upper: VWAP + std_dev * rolling_std
    - vwap_lower: VWAP - std_dev * rolling_std

    Useful for:
    - Mean reversion strategies
    - Identifying extended moves
    - Dynamic support/resistance
    """

    period: int = 20
    std_dev: float = 2.0
    normalized: bool = False
    norm_period: int | None = None

    requires = ["high", "low", "close", "volume"]
    outputs = ["vwap", "vwap_upper_{period}", "vwap_lower_{period}"]

    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        close = df["close"].to_numpy()
        volume = df["volume"].to_numpy()
        n = len(close)

        tp = (high + low + close) / 3

        cum_tp_vol = np.cumsum(tp * volume)
        cum_vol = np.cumsum(volume)
        vwap = cum_tp_vol / (cum_vol + 1e-10)

        upper = np.full(n, np.nan)
        lower = np.full(n, np.nan)

        for i in range(self.period - 1, n):
            window = tp[i - self.period + 1 : i + 1]
            std = np.std(window, ddof=1)
            upper[i] = vwap[i] + self.std_dev * std
            lower[i] = vwap[i] - self.std_dev * std

        # Normalization: z-score for unbounded price-like indicators
        if self.normalized:
            from signalflow.ta._normalization import normalize_zscore, get_norm_window

            norm_window = self.norm_period or get_norm_window(self.period)
            vwap = normalize_zscore(vwap, window=norm_window)
            upper = normalize_zscore(upper, window=norm_window)
            lower = normalize_zscore(lower, window=norm_window)

        col_vwap, col_upper, col_lower = self._get_output_names()
        return df.with_columns(
            [
                pl.Series(name=col_vwap, values=vwap),
                pl.Series(name=col_upper, values=upper),
                pl.Series(name=col_lower, values=lower),
            ]
        )

    def _get_output_names(self) -> tuple[str, str, str]:
        """Generate output column names with normalization suffix."""
        suffix = "_norm" if self.normalized else ""
        return (
            f"vwap{suffix}",
            f"vwap_upper_{self.period}{suffix}",
            f"vwap_lower_{self.period}{suffix}",
        )

    test_params: ClassVar[list[dict]] = [
        {"period": 20, "std_dev": 2.0},
        {"period": 20, "std_dev": 2.0, "normalized": True},
        {"period": 30, "std_dev": 2.0},
        {"period": 60, "std_dev": 2.5},
    ]

    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable, reproducible output."""
        base_warmup = self.period * 5
        if self.normalized:
            from signalflow.ta._normalization import get_norm_window

            norm_window = self.norm_period or get_norm_window(self.period)
            return base_warmup + norm_window
        return base_warmup

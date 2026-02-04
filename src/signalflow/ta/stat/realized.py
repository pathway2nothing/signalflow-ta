# src/signalflow/ta/stat/realized.py
"""Realized volatility estimators using OHLC data."""
from dataclasses import dataclass

import numpy as np
import polars as pl

from signalflow import sf_component
from signalflow.feature.base import Feature
from typing import ClassVar


@dataclass
@sf_component(name="stat/realized_vol")
class RealizedVolStat(Feature):
    """Rolling Realized Volatility (Close-to-Close).
    
    Standard volatility estimate from log returns.
    
    RV = √(Σ(log(Ct/Ct-1))² / n) * √252
    
    Annualized by default.
    
    Reference: https://en.wikipedia.org/wiki/Realized_variance
    """
    
    period: int = 30
    annualize: bool = True
    trading_periods: int = 252 
    
    requires = ["close"]
    outputs = ["realized_vol_{period}"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        close = df["close"].to_numpy()
        n = len(close)
        
        log_ret = np.diff(np.log(close), prepend=np.nan)
        
        rv = np.full(n, np.nan)
        factor = np.sqrt(self.trading_periods) if self.annualize else 1.0
        
        for i in range(self.period, n):
            window = log_ret[i - self.period + 1:i + 1]
            rv[i] = np.sqrt(np.nansum(window ** 2) / self.period) * factor
        
        return df.with_columns(
            pl.Series(name=f"realized_vol_{self.period}", values=rv)
        )
    
    test_params: ClassVar[list[dict]] = [
        {"period": 30, "annualize": True, "trading_periods": 252},
        {"period": 60, "annualize": True, "trading_periods": 252},
        {"period": 120, "annualize": False, "trading_periods": 252},
    ]



    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable, reproducible output."""
        return getattr(self, "period", getattr(self, "length", getattr(self, "window", 20))) * 5

@dataclass
@sf_component(name="stat/parkinson_vol")
class ParkinsonVolStat(Feature):
    """Rolling Parkinson Volatility.
    
    Uses high-low range. More efficient than close-to-close.
    ~5x more efficient under ideal conditions.
    
    PV = √(1/(4n*ln2) * Σ(ln(H/L))²)
    
    Reference: Parkinson (1980)
    https://en.wikipedia.org/wiki/Parkinson_volatility
    """
    
    period: int = 30
    annualize: bool = True
    trading_periods: int = 252
    
    requires = ["high", "low"]
    outputs = ["parkinson_vol_{period}"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        n = len(high)
        
        hl_log = np.log(high / low)
        factor_base = 1 / (4 * np.log(2))
        factor_annual = np.sqrt(self.trading_periods) if self.annualize else 1.0
        
        pv = np.full(n, np.nan)
        for i in range(self.period - 1, n):
            window = hl_log[i - self.period + 1:i + 1]
            pv[i] = np.sqrt(factor_base * np.nansum(window ** 2) / self.period) * factor_annual
        
        return df.with_columns(
            pl.Series(name=f"parkinson_vol_{self.period}", values=pv)
        )
    
    test_params: ClassVar[list[dict]] = [
        {"period": 30, "annualize": True, "trading_periods": 252},
        {"period": 60, "annualize": True, "trading_periods": 252},
        {"period": 120, "annualize": True, "trading_periods": 252},
    ]

    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable, reproducible output."""
        return self.period * 5


@dataclass
@sf_component(name="stat/garman_klass_vol")
class GarmanKlassVolStat(Feature):
    """Rolling Garman-Klass Volatility.
    
    Uses OHLC. More efficient than Parkinson.
    ~8x more efficient than close-to-close.
    
    GK = √(0.5*(ln(H/L))² - (2ln2-1)*(ln(C/O))²)
    
    Reference: Garman & Klass (1980)
    """
    
    period: int = 30
    annualize: bool = True
    trading_periods: int = 252
    
    requires = ["open", "high", "low", "close"]
    outputs = ["gk_vol_{period}"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        o = df["open"].to_numpy()
        h = df["high"].to_numpy()
        l = df["low"].to_numpy()
        c = df["close"].to_numpy()
        n = len(c)
        
        hl = np.log(h / l) ** 2
        co = np.log(c / o) ** 2
        
        factor = 2 * np.log(2) - 1
        factor_annual = np.sqrt(self.trading_periods) if self.annualize else 1.0
        
        gk = np.full(n, np.nan)
        for i in range(self.period - 1, n):
            hl_win = hl[i - self.period + 1:i + 1]
            co_win = co[i - self.period + 1:i + 1]
            
            variance = np.nanmean(0.5 * hl_win - factor * co_win)
            gk[i] = np.sqrt(max(variance, 0)) * factor_annual
        
        return df.with_columns(
            pl.Series(name=f"gk_vol_{self.period}", values=gk)
        )
    
    test_params: ClassVar[list[dict]] = [
        {"period": 30, "annualize": True, "trading_periods": 252},
        {"period": 60, "annualize": True, "trading_periods": 252},
        {"period": 120, "annualize": True, "trading_periods": 252},
    ]

    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable, reproducible output."""
        return self.period * 5


@dataclass
@sf_component(name="stat/rogers_satchell_vol")
class RogersSatchellVolStat(Feature):
    """Rolling Rogers-Satchell Volatility.
    
    Handles drift (trending markets) unlike Parkinson/GK.
    
    RS = √(ln(H/C)*ln(H/O) + ln(L/C)*ln(L/O))
    
    Reference: Rogers & Satchell (1991)
    """
    
    period: int = 30
    annualize: bool = True
    trading_periods: int = 252
    
    requires = ["open", "high", "low", "close"]
    outputs = ["rs_vol_{period}"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        o = df["open"].to_numpy()
        h = df["high"].to_numpy()
        l = df["low"].to_numpy()
        c = df["close"].to_numpy()
        n = len(c)
        
        term1 = np.log(h / c) * np.log(h / o)
        term2 = np.log(l / c) * np.log(l / o)
        
        factor_annual = np.sqrt(self.trading_periods) if self.annualize else 1.0
        
        rs = np.full(n, np.nan)
        for i in range(self.period - 1, n):
            win1 = term1[i - self.period + 1:i + 1]
            win2 = term2[i - self.period + 1:i + 1]
            
            variance = np.nanmean(win1 + win2)
            rs[i] = np.sqrt(max(variance, 0)) * factor_annual
        
        return df.with_columns(
            pl.Series(name=f"rs_vol_{self.period}", values=rs)
        )
    
    test_params: ClassVar[list[dict]] = [
        {"period": 30, "annualize": True, "trading_periods": 252},
        {"period": 60, "annualize": True, "trading_periods": 252},
        {"period": 120, "annualize": True, "trading_periods": 252},
    ]

    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable, reproducible output."""
        return self.period * 5


@dataclass
@sf_component(name="stat/yang_zhang_vol")
class YangZhangVolStat(Feature):
    """Rolling Yang-Zhang Volatility.
    
    Best overall OHLC estimator. Handles drift AND opening jumps.
    Combines overnight, open-close, and Rogers-Satchell.
    
    YZ = √(σ²_overnight + k*σ²_open + (1-k)*σ²_rs)
    
    Reference: Yang & Zhang (2000)
    """
    
    period: int = 30
    annualize: bool = True
    trading_periods: int = 252
    
    requires = ["open", "high", "low", "close"]
    outputs = ["yz_vol_{period}"]
    
    def compute_pair(self, df: pl.DataFrame) -> pl.DataFrame:
        o = df["open"].to_numpy()
        h = df["high"].to_numpy()
        l = df["low"].to_numpy()
        c = df["close"].to_numpy()
        n = len(c)
        
        log_oc = np.log(o[1:] / c[:-1])
        log_oc = np.insert(log_oc, 0, np.nan)
        
        log_co = np.log(c / o)
        
        rs_term = np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)
        
        factor_annual = np.sqrt(self.trading_periods) if self.annualize else 1.0
        k = 0.34 / (1.34 + (self.period + 1) / (self.period - 1))
        
        yz = np.full(n, np.nan)
        for i in range(self.period, n):
            oc_win = log_oc[i - self.period + 1:i + 1]
            co_win = log_co[i - self.period + 1:i + 1]
            rs_win = rs_term[i - self.period + 1:i + 1]
            
            var_overnight = np.nanvar(oc_win, ddof=1)
            var_open = np.nanvar(co_win, ddof=1)
            var_rs = np.nanmean(rs_win)
            
            variance = var_overnight + k * var_open + (1 - k) * max(var_rs, 0)
            yz[i] = np.sqrt(max(variance, 0)) * factor_annual
        
        return df.with_columns(
            pl.Series(name=f"yz_vol_{self.period}", values=yz)
        )
    
    test_params: ClassVar[list[dict]] = [
        {"period": 30, "annualize": True, "trading_periods": 252},
        {"period": 60, "annualize": True, "trading_periods": 252},
        {"period": 120, "annualize": True, "trading_periods": 252},
    ]

    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable, reproducible output."""
        return self.period * 5

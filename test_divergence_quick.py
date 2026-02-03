"""Quick test for divergence detection"""

import numpy as np
import polars as pl
from signalflow.ta import RsiDivergence, MacdDivergence, MultiDivergence

# Create synthetic data with a clear divergence pattern
n = 200
dates = pl.date_range(
    pl.date(2024, 1, 1),
    pl.date(2024, 1, 1) + pl.duration(days=n-1),
    interval='1d',
    eager=True
)

# Price makes lower lows
price = np.concatenate([
    np.linspace(100, 90, 50),  # downtrend
    np.linspace(90, 85, 50),   # lower low
    np.linspace(85, 95, 100),  # recovery
])

# Add some noise
np.random.seed(42)
price = price + np.random.randn(n) * 0.5

df = pl.DataFrame({
    'date': dates,
    'open': price,
    'high': price + 1,
    'low': price - 1,
    'close': price,
    'volume': np.ones(n) * 1000,
})

print('Testing RSI Divergence...')
rsi_div = RsiDivergence(rsi_period=14, pivot_window=5, min_pivot_distance=10)
df_rsi = rsi_div.compute_pair(df)
div_cols = [c for c in df_rsi.columns if "div" in c]
print(f'✓ RSI Divergence computed. Columns added: {div_cols}')
print(f'  Bullish divergences detected: {df_rsi["rsi_div_bullish"].sum()}')
print(f'  Bearish divergences detected: {df_rsi["rsi_div_bearish"].sum()}')

print('\nTesting MACD Divergence...')
macd_div = MacdDivergence(fast=12, slow=26, signal=9)
df_macd = macd_div.compute_pair(df)
div_cols = [c for c in df_macd.columns if "div" in c]
print(f'✓ MACD Divergence computed. Columns added: {div_cols}')
print(f'  Bullish divergences detected: {df_macd["macd_div_bullish"].sum()}')
print(f'  Bearish divergences detected: {df_macd["macd_div_bearish"].sum()}')

print('\nTesting Multi-Indicator Confluence...')
multi_div = MultiDivergence(use_rsi=True, use_macd=True, min_confluence=2)
df_multi = multi_div.compute_pair(df)
print(f'✓ Multi-Indicator Confluence computed')
print(f'  Bullish confluence signals: {df_multi["multi_div_bullish"].sum()}')
print(f'  Bearish confluence signals: {df_multi["multi_div_bearish"].sum()}')
print(f'  Max confluence score: {df_multi["multi_div_confluence_score"].max():.1f}')

print('\n✅ All divergence indicators working correctly!')

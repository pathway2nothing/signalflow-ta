"""Shared Numba-accelerated kernels for technical indicators.

All kernels use @njit(cache=True) for persistent compilation.
Pure float64[:] arrays in/out — no Python objects inside.
"""

import numpy as np
from numba import njit


@njit(cache=True)
def rma_sma_init(values: np.ndarray, period: int) -> np.ndarray:
    """RMA (Wilder's smoothing) with SMA initialization.

    alpha = 1/period (unlike EMA which uses 2/(period+1)).
    First (period-1) values are NaN.
    """
    n = len(values)
    alpha = 1.0 / period
    rma = np.full(n, np.nan)

    if n < period:
        return rma

    # SMA init
    s = 0.0
    for j in range(period):
        s += values[j]
    rma[period - 1] = s / period

    # Wilder's smoothing
    for i in range(period, n):
        rma[i] = alpha * values[i] + (1.0 - alpha) * rma[i - 1]

    return rma


@njit(cache=True)
def ema_sma_init(values: np.ndarray, period: int) -> np.ndarray:
    """EMA with SMA initialization. Handles leading NaNs.

    alpha = 2/(period+1). First (period-1) values are NaN.
    """
    n = len(values)
    alpha = 2.0 / (period + 1)
    ema = np.full(n, np.nan)

    if n < period:
        return ema

    # Find first non-NaN
    first_valid = -1
    for i in range(n):
        if not np.isnan(values[i]):
            first_valid = i
            break
    if first_valid < 0:
        return ema

    if first_valid + period > n:
        return ema

    # SMA of first `period` valid values
    init_idx = first_valid + period - 1
    s = 0.0
    for j in range(first_valid, first_valid + period):
        s += values[j]
    ema[init_idx] = s / period

    # Standard EMA
    for i in range(init_idx + 1, n):
        if not np.isnan(values[i]):
            ema[i] = alpha * values[i] + (1.0 - alpha) * ema[i - 1]

    return ema


@njit(cache=True)
def normalize_zscore_nb(values: np.ndarray, window: int) -> np.ndarray:
    """Rolling z-score normalization (standard: mean/std)."""
    n = len(values)
    result = np.full(n, np.nan)

    for i in range(window - 1, n):
        # Collect valid values in window
        count = 0
        s = 0.0
        for j in range(i - window + 1, i + 1):
            v = values[j]
            if not np.isnan(v):
                s += v
                count += 1

        if count > 1:
            mean = s / count
            ss = 0.0
            for j in range(i - window + 1, i + 1):
                v = values[j]
                if not np.isnan(v):
                    ss += (v - mean) * (v - mean)
            std = np.sqrt(ss / (count - 1))
            if std > 1e-10:
                result[i] = (values[i] - mean) / std

    return result


@njit(cache=True)
def normalize_zscore_robust_nb(values: np.ndarray, window: int) -> np.ndarray:
    """Rolling robust z-score normalization (median/MAD)."""
    n = len(values)
    result = np.full(n, np.nan)
    scale = 1.4826

    for i in range(window - 1, n):
        # Collect valid values
        valid = np.empty(window, dtype=np.float64)
        count = 0
        for j in range(i - window + 1, i + 1):
            v = values[j]
            if not np.isnan(v):
                valid[count] = v
                count += 1

        if count > 1:
            valid_slice = valid[:count]
            median = np.median(valid_slice)
            deviations = np.empty(count, dtype=np.float64)
            for k in range(count):
                deviations[k] = np.abs(valid_slice[k] - median)
            mad = np.median(deviations)
            if mad > 1e-10:
                result[i] = (values[i] - median) / (scale * mad)

    return result


@njit(cache=True)
def rolling_min(arr: np.ndarray, period: int) -> np.ndarray:
    """Rolling minimum over a window."""
    n = len(arr)
    result = np.full(n, np.nan)
    for i in range(period - 1, n):
        mn = arr[i]
        for j in range(i - period + 1, i):
            if arr[j] < mn:
                mn = arr[j]
        result[i] = mn
    return result


@njit(cache=True)
def rolling_max(arr: np.ndarray, period: int) -> np.ndarray:
    """Rolling maximum over a window."""
    n = len(arr)
    result = np.full(n, np.nan)
    for i in range(period - 1, n):
        mx = arr[i]
        for j in range(i - period + 1, i):
            if arr[j] > mx:
                mx = arr[j]
        result[i] = mx
    return result


@njit(cache=True)
def sma_nb(arr: np.ndarray, period: int) -> np.ndarray:
    """Simple Moving Average."""
    n = len(arr)
    result = np.full(n, np.nan)
    for i in range(period - 1, n):
        s = 0.0
        cnt = 0
        for j in range(i - period + 1, i + 1):
            if not np.isnan(arr[j]):
                s += arr[j]
                cnt += 1
        if cnt > 0:
            result[i] = s / cnt
    return result


@njit(cache=True)
def adx_kernel(
    tr: np.ndarray,
    pdm: np.ndarray,
    ndm: np.ndarray,
    period: int,
) -> tuple:
    """Compute ADX, +DI, -DI from true range and directional movement.

    Returns (adx, dmp, dmn) arrays.
    """
    n = len(tr)
    alpha = 1.0 / period

    atr = np.full(n, np.nan)
    smooth_pdm = np.full(n, np.nan)
    smooth_ndm = np.full(n, np.nan)

    # SMA init
    s_tr = 0.0
    s_pdm = 0.0
    s_ndm = 0.0
    for j in range(period):
        s_tr += tr[j]
        s_pdm += pdm[j]
        s_ndm += ndm[j]
    atr[period - 1] = s_tr / period
    smooth_pdm[period - 1] = s_pdm / period
    smooth_ndm[period - 1] = s_ndm / period

    # RMA smoothing
    for i in range(period, n):
        atr[i] = alpha * tr[i] + (1.0 - alpha) * atr[i - 1]
        smooth_pdm[i] = alpha * pdm[i] + (1.0 - alpha) * smooth_pdm[i - 1]
        smooth_ndm[i] = alpha * ndm[i] + (1.0 - alpha) * smooth_ndm[i - 1]

    dmp = np.full(n, np.nan)
    dmn = np.full(n, np.nan)
    dx = np.full(n, np.nan)

    for i in range(period - 1, n):
        if atr[i] > 0:
            dmp[i] = 100.0 * smooth_pdm[i] / atr[i]
            dmn[i] = 100.0 * smooth_ndm[i] / atr[i]
            denom = dmp[i] + dmn[i] + 1e-10
            dx[i] = 100.0 * np.abs(dmp[i] - dmn[i]) / denom

    # ADX = RMA of DX
    adx = np.full(n, np.nan)
    start = 2 * period - 1
    if start < n:
        s_dx = 0.0
        cnt = 0
        for j in range(period, start + 1):
            if not np.isnan(dx[j]):
                s_dx += dx[j]
                cnt += 1
        if cnt > 0:
            adx[start] = s_dx / cnt
        for i in range(start + 1, n):
            if not np.isnan(dx[i]) and not np.isnan(adx[i - 1]):
                adx[i] = alpha * dx[i] + (1.0 - alpha) * adx[i - 1]

    return adx, dmp, dmn


@njit(cache=True)
def stoch_kernel(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    k_period: int,
    smooth_k: int,
    d_period: int,
) -> tuple:
    """Compute Stochastic %K and %D."""
    n = len(close)

    # Raw %K
    raw_k = np.full(n, np.nan)
    for i in range(k_period - 1, n):
        hh = high[i]
        ll = low[i]
        for j in range(i - k_period + 1, i):
            if high[j] > hh:
                hh = high[j]
            if low[j] < ll:
                ll = low[j]
        if hh != ll:
            raw_k[i] = 100.0 * (close[i] - ll) / (hh - ll)
        else:
            raw_k[i] = 50.0

    # Smoothed %K (SMA)
    stoch_k = np.full(n, np.nan)
    start_k = k_period + smooth_k - 2
    for i in range(start_k, n):
        s = 0.0
        cnt = 0
        for j in range(i - smooth_k + 1, i + 1):
            if not np.isnan(raw_k[j]):
                s += raw_k[j]
                cnt += 1
        if cnt > 0:
            stoch_k[i] = s / cnt

    # %D (SMA of %K)
    stoch_d = np.full(n, np.nan)
    start_d = start_k + d_period - 1
    for i in range(start_d, n):
        s = 0.0
        cnt = 0
        for j in range(i - d_period + 1, i + 1):
            if not np.isnan(stoch_k[j]):
                s += stoch_k[j]
                cnt += 1
        if cnt > 0:
            stoch_d[i] = s / cnt

    return stoch_k, stoch_d


@njit(cache=True)
def jma_kernel(
    source: np.ndarray,
    period: int,
    phase: float,
) -> np.ndarray:
    """Jurik Moving Average kernel."""
    n = len(source)
    jma = np.full(n, np.nan)
    volty = np.zeros(n)
    v_sum = np.zeros(n)

    warmup = min(period, n)
    if warmup == 0:
        return jma

    init_val = 0.0
    for j in range(warmup):
        init_val += source[j]
    init_val /= warmup

    jma[warmup - 1] = init_val
    ma1 = init_val
    uBand = init_val
    lBand = init_val
    det0 = 0.0
    det1 = 0.0
    ma2 = 0.0

    length = 0.5 * (period - 1)
    if phase < -100.0:
        pr = 0.5
    elif phase > 100.0:
        pr = 2.5
    else:
        pr = 1.5 + phase * 0.01

    length1 = max(np.log(np.sqrt(length)) / np.log(2.0) + 2.0, 0.0)
    pow1 = max(length1 - 2.0, 0.5)
    length2 = length1 * np.sqrt(length)
    bet = length2 / (length2 + 1.0)
    beta = 0.45 * (period - 1) / (0.45 * (period - 1) + 2.0)

    sum_length = 10

    for i in range(warmup, n):
        price = source[i]

        del1 = price - uBand
        del2 = price - lBand
        if abs(del1) != abs(del2):
            volty[i] = max(abs(del1), abs(del2))
        else:
            volty[i] = 0.0

        start_idx = max(i - sum_length, 0)
        v_sum[i] = v_sum[i - 1] + (volty[i] - volty[start_idx]) / sum_length

        avg_idx = max(i - 65, 0)
        avg_volty = 0.0
        for j in range(avg_idx, i + 1):
            avg_volty += v_sum[j]
        avg_volty /= (i - avg_idx + 1)

        d_volty = volty[i] / avg_volty if avg_volty > 0.0 else 0.0
        r_volty = max(1.0, min(length1 ** (1.0 / pow1), d_volty))

        pow2 = r_volty ** pow1
        kv = bet ** np.sqrt(pow2)
        if del1 > 0.0:
            uBand = price
        else:
            uBand = price - kv * del1
        if del2 < 0.0:
            lBand = price
        else:
            lBand = price - kv * del2

        power = r_volty ** pow1
        alpha = beta ** power

        ma1 = (1.0 - alpha) * price + alpha * ma1
        det0 = (price - ma1) * (1.0 - beta) + beta * det0
        ma2 = ma1 + pr * det0
        det1 = (ma2 - jma[i - 1]) * (1.0 - alpha) ** 2 + alpha ** 2 * det1
        jma[i] = jma[i - 1] + det1

    for i in range(period - 1):
        jma[i] = np.nan

    return jma


@njit(cache=True)
def kama_kernel(
    source: np.ndarray,
    period: int,
    fast_sc: float,
    slow_sc: float,
) -> np.ndarray:
    """KAMA kernel — efficiency ratio + adaptive smoothing."""
    n = len(source)
    kama = np.full(n, np.nan)

    if n < period:
        return kama

    # SMA init
    s = 0.0
    for j in range(period):
        s += source[j]
    kama[period - 1] = s / period

    for i in range(period, n):
        change = abs(source[i] - source[i - period])
        vol = 0.0
        for j in range(i - period, i):
            vol += abs(source[j + 1] - source[j])

        er = change / vol if vol > 0.0 else 0.0
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = sc * source[i] + (1.0 - sc) * kama[i - 1]

    return kama


@njit(cache=True)
def vidya_kernel(
    source: np.ndarray,
    pos: np.ndarray,
    neg: np.ndarray,
    period: int,
    alpha: float,
) -> np.ndarray:
    """VIDYA kernel — CMO-based adaptive smoothing."""
    n = len(source)
    vidya = np.full(n, np.nan)

    if n <= period:
        return vidya

    # SMA init
    s = 0.0
    for j in range(period + 1):
        s += source[j]
    vidya[period] = s / (period + 1)

    for i in range(period + 1, n):
        pos_sum = 0.0
        neg_sum = 0.0
        for j in range(i - period + 1, i + 1):
            pos_sum += pos[j]
            neg_sum += neg[j]

        total = pos_sum + neg_sum
        if total > 0.0:
            cmo = (pos_sum - neg_sum) / total
        else:
            cmo = 0.0
        abs_cmo = abs(cmo)
        vidya[i] = alpha * abs_cmo * source[i] + (1.0 - alpha * abs_cmo) * vidya[i - 1]

    return vidya


@njit(cache=True)
def mcginley_kernel(
    source: np.ndarray,
    period: int,
    k: float,
) -> np.ndarray:
    """McGinley Dynamic kernel."""
    n = len(source)
    md = np.full(n, np.nan)
    md[0] = source[0]

    for i in range(1, n):
        if md[i - 1] != 0.0:
            ratio = source[i] / md[i - 1]
            denom = k * period * (ratio ** 4)
            md[i] = md[i - 1] + (source[i] - md[i - 1]) / denom
        else:
            md[i] = source[i]

    return md


@njit(cache=True)
def frama_kernel(
    source: np.ndarray,
    period: int,
) -> np.ndarray:
    """FRAMA kernel — fractal adaptive moving average."""
    n = len(source)
    half = period // 2
    frama = np.full(n, np.nan)
    frama[period - 1] = source[period - 1]

    for i in range(period, n):
        # N1: first half
        max1 = source[i - period + 1]
        min1 = source[i - period + 1]
        for j in range(i - period + 2, i - half + 1):
            if source[j] > max1:
                max1 = source[j]
            if source[j] < min1:
                min1 = source[j]
        n1 = (max1 - min1) / half

        # N2: second half
        max2 = source[i - half + 1]
        min2 = source[i - half + 1]
        for j in range(i - half + 2, i + 1):
            if source[j] > max2:
                max2 = source[j]
            if source[j] < min2:
                min2 = source[j]
        n2 = (max2 - min2) / half

        # N3: full period
        max3 = source[i - period + 1]
        min3 = source[i - period + 1]
        for j in range(i - period + 2, i + 1):
            if source[j] > max3:
                max3 = source[j]
            if source[j] < min3:
                min3 = source[j]
        n3 = (max3 - min3) / period

        if n1 + n2 > 0.0 and n3 > 0.0:
            d = (np.log(n1 + n2) - np.log(n3)) / np.log(2.0)
        else:
            d = 1.0

        alpha = np.exp(-4.6 * (d - 1.0))
        if alpha < 0.01:
            alpha = 0.01
        elif alpha > 1.0:
            alpha = 1.0

        frama[i] = alpha * source[i] + (1.0 - alpha) * frama[i - 1]

    return frama


@njit(cache=True)
def aroon_kernel(
    high: np.ndarray,
    low: np.ndarray,
    period: int,
) -> tuple:
    """Aroon Up/Down kernel."""
    n = len(high)
    aroon_up = np.full(n, np.nan)
    aroon_dn = np.full(n, np.nan)

    for i in range(period, n):
        # Find periods since highest high (looking backward from most recent)
        best_high_idx = i
        for j in range(i - 1, i - period - 1, -1):
            if high[j] > high[best_high_idx]:
                best_high_idx = j
        periods_from_hh = i - best_high_idx

        # Find periods since lowest low
        best_low_idx = i
        for j in range(i - 1, i - period - 1, -1):
            if low[j] < low[best_low_idx]:
                best_low_idx = j
        periods_from_ll = i - best_low_idx

        aroon_up[i] = 100.0 * (period - periods_from_hh) / period
        aroon_dn[i] = 100.0 * (period - periods_from_ll) / period

    return aroon_up, aroon_dn


@njit(cache=True)
def cmo_kernel(
    gains: np.ndarray,
    losses: np.ndarray,
    period: int,
) -> np.ndarray:
    """CMO kernel with running sums."""
    n = len(gains)
    cmo = np.full(n, np.nan)

    if n < period:
        return cmo

    # Initial sums
    sum_g = 0.0
    sum_l = 0.0
    for j in range(period):
        sum_g += gains[j]
        sum_l += losses[j]
    total = sum_g + sum_l
    if total > 0.0:
        cmo[period - 1] = 100.0 * (sum_g - sum_l) / total

    # Sliding window
    for i in range(period, n):
        sum_g += gains[i] - gains[i - period]
        sum_l += losses[i] - losses[i - period]
        total = sum_g + sum_l
        if total > 0.0:
            cmo[i] = 100.0 * (sum_g - sum_l) / total

    return cmo

"""Shared Numba-accelerated kernels for technical indicators.

All kernels use @njit(cache=True) for persistent compilation.
Pure float64[:] arrays in/out - no Python objects inside.
"""

import numpy as np

from signalflow.ta._numba_compat import njit


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

    s = 0.0
    for j in range(period):
        s += values[j]
    rma[period - 1] = s / period

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

    first_valid = -1
    for i in range(n):
        if not np.isnan(values[i]):
            first_valid = i
            break
    if first_valid < 0:
        return ema

    if first_valid + period > n:
        return ema

    init_idx = first_valid + period - 1
    s = 0.0
    for j in range(first_valid, first_valid + period):
        s += values[j]
    ema[init_idx] = s / period

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

    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)

    atr = np.full(n, np.nan)
    smooth_pdm = np.full(n, np.nan)
    smooth_ndm = np.full(n, np.nan)

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

    if n < period:
        return jma

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
        avg_volty /= i - avg_idx + 1

        d_volty = volty[i] / avg_volty if avg_volty > 0.0 else 0.0
        r_volty = max(1.0, min(length1 ** (1.0 / pow1), d_volty))

        pow2 = r_volty**pow1
        kv = bet ** np.sqrt(pow2)
        uBand = price if del1 > 0.0 else price - kv * del1
        lBand = price if del2 < 0.0 else price - kv * del2

        power = r_volty**pow1
        alpha = beta**power

        ma1 = (1.0 - alpha) * price + alpha * ma1
        det0 = (price - ma1) * (1.0 - beta) + beta * det0
        ma2 = ma1 + pr * det0
        det1 = (ma2 - jma[i - 1]) * (1.0 - alpha) ** 2 + alpha**2 * det1
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
    """KAMA kernel - efficiency ratio + adaptive smoothing."""
    n = len(source)
    kama = np.full(n, np.nan)

    if n < period:
        return kama

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
    """VIDYA kernel - CMO-based adaptive smoothing."""
    n = len(source)
    vidya = np.full(n, np.nan)

    if n <= period:
        return vidya

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
        cmo = (pos_sum - neg_sum) / total if total > 0.0 else 0.0
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
            denom = k * period * (ratio**4)
            md[i] = md[i - 1] + (source[i] - md[i - 1]) / denom
        else:
            md[i] = source[i]

    return md


@njit(cache=True)
def frama_kernel(
    source: np.ndarray,
    period: int,
) -> np.ndarray:
    """FRAMA kernel - fractal adaptive moving average."""
    n = len(source)
    half = period // 2
    frama = np.full(n, np.nan)

    if n < period:
        return frama

    frama[period - 1] = source[period - 1]

    for i in range(period, n):
        max1 = source[i - period + 1]
        min1 = source[i - period + 1]
        for j in range(i - period + 2, i - half + 1):
            if source[j] > max1:
                max1 = source[j]
            if source[j] < min1:
                min1 = source[j]
        n1 = (max1 - min1) / half

        max2 = source[i - half + 1]
        min2 = source[i - half + 1]
        for j in range(i - half + 2, i + 1):
            if source[j] > max2:
                max2 = source[j]
            if source[j] < min2:
                min2 = source[j]
        n2 = (max2 - min2) / half

        max3 = source[i - period + 1]
        min3 = source[i - period + 1]
        for j in range(i - period + 2, i + 1):
            if source[j] > max3:
                max3 = source[j]
            if source[j] < min3:
                min3 = source[j]
        n3 = (max3 - min3) / period

        d = (np.log(n1 + n2) - np.log(n3)) / np.log(2.0) if n1 + n2 > 0.0 and n3 > 0.0 else 1.0

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
        best_high_idx = i
        for j in range(i - 1, i - period - 1, -1):
            if high[j] > high[best_high_idx]:
                best_high_idx = j
        periods_from_hh = i - best_high_idx

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

    sum_g = 0.0
    sum_l = 0.0
    for j in range(period):
        sum_g += gains[j]
        sum_l += losses[j]
    total = sum_g + sum_l
    if total > 0.0:
        cmo[period - 1] = 100.0 * (sum_g - sum_l) / total

    for i in range(period, n):
        sum_g += gains[i] - gains[i - period]
        sum_l += losses[i] - losses[i - period]
        total = sum_g + sum_l
        if total > 0.0:
            cmo[i] = 100.0 * (sum_g - sum_l) / total

    return cmo


@njit(cache=True)
def rolling_sum(arr: np.ndarray, period: int) -> np.ndarray:
    """O(n) rolling sum using cumulative approach."""
    n = len(arr)
    out = np.full(n, np.nan)
    if n < period:
        return out

    s = 0.0
    for j in range(period):
        s += arr[j]
    out[period - 1] = s

    for i in range(period, n):
        s += arr[i] - arr[i - period]
        out[i] = s

    return out


@njit(cache=True)
def rolling_std(arr: np.ndarray, period: int) -> np.ndarray:
    """O(n) rolling standard deviation (ddof=1) using Welford online."""
    n = len(arr)
    out = np.full(n, np.nan)
    if n < period:
        return out

    mean_val = 0.0
    for j in range(period):
        mean_val += arr[j]
    mean_val /= period
    ss = 0.0
    for j in range(period):
        d = arr[j] - mean_val
        ss += d * d
    out[period - 1] = np.sqrt(ss / (period - 1))

    for i in range(period, n):
        old_val = arr[i - period]
        new_val = arr[i]
        old_mean = mean_val
        mean_val += (new_val - old_val) / period
        ss += (new_val - old_val) * (new_val - mean_val + old_val - old_mean)
        if ss < 0.0:
            ss = 0.0
        out[i] = np.sqrt(ss / (period - 1))

    return out


@njit(cache=True)
def velocity_kernel(close: np.ndarray) -> np.ndarray:
    """Log returns: log(close[i] / close[i-1])."""
    n = len(close)
    vel = np.empty(n)
    vel[0] = 0.0
    for i in range(1, n):
        if close[i - 1] > 0.0:
            vel[i] = np.log(close[i] / close[i - 1])
        else:
            vel[i] = 0.0
    return vel


@njit(cache=True)
def rolling_mean_nan(arr: np.ndarray, period: int) -> np.ndarray:
    """Rolling mean that skips NaN values."""
    n = len(arr)
    out = np.full(n, np.nan)
    for i in range(period - 1, n):
        s = 0.0
        cnt = 0
        for j in range(i - period + 1, i + 1):
            v = arr[j]
            if not np.isnan(v):
                s += v
                cnt += 1
        if cnt > 0:
            out[i] = s / cnt
    return out


@njit(cache=True)
def wma_kernel(source: np.ndarray, period: int) -> np.ndarray:
    """Weighted Moving Average. Weight = position in window."""
    n = len(source)
    out = np.full(n, np.nan)
    denom = period * (period + 1) / 2.0

    for i in range(period - 1, n):
        s = 0.0
        for j in range(period):
            s += source[i - period + 1 + j] * (j + 1)
        out[i] = s / denom

    return out


@njit(cache=True)
def cci_kernel(
    tp: np.ndarray,
    period: int,
    constant: float,
) -> np.ndarray:
    """CCI: (TP - SMA(TP)) / (constant * MAD(TP))."""
    n = len(tp)
    cci = np.full(n, np.nan)

    for i in range(period - 1, n):
        s = 0.0
        for j in range(i - period + 1, i + 1):
            s += tp[j]
        sma = s / period

        mad = 0.0
        for j in range(i - period + 1, i + 1):
            mad += abs(tp[j] - sma)
        mad /= period

        if mad > 1e-10:
            cci[i] = (tp[i] - sma) / (constant * mad)

    return cci


@njit(cache=True)
def uo_kernel(
    bp: np.ndarray,
    tr: np.ndarray,
    fast: int,
    medium: int,
    slow: int,
    fw: float,
    mw: float,
    sw: float,
) -> np.ndarray:
    """Ultimate Oscillator kernel with 3 rolling sum periods."""
    n = len(bp)
    uo = np.full(n, np.nan)
    if n < slow:
        return uo

    for i in range(slow - 1, n):
        bp_fast = 0.0
        tr_fast = 0.0
        for j in range(i - fast + 1, i + 1):
            bp_fast += bp[j]
            tr_fast += tr[j]

        bp_med = 0.0
        tr_med = 0.0
        for j in range(i - medium + 1, i + 1):
            bp_med += bp[j]
            tr_med += tr[j]

        bp_slow = 0.0
        tr_slow = 0.0
        for j in range(i - slow + 1, i + 1):
            bp_slow += bp[j]
            tr_slow += tr[j]

        if tr_fast > 1e-10 and tr_med > 1e-10 and tr_slow > 1e-10:
            avg_fast = bp_fast / tr_fast
            avg_med = bp_med / tr_med
            avg_slow = bp_slow / tr_slow
            raw = fw * avg_fast + mw * avg_med + sw * avg_slow
            uo[i] = 100.0 * raw / (fw + mw + sw)

    return uo


@njit(cache=True)
def bollinger_kernel(
    close: np.ndarray,
    period: int,
    std_dev: float,
) -> tuple:
    """Bollinger Bands: SMA ± std_dev * rolling_std."""
    n = len(close)
    upper = np.full(n, np.nan)
    middle = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    bw = np.full(n, np.nan)
    bp = np.full(n, np.nan)

    for i in range(period - 1, n):
        s = 0.0
        for j in range(i - period + 1, i + 1):
            s += close[j]
        sma = s / period

        ss = 0.0
        for j in range(i - period + 1, i + 1):
            d = close[j] - sma
            ss += d * d
        std = np.sqrt(ss / (period - 1))

        middle[i] = sma
        upper[i] = sma + std_dev * std
        lower[i] = sma - std_dev * std

        band_range = upper[i] - lower[i]
        if band_range > 1e-10:
            bw[i] = band_range / sma * 100.0
            bp[i] = (close[i] - lower[i]) / band_range

    return upper, middle, lower, bw, bp


@njit(cache=True)
def keltner_kernel(
    close: np.ndarray,
    range_vals: np.ndarray,
    period: int,
    mult: float,
) -> tuple:
    """Keltner Channels: EMA(close) ± mult * EMA(range)."""
    n = len(close)
    upper = np.full(n, np.nan)
    middle = np.full(n, np.nan)
    lower = np.full(n, np.nan)

    if n < period:
        return upper, middle, lower

    alpha = 2.0 / (period + 1)

    sma_c = 0.0
    sma_r = 0.0
    for j in range(period):
        sma_c += close[j]
        sma_r += range_vals[j]
    ema_c = sma_c / period
    ema_r = sma_r / period

    middle[period - 1] = ema_c
    upper[period - 1] = ema_c + mult * ema_r
    lower[period - 1] = ema_c - mult * ema_r

    for i in range(period, n):
        ema_c = alpha * close[i] + (1 - alpha) * ema_c
        ema_r = alpha * range_vals[i] + (1 - alpha) * ema_r
        middle[i] = ema_c
        upper[i] = ema_c + mult * ema_r
        lower[i] = ema_c - mult * ema_r

    return upper, middle, lower


@njit(cache=True)
def psar_kernel(
    high: np.ndarray,
    low: np.ndarray,
    af_step: float,
    af_max: float,
) -> tuple:
    """Parabolic SAR kernel."""
    n = len(high)
    psar = np.full(n, np.nan)
    direction = np.full(n, np.nan)

    if n < 2:
        return psar, direction

    is_long = True
    af = af_step
    ep = high[0]
    sar = low[0]

    psar[0] = sar
    direction[0] = 1.0

    for i in range(1, n):
        prev_sar = sar

        sar = prev_sar + af * (ep - prev_sar)

        if is_long:
            sar = min(sar, low[i - 1], low[i - 2]) if i >= 2 else min(sar, low[i - 1])

            if low[i] < sar:
                is_long = False
                sar = ep
                ep = low[i]
                af = af_step
            else:
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + af_step, af_max)
        else:
            sar = max(sar, high[i - 1], high[i - 2]) if i >= 2 else max(sar, high[i - 1])

            if high[i] > sar:
                is_long = True
                sar = ep
                ep = high[i]
                af = af_step
            else:
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + af_step, af_max)

        psar[i] = sar
        direction[i] = 1.0 if is_long else -1.0

    return psar, direction


@njit(cache=True)
def supertrend_kernel(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int,
    multiplier: float,
) -> tuple:
    """Supertrend kernel: ATR-based trend following."""
    n = len(high)
    supertrend = np.full(n, np.nan)
    direction = np.full(n, np.nan)

    if n < period:
        return supertrend, direction

    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, max(hc, lc))

    atr = np.full(n, np.nan)
    s = 0.0
    for j in range(period):
        s += tr[j]
    atr[period - 1] = s / period
    alpha = 1.0 / period
    for i in range(period, n):
        atr[i] = alpha * tr[i] + (1 - alpha) * atr[i - 1]

    upper_band = np.empty(n)
    lower_band = np.empty(n)

    for i in range(n):
        hl2 = (high[i] + low[i]) / 2.0
        if np.isnan(atr[i]):
            upper_band[i] = hl2
            lower_band[i] = hl2
        else:
            upper_band[i] = hl2 + multiplier * atr[i]
            lower_band[i] = hl2 - multiplier * atr[i]

    dir_val = 1

    start = period - 1
    supertrend[start] = upper_band[start]
    direction[start] = 1.0

    for i in range(start + 1, n):
        if lower_band[i] > lower_band[i - 1] or close[i - 1] < lower_band[i - 1]:
            pass
        else:
            lower_band[i] = lower_band[i - 1]

        if upper_band[i] < upper_band[i - 1] or close[i - 1] > upper_band[i - 1]:
            pass
        else:
            upper_band[i] = upper_band[i - 1]

        if dir_val == 1:
            if close[i] < lower_band[i]:
                dir_val = -1
        else:
            if close[i] > upper_band[i]:
                dir_val = 1

        if dir_val == 1:
            supertrend[i] = lower_band[i]
        else:
            supertrend[i] = upper_band[i]

        direction[i] = float(dir_val)

    return supertrend, direction


@njit(cache=True)
def ichimoku_midprice(
    high: np.ndarray,
    low: np.ndarray,
    period: int,
) -> np.ndarray:
    """Rolling (max + min) / 2 for Ichimoku lines."""
    n = len(high)
    out = np.full(n, np.nan)
    for i in range(period - 1, n):
        hi = high[i]
        lo = low[i]
        for j in range(i - period + 1, i):
            if high[j] > hi:
                hi = high[j]
            if low[j] < lo:
                lo = low[j]
        out[i] = (hi + lo) / 2.0
    return out


@njit(cache=True)
def mass_index_kernel(
    hl_range: np.ndarray,
    fast: int,
    slow: int,
) -> np.ndarray:
    """Mass Index: sum of EMA(HL) / EMA(EMA(HL)) over slow window."""
    n = len(hl_range)
    massi = np.full(n, np.nan)

    alpha = 2.0 / (fast + 1)

    ema1 = np.empty(n)
    ema2 = np.empty(n)
    ema1[0] = hl_range[0]
    ema2[0] = hl_range[0]

    for i in range(1, n):
        ema1[i] = alpha * hl_range[i] + (1 - alpha) * ema1[i - 1]
        ema2[i] = alpha * ema1[i] + (1 - alpha) * ema2[i - 1]

    ratio = np.empty(n)
    for i in range(n):
        denom = ema2[i]
        if abs(denom) > 1e-10:
            ratio[i] = ema1[i] / denom
        else:
            ratio[i] = 1.0

    for i in range(slow - 1, n):
        s = 0.0
        for j in range(i - slow + 1, i + 1):
            s += ratio[j]
        massi[i] = s

    return massi


@njit(cache=True)
def ssf_kernel(
    source: np.ndarray,
    period: int,
    poles: int,
) -> np.ndarray:
    """Ehlers Super Smoother Filter (2-pole or 3-pole)."""
    n = len(source)
    out = np.full(n, np.nan)

    if poles == 2:
        a1 = np.exp(-1.414 * np.pi / period)
        b1 = 2.0 * a1 * np.cos(1.414 * np.pi / period)
        c2 = b1
        c3 = -a1 * a1
        c1 = 1.0 - c2 - c3

        if n >= 3:
            out[0] = source[0]
            out[1] = source[1]
            for i in range(2, n):
                out[i] = c1 * (source[i] + source[i - 1]) / 2.0 + c2 * out[i - 1] + c3 * out[i - 2]
    else:
        a1 = np.exp(-np.pi / period)
        b1 = 2.0 * a1 * np.cos(1.738 * np.pi / period)
        c1_coeff = a1 * a1
        c3 = -c1_coeff * a1
        c2 = c1_coeff + b1
        _c1_val = 1.0 - c2 - c3 - (-b1 * c1_coeff)
        c2_actual = b1 + c1_coeff
        c3_actual = -c1_coeff * (1.0 + b1)
        c4 = c1_coeff * c1_coeff
        c0 = 1.0 - c2_actual - c3_actual - c4

        if n >= 4:
            out[0] = source[0]
            out[1] = source[1]
            out[2] = source[2]
            for i in range(3, n):
                out[i] = (
                    c0 * (source[i] + source[i - 1]) / 2.0
                    + c2_actual * out[i - 1]
                    + c3_actual * out[i - 2]
                    + c4 * out[i - 3]
                )

    return out


@njit(cache=True)
def ulcer_index_kernel(
    close: np.ndarray,
    period: int,
) -> np.ndarray:
    """Ulcer Index: sqrt(mean(pct_drawdown^2)) over rolling window."""
    n = len(close)
    ui = np.full(n, np.nan)

    for i in range(period - 1, n):
        highest = close[i - period + 1]
        ss_dd = 0.0
        for j in range(i - period + 1, i + 1):
            if close[j] > highest:
                highest = close[j]
            pct_dd = 100.0 * (close[j] - highest) / highest
            ss_dd += pct_dd * pct_dd
        ui[i] = np.sqrt(ss_dd / period)

    return ui


@njit(cache=True)
def rvi_kernel(
    close: np.ndarray,
    period: int,
    std_period: int,
) -> np.ndarray:
    """RVI: directional volatility using rolling std + EMA."""
    n = len(close)
    rvi = np.full(n, np.nan)

    std_arr = np.full(n, np.nan)
    for i in range(std_period - 1, n):
        s = 0.0
        for j in range(i - std_period + 1, i + 1):
            s += close[j]
        mean_val = s / std_period
        ss = 0.0
        for j in range(i - std_period + 1, i + 1):
            d = close[j] - mean_val
            ss += d * d
        std_arr[i] = np.sqrt(ss / (std_period - 1))

    diff = np.empty(n)
    diff[0] = 0.0
    for i in range(1, n):
        diff[i] = close[i] - close[i - 1]

    up_std = np.empty(n)
    dn_std = np.empty(n)
    for i in range(n):
        sv = std_arr[i] if not np.isnan(std_arr[i]) else 0.0
        if diff[i] > 0:
            up_std[i] = sv
            dn_std[i] = 0.0
        else:
            up_std[i] = 0.0
            dn_std[i] = sv

    alpha = 2.0 / (period + 1)
    init_idx = std_period + period - 2

    if n <= init_idx:
        return rvi

    start_std = std_period - 1
    end_init = start_std + period
    if end_init > n:
        end_init = n

    up_sum = 0.0
    dn_sum = 0.0
    cnt = 0
    for j in range(start_std, end_init):
        up_sum += up_std[j]
        dn_sum += dn_std[j]
        cnt += 1

    up_ema = up_sum / cnt
    dn_ema = dn_sum / cnt

    total = up_ema + dn_ema
    if total > 1e-10:
        rvi[init_idx] = 100.0 * up_ema / total

    for i in range(init_idx + 1, n):
        up_ema = alpha * up_std[i] + (1 - alpha) * up_ema
        dn_ema = alpha * dn_std[i] + (1 - alpha) * dn_ema
        total = up_ema + dn_ema
        if total > 1e-10:
            rvi[i] = 100.0 * up_ema / total

    return rvi


@njit(cache=True)
def historical_vol_kernel(
    close: np.ndarray,
    period: int,
    annualize: int,
) -> np.ndarray:
    """Historical volatility: std(log_returns) * sqrt(annualize)."""
    n = len(close)
    hv = np.full(n, np.nan)

    log_ret = np.empty(n)
    log_ret[0] = 0.0
    for i in range(1, n):
        if close[i - 1] > 0:
            log_ret[i] = np.log(close[i] / close[i - 1])
        else:
            log_ret[i] = 0.0

    ann_factor = np.sqrt(float(annualize))

    for i in range(period - 1, n):
        s = 0.0
        for j in range(i - period + 1, i + 1):
            s += log_ret[j]
        mean_val = s / period
        ss = 0.0
        for j in range(i - period + 1, i + 1):
            d = log_ret[j] - mean_val
            ss += d * d
        hv[i] = np.sqrt(ss / (period - 1)) * ann_factor

    return hv

# ------------------------- Utilities -------------------------
def random_fill_nans(X):
    import numpy as np
    nan_mask = np.isnan(X)
    X[nan_mask] = np.random.rand(*X[nan_mask].shape)
    return X

def binary_autocorrelation_with_spacing(
    row, 
    positions, 
    max_lag=1000, 
    assume_sorted=True,
    normalize: str = "sum", 
    return_counts: bool = False
):
    """
    Fast autocorrelation over real genomic spacing.

    Parameters
    ----------
    row : 1D array (float)
        Values per position (NaN = missing). Works for binary or real-valued.
    positions : 1D array (int)
        Genomic coordinates for each column of `row`.
    max_lag : int
        Max genomic lag (inclusive).
    assume_sorted : bool
        If True, assumes `positions` are strictly non-decreasing.
    normalize : {"sum", "pearson"}
        "sum": autocorr[l] = sum_{pairs at lag l} (xc_i * xc_j) / sum(xc^2)
               (fast; comparable across lags and molecules).
        "pearson": autocorr[l] = (mean_{pairs at lag l} (xc_i * xc_j)) / (mean(xc^2))
                   i.e., an estimate of Pearson-like correlation at that lag.
    return_counts : bool
        If True, return (autocorr, lag_counts). Otherwise just autocorr.

    Returns
    -------
    autocorr : 1D array, shape (max_lag+1,)
        Normalized autocorrelation; autocorr[0] = 1.0.
        Lags with no valid pairs are NaN.
    (optionally) lag_counts : 1D array, shape (max_lag+1,)
        Number of pairs contributing to each lag.
    """
    import numpy as np

    # mask valid entries
    valid = ~np.isnan(row)
    if valid.sum() < 2:
        out = np.full(max_lag + 1, np.nan, dtype=np.float32)
        return (out, np.zeros_like(out, dtype=int)) if return_counts else out

    x = row[valid].astype(np.float64, copy=False)
    pos = positions[valid].astype(np.int64, copy=False)

    # sort by position if needed
    if not assume_sorted:
        order = np.argsort(pos, kind="mergesort")
        pos = pos[order]
        x = x[order]

    n = x.size
    x_mean = x.mean()
    xc = x - x_mean
    sum_xc2 = np.sum(xc * xc)
    if sum_xc2 == 0.0:
        out = np.full(max_lag + 1, np.nan, dtype=np.float32)
        return (out, np.zeros_like(out, dtype=int)) if return_counts else out

    lag_sums = np.zeros(max_lag + 1, dtype=np.float64)
    lag_counts = np.zeros(max_lag + 1, dtype=np.int64)

    # sliding window upper pointer
    j = 1
    for i in range(n - 1):
        # ensure j starts at least i+1 (important correctness)
        if j <= i:
            j = i + 1
        # advance j to include all positions within max_lag
        while j < n and pos[j] - pos[i] <= max_lag:
            j += 1
        # consider pairs (i, i+1...j-1)
        if j - i > 1:
            diffs = pos[i+1:j] - pos[i]                 # 1..max_lag
            contrib = xc[i] * xc[i+1:j]                 # contributions for each pair
            # accumulate weighted sums and counts per lag
            # bincount returns length >= max(diffs)+1; we request minlength
            bc_vals = np.bincount(diffs, weights=contrib, minlength=max_lag+1)[:max_lag+1]
            bc_counts = np.bincount(diffs, minlength=max_lag+1)[:max_lag+1]
            lag_sums += bc_vals
            lag_counts += bc_counts

    autocorr = np.full(max_lag + 1, np.nan, dtype=np.float64)
    nz = lag_counts > 0

    if normalize == "sum":
        # matches your original: sum_pairs / sum_xc2
        autocorr[nz] = lag_sums[nz] / sum_xc2
    elif normalize == "pearson":
        # (mean of pairwise products) / (mean(xc^2)) -> more like correlation coeff
        mean_pair = lag_sums[nz] / lag_counts[nz]
        mean_var = sum_xc2 / n
        autocorr[nz] = mean_pair / mean_var
    else:
        raise ValueError("normalize must be 'sum' or 'pearson'")

    # define lag 0 as exactly 1.0 (by definition)
    autocorr[0] = 1.0

    if return_counts:
        return autocorr.astype(np.float32, copy=False), lag_counts
    return autocorr.astype(np.float32, copy=False)


import numpy as np
from numpy.fft import rfft, rfftfreq

# optionally use scipy for find_peaks (more robust)
try:
    from scipy.signal import find_peaks
    _have_scipy = True
except Exception:
    _have_scipy = False

# ---------- helpers ----------
def weighted_mean_autocorr(ac_matrix, counts_matrix, min_count=20):
    """
    Weighted mean across molecules: sum(ac * counts) / sum(counts) per lag.
    Mask lags with total counts < min_count (set NaN).
    """
    counts_total = counts_matrix.sum(axis=0)
    # replace NaNs in ac_matrix with 0 for weighted sum
    filled = np.where(np.isfinite(ac_matrix), ac_matrix, 0.0)
    s = (filled * counts_matrix).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_ac = np.where(counts_total > 0, s / counts_total, np.nan)
    # mask low support
    mean_ac[counts_total < min_count] = np.nan
    return mean_ac, counts_total

def psd_from_autocorr(mean_ac, lags, pad_factor=4):
    n = len(mean_ac)
    pad_n = int(max(2**10, pad_factor * n))  # pad to at least some min to stabilize FFT res
    ac_padded = np.zeros(pad_n, dtype=np.float64)
    ac_padded[:n] = np.where(np.isfinite(mean_ac), mean_ac, 0.0)
    A = rfft(ac_padded)
    power = np.abs(A) ** 2
    df = (lags[1] - lags[0]) if len(lags) > 1 else 1.0
    freqs = rfftfreq(pad_n, d=df)
    return freqs, power

def find_peak_in_nrl_band(freqs, power, nrl_search_bp=(120,260), prominence_frac=0.05):
    fmin = 1.0 / nrl_search_bp[1]
    fmax = 1.0 / nrl_search_bp[0]
    band_mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(band_mask):
        return None, None
    freqs_band = freqs[band_mask]
    power_band = power[band_mask]
    if _have_scipy:
        prom = max(np.max(power_band) * prominence_frac, 1e-12)
        peaks, props = find_peaks(power_band, prominence=prom)
        if peaks.size:
            rel = peaks[np.argmax(power_band[peaks])]
        else:
            rel = int(np.argmax(power_band))
    else:
        rel = int(np.argmax(power_band))
    band_indices = np.nonzero(band_mask)[0]
    idx = band_indices[rel]
    return freqs[idx], idx

def fwhm_freq_to_bp(freqs, power, peak_idx):
    # find half power
    pk = power[peak_idx]
    half = pk / 2.0
    # move left
    left = peak_idx
    while left > 0 and power[left] > half:
        left -= 1
    # left interpolation
    if left == peak_idx:
        left_f = freqs[peak_idx]
    else:
        x0, x1 = freqs[left], freqs[left+1]
        y0, y1 = power[left], power[left+1]
        left_f = x0 if y1 == y0 else x0 + (half - y0)*(x1-x0)/(y1-y0)
    # move right
    right = peak_idx
    while right < len(power)-1 and power[right] > half:
        right += 1
    if right == peak_idx:
        right_f = freqs[peak_idx]
    else:
        x0, x1 = freqs[right-1], freqs[right]
        y0, y1 = power[right-1], power[right]
        right_f = x1 if y1 == y0 else x0 + (half - y0)*(x1-x0)/(y1-y0)
    # convert to bp approximating delta_NRL = |1/left_f - 1/right_f|
    left_NRL = 1.0 / right_f if right_f > 0 else np.nan
    right_NRL = 1.0 / left_f if left_f > 0 else np.nan
    fwhm_bp = abs(left_NRL - right_NRL)
    return fwhm_bp, left_f, right_f

def estimate_snr(power, peak_idx, exclude_bins=5):
    pk = power[peak_idx]
    mask = np.ones_like(power, dtype=bool)
    lo = max(0, peak_idx-exclude_bins)
    hi = min(len(power), peak_idx+exclude_bins+1)
    mask[lo:hi] = False
    bg = power[mask]
    bg_med = np.median(bg) if bg.size else np.median(power)
    return pk / (bg_med if bg_med > 0 else np.finfo(float).eps), pk, bg_med

def sample_autocorr_at_harmonics(mean_ac, lags, nrl_bp, max_harmonics=6):
    sample_lags = []
    heights = []
    for m in range(1, max_harmonics+1):
        target = m * nrl_bp
        # stop if beyond observed lag range
        if target > lags[-1]:
            break
        idx = np.argmin(np.abs(lags - target))
        h = mean_ac[idx]
        if not np.isfinite(h):
            break
        sample_lags.append(lags[idx])
        heights.append(h)
    return np.array(sample_lags), np.array(heights)

def fit_exponential_envelope(sample_lags, heights, counts=None):
    # heights ~ A * exp(-lag / xi)
    mask = (heights > 0) & np.isfinite(heights)
    if mask.sum() < 2:
        return np.nan, np.nan, np.nan, np.nan
    x = sample_lags[mask].astype(float)
    y = np.log(heights[mask].astype(float))
    if counts is None:
        w = np.ones_like(y)
    else:
        w = np.asarray(counts[mask], dtype=float)
        w = w / (np.max(w) if np.max(w)>0 else 1.0)
    # weighted linear regression y = b0 + b1 * x
    X = np.vstack([np.ones_like(x), x]).T
    W = np.diag(w)
    XtWX = X.T.dot(W).dot(X)
    XtWy = X.T.dot(W).dot(y)
    try:
        b = np.linalg.solve(XtWX, XtWy)
    except np.linalg.LinAlgError:
        return np.nan, np.nan, np.nan, np.nan
    b0, b1 = b
    A = np.exp(b0)
    xi = -1.0 / b1 if b1 < 0 else np.nan
    # R^2
    y_pred = X.dot(b)
    ss_res = np.sum(w * (y - y_pred)**2)
    ss_tot = np.sum(w * (y - np.average(y, weights=w))**2)
    r2 = 1.0 - ss_res/ss_tot if ss_tot != 0 else np.nan
    return xi, A, b1, r2

# ---------- main analysis per site_type ----------
def analyze_autocorr_matrix(autocorr_matrix, counts_matrix, lags,
                            nrl_search_bp=(120,260), pad_factor=4,
                            min_count=20, max_harmonics=6):
    """
    Return dict: nrl_bp, peak_power, fwhm_bp, snr, xi, envelope points, freqs, power, mean_ac
    """
    mean_ac, counts_total = weighted_mean_autocorr(autocorr_matrix, counts_matrix, min_count=min_count)
    freqs, power = psd_from_autocorr(mean_ac, lags, pad_factor=pad_factor)
    f0, peak_idx = find_peak_in_nrl_band(freqs, power, nrl_search_bp=nrl_search_bp)
    if f0 is None:
        return {"error":"no_peak_found", "mean_ac":mean_ac, "counts":counts_total}
    nrl_bp = 1.0 / f0
    fwhm_bp, left_f, right_f = fwhm_freq_to_bp(freqs, power, peak_idx)
    snr, peak_power, bg = estimate_snr(power, peak_idx)
    sample_lags, heights = sample_autocorr_at_harmonics(mean_ac, lags, nrl_bp, max_harmonics=max_harmonics)
    xi, A, slope, r2 = fit_exponential_envelope(sample_lags, heights) if heights.size else (np.nan,)*4

    return dict(
        nrl_bp = nrl_bp,
        f0 = f0,
        peak_power = peak_power,
        fwhm_bp = fwhm_bp,
        snr = snr,
        bg_median = bg,
        envelope_sample_lags = sample_lags,
        envelope_heights = heights,
        xi = xi,
        xi_A = A,
        xi_slope = slope,
        xi_r2 = r2,
        freqs = freqs,
        power = power,
        mean_ac = mean_ac,
        counts = counts_total
    )

# ---------- bootstrap wrapper ----------
def bootstrap_periodicity(autocorr_matrix, counts_matrix, lags, n_boot=200, **kwargs):
    rng = np.random.default_rng()
    metrics = []
    n = autocorr_matrix.shape[0]
    for _ in range(n_boot):
        sample_idx = rng.integers(0, n, size=n)
        res = analyze_autocorr_matrix(autocorr_matrix[sample_idx], counts_matrix[sample_idx], lags, **kwargs)
        metrics.append(res)
    # extract key fields robustly
    nrls = np.array([m.get("nrl_bp", np.nan) for m in metrics])
    xis  = np.array([m.get("xi", np.nan) for m in metrics])
    return {"nrl_boot":nrls, "xi_boot":xis, "metrics":metrics}
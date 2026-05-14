"""
ls_periodicity.py — Lomb-Scargle periodogram utilities for autocorrelation-based
nucleosome analysis.

Uses scipy.signal.lombscargle which operates on finite (lag, autocorr) pairs and
ignores NaN lags. This is the correct tool when comparing masked vs unmasked
conditions where the number of valid lags differs.

Key difference from FFT
-----------------------
FFT: zero-pads NaN lags → gaps alter spectral shape and SNR.
LS:  drops NaN lags entirely → spectrum reflects only observed data.

Usage
-----
    from tools.ls_periodicity import analyze_ls_periodicity, analyze_fft_periodicity

    result = analyze_ls_periodicity(lags, ac_values)
    # result is None on failure, else dict with ls_nrl_bp, ls_snr, ls_fwhm_bp, …
"""

import numpy as np
from scipy.signal import lombscargle, find_peaks
from scipy.fft import rfft, rfftfreq

NRL_SEARCH_BP = (120, 260)
MIN_FINITE_LAGS = 10
LS_PERIOD_RANGE_BP = (80, 400)
FFT_SMOOTHING_WINDOW_BP = 25
MAX_HARMONICS = 8


def ls_periodogram_from_autocorr(
    lags,
    ac_vals,
    period_range_bp: tuple[float, float] = LS_PERIOD_RANGE_BP,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[None, None, None]:
    """
    Compute Lomb-Scargle periodogram from (lag, autocorr) pairs.

    Returns (freqs, power_norm, power_raw) or (None, None, None) on failure.
    """
    lags = np.asarray(lags, dtype=float)
    ac_vals = np.asarray(ac_vals, dtype=float)

    nonzero = lags > 0
    lags, ac_vals = lags[nonzero], ac_vals[nonzero]

    finite = np.isfinite(ac_vals)
    if np.sum(finite) < MIN_FINITE_LAGS:
        return None, None, None
    lags_f, ac_f = lags[finite], ac_vals[finite]
    ac_f = ac_f - np.mean(ac_f)

    period_min, period_max = period_range_bp
    periods = np.arange(period_max, period_min - 1, -1, dtype=float)
    freqs = 1.0 / periods
    if freqs.size == 0:
        return None, None, None

    omega = 2.0 * np.pi * freqs
    power_norm = lombscargle(lags_f, ac_f, omega, normalize=True)
    power_raw  = lombscargle(lags_f, ac_f, omega, normalize=False)
    return freqs, power_norm, power_raw


def find_peak_ls(
    freqs: np.ndarray,
    power: np.ndarray,
    nrl_search_bp: tuple[float, float] = NRL_SEARCH_BP,
    prominence_frac: float = 0.05,
) -> tuple[float, int] | tuple[None, None]:
    """Find the dominant LS peak in the NRL search band."""
    fmin = 1.0 / nrl_search_bp[1]
    fmax = 1.0 / nrl_search_bp[0]
    band_mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(band_mask):
        return None, None

    power_band = power[band_mask]
    prom = max(np.max(power_band) * prominence_frac, 1e-12)
    peaks, _ = find_peaks(power_band, prominence=prom)
    rel = peaks[np.argmax(power_band[peaks])] if peaks.size else int(np.argmax(power_band))
    band_indices = np.nonzero(band_mask)[0]
    idx = band_indices[rel]
    return freqs[idx], idx


def fwhm_ls(freqs: np.ndarray, power: np.ndarray, peak_idx: int) -> float:
    """Estimate FWHM in base pairs for a spectral peak."""
    half = power[peak_idx] / 2.0

    left = peak_idx
    while left > 0 and power[left] > half:
        left -= 1
    left_f = (freqs[left] if left == peak_idx else
              freqs[left] + (half - power[left]) * (freqs[left + 1] - freqs[left]) /
              (power[left + 1] - power[left]))

    right = peak_idx
    while right < len(power) - 1 and power[right] > half:
        right += 1
    right_f = (freqs[right] if right == peak_idx else
               freqs[right - 1] + (half - power[right - 1]) * (freqs[right] - freqs[right - 1]) /
               (power[right] - power[right - 1]))

    left_nrl  = 1.0 / right_f if right_f > 0 else np.nan
    right_nrl = 1.0 / left_f  if left_f  > 0 else np.nan
    return abs(left_nrl - right_nrl)


def snr_ls(
    power: np.ndarray,
    peak_idx: int,
    exclude_bins: int = 5,
) -> tuple[float, float, float]:
    """Estimate SNR around a spectral peak. Returns (snr, peak_power, bg_median)."""
    pk = power[peak_idx]
    mask = np.ones_like(power, dtype=bool)
    mask[max(0, peak_idx - exclude_bins): min(len(power), peak_idx + exclude_bins + 1)] = False
    bg = power[mask]
    bg_med = np.median(bg) if bg.size else np.median(power)
    snr = pk / (bg_med if bg_med > 0 else np.finfo(float).eps)
    return snr, pk, bg_med


def analyze_ls_periodicity(
    lags,
    ac_vals,
    nrl_search_bp: tuple[float, float] = NRL_SEARCH_BP,
    period_range_bp: tuple[float, float] = LS_PERIOD_RANGE_BP,
) -> dict | None:
    """
    Full LS periodicity analysis on a (lag, autocorr) sequence.

    Returns dict with keys: ls_nrl_bp, ls_snr, ls_peak_power, ls_peak_power_raw,
    ls_fwhm_bp, ls_freqs, ls_power, ls_power_raw — or None on failure.
    """
    freqs, power, power_raw = ls_periodogram_from_autocorr(lags, ac_vals, period_range_bp)
    if freqs is None:
        return None

    f0, peak_idx = find_peak_ls(freqs, power, nrl_search_bp)
    if f0 is None:
        return None

    return {
        "ls_nrl_bp":        1.0 / f0,
        "ls_snr":           snr_ls(power, peak_idx)[0],
        "ls_peak_power":    float(power[peak_idx]),
        "ls_peak_power_raw": float(power_raw[peak_idx]),
        "ls_fwhm_bp":       fwhm_ls(freqs, power, peak_idx),
        "ls_freqs":         freqs,
        "ls_power":         power,
        "ls_power_raw":     power_raw,
    }


# ---------------------------------------------------------------------------
# FFT helpers
# ---------------------------------------------------------------------------

def rolling_mean_nan(x, window: int = FFT_SMOOTHING_WINDOW_BP) -> np.ndarray:
    """Centered rolling mean that ignores NaNs."""
    x = np.asarray(x, dtype=float)
    if window <= 1:
        return x.copy()
    out = np.full_like(x, np.nan)
    half = window // 2
    for i in range(len(x)):
        vals = x[max(0, i - half): min(len(x), i + half + 1)]
        finite = np.isfinite(vals)
        if np.any(finite):
            out[i] = np.mean(vals[finite])
    return out


def psd_from_autocorr(mean_ac, lags, pad_factor: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """FFT power spectrum from an autocorrelation curve, zero-filling NaN lags."""
    mean_ac = np.asarray(mean_ac, dtype=float)
    lags = np.asarray(lags, dtype=float)
    n = len(mean_ac)
    pad_n = int(max(2**10, pad_factor * n))
    ac_padded = np.zeros(pad_n)
    ac_padded[:n] = np.where(np.isfinite(mean_ac), mean_ac, 0.0)
    spectrum = rfft(ac_padded)
    power = np.abs(spectrum) ** 2
    df = (lags[1] - lags[0]) if len(lags) > 1 else 1.0
    return rfftfreq(pad_n, d=df), power


def analyze_fft_periodicity(
    lags,
    ac_vals,
    smoothing_window: int = FFT_SMOOTHING_WINDOW_BP,
    nrl_search_bp: tuple[float, float] = NRL_SEARCH_BP,
) -> dict | None:
    """FFT periodicity analysis on a smoothed autocorrelation curve."""
    lags = np.asarray(lags, dtype=float)
    ac_vals = np.asarray(ac_vals, dtype=float)
    if lags.size == 0 or np.sum(np.isfinite(ac_vals)) < MIN_FINITE_LAGS:
        return None

    smoothed = rolling_mean_nan(ac_vals, window=smoothing_window)
    freqs, power = psd_from_autocorr(smoothed, lags)

    fmin = 1.0 / nrl_search_bp[1]
    fmax = 1.0 / nrl_search_bp[0]
    band_mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(band_mask):
        return None

    power_band = power[band_mask]
    peaks, _ = find_peaks(power_band, prominence=max(np.max(power_band) * 0.05, 1e-12))
    rel = peaks[np.argmax(power_band[peaks])] if peaks.size else int(np.argmax(power_band))
    global_idx = np.nonzero(band_mask)[0][rel]
    f0 = freqs[global_idx]
    nrl_bp = 1.0 / f0

    half = power[global_idx] / 2.0
    left = global_idx
    while left > 0 and power[left] > half:
        left -= 1
    right = global_idx
    while right < len(power) - 1 and power[right] > half:
        right += 1
    left_bp  = 1.0 / freqs[right] if freqs[right] > 0 else np.nan
    right_bp = 1.0 / freqs[left]  if freqs[left]  > 0 else np.nan
    fwhm_bp = abs(left_bp - right_bp)

    mask = np.ones_like(power, dtype=bool)
    mask[max(0, global_idx - 5): min(len(power), global_idx + 6)] = False
    bg_med = np.median(power[mask]) if mask.any() else np.median(power)
    snr = power[global_idx] / (bg_med if bg_med > 0 else np.finfo(float).eps)

    return {
        "nrl_bp": float(nrl_bp),
        "snr": float(snr),
        "peak_power": float(power[global_idx]),
        "fwhm_bp": float(fwhm_bp),
        "fft_freqs": freqs,
        "fft_power": power,
        "fft_smoothing_window_bp": smoothing_window,
    }

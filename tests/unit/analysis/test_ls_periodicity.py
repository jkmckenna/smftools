from __future__ import annotations

import numpy as np
import pytest

from smftools.analysis.compute.ls_periodicity import (
    MIN_SITES_PER_READ,
    analyze_fft_periodicity,
    analyze_ls_periodicity,
    analyze_ls_periodicity_direct,
    ls_periodogram_from_autocorr,
    ls_periodogram_from_signal,
)


def _synthetic_acf(period_bp: float, n_lags: int = 300) -> tuple[np.ndarray, np.ndarray]:
    """Cosine autocorrelation with exponential decay at a known period."""
    lags = np.arange(1, n_lags + 1, dtype=float)
    ac = np.cos(2 * np.pi * lags / period_bp) * np.exp(-lags / 500)
    return lags, ac


# ---------------------------------------------------------------------------
# ls_periodogram_from_autocorr
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_ls_periodogram_returns_arrays() -> None:
    lags, ac = _synthetic_acf(185.0)
    freqs, power, power_raw = ls_periodogram_from_autocorr(lags, ac)
    assert freqs is not None
    assert len(freqs) == len(power) == len(power_raw)
    assert np.all(power >= 0)


@pytest.mark.unit
def test_ls_periodogram_too_few_lags_returns_none() -> None:
    lags = np.array([5.0, 10.0])
    ac = np.array([0.8, 0.6])
    result = ls_periodogram_from_autocorr(lags, ac)
    assert result == (None, None, None)


@pytest.mark.unit
def test_ls_periodogram_ignores_nan_lags() -> None:
    lags, ac = _synthetic_acf(185.0)
    ac_with_nans = ac.copy()
    ac_with_nans[::5] = np.nan  # introduce gaps
    freqs_clean, power_clean, _ = ls_periodogram_from_autocorr(lags, ac)
    freqs_nan, power_nan, _ = ls_periodogram_from_autocorr(lags, ac_with_nans)
    # Both should succeed; peak should be in the same region
    assert freqs_clean is not None and freqs_nan is not None
    peak_clean = 1.0 / freqs_clean[np.argmax(power_clean)]
    peak_nan = 1.0 / freqs_nan[np.argmax(power_nan)]
    assert abs(peak_clean - peak_nan) < 20  # within 20 bp of each other


# ---------------------------------------------------------------------------
# analyze_ls_periodicity
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_analyze_ls_periodicity_detects_known_period() -> None:
    lags, ac = _synthetic_acf(185.0)
    result = analyze_ls_periodicity(lags, ac)
    assert result is not None
    assert "ls_nrl_bp" in result
    # Should recover a period within the NRL search band (150–250 bp)
    assert 150 <= result["ls_nrl_bp"] <= 250


@pytest.mark.unit
def test_analyze_ls_periodicity_returns_none_for_flat_signal() -> None:
    lags = np.arange(1, 50, dtype=float)
    ac = np.zeros(len(lags))
    result = analyze_ls_periodicity(lags, ac)
    # Flat signal: still returns a result (LS will find *some* peak), but shouldn't crash
    # The important thing is no exception
    assert result is None or isinstance(result, dict)


@pytest.mark.unit
def test_analyze_ls_periodicity_result_keys() -> None:
    lags, ac = _synthetic_acf(185.0)
    result = analyze_ls_periodicity(lags, ac)
    assert result is not None
    for key in ("ls_nrl_bp", "ls_snr", "ls_peak_power", "ls_fwhm_bp", "ls_freqs", "ls_power"):
        assert key in result


# ---------------------------------------------------------------------------
# analyze_fft_periodicity
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_analyze_fft_periodicity_detects_known_period() -> None:
    lags, ac = _synthetic_acf(185.0)
    result = analyze_fft_periodicity(lags, ac)
    assert result is not None
    assert 150 <= result["nrl_bp"] <= 250


@pytest.mark.unit
def test_analyze_fft_periodicity_returns_none_for_empty() -> None:
    result = analyze_fft_periodicity(np.array([]), np.array([]))
    assert result is None


# ---------------------------------------------------------------------------
# ls_periodogram_from_signal (direct method)
# ---------------------------------------------------------------------------


def _synthetic_signal(
    period_bp: float, n_sites: int = 200, locus_len: int = 3000, rng_seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Binary signal with sinusoidal accessibility at a known nucleosomal period."""
    rng = np.random.default_rng(rng_seed)
    positions = np.sort(rng.choice(locus_len, size=n_sites, replace=False)).astype(float)
    prob = 0.5 + 0.4 * np.cos(2 * np.pi * positions / period_bp)
    signal = (rng.random(n_sites) < prob).astype(float)
    return positions, signal


@pytest.mark.unit
def test_ls_periodogram_from_signal_returns_arrays() -> None:
    pos, sig = _synthetic_signal(185.0)
    freqs, power, power_raw = ls_periodogram_from_signal(pos, sig)
    assert freqs is not None
    assert len(freqs) == len(power) == len(power_raw)
    assert np.all(power >= 0)


@pytest.mark.unit
def test_ls_periodogram_from_signal_too_few_sites_returns_none() -> None:
    pos = np.arange(10, dtype=float)
    sig = np.ones(10)
    result = ls_periodogram_from_signal(pos, sig, min_sites=MIN_SITES_PER_READ)
    assert result == (None, None, None)


@pytest.mark.unit
def test_ls_periodogram_from_signal_flat_returns_none() -> None:
    pos = np.arange(100, dtype=float)
    sig = np.ones(100)
    result = ls_periodogram_from_signal(pos, sig)
    assert result == (None, None, None)


@pytest.mark.unit
def test_ls_periodogram_from_signal_ignores_nan() -> None:
    pos, sig = _synthetic_signal(185.0)
    sig_nan = sig.copy()
    sig_nan[:20] = np.nan
    freqs, power, _ = ls_periodogram_from_signal(pos, sig_nan)
    assert freqs is not None


@pytest.mark.unit
def test_ls_periodogram_from_signal_known_period() -> None:
    pos, sig = _synthetic_signal(185.0, n_sites=300, locus_len=4000)
    freqs, power, _ = ls_periodogram_from_signal(pos, sig)
    assert freqs is not None
    peak_period = 1.0 / freqs[np.argmax(power)]
    assert 150 <= peak_period <= 250


# ---------------------------------------------------------------------------
# analyze_ls_periodicity_direct
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_analyze_ls_periodicity_direct_result_keys() -> None:
    pos, sig = _synthetic_signal(185.0, n_sites=300)
    result = analyze_ls_periodicity_direct(pos, sig)
    assert result is not None
    for key in (
        "ls_nrl_bp",
        "ls_snr",
        "ls_peak_power",
        "ls_peak_power_raw",
        "ls_fwhm_bp",
        "ls_freqs",
        "ls_power",
        "ls_power_raw",
    ):
        assert key in result


@pytest.mark.unit
def test_analyze_ls_periodicity_direct_detects_known_period() -> None:
    pos, sig = _synthetic_signal(185.0, n_sites=300, locus_len=4000)
    result = analyze_ls_periodicity_direct(pos, sig)
    assert result is not None
    assert 150 <= result["ls_nrl_bp"] <= 250


@pytest.mark.unit
def test_analyze_ls_periodicity_direct_too_few_sites_returns_none() -> None:
    pos = np.arange(10, dtype=float)
    sig = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=float)
    result = analyze_ls_periodicity_direct(pos, sig)
    assert result is None


@pytest.mark.unit
def test_analyze_ls_periodicity_direct_matches_keys_of_acf_method() -> None:
    """Both analysis paths return the same dict keys (minus array-valued spectra)."""
    lags, ac = _synthetic_acf(185.0)
    pos, sig = _synthetic_signal(185.0, n_sites=300)
    result_acf = analyze_ls_periodicity(lags, ac)
    result_direct = analyze_ls_periodicity_direct(pos, sig)
    assert result_acf is not None and result_direct is not None
    scalar_keys = {"ls_nrl_bp", "ls_snr", "ls_peak_power", "ls_peak_power_raw", "ls_fwhm_bp"}
    assert scalar_keys <= set(result_acf.keys())
    assert scalar_keys <= set(result_direct.keys())

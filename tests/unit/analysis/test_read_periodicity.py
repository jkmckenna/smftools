"""RPG-02: per-read periodograms over a region (pure functions)."""

import numpy as np
import pytest

from smftools.analysis.compute.ls_periodicity import analyze_ls_periodicity_direct
from smftools.analysis.compute.read_periodicity import (
    LOW_COVERAGE,
    NO_SIGNAL,
    OK,
    REGION_TOO_SHORT,
    TOO_FEW_SITES,
    period_grid,
    read_periodograms,
)


def _reads(n_reads=6, length=1600, period=190.0, every=3, seed=0):
    """Binary calls at every ``every``-th position, modified with a periodic probability."""
    rng = np.random.default_rng(seed)
    positions = np.arange(length)
    sites = positions % every == 0
    probability = 0.5 + 0.45 * np.sin(2 * np.pi * positions / period)
    calls = (rng.random((n_reads, length)) < probability).astype(float)
    observed = np.broadcast_to(sites, calls.shape).copy()
    calls[~observed] = np.nan
    return positions, calls, observed, observed.copy()


def test_grid_keeps_the_requested_range_on_a_long_region():
    grid = period_grid(0, 4000)
    assert grid.status == OK and not grid.narrowed
    assert grid.periods[0] == 400 and grid.periods[-1] == 80 and grid.periods.size == 321
    assert grid.peak_range == (150.0, 250.0)


def test_grid_narrows_to_three_cycles():
    grid = period_grid(100, 700)  # 600 bp: longest period 200
    assert grid.status == OK and grid.narrowed
    assert grid.period_range == (80.0, 200.0)
    assert grid.peak_range == (150.0, 200.0)
    assert grid.periods[0] == 200 and grid.periods[-1] == 80
    assert period_grid(0, 600, min_cycles=2).period_range == (80.0, 300.0)


def test_grid_skips_a_region_too_short_for_the_peak_range():
    grid = period_grid(0, 400)  # longest 133 < peak range floor 150
    assert grid.status == REGION_TOO_SHORT and grid.periods.size == 0
    assert period_grid(0, 200).status == REGION_TOO_SHORT  # longest 66 < 80


def test_grid_rejects_bad_ranges():
    with pytest.raises(ValueError, match="empty"):
        period_grid(5, 5)
    with pytest.raises(ValueError, match="peak range"):
        period_grid(0, 4000, peak_range=(50, 250))


def test_planted_period_is_recovered():
    positions, calls, observed, design = _reads()
    grid = period_grid(0, 1600)  # longest period 533 -> 400
    power, stats = read_periodograms(positions, calls, observed, design, grid)
    assert (stats["status"] == OK).all()
    assert power.shape == (6, grid.periods.size)
    assert np.all(np.abs(stats["peak_period_bp"] - 190) <= 6)
    assert (stats["coverage"] == 1.0).all()


def test_equals_the_spatial_stage_method():
    positions, calls, observed, design = _reads(n_reads=2)
    grid = period_grid(0, 1600)
    power, stats = read_periodograms(positions, calls, observed, design, grid)
    sites = observed[0]
    direct = analyze_ls_periodicity_direct(
        positions[sites], calls[0, sites], nrl_search_bp=(150, 250), period_range_bp=(80, 400)
    )
    np.testing.assert_allclose(power[0], direct["ls_power"], rtol=1e-6)
    assert stats["peak_period_bp"].iat[0] == pytest.approx(direct["ls_nrl_bp"])


def test_only_the_region_is_used_and_thresholds_apply():
    positions, calls, observed, design = _reads(n_reads=4, length=2400)
    observed[1, 1200:] = False  # read 1 covers half the region
    calls[1, 1200:] = np.nan
    observed[2] = False
    observed[2, 1000:1030] = True  # read 2: a handful of sites
    calls[3] = np.where(observed[3], 1.0, np.nan)  # read 3: flat signal
    grid = period_grid(800, 2400)
    _, stats = read_periodograms(positions, calls, observed, design, grid)
    assert list(stats["status"]) == [OK, LOW_COVERAGE, LOW_COVERAGE, NO_SIGNAL]
    _, stats = read_periodograms(
        positions, calls, observed, design, grid, min_coverage=0.0, min_sites=40
    )
    assert stats["status"].iat[2] == TOO_FEW_SITES
    assert stats["n_sites"].iat[0] == design[0, 800:2400].sum()


def test_dense_signal_and_its_site_sampling():
    """A dense step track and the same track at sites only both find its period."""
    positions = np.arange(1800)
    track = ((positions // 95) % 2).astype(float)[None, :].repeat(2, axis=0)  # period 190
    dense = np.ones_like(track, dtype=bool)
    grid = period_grid(0, 1800)
    _, every = read_periodograms(positions, track, dense, dense, grid)
    sites = np.broadcast_to(positions % 4 == 0, track.shape)
    _, sampled = read_periodograms(positions, track, sites, sites, grid)
    for table in (every, sampled):
        assert (table["status"] == OK).all()
        assert np.all(np.abs(table["peak_period_bp"] - 190) <= 6)
    assert every["n_sites"].iat[0] == 1800 and sampled["n_sites"].iat[0] == 450


def test_a_region_too_short_marks_every_read():
    positions, calls, observed, design = _reads(n_reads=3)
    power, stats = read_periodograms(positions, calls, observed, design, period_grid(0, 300))
    assert power.shape == (3, 0)
    assert (stats["status"] == REGION_TOO_SHORT).all()


def test_peak_at_edge_marks_unresolved_peaks():
    """A short region narrows the band to 150-200 bp; a 230 bp period pins the peak at 200."""
    positions, calls, observed, design = _reads(period=230.0, length=600)
    _, stats = read_periodograms(positions, calls, observed, design, period_grid(0, 600))
    assert (stats["peak_period_bp"] == 200).all() and stats["peak_at_edge"].all()
    positions, calls, observed, design = _reads()  # 190 bp over 1.6 kb: resolved
    _, stats = read_periodograms(positions, calls, observed, design, period_grid(0, 1600))
    assert not stats["peak_at_edge"].any()


def test_reads_are_scored_with_single_threaded_blas(monkeypatch):
    """`F75`: a multi-threaded BLAS oversubscribes worker processes."""
    from threadpoolctl import threadpool_info

    from smftools.analysis.compute import read_periodicity as module

    seen = []
    original = module.analyze_ls_periodicity_direct

    def recording(*args, **kwargs):
        seen.append(
            {pool["num_threads"] for pool in threadpool_info() if pool["user_api"] == "blas"}
        )
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "analyze_ls_periodicity_direct", recording)
    positions, calls, observed, design = _reads(n_reads=2)
    read_periodograms(positions, calls, observed, design, period_grid(0, 1600))
    assert seen and all(threads <= {1} for threads in seen)

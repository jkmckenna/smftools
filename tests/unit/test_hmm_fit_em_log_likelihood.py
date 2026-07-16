from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from smftools.hmm.HMM import create_hmm


def _cfg(**overrides):
    defaults = dict(
        hmm_n_states=2,
        hmm_distance_bins=[1, 5, 10, 25, 50, 100],
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _synthetic_binary_data(rng, n_reads=40, n_positions=24):
    # Two clear regimes so EM has real structure to learn: the first half of
    # reads are mostly 1 in the first half of positions then 0, the second
    # half of reads are the mirror image, with a little label noise.
    X = np.zeros((n_reads, n_positions))
    half_reads, half_pos = n_reads // 2, n_positions // 2
    X[:half_reads, :half_pos] = 1
    X[half_reads:, half_pos:] = 1
    noise = rng.random(X.shape) < 0.05
    return np.logical_xor(X.astype(bool), noise).astype(float)


@pytest.mark.unit
@pytest.mark.parametrize("arch", ["single", "single_distance_binned"])
def test_fit_em_log_likelihood_is_not_a_near_zero_constant(arch):
    # Regression test for a real bug: ll_proxy used to be computed from
    # gamma.sum(dim=2), which _forward_backward already row-normalizes to 1 at
    # every iteration by construction -- so the "log-likelihood" was ~0 always,
    # completely independent of fit quality, and tol-based early stopping fired
    # at iteration 2 on every real fit regardless of true convergence.
    rng = np.random.default_rng(0)
    X = _synthetic_binary_data(rng)
    coords = np.arange(X.shape[1])
    model = create_hmm(_cfg(), arch=arch, device="cpu")

    hist = model.fit(X, coords, device="cpu", max_iter=25, tol=0.0, verbose=False)

    assert len(hist) == 25
    # A real per-read log-likelihood summed over 40 reads x 24 positions on a
    # genuinely structured dataset should be a large-magnitude negative
    # number, not clustered in the ~1e-8 noise band the old gamma-sum proxy
    # produced.
    assert abs(hist[0]) > 1.0
    assert max(abs(v) for v in hist) > 1.0


@pytest.mark.unit
def test_fit_em_log_likelihood_is_nondecreasing_across_iterations():
    # EM guarantees the true log-likelihood is monotonically non-decreasing;
    # a small numerical tolerance absorbs floating-point noise.
    rng = np.random.default_rng(1)
    X = _synthetic_binary_data(rng)
    coords = np.arange(X.shape[1])
    model = create_hmm(_cfg(), arch="single", device="cpu")

    hist = model.fit(X, coords, device="cpu", max_iter=25, tol=0.0, verbose=False)

    diffs = np.diff(np.asarray(hist))
    assert (diffs >= -1e-6).all(), f"log-likelihood decreased between iterations: {diffs.min()}"


@pytest.mark.unit
def test_fit_em_relative_tolerance_stops_before_max_iter_once_converged():
    # Regression test for the tol early-stop mechanism actually working: the
    # break condition compares the iteration-over-iteration change against
    # tol * max(abs(current_ll), 1.0), not a fixed absolute epsilon -- a real
    # log-likelihood's magnitude scales with dataset size (N*L), so a fixed
    # absolute tol (the pre-fix default) either fires immediately (irrelevant
    # scale) or never fires before max_iter (too-large scale). With the
    # default relative tol=1e-5 (BaseHMM.fit), a small, easily-learnable
    # synthetic dataset should converge and stop well short of a generous
    # max_iter budget.
    rng = np.random.default_rng(3)
    X = _synthetic_binary_data(rng)
    coords = np.arange(X.shape[1])
    model = create_hmm(_cfg(), arch="single", device="cpu")

    hist = model.fit(X, coords, device="cpu", max_iter=200, tol=1e-5, verbose=False)

    assert len(hist) < 200


@pytest.mark.unit
def test_multi_bernoulli_fit_em_log_likelihood_is_not_a_near_zero_constant():
    rng = np.random.default_rng(2)
    x_single = _synthetic_binary_data(rng, n_reads=30, n_positions=16)
    X = np.stack([x_single, 1.0 - x_single], axis=-1)  # (N, L, 2 channels)
    coords = np.arange(X.shape[1])
    model = create_hmm(_cfg(), arch="multi", device="cpu")

    hist = model.fit(X, coords, device="cpu", max_iter=20, tol=0.0, verbose=False)

    assert len(hist) == 20
    assert abs(hist[0]) > 1.0

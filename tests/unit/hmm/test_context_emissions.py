"""HCE-02: context-aware Bernoulli emissions with fixed weights."""

import numpy as np
import pytest
import torch

from smftools.hmm.HMM import ContextBernoulliHMM, SingleBernoulliHMM

N_CONTEXTS = 16


def _simulate(n_reads=60, length=900, seed=0):
    """Reads of accessible / protected blocks; accessible sites modified by context.

    Half the contexts are disfavoured (weight 0.2), half favoured (1.6), applied
    as the model applies them: on the log-odds of the accessible level 0.55.
    """
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, N_CONTEXTS, size=length)
    weights = np.where(np.arange(N_CONTEXTS) % 2 == 0, 0.2, 1.6)
    states = np.zeros((n_reads, length), dtype=int)  # 1 = accessible
    for read in range(n_reads):
        position, state = 0, int(rng.integers(0, 2))
        while position < length:
            run = int(rng.integers(60, 180))
            states[read, position : position + run] = state
            position, state = position + run, 1 - state
    logit = np.log(0.55 / 0.45) + np.log(weights[codes])
    p = np.where(states == 1, 1 / (1 + np.exp(-logit)), 0.04)
    calls = (rng.random(p.shape) < p).astype(float)
    calls[rng.random(p.shape) < 0.5] = np.nan  # half the positions are not sites
    return calls, states, codes, weights


def _fit(model, calls):
    model.fit(calls, np.arange(calls.shape[1]), max_iter=60, tol=1e-7, device="cpu")
    return model


def test_unit_weights_reproduce_the_single_bernoulli_model():
    calls, _, codes, _ = _simulate(n_reads=20, length=400)
    plain = _fit(SingleBernoulliHMM(init_emission=[0.1, 0.6]), calls)
    context = ContextBernoulliHMM(
        init_emission=[0.1, 0.6], log_weights=np.zeros(N_CONTEXTS), position_codes=codes
    )
    _fit(context, calls)
    torch.testing.assert_close(plain.emission, context.emission)
    torch.testing.assert_close(plain.trans, context.trans)
    _, gamma_plain = plain.decode(calls, np.arange(calls.shape[1]), device="cpu")
    _, gamma_context = context.decode(calls, np.arange(calls.shape[1]), device="cpu")
    np.testing.assert_allclose(gamma_plain, gamma_context)


def test_true_weights_recover_states_better_at_disfavoured_contexts():
    calls, states, codes, weights = _simulate()
    coords = np.arange(calls.shape[1])
    plain = _fit(SingleBernoulliHMM(init_emission=[0.1, 0.6]), calls)
    context = _fit(
        ContextBernoulliHMM(
            init_emission=[0.1, 0.6], log_weights=np.log(weights), position_codes=codes
        ),
        calls,
    )
    accuracy = {}
    for name, model in (("plain", plain), ("context", context)):
        called, _ = model.decode(calls, coords, device="cpu")
        accessible = model.modified_state_index()
        truth = states == 1
        observed = ~np.isnan(calls)
        disfavoured = np.broadcast_to(weights[codes] < 1, calls.shape) & observed
        accuracy[name] = (
            ((called == accessible) == truth)[observed].mean(),
            ((called == accessible) == truth)[disfavoured & truth].mean(),
        )
    assert accuracy["context"][0] > accuracy["plain"][0]
    assert accuracy["context"][1] > accuracy["plain"][1]  # accessible, disfavoured sites
    # The fitted accessible level is the context-free 0.55 the reads were made with.
    assert float(context.emission[context.modified_state_index()]) == pytest.approx(0.55, abs=0.05)


def test_weights_apply_only_to_the_modified_state_by_default():
    model = ContextBernoulliHMM(
        init_emission=[0.05, 0.5], log_weights=[np.log(4.0)], position_codes=[0, -1]
    )
    p = model._site_probabilities(2)  # columns 0 (weighted context), 1 (no context)
    assert float(p[0, 1]) == pytest.approx(0.8)  # logit(0.5) + log 4 -> 0.8
    assert float(p[1, 1]) == pytest.approx(0.5)
    assert float(p[0, 0]) == pytest.approx(0.05) and float(p[1, 0]) == pytest.approx(0.05)
    every = ContextBernoulliHMM(
        init_emission=[0.05, 0.5],
        log_weights=[np.log(4.0)],
        position_codes=[0],
        context_states="all",
    )
    assert float(every._site_probabilities(1)[0, 0]) > 0.05


def test_save_load_round_trip(tmp_path):
    calls, _, codes, weights = _simulate(n_reads=10, length=300)
    model = _fit(
        ContextBernoulliHMM(
            init_emission=[0.1, 0.6], log_weights=np.log(weights), position_codes=codes
        ),
        calls,
    )
    path = tmp_path / "model.pt"
    model.save(path)
    loaded = ContextBernoulliHMM.load(path)
    np.testing.assert_array_equal(loaded.position_codes, codes)
    torch.testing.assert_close(loaded.log_weights, model.log_weights)
    coords = np.arange(calls.shape[1])
    # Loading renormalizes the transition rows (as for every HMM): equal to ~1e-7.
    np.testing.assert_allclose(
        loaded.decode(calls, coords, device="cpu")[1],
        model.decode(calls, coords, device="cpu")[1],
        rtol=1e-5,
        atol=1e-6,
    )


def test_set_contexts_validates():
    model = ContextBernoulliHMM()
    with pytest.raises(ValueError, match="no weight"):
        model.set_contexts(log_weights=[0.0, 0.0], position_codes=[0, 5])
    with pytest.raises(ValueError, match="finite"):
        model.set_contexts(log_weights=[np.inf], position_codes=[0])
    with pytest.raises(ValueError, match="context_states"):
        ContextBernoulliHMM(context_states="some")

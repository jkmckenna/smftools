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


def _learned(codes, **kwargs):
    return ContextBernoulliHMM(
        init_emission=[0.1, 0.6],
        log_weights=np.zeros(N_CONTEXTS),
        position_codes=codes,
        learn=True,
        **kwargs,
    )


def test_learned_weights_converge_to_the_planted_ones():
    """`HCE-03`."""
    calls, _, codes, weights = _simulate(n_reads=120, length=1200, seed=1)
    model = _fit(_learned(codes), calls)
    # Weights are relative to the state's overall rate, which averages over the
    # contexts: they are identified up to a constant the level absorbs.
    learned = model.log_weights.numpy()
    truth = np.log(weights)
    learned, truth = learned - learned.mean(), truth - truth.mean()
    assert np.corrcoef(learned, truth)[0, 1] > 0.95
    assert np.abs(learned - truth).max() < 0.2
    assert np.array_equal(learned < 0, truth < 0)  # the split, on every context


def test_shrinkage_holds_a_rare_context_at_the_state_rate():
    calls, _, codes, _ = _simulate(n_reads=60, length=900, seed=2)
    codes = codes.copy()
    codes[codes == 3] = 0
    codes[:2] = 3  # context 3 now sits at two positions only
    model = _fit(_learned(codes, shrinkage=1e6), calls)
    assert np.abs(model.log_weights.numpy()).max() < 0.05  # everything pinned to the level
    loose = _fit(_learned(codes, shrinkage=0.0), calls)
    assert np.abs(loose.log_weights.numpy()).max() > 0.5


def test_no_context_effect_learns_no_weights():
    rng = np.random.default_rng(3)
    calls, states, codes, _ = _simulate(n_reads=60, length=900, seed=3)
    p = np.where(states == 1, 0.55, 0.04)
    calls = np.where(np.isnan(calls), np.nan, (rng.random(p.shape) < p).astype(float))
    model = _fit(_learned(codes), calls)
    assert np.abs(model.log_weights.numpy()).max() < 0.25


def test_cpg_exclude_ignores_cpg_sites():
    calls, _, codes, weights = _simulate(n_reads=20, length=400, seed=4)
    cpg_codes = [1, 5]
    model = ContextBernoulliHMM(
        init_emission=[0.1, 0.6],
        log_weights=np.log(weights),
        position_codes=codes,
        cpg_codes=cpg_codes,
        cpg="exclude",
    )
    masked = calls.copy()
    masked[:, np.isin(codes, cpg_codes)] = np.nan
    reference = ContextBernoulliHMM(
        init_emission=[0.1, 0.6], log_weights=np.log(weights), position_codes=codes
    )
    coords = np.arange(calls.shape[1])
    np.testing.assert_allclose(
        model.decode(calls, coords, device="cpu")[1],
        reference.decode(masked, coords, device="cpu")[1],
    )


def test_learned_weights_export_as_a_table(tmp_path):
    from smftools.analysis.compute.site_context_bias import (
        read_weight_table,
        weights_for,
        write_weight_table,
    )

    calls, _, codes, _ = _simulate(n_reads=30, length=600, seed=5)
    model = _fit(_learned(codes), calls)
    table = model.weight_table("CseDa01", k=3)
    assert (table["source"] == "learned").all() and len(table) == 16
    assert table["cpg"].sum() == 4  # N-C-G
    write_weight_table(table, tmp_path / "learned.parquet")
    weights = weights_for(read_weight_table(tmp_path / "learned.parquet"), "CseDa01", 3)
    np.testing.assert_allclose(np.log(weights), model.log_weights.numpy())
    with pytest.raises(ValueError, match="do not match"):
        model.weight_table("x", k=5)


def test_learned_mode_validation():
    with pytest.raises(ValueError, match="modified state only"):
        ContextBernoulliHMM(learn=True, context_states="all")
    with pytest.raises(ValueError, match="cpg"):
        ContextBernoulliHMM(cpg="maybe")
    with pytest.raises(ValueError, match="bracket"):
        ContextBernoulliHMM(weight_bounds=(2.0, 10.0))

"""`HPS-03`: tied EM -- shared emissions, per-group transitions, offsets."""

import numpy as np
import pytest
import torch

from smftools.hmm.HMM import ContextBernoulliHMM, SingleBernoulliHMM
from smftools.hmm.tied_fit import TiedMember, tied_em

N_CONTEXTS = 16


def _simulate(stay, efficiency, *, n_reads=120, length=600, seed=0, codes=None, weights=None):
    """Two-state chromatin with its own persistence; shared levels (logit
    -2.5 protected, 0.4 accessible), shifted by ``efficiency``; optional
    context weights on the accessible state."""
    rng = np.random.default_rng(seed)
    states = np.zeros((n_reads, length), dtype=int)
    for read in range(n_reads):
        state = int(rng.integers(0, 2))
        for position in range(length):
            states[read, position] = state
            if rng.random() > stay[state]:
                state = 1 - state
    logit = np.where(states == 1, 0.4, -2.5) + efficiency
    if codes is not None:
        logit = logit + np.where(states == 1, np.log(weights[codes])[None, :], 0.0)
    calls = (rng.random(states.shape) < 1 / (1 + np.exp(-logit))).astype(float)
    calls[rng.random(states.shape) < 0.4] = np.nan
    return calls


def _single():
    model = SingleBernoulliHMM(init_emission=[0.1, 0.5])
    model.trans.data = torch.tensor([[0.9, 0.1], [0.1, 0.9]], dtype=model.dtype)
    return model


def test_shared_levels_dose_slope_and_own_transitions():
    doses = np.log([1.0, 2.0, 4.0])
    true_beta = 0.5
    stays = [(0.95, 0.90), (0.97, 0.85), (0.92, 0.95)]
    members = []
    for i, (dose, stay) in enumerate(zip(doses, stays, strict=True)):
        efficiency = true_beta * (dose - doses.mean())
        X = _simulate(stay, efficiency, seed=i)
        members.append(TiedMember(_single(), X, np.arange(X.shape[1]), covariates=np.array([dose])))
    result = tied_em(
        members,
        free_w=np.zeros((2, 0), bool),
        fixed_w=np.zeros((2, 0)),
        rounds=8,
        transition_iters=5,
        shrinkage=50.0,
        bound=float(np.log(10)),
        prior_sd=1.0,
    )
    theta = np.sort(result.theta)
    np.testing.assert_allclose(theta, [-2.5, 0.4], atol=0.25)
    assert 0.25 < result.beta[0] < 0.75  # recovered slope on log dose
    assert result.offsets[0] < result.offsets[1] < result.offsets[2]
    for member, stay in zip(members, stays, strict=True):
        model = member.model
        modified = model.modified_state_index()
        learned = [
            model.trans[1 - modified, 1 - modified].item(),
            model.trans[modified, modified].item(),
        ]
        np.testing.assert_allclose(learned, stay, atol=0.03)
    assert result.constant_covariates == []


def test_constant_covariate_gives_no_offset():
    members = []
    for i in range(2):
        X = _simulate((0.95, 0.9), 0.0, seed=10 + i, n_reads=60, length=300)
        members.append(TiedMember(_single(), X, np.arange(X.shape[1]), covariates=np.array([1.0])))
    result = tied_em(
        members,
        free_w=np.zeros((2, 0), bool),
        fixed_w=np.zeros((2, 0)),
        rounds=3,
        transition_iters=3,
        shrinkage=50.0,
        bound=float(np.log(10)),
        prior_sd=1.0,
    )
    assert result.constant_covariates == [0]
    assert np.allclose(result.offsets, 0.0)


def test_learned_context_weights_are_shared():
    rng = np.random.default_rng(3)
    length = 600
    codes = rng.integers(0, N_CONTEXTS, size=length)
    weights = np.where(np.arange(N_CONTEXTS) % 2 == 0, 0.3, 2.0)
    members = []
    for i, stay in enumerate([(0.95, 0.9), (0.97, 0.85)]):
        X = _simulate(stay, 0.0, seed=20 + i, codes=codes, weights=weights)
        model = ContextBernoulliHMM(
            init_emission=[0.1, 0.5],
            log_weights=np.zeros((2, N_CONTEXTS)),
            position_codes=codes,
            context_states="all",
        )
        model.trans.data = torch.tensor([[0.9, 0.1], [0.1, 0.9]], dtype=model.dtype)
        members.append(TiedMember(model, X, np.arange(length)))
    free = np.ones((2, N_CONTEXTS), bool)
    result = tied_em(
        members,
        free_w=free,
        fixed_w=np.zeros((2, N_CONTEXTS)),
        rounds=6,
        transition_iters=4,
        shrinkage=50.0,
        bound=float(np.log(10)),
        prior_sd=1.0,
    )
    modified = int(np.argmax(result.theta))
    learned = result.w[modified] - result.w[modified].mean()
    truth = np.log(weights) - np.log(weights).mean()
    assert np.corrcoef(learned, truth)[0, 1] > 0.9
    # both members carry the same emissions
    np.testing.assert_allclose(
        members[0].model.log_weights.numpy(), members[1].model.log_weights.numpy()
    )


def test_tied_em_needs_members():
    with pytest.raises(ValueError, match="at least one"):
        tied_em(
            [],
            free_w=np.zeros((2, 0), bool),
            fixed_w=np.zeros((2, 0)),
            rounds=1,
            transition_iters=1,
            shrinkage=50.0,
            bound=1.0,
            prior_sd=1.0,
        )

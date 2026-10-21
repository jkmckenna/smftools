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


def _build_member(index):
    """A deterministic member, built where it is fitted (a shard worker)."""
    doses = np.log([1.0, 2.0, 4.0, 1.5, 3.0])
    stays = [(0.95, 0.90), (0.97, 0.85), (0.92, 0.95), (0.96, 0.88), (0.94, 0.92)]
    if index == 3:
        raise ValueError("no reads for this member")
    X = _simulate(stays[index], 0.5 * (doses[index] - doses.mean()), n_reads=60, seed=index)
    return TiedMember(_single(), X, np.arange(X.shape[1]), covariates=np.array([doses[index]]))


def test_sharded_members_match_the_in_process_fit():
    """Members built and fitted by worker processes give the in-process
    result exactly: same shared levels, offsets, transitions and history;
    a member whose input is unusable is reported as skipped."""
    from smftools.hmm.tied_fit import LocalMembers, ShardedMembers, tied_em_backend

    settings = dict(
        free_w=np.zeros((2, 0), bool),
        fixed_w=np.zeros((2, 0)),
        rounds=3,
        transition_iters=3,
        shrinkage=50.0,
        bound=float(np.log(10)),
        prior_sd=1.0,
    )
    kept = [0, 1, 2, 4]
    local = [_build_member(i) for i in kept]
    expected, local_models = tied_em_backend(
        LocalMembers(local),
        covariates=[m.covariates for m in local],
        categories=[None] * len(local),
        **settings,
    )
    backend = ShardedMembers(_build_member, [(i,) for i in range(5)], workers=2)
    try:
        assert backend.skipped == {3: "no reads for this member"}
        assert backend.indices == kept
        result, models = tied_em_backend(
            backend,
            covariates=[m.covariates for m in local],
            categories=[None] * len(local),
            **settings,
        )
    finally:
        backend.close()
    np.testing.assert_allclose(result.theta, expected.theta, rtol=0, atol=1e-10)
    np.testing.assert_allclose(result.offsets, expected.offsets, rtol=0, atol=1e-10)
    np.testing.assert_allclose(result.history, expected.history, rtol=1e-12)
    for model, reference in zip(models, local_models, strict=True):
        torch.testing.assert_close(model.trans, reference.trans, rtol=0, atol=1e-10)
        torch.testing.assert_close(model.emission, reference.emission, rtol=0, atol=1e-10)


def test_a_failing_shard_worker_is_reported():
    from smftools.hmm.tied_fit import ShardedMembers

    with pytest.raises(RuntimeError, match="tied-fit worker failed"):
        ShardedMembers(_explode, [(0,), (1,)], workers=2)


def _explode(index):
    raise RuntimeError(f"cannot build member {index}")


def _two_enzyme_member(index):
    """Members of two enzymes (different emission levels), one chromatin
    process (the same stay probabilities)."""
    stay = (0.96, 0.9)
    efficiency = -1.0 if index < 2 else 0.8  # enzyme A weaker than enzyme B
    X = _simulate(stay, efficiency, n_reads=120, seed=10 + index)
    return TiedMember(_single(), X, np.arange(X.shape[1]))


def _pooled_settings():
    return dict(
        free_w=np.zeros((2, 0), bool),
        fixed_w=np.zeros((2, 0)),
        rounds=6,
        transition_iters=4,
        shrinkage=50.0,
        bound=float(np.log(10)),
        prior_sd=1.0,
        emission_groups=["A", "A", "B", "B"],
        transition_groups=["ref"] * 4,
    )


def test_pooled_transitions_are_shared_and_emissions_stay_per_enzyme():
    """`HPS-16`: one transition model pooled over every member, across
    emission groups; each emission group keeps its own levels."""
    from smftools.hmm.tied_fit import LocalMembers, tied_em_backend

    members = [_two_enzyme_member(i) for i in range(4)]
    result, models = tied_em_backend(
        LocalMembers(members),
        covariates=[np.zeros(0)] * 4,
        categories=[None] * 4,
        **_pooled_settings(),
    )
    trans = [m.trans.detach().numpy() for m in models]
    for other in trans[1:]:
        np.testing.assert_allclose(other, trans[0], atol=1e-12)  # one shared model
    modified = models[0].modified_state_index()
    stays = [trans[0][1 - modified, 1 - modified], trans[0][modified, modified]]
    np.testing.assert_allclose(stays, (0.96, 0.9), atol=0.03)
    levels = {key: np.sort(_sig(g["theta"])) for key, g in result.groups.items()}
    assert levels["A"][1] < levels["B"][1]  # enzyme B's accessible level is higher
    assert set(result.transitions) == {"ref"}


def test_pooled_transitions_sharded_match_in_process():
    from smftools.hmm.tied_fit import LocalMembers, ShardedMembers, tied_em_backend

    local = [_two_enzyme_member(i) for i in range(4)]
    expected, local_models = tied_em_backend(
        LocalMembers(local),
        covariates=[np.zeros(0)] * 4,
        categories=[None] * 4,
        **_pooled_settings(),
    )
    backend = ShardedMembers(_two_enzyme_member, [(i,) for i in range(4)], workers=2)
    try:
        result, models = tied_em_backend(
            backend,
            covariates=[np.zeros(0)] * 4,
            categories=[None] * 4,
            **_pooled_settings(),
        )
    finally:
        backend.close()
    np.testing.assert_allclose(result.history, expected.history, rtol=1e-12)
    for model, reference in zip(models, local_models, strict=True):
        torch.testing.assert_close(model.trans, reference.trans, rtol=0, atol=1e-10)
        torch.testing.assert_close(model.emission, reference.emission, rtol=0, atol=1e-10)


def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x)))

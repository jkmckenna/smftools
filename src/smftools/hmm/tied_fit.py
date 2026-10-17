"""Tied EM: fit groups sharing one emission model (`HPS-03`).

Transitions describe chromatin and stay per fit group (sample x reference x
strand); emissions describe the measurement and are shared by every fit group
of an emission group (e.g. one enzyme). For site *i* of fit group *g* in
state *k*::

    logit P(modified) = theta[k] + w[k, c(i)] + delta[g]
    delta[g] = sum_j beta[j] * x[g, j]  +  gamma[e(g)]

- ``w``: per-state context weights -- learned (shrunk, bounded), fixed from a
  table, or absent;
- ``x``: technical covariates of the sample (e.g. log enzyme per cell),
  centred within the emission group; ``beta`` one slope each, Gaussian prior;
- ``gamma``: optional categorical technical offsets (e.g. a batch), Gaussian
  prior.

Biological factors (cell type, genotype, treatment, replicate) never enter
``delta``, so differences along them are read as chromatin.

Each round: every fit group's transitions are re-estimated with its emissions
fixed (the model's own EM), then its posteriors give hits / trials per state x
context, and one penalised logistic fit over the pooled table updates the
shared emissions and the offsets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np


@dataclass
class TiedMember:
    """One fit group: a configured model and its training reads."""

    model: Any
    X: np.ndarray
    coords: np.ndarray
    covariates: np.ndarray = field(default_factory=lambda: np.zeros(0))  # (J,) raw log values
    category: str | None = None


@dataclass
class TiedResult:
    theta: np.ndarray
    w: np.ndarray
    beta: np.ndarray
    gamma: dict[str, float]
    offsets: list[float]
    history: list[float]
    constant_covariates: list[int]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9)
    return np.log(p) - np.log1p(-p)


def _n_contexts(model) -> int:
    weights = getattr(model, "log_weights", None)
    return 0 if weights is None else int(weights.shape[-1])


def emission_tables(model, X: np.ndarray, coords: np.ndarray, device="cpu"):
    """(hits, trials), each (K, C + 1): posterior-weighted modified calls and
    observations per state and context code (last column: no context)."""
    X = np.asarray(X, dtype=float)
    _, gamma = model.decode(X, coords, device=device)
    gamma = np.asarray(gamma)
    n_states, n_contexts = gamma.shape[-1], _n_contexts(model)
    observed = np.isfinite(X)
    if n_contexts:
        codes = np.asarray(model._column_codes(X.shape[1]))
        excluded = model._excluded_columns(X.shape[1]).detach().cpu().numpy()
        codes = np.broadcast_to(codes, X.shape)
        observed = observed & ~np.broadcast_to(excluded, X.shape)
        codes = np.where(codes >= 0, codes, n_contexts)
    else:
        codes = np.zeros(X.shape, dtype=np.int64)
    calls = np.nan_to_num(X)
    hits = np.zeros((n_states, n_contexts + 1))
    trials = np.zeros((n_states, n_contexts + 1))
    flat_codes = codes[observed]
    for k in range(n_states):
        g = gamma[..., k][observed]
        np.add.at(hits[k], flat_codes, g * calls[observed])
        np.add.at(trials[k], flat_codes, g)
    return hits, trials


def fit_emission_table(
    hits: np.ndarray,
    trials: np.ndarray,
    *,
    free_w: np.ndarray,
    fixed_w: np.ndarray,
    covariates: np.ndarray,
    categories: Sequence[int | None],
    n_categories: int,
    ridge: np.ndarray,
    bound: float,
    prior_sd: float,
    init_theta: np.ndarray,
    init_w: np.ndarray,
):
    """Penalised binomial fit of theta (K), free w, beta (J), gamma.

    ``hits`` / ``trials``: (G, K, C+1); ``free_w`` / ``fixed_w``: (K, C);
    ``covariates``: (G, J), centred; ``categories``: per group an index or None.
    """
    from scipy.optimize import minimize

    G, K, C1 = hits.shape
    C = C1 - 1
    J = covariates.shape[1]
    free_idx = np.flatnonzero(free_w.ravel())
    n_free = free_idx.size
    cat = np.array([-1 if c is None else int(c) for c in categories])
    precision = 1.0 / prior_sd**2

    def unpack(x):
        theta = x[:K]
        w = fixed_w.astype(float).ravel().copy()
        w[free_idx] = x[K : K + n_free]
        w = w.reshape(K, C)
        beta = x[K + n_free : K + n_free + J]
        gamma = x[K + n_free + J :]
        delta = covariates @ beta + np.where(
            cat >= 0, gamma[np.maximum(cat, 0)] if gamma.size else 0.0, 0.0
        )
        return theta, w, beta, gamma, delta

    def objective(x):
        theta, w, beta, gamma, delta = unpack(x)
        w_full = np.concatenate([w, np.zeros((K, 1))], axis=1)
        eta = theta[None, :, None] + w_full[None] + delta[:, None, None]
        loss = float((trials * np.logaddexp(0.0, eta) - hits * eta).sum())
        resid = trials * _sigmoid(eta) - hits  # d loss / d eta, (G, K, C+1)
        g_theta = resid.sum(axis=(0, 2))
        g_w = resid.sum(axis=0)[:, :C].ravel()[free_idx]
        g_delta = resid.sum(axis=(1, 2))  # (G,)
        g_beta = covariates.T @ g_delta
        g_gamma = np.zeros(n_categories)
        for index in range(n_categories):
            g_gamma[index] = g_delta[cat == index].sum()
        x_w = x[K : K + n_free]
        ridge_w = np.repeat(ridge, C)[free_idx] if n_free else np.zeros(0)
        loss += 0.5 * float((ridge_w * x_w**2).sum())
        loss += 0.5 * precision * float((beta**2).sum() + (gamma**2).sum())
        grad = np.concatenate(
            [g_theta, g_w + ridge_w * x_w, g_beta + precision * beta, g_gamma + precision * gamma]
        )
        return loss, grad

    x0 = np.concatenate(
        [
            np.asarray(init_theta, float),
            np.asarray(init_w, float).ravel()[free_idx],
            np.zeros(J),
            np.zeros(n_categories),
        ]
    )
    bounds = [(None, None)] * K + [(-bound, bound)] * n_free + [(None, None)] * (J + n_categories)
    result = minimize(objective, x0, jac=True, method="L-BFGS-B", bounds=bounds)
    theta, w, beta, gamma, delta = unpack(result.x)
    return theta, w, beta, gamma, delta


def set_emissions(model, theta: np.ndarray, w: np.ndarray, offset: float) -> None:
    """Give a member model the shared emissions plus its offset."""
    import torch

    level = _sigmoid(np.asarray(theta) + offset)
    model.emission.data = torch.as_tensor(level, dtype=model.emission.dtype)
    if _n_contexts(model):
        model.set_contexts(
            log_weights=np.asarray(w),
            position_codes=model.position_codes,
            cpg_codes=model.cpg_codes,
            position_offset=int(model.context_position_offset),
        )


def tied_em(
    members: Sequence[TiedMember],
    *,
    free_w: np.ndarray,
    fixed_w: np.ndarray,
    rounds: int,
    transition_iters: int,
    shrinkage: float,
    bound: float,
    prior_sd: float,
    device="cpu",
) -> TiedResult:
    """Fit the members' shared emissions, offsets and own transitions."""
    if not members:
        raise ValueError("tied EM needs at least one member")
    raw = np.array([np.atleast_1d(m.covariates).astype(float) for m in members])
    if raw.ndim == 1:
        raw = raw[:, None]
    covariates = raw - raw.mean(axis=0, keepdims=True) if raw.size else np.zeros((len(members), 0))
    constant = [j for j in range(covariates.shape[1]) if np.allclose(covariates[:, j], 0.0)]
    names = sorted({m.category for m in members if m.category is not None})
    categories = [names.index(m.category) if m.category is not None else None for m in members]

    # Round 0: each member's own quick fit orients the states and starts the
    # shared levels at their mean.
    history: list[float] = []
    for member in members:
        member.model.fit(
            member.X, member.coords, device=device, max_iter=transition_iters, update_emission=True
        )
    theta = np.mean([_logit(m.model.emission.detach().cpu().numpy()) for m in members], axis=0)
    w = np.array(fixed_w, dtype=float)
    beta = np.zeros(covariates.shape[1])
    gamma = np.zeros(len(names))
    offsets = [0.0] * len(members)
    for round_ in range(rounds + 1):
        tables = []
        total = 0.0
        for member, offset in zip(members, offsets, strict=True):
            if round_ > 0:
                set_emissions(member.model, theta, w, offset)
                hist = member.model.fit(
                    member.X,
                    member.coords,
                    device=device,
                    max_iter=transition_iters,
                    update_emission=False,
                )
                total += float(hist[-1]) if hist else 0.0
            tables.append(emission_tables(member.model, member.X, member.coords, device=device))
        if round_ > 0:
            history.append(total)
        hits = np.stack([t[0] for t in tables])
        trials = np.stack([t[1] for t in tables])
        level = _sigmoid(theta)
        ridge = shrinkage * level * (1.0 - level)
        theta, w, beta, gamma_vec, delta = fit_emission_table(
            hits,
            trials,
            free_w=free_w,
            fixed_w=w,
            covariates=covariates,
            categories=categories,
            n_categories=len(names),
            ridge=ridge,
            bound=bound,
            prior_sd=prior_sd,
            init_theta=theta,
            init_w=w,
        )
        gamma = gamma_vec
        offsets = [float(d) for d in delta]
    for member, offset in zip(members, offsets, strict=True):
        set_emissions(member.model, theta, w, offset)
    return TiedResult(
        theta=theta,
        w=w,
        beta=beta,
        gamma={name: float(gamma[i]) for i, name in enumerate(names)},
        offsets=offsets,
        history=history,
        constant_covariates=constant,
    )

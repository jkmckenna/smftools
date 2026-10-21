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

import pickle
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from ..logging_utils import get_logger

logger = get_logger(__name__)


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
    #: Per emission group: theta, w, beta, gamma, constant covariates
    #: (several groups when transitions are pooled across them, `HPS-16`).
    groups: dict = field(default_factory=dict)
    #: Per transition group (pooled transitions only): start, trans.
    transitions: dict = field(default_factory=dict)
    #: Whether the relative log-likelihood gain fell below ``tol`` before
    #: the round cap, and the rounds run.
    converged: bool = False
    rounds_run: int = 0


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


class LocalMembers:
    """Members held in this process; the rounds run one member at a time."""

    def __init__(self, members: Sequence[TiedMember]):
        self.members = list(members)

    def initial_logits(self, transition_iters: int, device) -> list[np.ndarray]:
        return [_initial_logit(m, transition_iters, device) for m in self.members]

    def round(self, thetas, ws, offsets, *, refit, transition_iters, device):
        return [
            _member_round(m, theta, w, offset, refit, transition_iters, device)
            for m, theta, w, offset in zip(self.members, thetas, ws, offsets, strict=True)
        ]

    def set_emissions(self, thetas, ws, offsets) -> None:
        for m, theta, w, offset in zip(self.members, thetas, ws, offsets, strict=True):
            set_emissions(m.model, theta, w, offset)

    def transition_stats(self, device) -> list[dict]:
        return [_transition_stats(m, device) for m in self.members]

    def set_transitions(self, starts, transitions) -> None:
        for m, start, trans in zip(self.members, starts, transitions, strict=True):
            set_transitions(m.model, start, trans)

    def finalize(self, thetas, ws, offsets) -> list[Any]:
        self.set_emissions(thetas, ws, offsets)
        return [m.model for m in self.members]

    def close(self) -> None:
        pass


def _initial_logit(member: TiedMember, transition_iters: int, device) -> np.ndarray:
    member.model.fit(
        member.X, member.coords, device=device, max_iter=transition_iters, update_emission=True
    )
    return _logit(member.model.emission.detach().cpu().numpy())


def _member_round(member, theta, w, offset, refit, transition_iters, device):
    """(hits, trials, last log-likelihood or 0) for one member and round."""
    loglik = 0.0
    if refit:
        set_emissions(member.model, theta, w, offset)
        hist = member.model.fit(
            member.X,
            member.coords,
            device=device,
            max_iter=transition_iters,
            update_emission=False,
        )
        loglik = float(hist[-1]) if hist else 0.0
    hits, trials = emission_tables(member.model, member.X, member.coords, device=device)
    return hits, trials, loglik


def _transition_stats(member: TiedMember, device) -> dict:
    """Expected start / transition counts and log-likelihood of one member
    under its current parameters (one E step, nothing updated)."""
    stats: dict = {}
    member.model.fit(
        member.X,
        member.coords,
        device=device,
        max_iter=1,
        update_start=False,
        update_trans=False,
        update_emission=False,
        stats_out=stats,
    )
    return stats


def set_transitions(model, start: np.ndarray, trans: np.ndarray) -> None:
    """Give a member model shared (pooled) start and transition probabilities."""
    import torch

    with torch.no_grad():
        model.start.data = torch.as_tensor(start, dtype=model.start.dtype)
        model.trans.data = torch.as_tensor(trans, dtype=model.trans.dtype)
    model._normalize_params()


def pooled_transitions(stats: Sequence[dict], eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    """One start vector and transition matrix from several members' expected
    counts -- the M step of a transition model they share (`HPS-16`)."""
    start = np.sum([s["start"] for s in stats], axis=0) + eps
    trans = np.sum([s["trans"] for s in stats], axis=0) + eps
    rows = trans.sum(axis=1, keepdims=True)
    rows[rows == 0.0] = 1.0
    return start / start.sum(), trans / rows


def _shard_worker(conn, build, jobs, device, probe=None) -> None:
    """A persistent worker owning some members: builds them, then serves
    rounds until told to stop. ``jobs`` is [(index, build args)]; ``probe``
    (optional) is reported for every built member."""
    import traceback

    from ..parallel_utils import configure_worker_threads

    configure_worker_threads(1)
    try:
        members: dict[int, TiedMember] = {}
        skipped: dict[int, str] = {}
        for index, args in jobs:
            try:
                members[index] = build(*args)
            except (KeyError, ValueError) as exc:
                skipped[index] = str(exc)
        probes = {i: probe(m) for i, m in members.items()} if probe is not None else {}
        conn.send(("ok", (skipped, probes)))
        while True:
            command, payload = conn.recv()
            if command == "stop":
                break
            if command == "init":
                result = {i: _initial_logit(m, payload, device) for i, m in members.items()}
            elif command == "round":
                thetas, ws, offsets, refit, iters = payload
                result = {
                    i: _member_round(m, thetas[i], ws[i], offsets[i], refit, iters, device)
                    for i, m in members.items()
                }
            elif command == "set_emissions":
                thetas, ws, offsets = payload
                for i, m in members.items():
                    set_emissions(m.model, thetas[i], ws[i], offsets[i])
                result = {}
            elif command == "transition_stats":
                result = {i: _transition_stats(m, device) for i, m in members.items()}
            elif command == "set_transitions":
                starts, transitions = payload
                for i, m in members.items():
                    set_transitions(m.model, starts[i], transitions[i])
                result = {}
            elif command == "finalize":
                thetas, ws, offsets = payload
                for i, m in members.items():
                    set_emissions(m.model, thetas[i], ws[i], offsets[i])
                # Plain pickle bytes: a tensor sent through a multiprocessing
                # pipe goes by torch's shared-memory hand-off, which keeps
                # this process alive at exit until it times out.
                result = {i: pickle.dumps(m.model) for i, m in members.items()}
            else:
                raise ValueError(f"unknown command {command!r}")
            conn.send(("ok", result))
    except BaseException:
        conn.send(("error", traceback.format_exc()))
    finally:
        conn.close()


class ShardedMembers:
    """Members built and held by persistent worker processes (one shard
    each); a round runs every shard at once. Member order -- and so every
    pooled quantity -- is the same as :class:`LocalMembers`'."""

    def __init__(
        self, build, build_args: Sequence[tuple], *, workers: int, device="cpu", probe=None
    ):
        import multiprocessing

        context = multiprocessing.get_context()
        n = len(build_args)
        workers = max(1, min(int(workers), n))
        self._procs, self._conns = [], []
        for shard in range(workers):
            jobs = [(i, build_args[i]) for i in range(shard, n, workers)]
            parent, child = context.Pipe()
            proc = context.Process(target=_shard_worker, args=(child, build, jobs, device, probe))
            proc.start()
            child.close()
            self._procs.append(proc)
            self._conns.append(parent)
        self.skipped: dict[int, str] = {}
        self.probes: dict[int, Any] = {}
        for skipped, probes in self._gather():
            self.skipped.update(skipped)
            self.probes.update(probes)
        self.indices = [i for i in range(n) if i not in self.skipped]

    def _gather(self) -> list:
        replies = []
        for conn in self._conns:
            try:
                status, payload = conn.recv()
            except EOFError as exc:
                self.close()
                raise RuntimeError("a tied-fit worker exited unexpectedly") from exc
            if status != "ok":
                self.close()
                raise RuntimeError(f"tied-fit worker failed:\n{payload}")
            replies.append(payload)
        return replies

    def _call(self, command, payload) -> dict[int, Any]:
        for conn in self._conns:
            conn.send((command, payload))
        merged: dict[int, Any] = {}
        for reply in self._gather():
            merged.update(reply)
        return merged

    def initial_logits(self, transition_iters: int, device) -> list[np.ndarray]:
        result = self._call("init", transition_iters)
        return [result[i] for i in self.indices]

    def _by_index(self, values) -> dict[int, Any]:
        return dict(zip(self.indices, values, strict=True))

    def round(self, thetas, ws, offsets, *, refit, transition_iters, device):
        payload = (
            self._by_index(thetas),
            self._by_index(ws),
            self._by_index(offsets),
            refit,
            transition_iters,
        )
        result = self._call("round", payload)
        return [result[i] for i in self.indices]

    def set_emissions(self, thetas, ws, offsets) -> None:
        self._call(
            "set_emissions", (self._by_index(thetas), self._by_index(ws), self._by_index(offsets))
        )

    def transition_stats(self, device) -> list[dict]:
        result = self._call("transition_stats", None)
        return [result[i] for i in self.indices]

    def set_transitions(self, starts, transitions) -> None:
        self._call("set_transitions", (self._by_index(starts), self._by_index(transitions)))

    def finalize(self, thetas, ws, offsets) -> list[Any]:
        payload = (self._by_index(thetas), self._by_index(ws), self._by_index(offsets))
        result = self._call("finalize", payload)
        return [pickle.loads(result[i]) for i in self.indices]

    def close(self) -> None:
        for conn in self._conns:
            try:
                conn.send(("stop", None))
            except (BrokenPipeError, OSError):
                pass
        for proc in self._procs:
            proc.join(timeout=30)
            if proc.is_alive():
                proc.terminate()
        self._conns, self._procs = [], []


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
    tol: float = 1e-5,
) -> TiedResult:
    """Fit the members' shared emissions, offsets and own transitions (in
    this process; the members' models are updated in place)."""
    if not members:
        raise ValueError("tied EM needs at least one member")
    result, _ = tied_em_backend(
        LocalMembers(members),
        covariates=[m.covariates for m in members],
        categories=[m.category for m in members],
        free_w=free_w,
        fixed_w=fixed_w,
        rounds=rounds,
        transition_iters=transition_iters,
        shrinkage=shrinkage,
        bound=bound,
        prior_sd=prior_sd,
        device=device,
        tol=tol,
    )
    return result


def tied_em_backend(
    backend,
    *,
    covariates: Sequence[np.ndarray],
    categories: Sequence[str | None],
    free_w: np.ndarray,
    fixed_w: np.ndarray,
    rounds: int,
    transition_iters: int,
    shrinkage: float,
    bound: float,
    prior_sd: float,
    device="cpu",
    label: str = "",
    emission_groups: Sequence[str] | None = None,
    transition_groups: Sequence[str] | None = None,
    tol: float = 1e-5,
) -> tuple[TiedResult, list[Any]]:
    """Tied EM over a members backend (:class:`LocalMembers` or
    :class:`ShardedMembers`); returns the result and the fitted models in
    member order.

    ``emission_groups`` (one key per member; default: all one group) share one
    emission model each, with efficiency offsets fitted within the group.
    ``transition_groups`` (one key per member; default ``None``: every member
    re-estimates its own transitions) share one start / transition model each,
    pooled over its members' expected counts every round -- across emission
    groups too (`HPS-16`).

    ``rounds`` is a cap: the fit stops early once a round's log-likelihood gain
    is below ``tol * max(|log-likelihood|, 1)`` -- the relative rule of
    :meth:`BaseHMM.fit`."""
    import time

    n = len(covariates)
    if not n:
        raise ValueError("tied EM needs at least one member")
    egroups = list(emission_groups) if emission_groups is not None else [""] * n
    if len(egroups) != n or (transition_groups is not None and len(transition_groups) != n):
        raise ValueError("one emission / transition group per member")
    raw = np.array([np.atleast_1d(c).astype(float) for c in covariates])
    if raw.ndim == 1:
        raw = raw[:, None]
    started = time.monotonic()
    tag = f"tied fit {label}: " if label else "tied fit: "

    # Per emission group: members, centred covariates, categories.
    groups: dict[str, dict] = {}
    for key in dict.fromkeys(egroups):
        idx = [i for i, g in enumerate(egroups) if g == key]
        sub = raw[idx]
        centred = sub - sub.mean(axis=0, keepdims=True) if sub.size else np.zeros((len(idx), 0))
        cats = [categories[i] for i in idx]
        names = sorted({c for c in cats if c is not None})
        groups[key] = {
            "idx": idx,
            "centred": centred,
            "constant": [j for j in range(centred.shape[1]) if np.allclose(centred[:, j], 0.0)],
            "names": names,
            "category_index": [names.index(c) if c is not None else None for c in cats],
        }

    # Round 0: each member's own quick fit orients the states and starts each
    # group's shared levels at its members' mean.
    history: list[float] = []
    logits = backend.initial_logits(transition_iters, device)
    for key, g in groups.items():
        g["theta"] = np.mean([logits[i] for i in g["idx"]], axis=0)
        g["w"] = np.array(fixed_w, dtype=float)
        g["beta"] = np.zeros(g["centred"].shape[1])
        g["gamma"] = np.zeros(len(g["names"]))
    offsets = [0.0] * n
    tgroups = list(transition_groups) if transition_groups is not None else None
    pooled: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def member_params():
        thetas = [groups[egroups[i]]["theta"] for i in range(n)]
        ws = [groups[egroups[i]]["w"] for i in range(n)]
        return thetas, ws

    def pool_transitions(iterations: int) -> float:
        loglik = 0.0
        for _ in range(max(1, int(iterations))):
            stats = backend.transition_stats(device)
            loglik = float(sum(s.get("loglik", 0.0) for s in stats))
            for key in dict.fromkeys(tgroups):
                members_of = [i for i, t in enumerate(tgroups) if t == key]
                pooled[key] = pooled_transitions([stats[i] for i in members_of])
            backend.set_transitions(
                [pooled[tgroups[i]][0] for i in range(n)],
                [pooled[tgroups[i]][1] for i in range(n)],
            )
        return loglik

    converged = False
    for round_ in range(rounds + 1):
        thetas, ws = member_params()
        if tgroups is None:
            rows = backend.round(
                thetas,
                ws,
                offsets,
                refit=round_ > 0,
                transition_iters=transition_iters,
                device=device,
            )
            if round_ > 0:
                history.append(float(sum(row[2] for row in rows)))
        else:
            if round_ > 0:
                backend.set_emissions(thetas, ws, offsets)
            loglik = pool_transitions(transition_iters if round_ > 0 else 1)
            if round_ > 0:
                history.append(loglik)
            rows = backend.round(
                thetas, ws, offsets, refit=False, transition_iters=0, device=device
            )
        hits = np.stack([row[0] for row in rows])
        trials = np.stack([row[1] for row in rows])
        for key, g in groups.items():
            idx = g["idx"]
            level = _sigmoid(g["theta"])
            ridge = shrinkage * level * (1.0 - level)
            theta, w, beta, gamma_vec, delta = fit_emission_table(
                hits[idx],
                trials[idx],
                free_w=free_w,
                fixed_w=g["w"],
                covariates=g["centred"],
                categories=g["category_index"],
                n_categories=len(g["names"]),
                ridge=ridge,
                bound=bound,
                prior_sd=prior_sd,
                init_theta=g["theta"],
                init_w=g["w"],
            )
            g.update(theta=theta, w=w, beta=beta, gamma=gamma_vec)
            for position, member in enumerate(idx):
                offsets[member] = float(delta[position])
        logger.info(
            "%sround %d/%d over %d members (%.0fs)%s",
            tag,
            round_,
            rounds,
            len(rows),
            time.monotonic() - started,
            f"; log-likelihood {history[-1]:.1f}" if round_ > 0 else "",
        )
        if len(history) > 1 and abs(history[-1] - history[-2]) < float(tol) * max(
            abs(history[-1]), 1.0
        ):
            converged = True
            break
    logger.info(
        "%s%s after %d round(s)",
        tag,
        "converged" if converged else f"stopped at the {rounds}-round cap (not converged)",
        len(history),
    )
    thetas, ws = member_params()
    models = backend.finalize(thetas, ws, offsets)
    first = groups[egroups[0]]
    return (
        TiedResult(
            theta=first["theta"],
            w=first["w"],
            beta=first["beta"],
            gamma={name: float(first["gamma"][i]) for i, name in enumerate(first["names"])},
            offsets=offsets,
            history=history,
            converged=converged,
            rounds_run=len(history),
            constant_covariates=first["constant"],
            groups={
                key: {
                    "theta": g["theta"],
                    "w": g["w"],
                    "beta": g["beta"],
                    "gamma": {name: float(g["gamma"][i]) for i, name in enumerate(g["names"])},
                    "constant_covariates": g["constant"],
                }
                for key, g in groups.items()
            },
            transitions={key: {"start": v[0], "trans": v[1]} for key, v in pooled.items()},
        ),
        models,
    )

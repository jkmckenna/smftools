"""Versioned memory estimates and live resource decisions for latent analysis."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .memory_guard import PoolBudget, resolve_pool_budget

LATENT_RESOURCE_ESTIMATOR_VERSION = "1"
_MIB = 1024**2
_FLOAT_BYTES = 4


class LatentResourceError(MemoryError):
    """Raised when a required latent operation cannot fit live headroom."""


@dataclass(frozen=True)
class LatentMemoryEstimate:
    """Predicted peak memory for one latent operation."""

    operation: str
    estimator_version: str
    n_reads: int
    n_positions: int
    source_dtype: str
    fit_dtype: str
    breakdown_bytes: dict[str, int]
    predicted_peak_bytes: int

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable estimate."""
        return asdict(self)


@dataclass(frozen=True)
class LatentResourceDecision:
    """One live-headroom decision made before a latent allocation."""

    operation: str
    estimator_version: str
    requested_reads: int
    effective_reads: int
    minimum_reads: int
    n_positions: int
    usable_headroom_bytes: int
    predicted_peak_bytes: int
    limiting_operation: str | None
    pool_budget: dict[str, Any]
    estimate: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable resource decision."""
        return asdict(self)


def resource_envelope_id(envelope) -> str:
    """Return a stable ID for an immutable invocation resource envelope.

    Args:
        envelope: Resource envelope exposing ``as_dict()``.

    Returns:
        SHA-256 identity for the canonical envelope record.
    """
    payload = json.dumps(envelope.as_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _enabled_widths(cfg) -> tuple[int, int, int, int]:
    pca_components = (
        max(1, int(getattr(cfg, "latent_n_pcs", 10)))
        if bool(getattr(cfg, "latent_run_pca_umap", True))
        else 0
    )
    nmf_components = (
        max(1, int(getattr(cfg, "latent_nmf_components", 2)))
        if bool(getattr(cfg, "latent_run_nmf", True))
        else 0
    )
    cp_rank = (
        max(1, int(getattr(cfg, "latent_cp_rank", 2)))
        if bool(getattr(cfg, "latent_run_cp", True))
        else 0
    )
    neighbors = (
        max(2, int(getattr(cfg, "latent_knn_neighbors", 15)))
        if bool(getattr(cfg, "latent_run_pca_umap", True))
        else 0
    )
    return pca_components, nmf_components, cp_rank, neighbors


def estimate_latent_memory(
    cfg,
    operation: str,
    *,
    n_reads: int,
    n_positions: int,
    source_dtype: str = "float32",
    fit_dtype: str = "float32",
) -> LatentMemoryEstimate:
    """Estimate peak bytes for a material latent operation.

    The formulas deliberately favor transparent upper bounds over fitted
    coefficients. They account for the two source layers used by the executor,
    algorithm workspaces, retained output coordinates/loadings, serialization
    staging, and plotting arrays.

    Args:
        cfg: Resolved experiment configuration.
        operation: One of ``fit``, ``transform``, ``cp``, ``result``, ``write``,
            or ``plot``.
        n_reads: Reads allocated by the operation.
        n_positions: Positions allocated by the operation.
        source_dtype: NumPy dtype of materialized source matrices.
        fit_dtype: NumPy dtype used by model-fitting workspaces.

    Returns:
        Versioned byte estimate with a component breakdown.

    Raises:
        ValueError: If ``operation`` is unsupported.
    """
    operation = str(operation)
    if operation not in {"fit", "transform", "cp", "result", "write", "plot"}:
        raise ValueError(f"unsupported latent resource operation: {operation}")
    reads = max(0, int(n_reads))
    positions = max(1, int(n_positions))
    cells = reads * positions
    source_itemsize = int(np.dtype(source_dtype).itemsize)
    fit_itemsize = int(np.dtype(fit_dtype).itemsize)
    pca_components, nmf_components, cp_rank, neighbors = _enabled_widths(cfg)

    source_layers = 2 * cells * source_itemsize
    coordinate_bytes = (
        reads * (pca_components + (2 if pca_components else 0) + nmf_components + 1) * _FLOAT_BYTES
    )
    loading_bytes = positions * (pca_components + nmf_components) * _FLOAT_BYTES
    annotation_bytes = reads * 160 + positions * 64
    retained_result = coordinate_bytes + loading_bytes + annotation_bytes

    breakdown: dict[str, int] = {"fixed_overhead": 4 * _MIB}
    if operation == "fit":
        breakdown["source_layers"] = source_layers
        breakdown["fit_matrix_copies"] = 2 * cells * fit_itemsize
        if pca_components:
            breakdown["pca_workspace"] = (
                3 * cells * fit_itemsize + (reads + positions) * pca_components * 2 * fit_itemsize
            )
            breakdown["umap_workspace"] = (
                reads * neighbors * 24
                + reads * max(1, pca_components) * 2 * _FLOAT_BYTES
                + reads * 2 * _FLOAT_BYTES
            )
        if nmf_components:
            breakdown["nmf_workspace"] = (
                2 * cells * fit_itemsize + (reads + positions) * nmf_components * 4 * fit_itemsize
            )
        breakdown["retained_result"] = retained_result
    elif operation == "transform":
        breakdown["source_layers"] = source_layers
        breakdown["matrix_workspace"] = cells * fit_itemsize
        if pca_components:
            breakdown["pca_umap_workspace"] = (
                reads * (pca_components * 2 + neighbors * 2 + 2) * _FLOAT_BYTES
            )
        if nmf_components:
            breakdown["nmf_workspace"] = (
                cells * fit_itemsize + reads * nmf_components * 3 * fit_itemsize
            )
        breakdown["chunk_result"] = coordinate_bytes
    elif operation == "cp":
        breakdown["one_hot_tensor"] = cells * 4 * fit_itemsize
        breakdown["tensor_workspace"] = cells * 8 * fit_itemsize
        breakdown["factor_workspace"] = 6 * (reads + positions + 4) * cp_rank * 4 * _FLOAT_BYTES
        breakdown["retained_cp_result"] = 6 * (reads + positions) * cp_rank * _FLOAT_BYTES
    elif operation == "result":
        breakdown["result_arrays"] = retained_result
        breakdown["row_index"] = reads * 32
    elif operation == "write":
        staging_reads = min(reads, 1024)
        breakdown["zarr_staging"] = (
            2 * staging_reads * positions * source_itemsize
            + staging_reads * (pca_components + nmf_components + 2) * _FLOAT_BYTES
            + loading_bytes
            + reads * 16
        )
    else:
        color_count = max(
            1,
            1 + len(getattr(cfg, "umap_layers_to_plot", []) or []) + (1 if pca_components else 0),
        )
        embedding_count = (
            (2 if pca_components else 0) + (1 if nmf_components else 0) + (6 if cp_rank else 0)
        )
        breakdown["plot_coordinates"] = reads * max(1, embedding_count) * 2 * 8
        breakdown["plot_colors"] = reads * color_count * 32
        breakdown["plot_loadings"] = positions * (pca_components + nmf_components + 6 * cp_rank) * 8
        breakdown["render_workspace"] = reads * max(1, embedding_count) * 48

    peak = max(1, sum(max(0, int(value)) for value in breakdown.values()))
    return LatentMemoryEstimate(
        operation=operation,
        estimator_version=LATENT_RESOURCE_ESTIMATOR_VERSION,
        n_reads=reads,
        n_positions=positions,
        source_dtype=source_dtype,
        fit_dtype=fit_dtype,
        breakdown_bytes=breakdown,
        predicted_peak_bytes=peak,
    )


def memory_safe_read_count(
    cfg,
    operation: str,
    *,
    requested_reads: int,
    n_positions: int,
    usable_headroom_bytes: int,
    source_dtype: str = "float32",
    fit_dtype: str = "float32",
) -> int:
    """Return the largest requested read count whose estimate fits headroom.

    Args:
        cfg: Resolved experiment configuration.
        operation: Latent operation to size.
        requested_reads: User-bounded read ceiling.
        n_positions: Positions allocated by the operation.
        usable_headroom_bytes: Current memory available to new allocations.
        source_dtype: NumPy dtype of materialized source matrices.
        fit_dtype: NumPy dtype used by model-fitting workspaces.

    Returns:
        Largest count at or below ``requested_reads`` that fits.
    """
    requested = max(0, int(requested_reads))
    headroom = max(0, int(usable_headroom_bytes))
    if requested == 0:
        return 0
    if (
        estimate_latent_memory(
            cfg,
            operation,
            n_reads=requested,
            n_positions=n_positions,
            source_dtype=source_dtype,
            fit_dtype=fit_dtype,
        ).predicted_peak_bytes
        <= headroom
    ):
        return requested
    low, high = 0, requested
    while low < high:
        middle = (low + high + 1) // 2
        estimate = estimate_latent_memory(
            cfg,
            operation,
            n_reads=middle,
            n_positions=n_positions,
            source_dtype=source_dtype,
            fit_dtype=fit_dtype,
        )
        if estimate.predicted_peak_bytes <= headroom:
            low = middle
        else:
            high = middle - 1
    return low


def decide_latent_operation(
    cfg,
    operation: str,
    *,
    requested_reads: int,
    n_positions: int,
    minimum_reads: int,
    pool_budget: PoolBudget,
    source_dtype: str = "float32",
    fit_dtype: str = "float32",
) -> LatentResourceDecision:
    """Resolve an effective read count against one live budget snapshot.

    Args:
        cfg: Resolved experiment configuration.
        operation: Latent operation to size.
        requested_reads: User-bounded read ceiling.
        n_positions: Positions allocated by the operation.
        minimum_reads: Smallest viable count for this operation.
        pool_budget: Point-in-time live memory budget.
        source_dtype: NumPy dtype of materialized source matrices.
        fit_dtype: NumPy dtype used by model-fitting workspaces.

    Returns:
        Persistable requested/effective resource decision.

    Raises:
        LatentResourceError: If the minimum viable count cannot fit.
    """
    requested = max(0, int(requested_reads))
    minimum = max(0, int(minimum_reads))
    safe_reads = memory_safe_read_count(
        cfg,
        operation,
        requested_reads=requested,
        n_positions=n_positions,
        usable_headroom_bytes=pool_budget.usable_headroom_bytes,
        source_dtype=source_dtype,
        fit_dtype=fit_dtype,
    )
    effective = min(requested, safe_reads)
    if effective < minimum:
        minimum_estimate = estimate_latent_memory(
            cfg,
            operation,
            n_reads=minimum,
            n_positions=n_positions,
            source_dtype=source_dtype,
            fit_dtype=fit_dtype,
        )
        raise LatentResourceError(
            f"Latent resource estimator {LATENT_RESOURCE_ESTIMATOR_VERSION} cannot fit "
            f"operation {operation!r} at minimum_reads={minimum}: predicted "
            f"{minimum_estimate.predicted_peak_bytes / _MIB:.1f} MiB exceeds live usable "
            f"headroom {pool_budget.usable_headroom_bytes / _MIB:.1f} MiB."
        )
    estimate = estimate_latent_memory(
        cfg,
        operation,
        n_reads=effective,
        n_positions=n_positions,
        source_dtype=source_dtype,
        fit_dtype=fit_dtype,
    )
    return LatentResourceDecision(
        operation=operation,
        estimator_version=LATENT_RESOURCE_ESTIMATOR_VERSION,
        requested_reads=requested,
        effective_reads=effective,
        minimum_reads=minimum,
        n_positions=max(1, int(n_positions)),
        usable_headroom_bytes=int(pool_budget.usable_headroom_bytes),
        predicted_peak_bytes=estimate.predicted_peak_bytes,
        limiting_operation=operation if effective < requested else None,
        pool_budget=pool_budget.as_dict(),
        estimate=estimate.as_dict(),
    )


def resolve_latent_operation(
    cfg,
    operation: str,
    *,
    requested_reads: int,
    n_positions: int,
    minimum_reads: int,
    source_dtype: str = "float32",
    fit_dtype: str = "float32",
) -> LatentResourceDecision:
    """Snapshot live headroom and resolve one operation before allocation.

    Args:
        cfg: Resolved experiment configuration.
        operation: Latent operation to size.
        requested_reads: User-bounded read ceiling.
        n_positions: Positions allocated by the operation.
        minimum_reads: Smallest viable count for this operation.
        source_dtype: NumPy dtype of materialized source matrices.
        fit_dtype: NumPy dtype used by model-fitting workspaces.

    Returns:
        Persistable requested/effective resource decision.

    Raises:
        LatentResourceError: If the minimum viable count cannot fit.
    """
    budget = resolve_pool_budget(
        cfg,
        1,
        per_item_memory_mb=1.0,
        estimator=f"latent:{LATENT_RESOURCE_ESTIMATOR_VERSION}:{operation}",
    )
    return decide_latent_operation(
        cfg,
        operation,
        requested_reads=requested_reads,
        n_positions=n_positions,
        minimum_reads=minimum_reads,
        pool_budget=budget,
        source_dtype=source_dtype,
        fit_dtype=fit_dtype,
    )

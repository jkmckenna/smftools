"""Per-sample analysis catalog: project-computed, read-span-scoped analyses (currently
periodicity) layered onto per-sample store partitions.

Reuses ``smftools.analysis.compute.autocorrelation`` directly (Tier 2 -- pure,
array-in/result-out, no project-specific knowledge) rather than reimplementing;
this module is purely the project-level plumbing around it: load a partition's
molecules (from the per-sample store's cache, or through the registry +
``materialize()`` for a modern pointer partition), run the requested analysis, and
cache the result keyed by a hash of its full definition (layer, window, method,
LS parameters) so a different read-span definition never collides with -- or gets
silently served by -- a stale one.

``join_periodicity`` is the read side: it attaches an already-computed analysis
(never computes one itself) onto a materialized selection by read_id, the mechanism
``dev/project_sample_and_set_stores.md`` calls the set store's per-sample catalog
join. It's a separate, explicit step after ``set_store.materialize_set()`` rather
than automatic inside it, since which analysis/definition is relevant is caller
knowledge ``materialize_set`` has no way to guess.

Phase 4 (partial -- periodicity only, no embeddings yet) of the design doc.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .sample_store import list_per_sample_partitions, load_per_sample_partition, partition_dir_for

ANALYSES_DIRNAME = "analyses"
RESULT_FILENAME = "result.parquet"
DEFINITION_FILENAME = "definition.json"
PERIODICITY_ANALYSIS_NAME = "periodicity"
# Array-valued columns compute_single_molecule_periodicity_direct returns -- not
# parquet-safe as raw object-dtype numpy arrays, and not useful in a cached summary
# table; see that function's own docstring ("drop before saving").
_DROP_BEFORE_CACHE = ("ls_freqs", "ls_power")


def _definition_hash(definition: dict) -> str:
    encoded = json.dumps(definition, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _analysis_dir(
    project_dir: str | Path,
    experiment_id: str,
    reference_strand: str,
    sample: str,
    analysis_name: str,
    definition_hash: str,
) -> Path:
    return (
        partition_dir_for(project_dir, experiment_id, reference_strand, sample)
        / ANALYSES_DIRNAME
        / analysis_name
        / definition_hash
    )


def _periodicity_definition(*, layer, start, end, method, kwargs) -> dict:
    return {
        "analysis": PERIODICITY_ANALYSIS_NAME,
        "layer": layer,
        "start": start,
        "end": end,
        "method": method,
        **kwargs,
    }


def _load_partition_adata(project_dir, experiment_id: str, reference_strand: str, sample: str):
    """Load one per-sample-store partition's molecules, cache or pointer alike."""
    partitions = list_per_sample_partitions(project_dir, experiment_id)
    match = next(
        (p for p in partitions if p["reference_strand"] == reference_strand and p["sample"] == sample),
        None,
    )
    if match is None:
        raise FileNotFoundError(
            f"no per-sample store entry for {experiment_id!r}/{reference_strand!r}/{sample!r} "
            "-- run project add (or backfill_per_sample_store) first"
        )
    if match["kind"] == "cache":
        return load_per_sample_partition(project_dir, experiment_id, reference_strand, sample)

    from ..informatics.partition_read import materialize
    from .catalog import ProjectCatalog
    from .registry import resolve_experiment_spine

    catalog = ProjectCatalog.open(project_dir)
    entry = next((e for e in catalog.experiments() if e["id"] == experiment_id), None)
    if entry is None:
        raise FileNotFoundError(f"experiment {experiment_id!r} not found in project registry")
    resolved = resolve_experiment_spine(entry)
    if resolved is None:
        raise FileNotFoundError(f"no spine available for experiment {experiment_id!r}")
    _, spine_path = resolved
    return materialize(spine_path, references=[reference_strand], samples=[sample])


def compute_periodicity(
    project_dir: str | Path,
    experiment_id: str,
    reference_strand: str,
    sample: str,
    *,
    layer: str | None = None,
    start: int | None = None,
    end: int | None = None,
    method: str = "direct",
    force_recompute: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """Per-read Lomb-Scargle periodicity for one per-sample-store partition, cached.

    Wraps ``smftools.analysis.compute.autocorrelation.compute_single_molecule_periodicity_direct``
    (``method="direct"``, default -- more reliable for sparse single-molecule data
    per that function's own docstring) or ``compute_single_molecule_periodicity``
    (any other ``method``, ACF-intermediate). ``layer`` selects which ``adata.layers``
    matrix to run on;
    ``None`` uses ``adata.X``. ``start``/``end`` restrict to a genomic window first.
    Remaining ``**kwargs`` (``min_col_coverage``, ``min_row_coverage``,
    ``nrl_search_bp``, ``period_range_bp``, ``poly_degree``, ``min_sites``, ...) pass
    straight through to the underlying compute function -- see its docstring.

    Cached under this partition's ``analyses/periodicity/<definition_hash>/``, keyed
    by every parameter that changes the result, so a different read-span/analysis
    definition never collides with -- or is silently served by -- an old cache entry.

    Returns a DataFrame with one row per surviving read, indexed by ``read_id`` (not
    the positional ``row_index`` the underlying compute function returns), ready to
    join onto a materialized selection by read_id -- see :func:`join_periodicity`.
    """
    from ..analysis.compute import autocorrelation

    definition = _periodicity_definition(layer=layer, start=start, end=end, method=method, kwargs=kwargs)
    definition_hash = _definition_hash(definition)
    analysis_dir = _analysis_dir(
        project_dir, experiment_id, reference_strand, sample, PERIODICITY_ANALYSIS_NAME, definition_hash
    )
    result_path = analysis_dir / RESULT_FILENAME
    if not force_recompute and result_path.exists():
        return pd.read_parquet(result_path).set_index("read_id")

    adata = _load_partition_adata(project_dir, experiment_id, reference_strand, sample)
    positions = np.asarray(adata.var_names, dtype=np.int64)
    if start is not None or end is not None:
        window = np.ones(positions.shape[0], dtype=bool)
        if start is not None:
            window &= positions >= int(start)
        if end is not None:
            window &= positions < int(end)
        adata = adata[:, window]
        positions = positions[window]

    matrix_source = adata.layers[layer] if layer is not None else adata.X
    mat = np.asarray(matrix_source, dtype=np.float64)

    compute_fn = (
        autocorrelation.compute_single_molecule_periodicity_direct
        if method == "direct"
        else autocorrelation.compute_single_molecule_periodicity
    )
    result = compute_fn(mat, positions, **kwargs)
    read_ids = np.asarray(adata.obs_names)[result["row_index"].to_numpy()]
    result = result.drop(columns=[c for c in _DROP_BEFORE_CACHE if c in result.columns])
    result = result.drop(columns="row_index")
    result.insert(0, "read_id", read_ids)

    analysis_dir.mkdir(parents=True, exist_ok=True)
    result.to_parquet(result_path)
    (analysis_dir / DEFINITION_FILENAME).write_text(
        json.dumps(definition, indent=2, sort_keys=True, default=str)
    )
    return result.set_index("read_id")


def join_periodicity(
    adata,
    project_dir: str | Path,
    *,
    layer: str | None = None,
    start: int | None = None,
    end: int | None = None,
    method: str = "direct",
    ref_column: str = "Reference_strand",
    sample_column: str = "Sample",
    **kwargs,
):
    """Attach an already-computed periodicity result onto ``adata`` by read_id.

    Never computes anything -- looks up, for every ``(experiment, Reference_strand,
    sample)`` combination present in ``adata.obs``, the per-sample analysis cache for
    the exact definition given here (same parameters as :func:`compute_periodicity`;
    must already have been computed via that function). Reads with no matching cached
    result (partition never analyzed for this definition, or the read didn't survive
    that analysis's coverage filtering) get NaN. Adds one ``periodicity_<column>`` obs
    column per result column; returns ``adata`` unchanged if it lacks ``experiment``/
    ``Reference_strand``/``Sample`` obs columns (nothing to key the join on).
    """
    if (
        "experiment" not in adata.obs
        or ref_column not in adata.obs
        or sample_column not in adata.obs
    ):
        return adata

    definition = _periodicity_definition(layer=layer, start=start, end=end, method=method, kwargs=kwargs)
    definition_hash = _definition_hash(definition)

    frames = []
    combos = adata.obs[["experiment", ref_column, sample_column]].drop_duplicates()
    for _, row in combos.iterrows():
        analysis_dir = _analysis_dir(
            project_dir,
            str(row["experiment"]),
            str(row[ref_column]),
            str(row[sample_column]),
            PERIODICITY_ANALYSIS_NAME,
            definition_hash,
        )
        result_path = analysis_dir / RESULT_FILENAME
        if result_path.exists():
            frames.append(pd.read_parquet(result_path))

    if not frames:
        return adata

    combined = pd.concat(frames, ignore_index=True).drop_duplicates(subset="read_id", keep="first")
    combined = combined.set_index("read_id")
    aligned = combined.reindex(adata.obs_names)
    for column in aligned.columns:
        adata.obs[f"periodicity_{column}"] = aligned[column].to_numpy()
    return adata

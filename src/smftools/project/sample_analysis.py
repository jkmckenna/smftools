"""Per-sample analysis catalog: project-computed, read-span-scoped analyses (currently
periodicity) layered onto per-sample store partitions.

Reuses ``smftools.analysis.compute.autocorrelation`` directly (Tier 2 -- pure,
array-in/result-out, no project-specific knowledge) rather than reimplementing;
this module is purely the project-level plumbing around it: load a partition's
molecules (from the per-sample store's cache, or through the registry +
``materialize()`` for a modern pointer partition), run the requested analysis, and
cache the result keyed by a hash of its full definition (layer, window, method,
LS parameters, analysis algorithm version, and semantic graph version) so a
different read-span or code definition never collides with -- or gets silently
served by -- a stale one.

``join_periodicity`` is the read side: it attaches an already-computed analysis
(never computes one itself) onto a materialized selection by molecule UID, the mechanism
``dev/plans/completed/project_sample_and_set_stores.md`` calls the set store's per-sample catalog
join. It's a separate, explicit step after pooling a set (e.g. via
``catalog.project_adata`` or streaming ``set_store.iter_set_parts``) rather than
automatic inside it, since which analysis/definition is relevant is caller knowledge
the set materialization has no way to guess.

Phase 4 (partial -- periodicity only, no embeddings yet) of the design doc.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..constants import SEMANTIC_GRAPH_DEFINITION_VERSION
from ..informatics.molecule_identity import (
    EXPERIMENT_UID_COLUMN,
    MOLECULE_UID_COLUMN,
    READ_ID_COLUMN,
    molecule_uid,
)
from .sample_store import list_per_sample_partitions, load_per_sample_partition, partition_dir_for

ANALYSES_DIRNAME = "analyses"
RESULT_FILENAME = "result.parquet"
DEFINITION_FILENAME = "definition.json"
PERIODICITY_ANALYSIS_NAME = "periodicity"
PERIODICITY_CACHE_SCHEMA_VERSION = 2
PERIODICITY_ALGORITHM_VERSION = "1"
SCHEMA_VERSION_COLUMN = "schema_version"
_CACHE_IDENTITY_COLUMNS = (
    SCHEMA_VERSION_COLUMN,
    EXPERIMENT_UID_COLUMN,
    MOLECULE_UID_COLUMN,
    READ_ID_COLUMN,
)
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


def _periodicity_definition(*, layer, start, end, method, kwargs) -> dict[str, object]:
    return {
        "analysis": PERIODICITY_ANALYSIS_NAME,
        "algorithm_version": PERIODICITY_ALGORITHM_VERSION,
        "graph_definition_version": SEMANTIC_GRAPH_DEFINITION_VERSION,
        "layer": layer,
        "start": start,
        "end": end,
        "method": method,
        **kwargs,
    }


def _registered_experiment_uid(project_dir: str | Path, experiment_id: str) -> str:
    """Return the persistent identity for one registered experiment."""
    from .registry import list_experiments

    entry = next(
        (entry for entry in list_experiments(project_dir) if entry["id"] == experiment_id),
        None,
    )
    if entry is None:
        raise FileNotFoundError(
            f"experiment {experiment_id!r} is not registered in this project; "
            "run project add before computing project analyses"
        )
    return str(entry[EXPERIMENT_UID_COLUMN])


def _read_ids(adata) -> np.ndarray:
    """Return explicit instrument read IDs for one experiment-local AnnData."""
    values = (
        adata.obs[READ_ID_COLUMN].astype(str)
        if READ_ID_COLUMN in adata.obs
        else adata.obs.index.to_series().astype(str)
    )
    if values.duplicated().any():
        raise ValueError("periodicity input contains duplicate instrument read IDs")
    return values.to_numpy()


def _load_cached_result(
    result_path: Path,
    *,
    experiment_uid: str,
    migrate_legacy: bool = True,
) -> pd.DataFrame:
    """Load and validate one cache, migrating an unambiguously owned v1 result."""
    result = pd.read_parquet(result_path)
    if READ_ID_COLUMN not in result:
        raise ValueError(
            f"periodicity cache {result_path} has no {READ_ID_COLUMN!r} column; "
            "rerun with force_recompute=True"
        )

    result[READ_ID_COLUMN] = result[READ_ID_COLUMN].astype(str)
    if result[READ_ID_COLUMN].duplicated().any():
        raise ValueError(
            f"periodicity cache {result_path} contains duplicate instrument read IDs; "
            "rerun with force_recompute=True"
        )
    has_complete_identity = all(column in result for column in _CACHE_IDENTITY_COLUMNS)
    if not has_complete_identity:
        identity_columns_present = [
            column
            for column in _CACHE_IDENTITY_COLUMNS
            if column != READ_ID_COLUMN and column in result
        ]
        if identity_columns_present:
            raise ValueError(
                f"periodicity cache {result_path} has incomplete project identity columns "
                f"{identity_columns_present}; rerun with force_recompute=True"
            )
        if not migrate_legacy:
            raise ValueError(
                f"periodicity cache {result_path} predates project molecule identity; "
                "rerun with force_recompute=True"
            )
        # The cache directory belongs to exactly one registered experiment, so its
        # legacy read IDs can be stamped without inferring ownership from pooled names.
        result[SCHEMA_VERSION_COLUMN] = PERIODICITY_CACHE_SCHEMA_VERSION
        result[EXPERIMENT_UID_COLUMN] = experiment_uid
        result[MOLECULE_UID_COLUMN] = [
            molecule_uid(experiment_uid, read_id) for read_id in result[READ_ID_COLUMN]
        ]
        result.to_parquet(result_path, index=False)
        definition_path = result_path.with_name(DEFINITION_FILENAME)
        if definition_path.exists():
            definition = json.loads(definition_path.read_text(encoding="utf-8"))
            definition["cache_schema_version"] = PERIODICITY_CACHE_SCHEMA_VERSION
            definition_path.write_text(
                json.dumps(definition, indent=2, sort_keys=True, default=str),
                encoding="utf-8",
            )
        return result

    versions = set(result[SCHEMA_VERSION_COLUMN].dropna().astype(int))
    if not result.empty and versions != {PERIODICITY_CACHE_SCHEMA_VERSION}:
        raise ValueError(
            f"periodicity cache {result_path} has unsupported schema version(s) "
            f"{sorted(versions)}; rerun with force_recompute=True"
        )
    owners = set(result[EXPERIMENT_UID_COLUMN].dropna().astype(str))
    if not result.empty and owners != {experiment_uid}:
        raise ValueError(
            f"periodicity cache {result_path} belongs to experiment UID(s) "
            f"{sorted(owners)}, not {experiment_uid!r}; rerun with force_recompute=True"
        )
    expected_molecule_uids = [
        molecule_uid(experiment_uid, read_id) for read_id in result[READ_ID_COLUMN]
    ]
    if result[MOLECULE_UID_COLUMN].astype(str).tolist() != expected_molecule_uids:
        raise ValueError(
            f"periodicity cache {result_path} contains inconsistent molecule identities; "
            "rerun with force_recompute=True"
        )
    return result


def _load_partition_adata(project_dir, experiment_id: str, reference_strand: str, sample: str):
    """Load one per-sample-store partition's molecules, cache or pointer alike."""
    partitions = list_per_sample_partitions(project_dir, experiment_id)
    match = next(
        (
            p
            for p in partitions
            if p["reference_strand"] == reference_strand and p["sample"] == sample
        ),
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
    by every parameter and code-identity version that changes the result, so a
    different read-span/analysis definition never collides with -- or is silently
    served by -- an old cache entry.

    Returns a DataFrame with one row per surviving read, indexed by ``molecule_uid``
    (not the positional ``row_index`` the underlying compute function returns).
    ``experiment_uid`` and the original instrument ``read_id`` remain explicit
    columns for traceability.
    """
    from ..analysis.compute import autocorrelation

    definition = _periodicity_definition(
        layer=layer, start=start, end=end, method=method, kwargs=kwargs
    )
    definition_hash = _definition_hash(definition)
    analysis_dir = _analysis_dir(
        project_dir,
        experiment_id,
        reference_strand,
        sample,
        PERIODICITY_ANALYSIS_NAME,
        definition_hash,
    )
    result_path = analysis_dir / RESULT_FILENAME
    experiment_uid = _registered_experiment_uid(project_dir, experiment_id)
    if not force_recompute and result_path.exists():
        return _load_cached_result(
            result_path,
            experiment_uid=experiment_uid,
        ).set_index(MOLECULE_UID_COLUMN)

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
    input_read_ids = _read_ids(adata)
    # A partition where no read clears the coverage and site thresholds is an
    # ordinary outcome -- a small sample, or a narrow window -- and yields an
    # empty frame whose `row_index` column carries no dtype. Index with an
    # explicit integer array so that case produces an empty result rather than
    # failing the whole analysis on a numpy indexing error.
    row_index = np.asarray(result["row_index"].to_numpy(), dtype=np.int64)
    read_ids = input_read_ids[row_index]
    result = result.drop(columns=[c for c in _DROP_BEFORE_CACHE if c in result.columns])
    result = result.drop(columns="row_index")
    result.insert(0, READ_ID_COLUMN, read_ids)
    result.insert(
        0,
        MOLECULE_UID_COLUMN,
        [molecule_uid(experiment_uid, read_id) for read_id in read_ids],
    )
    result.insert(0, EXPERIMENT_UID_COLUMN, experiment_uid)
    result.insert(0, SCHEMA_VERSION_COLUMN, PERIODICITY_CACHE_SCHEMA_VERSION)

    analysis_dir.mkdir(parents=True, exist_ok=True)
    result.to_parquet(result_path, index=False)
    persisted_definition = {
        "cache_schema_version": PERIODICITY_CACHE_SCHEMA_VERSION,
        **definition,
    }
    (analysis_dir / DEFINITION_FILENAME).write_text(
        json.dumps(persisted_definition, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return result.set_index(MOLECULE_UID_COLUMN)


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
    """Attach an already-computed periodicity result onto ``adata`` by molecule UID.

    Never computes anything -- looks up, for every ``(experiment, Reference_strand,
    sample)`` combination present in ``adata.obs``, the per-sample analysis cache for
    the exact definition given here (same parameters as :func:`compute_periodicity`;
    must already have been computed via that function). Reads with no matching cached
    result (partition never analyzed for this definition, or the read didn't survive
    that analysis's coverage filtering) get NaN. Adds one ``periodicity_<column>`` obs
    column per result column; returns ``adata`` unchanged if it lacks ``experiment``/
    ``Reference_strand``/``Sample`` obs columns (nothing to key the join on).
    ``molecule_uid`` must be present explicitly, or derivable from explicit
    ``experiment_uid`` and ``read_id`` columns; pooled observation names are never
    decoded to infer identity.
    """
    if (
        "experiment" not in adata.obs
        or ref_column not in adata.obs
        or sample_column not in adata.obs
    ):
        return adata
    definition = _periodicity_definition(
        layer=layer, start=start, end=end, method=method, kwargs=kwargs
    )
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
            experiment_uid = _registered_experiment_uid(project_dir, str(row["experiment"]))
            frames.append(
                _load_cached_result(
                    result_path,
                    experiment_uid=experiment_uid,
                )
            )

    if not frames:
        return adata

    if MOLECULE_UID_COLUMN in adata.obs:
        target_molecule_uids = adata.obs[MOLECULE_UID_COLUMN].astype(str).to_numpy()
    elif EXPERIMENT_UID_COLUMN in adata.obs and READ_ID_COLUMN in adata.obs:
        target_molecule_uids = np.asarray(
            [
                molecule_uid(experiment_uid, read_id)
                for experiment_uid, read_id in zip(
                    adata.obs[EXPERIMENT_UID_COLUMN],
                    adata.obs[READ_ID_COLUMN],
                )
            ],
            dtype=object,
        )
    else:
        raise ValueError(
            "join_periodicity requires obs['molecule_uid'] or both "
            "obs['experiment_uid'] and obs['read_id']; pooled observation names "
            "are not an identity source"
        )

    combined = pd.concat(frames, ignore_index=True)
    if combined[MOLECULE_UID_COLUMN].duplicated().any():
        duplicates = sorted(
            combined.loc[
                combined[MOLECULE_UID_COLUMN].duplicated(keep=False),
                MOLECULE_UID_COLUMN,
            ]
            .astype(str)
            .unique()
        )
        raise ValueError(f"periodicity caches contain duplicate molecule UIDs: {duplicates[:5]}")
    combined = combined.set_index(MOLECULE_UID_COLUMN)
    aligned = combined.reindex(target_molecule_uids)
    result_columns = [column for column in aligned.columns if column not in _CACHE_IDENTITY_COLUMNS]
    for column in result_columns:
        adata.obs[f"periodicity_{column}"] = aligned[column].to_numpy()
    return adata

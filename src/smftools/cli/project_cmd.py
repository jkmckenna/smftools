"""CLI logic for the project-level cross-experiment catalog."""

from __future__ import annotations

from pathlib import Path

from smftools.logging_utils import get_logger

logger = get_logger(__name__)


def project_init(project_dir: str | Path, *, name: str | None = None) -> tuple[Path, list[Path]]:
    """Initialize a project directory + empty registry, plus starter docs/dirs.

    Returns ``(registry_path, scaffolded_paths)``. ``scaffolded_paths`` covers the
    README/AGENTS/CLAUDE/PLAN/project.yaml starter files and the project_scripts/
    project_outputs working directories (see ``project.scaffold``) -- only the
    ones actually created this call, since re-running ``init`` never overwrites
    existing files.
    """
    from ..project.registry import init_project
    from ..project.scaffold import scaffold_project

    registry_path = init_project(project_dir)
    scaffolded = scaffold_project(project_dir, name=name)
    return registry_path, scaffolded


def project_add(
    project_dir: str | Path,
    experiment_dir: str | Path,
    *,
    experiment_id: str | None = None,
    name: str | None = None,
    stage: str | None = None,
) -> tuple[str, dict, list[str]]:
    """Register an experiment and return ``(id, entry, reference_conflicts)``.

    ``stage`` only applies when ``experiment_dir`` is a legacy monolithic
    ``.h5ad``/``.h5ad.gz`` file (see :func:`smftools.project.registry.add_experiment`);
    it names which pipeline stage that file represents, overriding the
    filename-based guess. Ignored for directory-based registration, since that
    path discovers every stage automatically.

    Also backfills the project's per-sample store (see
    ``smftools.project.sample_store`` and ``dev/plans/completed/project_sample_and_set_stores.md``)
    from this experiment's most-derived available stage: modern (partitioned-store)
    experiments get a pointer catalog only, legacy experiments get their molecules
    cached once (their only read path is a full eager load, so repeating it on every
    later query would be wasteful).
    """
    from ..project.catalog import ProjectCatalog
    from ..project.reference_registry import detect_reference_conflicts
    from ..project.registry import add_experiment, resolve_experiment_spine
    from ..project.sample_store import backfill_per_sample_store

    exp_id, entry = add_experiment(
        project_dir, experiment_dir, experiment_id=experiment_id, name=name, stage=stage
    )

    catalog = ProjectCatalog.open(project_dir)
    resolved_entry = next((e for e in catalog.experiments() if e["id"] == exp_id), None)
    if resolved_entry is not None:
        resolved = resolve_experiment_spine(resolved_entry)
        if resolved is not None:
            _, spine_path = resolved
            backfill_per_sample_store(project_dir, exp_id, spine_path)

    conflicts = detect_reference_conflicts(catalog.references())
    for warning in conflicts:
        logger.warning("reference conflict: %s", warning)
    return exp_id, entry, conflicts


def project_remove(project_dir: str | Path, experiment_id: str) -> None:
    """Mark an experiment inactive in the project."""
    from ..project.registry import remove_experiment

    remove_experiment(project_dir, experiment_id)


def project_list(project_dir: str | Path):
    """Return ``(experiments, harmonized_references)`` for display.

    Each experiment carries ``locality`` (``reachable``/``offline``/``missing``)
    and, when detached, ``locality_volume``. A project references run directories
    it does not own, so a registered experiment is not necessarily a readable one,
    and listing them identically hides which half of the project can answer
    (`PSR-18`).
    """
    from ..project.catalog import ProjectCatalog
    from ..project.locality import resolve_experiment_locality

    catalog = ProjectCatalog.open(project_dir)
    experiments = []
    for entry in catalog.experiments():
        locality = resolve_experiment_locality(entry["id"], entry["path"])
        experiments.append(
            {
                **entry,
                "locality": locality.state,
                "locality_volume": str(locality.volume) if locality.volume else None,
            }
        )
    return experiments, catalog.references()


def project_add_set(
    project_dir: str | Path,
    name: str,
    *,
    experiments: list[str] | None = None,
    query: str | None = None,
    allow_unresolved: bool = False,
):
    """Define a named experiment set and return its resolved membership.

    Args:
        project_dir: Project root.
        name: Set name, as later passed to ``--set``.
        experiments: Explicit membership. Mutually exclusive with *query*.
        query: Saved SQL predicate over the harmonized ``refs`` table. Mutually
            exclusive with *experiments*.
        allow_unresolved: Define the set even when it names an experiment that is
            not registered, is deactivated, or is repeated. Useful when the set
            is written before the experiments it names are registered.

    Returns:
        The stored :class:`~smftools.project.registry.SetMembership`.
    """
    from ..project.registry import add_set, resolve_set_membership

    add_set(
        project_dir,
        name,
        experiments=experiments,
        query=query,
        validate=not allow_unresolved,
    )
    return resolve_set_membership(project_dir, name)


def project_list_sets(project_dir: str | Path) -> list[dict]:
    """Return one display record per named set, most useful fields first.

    A query set is described but not executed here, so listing stays cheap and
    does not require DuckDB; use :func:`project_show_set` to resolve one.
    """
    from ..project.registry import list_sets

    records = []
    for name, definition in sorted(list_sets(project_dir).items()):
        kind = str(definition.get("kind", ""))
        records.append(
            {
                "name": name,
                "kind": kind,
                "n_declared": len(definition.get("experiments", ())) if kind == "list" else None,
                "query": str(definition.get("sql", "")) if kind == "query" else None,
            }
        )
    return records


def project_show_set(project_dir: str | Path, name: str):
    """Return one named set's resolved membership.

    Returns:
        The :class:`~smftools.project.registry.SetMembership` that every ``--set``
        consumer resolves to.
    """
    from ..project.registry import resolve_set_membership

    return resolve_set_membership(project_dir, name)


def project_remove_set(project_dir: str | Path, name: str) -> None:
    """Delete a named set. No experiment registration is affected."""
    from ..project.registry import remove_set

    remove_set(project_dir, name)


def project_plan(
    project_dir: str | Path,
    target: str,
    canonical_reference: str,
    *,
    set_name: str | None = None,
    modality: str | None = None,
    experiments=None,
    stage: str | None = None,
    start: int | None = None,
    end: int | None = None,
    layers: list[str] | None = None,
    read_metrics: bool = False,
    partitioned: bool = False,
):
    """Build a read-only semantic plan for a project analysis target."""
    from ..pipeline.project_graph import build_project_plan
    from ..project.catalog import ProjectCatalog

    project_dir = Path(project_dir)
    catalog = ProjectCatalog.open(project_dir)
    request = {
        "project_identity": catalog.registry.get("project_uid", project_dir.name),
        "canonical_reference": canonical_reference,
        "set_name": set_name,
        "modality": modality,
        "experiments": list(experiments) if experiments else None,
        "stage": stage,
        "layers": layers,
        "start": start,
        "end": end,
        "read_metrics": read_metrics,
        "partitioned": partitioned,
    }
    return build_project_plan(project_dir, target, request)


def project_upgrade_impact(
    project_dir: str | Path,
    target: str,
    canonical_reference: str,
    *,
    set_name: str | None = None,
    modality: str | None = None,
    experiments=None,
    stage: str | None = None,
    start: int | None = None,
    end: int | None = None,
    layers: list[str] | None = None,
    read_metrics: bool = False,
    partitioned: bool = False,
):
    """Build grouped installed-code impact without adding compatibility rules."""
    from ..pipeline.upgrade_impact import build_upgrade_impact

    plan = project_plan(
        project_dir,
        target,
        canonical_reference,
        set_name=set_name,
        modality=modality,
        experiments=experiments,
        stage=stage,
        start=start,
        end=end,
        layers=layers,
        read_metrics=read_metrics,
        partitioned=partitioned,
    )
    return build_upgrade_impact(plan, scope="project")


def project_materialize(
    project_dir: str | Path,
    canonical_reference: str,
    output_path: str | Path,
    *,
    set_name: str | None = None,
    modality: str | None = None,
    experiments=None,
    stage: str | None = None,
    start: int | None = None,
    end: int | None = None,
    layers: list[str] | None = None,
    read_metrics: bool = False,
    allow_large: bool = False,
    partitioned: bool = False,
    max_memory_gb: float | None = None,
    max_memory_percent: float | None = 60.0,
) -> Path:
    """Pool a canonical reference across matching experiments into one AnnData and write it.

    This is the explicit "give me one pooled object" path (``project.catalog.project_adata``).
    ``experiments`` restricts the pool to explicit experiment IDs, intersected
    with ``set_name`` when both are given.
    ``stage`` picks a genomic pipeline stage per experiment (``raw``,
    ``preprocess``, ``spatial``, ``hmm``, ...); the default falls back through
    the most-derived stage available per experiment. Latent task coordinates
    require :func:`project_export_latent` because their owners cannot be pooled.

    Prefer a narrow ``layers`` subset and/or a ``start``/``end`` window. Pooled output
    is preflighted before allocation. ``allow_large`` acknowledges the soft 8-GiB
    warning but never bypasses the resolved hard memory ceiling. ``partitioned`` writes
    independently readable Zarr parts and avoids a final in-memory concatenation.
    """
    from ..project.catalog import export_project_partitions, project_adata
    from ..readwrite import safe_write_h5ad

    if partitioned:
        output_path = export_project_partitions(
            project_dir,
            canonical_reference,
            output_path,
            set_name=set_name,
            modality=modality,
            experiments=experiments,
            stage=stage,
            start=start,
            end=end,
            layers=layers,
            read_metrics=read_metrics,
            max_memory_gb=max_memory_gb,
            max_memory_percent=max_memory_percent,
        )
        logger.info("Wrote partitioned project materialization -> %s", output_path)
        return output_path
    adata = project_adata(
        project_dir,
        canonical_reference,
        set_name=set_name,
        modality=modality,
        experiments=experiments,
        stage=stage,
        start=start,
        end=end,
        layers=layers,
        read_metrics=read_metrics,
        allow_large=allow_large,
        max_memory_gb=max_memory_gb,
        max_memory_percent=max_memory_percent,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    safe_write_h5ad(adata, output_path, backup=False, verbose=False)
    logger.info(
        "Wrote %d molecules from %d experiment(s) -> %s",
        adata.n_obs,
        adata.obs["experiment"].nunique() if "experiment" in adata.obs else 1,
        output_path,
    )
    return output_path


def project_export_latent(
    project_dir: str | Path,
    output_dir: str | Path,
    *,
    canonical_reference: str | None = None,
    experiments=None,
    set_name: str | None = None,
    molecule_uids=None,
    analysis_core_ids=None,
    representations=None,
    labels=None,
) -> Path:
    """Export one portable task-local artifact per latent coordinate owner."""
    from ..project.catalog import ProjectCatalog
    from ..project.latent_store import export_latent_parts

    catalog = ProjectCatalog.open(project_dir)
    output = export_latent_parts(
        catalog,
        output_dir,
        canonical_reference=canonical_reference,
        experiments=experiments,
        set_name=set_name,
        molecule_uids=molecule_uids,
        analysis_core_ids=analysis_core_ids,
        representations=representations,
        labels=labels,
    )
    logger.info("Wrote scoped latent project export -> %s", output)
    return output


SAMPLE_ANALYSIS_SCHEMA_VERSION = 1


def project_sample_analysis_partitions(
    project_dir: str | Path,
    canonical_reference: str,
    *,
    set_name: str | None = None,
    modality: str | None = None,
    experiments=None,
    stage: str | None = None,
) -> list[dict]:
    """Resolve which per-sample-store partitions one selection would analyze.

    Sample analysis is per ``(experiment, reference_strand, sample)``, one scope
    finer than materialization's per-experiment selection, so the selection is
    resolved to experiments first and then expanded through the per-sample store.

    Args:
        project_dir: Project root.
        canonical_reference: Harmonized reference to analyze.
        set_name: Optional named set restricting the experiments.
        modality: Optional modality restriction.
        experiments: Optional explicit experiment IDs.
        stage: Optional pipeline stage for spine resolution.

    Returns:
        One record per partition, sorted, each with ``experiment``,
        ``reference_strand``, ``sample``, and the store ``kind``.
    """
    from ..project.catalog import ProjectCatalog, resolve_set_members
    from ..project.sample_store import list_per_sample_partitions

    catalog = ProjectCatalog.open(project_dir)
    members = resolve_set_members(
        catalog,
        canonical_reference,
        set_name=set_name,
        modality=modality,
        experiments=experiments,
        stage=stage,
    )
    records = []
    for member in members:
        wanted = set(member["reference_strands"])
        for partition in list_per_sample_partitions(project_dir, member["experiment"]):
            if str(partition["reference_strand"]) not in wanted:
                continue
            records.append(
                {
                    "experiment": str(member["experiment"]),
                    "experiment_uid": str(member["experiment_uid"]),
                    "reference_strand": str(partition["reference_strand"]),
                    "sample": str(partition["sample"]),
                    "kind": str(partition["kind"]),
                }
            )
    return sorted(
        records, key=lambda item: (item["experiment"], item["reference_strand"], item["sample"])
    )


def project_sample_analysis(
    project_dir: str | Path,
    canonical_reference: str,
    output_path: str | Path,
    *,
    set_name: str | None = None,
    modality: str | None = None,
    experiments=None,
    stage: str | None = None,
    layer: str | None = None,
    start: int | None = None,
    end: int | None = None,
    method: str = "direct",
    force_recompute: bool = False,
) -> dict:
    """Run per-sample periodicity across a project selection into one task-local table.

    Each partition's result is computed and cached in the project's per-sample
    store under its own definition hash (see
    :func:`smftools.project.sample_analysis.compute_periodicity`), so re-running
    this reuses those caches. What lands in *output_path* is the joined,
    task-local product: one row per read, with the partition identity kept
    explicit so rows from different experiments never become
    indistinguishable.

    Args:
        project_dir: Project root.
        canonical_reference: Harmonized reference to analyze.
        output_path: Task-local parquet path to write.
        set_name: Optional named set restricting the experiments.
        modality: Optional modality restriction.
        experiments: Optional explicit experiment IDs.
        stage: Optional pipeline stage for spine resolution.
        layer: Layer to analyze; ``None`` uses ``X``.
        start: Optional genomic window start.
        end: Optional genomic window end.
        method: Periodicity method passed through to the compute function.
        force_recompute: Recompute each partition instead of reading its cache.

    Returns:
        A summary with the schema version, the analyzed partitions, and row counts.

    Raises:
        ValueError: The selection matched no per-sample-store partition.
    """
    import pandas as pd

    from ..project.sample_analysis import compute_periodicity

    partitions = project_sample_analysis_partitions(
        project_dir,
        canonical_reference,
        set_name=set_name,
        modality=modality,
        experiments=experiments,
        stage=stage,
    )
    if not partitions:
        raise ValueError(
            f"no per-sample-store partition matches {canonical_reference!r} for this selection; "
            "run `smftools project add` so the per-sample store is cataloged"
        )
    output_path = Path(output_path)
    frames = []
    analyzed = []
    for partition in partitions:
        result = compute_periodicity(
            project_dir,
            partition["experiment"],
            partition["reference_strand"],
            partition["sample"],
            layer=layer,
            start=start,
            end=end,
            method=method,
            force_recompute=force_recompute,
        )
        frame = result.reset_index()
        for column in ("experiment", "reference_strand", "sample"):
            frame.insert(0, column, partition[column])
        frames.append(frame)
        analyzed.append({**partition, "n_reads": int(len(frame))})
    joined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joined.to_parquet(output_path, index=False)
    logger.info(
        "Wrote project sample analysis for %d partition(s), %d read(s) -> %s",
        len(analyzed),
        len(joined),
        output_path,
    )
    return {
        "schema_version": SAMPLE_ANALYSIS_SCHEMA_VERSION,
        "canonical_reference": str(canonical_reference),
        "analysis": "periodicity",
        "method": str(method),
        "partitions": analyzed,
        "n_partitions": len(analyzed),
        "n_reads": int(len(joined)),
    }


EMBEDDING_EXPORT_SCHEMA_VERSION = 1


def project_embedding(
    project_dir: str | Path,
    canonical_reference: str,
    output_path: str | Path,
    *,
    set_name: str | None = None,
    modality: str | None = None,
    experiments=None,
    stage: str | None = None,
    layer: str | None = None,
    start: int | None = None,
    end: int | None = None,
    feature_kind: str = "raw",
    leiden_resolution: float = 0.5,
    n_neighbors: int = 15,
    min_reads: int = 10,
    random_state: int = 42,
    force_recompute: bool = False,
    trust_local_models: bool = False,
) -> dict:
    """Fit or extend one shared project embedding and export its coordinates.

    The durable product stays where it belongs: an immutable, checksummed
    generation inside the project, selected by the embedding's own current
    pointer. What lands in *output_path* is a portable table of the coordinates
    and cluster assignments -- never the estimator pickles, which are
    trusted-local runtime artifacts and must not be spread into task outputs by
    a workflow that merely read them.

    Args:
        project_dir: Project root.
        canonical_reference: Harmonized reference to embed.
        output_path: Task-local parquet path for the exported coordinates.
        set_name: Optional named set restricting the experiments.
        modality: Optional modality restriction.
        experiments: Optional explicit experiment IDs.
        stage: Optional pipeline stage for spine resolution.
        layer: Layer to embed; ``None`` uses ``X``.
        start: Optional genomic window start.
        end: Optional genomic window end.
        feature_kind: ``"raw"`` or ``"acf"`` feature construction.
        leiden_resolution: Leiden clustering resolution.
        n_neighbors: Neighborhood size for UMAP and clustering.
        min_reads: Minimum reads required to fit.
        random_state: Deterministic seed.
        force_recompute: Refit from scratch, retaining prior generations.
        trust_local_models: Permit loading this project's persisted estimator
            pickles, which incremental growth requires.

    Returns:
        A summary with the selected generation, fit kind, and molecule counts.
    """
    import numpy as np
    import pandas as pd

    from ..project.embedding_store import fit_or_extend_embedding

    result = fit_or_extend_embedding(
        project_dir,
        canonical_reference,
        set_name=set_name,
        modality=modality,
        experiments=experiments,
        stage=stage,
        layer=layer,
        start=start,
        end=end,
        feature_kind=feature_kind,
        leiden_resolution=leiden_resolution,
        n_neighbors=n_neighbors,
        min_reads=min_reads,
        random_state=random_state,
        force_recompute=force_recompute,
        trust_local_models=trust_local_models,
    )
    obs_names = [str(name) for name in result["obs_names"]]
    umap = np.asarray(result["X_umap"], dtype=np.float64)
    pca = np.asarray(result["X_pca"], dtype=np.float64)
    frame = pd.DataFrame({"molecule_uid": obs_names, "cluster": np.asarray(result["clusters"])})
    for index in range(umap.shape[1]):
        frame[f"umap_{index + 1}"] = umap[:, index]
    for index in range(pca.shape[1]):
        frame[f"pca_{index + 1}"] = pca[:, index]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    # Generation identity lives on the published manifest the store returns as
    # `meta`, not on the coordinate payload.
    meta = result.get("meta", {}) or {}
    summary = {
        "schema_version": EMBEDDING_EXPORT_SCHEMA_VERSION,
        "canonical_reference": str(canonical_reference),
        "generation_id": str(meta.get("generation_id", "")),
        "fit_kind": str(meta.get("fit_kind", "")),
        "prior_generation_id": meta.get("prior_generation_id"),
        "n_molecules": int(len(obs_names)),
        # From the published manifest rather than the in-memory fit result: the
        # generation read back after publication does not carry the growth
        # counters, so sourcing them there silently reports zero new molecules.
        "n_new_molecules": int(meta.get("n_new_reads", 0) or 0),
        "n_clusters": int(len(set(map(str, result["clusters"])))),
        "feature_kind": str(feature_kind),
    }
    logger.info(
        "Exported project embedding generation %s (%s fit, %d molecules) -> %s",
        summary["generation_id"] or "unknown",
        summary["fit_kind"] or "unknown",
        summary["n_molecules"],
        output_path,
    )
    return summary


def project_sample_store_list(
    project_dir: str | Path, experiment_id: str | None = None
) -> list[dict]:
    """List cataloged per-sample-store partitions, optionally filtered to one experiment."""
    from ..project.sample_store import list_per_sample_partitions

    return list_per_sample_partitions(project_dir, experiment_id)

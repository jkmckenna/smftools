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
    ``smftools.project.sample_store`` and ``dev/project_sample_and_set_stores.md``)
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
    """Return ``(experiments, harmonized_references)`` for display."""
    from ..project.catalog import ProjectCatalog

    catalog = ProjectCatalog.open(project_dir)
    return catalog.experiments(), catalog.references()


def project_materialize(
    project_dir: str | Path,
    canonical_reference: str,
    output_path: str | Path,
    *,
    set_name: str | None = None,
    modality: str | None = None,
    stage: str | None = None,
    start: int | None = None,
    end: int | None = None,
    read_metrics: bool = False,
) -> Path:
    """Materialize a canonical reference across matching experiments and write it.

    ``stage`` picks a specific pipeline stage per experiment (``raw``,
    ``preprocess``, ``spatial``, ``hmm``, ...); the default falls back through
    the most-derived stage available per experiment, since a later stage's
    spine already carries forward everything earlier stages produced.
    """
    from ..project.catalog import project_adata
    from ..readwrite import safe_write_h5ad

    adata = project_adata(
        project_dir,
        canonical_reference,
        set_name=set_name,
        modality=modality,
        stage=stage,
        start=start,
        end=end,
        read_metrics=read_metrics,
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

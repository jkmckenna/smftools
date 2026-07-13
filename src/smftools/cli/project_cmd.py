"""CLI logic for the project-level cross-experiment catalog."""

from __future__ import annotations

from pathlib import Path

from smftools.logging_utils import get_logger

logger = get_logger(__name__)


def project_init(project_dir: str | Path) -> Path:
    """Initialize a project directory + empty registry."""
    from ..project.registry import init_project

    return init_project(project_dir)


def project_add(
    project_dir: str | Path,
    experiment_dir: str | Path,
    *,
    experiment_id: str | None = None,
    name: str | None = None,
) -> tuple[str, dict, list[str]]:
    """Register an experiment and return ``(id, entry, reference_conflicts)``."""
    from ..project.catalog import ProjectCatalog
    from ..project.reference_registry import detect_reference_conflicts
    from ..project.registry import add_experiment

    exp_id, entry = add_experiment(
        project_dir, experiment_dir, experiment_id=experiment_id, name=name
    )
    conflicts = detect_reference_conflicts(ProjectCatalog.open(project_dir).references())
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
    start: int | None = None,
    end: int | None = None,
) -> Path:
    """Materialize a canonical reference across matching experiments and write it."""
    from ..project.catalog import project_adata
    from ..readwrite import safe_write_h5ad

    adata = project_adata(
        project_dir,
        canonical_reference,
        set_name=set_name,
        modality=modality,
        start=start,
        end=end,
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

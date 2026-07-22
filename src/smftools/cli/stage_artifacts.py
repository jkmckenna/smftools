"""Standard output layout and artifact catalogs for partitioned CLI stages."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from ..readwrite import atomic_write_json

DEFAULT_PLOT_CATEGORIES = (
    "global",
    "reference",
    "sample",
    "diagnostics",
)
STAGE_PLOT_CATEGORIES = {
    "preprocess": (
        "barcode_summary",
        "read_qc",
        "modification_qc",
        "duplicate_qc",
        "library_complexity",
        "read_span_quality",
        "coverage",
        "task_diagnostics",
    ),
    "variant": (
        "mismatch_frequency",
        "variant_calls",
        "variant_segments",
        "clustermaps",
        "diagnostics",
    ),
    "spatial": (
        "clustermaps",
        "autocorrelation",
        "periodicity",
        "rolling_metrics",
        "position_correlation",
        "embeddings",
        "diagnostics",
    ),
    "hmm": (
        "training",
        "emissions",
        "features",
        "clustermaps",
        "diagnostics",
    ),
    "latent": ("embeddings", "loadings", "clusters", "diagnostics"),
    "chimeric": ("segments", "classifications", "clustermaps", "diagnostics"),
}
PLOT_CATALOG_COLUMNS = (
    "artifact_id",
    "stage",
    "category",
    "plot_type",
    "reference",
    "sample",
    "core_start",
    "core_end",
    "path",
    "created_at",
)


@dataclass(frozen=True)
class StagePlotPaths:
    """Paths for one stage's automated plot artifacts."""

    root: Path
    catalog: Path
    context: Path
    categories: dict[str, Path]


def prepare_stage_plot_layout(
    stage_root: str | Path,
    *,
    stage: str,
    source_spine: str | Path | None = None,
    categories: Iterable[str] = DEFAULT_PLOT_CATEGORIES,
) -> StagePlotPaths:
    """Create a predictable plotting tree and initialize its artifact catalog."""
    stage_root = Path(stage_root)
    plot_root = stage_root / "plots"
    category_paths = {str(category): plot_root / str(category) for category in categories}
    for path in category_paths.values():
        path.mkdir(parents=True, exist_ok=True)
    catalog = plot_root / "catalog.parquet"
    if not catalog.exists():
        pd.DataFrame(columns=PLOT_CATALOG_COLUMNS).to_parquet(catalog, index=False)
    context = plot_root / "context.json"
    atomic_write_json(
        context,
        {
            "stage": str(stage),
            "source_spine": (
                str(Path(source_spine).resolve()) if source_spine is not None else None
            ),
            "categories": {
                name: path.relative_to(stage_root).as_posix()
                for name, path in category_paths.items()
            },
        },
    )
    return StagePlotPaths(
        root=plot_root,
        catalog=catalog,
        context=context,
        categories=category_paths,
    )


def prepare_analysis_plot_layout(
    stage_root: str | Path,
    *,
    stage: str,
    source_spine: str | Path | None = None,
) -> StagePlotPaths:
    """Create the standard category tree for a known analysis stage."""
    stage = str(stage).lower()
    categories = STAGE_PLOT_CATEGORIES.get(stage, DEFAULT_PLOT_CATEGORIES)
    return prepare_stage_plot_layout(
        stage_root,
        stage=stage,
        source_spine=source_spine,
        categories=categories,
    )


def register_plot_artifact(
    layout: StagePlotPaths,
    plot_path: str | Path,
    *,
    stage: str,
    category: str,
    plot_type: str,
    reference: str | None = None,
    sample: str | None = None,
    core_start: int | None = None,
    core_end: int | None = None,
) -> Path:
    """Append one generated plot to the stage plot catalog."""
    if category not in layout.categories:
        raise KeyError(f"unknown plot category {category!r}")
    plot_path = Path(plot_path)
    try:
        relative_path = plot_path.resolve().relative_to(layout.root.parent.resolve())
    except ValueError as exc:
        raise ValueError("plot artifacts must be stored inside the stage directory") from exc
    catalog = pd.read_parquet(layout.catalog)
    artifact_id = f"{stage}:{category}:{plot_type}:{len(catalog):06d}"
    row = pd.DataFrame(
        [
            {
                "artifact_id": artifact_id,
                "stage": str(stage),
                "category": str(category),
                "plot_type": str(plot_type),
                "reference": reference,
                "sample": sample,
                "core_start": core_start,
                "core_end": core_end,
                "path": relative_path.as_posix(),
                "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            }
        ]
    )
    pd.concat([catalog, row], ignore_index=True).to_parquet(layout.catalog, index=False)
    return layout.catalog

"""Project catalog: harmonized cross-experiment selection + cross-experiment materialize.

Reference-level selection (which experiments/references match) runs in-memory over the
registry-derived alias table and needs no extra dependency. Row/SQL-level queries over
the registered experiments' interval catalogs use **DuckDB** when available (optional
``catalog`` extra); a pandas union is the fallback. Cross-experiment analysis resolves a
canonical reference to each experiment's own reference name and ``materialize``s + concats
the matching slices -- never a global merge.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from smftools.logging_utils import get_logger

logger = get_logger(__name__)


def _load_duckdb():
    try:
        import duckdb

        return duckdb
    except ImportError:
        return None


class ProjectCatalog:
    """Harmonized view over a project's registered experiments."""

    def __init__(self, project_dir, registry, reference_registry, alias_table):
        self.project_dir = Path(project_dir)
        self.registry = registry
        self.reference_registry = reference_registry
        self.alias_table = alias_table

    @classmethod
    def open(cls, project_dir: str | Path) -> "ProjectCatalog":
        from .reference_registry import (
            REFERENCE_REGISTRY_FILENAME,
            ReferenceRegistry,
            build_reference_alias_table,
        )
        from .registry import load_registry

        project_dir = Path(project_dir)
        registry = load_registry(project_dir)
        reference_registry = ReferenceRegistry.load(project_dir / REFERENCE_REGISTRY_FILENAME)
        active = [
            {"id": exp_id, **entry}
            for exp_id, entry in registry["experiments"].items()
            if entry.get("status") == "active"
        ]
        alias = build_reference_alias_table(active, reference_registry)
        if not alias.empty:
            modality = {e["id"]: e["modality"] for e in active}
            alias["modality"] = alias["experiment"].map(modality)
        return cls(project_dir, registry, reference_registry, alias)

    def experiments(self, *, active_only: bool = True) -> list[dict]:
        from .registry import list_experiments

        return list_experiments(self.project_dir, active_only=active_only)

    def references(self) -> pd.DataFrame:
        """Harmonized (experiment, reference_strand, reference_uid, canonical_reference)."""
        return self.alias_table

    def _resolve_experiment_filter(self, experiments, set_name) -> set[str] | None:
        ids: set[str] | None = None
        if experiments is not None:
            ids = {str(e) for e in experiments}
        if set_name is not None:
            from .registry import resolve_set

            saved = resolve_set(self.project_dir, set_name)
            if saved["kind"] == "list":
                set_ids = set(saved["experiments"])
            else:
                result = self.query(f"SELECT DISTINCT experiment FROM refs WHERE {saved['sql']}")
                set_ids = set(result["experiment"].astype(str))
            ids = set_ids if ids is None else (ids & set_ids)
        return ids

    def select(
        self,
        *,
        canonical_reference=None,
        modality=None,
        experiments=None,
        set_name=None,
    ) -> pd.DataFrame:
        """Select harmonized experiment-references matching the filters (pandas, no dep)."""
        table = self.alias_table
        if table.empty:
            return table
        mask = pd.Series(True, index=table.index)
        if canonical_reference is not None:
            wanted = (
                {canonical_reference}
                if isinstance(canonical_reference, str)
                else set(canonical_reference)
            )
            mask &= table["canonical_reference"].isin(wanted)
        if modality is not None:
            wanted = {modality} if isinstance(modality, str) else set(modality)
            mask &= table["modality"].isin(wanted)
        exp_filter = self._resolve_experiment_filter(experiments, set_name)
        if exp_filter is not None:
            mask &= table["experiment"].isin(exp_filter)
        return table.loc[mask]

    def interval_catalog(self) -> pd.DataFrame:
        """Union the registered experiments' interval catalogs (one row per raw shard)."""
        frames = []
        for entry in self.experiments():
            path = entry.get("catalogs", {}).get("interval_catalog.parquet")
            if path and Path(path).exists():
                frame = pd.read_parquet(path)
                frame["experiment"] = entry["id"]
                frames.append(frame)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def region_catalog(self, scope: str) -> pd.DataFrame:
        """Union one original-coordinate region scope across experiments."""
        if scope not in {"alignment", "analysis", "plot"}:
            raise ValueError("scope must be one of: alignment, analysis, plot")
        frames = []
        for entry in self.experiments():
            path = entry.get("catalogs", {}).get(f"{scope}_regions")
            if path and Path(path).exists():
                frame = pd.read_parquet(path)
                frame["experiment"] = entry["id"]
                frames.append(frame)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def reference_interval_map(self) -> pd.DataFrame:
        """Union stored-to-original reference maps across experiments."""
        frames = []
        for entry in self.experiments():
            path = entry.get("catalogs", {}).get("reference_interval_map")
            if path and Path(path).exists():
                frame = pd.read_parquet(path)
                frame["experiment"] = entry["id"]
                frames.append(frame)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def lookup_molecule(
        self,
        *,
        experiment_uid: str | None = None,
        read_id: str | None = None,
        molecule_uid: str | None = None,
    ) -> pd.DataFrame:
        """Locate one molecule's raw and derived records using Parquet indexes only.

        Supply either ``molecule_uid`` or both ``experiment_uid`` and ``read_id``.
        Task Zarr stores and experiment spines are not opened.
        """
        import pyarrow.dataset as ds

        if molecule_uid is None and (experiment_uid is None or read_id is None):
            raise ValueError("supply molecule_uid or both experiment_uid and read_id")
        if molecule_uid is not None and (experiment_uid is not None or read_id is not None):
            raise ValueError("molecule_uid cannot be combined with experiment_uid/read_id")

        frames: list[pd.DataFrame] = []
        for entry in self.experiments():
            if experiment_uid is not None and str(entry["experiment_uid"]) != str(experiment_uid):
                continue
            for key, path_value in entry.get("catalogs", {}).items():
                if key != "molecule_index" and not key.endswith("_read_index"):
                    continue
                path = Path(path_value)
                if not path.exists():
                    continue
                dataset = ds.dataset(path, format="parquet")
                expression = (
                    ds.field("molecule_uid") == str(molecule_uid)
                    if molecule_uid is not None
                    else (ds.field("experiment_uid") == str(experiment_uid))
                    & (ds.field("read_id") == str(read_id))
                )
                frame = dataset.to_table(filter=expression).to_pandas()
                if frame.empty:
                    continue
                if "stage" not in frame:
                    frame["stage"] = "raw"
                frame["experiment"] = entry["id"]
                frames.append(frame)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True, sort=False)

    def query(self, sql: str) -> pd.DataFrame:
        """Run SQL over the ``refs`` (harmonized references) and ``intervals`` tables.

        Needs DuckDB. ``refs`` avoids the reserved word ``references``.
        """
        duckdb = _load_duckdb()
        if duckdb is None:
            raise RuntimeError(
                "duckdb is required for SQL project queries; install `smftools[catalog]` "
                "(reference-level select() works without it)."
            )
        connection = duckdb.connect()
        try:
            connection.register("refs", self.alias_table)
            connection.register("intervals", self.interval_catalog())
            return connection.execute(sql).fetch_df()
        finally:
            connection.close()


def resolve_set_members(
    catalog: "ProjectCatalog",
    canonical_reference: str,
    *,
    set_name=None,
    modality=None,
    experiments=None,
    stage: str | None = None,
) -> list[dict]:
    """Resolve which experiments/stages/reference-strands a query would materialize.

    Cheap -- selects over the in-memory harmonized reference table and the registry's
    already-resolved spine paths, without opening any matrix data. Shared by
    :func:`project_adata` (which does the actual materialize) and
    ``project.set_store`` (which uses this alone to compute a cache key without
    paying for a materialize just to check whether one is needed).

    Returns one dict per matched, spine-available experiment:
    ``{"experiment": exp_id, "stage": resolved_stage, "spine_path": Path,
    "reference_strands": [...]}. Experiments with no spine available for ``stage``
    (or, in fallback mode, no stage at all) are skipped with a warning, not included.
    """
    from .registry import resolve_experiment_spine

    selection = catalog.select(
        canonical_reference=canonical_reference,
        modality=modality,
        experiments=experiments,
        set_name=set_name,
    )
    if selection.empty:
        return []

    entries = {entry["id"]: entry for entry in catalog.experiments()}
    members = []
    for exp_id, group in selection.groupby("experiment", sort=True):
        entry = entries.get(exp_id)
        resolved = resolve_experiment_spine(entry, stage) if entry is not None else None
        if resolved is None:
            logger.warning(
                "skipping experiment %r: no spine available for stage=%r",
                exp_id,
                stage or "any (auto)",
            )
            continue
        resolved_stage, spine_path = resolved
        members.append(
            {
                "experiment": str(exp_id),
                "experiment_uid": str(entry["experiment_uid"]),
                "stage": resolved_stage,
                "spine_path": spine_path,
                "reference_strands": sorted(group["reference_strand"].unique()),
            }
        )
    return members


DEFAULT_MAX_POOL_BYTES = 8 * 1024**3  # 8 GiB
PROJECT_EXPORT_SCHEMA_VERSION = 1
PROJECT_EXPORT_FORMAT = "anndata-zarr-v3"
PROJECT_EXPORT_CATALOG_COLUMNS = (
    "schema_version",
    "part_id",
    "artifact_format",
    "experiment",
    "experiment_uid",
    "stage",
    "canonical_reference",
    "reference_strands",
    "barcode",
    "start",
    "end",
    "chunk_index",
    "n_reads",
    "n_positions",
    "layers",
    "read_metrics",
    "path",
)
PROJECT_DEFAULT_ALL_LAYER_ESTIMATE = 32


@dataclass(frozen=True)
class ProjectSelectionEstimate:
    """Conservative pre-allocation estimate for one project selection."""

    total_bytes: int
    n_reads: int
    n_positions: int
    n_arrays: int
    members: tuple[dict, ...]


def _project_resource_config(
    *, max_memory_gb: float | None = None, max_memory_percent: float | None = 60.0
):
    """Build the small config surface consumed by the shared resource resolver."""
    return SimpleNamespace(
        threads=1,
        max_memory_gb=max_memory_gb,
        max_memory_percent=max_memory_percent,
        memory_reserve_gb=1.0,
        target_task_memory_mb=256,
    )


def _fallback_member_width(member: dict, start: int | None, end: int | None) -> int:
    """Read only backed metadata when a legacy/noncanonical index cannot size a width."""
    if start is not None:
        return int(end) - int(start)
    import anndata as ad

    backed = ad.read_h5ad(member["spine_path"], backed="r")
    try:
        reference_lengths = backed.uns.get("reference_lengths", {})
        widths = [
            int(reference_lengths[reference])
            for reference in member["reference_strands"]
            if reference in reference_lengths
        ]
        return max(widths, default=int(backed.n_vars))
    finally:
        backed.file.close()


def _index_filter(dataset, member: dict, start: int | None, end: int | None):
    """Build the project member's predicate without materializing index rows."""
    import pyarrow.dataset as ds

    available = set(dataset.schema.names)
    expression = ds.field("Reference_strand").isin(member["reference_strands"])
    if "experiment_uid" in available:
        expression &= ds.field("experiment_uid") == str(member["experiment_uid"])
    if start is not None:
        expression &= (ds.field("reference_start") < int(end)) & (
            ds.field("reference_end") > int(start)
        )
    return expression


def _indexed_selection_summary(
    index_path: str | Path,
    member: dict,
    start: int | None,
    end: int | None,
) -> tuple[bool, int, int]:
    """Return index ownership, selected rows, and width using bounded Arrow scans."""
    import pyarrow.dataset as ds

    dataset = ds.dataset(Path(index_path), format="parquet")
    available = set(dataset.schema.names)
    owner_expression = ds.field("Reference_strand").isin(member["reference_strands"])
    if "experiment_uid" in available:
        owner_expression &= ds.field("experiment_uid") == str(member["experiment_uid"])
    owns_index = dataset.count_rows(filter=owner_expression) > 0
    if not owns_index:
        return False, 0, 0
    expression = _index_filter(dataset, member, start, end)
    n_reads = int(dataset.count_rows(filter=expression))
    if start is not None:
        return True, n_reads, int(end) - int(start)
    width = 0
    for batch in dataset.scanner(
        columns=["reference_end"], filter=expression, batch_size=65536
    ).to_batches():
        values = batch.column(0).to_numpy(zero_copy_only=False)
        if values.size:
            width = max(width, int(values.max()))
    return True, n_reads, width


def _iter_index_selection_batches(
    member: dict,
    *,
    start: int | None,
    end: int | None,
    batch_size: int,
):
    """Yield bounded ``(barcode, read_ids)`` batches from one molecule index."""
    import pyarrow.dataset as ds

    dataset = ds.dataset(Path(member["molecule_index"]), format="parquet")
    available = set(dataset.schema.names)
    columns = ["read_id", *(["Barcode"] if "Barcode" in available else [])]
    scanner = dataset.scanner(
        columns=columns,
        filter=_index_filter(dataset, member, start, end),
        batch_size=max(1, int(batch_size)),
    )
    for batch in scanner.to_batches():
        frame = batch.to_pandas()
        if frame.empty:
            continue
        if "Barcode" not in frame:
            yield "all", frame["read_id"].astype(str).tolist()
            continue
        for barcode, group in frame.groupby("Barcode", sort=True, observed=True):
            yield str(barcode), group["read_id"].astype(str).tolist()


def _project_export_part_id(member: dict, barcode: str, chunk_index: int) -> str:
    """Return a stable, collision-resistant identifier for one exported part."""
    identity = "\0".join(
        (
            str(member["experiment_uid"]),
            str(member["stage"]),
            str(barcode),
            str(chunk_index),
        )
    )
    return hashlib.sha256(identity.encode()).hexdigest()[:20]


def _validate_project_export_catalog(catalog: pd.DataFrame, root: Path) -> None:
    """Verify that a completed catalog describes every exported Zarr part exactly once."""
    missing_columns = set(PROJECT_EXPORT_CATALOG_COLUMNS).difference(catalog.columns)
    if missing_columns:
        raise RuntimeError(
            "project export catalog is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )
    if catalog.empty:
        raise RuntimeError("project export produced no cataloged parts")
    if catalog["part_id"].duplicated().any():
        raise RuntimeError("project export produced duplicate part identifiers")
    if catalog["path"].duplicated().any():
        raise RuntimeError("project export catalog contains duplicate paths")
    if (catalog["n_reads"] <= 0).any() or (catalog["n_positions"] < 0).any():
        raise RuntimeError("project export catalog contains invalid part dimensions")

    catalog_paths: set[str] = set()
    for value in catalog["path"]:
        relative = Path(str(value))
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"project export catalog path is not portable: {value!r}")
        if not (root / relative).is_dir():
            raise RuntimeError(f"project export catalog path does not exist: {value!r}")
        catalog_paths.add(relative.as_posix())
    disk_paths = {path.relative_to(root).as_posix() for path in (root / "parts").rglob("*.zarr")}
    if catalog_paths != disk_paths:
        uncataloged = sorted(disk_paths - catalog_paths)
        missing = sorted(catalog_paths - disk_paths)
        raise RuntimeError(
            "project export catalog/part mismatch: "
            f"uncataloged={uncataloged!r}, missing={missing!r}"
        )


def _sha256_file(path: Path) -> str:
    """Hash a file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def estimate_project_selection(
    project_dir: str | Path,
    canonical_reference: str,
    *,
    set_name=None,
    modality=None,
    experiments=None,
    stage: str | None = None,
    start: int | None = None,
    end: int | None = None,
    layers=None,
    read_metrics: bool = False,
) -> ProjectSelectionEstimate:
    """Estimate selected arrays and metadata before any AnnData part is allocated."""

    catalog = ProjectCatalog.open(project_dir)
    members = resolve_set_members(
        catalog,
        canonical_reference,
        set_name=set_name,
        modality=modality,
        experiments=experiments,
        stage=stage,
    )
    entries = {entry["id"]: entry for entry in catalog.experiments()}
    n_arrays = 1 + (PROJECT_DEFAULT_ALL_LAYER_ESTIMATE if layers is None else len(layers))
    estimates = []
    total_reads = 0
    maximum_width = 0
    for member in members:
        entry = entries[member["experiment"]]
        index_path = entry.get("catalogs", {}).get("molecule_index")
        if index_path and Path(index_path).exists():
            owns_index, n_reads, width = _indexed_selection_summary(index_path, member, start, end)
            if not owns_index:
                # Older/noncanonical registrations can point at a shared or absent
                # index. Fall back to the registered count plus backed spine metadata.
                n_reads = int(entry.get("n_reads", 0))
                width = _fallback_member_width(member, start, end)
        else:
            # Legacy stores lack a predicate-pruned molecule index. Their registered
            # read count still provides an early conservative allocation check.
            n_reads = int(entry.get("n_reads", 0))
            width = _fallback_member_width(member, start, end)
            owns_index = False
        # float64-equivalent arrays, a second destination/serialization copy,
        # scalar obs/var overhead, and optional obsm/read-metric allowance.
        array_bytes = n_reads * max(1, width) * max(1, n_arrays) * 8 * 2
        metadata_bytes = n_reads * (2048 + (4096 if read_metrics else 0)) + width * 512
        estimated_bytes = int(array_bytes + metadata_bytes)
        estimates.append(
            {
                **member,
                "molecule_index": index_path,
                "indexed": bool(index_path and owns_index),
                "n_reads": n_reads,
                "n_positions": width,
                "estimated_bytes": estimated_bytes,
            }
        )
        total_reads += n_reads
        maximum_width = max(maximum_width, width)
    # Pooled AnnData needs both resident inputs and a destination concat object.
    total_bytes = sum(item["estimated_bytes"] for item in estimates) * 2
    return ProjectSelectionEstimate(
        total_bytes=int(total_bytes),
        n_reads=total_reads,
        n_positions=maximum_width,
        n_arrays=n_arrays,
        members=tuple(estimates),
    )


def project_adata(
    project_dir: str | Path,
    canonical_reference: str,
    *,
    set_name=None,
    modality=None,
    experiments=None,
    stage: str | None = None,
    start: int | None = None,
    end: int | None = None,
    layers=None,
    read_metrics=False,
    lazy: bool | None = None,
    allow_large: bool = False,
    max_bytes: int = DEFAULT_MAX_POOL_BYTES,
    max_memory_gb: float | None = None,
    max_memory_percent: float | None = 60.0,
):
    """Materialize + concat a canonical reference across the matching experiments.

    This is the explicit, size-guarded "give me one pooled AnnData" opt-in over a set.
    It consumes ``set_store.iter_set_parts`` (streamed, projected) and concatenates the
    stream along obs (shared coordinate system) -- no global merge. Adds an
    ``experiment`` obs column and keeps only the shared genomic-position axis for
    ``var`` (see ``set_store.normalize_part``).

    Each experiment's spine is picked independently: ``stage`` picks a specific
    pipeline stage (``"raw"``, ``"preprocess"``, ``"spatial"``, ``"hmm"``, ...) and
    skips experiments that haven't reached it; ``None`` (default) falls back through the
    most-derived stage available per experiment. Prefer a narrow ``layers`` subset
    and/or a ``start``/``end`` window -- pooling *all* layers at full locus across many
    experiments is what produced the >200 GB objects the redesign exists to avoid.

    The full selection and concat destination are estimated from project indexes before
    the first part is materialized. ``max_bytes`` is a soft warning threshold;
    ``allow_large`` acknowledges that warning, while the resolved resource envelope
    remains a hard ceiling. Use :func:`export_project_partitions` when the selection
    should remain partitioned instead of allocating a final concat destination.
    """
    import anndata as ad

    from ..memory_guard import require_memory_headroom
    from .set_store import iter_set_parts

    estimate = estimate_project_selection(
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
    )
    if not estimate.members:
        raise ValueError(
            f"no experiment matched canonical_reference={canonical_reference!r} had a spine "
            f"available for stage={stage or 'any (auto)'!r}"
        )
    if estimate.n_reads == 0:
        raise ValueError("project selection matched no molecules")
    if not allow_large and estimate.total_bytes > max_bytes:
        raise ValueError(
            f"pooled object is estimated at {estimate.total_bytes / 1024**3:.1f} GiB, "
            f"above the {max_bytes / 1024**3:.1f} GiB warning threshold. Narrow it with "
            "layers/start/end, use partitioned project materialization, or pass allow_large=True "
            "to acknowledge this warning. The resolved hard memory ceiling still applies."
        )
    resource_cfg = _project_resource_config(
        max_memory_gb=max_memory_gb, max_memory_percent=max_memory_percent
    )
    require_memory_headroom(
        resource_cfg,
        estimated_memory_mb=max(1, estimate.total_bytes) / 1024**2,
        operation_label="pooled project materialization",
        estimator="project_pool_peak",
    )

    parts = []
    for sub in iter_set_parts(
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
        lazy=lazy,
    ):
        parts.append(sub)

    if not parts:
        raise ValueError(
            f"no experiment matched canonical_reference={canonical_reference!r} had a spine "
            f"available for stage={stage or 'any (auto)'!r}"
        )
    if len(parts) == 1:
        return parts[0]
    return ad.concat(parts, join="outer", merge="first", uns_merge="first")


def export_project_partitions(
    project_dir: str | Path,
    canonical_reference: str,
    output_dir: str | Path,
    *,
    set_name=None,
    modality=None,
    experiments=None,
    stage: str | None = None,
    start: int | None = None,
    end: int | None = None,
    layers=None,
    read_metrics: bool = False,
    max_memory_gb: float | None = None,
    max_memory_percent: float | None = 60.0,
) -> Path:
    """Transactionally export bounded experiment/barcode Zarr parts plus a catalog."""
    from ..informatics.partition_read import materialize
    from ..informatics.physical_layout import portable_matrix_chunks
    from ..memory_guard import require_memory_headroom, resource_envelope_for_config
    from ..readwrite import atomic_write_json, safe_write_zarr
    from .set_store import normalize_part, slug

    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"partitioned project output already exists: {output_dir}")
    estimate = estimate_project_selection(
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
    )
    if not estimate.members:
        raise ValueError(f"no experiments match canonical_reference={canonical_reference!r}")
    if estimate.n_reads == 0:
        raise ValueError("project selection matched no molecules")
    resource_cfg = _project_resource_config(
        max_memory_gb=max_memory_gb, max_memory_percent=max_memory_percent
    )
    envelope = resource_envelope_for_config(resource_cfg)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(dir=output_dir.parent, prefix=f".{output_dir.name}."))
    catalog_rows: list[dict] = []
    try:
        for member in estimate.members:
            bytes_per_read = max(1, member["estimated_bytes"] // max(1, member["n_reads"]))
            chunk_reads = max(1, envelope.resolved_memory_bytes // max(1, bytes_per_read * 2))
            if member["indexed"]:
                groups = _iter_index_selection_batches(
                    member,
                    start=start,
                    end=end,
                    batch_size=chunk_reads,
                )
            else:
                # Legacy/noncanonical stores have no bounded index path. They are
                # admitted only as one preflighted member-sized materialization.
                groups = iter([("all", None)])
            chunk_by_barcode: dict[str, int] = {}
            for barcode, chunk_ids in groups:
                chunk_index = chunk_by_barcode.get(str(barcode), 0)
                chunk_by_barcode[str(barcode)] = chunk_index + 1
                group_size = member["n_reads"] if chunk_ids is None else len(chunk_ids)
                estimated_bytes = group_size * bytes_per_read
                require_memory_headroom(
                    resource_cfg,
                    estimated_memory_mb=max(1, estimated_bytes) / 1024**2,
                    operation_label=(
                        f"project export {member['experiment']} barcode={barcode} "
                        f"chunk={chunk_index}"
                    ),
                    estimator="project_export_part_peak",
                )
                part = materialize(
                    member["spine_path"],
                    references=member["reference_strands"],
                    read_ids=chunk_ids,
                    start=start,
                    end=end,
                    layers=layers,
                    read_metrics=read_metrics,
                )
                part = normalize_part(
                    part,
                    member["experiment"],
                    member["stage"],
                    member["experiment_uid"],
                )
                if part.n_obs == 0:
                    del part
                    continue
                part_id = _project_export_part_id(member, str(barcode), chunk_index)
                relative = (
                    Path("parts")
                    / f"experiment={slug(member['experiment'])}"
                    / (
                        f"barcode={slug(str(barcode))}__chunk={chunk_index:05d}"
                        f"__part={part_id}.zarr"
                    )
                )
                path = temporary / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                safe_write_zarr(
                    part,
                    path,
                    backup=False,
                    verbose=False,
                    zarr_format=3,
                    chunks=portable_matrix_chunks(part.shape, getattr(part.X, "dtype", "float32")),
                )
                catalog_rows.append(
                    {
                        "schema_version": PROJECT_EXPORT_SCHEMA_VERSION,
                        "part_id": part_id,
                        "artifact_format": PROJECT_EXPORT_FORMAT,
                        "experiment": member["experiment"],
                        "experiment_uid": member["experiment_uid"],
                        "stage": member["stage"],
                        "canonical_reference": canonical_reference,
                        "reference_strands": member["reference_strands"],
                        "barcode": str(barcode),
                        "start": start,
                        "end": end,
                        "chunk_index": chunk_index,
                        "n_reads": part.n_obs,
                        "n_positions": part.n_vars,
                        "layers": sorted(part.layers.keys()),
                        "read_metrics": bool(read_metrics),
                        "path": relative.as_posix(),
                    }
                )
                del part
        catalog = pd.DataFrame(catalog_rows, columns=PROJECT_EXPORT_CATALOG_COLUMNS)
        catalog = catalog.sort_values(
            ["experiment", "barcode", "chunk_index", "part_id"], ignore_index=True
        )
        _validate_project_export_catalog(catalog, temporary)
        catalog_path = temporary / "catalog.parquet"
        catalog.to_parquet(catalog_path, index=False)
        atomic_write_json(
            temporary / "manifest.json",
            {
                "schema_version": PROJECT_EXPORT_SCHEMA_VERSION,
                "status": "complete",
                "canonical_reference": canonical_reference,
                "start": start,
                "end": end,
                "layers": None if layers is None else list(layers),
                "read_metrics": bool(read_metrics),
                "catalog": catalog_path.relative_to(temporary).as_posix(),
                "catalog_schema_version": PROJECT_EXPORT_SCHEMA_VERSION,
                "catalog_sha256": _sha256_file(catalog_path),
                "n_parts": len(catalog),
                "n_reads": int(catalog["n_reads"].sum()),
                "resource_envelope": envelope.as_dict(),
            },
        )
        os.replace(temporary, output_dir)
    except Exception:
        import shutil

        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output_dir

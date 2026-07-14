"""Project catalog: harmonized cross-experiment selection + cross-experiment materialize.

Reference-level selection (which experiments/references match) runs in-memory over the
registry-derived alias table and needs no extra dependency. Row/SQL-level queries over
the registered experiments' interval catalogs use **DuckDB** when available (optional
``catalog`` extra); a pandas union is the fallback. Cross-experiment analysis resolves a
canonical reference to each experiment's own reference name and ``materialize``s + concats
the matching slices -- never a global merge.
"""

from __future__ import annotations

from pathlib import Path

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
            wanted = {canonical_reference} if isinstance(canonical_reference, str) else set(canonical_reference)
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
                "stage": resolved_stage,
                "spine_path": spine_path,
                "reference_strands": sorted(group["reference_strand"].unique()),
            }
        )
    return members


DEFAULT_MAX_POOL_BYTES = 8 * 1024**3  # 8 GiB


def _part_nbytes(sub) -> int:
    """Rough in-memory footprint of a materialized part (X + layers)."""
    total = int(getattr(sub.X, "nbytes", 0) or 0)
    for layer in sub.layers.values():
        total += int(getattr(layer, "nbytes", 0) or 0)
    return total


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
    lazy: bool = False,
    allow_large: bool = False,
    max_bytes: int = DEFAULT_MAX_POOL_BYTES,
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

    Guardrail: parts are summed as the stream is consumed and, once the running total
    would exceed ``max_bytes`` (default 8 GiB), a ``ValueError`` is raised naming the
    windowing/layer/``allow_large`` escape hatches -- so an accidental full-project,
    all-layers pool can't silently build a machine-filling object. Pass
    ``allow_large=True`` to bypass the guardrail when you know it fits (e.g. after
    projecting to one layer + a window).
    """
    import anndata as ad

    from .set_store import iter_set_parts

    parts = []
    running = 0
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
        running += _part_nbytes(sub)
        if not allow_large and running > max_bytes:
            raise ValueError(
                f"pooled object exceeds {max_bytes / 1024**3:.1f} GiB "
                f"(already {running / 1024**3:.1f} GiB across {len(parts)} experiment(s)). "
                "Narrow it with a layer subset (layers=[...]) and/or a genomic window "
                "(start=/end=), stream it with set_store.iter_set_parts instead of "
                "pooling, or pass allow_large=True if you're sure it fits."
            )

    if not parts:
        raise ValueError(
            f"no experiment matched canonical_reference={canonical_reference!r} had a spine "
            f"available for stage={stage or 'any (auto)'!r}"
        )
    if len(parts) == 1:
        return parts[0]
    return ad.concat(parts, join="outer", merge="first", uns_merge="first")

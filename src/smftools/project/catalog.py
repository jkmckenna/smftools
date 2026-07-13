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


def project_adata(
    project_dir: str | Path,
    canonical_reference: str,
    *,
    set_name=None,
    modality=None,
    experiments=None,
    start: int | None = None,
    end: int | None = None,
    layers=None,
    lazy: bool = False,
):
    """Materialize + concat a canonical reference across the matching experiments.

    Resolves the canonical reference back to each experiment's own reference name(s),
    ``materialize``s each experiment's slice, and concatenates along obs (shared
    coordinate system) -- no global merge. Adds an ``experiment`` obs column.
    """
    import anndata as ad

    from ..informatics.partition_read import materialize

    catalog = ProjectCatalog.open(project_dir)
    selection = catalog.select(
        canonical_reference=canonical_reference,
        modality=modality,
        experiments=experiments,
        set_name=set_name,
    )
    if selection.empty:
        raise ValueError(f"no experiment references match canonical_reference={canonical_reference!r}")

    exp_paths = {
        entry["id"]: (Path(entry["path"]), entry.get("spine", "spine.h5ad"))
        for entry in catalog.experiments()
    }
    parts = []
    for exp_id, group in selection.groupby("experiment", sort=True):
        exp_dir, spine_rel = exp_paths[exp_id]
        reference_strands = sorted(group["reference_strand"].unique())
        sub = materialize(
            exp_dir / spine_rel,
            references=reference_strands,
            start=start,
            end=end,
            layers=layers,
            lazy=lazy,
        )
        sub.obs["experiment"] = str(exp_id)
        parts.append(sub)

    if len(parts) == 1:
        return parts[0]
    return ad.concat(parts, join="outer", merge="first", uns_merge="first")

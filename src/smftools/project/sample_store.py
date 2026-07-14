"""Per-sample store: catalogs which ``(experiment, Reference_strand, sample)``
partitions are available across a project.

Phase 1: modern (partitioned-store) experiments are cataloged as pointers -- nothing
is copied, since ``materialize()`` can already pull directly from the registry's spine
filtered by reference + sample.

Phase 2 (this module, current): legacy monolithic experiments are cached instead of
pointed at, since their only read path is a full eager load of the whole file (see
``informatics.partition_read._materialize_legacy``) -- repeating that on every later
query is wasteful when the source file never changes. Registering a legacy experiment
now materializes its latest stage once and writes each ``(Reference_strand, sample)``
partition out as its own small, uncompressed ``.h5ad`` under the project's per-sample
store; every later per-sample query reads that cache, not the original file. The
source legacy file itself is still never mutated (same invariant as the registry's
legacy adapter).

Still not implemented here: the project-computed per-sample analysis catalog
(autocorrelation, periodicity, ...) that will live alongside each partition -- see
``dev/project_sample_and_set_stores.md`` for the full design and remaining phases.
"""

from __future__ import annotations

import json
from pathlib import Path

PER_SAMPLE_DIRNAME = "per_sample"
POINTER_FILENAME = "pointer.json"
CACHE_FILENAME = "cache.h5ad"


def _per_sample_root(project_dir: str | Path) -> Path:
    return Path(project_dir) / "project_outputs" / PER_SAMPLE_DIRNAME


def partition_dir_for(
    project_dir: str | Path, experiment_id: str, reference_strand: str, sample: str
) -> Path:
    return _per_sample_root(project_dir) / experiment_id / reference_strand / sample


def backfill_per_sample_store(
    project_dir: str | Path,
    experiment_id: str,
    spine_path: str | Path,
    *,
    ref_column: str = "Reference_strand",
    sample_column: str = "Sample",
) -> list[Path]:
    """Catalog every ``(Reference_strand, sample)`` partition found in ``spine_path``.

    Returns ``[]`` if either partition column is missing. Two different behaviors
    depending on the spine, both idempotent (safe to call again against an updated
    experiment -- each partition's entry is simply overwritten, not accumulated):

    - **Modern** (``uns["is_spine"]``): writes a ``pointer.json`` recording the
      partition's read count only. No data is copied -- a later reader resolves the
      experiment's spine through the registry as usual and calls ``materialize()``
      filtered to this partition's reference + sample.
    - **Legacy** (no ``uns["is_spine"]``): writes each partition's molecules out as its
      own ``cache.h5ad`` (uncompressed -- these are meant to be re-read repeatedly, see
      :func:`smftools.readwrite.safe_write_h5ad`'s compression guidance) alongside a
      ``pointer.json`` recording where to find it. The source file is only ever read
      here, never written to.

    Does not remove a partition's entry if that partition disappears from a later
    stage (no garbage collection yet -- see the design doc's open questions).
    """
    from ..readwrite import safe_read_h5ad, safe_write_h5ad

    spine, _ = safe_read_h5ad(spine_path, verbose=False)
    if ref_column not in spine.obs or sample_column not in spine.obs:
        return []
    is_modern = bool(spine.uns.get("is_spine", False))

    written: list[Path] = []
    groups = spine.obs.groupby([ref_column, sample_column], observed=True).groups
    for (reference_strand, sample), read_ids in groups.items():
        if len(read_ids) == 0:
            continue
        reference_strand = str(reference_strand)
        sample = str(sample)
        partition_dir = partition_dir_for(project_dir, experiment_id, reference_strand, sample)
        partition_dir.mkdir(parents=True, exist_ok=True)
        pointer = {
            "kind": "pointer" if is_modern else "cache",
            "experiment_id": experiment_id,
            "reference_strand": reference_strand,
            "sample": sample,
            "n_reads": int(len(read_ids)),
        }
        if not is_modern:
            cache_path = partition_dir / CACHE_FILENAME
            safe_write_h5ad(spine[list(read_ids)].copy(), cache_path, backup=False, verbose=False)
            pointer["cache_path"] = CACHE_FILENAME
        path = partition_dir / POINTER_FILENAME
        path.write_text(json.dumps(pointer, indent=2, sort_keys=True))
        written.append(path)
    return written


def list_per_sample_partitions(
    project_dir: str | Path, experiment_id: str | None = None
) -> list[dict]:
    """Return every cataloged per-sample partition, optionally filtered to one experiment."""
    root = _per_sample_root(project_dir)
    if not root.exists():
        return []
    search_root = root / experiment_id if experiment_id else root
    depth = "*/*" if experiment_id else "*/*/*"
    results = []
    for pointer_path in sorted(search_root.glob(f"{depth}/{POINTER_FILENAME}")):
        results.append(json.loads(pointer_path.read_text()))
    return results


def load_per_sample_partition(
    project_dir: str | Path, experiment_id: str, reference_strand: str, sample: str
):
    """Load a cached legacy partition's molecules directly from the per-sample store.

    Only valid for ``kind == "cache"`` partitions (legacy experiments -- see
    :func:`backfill_per_sample_store`). Modern (``kind == "pointer"``) partitions have
    no cached copy to load here by design; resolve the experiment's spine through the
    registry and call ``materialize()`` filtered to this reference + sample instead.
    """
    from ..readwrite import safe_read_h5ad

    partition_dir = partition_dir_for(project_dir, experiment_id, reference_strand, sample)
    pointer_path = partition_dir / POINTER_FILENAME
    if not pointer_path.exists():
        raise FileNotFoundError(f"no per-sample store entry at {partition_dir}")
    pointer = json.loads(pointer_path.read_text())
    if pointer.get("kind") != "cache":
        raise ValueError(
            f"per-sample partition {reference_strand!r}/{sample!r} for {experiment_id!r} "
            f"is a {pointer.get('kind')!r} entry, not a cache -- resolve it through the "
            "registry + materialize() instead"
        )
    adata, _ = safe_read_h5ad(partition_dir / pointer["cache_path"], verbose=False)
    return adata

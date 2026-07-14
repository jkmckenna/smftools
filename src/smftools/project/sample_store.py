"""Per-sample store: catalogs which ``(experiment, Reference_strand, sample)``
partitions are available across a project.

Phase 1 (this module): modern (partitioned-store) experiments only. Registering an
experiment catalogs which partitions exist and how many reads each holds -- nothing is
copied, since ``materialize()`` can already pull directly from the registry's spine
pointer, filtered by reference + sample. Legacy-experiment caching (actually copying
data once at registration time, since legacy reads have no lazy path -- see
``informatics.partition_read._materialize_legacy``) is a later phase, along with the
project-computed per-sample analysis catalog (autocorrelation, periodicity, ...) that
will live alongside each partition's pointer. See
``dev/project_sample_and_set_stores.md`` for the full design.
"""

from __future__ import annotations

import json
from pathlib import Path

PER_SAMPLE_DIRNAME = "per_sample"
POINTER_FILENAME = "pointer.json"


def _per_sample_root(project_dir: str | Path) -> Path:
    return Path(project_dir) / "project_outputs" / PER_SAMPLE_DIRNAME


def _partition_dir(
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

    Only modern (partitioned-store) spines are cataloged -- returns ``[]`` for a legacy
    spine (no ``uns["is_spine"]``) or one missing either partition column, since there's
    nothing yet to point at for those (see module docstring). Overwrites each
    partition's ``pointer.json`` with current read counts on every call, so re-running
    ``project add`` against an updated experiment keeps counts current; it does not
    remove a partition's entry if that partition disappears from a later stage (no
    garbage collection yet -- see the design doc's open questions).
    """
    from ..readwrite import safe_read_h5ad

    spine, _ = safe_read_h5ad(spine_path, verbose=False)
    if not bool(spine.uns.get("is_spine", False)):
        return []
    if ref_column not in spine.obs or sample_column not in spine.obs:
        return []

    written: list[Path] = []
    counts = spine.obs.groupby([ref_column, sample_column], observed=True).size()
    for (reference_strand, sample), n_reads in counts.items():
        if n_reads <= 0:
            continue
        reference_strand = str(reference_strand)
        sample = str(sample)
        partition_dir = _partition_dir(project_dir, experiment_id, reference_strand, sample)
        partition_dir.mkdir(parents=True, exist_ok=True)
        pointer = {
            "kind": "pointer",
            "experiment_id": experiment_id,
            "reference_strand": reference_strand,
            "sample": sample,
            "n_reads": int(n_reads),
        }
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

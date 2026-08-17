"""A staging spine must point at artifacts that exist *now* (`F12`).

``staged_generation`` builds a preprocess generation in a staging directory and
only moves it to ``generations/<id>/`` at publish. ``execute_partitioned
_preprocessing`` used to stamp the final publication paths into the spine's uns
pointers before that move, then hand that spine to ``reduce_duplicate_reads``.

Every pointer therefore named a directory that did not exist yet.
``_overlay_preprocess_var`` returns quietly when the var catalog is missing, so
duplicate detection got a slice with no site-type and no membership columns,
built an empty comparison mask, and reported every read unique -- silently, and
regardless of how much genuine duplication the data contained.

These tests pin the invariant the fix restores: whatever spine a mid-execute
consumer is handed, its pointers resolve.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from smftools.informatics.partition_read import materialize, resolve_relative_path
from smftools.informatics.raw_store import write_raw_store
from smftools.preprocessing import partitioned_executor
from smftools.preprocessing.partitioned_executor import execute_partitioned_preprocessing
from smftools.readwrite import safe_read_h5ad

from .test_partitioned_preprocess_executor import _cfg, _frame

pytestmark = pytest.mark.unit

_POINTER_KEYS = (
    "preprocess_store",
    "preprocess_catalog",
    "preprocess_task_catalog",
    "preprocess_var",
    "preprocess_obs",
    "preprocess_read_index",
)


def _run(tmp_path, monkeypatch):
    """Run preprocessing with staging and publication dirs deliberately split.

    Captures the spine as ``reduce_duplicate_reads`` actually receives it --
    the only moment that matters, and the one the old code got wrong.
    """
    raw = write_raw_store(
        _frame(),
        tmp_path / "raw_outputs",
        reference_lengths={"ref_top": 12},
        extra_uns={"References": {"ref_FASTA_sequence": "ACGCGTACGTAC"}},
    )
    staging_dir = tmp_path / "preprocess_outputs" / ".staging" / "abc123"
    # Never created: mirrors a generation directory that only appears at publish.
    final_dir = tmp_path / "preprocess_outputs" / "generations" / "abc123"

    seen: dict[str, object] = {}
    original = partitioned_executor.reduce_duplicate_reads

    def _spy(preprocess_spine_path, obs_path, cfg):
        # Everything must be captured here: the staging spine is unlinked once
        # execute finishes, so there is no "after" in which to inspect it.
        spine, _ = safe_read_h5ad(str(preprocess_spine_path))
        seen["uns"] = dict(spine.uns)
        window = materialize(
            preprocess_spine_path, references="ref_top", start=0, end=12, layers=[]
        )
        seen["var_columns"] = list(window.var.columns)
        if "position_in_ref_top" in window.var:
            seen["membership"] = np.asarray(window.var["position_in_ref_top"], dtype=bool)
        return original(preprocess_spine_path, obs_path, cfg)

    monkeypatch.setattr(partitioned_executor, "reduce_duplicate_reads", _spy)
    outputs = execute_partitioned_preprocessing(
        raw["spine"],
        _cfg(),
        staging_dir,
        publication_dir=final_dir,
        run_root=tmp_path,
    )
    return outputs, seen, tmp_path, final_dir


def test_staging_spine_pointers_resolve_before_publication(tmp_path, monkeypatch):
    """Every artifact pointer dedup may follow must exist when dedup runs."""
    pytest.importorskip("pyarrow")
    _outputs, seen, run_root, final_dir = _run(tmp_path, monkeypatch)

    assert seen, "reduce_duplicate_reads was never called"
    assert not final_dir.exists(), "fixture invalid: publication dir must not exist yet"

    unresolved = {}
    for key in _POINTER_KEYS:
        pointer = seen["uns"].get(key)
        if pointer is None:
            continue
        path = resolve_relative_path(pointer, run_root)
        if path is None or not path.exists():
            unresolved[key] = pointer
    assert not unresolved, f"staging spine points at not-yet-published paths: {unresolved}"


def test_dedup_slice_carries_reference_context_columns(tmp_path, monkeypatch):
    """The consequence, stated as data rather than as paths.

    Pointer correctness is only interesting because of what it delivers: the
    columns duplicate detection builds its comparison mask from. Asserting the
    columns (not just the paths) is what would have caught `F12` even if the
    pointer had been wrong for some other reason.
    """
    pytest.importorskip("pyarrow")
    _outputs, seen, _run_root, _final_dir = _run(tmp_path, monkeypatch)

    assert "position_in_ref_top" in seen["var_columns"], (
        "membership column absent from the slice dedup compares on; "
        "the comparison mask would be empty and every read would look unique"
    )
    assert bool(seen["membership"].any())
    assert any(column.endswith("_site") for column in seen["var_columns"]), (
        "no site-type columns on the dedup slice; the comparison mask starts empty"
    )


def test_published_spine_keeps_publication_pointers(tmp_path, monkeypatch):
    """The staging rebind must not leak staging paths into the published spine.

    Guards the over-correction: it would be easy to fix dedup by pointing
    everything at ``output_dir`` and leaving it there, which would make every
    published generation reference a staging directory that gets removed.
    """
    pytest.importorskip("pyarrow")
    outputs, _seen, run_root, final_dir = _run(tmp_path, monkeypatch)

    spine, _ = safe_read_h5ad(str(outputs["spine"]))
    for key in _POINTER_KEYS:
        pointer = spine.uns.get(key)
        if pointer is None:
            continue
        resolved = resolve_relative_path(pointer, run_root)
        assert final_dir in resolved.parents or resolved == final_dir, (
            f"{key} should point into the publication directory, got {resolved}"
        )


def test_missing_var_catalog_is_reported_not_swallowed(tmp_path, monkeypatch, caplog):
    """A spine claiming a var catalog it cannot produce must say so.

    The silence is the reason `F12` survived three published generations: the
    columns simply were not there, and nothing anywhere said why.
    """
    import anndata as ad

    from smftools.informatics.partition_read import _overlay_preprocess_var

    spine = ad.AnnData(obs=pd.DataFrame(index=["r1"]))
    spine.uns["preprocess_var"] = "generations/not_published_yet/var.parquet"
    result = ad.AnnData(
        X=np.zeros((1, 3), dtype=float),
        obs=pd.DataFrame({"Reference_strand": ["ref_top"]}, index=["r1"]),
        var=pd.DataFrame(index=["0", "1", "2"]),
    )

    with caplog.at_level("WARNING"):
        _overlay_preprocess_var(spine, result, tmp_path)

    assert "position_in_ref_top" not in result.var, (
        "a missing catalog must not fabricate membership"
    )
    assert any("preprocess var catalog pointer" in record.message for record in caplog.records), (
        "a broken var pointer must warn rather than silently strip columns"
    )


def test_absent_preprocess_pointer_stays_quiet(tmp_path, caplog):
    """A raw slice has no preprocess stage; that is not a defect.

    Guards the over-correction of warning on every pre-preprocess
    materialization, which would bury the real signal in noise.
    """
    import anndata as ad

    from smftools.informatics.partition_read import _overlay_preprocess_var

    spine = ad.AnnData(obs=pd.DataFrame(index=["r1"]))
    result = ad.AnnData(
        X=np.zeros((1, 3), dtype=float),
        obs=pd.DataFrame({"Reference_strand": ["ref_top"]}, index=["r1"]),
        var=pd.DataFrame(index=["0", "1", "2"]),
    )

    with caplog.at_level("WARNING"):
        _overlay_preprocess_var(spine, result, tmp_path)

    assert not [r for r in caplog.records if "preprocess var catalog pointer" in r.message]

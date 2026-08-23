"""Rebuilding a raw generation's obs from its existing shards (`F34`)."""

from __future__ import annotations

import pandas as pd
import pytest

from smftools.informatics.ragged_store import RAGGED_ARRAY_COLUMNS
from smftools.informatics.raw_reassembly import (
    iter_shard_scalars,
    reassemble_obs,
    shard_relative_paths,
)
from smftools.informatics.raw_store import write_raw_store

pytestmark = pytest.mark.unit


def _read(read_id, barcode, seq_ints, *, bm=None, bc=None):
    row = dict(
        read_id=read_id,
        reference="ref",
        Reference_strand="ref_top",
        barcode=barcode,
        sample=barcode,
        reference_start=0,
        cigar=f"{len(seq_ints)}M",
        aligned_length=len(seq_ints),
        sequence=seq_ints,
        quality=[30] * len(seq_ints),
        mismatch=[4] * len(seq_ints),
        read_length=len(seq_ints),
        mapped_length=len(seq_ints),
        reference_length=12,
        read_quality=30,
        mapping_quality=60,
        read_length_to_reference_length_ratio=1.0,
        mapped_length_to_reference_length_ratio=1.0,
        mapped_length_to_read_length_ratio=1.0,
    )
    if bm is not None:
        row["BM"] = bm
    if bc is not None:
        row["BC"] = bc
    return row


def _store(tmp_path, rows, *, bam_path=None):
    return write_raw_store(
        pd.DataFrame(rows),
        tmp_path / "raw_outputs",
        reference_lengths={"ref_top": 12},
        analysis_mode="locus",
        bam_path=bam_path,
        extra_uns={"References": {"ref_FASTA_sequence": "ACGCGTACGTAC"}},
    )


def test_reassembly_without_annotation_reproduces_the_stored_obs(tmp_path):
    """The rebuild must be the same function of the same inputs as the live path.

    With annotation off, any difference is a defect in the reconstruction rather
    than an intended re-annotation -- so this pins the shard-pointer arithmetic
    (`ragged_shard`, `ragged_row`) and the molecule ordering.
    """
    rows = [
        _read("read1", "bc01", [0, 1, 2, 3]),
        _read("read2", "bc01", [3, 2, 1, 0]),
        _read("read3", "bc02", [1, 1, 1, 1]),
    ]
    raw_out = _store(tmp_path, rows)
    generation_dir = raw_out["spine"].parent

    rebuilt = reassemble_obs(generation_dir, annotate=None)
    stored = pd.read_parquet(generation_dir / "obs.parquet")

    assert len(rebuilt) == len(stored)
    shared = [column for column in stored.columns if column in rebuilt.columns]
    assert "ragged_shard" in shared and "ragged_row" in shared
    left = stored.reset_index(drop=True)[shared]
    right = rebuilt.reset_index(drop=True)[shared]
    differing = [column for column in shared if not left[column].equals(right[column])]
    assert differing == []


def test_reassembly_recovers_demux_type_from_shards(tmp_path):
    """The `F31` columns come back without re-extracting anything.

    `BM` survives into the shards, so the single- vs double-ended status is
    derivable from a generation that never recorded `demux_type` at all.
    """
    rows = [
        _read("read1", "bc01", [0, 1, 2, 3], bm="both", bc="bc01"),
        _read("read2", "bc01", [3, 2, 1, 0], bm="read_start_only", bc="bc01"),
        _read("read3", "bc02", [1, 1, 1, 1], bm="both", bc="bc02"),
    ]
    raw_out = _store(tmp_path, rows)
    generation_dir = raw_out["spine"].parent

    rebuilt = reassemble_obs(generation_dir).set_index("read_id")

    assert rebuilt.loc["read1", "demux_type"] == "double"
    assert rebuilt.loc["read2", "demux_type"] == "single"
    assert rebuilt.loc["read3", "demux_type"] == "double"
    assert set(rebuilt["demux_type_source"]) == {"bm_tag"}
    assert set(rebuilt["barcode_agreement"]) == {"agree"}


def test_reassembly_never_reads_the_ragged_arrays(tmp_path):
    """Projecting the arrays away is what makes this cheap rather than a re-read.

    Reading them back would turn a seconds-scale rebuild into a scan of the
    whole store, which is the cost the feature exists to avoid.
    """
    rows = [_read("read1", "bc01", [0, 1, 2, 3])]
    raw_out = _store(tmp_path, rows)
    generation_dir = raw_out["spine"].parent

    for _relative, frame in iter_shard_scalars(generation_dir):
        assert not set(frame.columns) & set(RAGGED_ARRAY_COLUMNS)


def test_shard_paths_come_from_the_spine_not_a_glob(tmp_path):
    """`ragged_row` pointers were assigned against the spine's shard order.

    A stray parquet beside the store must not enter the rebuild.
    """
    rows = [_read("read1", "bc01", [0, 1, 2, 3])]
    raw_out = _store(tmp_path, rows)
    generation_dir = raw_out["spine"].parent
    expected = shard_relative_paths(generation_dir)

    stray = generation_dir / "raw" / "reference=ref_top" / "not-a-shard.parquet"
    stray.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"read_id": ["ghost"]}).to_parquet(stray, index=False)

    assert shard_relative_paths(generation_dir) == expected
    assert all("not-a-shard" not in path for path in expected)


def _publish_real_generation(tmp_path, rows, *, generation_id):
    """Publish a generation whose shards are real parquet, not placeholder bytes."""
    from smftools.informatics.raw_generation import publish_raw_generation

    run_root = tmp_path
    raw_root = run_root / "raw_outputs"
    bam_path = raw_root / "intermediates" / "alignment.bam"
    bam_path.parent.mkdir(parents=True, exist_ok=True)
    bam_path.write_bytes(b"bam")
    raw_out = _store(tmp_path, rows, bam_path=bam_path)

    pd.DataFrame({"reference": ["ref_top"]}).to_parquet(
        run_root / "reference_interval_map.parquet", index=False
    )
    input_manifest = raw_root / "input_manifest"
    input_manifest.mkdir(parents=True, exist_ok=True)
    (input_manifest / "resolved_input_manifest.csv").write_text(
        "schema_version,path\n1,input.bam\n", encoding="utf-8"
    )
    (input_manifest / "resolved_input_manifest.json").write_text("{}\n", encoding="utf-8")
    (input_manifest / "input_resolution_report.json").write_text("{}\n", encoding="utf-8")

    sources = {
        "spine": raw_out["spine"],
        "ragged_store": raw_root / "raw",
        "interval_catalog": raw_out["interval_catalog"],
        "obs": raw_out["obs"],
        "molecules": raw_out["molecules"],
        "molecule_index": raw_out["molecule_index"],
        "segments": raw_out["segments"],
        "segment_index": raw_out["segment_index"],
        "reference_interval_map": run_root / "reference_interval_map.parquet",
        "input_manifest_csv": input_manifest / "resolved_input_manifest.csv",
        "input_manifest_json": input_manifest / "resolved_input_manifest.json",
        "input_resolution_report": input_manifest / "input_resolution_report.json",
    }
    return publish_raw_generation(
        run_root,
        sources,
        config_hash="config-a",
        input_artifact_ids=["input-manifest:abc"],
        generation_id=generation_id,
    )


def test_reassembled_generation_publishes_and_shares_its_parent_shards(tmp_path):
    """The rebuild publishes an immutable sibling rather than editing in place.

    Shards are the expensive artifact and are unchanged, so they must be
    hardlinked to the parent's -- copying 7.9 GB to rewrite a 217 MB obs would
    defeat the point.
    """
    from smftools.informatics.raw_generation import (
        resolve_current_raw_generation,
        validate_raw_generation,
    )
    from smftools.informatics.raw_reassembly import reassemble_raw_generation

    rows = [
        _read("read1", "bc01", [0, 1, 2, 3], bm="both", bc="bc01"),
        _read("read2", "bc01", [3, 2, 1, 0], bm="read_end_only", bc="bc01"),
    ]
    published = _publish_real_generation(tmp_path, rows, generation_id="generation-a")
    parent = published["generation"]
    assert "demux_type" not in pd.read_parquet(parent / "obs.parquet").columns

    result = reassemble_raw_generation(tmp_path)
    rebuilt = result["generation"]

    assert rebuilt != parent
    validate_raw_generation(rebuilt, run_root=tmp_path)
    current, _manifest = resolve_current_raw_generation(tmp_path / "raw_outputs")
    assert current == rebuilt

    obs = pd.read_parquet(rebuilt / "obs.parquet").set_index("read_id")
    assert obs.loc["read1", "demux_type"] == "double"
    assert obs.loc["read2", "demux_type"] == "single"
    assert set(obs["bam_path"]) == set(pd.read_parquet(parent / "obs.parquet")["bam_path"])

    for relative in shard_relative_paths(rebuilt):
        assert (rebuilt / relative).stat().st_ino == (parent / relative).stat().st_ino

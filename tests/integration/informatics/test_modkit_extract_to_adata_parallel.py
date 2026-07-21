"""Integration test: parallel (multiprocessing.Pool) vs serial dispatch must agree.

Uses a tiny real BAM/TSV/FASTA fixture (checked into tests/_test_inputs/parallel_dispatch/,
derived from the direct-modality e2e fixture) and splits its handful of reads into
one pseudo-sample per read via a synthetic barcode sidecar, forcing
modkit_extract_to_adata's non-split mode into multiple batches -- enough to exercise
genuine multi-worker Pool dispatch, not just the batch_size=1/n_batches=1 trivial case.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pysam
import pytest

from smftools.constants import H5_DIR
from smftools.informatics.modkit_extract_to_adata import modkit_extract_to_adata

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "_test_inputs" / "parallel_dispatch"

pytestmark = pytest.mark.integration


def _build_barcode_sidecar(bam_path: Path, out_path: Path) -> int:
    with pysam.AlignmentFile(str(bam_path), "rb") as bam:
        read_names = [
            r.query_name
            for r in bam.fetch(until_eof=True)
            if not r.is_secondary and not r.is_supplementary
        ]
    df = pd.DataFrame(
        {
            "read_name": read_names,
            "BC": [f"pseudo_barcode_{i}" for i in range(len(read_names))],
        }
    )
    df.to_parquet(out_path, index=False)
    return len(read_names)


def _run(tmp_path: Path, label: str, max_workers) -> "anndata.AnnData":  # noqa: F821
    out_dir = tmp_path / label
    out_dir.mkdir()
    sidecar = tmp_path / "barcode_sidecar.parquet"
    if not sidecar.exists():
        n_reads = _build_barcode_sidecar(FIXTURE_DIR / "sample.bam", sidecar)
        assert n_reads > 1, "fixture must have more than one read to exercise multiple batches"

    adata, _path = modkit_extract_to_adata(
        fasta=FIXTURE_DIR / "sample.fasta",
        bam_dir=None,
        out_dir=out_dir,
        input_already_demuxed=False,
        mapping_threshold=0.05,
        experiment_name="parallel_dispatch_test",
        mods=["6mA", "5mC"],
        batch_size=1,
        mod_tsv_dir=FIXTURE_DIR,
        delete_batch_hdfs=False,
        threads=4,
        double_barcoded_path=None,
        samtools_backend="auto",
        demux_backend="dorado",
        single_bam=FIXTURE_DIR / "sample.bam",
        barcode_sidecar=sidecar,
        max_workers=max_workers,
    )
    assert adata is not None
    return adata


def _summarize(adata) -> dict:
    return {
        "shape": adata.shape,
        "layers": sorted(adata.layers.keys()),
        "layer_sha256": {
            name: hashlib.sha256(np.nan_to_num(np.asarray(mat), nan=-99999.0).tobytes()).hexdigest()
            for name, mat in adata.layers.items()
        },
        "obs_names_sorted_sha256": hashlib.sha256(
            "\n".join(sorted(adata.obs_names)).encode()
        ).hexdigest(),
        "var_names_sha256": hashlib.sha256("\n".join(adata.var_names).encode()).hexdigest(),
    }


def test_parallel_dispatch_matches_serial_dispatch(tmp_path):
    serial = _summarize(_run(tmp_path, "serial", max_workers=None))
    parallel = _summarize(_run(tmp_path, "parallel", max_workers=4))

    assert serial["shape"] == parallel["shape"]
    assert serial["layers"] == parallel["layers"]
    assert serial["layer_sha256"] == parallel["layer_sha256"]
    assert serial["obs_names_sorted_sha256"] == parallel["obs_names_sorted_sha256"]
    assert serial["var_names_sha256"] == parallel["var_names_sha256"]


def test_auto_worker_count_matches_serial_dispatch(tmp_path):
    # "auto" must never crash and must still agree with serial on this tiny fixture,
    # regardless of what worker count it resolves to on the current machine.
    serial = _summarize(_run(tmp_path, "serial_for_auto", max_workers=None))
    auto = _summarize(_run(tmp_path, "auto", max_workers="auto"))

    assert serial["shape"] == auto["shape"]
    assert serial["layer_sha256"] == auto["layer_sha256"]


def test_nonsplit_chunks_are_genuinely_scoped_not_aliased(tmp_path):
    """Regression test for a real production OOM: non-split mode used to write one
    full-corpus cache/TSV per pseudo-sample "chunk" and alias the *same* file list
    onto every chunk's key, so every batch's worker had to load and filter down
    from the entire dataset just to use its own read subset. Every chunk here
    should now get its own distinct, non-overlapping on-disk files.
    """
    import anndata as ad
    import pandas as pd

    out_dir = tmp_path / "chunk_scoping"
    out_dir.mkdir()
    sidecar = tmp_path / "barcode_sidecar.parquet"
    n_reads = _build_barcode_sidecar(FIXTURE_DIR / "sample.bam", sidecar)
    assert n_reads >= 4, "fixture must have at least as many reads as chunks to be meaningful"

    modkit_extract_to_adata(
        fasta=FIXTURE_DIR / "sample.fasta",
        bam_dir=None,
        out_dir=out_dir,
        input_already_demuxed=False,
        mapping_threshold=0.05,
        experiment_name="chunk_scoping_test",
        mods=["6mA", "5mC"],
        batch_size=1,
        mod_tsv_dir=FIXTURE_DIR,
        delete_batch_hdfs=False,
        threads=4,
        double_barcoded_path=None,
        samtools_backend="auto",
        demux_backend="dorado",
        single_bam=FIXTURE_DIR / "sample.bam",
        barcode_sidecar=sidecar,
        max_workers=None,
    )

    tmp_dir = out_dir / "tmp"

    # Every chunk got its own pre-split, filtered TSV (not all four pointing at
    # the same full-corpus file).
    chunk_tsvs = sorted(tmp_dir.glob("tmp_extract_chunk_*.tsv.gz"))
    assert len(chunk_tsvs) == 4
    chunk_read_ids = [set(pd.read_csv(p, sep="\t")["read_id"]) for p in chunk_tsvs]
    # Disjoint: no read appears in more than one chunk's TSV.
    for i in range(len(chunk_read_ids)):
        for j in range(i + 1, len(chunk_read_ids)):
            assert chunk_read_ids[i].isdisjoint(chunk_read_ids[j])

    # Every chunk's sequence-cache h5ad file(s) hold a distinct read set too --
    # not every chunk sharing the identical full-corpus file(s).
    seq_files_by_chunk: dict[int, set[str]] = {}
    for h5 in tmp_dir.glob("tmp_*_fwd_*.h5ad"):
        chunk_idx = int(h5.name.split("_")[1])
        reads = set(ad.read_h5ad(h5).uns.keys())
        seq_files_by_chunk.setdefault(chunk_idx, set()).update(reads)
    assert len(seq_files_by_chunk) >= 2, "expected multiple distinct chunks to compare"
    chunks = list(seq_files_by_chunk.values())
    for i in range(len(chunks)):
        for j in range(i + 1, len(chunks)):
            assert chunks[i].isdisjoint(chunks[j]) or (not chunks[i] and not chunks[j])


def test_partial_batch_output_without_marker_is_reprocessed(tmp_path):
    """Regression test for a real production incident: a worker killed (by the
    memory watchdog) partway through writing a batch's several per-dict-type
    output files left some, but not all, of them on disk. Resumability used to
    be judged by "does any output file for this batch exist", which would treat
    that batch as done and permanently skip whichever dict types hadn't been
    written yet -- silent, hard-to-notice data loss. It must instead be judged
    by an explicit completion marker written only after every dict type for the
    batch has been handled.
    """
    out_dir = tmp_path / "partial_batch"
    out_dir.mkdir()
    sidecar = tmp_path / "barcode_sidecar.parquet"
    _build_barcode_sidecar(FIXTURE_DIR / "sample.bam", sidecar)

    run_kwargs = dict(
        fasta=FIXTURE_DIR / "sample.fasta",
        bam_dir=None,
        out_dir=out_dir,
        input_already_demuxed=False,
        mapping_threshold=0.05,
        experiment_name="partial_batch_test",
        mods=["6mA", "5mC"],
        batch_size=1,
        mod_tsv_dir=FIXTURE_DIR,
        delete_batch_hdfs=False,
        threads=4,
        double_barcoded_path=None,
        samtools_backend="auto",
        demux_backend="dorado",
        single_bam=FIXTURE_DIR / "sample.bam",
        barcode_sidecar=sidecar,
        max_workers=None,
    )

    modkit_extract_to_adata(**run_kwargs)

    h5_dir = out_dir / H5_DIR
    marker = h5_dir / "_batch_0_complete.marker"
    assert marker.exists(), "expected batch 0 to have completed on the first run"

    batch0_outputs = sorted(h5_dir.glob("*_0_*_SMF_binarized_sample_hdf5.h5ad.gz"))
    assert batch0_outputs, "expected batch 0 to have written at least one output file"

    # Reproduce the exact on-disk state a mid-write kill leaves behind: some
    # (but not all) of batch 0's output files, and no completion marker --
    # without needing to actually race a real kill.
    removed_file = batch0_outputs[0]
    removed_file.unlink()
    marker.unlink()

    modkit_extract_to_adata(**run_kwargs)

    assert marker.exists(), "batch 0 should have been reprocessed and re-marked complete"
    assert removed_file.exists(), "reprocessing batch 0 should have restored the deleted output"

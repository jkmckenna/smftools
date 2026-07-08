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

from smftools.informatics.modkit_extract_to_adata import modkit_extract_to_adata

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "_test_inputs" / "parallel_dispatch"


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
            name: hashlib.sha256(
                np.nan_to_num(np.asarray(mat), nan=-99999.0).tobytes()
            ).hexdigest()
            for name, mat in adata.layers.items()
        },
        "obs_names_sorted_sha256": hashlib.sha256(
            "\n".join(sorted(adata.obs_names)).encode()
        ).hexdigest(),
        "var_names_sha256": hashlib.sha256("\n".join(adata.var_names).encode()).hexdigest(),
    }


@pytest.mark.e2e
def test_parallel_dispatch_matches_serial_dispatch(tmp_path):
    serial = _summarize(_run(tmp_path, "serial", max_workers=None))
    parallel = _summarize(_run(tmp_path, "parallel", max_workers=4))

    assert serial["shape"] == parallel["shape"]
    assert serial["layers"] == parallel["layers"]
    assert serial["layer_sha256"] == parallel["layer_sha256"]
    assert serial["obs_names_sorted_sha256"] == parallel["obs_names_sorted_sha256"]
    assert serial["var_names_sha256"] == parallel["var_names_sha256"]


@pytest.mark.e2e
def test_auto_worker_count_matches_serial_dispatch(tmp_path):
    # "auto" must never crash and must still agree with serial on this tiny fixture,
    # regardless of what worker count it resolves to on the current machine.
    serial = _summarize(_run(tmp_path, "serial_for_auto", max_workers=None))
    auto = _summarize(_run(tmp_path, "auto", max_workers="auto"))

    assert serial["shape"] == auto["shape"]
    assert serial["layer_sha256"] == auto["layer_sha256"]

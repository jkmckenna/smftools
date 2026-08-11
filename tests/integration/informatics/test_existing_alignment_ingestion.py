import json
from array import array
from importlib import resources

import pytest

from smftools.cli.helpers import get_adata_paths
from smftools.cli.load_adata import load_adata_core
from smftools.config import ExperimentConfig
from smftools.informatics.sidecar_manifest import resolve_sidecar

pysam = pytest.importorskip("pysam")
pytestmark = pytest.mark.integration


def test_existing_direct_alignment_reaches_raw_without_aligner(tmp_path, monkeypatch):
    fasta = tmp_path / "reference.fa"
    fasta.write_text(">ref\nACGTACGTACGT\n", encoding="utf-8")
    bam_path = tmp_path / "external.bam"
    header = {
        "HD": {"VN": "1.6", "SO": "coordinate"},
        "SQ": [{"SN": "ref", "LN": 12}],
        "PG": [{"ID": "external", "PN": "external-aligner", "VN": "1"}],
    }
    with pysam.AlignmentFile(str(bam_path), "wb", header=header) as bam:
        read = pysam.AlignedSegment()
        read.query_name = "read-1"
        read.query_sequence = "ACGT"
        read.query_qualities = pysam.qualitystring_to_array("IIII")
        read.reference_id = 0
        read.reference_start = 0
        read.cigarstring = "4M"
        read.mapping_quality = 60
        read.set_tag("BC", "barcode01")
        read.set_tag("MM", "C+m,0;")
        read.set_tag("ML", array("B", [200]))
        bam.write(read)
    pysam.index(str(bam_path))
    defaults_dir = resources.files("smftools").joinpath("config")
    cfg, _ = ExperimentConfig.from_var_dict(
        {
            "alignment_mode": "existing",
            "input_data_path": str(bam_path),
            "output_directory": str(tmp_path / "run"),
            "fasta": str(fasta),
            "experiment_name": "existing-direct",
            "smf_modality": "direct",
            "direct_signal_backend": "pysam",
            "input_already_demuxed": True,
            "skip_bam_split": True,
            "skip_bam_qc": True,
            "threads": 1,
        },
        defaults_dir=defaults_dir,
    )
    monkeypatch.setattr(
        "smftools.informatics.bam_functions.align_and_sort_BAM",
        lambda *_args, **_kwargs: pytest.fail("existing mode invoked an aligner"),
    )

    spine, spine_path, _ = load_adata_core(
        cfg,
        get_adata_paths(cfg),
        raw_only=True,
    )

    assert spine_path.is_file()
    assert spine.n_obs == 1
    assert spine.obs.iloc[0]["Barcode"] == "barcode01"
    sidecars = tmp_path / "run" / "raw_outputs" / "sidecar_manifest.json"
    alignment_manifest = resolve_sidecar(sidecars, "alignment_manifest")
    assert alignment_manifest is not None and alignment_manifest.is_file()
    payload = json.loads(alignment_manifest.read_text())
    assert payload["validation"]["source"]["source_index_valid"] is True
    assert payload["validation"]["source"]["external_aligner"] == "external-aligner"

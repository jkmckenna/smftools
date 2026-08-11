import json
from types import SimpleNamespace

import pandas as pd
import pytest

from smftools.cli.load_adata import (
    _attach_dense_barcode_identity,
    _barcode_classifier_source,
    _publish_canonical_barcode_identity,
)
from smftools.cli.raw_adata import _attach_obs_metadata
from smftools.informatics.bam_functions import concatenate_fastqs_to_bam
from smftools.informatics.barcode_sidecar import (
    BARCODE_IDENTITY_COLUMNS,
    publish_barcode_identity_sidecar,
    read_barcode_identity_sidecar,
)
from smftools.informatics.sidecar_manifest import resolve_sidecar

pysam = pytest.importorskip("pysam")


def _bam(path, records, *, read_groups=()):
    header = {
        "HD": {"VN": "1.6", "SO": "unknown"},
        "SQ": [],
        "RG": list(read_groups),
    }
    with pysam.AlignmentFile(str(path), "wb", header=header) as bam:
        for name, tags in records:
            read = pysam.AlignedSegment()
            read.query_name = name
            read.query_sequence = "ACGT"
            read.query_qualities = pysam.qualitystring_to_array("IIII")
            read.is_unmapped = True
            for key, value in tags.items():
                read.set_tag(key, value)
            bam.write(read)
    return path


def _manifest_row(path, **values):
    defaults = {
        "path": str(path),
        "source_kind": "",
        "source_id": "",
        "pair_id": "",
        "barcode": "",
        "sample": "",
        "read_group": "",
        "namespace": "",
    }
    defaults.update(values)
    return SimpleNamespace(**defaults)


def test_manifest_overrides_lower_authorities_and_records_conflict(tmp_path):
    bam = _bam(
        tmp_path / "reads.bam",
        [("read-1", {"BC": "bam-barcode", "RG": "rg-1"})],
        read_groups=[{"ID": "rg-1", "SM": "bam-sample"}],
    )
    classifier = tmp_path / "classifier.parquet"
    pd.DataFrame({"read_name": ["read-1"], "BC": ["sequence-barcode"], "BM": ["both"]}).to_parquet(
        classifier, index=False
    )
    manifest = [
        _manifest_row(
            bam,
            barcode="manifest-barcode",
            sample="manifest-sample",
            namespace="experiment-a",
        )
    ]

    sidecar, report = publish_barcode_identity_sidecar(
        bam,
        tmp_path / "identity.parquet",
        input_manifest=manifest,
        classifier_sidecar=classifier,
        classifier_source="sequence:smftools",
    )

    row = pd.read_parquet(sidecar).iloc[0]
    assert set(BARCODE_IDENTITY_COLUMNS).issubset(row.index)
    assert row["barcode"] == "manifest-barcode"
    assert row["barcode_source"] == "manifest"
    assert row["sample"] == "manifest-sample"
    assert row["sample_source"] == "manifest"
    assert row["namespace"] == "experiment-a"
    assert row["identity_status"] == "conflicting"
    assert {item["source"] for item in json.loads(row["identity_conflicts"])} >= {
        "bam:BC",
        "bam:SM",
    }
    assert json.loads(report.read_text())["status_counts"]["conflicting"] == 1


def test_rg_sm_only_bam_resolves_barcode_and_sample(tmp_path):
    bam = _bam(
        tmp_path / "reads.bam",
        [("read-1", {"RG": "group-a"})],
        read_groups=[{"ID": "group-a", "SM": "sample-a"}],
    )

    sidecar, _ = publish_barcode_identity_sidecar(bam, tmp_path / "identity.parquet")

    row = pd.read_parquet(sidecar).iloc[0]
    assert row["barcode"] == "group-a"
    assert row["barcode_source"] == "bam:RG"
    assert row["sample"] == "sample-a"
    assert row["sample_source"] == "bam:SM"
    assert row["identity_status"] == "classified"


@pytest.mark.parametrize("source", ["sequence:dorado", "sequence:smftools"])
def test_classifiers_publish_the_same_canonical_columns(tmp_path, source):
    bam = _bam(tmp_path / "reads.bam", [("read-1", {})])
    classifier = tmp_path / "classifier.parquet"
    pd.DataFrame({"read_name": ["read-1"], "BC": ["barcode01"], "BM": ["both"]}).to_parquet(
        classifier, index=False
    )

    sidecar, _ = publish_barcode_identity_sidecar(
        bam,
        tmp_path / "identity.parquet",
        classifier_sidecar=classifier,
        classifier_source=source,
    )

    row = pd.read_parquet(sidecar).iloc[0]
    assert row["barcode"] == "barcode01"
    assert row["sample"] == "barcode01"
    assert row["barcode_source"] == source
    assert row["BM"] == "both"


@pytest.mark.parametrize(
    ("already_demuxed", "barcode_kit", "backend", "expected"),
    [
        (True, "kit", "smftools", "filename"),
        (False, None, "smftools", "filename"),
        (False, "kit", "smftools", "sequence:smftools"),
        (False, "kit", "dorado", "sequence:dorado"),
    ],
)
def test_route_classifier_source(already_demuxed, barcode_kit, backend, expected):
    cfg = SimpleNamespace(
        input_already_demuxed=already_demuxed,
        barcode_kit=barcode_kit,
    )

    assert _barcode_classifier_source(cfg, demux_backend=backend) == expected


def test_filename_fallback_warns_and_does_not_use_mate_token(tmp_path):
    bam = _bam(tmp_path / "reads.bam", [("read-1", {})])
    manifest = [_manifest_row(tmp_path / "sample_R1.fastq.gz")]

    with pytest.warns(UserWarning, match="filename fallback"):
        sidecar, _ = publish_barcode_identity_sidecar(
            bam,
            tmp_path / "identity.parquet",
            input_manifest=manifest,
            classifier_source="filename",
        )

    row = pd.read_parquet(sidecar).iloc[0]
    assert row["barcode"] == "sample"
    assert row["barcode"] != "R1"
    assert row["barcode_source"] == "filename"


def test_duplicate_classifier_assignments_are_conflicting(tmp_path):
    bam = _bam(tmp_path / "reads.bam", [("read-1", {})])
    classifier = tmp_path / "classifier.parquet"
    pd.DataFrame({"read_name": ["read-1", "read-1"], "BC": ["barcode01", "barcode02"]}).to_parquet(
        classifier, index=False
    )

    sidecar, _ = publish_barcode_identity_sidecar(
        bam, tmp_path / "identity.parquet", classifier_sidecar=classifier
    )

    row = pd.read_parquet(sidecar).iloc[0]
    assert row["identity_status"] == "conflicting"
    assert "barcode01|barcode02" in row["identity_conflicts"]


def test_missing_classifier_assignment_is_reported_unclassified(tmp_path):
    bam = _bam(tmp_path / "reads.bam", [("read-1", {}), ("read-2", {})])
    classifier = tmp_path / "classifier.parquet"
    pd.DataFrame({"read_name": ["read-1"], "BC": ["barcode01"]}).to_parquet(classifier, index=False)

    sidecar, report = publish_barcode_identity_sidecar(
        bam,
        tmp_path / "identity.parquet",
        classifier_sidecar=classifier,
        classifier_source="sequence:smftools",
    )

    frame = pd.read_parquet(sidecar).set_index("read_name")
    assert frame.loc["read-2", "barcode"] == "unclassified"
    assert frame.loc["read-2", "identity_status"] == "unclassified"
    payload = json.loads(report.read_text())
    assert payload["status_counts"]["classified"] == 1
    assert payload["status_counts"]["unclassified"] == 1
    assert payload["status_fractions"]["unclassified"] == 0.5


def test_legacy_sidecar_has_deterministic_compatibility(tmp_path):
    path = tmp_path / "legacy.parquet"
    pd.DataFrame({"read_name": ["read-1"], "BC": ["barcode01"]}).to_parquet(path, index=False)

    with pytest.warns(UserWarning, match="legacy barcode sidecar"):
        frame = read_barcode_identity_sidecar(path)

    assert frame.loc[0, "barcode"] == "barcode01"
    assert frame.loc[0, "sample"] == "barcode01"
    assert frame.loc[0, "barcode_source"] == "legacy_sidecar"


def test_fastq_normalization_preserves_per_source_rg_and_sample(tmp_path):
    fastq = tmp_path / "reads.fastq"
    fastq.write_text("@read-1\nACGT\n+\nIIII\n", encoding="utf-8")
    output = tmp_path / "reads.bam"

    concatenate_fastqs_to_bam(
        [fastq],
        output,
        barcode_map={fastq: "barcode01"},
        read_group_map={fastq: "source-group"},
        sample_map={fastq: "sample-a"},
        progress=False,
        auto_pair=False,
        samtools_backend="python",
    )

    with pysam.AlignmentFile(str(output), "rb", check_sq=False) as bam:
        read = next(iter(bam))
        assert read.get_tag("BC") == "barcode01"
        assert read.get_tag("RG") == "source-group"
        assert bam.header.to_dict()["RG"] == [{"ID": "source-group", "SM": "sample-a"}]


def test_generated_fastq_tags_preserve_filename_fallback_provenance(tmp_path):
    fastq = tmp_path / "sample_R1.fastq"
    fastq.write_text("@read-1\nACGT\n+\nIIII\n", encoding="utf-8")
    bam = tmp_path / "reads.bam"
    concatenate_fastqs_to_bam(
        [fastq],
        bam,
        read_group_map={fastq: "source-1"},
        progress=False,
        auto_pair=False,
        samtools_backend="python",
    )
    manifest = [_manifest_row(fastq, source_kind="fastq", source_id="source-1")]

    with pytest.warns(UserWarning, match="filename fallback"):
        sidecar, _ = publish_barcode_identity_sidecar(
            bam,
            tmp_path / "identity.parquet",
            input_manifest=manifest,
            classifier_source="filename",
        )

    row = pd.read_parquet(sidecar).iloc[0]
    assert row["barcode"] == "sample"
    assert row["barcode_source"] == "filename"


def test_raw_metadata_consumes_canonical_sample_and_namespace(tmp_path):
    sidecar = tmp_path / "identity.parquet"
    row = {column: "" for column in BARCODE_IDENTITY_COLUMNS}
    row.update(
        {
            "identity_schema_version": 1,
            "read_name": "read-1",
            "barcode": "barcode01",
            "barcode_source": "manifest",
            "barcode_confidence": 1.0,
            "sample": "sample-a",
            "sample_source": "manifest",
            "sample_confidence": 1.0,
            "namespace": "project-exp-a",
            "identity_status": "classified",
            "identity_conflicts": "[]",
        }
    )
    unclassified = row.copy()
    unclassified.update(
        {
            "read_name": "read-2",
            "barcode": "unclassified",
            "barcode_source": "sequence:smftools",
            "barcode_confidence": 0.75,
            "sample": "unclassified",
            "sample_source": "sequence:smftools:barcode",
            "sample_confidence": 0.75,
            "identity_status": "unclassified",
        }
    )
    pd.DataFrame([row, unclassified]).to_parquet(sidecar, index=False)
    frame = pd.DataFrame(
        {
            "read_id": ["read-1", "read-2"],
            "cigar": ["4M", "4M"],
            "reference": ["ref", "ref"],
            "reference_start": [0, 0],
        }
    )
    cfg = SimpleNamespace(
        experiment_name="experiment-a",
        samtools_backend="python",
        skip_unclassified=False,
    )

    result = _attach_obs_metadata(
        frame,
        cfg=cfg,
        bam_path=tmp_path / "unused.bam",
        barcode_sidecar=sidecar,
        umi_sidecar=None,
        metrics={
            "read-1": (4, 30, 4, 4, 60, 0, 4),
            "read-2": (4, 30, 4, 4, 60, 0, 4),
        },
    )

    assert result.loc[0, "barcode"] == "barcode01"
    assert result.loc[0, "sample"] == "sample-a"
    assert result.loc[0, "barcode_source"] == "manifest"
    assert result.loc[0, "Experiment_name_and_barcode"] == "project-exp-a_barcode01"
    cfg.skip_unclassified = True
    filtered = _attach_obs_metadata(
        frame,
        cfg=cfg,
        bam_path=tmp_path / "unused.bam",
        barcode_sidecar=sidecar,
        umi_sidecar=None,
        metrics={
            "read-1": (4, 30, 4, 4, 60, 0, 4),
            "read-2": (4, 30, 4, 4, 60, 0, 4),
        },
    )
    assert filtered["read_id"].tolist() == ["read-1"]


def test_dense_metadata_preserves_manifest_namespace(tmp_path):
    sidecar = tmp_path / "identity.parquet"
    row = {column: "" for column in BARCODE_IDENTITY_COLUMNS}
    row.update(
        {
            "identity_schema_version": 1,
            "read_name": "read-1",
            "barcode": "barcode01",
            "barcode_source": "manifest",
            "barcode_confidence": 1.0,
            "sample": "sample-a",
            "sample_source": "manifest",
            "sample_confidence": 1.0,
            "namespace": "project-exp-a",
            "identity_status": "classified",
            "identity_conflicts": "[]",
        }
    )
    pd.DataFrame([row]).to_parquet(sidecar, index=False)
    raw_adata = SimpleNamespace(
        obs=pd.DataFrame({"Barcode": ["legacy"]}, index=["read-1"]),
        obs_names=pd.Index(["read-1"]),
    )

    _attach_dense_barcode_identity(raw_adata, sidecar, "experiment-a")

    assert raw_adata.obs.loc["read-1", "sample"] == "sample-a"
    assert raw_adata.obs.loc["read-1", "Experiment_name_and_barcode"] == "project-exp-a_barcode01"


def test_non_split_bc_tagged_bam_publishes_and_reuses_canonical_sidecar(tmp_path):
    bam = _bam(tmp_path / "already-demuxed.bam", [("read-1", {"BC": "barcode01"})])
    manifest = SimpleNamespace(
        digest="manifest-digest",
        rows=(_manifest_row(bam),),
    )
    sidecar_manifest = tmp_path / "raw_outputs" / "sidecar_manifest.json"

    first = _publish_canonical_barcode_identity(
        output_directory=tmp_path,
        aligned_bam=bam,
        resolved_input_manifest=manifest,
        route_sidecar=None,
        classifier_source="filename",
        sidecar_manifest=sidecar_manifest,
        force_redo=False,
    )
    second = _publish_canonical_barcode_identity(
        output_directory=tmp_path,
        aligned_bam=bam,
        resolved_input_manifest=manifest,
        route_sidecar=None,
        classifier_source="filename",
        sidecar_manifest=sidecar_manifest,
        force_redo=False,
    )

    assert second == first
    row = pd.read_parquet(first[0]).iloc[0]
    assert row["barcode"] == "barcode01"
    assert row["barcode_source"] == "bam:BC"
    assert json.loads(first[1].read_text())["total_reads"] == 1
    assert resolve_sidecar(sidecar_manifest, "barcode") == first[0]
    assert resolve_sidecar(sidecar_manifest, "barcode_identity_report") == first[1]

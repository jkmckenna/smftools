from types import SimpleNamespace

from smftools.cli.helpers import (
    artifact_is_ready,
    artifact_manifest_path,
    get_artifact_paths,
    read_artifact_manifest,
    record_artifact_step,
    register_artifact,
    write_artifact_manifest,
)


def test_get_artifact_paths_for_direct_pod5(tmp_path):
    cfg = SimpleNamespace(
        output_directory=str(tmp_path / "outputs"),
        split_path=tmp_path / "outputs" / "bam_outputs" / "split_bams",
        bam_suffix=".bam",
        input_type="pod5",
        smf_modality="direct",
        model="hac",
        mod_list=["6mA"],
        experiment_name="exp1",
    )

    paths = get_artifact_paths(cfg)

    assert paths.load_directory.name == "load_adata_outputs"
    assert paths.bam_outputs_directory.name == "bam_outputs"
    assert paths.fasta_outputs_directory.name == "fasta_outputs"
    assert paths.bed_outputs_directory.name == "bed_outputs"
    assert paths.modkit_outputs_directory.name == "modkit_outputs"
    assert paths.mod_tsv_directory.parent == paths.modkit_outputs_directory
    assert paths.mod_bed_directory.parent == paths.modkit_outputs_directory
    assert paths.unaligned_bam.name == "hac_6mA_calls.bam"
    assert paths.unaligned_bam.parent == paths.bam_outputs_directory
    assert paths.aligned_sorted_bam.name == "hac_6mA_calls_aligned_sorted.bam"
    assert paths.aligned_sorted_bam.parent == paths.bam_outputs_directory
    assert paths.barcode_sidecar.name.endswith(".barcode_tags.parquet")
    assert paths.umi_oriented_sidecar.name.endswith(".umi_tags.parquet")


def test_get_artifact_paths_for_bam_input(tmp_path):
    cfg = SimpleNamespace(
        output_directory=str(tmp_path / "outputs"),
        split_path=tmp_path / "outputs" / "bam_outputs" / "split_bams",
        bam_suffix=".bam",
        input_type="bam",
        input_data_path=tmp_path / "inputs" / "sample_input.bam",
        smf_modality="conversion",
        experiment_name="exp1",
    )

    paths = get_artifact_paths(cfg)
    assert paths.bam_outputs_directory.name == "bam_outputs"
    assert paths.fasta_outputs_directory.name == "fasta_outputs"
    assert paths.bed_outputs_directory.name == "bed_outputs"
    assert paths.modkit_outputs_directory.name == "modkit_outputs"
    assert paths.unaligned_bam.name == "sample_input.bam"
    assert paths.aligned_bam.name == "sample_input_aligned.bam"
    assert paths.aligned_sorted_bam.name == "sample_input_aligned_sorted.bam"


def test_artifact_manifest_roundtrip(tmp_path):
    out_dir = tmp_path / "outputs"
    manifest_path = artifact_manifest_path(out_dir)

    manifest = read_artifact_manifest(manifest_path)
    assert manifest["version"] == 1
    assert "artifacts" in manifest
    assert "steps" in manifest

    register_artifact(
        manifest,
        key="aligned_sorted_bam",
        path=out_dir / "load_adata_outputs" / "x_aligned_sorted.bam",
        producer_step="align",
        status="ready",
        metadata={"reference": "ref.fa"},
    )
    record_artifact_step(
        manifest,
        step="align",
        inputs=["unaligned_bam"],
        outputs=["aligned_sorted_bam"],
        params={"aligner": "minimap2"},
    )

    write_artifact_manifest(manifest_path, manifest)
    loaded = read_artifact_manifest(manifest_path)

    assert artifact_is_ready(loaded, "aligned_sorted_bam")
    assert loaded["artifacts"]["aligned_sorted_bam"]["producer_step"] == "align"
    assert loaded["steps"][-1]["step"] == "align"

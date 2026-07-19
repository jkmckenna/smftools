from pathlib import Path

import smftools.informatics.bam_functions as bam_functions


def test_rebuild_barcode_sidecar_runs_real_classification_and_discards_split_bams(
    tmp_path, monkeypatch
):
    # Regression test: skip_bam_split=True callers whose aligned BAM has no
    # usable BC/bi tags used to fall back to scanning the aligned BAM itself
    # for BC tags that were never written, silently producing an
    # almost-empty sidecar (observed: 156 reads recovered out of >1M) rather
    # than failing loudly. rebuild_barcode_sidecar_via_dorado_classification
    # replaces that with a real dorado demux classification pass.
    aligned_bam = tmp_path / "aligned_sorted.bam"
    aligned_bam.write_bytes(b"stub")
    barcode_sidecar = tmp_path / "aligned_sorted.barcode_tags.parquet"

    captured_demux_kwargs = {}
    fake_split_dir_seen = {}

    def fake_demux_and_index_BAM(
        input_bam,
        split_dir,
        bam_suffix,
        barcode_kit,
        barcode_both_ends,
        trim,
        threads,
        no_classify=False,
        file_prefix=None,
    ):
        # The split directory must be a real, writable temp dir that gets
        # cleaned up afterward -- not cfg.split_path (the "real" split-mode
        # output location), honoring skip_bam_split=True's promise that no
        # split BAMs persist.
        assert Path(split_dir).is_dir()
        fake_split_dir_seen["path"] = Path(split_dir)
        captured_demux_kwargs.update(
            input_bam=input_bam,
            barcode_kit=barcode_kit,
            barcode_both_ends=barcode_both_ends,
            trim=trim,
            threads=threads,
            no_classify=no_classify,
        )
        classified = Path(split_dir) / "barcode01.bam"
        classified.write_bytes(b"stub")
        unclassified = Path(split_dir) / "unclassified.bam"
        unclassified.write_bytes(b"stub")
        return [classified, unclassified]

    captured_sidecar_kwargs = {}

    def fake_build_barcode_sidecar_from_split_bams(bam_files, output_path, samtools_backend=None):
        captured_sidecar_kwargs["bam_files"] = bam_files
        captured_sidecar_kwargs["output_path"] = output_path
        captured_sidecar_kwargs["samtools_backend"] = samtools_backend
        Path(output_path).write_bytes(b"stub-sidecar")
        return output_path

    monkeypatch.setattr(bam_functions, "demux_and_index_BAM", fake_demux_and_index_BAM)
    monkeypatch.setattr(
        bam_functions,
        "build_barcode_sidecar_from_split_bams",
        fake_build_barcode_sidecar_from_split_bams,
    )

    result = bam_functions.rebuild_barcode_sidecar_via_dorado_classification(
        aligned_bam,
        barcode_sidecar,
        barcode_kit="SQK-NBD114-24",
        barcode_both_ends=False,
        trim=False,
        threads=12,
        samtools_backend="auto",
    )

    assert result == barcode_sidecar
    assert captured_demux_kwargs["input_bam"] == aligned_bam
    assert captured_demux_kwargs["barcode_kit"] == "SQK-NBD114-24"
    assert captured_demux_kwargs["no_classify"] is False, (
        "must run real classification, not --no-classify (which just splits by "
        "whatever BC tags already exist -- the exact thing that produced an "
        "almost-empty sidecar in the first place)"
    )

    # Only the classified BAM is passed through, never "unclassified".
    sidecar_bam_names = [p.name for p in captured_sidecar_kwargs["bam_files"]]
    assert sidecar_bam_names == ["barcode01.bam"]
    assert captured_sidecar_kwargs["output_path"] == barcode_sidecar

    # The temporary split directory is cleaned up afterward -- no split BAMs
    # persist alongside the aligned BAM.
    assert not fake_split_dir_seen["path"].exists()

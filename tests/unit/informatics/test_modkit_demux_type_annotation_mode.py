from smftools.informatics.modkit_extract_to_adata import (
    _resolve_demux_type_annotation_mode,
)


def test_input_already_demuxed_wins_regardless_of_backend() -> None:
    assert _resolve_demux_type_annotation_mode(True, "dorado", None) == "already"
    assert _resolve_demux_type_annotation_mode(True, "smftools", "some/path") == "already"


def test_smftools_backend_skips_annotation() -> None:
    assert _resolve_demux_type_annotation_mode(False, "smftools", None) == "skip_smftools"
    assert _resolve_demux_type_annotation_mode(False, "SMFtools", "some/path") == "skip_smftools"


def test_dorado_backend_with_double_barcoded_path_uses_barcoding_summary() -> None:
    assert (
        _resolve_demux_type_annotation_mode(False, "dorado", "some/path")
        == "dorado_barcoding_summary"
    )


def test_dorado_backend_without_double_barcoded_path_skips_instead_of_crashing() -> None:
    # Regression test: skip_bam_split=True + demux_backend=dorado leaves
    # double_barcoded_path=None (no physical BAM splitting -> no dorado
    # barcoding_summary.txt). This must resolve to a skip, not attempt
    # `None / "barcoding_summary.txt"` (TypeError).
    assert (
        _resolve_demux_type_annotation_mode(False, "dorado", None) == "skip_no_double_barcoded_path"
    )
    assert _resolve_demux_type_annotation_mode(False, None, None) == "skip_no_double_barcoded_path"

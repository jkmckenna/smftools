from pathlib import Path

from smftools.informatics.converted_BAM_to_adata import _barcode_label_from_sample_name
from smftools.informatics.modkit_extract_to_adata import _build_sample_maps


def test_barcode_label_from_sample_name_handles_numeric_suffix():
    assert _barcode_label_from_sample_name("sample_barcode0007") == "0007"


def test_barcode_label_from_sample_name_falls_back_to_sample_name():
    assert _barcode_label_from_sample_name("sample_NB01") == "sample_NB01"


def test_build_sample_maps_supports_non_numeric_barcode_suffix():
    bams = [
        Path("sampleA_barcode0001.bam"),
        Path("sampleB_NB01.bam"),
    ]

    sample_map, barcode_map = _build_sample_maps(bams)

    assert sample_map[0] == "sampleA_barcode0001"
    assert barcode_map[0] == "0001"
    assert sample_map[1] == "sampleB_NB01_sampleB_NB01"
    assert barcode_map[1] == "sampleB_NB01"

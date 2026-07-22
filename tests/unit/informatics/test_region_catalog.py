from types import SimpleNamespace

import pandas as pd
import pyarrow.parquet as pq
import pytest

from smftools.informatics.region_catalog import (
    COORDINATE_SYSTEM,
    build_reference_interval_map,
    normalize_bed_catalog,
    write_reference_interval_map,
    write_region_catalogs,
)


def _fasta(path, records):
    path.write_text(
        "".join(f">{name}\n{sequence}\n" for name, sequence in records.items()),
        encoding="utf-8",
    )
    return path


def test_normalize_bed_preserves_records_and_annotates_overlap_and_adjacency(tmp_path):
    bed = tmp_path / "regions.bed"
    bed.write_text(
        "# original coordinates\n"
        "chr1\t15\t25\tthird\t30\t.\n"
        "chr1\t0\t10\tfirst\t10\t+\n"
        "chr1\t10\t20\tsecond\t20\t-\n",
        encoding="utf-8",
    )

    catalog = normalize_bed_catalog(bed, scope="analysis", reference_lengths={"chr1": 30})

    assert catalog["name"].tolist() == ["first", "second", "third"]
    assert catalog["overlaps_previous"].tolist() == [False, False, True]
    assert catalog["adjacent_previous"].tolist() == [False, True, False]
    assert catalog["coordinate_system"].unique().tolist() == [COORDINATE_SYSTEM]
    assert catalog["region_id"].is_unique
    assert catalog["source_sha256"].str.fullmatch(r"[0-9a-f]{64}").all()


def test_region_ids_do_not_depend_on_source_row_order(tmp_path):
    first = tmp_path / "first.bed"
    second = tmp_path / "second.bed"
    first.write_text("chr1\t0\t10\ta\nchr1\t20\t30\tb\n", encoding="utf-8")
    second.write_text("chr1\t20\t30\tb\nchr1\t0\t10\ta\n", encoding="utf-8")

    a = normalize_bed_catalog(first, scope="plot", reference_lengths={"chr1": 30})
    b = normalize_bed_catalog(second, scope="plot", reference_lengths={"chr1": 30})

    assert a[["name", "region_id"]].to_dict("records") == b[["name", "region_id"]].to_dict(
        "records"
    )
    assert a["source_sha256"].iloc[0] != b["source_sha256"].iloc[0]


def test_comment_only_bed_has_portable_empty_schema(tmp_path):
    bed = tmp_path / "empty.bed"
    bed.write_text("# no regions\ntrack name=empty\n", encoding="utf-8")

    catalog = normalize_bed_catalog(bed, scope="analysis", reference_lengths={"chr1": 30})

    assert catalog.empty
    assert catalog.dtypes.astype(str).to_dict() == {
        "schema_version": "int16",
        "scope": "string",
        "region_id": "string",
        "original_reference": "string",
        "original_start": "int64",
        "original_end": "int64",
        "name": "string",
        "score": "Float64",
        "strand": "string",
        "source_row": "int64",
        "source_filename": "string",
        "source_sha256": "string",
        "coordinate_system": "string",
        "overlaps_previous": "bool",
        "adjacent_previous": "bool",
    }


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ("chr1\t0\n", "BED3-BED6"),
        ("missing\t0\t2\n", "absent from the original FASTA"),
        ("chr1\t-1\t2\n", "0 <= start < end"),
        ("chr1\t2\t2\n", "0 <= start < end"),
        ("chr1\t0\t31\n", "exceeds original FASTA length"),
        ("chr1\t0\t2\ta\tinvalid\n", "score must be numeric"),
        ("chr1\t0\t2\ta\t1\t?\n", "strand must be"),
    ],
)
def test_invalid_bed_rows_fail_clearly(tmp_path, row, message):
    bed = tmp_path / "invalid.bed"
    bed.write_text(row, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        normalize_bed_catalog(bed, scope="analysis", reference_lengths={"chr1": 30})


def test_duplicate_names_and_alignment_intervals_are_rejected(tmp_path):
    duplicate_names = tmp_path / "names.bed"
    duplicate_names.write_text("chr1\t0\t5\ta\nchr1\t5\t10\ta\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate BED name"):
        normalize_bed_catalog(duplicate_names, scope="plot", reference_lengths={"chr1": 20})

    duplicate_intervals = tmp_path / "alignment.bed"
    duplicate_intervals.write_text("chr1\t0\t5\ta\nchr1\t0\t5\tb\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate alignment interval"):
        normalize_bed_catalog(
            duplicate_intervals, scope="alignment", reference_lengths={"chr1": 20}
        )


@pytest.mark.parametrize("modality", ["direct", "deaminase"])
def test_full_reference_map_covers_both_stored_strands(tmp_path, modality):
    fasta = _fasta(tmp_path / "reference.fa", {"chr1": "ACGT", "chr2": "AAAAAA"})

    mapping = build_reference_interval_map(
        original_fasta=fasta,
        alignment_fasta=fasta,
        modality=modality,
        conversions=["unconverted", "5mC"],
        strands=["bottom", "top"],
    )

    assert set(mapping["stored_reference"]) == {
        "chr1_top",
        "chr1_bottom",
        "chr2_top",
        "chr2_bottom",
    }
    assert set(mapping["alignment_reference"]) == {"chr1", "chr2"}
    assert set(mapping["reference_kind"]) == {"full_reference"}
    assert mapping["mapping_id"].is_unique


def test_conversion_map_connects_all_alignment_variants_to_stored_strands(tmp_path):
    original = _fasta(tmp_path / "reference.fa", {"chr1": "ACGT"})
    converted = _fasta(
        tmp_path / "reference_converted.fa",
        {
            "chr1_unconverted_top": "ACGT",
            "chr1_5mC_bottom": "ACGT",
            "chr1_5mC_top": "ACGT",
        },
    )

    mapping = build_reference_interval_map(
        original_fasta=original,
        alignment_fasta=converted,
        modality="conversion",
        conversions=["unconverted", "5mC"],
        strands=["bottom", "top"],
    )

    assert set(mapping["alignment_reference"]) == {
        "chr1_unconverted_top",
        "chr1_5mC_bottom",
        "chr1_5mC_top",
    }
    assert mapping.groupby("stored_reference").size().to_dict() == {
        "chr1_bottom": 1,
        "chr1_top": 2,
    }
    assert set(mapping["original_reference"]) == {"chr1"}
    assert set(mapping["original_start"]) == {0}
    assert set(mapping["original_end"]) == {4}


def test_reduced_conversion_map_restores_original_offsets(tmp_path):
    original = _fasta(tmp_path / "reference.fa", {"chr1": "A" * 30})
    bed = tmp_path / "alignment.bed"
    bed.write_text("chr1\t10\t20\tpeak\n", encoding="utf-8")
    catalog = normalize_bed_catalog(bed, scope="alignment", reference_lengths={"chr1": 30})
    converted = _fasta(
        tmp_path / "reduced_converted.fa",
        {
            "chr1:10-20_unconverted_top": "A" * 10,
            "chr1:10-20_5mC_bottom": "A" * 10,
            "chr1:10-20_5mC_top": "A" * 10,
        },
    )

    mapping = build_reference_interval_map(
        original_fasta=original,
        alignment_fasta=converted,
        modality="conversion",
        conversions=["unconverted", "5mC"],
        strands=["bottom", "top"],
        alignment_catalog=catalog,
    )

    assert set(mapping["stored_reference"]) == {
        "chr1:10-20_top",
        "chr1:10-20_bottom",
    }
    assert set(mapping["original_reference"]) == {"chr1"}
    assert set(mapping["stored_start"]) == {0}
    assert set(mapping["stored_end"]) == {10}
    assert set(mapping["original_start"]) == {10}
    assert set(mapping["original_end"]) == {20}
    assert set(mapping["source_region_id"]) == {catalog["region_id"].iloc[0]}


@pytest.mark.parametrize("modality", ["direct", "deaminase"])
def test_reduced_direct_and_deaminase_maps_restore_original_offsets(tmp_path, modality):
    original = _fasta(tmp_path / "reference.fa", {"chr1": "A" * 30})
    bed = tmp_path / "alignment.bed"
    bed.write_text("chr1\t10\t20\tpeak\n", encoding="utf-8")
    catalog = normalize_bed_catalog(bed, scope="alignment", reference_lengths={"chr1": 30})
    reduced = _fasta(tmp_path / "reduced.fa", {"chr1:10-20": "A" * 10})

    mapping = build_reference_interval_map(
        original_fasta=original,
        alignment_fasta=reduced,
        modality=modality,
        conversions=["unconverted", "5mC"],
        strands=["bottom", "top"],
        alignment_catalog=catalog,
    )

    assert set(mapping["stored_reference"]) == {
        "chr1:10-20_top",
        "chr1:10-20_bottom",
    }
    assert set(mapping["original_start"]) == {10}
    assert set(mapping["original_end"]) == {20}


def test_catalog_and_reference_map_publication(tmp_path):
    original = _fasta(tmp_path / "reference.fa", {"chr1": "ACGT"})
    analysis = tmp_path / "analysis.bed"
    plot = tmp_path / "plot.bed"
    analysis.write_text("chr1\t0\t2\ta\n", encoding="utf-8")
    plot.write_text("# intentionally empty\n", encoding="utf-8")
    cfg = SimpleNamespace(
        alignment_regions_bed=None,
        analysis_regions_bed=str(analysis),
        plot_regions_bed=str(plot),
    )

    catalogs = write_region_catalogs(cfg, original_fasta=original, run_root=tmp_path / "run")
    map_path = write_reference_interval_map(
        run_root=tmp_path / "run",
        original_fasta=original,
        alignment_fasta=original,
        modality="direct",
        conversions=["unconverted"],
        strands=["bottom", "top"],
    )

    assert set(catalogs) == {"analysis", "plot"}
    assert pd.read_parquet(catalogs["analysis"])["name"].tolist() == ["a"]
    assert pd.read_parquet(catalogs["plot"]).empty
    plot_metadata = pq.read_metadata(catalogs["plot"]).metadata
    assert plot_metadata[b"scope"] == b"plot"
    assert len(plot_metadata[b"source_sha256"]) == 64
    assert map_path.name == "reference_interval_map.parquet"
    assert len(pd.read_parquet(map_path)) == 2


def test_empty_alignment_catalog_is_rejected(tmp_path):
    original = _fasta(tmp_path / "reference.fa", {"chr1": "ACGT"})
    alignment = tmp_path / "alignment.bed"
    alignment.write_text("# empty\n", encoding="utf-8")
    cfg = SimpleNamespace(
        alignment_regions_bed=str(alignment),
        analysis_regions_bed=None,
        plot_regions_bed=None,
    )

    with pytest.raises(ValueError, match="contains no alignment intervals"):
        write_region_catalogs(cfg, original_fasta=original, run_root=tmp_path / "run")

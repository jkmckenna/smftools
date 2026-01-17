import pytest

import smftools.informatics.bam_functions as bam_functions
import smftools.informatics.bed_functions as bed_functions


def test_resolve_samtools_backend_auto_prefers_cli(monkeypatch):
    monkeypatch.setattr(bam_functions, "pysam", object())
    monkeypatch.setattr(bam_functions.shutil, "which", lambda name: "/usr/bin/samtools")
    assert bam_functions._resolve_samtools_backend("auto") == "cli"


def test_resolve_samtools_backend_cli_fallback(monkeypatch):
    monkeypatch.setattr(bam_functions, "pysam", None)
    monkeypatch.setattr(bam_functions.shutil, "which", lambda name: "/usr/bin/samtools")
    assert bam_functions._resolve_samtools_backend("auto") == "cli"


def test_resolve_samtools_backend_auto_falls_back_to_python(monkeypatch):
    monkeypatch.setattr(bam_functions, "pysam", object())
    monkeypatch.setattr(bam_functions.shutil, "which", lambda name: None)
    assert bam_functions._resolve_samtools_backend("auto") == "python"


def test_resolve_samtools_backend_requires_tool(monkeypatch):
    monkeypatch.setattr(bam_functions, "pysam", None)
    monkeypatch.setattr(bam_functions.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="Neither pysam nor samtools"):
        bam_functions._resolve_samtools_backend("auto")


def test_resolve_bedtools_backend_python_requires_pkg(monkeypatch):
    monkeypatch.setattr(bed_functions, "pybedtools", None)
    monkeypatch.setattr(bed_functions.shutil, "which", lambda name: "/usr/bin/bedtools")
    with pytest.raises(RuntimeError, match="Python package"):
        bed_functions._resolve_backend(
            "python", tool="bedtools", python_available=False, cli_name="bedtools"
        )


def test_resolve_bedtools_backend_cli_requires_tool(monkeypatch):
    monkeypatch.setattr(bed_functions, "pybedtools", object())
    monkeypatch.setattr(bed_functions.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="bedtools in PATH"):
        bed_functions._resolve_backend(
            "cli", tool="bedtools", python_available=True, cli_name="bedtools"
        )


def test_resolve_bedtools_backend_auto_prefers_cli(monkeypatch):
    monkeypatch.setattr(bed_functions.shutil, "which", lambda name: "/usr/bin/bedtools")
    assert (
        bed_functions._resolve_backend(
            "auto", tool="bedtools", python_available=True, cli_name="bedtools"
        )
        == "cli"
    )


def test_parse_idxstats_output():
    output = "chr1\t1000\t10\t0\nchr2\t2000\t5\t0\n*\t0\t0\t2\n"
    aligned, unaligned, record_counts = bam_functions._parse_idxstats_output(output)
    assert aligned == 15
    assert unaligned == 2
    assert record_counts["chr1"] == (10, 10 / 15)


def test_extract_readnames_from_bam_python_backend(tmp_path, monkeypatch):
    class FakeAlignmentFile:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def __iter__(self):
            return iter(
                [
                    type("Read", (), {"query_name": "read1"})(),
                    type("Read", (), {"query_name": "read2"})(),
                ]
            )

    monkeypatch.setattr(
        bam_functions,
        "pysam",
        type("FakePysam", (), {"AlignmentFile": FakeAlignmentFile})(),
    )

    bam_path = tmp_path / "sample.bam"
    bam_path.write_text("stub")
    bam_functions.extract_readnames_from_bam(str(bam_path), samtools_backend="python")
    output_path = tmp_path / "sample_read_names.txt"
    assert output_path.read_text().splitlines() == ["read1", "read2"]


def test_extract_read_features_from_bam_python_backend(monkeypatch, tmp_path):
    class FakeRead:
        def __init__(self):
            self.is_unmapped = False
            self.query_qualities = [10, 20, 30, 40]
            self.reference_name = "chr1"
            self.mapping_quality = 60
            self.query_length = 4
            self.query_name = "read1"

        def get_blocks(self):
            return [(0, 2), (2, 4)]

    class FakeAlignmentFile:
        def __init__(self, *args, **kwargs):
            self.references = ["chr1"]
            self.lengths = [100]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def __iter__(self):
            return iter([FakeRead()])

    monkeypatch.setattr(
        bam_functions,
        "pysam",
        type("FakePysam", (), {"AlignmentFile": FakeAlignmentFile})(),
    )

    bam_path = tmp_path / "sample.bam"
    bam_path.write_text("stub")
    metrics = bam_functions.extract_read_features_from_bam(bam_path, samtools_backend="python")
    assert metrics["read1"] == [4.0, 25.0, 100.0, 4.0, 60.0]


def test_extract_read_features_from_bam_cli_backend(monkeypatch, tmp_path):
    bam_path = tmp_path / "sample.bam"
    bam_path.write_text("stub")

    header_text = "@HD\tVN:1.6\n@SQ\tSN:chr1\tLN:100\n"
    view_text = "read1\t0\tchr1\t1\t60\t10M\t*\t0\t0\tACGTACGTAA\tIIIIIIIIII\n"

    def fake_run(cmd, stdout, stderr, text, check):
        assert cmd[:3] == ["samtools", "view", "-H"]
        return type(
            "CP",
            (),
            {"returncode": 0, "stdout": header_text, "stderr": ""},
        )()

    class FakePopen:
        def __init__(self, cmd, stdout, stderr, text):
            assert cmd[:2] == ["samtools", "view"]
            self.stdout = iter(view_text.splitlines(True))
            self.stderr = None

        def wait(self):
            return 0

    monkeypatch.setattr(bam_functions.subprocess, "run", fake_run)
    monkeypatch.setattr(bam_functions.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(bam_functions.shutil, "which", lambda name: "/usr/bin/samtools")

    metrics = bam_functions.extract_read_features_from_bam(bam_path, samtools_backend="cli")
    assert metrics["read1"] == [10.0, 40.0, 100.0, 10.0, 60.0]

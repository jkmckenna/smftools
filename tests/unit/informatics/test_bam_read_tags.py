import io

from smftools.informatics import bam_functions


def test_extract_read_tags_from_bam_python(monkeypatch, tmp_path):
    class FakeRead:
        def __init__(self):
            self.query_name = "read1"
            self.flag = 99
            self.cigarstring = "4M"
            self._tags = {
                "NM": 1,
                "MD": "4",
                "MM": "C+m,0;",
                "ML": [200],
                "pi": "pod5-parent",
            }

        def get_tag(self, tag):
            return self._tags[tag]

    class FakeAlignmentFile:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def fetch(self, until_eof=True):
            return iter([FakeRead()])

    monkeypatch.setattr(
        bam_functions,
        "pysam",
        type("FakePysam", (), {"AlignmentFile": FakeAlignmentFile})(),
    )

    bam_path = tmp_path / "sample.bam"
    bam_path.write_text("stub")

    read_tags = bam_functions.extract_read_tags_from_bam(
        bam_path,
        tag_names=["NM", "MD", "MM", "ML", "pi"],
        include_flags=True,
        include_cigar=True,
        samtools_backend="python",
    )

    assert read_tags["read1"]["CIGAR"] == "4M"
    assert "proper_pair" in read_tags["read1"]["FLAGS"]
    assert read_tags["read1"]["NM"] == 1
    assert read_tags["read1"]["MD"] == "4"
    assert read_tags["read1"]["MM"] == "C+m,0;"
    assert read_tags["read1"]["ML"] == [200]
    assert read_tags["read1"]["pi"] == "pod5-parent"


def test_extract_read_tags_from_bam_cli_preserves_lowercase_pi(monkeypatch, tmp_path):
    class FakeProcess:
        stdout = io.StringIO(
            "split-child\t0\tchr1\t1\t60\t4M\t*\t0\t0\tACGT\t????\tpi:Z:pod5-parent\n"
        )
        stderr = io.StringIO("")

        @staticmethod
        def wait():
            return 0

    monkeypatch.setattr(bam_functions.shutil, "which", lambda _name: "/usr/bin/samtools")
    monkeypatch.setattr(bam_functions.subprocess, "Popen", lambda *_args, **_kwargs: FakeProcess())
    bam_path = tmp_path / "sample.bam"
    bam_path.write_text("stub")

    read_tags = bam_functions.extract_read_tags_from_bam(
        bam_path,
        tag_names=["pi"],
        include_flags=False,
        include_cigar=False,
        samtools_backend="cli",
    )

    assert read_tags == {"split-child": {"pi": "pod5-parent"}}

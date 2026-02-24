import tempfile
from pathlib import Path

from smftools.informatics import bam_functions


def test_bam_qc_barcode_readname_fallback_does_not_create_temp_bam(monkeypatch, tmp_path):
    bam_path = tmp_path / "reads.bam"
    bam_path.write_text("stub")
    # Skip index creation path.
    bam_path.with_suffix(".bam.bai").write_text("stub")

    real_named_temporary_file = tempfile.NamedTemporaryFile

    def guard_named_temporary_file(*args, **kwargs):
        suffix = kwargs.get("suffix", "")
        if suffix == ".bam":
            raise AssertionError("Temporary BAM files should not be created during barcode QC fallback")
        return real_named_temporary_file(*args, **kwargs)

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", guard_named_temporary_file)
    monkeypatch.setattr(bam_functions, "_resolve_samtools_backend", lambda _backend: "cli")

    class DummyCompletedProcess:
        def __init__(self):
            self.returncode = 0
            # No -N support to force the readname fallback path.
            self.stdout = ""
            self.stderr = ""

    monkeypatch.setattr(bam_functions.subprocess, "run", lambda *args, **kwargs: DummyCompletedProcess())

    class _Stream:
        def __init__(self, lines=None):
            self._lines = list(lines or [])

        def __iter__(self):
            return iter(self._lines)

        def write(self, _text):
            return None

        def close(self):
            return None

    class FakePopen:
        def __init__(self, cmd, stdout=None, stderr=None, text=True, stdin=None):
            self.cmd = cmd
            self.stdout = _Stream()
            self.stderr = _Stream()
            self.stdin = _Stream()
            if cmd[:3] == ["samtools", "view", "-h"]:
                self.stdout = _Stream(
                    [
                        "@HD\tVN:1.6\n",
                        "read_keep\t0\tchr1\t1\t60\t10M\t*\t0\t0\tACGTACGTAA\tIIIIIIIIII\n",
                        "read_drop\t0\tchr1\t1\t60\t10M\t*\t0\t0\tACGTACGTAA\tIIIIIIIIII\n",
                    ]
                )

        def wait(self):
            return 0

    monkeypatch.setattr(bam_functions.subprocess, "Popen", FakePopen)

    out_dir = tmp_path / "bam_qc"
    bam_functions.bam_qc(
        bam_files=[bam_path],
        bam_qc_dir=out_dir,
        threads=1,
        modality="conversion",
        stats=False,
        flagstats=False,
        idxstats=False,
        samtools_backend="cli",
        barcodes=["barcode01"],
        barcode_readname_map={"barcode01": {"read_keep"}},
    )

    assert isinstance(out_dir, Path)


def test_bam_qc_barcode_readname_python_backend_materializes_temp_bam(monkeypatch, tmp_path):
    bam_path = tmp_path / "reads.bam"
    bam_path.write_text("stub")
    # Skip index creation path.
    bam_path.with_suffix(".bam.bai").write_text("stub")

    created_temp_bam = {"value": False}
    real_named_temporary_file = tempfile.NamedTemporaryFile

    def track_named_temporary_file(*args, **kwargs):
        if kwargs.get("suffix", "") == ".bam":
            created_temp_bam["value"] = True
        return real_named_temporary_file(*args, **kwargs)

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", track_named_temporary_file)
    monkeypatch.setattr(bam_functions, "_resolve_samtools_backend", lambda _backend: "python")

    class FakeRead:
        def __init__(self, query_name):
            self.query_name = query_name

    class FakeAlignmentFile:
        def __init__(self, path, mode, header=None, **kwargs):
            self.path = path
            self.mode = mode
            self.header = {"HD": {"VN": "1.6"}}
            self._written = []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def fetch(self, until_eof=True):
            return iter([FakeRead("read_keep"), FakeRead("read_drop")])

        def write(self, read):
            self._written.append(read)

    class FakePysam:
        AlignmentFile = FakeAlignmentFile

        @staticmethod
        def stats(path):
            return f"stats for {path}\n"

        @staticmethod
        def flagstat(path):
            return f"flagstat for {path}\n"

        @staticmethod
        def idxstats(path):
            return f"idxstats for {path}\n"

        @staticmethod
        def index(path):
            return None

    monkeypatch.setattr(bam_functions, "_require_pysam", lambda: FakePysam)

    out_dir = tmp_path / "bam_qc"
    bam_functions.bam_qc(
        bam_files=[bam_path],
        bam_qc_dir=out_dir,
        threads=1,
        modality="conversion",
        stats=True,
        flagstats=True,
        idxstats=False,
        samtools_backend="python",
        barcodes=["barcode01"],
        barcode_readname_map={"barcode01": {"read_keep"}},
    )

    assert created_temp_bam["value"] is True
    assert (out_dir / "barcode01_stats.txt").exists()
    assert (out_dir / "barcode01_flagstat.txt").exists()

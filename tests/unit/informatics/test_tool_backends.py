import pytest

import smftools.informatics.bam_functions as bam_functions
import smftools.informatics.bed_functions as bed_functions


def test_resolve_samtools_backend_auto_prefers_pysam(monkeypatch):
    monkeypatch.setattr(bam_functions, "pysam", object())
    monkeypatch.setattr(bam_functions.shutil, "which", lambda name: None)
    assert bam_functions._resolve_samtools_backend("auto") == "python"


def test_resolve_samtools_backend_cli_fallback(monkeypatch):
    monkeypatch.setattr(bam_functions, "pysam", None)
    monkeypatch.setattr(bam_functions.shutil, "which", lambda name: "/usr/bin/samtools")
    assert bam_functions._resolve_samtools_backend("auto") == "cli"


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

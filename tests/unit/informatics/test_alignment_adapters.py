import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from smftools.cli.load_adata import load_adata_core
from smftools.config.experiment_config import SUPPORTED_ALIGNERS
from smftools.informatics.alignment_adapters import (
    AlignmentAdapterError,
    AlignmentEnvironment,
    AlignmentRequest,
    adapter_names,
    get_alignment_adapter,
)
from smftools.informatics.alignment_adapters.base import probe_executable_version
from smftools.informatics.alignment_adapters.builtin import DoradoAdapter, Minimap2Adapter
from smftools.informatics.alignment_manifest import (
    read_alignment_manifest,
    write_alignment_manifest,
)


def _request(tmp_path, **overrides):
    values = {
        "reference_fasta": tmp_path / "reference with spaces.fa",
        "input_bam": tmp_path / "input [reads].bam",
        "aligned_bam": tmp_path / "workspace" / "aligned.bam",
        "source_layout": "single_bam",
        "modality": "conversion",
        "aligner_args": ("--rg", "@RG ID:sample one", "--flag=a;b"),
        "threads": 3,
        "align_from_bam": True,
    }
    values.update(overrides)
    return AlignmentRequest(**values)


def _environment(name="minimap2 2.27-r1193", version=(2, 27, 0)):
    return AlignmentEnvironment(
        adapter_version=name,
        adapter_version_tuple=version,
        samtools_backend="python",
        sort_index_version="pysam 0.23.3",
    )


def test_registry_is_the_supported_aligner_authority():
    assert adapter_names() == ("dorado", "minimap2")
    assert set(adapter_names()) == set(SUPPORTED_ALIGNERS)
    with pytest.raises(AlignmentAdapterError, match="Unknown alignment adapter"):
        get_alignment_adapter("shell-command")


def test_minimap2_and_dorado_argv_preserve_legacy_order_and_argument_boundaries(tmp_path):
    request = _request(tmp_path)

    minimap_argv = Minimap2Adapter().build_argv(request, request.input_bam)
    dorado_argv = DoradoAdapter().build_argv(request, request.input_bam)

    assert minimap_argv == [
        "minimap2",
        "--rg",
        "@RG ID:sample one",
        "--flag=a;b",
        "-t",
        "3",
        str(request.reference_fasta),
        str(request.input_bam),
    ]
    assert dorado_argv == [
        "dorado",
        "aligner",
        "-t",
        "3",
        "--rg",
        "@RG ID:sample one",
        "--flag=a;b",
        str(request.reference_fasta),
        str(request.input_bam),
    ]
    assert "@RG ID:sample one" in minimap_argv
    assert "--flag=a;b" in dorado_argv


def test_version_probe_rejects_missing_unparseable_and_old_tools(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _name: None)
    with pytest.raises(AlignmentAdapterError, match="requires executable"):
        probe_executable_version("minimap2", (2, 24, 0))

    monkeypatch.setattr("shutil.which", lambda name: f"/tools/{name}")
    monkeypatch.setattr(
        "subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout="unknown", stderr=""),
    )
    with pytest.raises(AlignmentAdapterError, match="Could not parse"):
        probe_executable_version("minimap2", (2, 24, 0))

    monkeypatch.setattr(
        "subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout="2.23-r1", stderr=""),
    )
    with pytest.raises(AlignmentAdapterError, match="minimap2 >= 2.24.0"):
        probe_executable_version("minimap2", (2, 24, 0))


def test_missing_adapter_executable_fails_before_task_output_creation(tmp_path, monkeypatch):
    output = tmp_path / "run"
    cfg = SimpleNamespace(
        alignment_mode="align",
        input_type="bam",
        input_already_demuxed=True,
        log_level="INFO",
        aligner="minimap2",
        samtools_backend="python",
        output_directory=output,
        bam_outputs_path=output / "bam",
        fasta_outputs_path=output / "fasta",
        bed_outputs_path=output / "bed",
        modkit_outputs_path=output / "modkit",
    )
    monkeypatch.setattr("shutil.which", lambda _name: None)

    with pytest.raises(AlignmentAdapterError, match="requires executable 'minimap2'"):
        load_adata_core(cfg, SimpleNamespace(), raw_only=True)

    assert not output.exists()


def test_adapter_capability_errors_name_layout_and_tag_preserving_remedy(tmp_path):
    adapter = Minimap2Adapter()
    with pytest.raises(AlignmentAdapterError, match="paired_bam.*IAR-08"):
        adapter.validate_request(_request(tmp_path, source_layout="paired_bam"))
    with pytest.raises(AlignmentAdapterError, match="discard MM/ML tags"):
        adapter.validate_request(_request(tmp_path, modality="direct", align_from_bam=False))
    DoradoAdapter().validate_request(_request(tmp_path, modality="direct", align_from_bam=False))


def test_reference_plan_identity_uses_only_semantic_inputs(tmp_path):
    adapter = Minimap2Adapter()
    environment = _environment()

    first = adapter.reference_plan("a" * 64, environment)
    repeated = adapter.reference_plan("a" * 64, environment)
    changed_reference = adapter.reference_plan("b" * 64, environment)
    changed_version = adapter.reference_plan(
        "a" * 64,
        _environment(name="minimap2 2.28", version=(2, 28, 0)),
    )

    assert first == repeated
    assert first["identity"] != changed_reference["identity"]
    assert first["identity"] != changed_version["identity"]
    assert "threads" not in first["index_parameters"]


def test_failed_adapter_execution_removes_partial_outputs(tmp_path, monkeypatch):
    adapter = Minimap2Adapter()
    request = _request(tmp_path)
    request.input_bam.write_bytes(b"input")

    def fail(_argv, output):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"partial")
        raise AlignmentAdapterError("injected aligner failure")

    monkeypatch.setattr(adapter, "_run_aligner", fail)
    with pytest.raises(AlignmentAdapterError, match="injected"):
        adapter.execute(request, _environment(), "a" * 64)

    assert not request.aligned_bam.exists()
    assert not request.aligned_bam.with_name("aligned_sorted.bam").exists()
    assert not Path(f"{request.aligned_bam.with_name('aligned_sorted.bam')}.bai").exists()


@pytest.mark.parametrize("failure_stage", ["sort", "index"])
def test_failed_sort_or_index_removes_partial_outputs(tmp_path, monkeypatch, failure_stage):
    adapter = Minimap2Adapter()
    request = _request(tmp_path)
    request.input_bam.write_bytes(b"input")
    sorted_bam = request.aligned_bam.with_name("aligned_sorted.bam")
    bai = Path(f"{sorted_bam}.bai")

    def align(_argv, output):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"alignment")

    def sort(_input, output, *, threads):
        del threads
        output.write_bytes(b"sorted-partial")
        if failure_stage == "sort":
            raise RuntimeError("injected sort failure")

    def index(output, *, threads):
        del output, threads
        bai.write_bytes(b"index-partial")
        raise RuntimeError("injected index failure")

    monkeypatch.setattr(adapter, "_run_aligner", align)
    monkeypatch.setattr("smftools.informatics.bam_functions._sort_bam_with_pysam", sort)
    monkeypatch.setattr("smftools.informatics.bam_functions._index_bam_with_pysam", index)

    with pytest.raises(RuntimeError, match=f"injected {failure_stage} failure"):
        adapter.execute(request, _environment(), "a" * 64)

    assert not request.aligned_bam.exists()
    assert not sorted_bam.exists()
    assert not bai.exists()


def test_generated_alignment_manifest_is_deterministic_and_relocatable(tmp_path):
    def publish(root):
        root.mkdir()
        bam = root / "aligned_sorted.bam"
        bai = root / "aligned_sorted.bam.bai"
        bam.write_bytes(b"bam")
        bai.write_bytes(b"bai")
        return write_alignment_manifest(
            root / "alignment_manifest.json",
            input_manifest_digest="input-digest",
            reference_bundle={"schema_version": 1, "digest": "reference-digest"},
            prepared_reference_sha256="a" * 64,
            source_bam=root.parent / "external" / "source.bam",
            source_sha256="b" * 64,
            normalized_bam=bam,
            normalized_bai=bai,
            validation={"normalized": {"reference_records": []}},
            alignment_mode="align",
            adapter={
                "schema_version": 1,
                "name": "minimap2",
                "version": "2.27-r1193",
                "normalized_argv": ["minimap2", "$REFERENCE", "$INPUT_FASTQ"],
            },
        )

    first = publish(tmp_path / "first")
    second = publish(tmp_path / "second")

    first_payload = json.loads(first.read_text())
    second_payload = json.loads(second.read_text())
    assert first_payload == second_payload
    assert first_payload["source"] == {"path_hint": "source.bam", "sha256": "b" * 64}
    assert read_alignment_manifest(second)["adapter"]["name"] == "minimap2"

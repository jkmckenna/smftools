from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from click.testing import CliRunner

from smftools import cli_entry
from smftools.cli import workflow_contract

pytestmark = pytest.mark.unit


class _Plan:
    def __init__(self, state: str):
        self.decisions = (
            SimpleNamespace(
                analysis_id="experiment.raw.complete",
                state=SimpleNamespace(value=state),
                reason="reason",
                reason_code="reason_code",
            ),
        )

    def to_dict(self):
        return {
            "schema_version": 1,
            "requested_target": "experiment.raw.complete",
            "decisions": [
                {
                    "analysis_id": "experiment.raw.complete",
                    "state": self.decisions[0].state.value,
                }
            ],
        }


class _ProjectPlan:
    def __init__(self, token: str = "source-a"):
        self.token = token

    def to_dict(self):
        return {
            "schema_version": 1,
            "requested_target": "project.materialization",
            "decisions": [
                {
                    "analysis_id": "project.genomic_selection",
                    "state": "compatible",
                    "selected_result_id": f"selection:{self.token}",
                },
                {
                    "analysis_id": "project.materialization",
                    "state": "missing",
                    "compatibility_key": f"materialization:{self.token}",
                },
            ],
        }


def _source_config(tmp_path: Path) -> Path:
    path = tmp_path / "source.csv"
    path.write_text(
        "variable,value,type\n"
        f"output_directory,{tmp_path / 'configured-output'},string\n"
        "threads,8,int\n",
        encoding="utf-8",
    )
    return path


def _cfg(tmp_path: Path):
    envelope = SimpleNamespace(
        resolved_threads=4,
        resolved_memory_bytes=8 * 1024**3,
        as_dict=lambda: {
            "resolved_threads": 4,
            "resolved_memory_bytes": 8 * 1024**3,
        },
    )
    return SimpleNamespace(
        output_directory=tmp_path,
        _resource_envelope=envelope,
        device="auto",
        skip_bam_qc=True,
        input_type="bam",
        aligner="",
        demux_backend="smftools",
        input_already_demuxed=True,
        smf_modality="conversion",
        direct_signal_backend="pysam",
        samtools_backend="python",
    )


def _patch_execution(monkeypatch, tmp_path: Path, *, state="missing", failure=None):
    cfg = _cfg(tmp_path)
    plan_calls = 0

    def plan(_path, _target):
        nonlocal plan_calls
        plan_calls += 1
        return _Plan(state if plan_calls == 1 else "compatible")

    monkeypatch.setattr(
        "smftools.cli.helpers.load_experiment_config",
        lambda _path: cfg,
    )
    monkeypatch.setattr(
        "smftools.pipeline.experiment_graph.plan_experiment",
        plan,
    )

    def execute(_path, _target):
        if failure is not None:
            raise failure
        return None

    monkeypatch.setattr("smftools.cli.recipes.run_experiment_target", execute)
    monkeypatch.setattr(workflow_contract, "stage_is_complete", lambda *_args, **_kwargs: True)
    return cfg


def _patch_project_execution(monkeypatch, *, token="source-a", failure=None):
    monkeypatch.setattr(
        "smftools.cli.project_cmd.project_plan",
        lambda *_args, **_kwargs: _ProjectPlan(token),
    )

    def materialize(_project, _reference, output, **_kwargs):
        if failure is not None:
            raise failure
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"materialized")
        return output

    monkeypatch.setattr("smftools.cli.project_cmd.project_materialize", materialize)

    def sample_analysis(_project, _reference, output, **_kwargs):
        if failure is not None:
            raise failure
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"sample-analysis")
        return {"schema_version": 1, "n_partitions": 2, "n_reads": 12, "partitions": []}

    monkeypatch.setattr("smftools.cli.project_cmd.project_sample_analysis", sample_analysis)

    def embedding(_project, _reference, output, **kwargs):
        if failure is not None:
            raise failure
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"embedding")
        return {
            "schema_version": 1,
            "generation_id": "generation-embed",
            "fit_kind": "extended" if kwargs.get("trust_local_models") else "full",
            "n_molecules": 30,
            "n_new_molecules": 10 if kwargs.get("trust_local_models") else 0,
        }

    monkeypatch.setattr("smftools.cli.project_cmd.project_embedding", embedding)
    monkeypatch.setattr(
        workflow_contract,
        "_project_resource_decision",
        lambda **_kwargs: {
            "requested": {},
            "ceiling": {},
            "resolved": {
                "cpus": 2,
                "memory_bytes": 4 * 1024**3,
                "accelerator": "cpu",
            },
        },
    )


def test_workflow_success_writes_only_declared_root_and_preserves_staged_inputs(
    tmp_path,
    monkeypatch,
):
    source = _source_config(tmp_path)
    staged_input = tmp_path / "staged.bam"
    staged_fasta = tmp_path / "reference.fa"
    staged_input.write_bytes(b"bam")
    staged_fasta.write_text(">ref\nACGT\n", encoding="utf-8")
    before = {path: path.read_bytes() for path in (source, staged_input, staged_fasta)}
    output = tmp_path / "task-output"
    _patch_execution(monkeypatch, output)

    result_path = workflow_contract.run_experiment_workflow(
        source,
        target="raw",
        output_root=output,
        input_path=staged_input,
        fasta_path=staged_fasta,
        cpus=2,
        memory_gb=4,
        accelerator="cpu",
    )

    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["outcome"] == "success"
    assert result["result_id"]
    assert result["run_id"]
    assert result["run_root"] == result["output_root"] == "."
    assert result["resources"]["resolved"] == {
        "cpus": 2,
        "memory_bytes": 4 * 1024**3,
        "accelerator": "cpu",
    }
    runtime_values = pd.read_csv(
        output / result["runtime_config"],
    ).set_index("variable")["value"]
    assert Path(runtime_values["input_data_path"]).is_relative_to(output)
    assert Path(runtime_values["input_data_path"]).resolve() == staged_input
    assert Path(runtime_values["fasta"]).is_relative_to(output)
    assert Path(runtime_values["fasta"]).resolve() == staged_fasta
    assert all(not Path(item["path"]).is_absolute() for item in result["artifacts"])
    assert {path: path.read_bytes() for path in before} == before
    assert not (tmp_path / "configured-output").exists()


def test_runtime_config_preserves_manifest_contract(tmp_path, monkeypatch):
    source_file = tmp_path / "reads.fastq"
    source_file.write_bytes(b"reads")
    input_manifest = tmp_path / "inputs.csv"
    input_manifest.write_text("path\nreads.fastq\n", encoding="utf-8")
    source_config = tmp_path / "source.csv"
    source_config.write_text(
        "variable,value,type\n"
        f"input_manifest_path,{input_manifest},string\n"
        f"output_directory,{tmp_path / 'configured-output'},string\n",
        encoding="utf-8",
    )
    envelope = SimpleNamespace(
        resolved_threads=4,
        resolved_memory_bytes=8 * 1024**3,
        as_dict=lambda: {},
    )
    cfg = SimpleNamespace(
        input_manifest_path=input_manifest,
        input_data_path=source_file,
        input_files=[source_file],
        fasta=None,
        device="cpu",
        _resource_envelope=envelope,
    )
    monkeypatch.setattr("smftools.cli.helpers.load_experiment_config", lambda _path: cfg)
    output = tmp_path / "task-output"

    runtime_config, _, effective_sources = workflow_contract._write_runtime_config(
        source_config,
        output,
        input_path=None,
        fasta_path=None,
        cpus=None,
        memory_gb=None,
        accelerator=None,
    )

    runtime_values = pd.read_csv(runtime_config, keep_default_na=False).set_index("variable")[
        "value"
    ]
    assert runtime_values["input_data_path"] == ""
    assert Path(runtime_values["input_manifest_path"]).resolve() == input_manifest
    assert effective_sources == {
        "input:000000": source_file.resolve(),
        "input_manifest": input_manifest.resolve(),
    }


def test_compatible_skip_has_distinct_successful_outcome(tmp_path, monkeypatch):
    source = _source_config(tmp_path)
    output = tmp_path / "task-output"
    _patch_execution(monkeypatch, output, state="compatible")

    path = workflow_contract.run_experiment_workflow(
        source,
        target="raw",
        output_root=output,
    )

    assert json.loads(path.read_text(encoding="utf-8"))["outcome"] == "compatible_skip"


def test_failure_writes_structured_result_json(tmp_path, monkeypatch):
    source = _source_config(tmp_path)
    output = tmp_path / "task-output"
    _patch_execution(monkeypatch, output, failure=ValueError("injected failure"))

    with pytest.raises(workflow_contract.WorkflowContractError, match="injected failure"):
        workflow_contract.run_experiment_workflow(
            source,
            target="raw",
            output_root=output,
        )

    result = json.loads(
        (output / workflow_contract.WORKFLOW_RESULT_FILENAME).read_text(encoding="utf-8")
    )
    assert result["outcome"] == "failed"
    assert result["failure"] == {
        "type": "ValueError",
        "message": "injected failure",
        "stage": "workflow",
    }


def test_resource_overrides_are_bounded_and_written_to_runtime_copy(tmp_path, monkeypatch):
    source = _source_config(tmp_path)
    output = tmp_path / "task-output"
    output.mkdir()
    cfg = _cfg(output)
    monkeypatch.setattr("smftools.cli.helpers.load_experiment_config", lambda _path: cfg)

    runtime, resources, _sources = workflow_contract._write_runtime_config(
        source,
        output,
        input_path=None,
        fasta_path=None,
        cpus=20,
        memory_gb=20,
        accelerator="cpu",
    )

    values = pd.read_csv(runtime).set_index("variable")["value"]
    assert int(values["threads"]) == 4
    assert float(values["max_memory_gb"]) == 8.0
    assert resources["resolved"]["cpus"] == 4
    assert resources["resolved"]["memory_bytes"] == 8 * 1024**3
    assert "output_directory" in source.read_text(encoding="utf-8")
    assert str(output) not in source.read_text(encoding="utf-8")


def test_strict_mode_rejects_missing_required_tool(tmp_path, monkeypatch):
    source = _source_config(tmp_path)
    output = tmp_path / "task-output"
    cfg = _patch_execution(monkeypatch, output)
    cfg.input_type = "pod5"
    monkeypatch.setattr(workflow_contract.shutil, "which", lambda _name: None)

    with pytest.raises(workflow_contract.WorkflowContractError, match="dorado"):
        workflow_contract.run_experiment_workflow(
            source,
            target="raw",
            output_root=output,
            strict=True,
        )

    result = json.loads(
        (output / workflow_contract.WORKFLOW_RESULT_FILENAME).read_text(encoding="utf-8")
    )
    assert result["outcome"] == "failed"


def test_versions_include_smftools_and_requested_external_tool(monkeypatch):
    run_kwargs = {}

    def run(*_args, **kwargs):
        run_kwargs.update(kwargs)
        return SimpleNamespace(
            stdout="samtools 1.21\n",
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr(workflow_contract.shutil, "which", lambda _name: "/bin/tool")
    monkeypatch.setattr(workflow_contract.subprocess, "run", run)

    versions = workflow_contract.software_versions(tools=("samtools",))

    assert versions["smftools"]
    assert versions["external_tools"]["samtools"]["version"] == "samtools 1.21"
    assert run_kwargs["encoding"] == "utf-8"
    assert run_kwargs["errors"] == "replace"


def test_versions_include_supplied_container_identity(monkeypatch):
    identity = {
        "SMFTOOLS_CONTAINER_IMAGE": "ghcr.io/jkmckenna/smftools",
        "SMFTOOLS_CONTAINER_TAG": "sha-abc123",
        "SMFTOOLS_CONTAINER_DIGEST": "sha256:deadbeef",
        "SMFTOOLS_CONTAINER_REVISION": "abc123",
        "SMFTOOLS_CONTAINER_PROFILE": "cpu-bam",
    }
    for name, value in identity.items():
        monkeypatch.setenv(name, value)

    versions = workflow_contract.software_versions()

    assert versions["schema_version"] == 2
    assert versions["container"] == {
        "image": "ghcr.io/jkmckenna/smftools",
        "tag": "sha-abc123",
        "digest": "sha256:deadbeef",
        "revision": "abc123",
        "profile": "cpu-bam",
    }


def test_version_plan_covers_configured_external_backends(monkeypatch, tmp_path):
    cfg = _cfg(tmp_path)
    cfg.input_type = "fast5"
    cfg.smf_modality = "direct"
    cfg.direct_signal_backend = "modkit"
    cfg.samtools_backend = "auto"
    cfg.skip_bam_qc = False
    cfg.make_beds = True
    cfg.make_bigwigs = True
    cfg.bedtools_backend = "auto"
    cfg.bigwig_backend = "auto"
    monkeypatch.setattr(workflow_contract.shutil, "which", lambda name: f"/bin/{name}")

    tools = workflow_contract._required_external_tools(cfg, raw_will_run=True)

    assert set(tools) == {
        "bedGraphToBigWig",
        "bedtools",
        "dorado",
        "gzip",
        "modkit",
        "multiqc",
        "pod5",
        "samtools",
    }


@pytest.mark.parametrize(
    ("aligner", "expected"),
    [
        ("minimap2", {"minimap2"}),
        ("bwa-mem2", {"bwa-mem2"}),
        # bowtie2 builds its native index with a separate binary.
        ("bowtie2", {"bowtie2", "bowtie2-build"}),
    ],
)
def test_version_plan_covers_every_supported_aligner(monkeypatch, tmp_path, aligner, expected):
    cfg = _cfg(tmp_path)
    cfg.aligner = aligner
    monkeypatch.setattr(workflow_contract.shutil, "which", lambda name: None)

    tools = set(workflow_contract._required_external_tools(cfg, raw_will_run=True))

    assert expected <= tools
    assert all(tool in workflow_contract._TOOL_VERSION_COMMANDS for tool in tools)


def _write_bundle(root: Path, *, bundle_kind: str = "sequence_only") -> Path:
    """Build a real, checksum-valid re-ingestion bundle."""
    from smftools.informatics.export_bundle import (
        artifact_record,
        write_bundle_manifest,
        write_canonical_input_manifest,
        write_identity_map,
    )

    root.mkdir(parents=True, exist_ok=True)
    reads = root / "reads.fastq"
    reads.write_text("@r1\nACGT\n+\nIIII\n", encoding="utf-8")
    write_canonical_input_manifest(
        root / "inputs.csv",
        [{"path": "reads.fastq", "source_kind": "fastq", "sample": "s", "barcode": "bc01"}],
        alignment_mode="align",
        modality="conversion",
    )
    write_identity_map(
        root / "identity_map.csv", [{"bundle_read_id": "r1", "source_read_id": "r1"}]
    )
    return write_bundle_manifest(
        root,
        bundle_kind=bundle_kind,
        scope="experiment",
        modality="conversion",
        selection={},
        # `raw_generation_id` is the field both exporters actually write.
        source_generations=[{"raw_generation_id": "generation-a"}],
        artifacts=[artifact_record(root, reads)],
        lost_capabilities=["direct_modification"],
    )


def test_staged_bundle_keeps_relative_artifacts_resolvable(tmp_path):
    """Staging symlinks the manifest, so bundle-relative artifacts must still resolve.

    The raw stage resolves the manifest path before anchoring relative rows, so a
    read-only alias outside the bundle still points back into it. Locks that in so
    hardening the staging path cannot silently break bundle re-ingestion.
    """
    from smftools.informatics.input_manifest import resolve_input_manifest_readonly

    bundle_manifest = _write_bundle(tmp_path / "bundle")
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()

    alias = workflow_contract._stage_readonly_alias(
        bundle_manifest, runtime_dir, stem="input_manifest"
    )

    assert alias.suffix == ".json"
    resolved = resolve_input_manifest_readonly(
        input_manifest_path=alias, alignment_mode="align", modality="conversion"
    )
    assert [Path(row.path).name for row in resolved.rows] == ["reads.fastq"]
    assert all(Path(row.path).is_file() for row in resolved.rows)


def test_source_snapshot_records_bundle_kind_and_lost_capabilities(tmp_path):
    """A file fingerprint cannot express what re-ingesting a bundle costs."""
    bundle_manifest = _write_bundle(tmp_path / "bundle")

    snapshot = workflow_contract._snapshot_sources({"input_manifest": bundle_manifest})
    records = workflow_contract._source_result_records(snapshot)

    bundle = records["input_manifest"]["bundle"]
    assert bundle["bundle_kind"] == "sequence_only"
    assert bundle["lost_capabilities"] == ["direct_modification"]
    assert bundle["source_generation_ids"] == ["generation-a"]
    assert bundle["modality"] == "conversion"
    # The ordinary fingerprint contract still applies to the manifest file.
    assert records["input_manifest"]["fingerprint"] == snapshot["input_manifest"]["fingerprint"]


def test_source_snapshot_accepts_the_legacy_generation_id_field(tmp_path):
    """A bundle naming its origin `generation_id` is still recorded, not dropped."""
    bundle_manifest = _write_bundle(tmp_path / "bundle")
    payload = json.loads(bundle_manifest.read_text(encoding="utf-8"))
    payload["source_generations"] = [{"generation_id": "generation-b"}]
    bundle_manifest.write_text(json.dumps(payload), encoding="utf-8")

    snapshot = workflow_contract._snapshot_sources({"input_manifest": bundle_manifest})

    assert snapshot["input_manifest"]["bundle"]["source_generation_ids"] == ["generation-b"]


def test_non_bundle_sources_carry_no_bundle_provenance(tmp_path):
    plain = tmp_path / "inputs.csv"
    plain.write_text("path\nreads.fastq\n", encoding="utf-8")
    unrelated_json = tmp_path / "other.json"
    unrelated_json.write_text('{"schema_version": 1}\n', encoding="utf-8")

    records = workflow_contract._source_result_records(
        workflow_contract._snapshot_sources({"input_manifest": plain, "sidecar": unrelated_json})
    )

    assert "bundle" not in records["input_manifest"]
    assert "bundle" not in records["sidecar"]


def test_validation_detects_corruption_and_relocated_output_validates(
    tmp_path,
    monkeypatch,
):
    source = _source_config(tmp_path)
    output = tmp_path / "task-output"
    _patch_execution(monkeypatch, output, state="compatible")
    result_path = workflow_contract.run_experiment_workflow(
        source,
        target="raw",
        output_root=output,
    )
    monkeypatch.setattr(
        "smftools.pipeline.experiment_graph.plan_experiment",
        lambda _path, _target: _Plan("compatible"),
    )

    relocated = tmp_path / "relocated"
    shutil.move(output, relocated)
    validation = workflow_contract.validate_workflow_output(relocated)
    assert validation["valid"] is True

    versions = relocated / workflow_contract.WORKFLOW_VERSIONS_FILENAME
    versions.write_text("corrupt", encoding="utf-8")
    validation = workflow_contract.validate_workflow_output(relocated)
    assert validation["valid"] is False
    assert "artifact_checksum_mismatch" in {issue["code"] for issue in validation["issues"]}
    assert not result_path.exists()


def test_validation_rejects_pointer_escape(tmp_path, monkeypatch):
    source = _source_config(tmp_path)
    output = tmp_path / "task-output"
    _patch_execution(monkeypatch, output, state="compatible")
    result_path = workflow_contract.run_experiment_workflow(
        source,
        target="raw",
        output_root=output,
    )
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["artifacts"][0]["path"] = "../outside"
    result_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = workflow_contract.validate_workflow_output(output)

    assert validation["valid"] is False
    assert "artifact_pointer_escape" in {issue["code"] for issue in validation["issues"]}


def test_validation_reports_incomplete_and_semantically_stale_output(
    tmp_path,
    monkeypatch,
):
    source = _source_config(tmp_path)
    output = tmp_path / "task-output"
    _patch_execution(monkeypatch, output, state="compatible")
    workflow_contract.run_experiment_workflow(
        source,
        target="raw",
        output_root=output,
    )
    result_path = output / workflow_contract.WORKFLOW_RESULT_FILENAME
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["post_plan"]["decisions"][0]["state"] = "stale_input"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    monkeypatch.setattr(workflow_contract, "stage_is_complete", lambda *_args, **_kwargs: False)

    validation = workflow_contract.validate_workflow_output(output)

    codes = {issue["code"] for issue in validation["issues"]}
    assert "publication_incompatible" in codes
    assert "stage_incomplete" in codes


def test_same_output_root_cannot_be_owned_concurrently(tmp_path):
    output = tmp_path / "task-output"
    output.mkdir()

    with workflow_contract._exclusive_run(output):
        with pytest.raises(workflow_contract.WorkflowContractError, match="already owned"):
            with workflow_contract._exclusive_run(output):
                pass


def test_remote_input_uri_is_rejected():
    with pytest.raises(workflow_contract.WorkflowContractError, match="unsupported URI"):
        workflow_contract._local_path("s3://bucket/input.bam", label="staged input")


def test_result_json_must_be_task_root_owned(tmp_path):
    output = tmp_path / "output"
    output.mkdir()

    with pytest.raises(workflow_contract.WorkflowContractError, match="direct child"):
        workflow_contract._result_path(output, "nested/result.json")


def test_workflow_staging_rejects_directory_inputs(tmp_path):
    source = tmp_path / "input"
    source.mkdir()
    runtime = tmp_path / "output" / workflow_contract.WORKFLOW_RUNTIME_DIRECTORY
    runtime.mkdir(parents=True)

    with pytest.raises(workflow_contract.WorkflowContractError, match="concrete file"):
        workflow_contract._stage_readonly_alias(source, runtime, stem="input")


def test_workflow_cli_and_validation_exit_codes(tmp_path, monkeypatch):
    source = _source_config(tmp_path)
    output = tmp_path / "task-output"
    _patch_execution(monkeypatch, output, state="compatible")
    runner = CliRunner()

    result = runner.invoke(
        cli_entry.cli,
        [
            "experiment",
            "run",
            str(source),
            "--target",
            "raw",
            "--output-root",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output

    valid = runner.invoke(
        cli_entry.cli,
        ["experiment", "validate", str(output), "--json"],
    )
    assert valid.exit_code == 0, valid.output
    assert json.loads(valid.output)["valid"] is True

    (output / workflow_contract.WORKFLOW_VERSIONS_FILENAME).unlink()
    invalid = runner.invoke(
        cli_entry.cli,
        ["experiment", "validate", str(output), "--json"],
    )
    assert invalid.exit_code == 1


def test_project_workflow_success_skip_and_validation(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    (project / "registry.json").write_text("{}", encoding="utf-8")
    output = tmp_path / "project-output"
    _patch_project_execution(monkeypatch)

    first = workflow_contract.run_project_materialization_workflow(
        project,
        "uid-ref",
        output_root=output,
    )
    first_result = json.loads(first.read_text(encoding="utf-8"))
    assert first_result["command"] == "project.materialize"
    assert first_result["outcome"] == "success"
    assert first_result["artifacts"][0]["path"] == "materialized.h5ad.gz"

    second = workflow_contract.run_project_materialization_workflow(
        project,
        "uid-ref",
        output_root=output,
    )
    assert json.loads(second.read_text(encoding="utf-8"))["outcome"] == "compatible_skip"
    assert workflow_contract.validate_workflow_output(
        output,
        project_dir=project,
    )["valid"]


def test_project_validation_detects_source_staleness(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    output = tmp_path / "project-output"
    _patch_project_execution(monkeypatch, token="source-a")
    workflow_contract.run_project_materialization_workflow(
        project,
        "uid-ref",
        output_root=output,
    )
    _patch_project_execution(monkeypatch, token="source-b")

    validation = workflow_contract.validate_workflow_output(
        output,
        project_dir=project,
    )

    assert validation["valid"] is False
    assert "project_source_stale" in {issue["code"] for issue in validation["issues"]}


def test_project_failure_writes_structured_result(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    output = tmp_path / "project-output"
    _patch_project_execution(monkeypatch, failure=ValueError("project failure"))

    with pytest.raises(workflow_contract.WorkflowContractError, match="project failure"):
        workflow_contract.run_project_materialization_workflow(
            project,
            "uid-ref",
            output_root=output,
        )

    result = json.loads(
        (output / workflow_contract.WORKFLOW_RESULT_FILENAME).read_text(encoding="utf-8")
    )
    assert result["outcome"] == "failed"
    assert result["failure"]["stage"] == "project.materialization"


def test_project_sample_analysis_workflow_success_skip_and_validation(tmp_path, monkeypatch):
    """Sample analysis is a workflow product, not a Python-only API.

    The `sample-analysis` target has been advertised by `project plan` with no
    matching executable, so a planner could ask for something no CLI could run
    or validate. This covers that lifecycle end to end.
    """
    project = tmp_path / "project"
    project.mkdir()
    output = tmp_path / "analysis-output"
    _patch_project_execution(monkeypatch)

    first = workflow_contract.run_project_sample_analysis_workflow(
        project,
        "uid-ref",
        output_root=output,
    )
    result = json.loads(first.read_text(encoding="utf-8"))
    assert result["command"] == "project.sample-analysis"
    assert result["target"] == "sample-analysis"
    assert result["outcome"] == "success"
    assert result["artifacts"][0]["artifact_id"] == "project:sample-analysis"
    assert result["artifacts"][0]["path"] == "sample_analysis.parquet"
    assert result["summary"]["n_partitions"] == 2

    second = workflow_contract.run_project_sample_analysis_workflow(
        project,
        "uid-ref",
        output_root=output,
    )
    assert json.loads(second.read_text(encoding="utf-8"))["outcome"] == "compatible_skip"
    assert workflow_contract.validate_workflow_output(output, project_dir=project)["valid"]

    _patch_project_execution(monkeypatch, token="source-b")
    validation = workflow_contract.validate_workflow_output(output, project_dir=project)
    assert validation["valid"] is False
    assert "project_source_stale" in {issue["code"] for issue in validation["issues"]}


def test_project_sample_analysis_failure_writes_structured_result(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    output = tmp_path / "analysis-output"
    _patch_project_execution(monkeypatch, failure=ValueError("no partitions"))

    with pytest.raises(workflow_contract.WorkflowContractError, match="no partitions"):
        workflow_contract.run_project_sample_analysis_workflow(
            project,
            "uid-ref",
            output_root=output,
        )

    result = json.loads(
        (output / workflow_contract.WORKFLOW_RESULT_FILENAME).read_text(encoding="utf-8")
    )
    assert result["outcome"] == "failed"
    assert result["command"] == "project.sample-analysis"
    assert result["failure"]["stage"] == "project.sample_analysis"


def test_one_project_product_never_satisfies_another_in_the_same_root(tmp_path, monkeypatch):
    """Two products share an output root only if neither can skip on the other.

    Both write `workflow_result.json`, so reuse has to key on the command and
    artifact ID, not just the request and plan -- otherwise a materialized
    selection would silently stand in for an unrun sample analysis.
    """
    project = tmp_path / "project"
    project.mkdir()
    output = tmp_path / "shared-output"
    _patch_project_execution(monkeypatch)

    workflow_contract.run_project_materialization_workflow(project, "uid-ref", output_root=output)
    analysis = workflow_contract.run_project_sample_analysis_workflow(
        project,
        "uid-ref",
        output_root=output,
    )

    result = json.loads(analysis.read_text(encoding="utf-8"))
    assert result["command"] == "project.sample-analysis"
    assert result["outcome"] == "success"
    assert (output / "sample_analysis.parquet").is_file()


def test_project_embedding_workflow_success_skip_and_validation(tmp_path, monkeypatch):
    """The `embedding` plan target needs a matching executable and validation path."""
    project = tmp_path / "project"
    project.mkdir()
    output = tmp_path / "embedding-output"
    _patch_project_execution(monkeypatch)

    first = workflow_contract.run_project_embedding_workflow(
        project,
        "uid-ref",
        output_root=output,
    )
    result = json.loads(first.read_text(encoding="utf-8"))
    assert result["command"] == "project.embedding"
    assert result["target"] == "embedding"
    assert result["outcome"] == "success"
    assert result["artifacts"][0]["artifact_id"] == "project:embedding"
    # The shared coordinate system is a project-scoped generation; the result
    # names it so a task output can be traced back to it.
    assert result["generation_ids"] == {"project.embedding": "generation-embed"}
    assert result["trust_local_models"] is False

    second = workflow_contract.run_project_embedding_workflow(
        project,
        "uid-ref",
        output_root=output,
    )
    assert json.loads(second.read_text(encoding="utf-8"))["outcome"] == "compatible_skip"
    assert workflow_contract.validate_workflow_output(output, project_dir=project)["valid"]

    _patch_project_execution(monkeypatch, token="source-b")
    validation = workflow_contract.validate_workflow_output(output, project_dir=project)
    assert validation["valid"] is False
    assert "project_source_stale" in {issue["code"] for issue in validation["issues"]}


def test_project_embedding_records_the_trust_decision_it_ran_under(tmp_path, monkeypatch):
    """Unpickling this project's estimators is a decision, so the result records it."""
    project = tmp_path / "project"
    project.mkdir()
    output = tmp_path / "embedding-output"
    _patch_project_execution(monkeypatch)

    path = workflow_contract.run_project_embedding_workflow(
        project,
        "uid-ref",
        output_root=output,
        trust_local_models=True,
    )

    result = json.loads(path.read_text(encoding="utf-8"))
    assert result["trust_local_models"] is True
    assert result["summary"]["fit_kind"] == "extended"
    assert result["summary"]["n_new_molecules"] == 10


def test_project_embedding_failure_writes_structured_result(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    output = tmp_path / "embedding-output"
    _patch_project_execution(monkeypatch, failure=RuntimeError("trust required"))

    with pytest.raises(workflow_contract.WorkflowContractError, match="trust required"):
        workflow_contract.run_project_embedding_workflow(project, "uid-ref", output_root=output)

    result = json.loads(
        (output / workflow_contract.WORKFLOW_RESULT_FILENAME).read_text(encoding="utf-8")
    )
    assert result["outcome"] == "failed"
    assert result["command"] == "project.embedding"
    assert result["failure"]["stage"] == "project.embedding"


def test_no_project_product_skips_against_another_in_one_root(tmp_path, monkeypatch):
    """Three products, one output root: each must run on its own terms."""
    project = tmp_path / "project"
    project.mkdir()
    output = tmp_path / "shared-output"
    _patch_project_execution(monkeypatch)

    workflow_contract.run_project_materialization_workflow(project, "uid-ref", output_root=output)
    workflow_contract.run_project_sample_analysis_workflow(project, "uid-ref", output_root=output)
    path = workflow_contract.run_project_embedding_workflow(project, "uid-ref", output_root=output)

    assert json.loads(path.read_text(encoding="utf-8"))["outcome"] == "success"
    for name in ("materialized.h5ad.gz", "sample_analysis.parquet", "embedding.parquet"):
        assert (output / name).exists(), name


def test_every_project_product_validates_after_relocation(tmp_path, monkeypatch):
    """A finished project task output is handed onward, often at another path.

    Project validation re-plans against the source project, so it has to tolerate
    the output tree itself having moved -- every artifact pointer inside the
    result is relative for exactly that reason.
    """
    project = tmp_path / "project"
    project.mkdir()
    _patch_project_execution(monkeypatch)
    runners = {
        "materialization": workflow_contract.run_project_materialization_workflow,
        "sample-analysis": workflow_contract.run_project_sample_analysis_workflow,
        "embedding": workflow_contract.run_project_embedding_workflow,
    }
    for name, runner in runners.items():
        output = tmp_path / f"{name}-output"
        runner(project, "uid-ref", output_root=output)
        assert workflow_contract.validate_workflow_output(output, project_dir=project)["valid"]

        relocated = tmp_path / "moved" / name
        relocated.parent.mkdir(exist_ok=True)
        shutil.move(str(output), str(relocated))

        validation = workflow_contract.validate_workflow_output(relocated, project_dir=project)
        assert validation["valid"] is True, (name, validation["issues"])
        # And the relocated tree still detects a tampered artifact.
        result = json.loads(
            (relocated / workflow_contract.WORKFLOW_RESULT_FILENAME).read_text(encoding="utf-8")
        )
        artifact = next(
            item for item in result["artifacts"] if item["artifact_id"].startswith("project:")
        )
        (relocated / artifact["path"]).write_bytes(b"tampered")
        tampered = workflow_contract.validate_workflow_output(relocated, project_dir=project)
        assert tampered["valid"] is False, name
        assert "artifact_checksum_mismatch" in {issue["code"] for issue in tampered["issues"]}

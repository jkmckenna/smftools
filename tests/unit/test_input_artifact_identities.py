"""Per-source input identity, shared by raw and basecall generations (`BCS-07`)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from smftools.cli.helpers import (
    basecall_input_artifact_ids,
    load_experiment_config,
    raw_input_artifact_ids,
    resolved_input_source_identities,
)

pytestmark = pytest.mark.unit


def _pod5_config(tmp_path: Path) -> Path:
    pod5_dir = tmp_path / "pod5"
    pod5_dir.mkdir()
    (pod5_dir / "signal.pod5").write_bytes(b"fake-pod5-bytes")
    fasta = tmp_path / "ref.fasta"
    fasta.write_text(">ref\nACGT\n", encoding="utf-8")
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    config = tmp_path / "experiment_config.csv"
    config.write_text(
        "variable,value\n"
        "smf_modality,deaminase\n"
        f"input_data_path,{pod5_dir}\n"
        "model,hac\n"
        f"model_dir,{model_dir}\n"
        f"fasta,{fasta}\n"
        f"output_directory,{tmp_path / 'store'}\n"
        "experiment_id,probe\n",
        encoding="utf-8",
    )
    return config


def test_basecall_input_artifact_ids_are_per_source_not_one_aggregate_digest(
    tmp_path: Path,
) -> None:
    cfg = load_experiment_config(_pod5_config(tmp_path))

    identities = basecall_input_artifact_ids(cfg)

    assert identities[0].startswith("input-manifest:")
    assert identities[1:] == [entry for entry in identities[1:] if entry.startswith("source:")]
    assert len(identities) == 2  # one manifest digest + one pod5 source row


def test_basecall_input_artifact_ids_omit_the_alignment_reference_raw_includes(
    tmp_path: Path,
) -> None:
    cfg = load_experiment_config(_pod5_config(tmp_path))

    basecall_ids = basecall_input_artifact_ids(cfg)
    raw_ids = raw_input_artifact_ids(cfg)

    assert not any(entry.startswith("alignment-reference-bundle:") for entry in basecall_ids)
    assert raw_ids[: len(basecall_ids)] == basecall_ids
    assert raw_ids[-1].startswith("alignment-reference-bundle:")


def test_resolved_input_source_identities_is_the_shared_prefix(tmp_path: Path) -> None:
    cfg = load_experiment_config(_pod5_config(tmp_path))

    assert resolved_input_source_identities(cfg) == basecall_input_artifact_ids(cfg)


def test_resolved_input_source_identities_empty_without_input_files() -> None:
    cfg = SimpleNamespace(input_manifest_path=None, input_files=None, fasta=None)

    assert resolved_input_source_identities(cfg) == []
    assert raw_input_artifact_ids(cfg) == []

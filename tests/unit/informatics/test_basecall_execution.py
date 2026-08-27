"""Dorado basecalling with intermediate reuse (`BCS-05` execution step)."""

from __future__ import annotations

from pathlib import Path

import pytest

from smftools.informatics import basecall_execution as be

pytestmark = pytest.mark.unit


def _params(tmp_path: Path, pod5: Path, **overrides) -> dict:
    kwargs = {
        "input_data_path": pod5,
        "output_directory": tmp_path / "run",
        "workspace_directory": tmp_path / "workspace",
        "model": "dna_r10.4.1_e8.2_400bps_hac@v5.0.0",
        "model_dir": tmp_path / "models",
        "modality": "deaminase",
        "barcode_kit": None,
        "barcode_both_ends": False,
        "device": "cpu",
        "emit_moves": False,
        "trim": True,
    }
    kwargs.update(overrides)
    return kwargs


@pytest.fixture
def pod5(tmp_path: Path) -> Path:
    path = tmp_path / "signal.pod5"
    path.write_bytes(b"fake-pod5-bytes")
    return path


def _fake_canoncall_writer(calls: list):
    def fake(model_dir, model, input_path, kit, out_prefix, suffix, *rest):
        calls.append(("canoncall", model))
        Path(out_prefix + suffix).write_bytes(b"fake-bam-bytes")

    return fake


def _fake_modcall_writer(calls: list):
    def fake(model_dir, model, input_path, kit, mods, out_prefix, suffix, *rest):
        calls.append(("modcall", model))
        Path(out_prefix + suffix).write_bytes(b"fake-bam-bytes")

    return fake


def test_canonical_modality_dispatches_to_canoncall(tmp_path, pod5, monkeypatch):
    calls: list = []
    monkeypatch.setattr(be, "canoncall", _fake_canoncall_writer(calls))
    monkeypatch.setattr(be, "modcall", _fake_modcall_writer(calls))

    result = be.run_dorado_basecall(**_params(tmp_path, pod5))

    assert calls == [("canoncall", "dna_r10.4.1_e8.2_400bps_hac@v5.0.0")]
    assert result.bam_path.is_file()
    assert result.reused is False


def test_direct_modality_dispatches_to_modcall(tmp_path, pod5, monkeypatch):
    calls: list = []
    monkeypatch.setattr(be, "canoncall", _fake_canoncall_writer(calls))
    monkeypatch.setattr(be, "modcall", _fake_modcall_writer(calls))

    result = be.run_dorado_basecall(
        **_params(tmp_path, pod5, modality="direct", mod_list=["5mCG_5hmCG"])
    )

    assert calls == [("modcall", "dna_r10.4.1_e8.2_400bps_hac@v5.0.0")]
    assert result.bam_path.is_file()


def test_second_call_with_identical_inputs_reuses_the_intermediate(tmp_path, pod5, monkeypatch):
    calls: list = []
    monkeypatch.setattr(be, "canoncall", _fake_canoncall_writer(calls))
    monkeypatch.setattr(be, "modcall", _fake_modcall_writer(calls))

    first = be.run_dorado_basecall(**_params(tmp_path, pod5))
    second = be.run_dorado_basecall(**_params(tmp_path, pod5))

    assert len(calls) == 1  # dorado ran exactly once
    assert first.reused is False
    assert second.reused is True
    assert second.bam_path.read_bytes() == b"fake-bam-bytes"


def test_force_redo_bypasses_reuse(tmp_path, pod5, monkeypatch):
    calls: list = []
    monkeypatch.setattr(be, "canoncall", _fake_canoncall_writer(calls))
    monkeypatch.setattr(be, "modcall", _fake_modcall_writer(calls))

    be.run_dorado_basecall(**_params(tmp_path, pod5))
    be.run_dorado_basecall(**_params(tmp_path, pod5, force_redo=True))

    assert len(calls) == 2


def test_a_different_model_selector_is_not_reused(tmp_path, pod5, monkeypatch):
    calls: list = []
    monkeypatch.setattr(be, "canoncall", _fake_canoncall_writer(calls))
    monkeypatch.setattr(be, "modcall", _fake_modcall_writer(calls))

    be.run_dorado_basecall(**_params(tmp_path, pod5, model="dna_r10.4.1_e8.2_400bps_hac@v5.0.0"))
    result = be.run_dorado_basecall(
        **_params(tmp_path, pod5, model="dna_r10.4.1_e8.2_400bps_sup@v5.0.0")
    )

    assert len(calls) == 2
    assert result.reused is False


def test_before_run_hook_fires_only_when_dorado_actually_runs(tmp_path, pod5, monkeypatch):
    calls: list = []
    monkeypatch.setattr(be, "canoncall", _fake_canoncall_writer(calls))
    monkeypatch.setattr(be, "modcall", _fake_modcall_writer(calls))
    hook_calls = []

    be.run_dorado_basecall(**_params(tmp_path, pod5, before_run=lambda: hook_calls.append(1)))
    assert hook_calls == [1]

    be.run_dorado_basecall(**_params(tmp_path, pod5, before_run=lambda: hook_calls.append(1)))
    assert hook_calls == [1]  # not called again on the reuse path


def test_reused_bam_path_has_the_original_content(tmp_path, pod5, monkeypatch):
    monkeypatch.setattr(be, "canoncall", _fake_canoncall_writer([]))
    monkeypatch.setattr(be, "modcall", _fake_modcall_writer([]))

    first = be.run_dorado_basecall(**_params(tmp_path, pod5))
    second = be.run_dorado_basecall(**_params(tmp_path, pod5))

    assert first.bam_path.read_bytes() == second.bam_path.read_bytes()

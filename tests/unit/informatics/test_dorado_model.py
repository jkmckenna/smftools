from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from smftools.informatics.dorado_model import (
    DoradoBasecallOptions,
    DoradoModelError,
    DoradoRunCondition,
    build_dorado_basecaller_argv,
    read_pod5_run_conditions,
    resolve_dorado_basecall,
)

pytestmark = pytest.mark.unit

_FLAGS = (
    "--barcode-both-ends",
    "--batchsize",
    "--device",
    "--disable-read-splitting",
    "--emit-moves",
    "--emit-summary",
    "--kit-name",
    "--min-qscore",
    "--modified-bases-models",
    "--output-dir",
    "--read-ids",
    "--trim",
)


def _catalog():
    return {
        "dna_test_5khz": {
            "sample_type": "DNA",
            "sampling_rate": 5000,
            "flowcells": ["FLOW-A"],
            "kits": ["KIT-A"],
            "simplex_models": {
                "dna_test_5khz_hac@v1.0.0": {
                    "variant": "hac",
                    "outdated": True,
                },
                "dna_test_5khz_hac@v2.0.0": {
                    "variant": "hac",
                    "modified_models": {
                        "dna_test_5khz_hac@v2.0.0_6mA@v1": {
                            "variant": "6mA",
                            "canonical_base": "A",
                        },
                        "dna_test_5khz_hac@v2.0.0_6mA@v2": {
                            "variant": "6mA",
                            "canonical_base": "A",
                        },
                    },
                },
                "dna_test_5khz_sup@v2.0.0": {"variant": "sup"},
            },
        }
    }


def _runner(catalog=None, flags=_FLAGS, version="1.3.1+fake"):
    catalog = _catalog() if catalog is None else catalog

    def run(command, **_kwargs):
        if command[-1] == "--version":
            return subprocess.CompletedProcess(command, 0, stdout=f"{version}\n", stderr="")
        if command[-2:] == ["basecaller", "--help"]:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout="\n".join(f"  {flag}" for flag in flags),
                stderr="",
            )
        if command[-2:] == ["download", "--list-structured"]:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps(catalog),
                stderr="",
            )
        raise AssertionError(command)

    return run


def _install_model(root, name, content=b"weights"):
    model = root / name
    model.mkdir(parents=True)
    (model / "config.toml").write_text(f'name = "{name}"\n', encoding="utf-8")
    (model / "weights.tensor").write_bytes(content)
    return model


def _resolve(tmp_path, options=None, *, model_root=None, runner=None, conditions=None):
    executable = tmp_path / "dorado"
    executable.touch(exist_ok=True)
    model_root = model_root or tmp_path / "models"
    conditions = conditions or (DoradoRunCondition("FLOW-A", "KIT-A", 5000),)
    return resolve_dorado_basecall(
        options or DoradoBasecallOptions(model="hac@latest", emit_moves=False),
        (tmp_path / "reads.pod5",),
        model_root,
        executable=executable,
        runner=runner or _runner(),
        condition_reader=lambda _paths: conditions,
    )


def test_resolves_latest_installed_simplex_and_exact_modification_bundle(tmp_path):
    simplex = "dna_test_5khz_hac@v2.0.0"
    modification = f"{simplex}_6mA@v2"
    _install_model(tmp_path / "models", simplex)
    _install_model(tmp_path / "models", modification, b"modified")
    options = DoradoBasecallOptions(
        model="hac@latest",
        modified_bases=("6mA",),
        read_splitting="disable",
        trim="all",
        emit_moves=True,
        min_qscore=7.5,
        barcode_kit="KIT-A",
        barcode_both_ends=True,
        device="cpu",
    )

    resolution = _resolve(tmp_path, options)
    argv = build_dorado_basecaller_argv(
        resolution,
        tmp_path / "signal",
        tmp_path / "read_ids.txt",
        tmp_path / "output",
    )

    assert resolution.simplex_model.name == simplex
    assert [model.name for model in resolution.modification_models] == [modification]
    assert resolution.dorado_version == "1.3.1+fake"
    assert resolution.chemistry == "dna_test_5khz"
    assert len(resolution.model_bundle_digest) == 64
    assert "--read-ids" in argv
    assert "--disable-read-splitting" in argv
    assert argv[argv.index("--trim") + 1] == "all"
    assert argv[argv.index("--modified-bases-models") + 1] == str(
        tmp_path / "models" / modification
    )
    assert "<signal-input>" in resolution.normalized_argv
    assert str(tmp_path) not in " ".join(resolution.normalized_argv)


def test_exact_version_selector_can_resolve_an_outdated_installed_model(tmp_path):
    model = "dna_test_5khz_hac@v1.0.0"
    _install_model(tmp_path / "models", model)

    resolution = _resolve(
        tmp_path,
        DoradoBasecallOptions(model="hac@v1.0.0", emit_moves=False),
    )

    assert resolution.simplex_model.name == model


def test_latest_selector_does_not_silently_fall_back_to_old_installed_model(tmp_path):
    _install_model(tmp_path / "models", "dna_test_5khz_hac@v1.0.0")

    with pytest.raises(DoradoModelError, match="not installed") as error:
        _resolve(tmp_path)

    assert error.value.code == "dorado_model_not_installed"


def test_bundle_digest_is_path_independent_and_changes_with_model_bytes(tmp_path):
    model = "dna_test_5khz_hac@v2.0.0"
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    _install_model(first_root, model)
    second_model = _install_model(second_root, model)

    first = _resolve(tmp_path, model_root=first_root)
    second = _resolve(tmp_path, model_root=second_root)
    (second_model / "weights.tensor").write_bytes(b"changed")
    changed = _resolve(tmp_path, model_root=second_root)

    assert first.model_bundle_digest == second.model_bundle_digest
    assert first.model_bundle_digest != changed.model_bundle_digest
    assert first.semantic_payload() == second.semantic_payload()


def test_missing_required_capability_has_stable_error(tmp_path):
    _install_model(tmp_path / "models", "dna_test_5khz_hac@v2.0.0")
    flags = tuple(flag for flag in _FLAGS if flag != "--read-ids")

    with pytest.raises(DoradoModelError, match="--read-ids") as error:
        _resolve(tmp_path, runner=_runner(flags=flags))

    assert error.value.code == "dorado_capability_missing"


def test_older_no_trim_flag_is_used_when_structured_trim_is_unavailable(tmp_path):
    model = "dna_test_5khz_hac@v2.0.0"
    _install_model(tmp_path / "models", model)
    flags = tuple(flag for flag in _FLAGS if flag != "--trim") + ("--no-trim",)

    resolution = _resolve(tmp_path, runner=_runner(flags=flags))

    assert "--no-trim" in resolution.normalized_argv
    assert "--trim" not in resolution.normalized_argv


def test_missing_compatible_modification_model_has_stable_error(tmp_path):
    _install_model(tmp_path / "models", "dna_test_5khz_hac@v2.0.0")

    with pytest.raises(DoradoModelError, match="5mCG") as error:
        _resolve(
            tmp_path,
            DoradoBasecallOptions(
                model="hac@latest",
                modified_bases=("5mCG",),
                emit_moves=False,
            ),
        )

    assert error.value.code == "dorado_modification_model_unavailable"


def test_multiple_source_chemistries_are_rejected(tmp_path):
    _install_model(tmp_path / "models", "dna_test_5khz_hac@v2.0.0")
    catalog = _catalog()
    catalog["dna_other"] = {
        "sampling_rate": 4000,
        "flowcells": ["FLOW-B"],
        "kits": ["KIT-B"],
        "simplex_models": {"dna_other_hac@v1.0.0": {"variant": "hac"}},
    }
    conditions = (
        DoradoRunCondition("FLOW-A", "KIT-A", 5000),
        DoradoRunCondition("FLOW-B", "KIT-B", 4000),
    )

    with pytest.raises(DoradoModelError, match="multiple Dorado chemistries") as error:
        _resolve(tmp_path, runner=_runner(catalog=catalog), conditions=conditions)

    assert error.value.code == "dorado_chemistry_ambiguous"


def test_checked_in_pod5_exposes_deterministic_run_conditions():
    fixture = Path(__file__).parents[2] / "_test_inputs" / "_test_pod5_I.pod5"

    conditions = read_pod5_run_conditions((fixture,))

    assert conditions == (DoradoRunCondition("FLO-MIN114", "SQK-NBD114-24", 5000),)


def test_fake_dorado_executable_is_probed_without_running_basecalls(tmp_path):
    catalog = _catalog()
    model = "dna_test_5khz_hac@v2.0.0"
    _install_model(tmp_path / "models", model)
    executable = tmp_path / "fake-dorado"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys\n"
        f"catalog = {catalog!r}\n"
        f"flags = {_FLAGS!r}\n"
        "if sys.argv[1:] == ['--version']:\n"
        "    print('1.3.1+fake-executable')\n"
        "elif sys.argv[1:] == ['basecaller', '--help']:\n"
        "    print('\\n'.join(flags))\n"
        "elif sys.argv[1:] == ['download', '--list-structured']:\n"
        "    print(json.dumps(catalog))\n"
        "else:\n"
        "    raise SystemExit(2)\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)

    resolution = resolve_dorado_basecall(
        DoradoBasecallOptions(model="hac@latest", emit_moves=False),
        (tmp_path / "reads.pod5",),
        tmp_path / "models",
        executable=executable,
        condition_reader=lambda _paths: (DoradoRunCondition("FLOW-A", "KIT-A", 5000),),
    )

    assert resolution.dorado_version == "1.3.1+fake-executable"
    assert resolution.simplex_model.name == model

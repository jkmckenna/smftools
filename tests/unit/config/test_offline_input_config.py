"""Config loading must survive an archived input volume (`PSR-01`).

Before this, `ExperimentConfig.from_var_dict` discovered input files eagerly and
`discover_input_files` raised on an absent path, so `smftools experiment hmm` on
a fully-processed run failed the moment the raw data's drive was unplugged --
despite hmm reading nothing from it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from smftools.cli.helpers import stage_config_hash
from smftools.config import ExperimentConfig, LoadExperimentConfig
from smftools.config.input_availability import INPUT_OFFLINE

pytestmark = pytest.mark.unit

OFFLINE_INPUT = "/Volumes/ArchiveDriveForTests/run/fastq_pass"


def _config(tmp_path: Path, input_path: str) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    fasta = tmp_path / "ref.fasta"
    fasta.write_text(">ref\nACGT\n", encoding="utf-8")
    config = tmp_path / "experiment_config.csv"
    config.write_text(
        "variable,value\n"
        "smf_modality,deaminase\n"
        f"input_data_path,{input_path}\n"
        f"fasta,{fasta}\n"
        f"output_directory,{tmp_path / 'store'}\n"
        "experiment_id,probe\n",
        encoding="utf-8",
    )
    return config


def _load(config_path: Path) -> ExperimentConfig:
    cfg, _ = ExperimentConfig.from_var_dict(
        LoadExperimentConfig(config_path).var_dict, date_str="260101"
    )
    return cfg


def test_offline_input_parses_instead_of_raising(tmp_path):
    cfg = _load(_config(tmp_path, OFFLINE_INPUT))
    assert cfg.input_availability == INPUT_OFFLINE
    assert cfg.input_unavailable_volume == "/Volumes/ArchiveDriveForTests"


def test_missing_input_still_fails_at_config_load(tmp_path):
    """Deferring discovery must not delay catching a mistyped path."""
    with pytest.raises(ValueError, match="does not exist"):
        _load(_config(tmp_path, str(tmp_path / "typo")))


def test_stage_requiring_raw_input_refuses_while_offline(tmp_path):
    cfg = _load(_config(tmp_path, OFFLINE_INPUT))
    with pytest.raises(FileNotFoundError, match="not attached"):
        cfg.require_input_available("raw")


def test_validate_does_not_report_archived_input_as_broken(tmp_path):
    cfg = _load(_config(tmp_path, OFFLINE_INPUT))
    errors = cfg.validate(require_paths=True, raise_on_error=False)
    assert not [e for e in errors if "input_data_path" in e]


def test_offline_run_hashes_identically_to_the_attached_run(tmp_path, monkeypatch):
    """The decisive property.

    `input_type`/`input_files` feed every stage's config hash. If detaching the
    volume moved that hash, a finished raw generation would read as incompatible
    and each downstream stage would try to re-ingest unreachable data -- so the
    offline path would still be broken, just later and less legibly.

    The declared path is identical in both loads, as it is in reality; only
    whether the volume answers changes.
    """
    import smftools.config.experiment_config as experiment_config
    from smftools.config.input_availability import InputAvailability

    run_root = tmp_path / "run"
    source_dir = tmp_path / "archive" / "fastq_pass"
    source_dir.mkdir(parents=True)
    names = ("a.fastq.gz", "b.fastq.gz")
    for name in names:
        (source_dir / name).write_bytes(b"")

    config_path = _config(run_root, str(source_dir))
    attached_hash = stage_config_hash(_load(config_path), "raw")

    # What the completed run recorded, which is where the offline load recovers
    # the input identity discovery can no longer produce.
    manifest_dir = run_root / "store" / "raw_outputs" / "input_manifest"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "resolved_input_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "base_directory": str(source_dir),
                "sources": [{"path": str(source_dir / name)} for name in names],
            }
        ),
        encoding="utf-8",
    )

    real = experiment_config.resolve_input_availability

    def detached(path, **kwargs):
        if path is not None and str(path) == str(source_dir):
            return InputAvailability(
                state=INPUT_OFFLINE,
                path=Path(path),
                volume=Path("/Volumes/ArchiveDriveForTests"),
                detail="simulated detach",
            )
        return real(path, **kwargs)

    monkeypatch.setattr(experiment_config, "resolve_input_availability", detached)
    offline_cfg = _load(config_path)

    assert offline_cfg.input_availability == INPUT_OFFLINE
    assert offline_cfg.input_type == "fastq"
    assert len(offline_cfg.input_files) == len(names)
    assert stage_config_hash(offline_cfg, "raw") == attached_hash


def test_config_load_relocates_input_through_the_replica_catalog(tmp_path, monkeypatch):
    """End-to-end `PSR-12`: the config still names the old path, unedited.

    A finished run's input volume is reattached under a different mount point
    and name. With no catalog, this is indistinguishable from a genuinely
    unplugged drive (`test_offline_input_parses_instead_of_raising`). With one,
    `from_var_dict` must resolve straight through to `present` and discover the
    relocated files -- not merely report `present` and then crash trying to
    read the stale path (`discover_input_files` raises on an absent path).
    """
    import shutil

    from smftools.data import volume_discovery
    from smftools.data.replica_catalog import add_replica, save_catalog
    from smftools.data.volume_stamp import init_volume
    from smftools.informatics.input_manifest import resolve_input_manifest

    monkeypatch.setattr(volume_discovery, "platform_mount_root_candidates", lambda: [])
    monkeypatch.setenv("SMFTOOLS_CONFIG_DIR", str(tmp_path / "no-user-config"))

    run_root = tmp_path / "run"
    store_dir = run_root / "store"
    original_source = tmp_path / "orig_mount" / "fastq_pass"
    names = ("a.fastq.gz", "b.fastq.gz")
    original_source.mkdir(parents=True)
    for index, name in enumerate(names):
        (original_source / name).write_bytes(f"@r{index}\nAC\n+\n!!\n".encode())

    config_path = _config(run_root, str(original_source))
    digest = resolve_input_manifest(
        output_directory=store_dir,
        input_paths=[original_source / name for name in names],
    ).digest

    new_mount = tmp_path / "new_mount"
    new_mount.mkdir()
    stamp, _ = init_volume(new_mount, label="renamed-drive", kind="archive")
    shutil.copytree(original_source, new_mount / "fastq_pass")
    shutil.rmtree(tmp_path / "orig_mount")  # the old mount point is well and truly gone

    catalog = add_replica({}, digest, volume_id=stamp.volume_id, path="fastq_pass")
    save_catalog(catalog)
    monkeypatch.setenv("SMFTOOLS_VOLUME_SEARCH_PATHS", str(new_mount))

    cfg = _load(config_path)

    assert cfg.input_availability == "present"
    assert cfg.input_data_path == new_mount / "fastq_pass"
    assert cfg.input_files is not None
    assert len(cfg.input_files) == len(names)
    cfg.require_input_available("raw")  # must not raise

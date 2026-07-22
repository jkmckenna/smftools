## test_LoadExperimentConfig

from importlib.resources import as_file, files

import pytest

from smftools.config.experiment_config import LoadExperimentConfig

csv_resource = files("tests._test_inputs").joinpath("test_experiment_config_direct_I.csv")


@pytest.fixture
def config():
    with as_file(csv_resource) as csv_path:
        return LoadExperimentConfig(csv_path)


def test_var_dict(config):
    assert isinstance(config.var_dict, dict)
    assert all(isinstance(k, str) for k in config.var_dict.keys())


def test_latent_partitioned_config_defaults_and_bool_parsing(tmp_path):
    from smftools.cli.helpers import load_experiment_config

    with as_file(csv_resource) as csv_path:
        rows = csv_path.read_text(encoding="utf-8-sig").splitlines()
    rows.extend(
        [
            "latent_execution_mode,partitioned,,,str",
            "latent_max_fit_reads,123,,,int",
            "latent_run_cp,False,,,bool",
        ]
    )
    config_path = tmp_path / "latent_config.csv"
    config_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    cfg = load_experiment_config(str(config_path))

    assert cfg.latent_execution_mode == "partitioned"
    assert cfg.latent_max_fit_reads == 123
    assert cfg.latent_run_cp is False
    assert cfg.latent_n_pcs == 10


def test_partitioned_hmm_fit_selection_defaults():
    from smftools.cli.helpers import load_experiment_config

    with as_file(csv_resource) as csv_path:
        config = load_experiment_config(str(csv_path))

    assert config.hmm_max_fit_reads == 1000
    assert config.hmm_fit_selection_seed == 0


def test_region_catalog_config_defaults_are_independent():
    from smftools.cli.helpers import load_experiment_config

    with as_file(csv_resource) as csv_path:
        config = load_experiment_config(str(csv_path))

    assert config.alignment_regions_bed == config.fasta_regions_of_interest
    assert config.analysis_regions_bed is None
    assert config.plot_regions_bed is None
    assert config.spatial_regions_bed is None


def test_legacy_fasta_regions_alias_resolves_with_warning(tmp_path):
    from smftools.config import ExperimentConfig

    legacy = tmp_path / "alignment.bed"
    with pytest.warns(FutureWarning, match="deprecated"):
        config, _ = ExperimentConfig.from_var_dict(
            {"fasta_regions_of_interest": str(legacy)}, defaults_map={}
        )

    assert config.alignment_regions_bed == str(legacy)
    assert config.fasta_regions_of_interest == str(legacy)


def test_equal_alignment_and_legacy_alias_values_are_accepted(tmp_path):
    from smftools.config import ExperimentConfig

    bed = tmp_path / "alignment.bed"
    with pytest.warns(FutureWarning, match="deprecated"):
        config, _ = ExperimentConfig.from_var_dict(
            {
                "alignment_regions_bed": str(bed),
                "fasta_regions_of_interest": str(bed),
            },
            defaults_map={},
        )

    assert config.alignment_regions_bed == str(bed)


def test_conflicting_alignment_and_legacy_alias_values_fail(tmp_path):
    from smftools.config import ExperimentConfig

    with pytest.warns(FutureWarning, match="deprecated"):
        with pytest.raises(ValueError, match="conflict"):
            ExperimentConfig.from_var_dict(
                {
                    "alignment_regions_bed": str(tmp_path / "new.bed"),
                    "fasta_regions_of_interest": str(tmp_path / "legacy.bed"),
                },
                defaults_map={},
            )


def test_spatial_regions_bed_is_not_promoted_to_new_scopes(tmp_path):
    from smftools.config import ExperimentConfig

    config, _ = ExperimentConfig.from_var_dict(
        {"spatial_regions_bed": str(tmp_path / "legacy-spatial.bed")}, defaults_map={}
    )

    assert config.spatial_regions_bed == str(tmp_path / "legacy-spatial.bed")
    assert config.alignment_regions_bed is None
    assert config.analysis_regions_bed is None
    assert config.plot_regions_bed is None


def test_region_catalog_paths_participate_in_config_validation(tmp_path):
    from smftools.config import ExperimentConfig

    config = ExperimentConfig(
        input_data_path=str(tmp_path),
        output_directory=str(tmp_path / "output"),
        fasta=str(tmp_path / "missing.fa"),
        analysis_regions_bed=str(tmp_path / "missing.bed"),
    )

    with pytest.raises(ValueError, match="analysis_regions_bed does not exist"):
        config.validate(require_paths=True)


def test_repeated_stage_loads_reuse_immutable_resource_envelope():
    from smftools.cli import helpers

    helpers._RESOURCE_ENVELOPE_CACHE.clear()
    with as_file(csv_resource) as csv_path:
        first = helpers.load_experiment_config(str(csv_path))
        second = helpers.load_experiment_config(str(csv_path))

    assert first._resource_envelope is second._resource_envelope
    assert first.threads == second.threads == first._resource_envelope.resolved_threads


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("threads", "0", "threads must be a positive integer"),
        ("threads", "1.5", "threads must be an integer"),
        ("max_memory_percent", "101", "max_memory_percent must be in"),
        ("max_memory_percent", "invalid", "max_memory_percent must be numeric"),
        ("max_memory_gb", "-1", "max_memory_gb must be positive"),
        ("memory_reserve_gb", "-1", "memory_reserve_gb must be non-negative"),
        ("target_task_memory_mb", "0", "target_task_memory_mb must be positive"),
        ("hmm_max_fit_reads", "0", "hmm_max_fit_reads must be positive"),
    ],
)
def test_invalid_resource_values_fail_during_config_loading(tmp_path, key, value, message):
    from smftools.cli.helpers import load_experiment_config

    with as_file(csv_resource) as csv_path:
        rows = csv_path.read_text(encoding="utf-8-sig").splitlines()
    replacement = f"{key},{value},invalid test value,,float"
    found = any(row.startswith(f"{key},") for row in rows)
    rows = [replacement if row.startswith(f"{key},") else row for row in rows]
    if not found:
        rows.append(replacement)
    config_path = tmp_path / f"invalid_{key}.csv"
    config_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_experiment_config(str(config_path))

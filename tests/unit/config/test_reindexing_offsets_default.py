import json
from importlib import resources

from smftools.config.experiment_config import ExperimentConfig

_DEFAULTS_DIR = resources.files("smftools").joinpath("config")


def _base_var_dict(tmp_path, **extra):
    input_bam = tmp_path / "input.bam"
    input_bam.write_text("stub")
    var_dict = {
        "input_data_path": str(input_bam),
        "output_directory": str(tmp_path / "outputs"),
        "experiment_name": "test_experiment",
    }
    var_dict.update(extra)
    return var_dict


def test_reindexing_offsets_defaults_to_empty_dict_not_none_placeholder(tmp_path):
    # default.yaml used to spell its "no offsets configured" default as
    # {null: null} (a YAML mapping with an explicit null key) rather than a
    # genuinely empty mapping. deep_merge unions dict-valued fields rather
    # than replacing them, so that placeholder key survived forever inside
    # cfg.reindexing_offsets even once a real config supplied real entries --
    # and a dict with both a None key and str keys can't be json.dump'd with
    # sort_keys=True (used by experiment_manifest.py), crashing raw ingestion.
    cfg, _ = ExperimentConfig.from_var_dict(
        _base_var_dict(tmp_path),
        defaults_dir=_DEFAULTS_DIR,
    )

    assert cfg.reindexing_offsets == {}


def test_reindexing_offsets_from_csv_has_no_leaked_none_key(tmp_path):
    cfg, _ = ExperimentConfig.from_var_dict(
        _base_var_dict(tmp_path, reindexing_offsets={"6B6_top": -3052}),
        defaults_dir=_DEFAULTS_DIR,
    )

    assert cfg.reindexing_offsets == {"6B6_top": -3052}
    # The actual crash: experiment_manifest.py writes the resolved config
    # through json.dump(..., sort_keys=True), which fails if any dict value
    # mixes None and str keys.
    json.dumps({"reindexing_offsets": cfg.reindexing_offsets}, sort_keys=True, default=str)

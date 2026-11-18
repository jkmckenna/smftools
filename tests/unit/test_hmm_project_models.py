"""Project HMMs (`HPS-06`): fit once, store, decode other data with the stored model."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from smftools.hmm.project_models import FrozenModels
from smftools.informatics.partition_read import load_spine, materialize
from smftools.informatics.raw_store import write_raw_store
from smftools.preprocessing.partitioned_executor import execute_partitioned_preprocessing
from smftools.project import hmm_models
from smftools.tools.partitioned_hmm import execute_partitioned_hmm

from .test_hmm_partitioned_cli import _frame, _hmm_cfg, _preprocess_cfg

pytestmark = pytest.mark.unit

SEQUENCE = "ACGCGTACGTAC"


def _quiet(monkeypatch):
    from smftools import memory_guard
    from smftools.tools import partitioned_hmm

    monkeypatch.setattr(
        memory_guard,
        "run_tasks_parallel",
        lambda worker, task_args_list, **kwargs: [worker(*args) for args in task_args_list],
    )
    for name in (
        "_plot_feature_fractions",
        "_plot_feature_count_size_histograms",
        "_plot_molecule_fractions",
        "_plot_hmm_vs_raw_scatter",
        "_plot_hmm_parameters_across_barcodes",
        "_plot_hmm_fit_history",
        "_plot_feature_clustermaps",
    ):
        monkeypatch.setattr(partitioned_hmm, name, lambda *args, **kwargs: None)


def _two_sample_spine(tmp_path):
    """bc1 (enzyme A) and bc2 (enzyme B) reads on ``ref_top``."""
    frame = _frame()
    extra = frame.copy()
    extra["read_id"] = ["read3", "read4"]
    extra["barcode"] = extra["sample"] = "bc2"
    frame = pd.concat([frame, extra], ignore_index=True)
    raw = write_raw_store(
        frame,
        tmp_path / "raw_outputs",
        reference_lengths={"ref_top": 12},
        analysis_mode="locus",
        extra_uns={"References": {"ref_FASTA_sequence": SEQUENCE}},
    )
    preprocess = execute_partitioned_preprocessing(
        raw["spine"], _preprocess_cfg(), tmp_path / "preprocess_outputs"
    )
    sheet = tmp_path / "sheet.csv"
    pd.DataFrame({"Barcode": ["bc1", "bc2"], "enzyme": ["A", "B"], "dose": ["2", "4"]}).to_csv(
        sheet, index=False
    )
    return preprocess["spine"], sheet


def _cfg(sheet, **overrides):
    return _hmm_cfg(
        hmm_methbases=["C"],
        hmm_max_iter=2,
        target_task_memory_mb=1,
        sample_sheet_path=str(sheet),
        sample_sheet_mapping_column="Barcode",
        **overrides,
    )


def _task_artifacts(outputs) -> dict[str, dict]:
    """barcode -> the artifact record its task was decoded with."""
    catalog = pd.read_parquet(outputs["task_catalog"])
    from smftools.readwrite import safe_read_zarr

    found = {}
    for _, row in catalog.iterrows():
        task, _ = safe_read_zarr(outputs["task_catalog"].parent / row["group_path"])
        found[str(task.uns["barcode"])] = task.uns["hmm_model_artifacts"][0]
    return found


def test_frozen_models_map_references_by_name_then_strand_suffix(tmp_path):
    frozen = FrozenModels(
        name="m",
        generation_id="g",
        models_dir=tmp_path,
        artifacts={"top": {"C": {}}, "bottom": {"C": {}}},
        reference_groups={"top": ["6B6_top", "6BALB_cJ_top"], "bottom": ["6B6_bottom"]},
        exclude_references=frozenset({"ctcf_mNanog_top"}),
    )
    assert frozen.group_for("6BALB_cJ_top") == "top"
    assert frozen.group_for("6B6_enh_del_top") == "top"
    assert frozen.group_for("6B6_enh_del_bottom") == "bottom"
    assert frozen.group_for("ctcf_mNanog_top") is None
    assert frozen.group_for("other") is None


def test_matching_barcodes_compares_numbers_numerically():
    rows = {"bc1": {"enzyme": "A", "dose": "2.0"}, "bc2": {"enzyme": "A", "dose": "4"}}
    assert hmm_models.matching_barcodes(rows, {"enzyme": "A", "dose": 2}) == {"bc1"}
    assert hmm_models.matching_barcodes(rows, {"enzyme": "A"}) == {"bc1", "bc2"}
    assert hmm_models.matching_barcodes(rows, {"enzyme": "B"}) == set()


def test_spec_needs_its_keys_and_defaults_apply_samples(tmp_path):
    path = tmp_path / "spec.json"
    path.write_text(json.dumps({"name": "m", "experiment": "e"}))
    with pytest.raises(hmm_models.ProjectModelError, match="samples"):
        hmm_models.load_spec(path)
    path.write_text(
        json.dumps(
            {
                "name": "m",
                "experiment": "e",
                "samples": {"enzyme": "A", "dose": 2},
                "reference_groups": {"g": ["ref"]},
            }
        )
    )
    spec = hmm_models.load_spec(path)
    assert spec["apply_samples"] == {"enzyme": "A", "dose": 2}
    assert spec["exclude_references"] == []


def test_frozen_apply_decodes_only_matching_samples_with_the_stored_model(tmp_path, monkeypatch):
    _quiet(monkeypatch)
    spine, sheet = _two_sample_spine(tmp_path)
    fitted = execute_partitioned_hmm(spine, _cfg(sheet), tmp_path / "fitted")
    stored = _task_artifacts(fitted)["bc1"]

    cfg = _cfg(sheet)
    cfg._hmm_frozen_models = FrozenModels(
        name="m",
        generation_id="g1",
        models_dir=fitted["models"],
        artifacts={"top": {stored["model_key"]["label"]: stored}},
        reference_groups={"top": ["ref_top"]},
        barcodes=frozenset({"bc1"}),
    )
    applied = execute_partitioned_hmm(spine, cfg, tmp_path / "applied")

    fits = pd.read_parquet(applied["fit_catalog"])
    assert set(fits["fit_kind"]) == {"PROJECT"}
    assert set(fits["model_id"]) == {stored["model_id"]}
    used = _task_artifacts(applied)
    assert set(used) == {"bc1"}
    assert used["bc1"]["model_id"] == stored["model_id"]
    # The stored checkpoint travels with the generation.
    assert (applied["models"] / Path(stored["checkpoint"]).relative_to("models")).is_file()
    # Only decoded reads are on the spine; bc2's are absent, not zero-filled.
    decoded = load_spine(applied["spine"])
    assert set(decoded.obs_names.astype(str)) == {"read1", "read2"}
    # The same model decodes bc1 exactly as the fitting run did.
    layers = sorted(load_spine(fitted["spine"]).uns["hmm_layer_absent_fill"])
    assert layers
    layer = next((name for name in layers if "accessible" in name), layers[0])
    before = materialize(
        fitted["spine"],
        references="ref_top",
        read_ids=["read1", "read2"],
        start=0,
        end=12,
        layers=[layer],
    )
    after = materialize(
        applied["spine"],
        references="ref_top",
        read_ids=["read1", "read2"],
        start=0,
        end=12,
        layers=[layer],
    )
    np.testing.assert_allclose(after.layers[layer], before.layers[layer], equal_nan=True)


def test_frozen_apply_refuses_when_no_task_matches(tmp_path, monkeypatch):
    _quiet(monkeypatch)
    spine, sheet = _two_sample_spine(tmp_path)
    fitted = execute_partitioned_hmm(spine, _cfg(sheet), tmp_path / "fitted")
    stored = _task_artifacts(fitted)["bc1"]
    cfg = _cfg(sheet)
    cfg._hmm_frozen_models = FrozenModels(
        name="m",
        generation_id="g1",
        models_dir=fitted["models"],
        artifacts={"bottom": {stored["model_key"]["label"]: stored}},
        reference_groups={"bottom": ["ref_bottom"]},
    )
    with pytest.raises(RuntimeError, match="decodes no task"):
        execute_partitioned_hmm(spine, cfg, tmp_path / "applied")


def test_fit_project_model_stores_one_model_and_reuses_it(tmp_path, monkeypatch):
    _quiet(monkeypatch)
    spine, sheet = _two_sample_spine(tmp_path)
    cfg = _cfg(sheet)
    monkeypatch.setattr(
        hmm_models, "experiment_config", lambda project, experiment, paths=None: (cfg, tmp_path)
    )
    monkeypatch.setattr(hmm_models, "_spatial", lambda cfg, run_root: (Path(spine), "spatial1"))
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps(
            {
                "name": "enzA",
                "experiment": "exp",
                "samples": {"enzyme": "A"},
                "reference_groups": {"locus": ["ref"]},
            }
        )
    )
    project = tmp_path / "project"
    first = hmm_models.fit_project_model(project, spec)
    assert not first["reused"]
    generation, manifest, directory = hmm_models.current_model(project, "enzA")
    assert generation == first["generation_id"]
    assert manifest["training"]["barcodes"] == ["bc1"]
    assert set(manifest["artifacts"]) == {"top"}
    assert manifest["reference_groups"] == {"top": ["ref_top"]}
    (artifact,) = manifest["artifacts"]["top"].values()
    assert (directory / "models" / Path(artifact["checkpoint"]).relative_to("models")).is_file()
    reads = pd.read_parquet(directory / hmm_models.TRAINING_READS)
    assert set(reads["read_id"]) <= {"read1", "read2"}
    assert manifest["settings"]["hmm_max_iter"] == 2

    again = hmm_models.fit_project_model(project, spec)
    assert again["reused"] and again["generation_id"] == generation
    forced = hmm_models.fit_project_model(project, spec, force=True)
    assert not forced["reused"] and forced["generation_id"] != generation
    assert [row["name"] for row in hmm_models.list_project_models(project)] == ["enzA"]


def test_apply_project_model_publishes_a_layer_set_and_reuses_it(tmp_path, monkeypatch):
    _quiet(monkeypatch)
    spine, sheet = _two_sample_spine(tmp_path)
    cfg = _cfg(sheet)
    monkeypatch.setattr(
        hmm_models, "experiment_config", lambda project, experiment, paths=None: (cfg, tmp_path)
    )
    monkeypatch.setattr(hmm_models, "_spatial", lambda cfg, run_root: (Path(spine), "spatial1"))
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps(
            {
                "name": "enzA",
                "experiment": "exp",
                "samples": {"enzyme": "A", "dose": 2},
                "apply_samples": {"enzyme": "A"},
                "reference_groups": {"locus": ["ref"]},
            }
        )
    )
    project = tmp_path / "project"
    fitted = hmm_models.fit_project_model(project, spec)

    (first,) = hmm_models.apply_project_model(project, "enzA", experiments=["exp"])
    assert first["status"] == "decoded" and not first["reused"]
    assert Path(first["spine"]).parent.parent.parent.name == "project-enzA"
    decoded = load_spine(first["spine"])
    assert set(decoded.obs_names.astype(str)) == {"read1", "read2"}
    # The training experiment marks the reads the model was fitted on.
    assert decoded.obs["hmm_fit_read"].any()
    manifest = json.loads((Path(first["spine"]).parent / "generation_manifest.json").read_text())
    assert manifest["project_model"]["generation_id"] == fitted["generation_id"]

    (again,) = hmm_models.apply_project_model(project, "enzA", experiments=["exp"])
    assert again["reused"] and again["generation_id"] == first["generation_id"]
    hmm_models.fit_project_model(project, spec, force=True)
    (refit,) = hmm_models.apply_project_model(project, "enzA", experiments=["exp"])
    assert not refit["reused"]


def test_apply_skips_an_experiment_without_matching_samples(tmp_path, monkeypatch):
    _quiet(monkeypatch)
    spine, sheet = _two_sample_spine(tmp_path)
    cfg = _cfg(sheet)
    monkeypatch.setattr(
        hmm_models, "experiment_config", lambda project, experiment, paths=None: (cfg, tmp_path)
    )
    monkeypatch.setattr(hmm_models, "_spatial", lambda cfg, run_root: (Path(spine), "spatial1"))
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps(
            {
                "name": "enzC",
                "experiment": "exp",
                "samples": {"enzyme": "A"},
                "apply_samples": {"enzyme": "C"},
                "reference_groups": {"locus": ["ref"]},
            }
        )
    )
    project = tmp_path / "project"
    hmm_models.fit_project_model(project, spec)
    (result,) = hmm_models.apply_project_model(project, "enzC", experiments=["exp"])
    assert result == {"experiment": "exp", "status": "skipped", "reason": "no matching sample"}


def test_a_project_sample_sheet_matches_barcodes_by_number(tmp_path):
    from types import SimpleNamespace

    sheet = tmp_path / "samples.csv"
    pd.DataFrame(
        {
            "experiment_id": ["exp", "exp", "other"],
            "barcode": ["11", "12", "11"],
            "enzyme": [None, "M.CviPI", "X"],
        }
    ).to_csv(sheet, index=False)
    spine = SimpleNamespace(
        obs=pd.DataFrame({"Barcode": ["SQK-NBD114-24_barcode11", "SQK-NBD114-24_barcode12"]})
    )
    spec = {
        "sample_sheet": {
            "path": "samples.csv",
            "experiment_column": "experiment_id",
            "barcode_column": "barcode",
            "match_on": "barcode_number",
        }
    }
    rows = hmm_models.sample_attributes(tmp_path, spec, spine, None, "exp")
    assert rows["SQK-NBD114-24_barcode12"]["enzyme"] == "M.CviPI"
    assert "enzyme" not in rows["SQK-NBD114-24_barcode11"]  # blank, not "nan"
    assert hmm_models.matching_barcodes(rows, {"enzyme": "M.CviPI"}) == {"SQK-NBD114-24_barcode12"}


def test_fit_references_reads_tiled_and_pooled_fits_from_their_members():
    from types import SimpleNamespace

    tiled = SimpleNamespace(
        reference="genome",
        members=(
            SimpleNamespace(reference="6B6_top"),
            SimpleNamespace(reference="6BALB_cJ_top"),
            SimpleNamespace(reference="6B6_top"),
        ),
    )
    assert hmm_models.fit_references(tiled) == (["6B6_top", "6BALB_cJ_top"], "top")
    single = SimpleNamespace(reference="6B6_bottom", members=())
    assert hmm_models.fit_references(single) == (["6B6_bottom"], "bottom")
    mixed = SimpleNamespace(
        reference="genome",
        members=(SimpleNamespace(reference="6B6_top"), SimpleNamespace(reference="6B6_bottom")),
    )
    with pytest.raises(hmm_models.ProjectModelError, match="spans strands"):
        hmm_models.fit_references(mixed)

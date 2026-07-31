from __future__ import annotations

import copy
import json

import pytest

from smftools.machine_learning.plan import (
    MLPlanValidationError,
    load_ml_plan,
    parse_ml_plan,
)

pytestmark = pytest.mark.unit


def _base_plan() -> dict:
    return {
        "schema_version": 1,
        "scope": {"kind": "project", "set": "dafseq_training"},
        "datasets": {
            "activity_reads": {
                "modalities": ["deaminase"],
                "experiments": {"include": ["exp_01", "exp_02", "exp_03"]},
                "samples": {
                    "include": [
                        "exp_01/sample_A",
                        "exp_01/sample_B",
                        "exp_02/sample_C",
                        "exp_02/sample_D",
                        "exp_03/sample_E",
                    ]
                },
                "references": ["Nkg2a"],
                "filters": {"mapping_quality_min": 20},
                "labels": {
                    "column": "activity_status",
                    "classes": {"inactive": 0, "active": 1},
                    "positive_class": "active",
                },
            },
            "new_activity_reads": {
                "modalities": ["deaminase"],
                "samples": {"include": ["exp_04/sample_F"]},
                "references": ["Nkg2a"],
            },
        },
        "splits": {
            "sample_holdout": {
                "strategy": "explicit_groups",
                "group_by": ["experiment_uid", "Sample"],
                "train_groups": [
                    "exp_01/sample_A",
                    "exp_01/sample_B",
                    "exp_02/sample_C",
                ],
                "validation_groups": ["exp_02/sample_D"],
                "test_groups": ["exp_03/sample_E"],
                "seed": 42,
            }
        },
        "balancing": {
            "weighted_training": {
                "train": {"method": "class_weight"},
                "validation": {"method": "natural"},
                "test": {"method": "natural"},
            }
        },
        "models": {
            "nb_baseline": {
                "backend": "sklearn",
                "family": "bernoulli_nb",
                "parameters": {"alpha": 1.0},
            },
            "cnn_small": {
                "backend": "torch",
                "recipe": "residual_dilated_cnn_v1",
                "overrides": {"channels": [32, 64, 128]},
            },
        },
        "jobs": {
            "train_activity": {
                "action": "train",
                "dataset": "activity_reads",
                "split": "sample_holdout",
                "balancing": "weighted_training",
                "models": ["nb_baseline", "cnn_small"],
                "evaluate": ["validation", "test"],
                "explain": ["native", "permutation"],
            },
            "apply_activity": {
                "action": "apply",
                "model": "model:immutable-model-id",
                "dataset": "new_activity_reads",
            },
            "evaluate_activity": {
                "action": "evaluate",
                "dataset": "activity_reads",
                "source_job": "train_activity",
                "evaluate": ["validation", "test"],
            },
            "explain_activity": {
                "action": "explain",
                "dataset": "activity_reads",
                "model": "nb_baseline",
                "source_job": "train_activity",
                "explain": ["native", "permutation"],
            },
            "compare_activity": {
                "action": "plot",
                "runs": ["train_activity", "run:immutable-run-id"],
                "plots": ["roc_pr", "calibration", "feature_importance"],
            },
        },
        "tracking": {"provider": "none"},
    }


def test_plan_resolves_all_job_actions_and_deaminase_default() -> None:
    plan = parse_ml_plan(_base_plan())

    assert {job.action for job in plan.jobs.values()} == {
        "train",
        "apply",
        "evaluate",
        "explain",
        "plot",
    }
    channels = plan.datasets["activity_reads"].channels
    assert [(channel.name, channel.biological_role) for channel in channels] == [
        ("accessibility", "accessibility")
    ]
    assert channels[0].sources[0].layer == "C_site_binary"
    assert plan.datasets["activity_reads"].channel_policy == "single_modality"
    assert plan.balancing["weighted_training"].validation.method == "natural"


def test_conversion_defaults_keep_accessibility_and_methylation_separate() -> None:
    raw = _base_plan()
    raw["datasets"]["activity_reads"]["modalities"] = ["conversion"]

    plan = parse_ml_plan(raw)

    channels = plan.datasets["activity_reads"].channels
    assert [
        (channel.name, channel.sources[0].layer, channel.biological_role) for channel in channels
    ] == [
        ("accessibility", "GpC_site_binary", "accessibility"),
        (
            "endogenous_methylation",
            "CpG_site_binary",
            "endogenous_methylation",
        ),
    ]


def test_direct_channels_must_be_declared_explicitly() -> None:
    raw = _base_plan()
    raw["datasets"]["activity_reads"]["modalities"] = ["direct"]

    with pytest.raises(MLPlanValidationError, match="direct-modality channels"):
        parse_ml_plan(raw)


def test_harmonized_mixed_modality_channel_requires_every_source() -> None:
    raw = _base_plan()
    raw["datasets"]["activity_reads"].update(
        {
            "modalities": ["deaminase", "conversion"],
            "channel_policy": "harmonized",
            "channels": [
                {
                    "name": "accessibility",
                    "biological_role": "accessibility",
                    "sources": [
                        {
                            "modality": "deaminase",
                            "stage": "preprocess",
                            "layer": "C_site_binary",
                            "site_context": "C",
                        }
                    ],
                }
            ],
        }
    )

    with pytest.raises(MLPlanValidationError, match=r"missing \['conversion'\]"):
        parse_ml_plan(raw)

    raw["datasets"]["activity_reads"]["channels"][0]["sources"].append(
        {
            "modality": "conversion",
            "stage": "preprocess",
            "layer": "GpC_site_binary",
            "site_context": "GpC",
        }
    )
    plan = parse_ml_plan(raw)
    assert plan.datasets["activity_reads"].channel_policy == "harmonized"


def test_union_mixed_modality_channels_allow_declared_unavailable_channels() -> None:
    raw = _base_plan()
    raw["datasets"]["activity_reads"].update(
        {
            "modalities": ["deaminase", "direct"],
            "channel_policy": "union",
            "channels": [
                {
                    "name": "deaminase_accessibility",
                    "biological_role": "accessibility",
                    "sources": [
                        {
                            "modality": "deaminase",
                            "stage": "preprocess",
                            "layer": "C_site_binary",
                            "site_context": "C",
                        }
                    ],
                },
                {
                    "name": "direct_a_accessibility",
                    "biological_role": "accessibility",
                    "sources": [
                        {
                            "modality": "direct",
                            "stage": "preprocess",
                            "layer": "A_site_binary",
                            "site_context": "A",
                        }
                    ],
                },
            ],
        }
    )

    plan = parse_ml_plan(raw)

    assert [channel.name for channel in plan.datasets["activity_reads"].channels] == [
        "deaminase_accessibility",
        "direct_a_accessibility",
    ]


def test_union_requires_at_least_one_source_for_each_selected_modality() -> None:
    raw = _base_plan()
    raw["datasets"]["activity_reads"].update(
        {
            "modalities": ["deaminase", "direct"],
            "channel_policy": "union",
            "channels": [
                {
                    "name": "accessibility",
                    "biological_role": "accessibility",
                    "sources": [
                        {
                            "modality": "deaminase",
                            "stage": "preprocess",
                            "layer": "C_site_binary",
                            "site_context": "C",
                        }
                    ],
                }
            ],
        }
    )

    with pytest.raises(MLPlanValidationError, match=r"selected modalities: \['direct'\]"):
        parse_ml_plan(raw)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda raw: raw["datasets"]["activity_reads"].update({"unexpected": True}),
            "unknown fields",
        ),
        (
            lambda raw: raw["jobs"]["train_activity"].update({"dataset": "missing"}),
            "unknown dataset",
        ),
        (
            lambda raw: raw["jobs"]["train_activity"].update({"models": ["missing"]}),
            "unknown model",
        ),
        (
            lambda raw: raw["splits"]["sample_holdout"]["test_groups"].append("exp_01/sample_A"),
            "appears in both train and test",
        ),
        (
            lambda raw: raw["balancing"]["weighted_training"]["test"].update(
                {"method": "upsample"}
            ),
            r"must be one of \['natural'\]",
        ),
    ],
)
def test_invalid_plan_contracts_fail_before_data_access(mutate, message: str) -> None:
    raw = _base_plan()
    mutate(raw)

    with pytest.raises(MLPlanValidationError, match=message):
        parse_ml_plan(raw)


def test_unsupported_schema_version_is_actionable() -> None:
    raw = _base_plan()
    raw["schema_version"] = 2

    with pytest.raises(
        MLPlanValidationError,
        match="unsupported version 2; supported version is 1",
    ):
        parse_ml_plan(raw)


def test_resolved_serialization_and_hash_are_order_stable() -> None:
    first_raw = _base_plan()
    second_raw = json.loads(json.dumps(first_raw, sort_keys=True))

    first = parse_ml_plan(first_raw)
    second = parse_ml_plan(second_raw)

    assert first.to_dict() == second.to_dict()
    assert first.canonical_json() == second.canonical_json()
    assert first.plan_hash == second.plan_hash
    assert parse_ml_plan(first.to_dict()).plan_hash == first.plan_hash
    with pytest.raises(TypeError):
        first.models["new"] = first.models["nb_baseline"]


def test_explicit_overrides_take_precedence_over_file_values() -> None:
    plan = parse_ml_plan(
        _base_plan(),
        overrides={
            "models": {
                "nb_baseline": {
                    "parameters": {"alpha": 0.25},
                }
            }
        },
    )

    assert plan.models["nb_baseline"].parameters["alpha"] == 0.25
    assert plan.models["nb_baseline"].family == "bernoulli_nb"


def test_load_json_and_yaml_produce_the_same_resolved_plan(tmp_path) -> None:
    yaml = pytest.importorskip("yaml")
    raw = _base_plan()
    json_path = tmp_path / "plan.json"
    yaml_path = tmp_path / "plan.yaml"
    json_path.write_text(json.dumps(raw), encoding="utf-8")
    yaml_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    from_json = load_ml_plan(json_path)
    from_yaml = load_ml_plan(yaml_path)

    assert from_json.canonical_json() == from_yaml.canonical_json()


def test_yaml_duplicate_named_declaration_is_rejected(tmp_path) -> None:
    yaml_path = tmp_path / "duplicate.yaml"
    yaml_path.write_text(
        """
schema_version: 1
scope: {kind: project}
datasets:
  repeated: {}
  repeated: {}
splits: {}
models: {}
jobs: {}
""",
        encoding="utf-8",
    )

    with pytest.raises(MLPlanValidationError, match="duplicate key 'repeated'"):
        load_ml_plan(yaml_path)


def test_parse_does_not_mutate_user_mapping() -> None:
    raw = _base_plan()
    before = copy.deepcopy(raw)

    parse_ml_plan(raw)

    assert raw == before

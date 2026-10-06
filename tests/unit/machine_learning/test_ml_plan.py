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


def test_leave_one_group_out_rejects_role_lists_and_fractions() -> None:
    raw = _base_plan()
    split = raw["splits"]["sample_holdout"]
    split["strategy"] = "leave_one_group_out"

    with pytest.raises(
        MLPlanValidationError,
        match="takes only train_groups",
    ):
        parse_ml_plan(raw)

    # Train-only groups are allowed (MLX-07); validation/test lists are not.
    for field in ("validation_groups", "test_groups"):
        split.pop(field)
    plan = parse_ml_plan(raw)

    assert plan.splits["sample_holdout"].strategy == "leave_one_group_out"
    assert plan.splits["sample_holdout"].train_groups


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


# --- MLX-01: label tables --------------------------------------------------


def _label_table_document(labels: dict, scope: str = "project") -> dict:
    return {
        "schema_version": 1,
        "scope": {"kind": scope},
        "datasets": {
            "reads": {
                "modalities": ["deaminase"],
                "labels": {"column": "label", "classes": {"inactive": 0, "active": 1}, **labels},
            }
        },
        "splits": {"s": {"strategy": "leave_one_group_out", "group_by": ["experiment_uid"]}},
        "models": {"nb": {"backend": "sklearn", "family": "bernoulli_nb"}},
        "jobs": {"t": {"action": "train", "dataset": "reads", "split": "s", "models": ["nb"]}},
    }


def test_table_labels_parse_and_round_trip():
    from smftools.machine_learning.plan import parse_ml_plan

    plan = parse_ml_plan(
        _label_table_document(
            {"source": "table", "table": "ml/labels.parquet", "keys": ["experiment_id", "barcode"]}
        )
    )
    labels = plan.datasets["reads"].labels
    assert (labels.source, labels.table, labels.keys) == (
        "table",
        "ml/labels.parquet",
        ("experiment_id", "barcode"),
    )
    assert parse_ml_plan(plan.to_dict()).plan_hash == plan.plan_hash


def test_obs_labels_serialise_without_table_fields():
    from smftools.machine_learning.plan import parse_ml_plan

    plan = parse_ml_plan(_label_table_document({}))
    # Plans written before MLX-01 keep their hash.
    assert set(plan.to_dict()["datasets"]["reads"]["labels"]) == {
        "column",
        "classes",
        "source",
        "missing",
        "positive_class",
    }


@pytest.mark.parametrize(
    "labels, scope, message",
    [
        (
            {"source": "table", "table": "ml/l.parquet", "keys": ["barcode"]},
            "experiment",
            "project scope",
        ),
        (
            {"source": "table", "table": "/abs/l.parquet", "keys": ["barcode"]},
            "project",
            "inside the project",
        ),
        (
            {"source": "table", "table": "../l.parquet", "keys": ["barcode"]},
            "project",
            "inside the project",
        ),
        (
            {"source": "table", "table": "ml/l.tsv", "keys": ["barcode"]},
            "project",
            ".parquet or .csv",
        ),
        ({"source": "table", "table": "ml/l.parquet", "keys": ["well"]}, "project", "unknown keys"),
        ({"source": "table", "table": "ml/l.parquet"}, "project", "keys"),
        (
            {"source": "table", "table": "ml/l.parquet", "keys": ["label"]},
            "project",
            "unknown keys",
        ),
        ({"table": "ml/l.parquet"}, "project", "only to source 'table'"),
        ({"source": "sheet"}, "project", "'obs' or 'table'"),
    ],
)
def test_table_label_declarations_are_validated(labels, scope, message):
    from smftools.machine_learning.plan import MLPlanValidationError, parse_ml_plan

    with pytest.raises(MLPlanValidationError, match=message):
        parse_ml_plan(_label_table_document(labels, scope=scope))


# --- MLX-02: position masks ------------------------------------------------


def _positions_document(positions=None, filters=None) -> dict:
    document = _label_table_document({})
    dataset = document["datasets"]["reads"]
    if positions is not None:
        dataset["positions"] = positions
    if filters is not None:
        dataset["filters"] = filters
    return document


def test_positions_resolve_to_kept_windows_and_round_trip():
    from smftools.machine_learning.plan import parse_ml_plan

    plan = parse_ml_plan(
        _positions_document({"include": [[995, 3718]], "exclude": [[3127, 3528], [1462, 1763]]})
    )
    assert plan.datasets["reads"].positions.windows() == ((995, 1462), (1763, 3127), (3528, 3718))
    assert parse_ml_plan(plan.to_dict()).plan_hash == plan.plan_hash


def test_plans_without_positions_serialise_without_them():
    from smftools.machine_learning.plan import parse_ml_plan

    assert "positions" not in parse_ml_plan(_positions_document()).to_dict()["datasets"]["reads"]


@pytest.mark.parametrize(
    "positions, filters, message",
    [
        ({"include": []}, None, "at least one window"),
        ({"include": [[10, 5]]}, None, "0 <= start < end"),
        ({"include": [[0, 10.5]]}, None, "integer"),
        ({"include": [[0, 10]], "exclude": [[0, 10]]}, None, "excludes every"),
        ({"exclude": [[0, 10]]}, None, "include"),
        ({"include": [[0, 10]]}, {"start": 0, "end": 10}, "filters.start/end"),
    ],
)
def test_position_declarations_are_validated(positions, filters, message):
    from smftools.machine_learning.plan import MLPlanValidationError, parse_ml_plan

    with pytest.raises(MLPlanValidationError, match=message):
        parse_ml_plan(_positions_document(positions, filters))


# --- MLX-03: coordinate frames ---------------------------------------------


def _frame_document(frame, scope="project") -> dict:
    document = _label_table_document({}, scope=scope)
    document["datasets"]["reads"]["coordinate_frame"] = frame
    return document


def test_coordinate_frame_parses_and_round_trips():
    from smftools.machine_learning.plan import parse_ml_plan

    plan = parse_ml_plan(
        _frame_document({"reference": "6B6", "maps": {"6B6_enh_del": "ml/maps/del.parquet"}})
    )
    frame = plan.datasets["reads"].coordinate_frame
    assert (frame.reference, dict(frame.maps)) == ("6B6", {"6B6_enh_del": "ml/maps/del.parquet"})
    assert parse_ml_plan(plan.to_dict()).plan_hash == plan.plan_hash
    assert (
        "coordinate_frame"
        not in parse_ml_plan(_positions_document()).to_dict()["datasets"]["reads"]
    )


@pytest.mark.parametrize(
    "frame, scope, message",
    [
        ({"reference": "a", "maps": {"b": "m.parquet"}}, "experiment", "project scope"),
        ({"reference": "a", "maps": {}}, "project", "at least one"),
        ({"reference": "a", "maps": {"a": "m.parquet"}}, "project", "onto itself"),
        ({"reference": "a", "maps": {"b": "/abs/m.parquet"}}, "project", "inside the project"),
        ({"reference": "a", "maps": {"b": "m.tsv"}}, "project", ".parquet or .csv"),
        ({"maps": {"b": "m.parquet"}}, "project", "reference"),
    ],
)
def test_coordinate_frame_declarations_are_validated(frame, scope, message):
    from smftools.machine_learning.plan import MLPlanValidationError, parse_ml_plan

    with pytest.raises(MLPlanValidationError, match=message):
        parse_ml_plan(_frame_document(frame, scope=scope))


# --- MLX-07: training-only groups ------------------------------------------


def test_single_class_groups_policy_is_validated_and_hash_neutral():
    raw = _base_plan()
    split = raw["splits"]["sample_holdout"]
    for field in ("train_groups", "validation_groups", "test_groups"):
        split.pop(field)
    split["strategy"] = "leave_one_group_out"
    default = parse_ml_plan(raw)
    assert "single_class_groups" not in default.to_dict()["splits"]["sample_holdout"]

    split["single_class_groups"] = "train"
    plan = parse_ml_plan(raw)
    assert plan.splits["sample_holdout"].single_class_groups == "train"
    assert plan.plan_hash != default.plan_hash
    assert parse_ml_plan(plan.to_dict()).plan_hash == plan.plan_hash

    split["single_class_groups"] = "drop"
    with pytest.raises(MLPlanValidationError, match="'refuse' or 'train'"):
        parse_ml_plan(raw)

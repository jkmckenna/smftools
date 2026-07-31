from __future__ import annotations

import copy

import numpy as np
import pytest

from smftools.machine_learning.contracts import (
    ML_CAPABILITY_SCHEMA_VERSION,
    InputCompatibilityError,
    InputSchema,
    LabelSchema,
    MaskSpec,
    MLContractError,
    PredictorCapabilities,
    assert_input_compatible,
    validate_mask_arrays,
    validate_mask_relationships,
    validate_mask_usage,
    validate_predictor_masks,
)
from smftools.machine_learning.plan import parse_ml_plan

pytestmark = pytest.mark.unit


def _plan_dataset(
    *,
    modality: str | tuple[str, ...] = "deaminase",
    channels: list[dict] | None = None,
    channel_policy: str | None = None,
):
    modalities = [modality] if isinstance(modality, str) else list(modality)
    dataset = {
        "modalities": modalities,
        "labels": {
            "column": "activity_status",
            "classes": {"inactive": 0, "active": 1},
            "positive_class": "active",
        },
    }
    if channels is not None:
        dataset["channels"] = channels
    if channel_policy is not None:
        dataset["channel_policy"] = channel_policy
    plan = parse_ml_plan(
        {
            "schema_version": 1,
            "scope": {"kind": "project"},
            "datasets": {"reads": dataset},
            "splits": {
                "groups": {
                    "strategy": "explicit_groups",
                    "group_by": ["experiment_uid", "Sample"],
                    "train_groups": ["exp1/a"],
                    "validation_groups": ["exp2/b"],
                    "test_groups": ["exp3/c"],
                }
            },
            "models": {
                "nb": {
                    "backend": "sklearn",
                    "family": "bernoulli_nb",
                }
            },
            "jobs": {
                "train": {
                    "action": "train",
                    "dataset": "reads",
                    "split": "groups",
                    "models": ["nb"],
                }
            },
        }
    )
    return plan.datasets["reads"]


def _input_schema(
    *,
    modality: str = "deaminase",
    masks: tuple[MaskSpec, ...] | None = None,
) -> InputSchema:
    return InputSchema.from_dataset(
        _plan_dataset(modality=modality),
        reference="Nkg2a",
        n_positions=8,
        masks=masks,
    )


def _capabilities(
    *,
    supported: tuple[str, ...] = (),
    required: tuple[str, ...] = (),
    position_masks: bool = False,
) -> PredictorCapabilities:
    return PredictorCapabilities.from_dict(
        {
            "schema_version": ML_CAPABILITY_SCHEMA_VERSION,
            "backend": "test",
            "probability_output": True,
            "incremental_fit": False,
            "sample_weights": False,
            "position_masks": position_masks,
            "gradients": False,
            "convolutional_layers": False,
            "attention_data": False,
            "supported_mask_kinds": list(supported),
            "required_mask_kinds": list(required),
        }
    )


def test_deaminase_input_schema_round_trip_is_hash_stable() -> None:
    schema = _input_schema()

    assert schema.axes == ("observation", "position", "channel")
    assert schema.channels[0].name == "accessibility"
    assert schema.channels[0].sources[0].layer == "C_site_binary"
    assert [mask.kind for mask in schema.masks] == [
        "observed",
        "availability",
        "design",
    ]

    restored = InputSchema.from_dict(schema.to_dict())
    assert restored == schema
    assert restored.schema_hash == schema.schema_hash


def test_conversion_preserves_gpc_and_cpg_biological_roles() -> None:
    schema = _input_schema(modality="conversion")

    assert [
        (channel.sources[0].site_context, channel.biological_role) for channel in schema.channels
    ] == [
        ("GpC", "accessibility"),
        ("CpG", "endogenous_methylation"),
    ]


def test_direct_smf_explicit_independent_channels_round_trip() -> None:
    dataset = _plan_dataset(
        modality="direct",
        channels=[
            {
                "name": "a_accessibility",
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
            {
                "name": "cpg_methylation",
                "biological_role": "endogenous_methylation",
                "sources": [
                    {
                        "modality": "direct",
                        "stage": "preprocess",
                        "layer": "CpG_site_binary",
                        "site_context": "CpG",
                    }
                ],
            },
        ],
    )
    schema = InputSchema.from_dataset(
        dataset,
        reference="Nkg2a",
        n_positions=8,
        transforms={"a_accessibility": "binary_v1"},
    )

    assert [channel.name for channel in schema.channels] == [
        "a_accessibility",
        "cpg_methylation",
    ]
    assert schema.channels[0].transform_id == "binary_v1"
    assert InputSchema.from_dict(schema.to_dict()) == schema


def test_union_schema_preserves_modality_specific_sources_and_masks() -> None:
    dataset = _plan_dataset(
        modality=("direct", "deaminase"),
        channel_policy="union",
        channels=[
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
    )

    schema = InputSchema.from_dataset(
        dataset,
        reference="Nkg2a",
        n_positions=8,
    )

    assert schema.modalities == ("deaminase", "direct")
    assert [source.modality for channel in schema.channels for source in channel.sources] == [
        "deaminase",
        "direct",
    ]
    assert {mask.kind for mask in schema.masks} >= {
        "observed",
        "availability",
        "design",
    }


def test_equal_shape_with_different_channel_order_is_incompatible() -> None:
    schema = _input_schema(modality="conversion")
    raw = schema.to_dict()
    raw["channels"] = list(reversed(raw["channels"]))
    reordered = InputSchema.from_dict(raw)

    with pytest.raises(InputCompatibilityError, match="ordered channel"):
        assert_input_compatible(schema, reordered)


def test_equal_shape_with_different_cpg_role_is_incompatible() -> None:
    schema = _input_schema(modality="conversion")
    raw = schema.to_dict()
    raw["channels"][1]["biological_role"] = "accessibility"
    changed = InputSchema.from_dict(raw)

    with pytest.raises(InputCompatibilityError, match="ordered channel"):
        assert_input_compatible(schema, changed)


def test_mask_contracts_fix_polarity_consumers_and_phases() -> None:
    observed = MaskSpec.standard("observed")
    design = MaskSpec.standard("design")
    padding = MaskSpec.standard("padding")
    corruption = MaskSpec.standard("corruption")
    loss = MaskSpec.standard("loss", axes=("observation",))

    assert observed.true_means == "value_is_measured"
    assert design.true_means == "value_is_enabled_by_design"
    assert design.axes == ("position", "channel")
    assert padding.true_means == "position_is_padding"
    assert corruption.phases == ("train",)
    assert loss.consumers == ("loss",)
    assert MaskSpec.from_dict(loss.to_dict()) == loss


def test_mask_serialization_rejects_reversed_polarity() -> None:
    raw = MaskSpec.standard("padding").to_dict()
    raw["true_means"] = "position_is_valid"

    with pytest.raises(MLContractError, match="must be 'position_is_padding'"):
        MaskSpec.from_dict(raw)


def test_mask_array_shape_and_dtype_follow_declared_axes() -> None:
    schema = _input_schema()
    arrays = {
        "observed": np.ones((3, 8, 1), dtype=bool),
        "availability": np.ones((3, 1), dtype=bool),
        "design": np.ones((8, 1), dtype=bool),
    }

    validate_mask_arrays(schema, arrays, batch_size=3)

    arrays["availability"] = np.ones((3, 8), dtype=bool)
    with pytest.raises(MLContractError, match="does not match axes"):
        validate_mask_arrays(schema, arrays, batch_size=3)


def test_full_mask_array_validation_rejects_omitted_design_mask() -> None:
    schema = _input_schema()

    with pytest.raises(MLContractError, match=r"missing declared masks: \['design'\]"):
        validate_mask_arrays(
            schema,
            {
                "observed": np.ones((2, 8, 1), dtype=bool),
                "availability": np.ones((2, 1), dtype=bool),
            },
            batch_size=2,
        )


def test_unavailable_channel_cannot_be_marked_as_measured() -> None:
    schema = _input_schema()
    arrays = {
        "observed": np.ones((2, 8, 1), dtype=bool),
        "availability": np.array([[True], [False]], dtype=bool),
    }

    with pytest.raises(MLContractError, match="cannot be marked observed"):
        validate_mask_relationships(schema, arrays)

    arrays["observed"][1, :, :] = False
    validate_mask_relationships(schema, arrays)


def test_corruption_only_selects_observed_values() -> None:
    schema = _input_schema(
        masks=(
            MaskSpec.standard("observed"),
            MaskSpec.standard("availability"),
            MaskSpec.standard("design"),
            MaskSpec.standard("corruption"),
        )
    )
    observed = np.ones((1, 8, 1), dtype=bool)
    observed[:, 3, :] = False
    corruption = np.zeros_like(observed)
    corruption[:, 3, :] = True

    with pytest.raises(MLContractError, match="only select observed"):
        validate_mask_relationships(
            schema,
            {
                "observed": observed,
                "availability": np.ones((1, 1), dtype=bool),
                "corruption": corruption,
            },
        )


def test_padding_positions_cannot_be_attendable() -> None:
    schema = _input_schema(
        masks=(
            MaskSpec.standard("observed"),
            MaskSpec.standard("availability"),
            MaskSpec.standard("design"),
            MaskSpec.standard("padding"),
            MaskSpec.standard("attention"),
        )
    )
    padding = np.zeros((1, 8), dtype=bool)
    attention = np.ones((1, 8), dtype=bool)
    padding[:, -1] = True

    with pytest.raises(MLContractError, match="cannot be attendable"):
        validate_mask_relationships(
            schema,
            {"padding": padding, "attention": attention},
        )


def test_phase_and_consumer_validation_separates_training_only_masks() -> None:
    schema = _input_schema(
        masks=(
            MaskSpec.standard("observed"),
            MaskSpec.standard("availability"),
            MaskSpec.standard("design"),
            MaskSpec.standard("corruption"),
            MaskSpec.standard("loss"),
        )
    )

    validate_mask_usage(
        schema,
        ["corruption"],
        consumer="pretraining_task",
        phase="train",
    )
    with pytest.raises(MLContractError, match="not valid during 'inference'"):
        validate_mask_usage(
            schema,
            ["corruption"],
            consumer="pretraining_task",
            phase="inference",
        )
    with pytest.raises(MLContractError, match="not consumed by 'predictor'"):
        validate_mask_usage(schema, ["loss"], consumer="predictor", phase="train")


def test_predictor_rejects_unsupported_and_missing_required_masks() -> None:
    schema = _input_schema()
    no_masks = _capabilities()

    with pytest.raises(MLContractError, match="does not support"):
        validate_predictor_masks(
            schema,
            no_masks,
            ["observed"],
            phase="inference",
        )

    requires_observed = _capabilities(
        supported=("observed",),
        required=("observed",),
        position_masks=True,
    )
    with pytest.raises(MLContractError, match="required masks were not provided"):
        validate_predictor_masks(
            schema,
            requires_observed,
            [],
            phase="inference",
        )
    selected = validate_predictor_masks(
        schema,
        requires_observed,
        ["observed"],
        phase="inference",
    )
    assert selected[0].kind == "observed"


def test_capabilities_round_trip_and_validate_position_mask_flag() -> None:
    capabilities = _capabilities(
        supported=("observed", "availability"),
        required=("observed",),
        position_masks=True,
    )

    restored = PredictorCapabilities.from_dict(capabilities.to_dict())
    assert restored == capabilities
    assert restored.schema_hash == capabilities.schema_hash

    raw = capabilities.to_dict()
    raw["position_masks"] = False
    with pytest.raises(MLContractError, match="must be true"):
        PredictorCapabilities.from_dict(raw)


def test_label_schema_uses_explicit_mapping_not_input_order() -> None:
    dataset = _plan_dataset()
    label = LabelSchema.from_plan_label(dataset.labels)

    assert label.class_order == ("inactive", "active")
    assert label.positive_class == "active"
    assert label.encode(["active", "inactive", None]) == (1, 0, None)
    assert LabelSchema.from_dict(label.to_dict()) == label

    with pytest.raises(MLContractError, match="unknown value 1"):
        label.encode([1])


def test_label_schema_rejects_transient_or_noncontiguous_codes() -> None:
    dataset = _plan_dataset()
    label = LabelSchema.from_plan_label(dataset.labels)
    raw = copy.deepcopy(label.to_dict())
    raw["value_to_class"] = {"inactive": 1, "active": 2}

    with pytest.raises(MLContractError, match="contiguous and match class_order"):
        LabelSchema.from_dict(raw)


@pytest.mark.parametrize(
    ("factory", "field"),
    [
        (_input_schema, "input_schema.schema_version"),
        (
            lambda: LabelSchema.from_plan_label(_plan_dataset().labels),
            "label_schema.schema_version",
        ),
        (_capabilities, "capabilities.schema_version"),
    ],
)
def test_contracts_reject_unsupported_schema_versions(factory, field: str) -> None:
    contract = factory()
    raw = contract.to_dict()
    raw["schema_version"] = 999

    with pytest.raises(MLContractError, match=field):
        type(contract).from_dict(raw)

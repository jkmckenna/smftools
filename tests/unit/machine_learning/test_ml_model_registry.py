from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB

from smftools.machine_learning.contracts import InputSchema
from smftools.machine_learning.models.registry import (
    BUILTIN_MODEL_REGISTRY,
    ModelRecipe,
    ModelRegistry,
    ModelRegistryError,
)
from smftools.machine_learning.plan import parse_ml_plan

pytestmark = pytest.mark.unit


def _input_schema(*, modality: str = "deaminase") -> InputSchema:
    dataset = {
        "modalities": [modality],
        "labels": {
            "column": "activity",
            "classes": {"inactive": 0, "active": 1},
        },
    }
    if modality == "direct":
        dataset["channels"] = [
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
            }
        ]
    plan = parse_ml_plan(
        {
            "schema_version": 1,
            "scope": {"kind": "experiment"},
            "datasets": {"reads": dataset},
            "splits": {
                "groups": {
                    "strategy": "explicit_groups",
                    "group_by": ["sample_id"],
                    "train_groups": ["a"],
                    "validation_groups": ["b"],
                    "test_groups": ["c"],
                }
            },
            "models": {"nb": {"backend": "sklearn", "family": "bernoulli_nb"}},
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
    return InputSchema.from_dataset(plan.datasets["reads"], reference="locus", n_positions=8)


def test_builtin_registry_is_explicit_and_deterministically_ordered() -> None:
    assert BUILTIN_MODEL_REGISTRY.names == (
        "bernoulli_nb",
        "logistic_regression",
        "random_forest",
    )
    assert BUILTIN_MODEL_REGISTRY.recipe_names == (
        "bernoulli_nb_v1",
        "logistic_regression_v1",
        "random_forest_v1",
    )
    assert BUILTIN_MODEL_REGISTRY.definition("bernoulli_nb").capabilities.incremental_fit
    assert not BUILTIN_MODEL_REGISTRY.definition("random_forest").capabilities.incremental_fit


@pytest.mark.parametrize(
    ("family", "expected_type"),
    [
        ("bernoulli_nb", BernoulliNB),
        ("logistic_regression", LogisticRegression),
        ("random_forest", RandomForestClassifier),
    ],
)
def test_builtin_definitions_build_approved_sklearn_families(
    family: str, expected_type: type
) -> None:
    resolved = BUILTIN_MODEL_REGISTRY.resolve(
        family,
        input_schema=_input_schema(),
    )

    model = BUILTIN_MODEL_REGISTRY.build(resolved)

    assert isinstance(model, expected_type)
    assert resolved.backend == "sklearn"
    assert resolved.architecture.parameters == resolved.config.to_dict()


def test_resolution_persists_all_defaults_and_validated_overrides() -> None:
    resolved = BUILTIN_MODEL_REGISTRY.resolve(
        "bernoulli_nb",
        input_schema=_input_schema(),
        parameters={"alpha": 0.25},
    )

    assert resolved.architecture.name == "bernoulli_nb_v1"
    assert resolved.architecture.parameters == {
        "alpha": 0.25,
        "binarize": 0.5,
        "fit_prior": True,
        "force_alpha": True,
    }
    assert resolved.config.alpha == 0.25

    with pytest.raises(ModelRegistryError, match="fields must be exactly"):
        BUILTIN_MODEL_REGISTRY.resolve(
            "bernoulli_nb",
            input_schema=_input_schema(),
            parameters={"unknown": 1},
        )


def test_recipe_round_trip_and_tamper_detection() -> None:
    recipe = BUILTIN_MODEL_REGISTRY.recipe("bernoulli_nb_v1")
    restored = ModelRecipe.from_dict(recipe.to_dict())

    assert restored == recipe
    assert restored.to_dict() == recipe.to_dict()

    tampered = deepcopy(recipe.to_dict())
    tampered["parameters"]["alpha"] = 9.0
    with pytest.raises(ModelRegistryError, match="recipe_id"):
        ModelRecipe.from_dict(tampered)


def test_recipe_parameters_are_deeply_immutable() -> None:
    recipe = ModelRecipe.create(
        name="nested_v1",
        version="1",
        family="bernoulli_nb",
        backend="sklearn",
        parameters={"nested": {"values": [1, 2]}},
        supported_modalities=("deaminase",),
        supported_channel_roles=("*",),
    )

    with pytest.raises(TypeError):
        recipe.parameters["nested"]["values"] = (9,)
    assert recipe.to_dict()["parameters"] == {"nested": {"values": [1, 2]}}


@pytest.mark.parametrize("modality", ["deaminase", "conversion", "direct"])
def test_builtin_recipes_declare_supported_smftools_modalities(modality: str) -> None:
    recipe = BUILTIN_MODEL_REGISTRY.recipe("bernoulli_nb_v1")

    recipe.assert_input_compatible(_input_schema(modality=modality))


def test_recipe_channel_role_and_count_incompatibility_fails_before_build() -> None:
    schema = _input_schema()
    unsupported_role = ModelRecipe.create(
        name="restricted_v1",
        version="1",
        family="bernoulli_nb",
        backend="sklearn",
        parameters=dict(BUILTIN_MODEL_REGISTRY.recipe("bernoulli_nb_v1").parameters),
        supported_modalities=("deaminase",),
        supported_channel_roles=("endogenous_methylation",),
    )
    too_many_channels = ModelRecipe.create(
        name="single_channel_v1",
        version="1",
        family="bernoulli_nb",
        backend="sklearn",
        parameters=dict(BUILTIN_MODEL_REGISTRY.recipe("bernoulli_nb_v1").parameters),
        supported_modalities=("conversion",),
        supported_channel_roles=("*",),
        maximum_channels=1,
    )

    with pytest.raises(ModelRegistryError, match="channel roles"):
        unsupported_role.assert_input_compatible(schema)
    with pytest.raises(ModelRegistryError, match="2 input channels"):
        too_many_channels.assert_input_compatible(_input_schema(modality="conversion"))


def test_registry_rejects_duplicate_or_cross_family_records() -> None:
    definition = BUILTIN_MODEL_REGISTRY.definition("bernoulli_nb")
    recipe = BUILTIN_MODEL_REGISTRY.recipe("bernoulli_nb_v1")

    with pytest.raises(ModelRegistryError, match="definition names must be unique"):
        ModelRegistry((definition, definition), (recipe,))
    with pytest.raises(ModelRegistryError, match="recipe names must be unique"):
        ModelRegistry((definition,), (recipe, recipe))


def test_unknown_family_and_recipe_fail_without_dynamic_registration() -> None:
    with pytest.raises(ModelRegistryError, match="unknown model family"):
        BUILTIN_MODEL_REGISTRY.definition("plugin_model")
    with pytest.raises(ModelRegistryError, match="unknown model recipe"):
        BUILTIN_MODEL_REGISTRY.recipe("plugin_recipe")


def test_build_rejects_tampered_runtime_resolution() -> None:
    resolved = BUILTIN_MODEL_REGISTRY.resolve(
        "bernoulli_nb",
        input_schema=_input_schema(),
    )

    with pytest.raises(ModelRegistryError, match="recipe identity"):
        BUILTIN_MODEL_REGISTRY.build(replace(resolved, recipe_id="f" * 64))

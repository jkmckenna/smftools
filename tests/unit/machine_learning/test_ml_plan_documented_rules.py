"""The plan rules stated in docs/source/ml/plan_reference.md, asserted.

Documentation examples rot silently. Writing that page surfaced three claims
that would have been wrong -- the scope key is ``set`` and not ``set_name``,
sklearn declares ``family`` while Torch declares ``recipe``, and an ``explain``
job needs at least one method -- each found only by running the example through
the real parser.

These tests pin the documented rules to the parser rather than the prose. When
one changes, this fails and names the page that needs updating.
"""

from __future__ import annotations

import pytest

from smftools.machine_learning.plan import MLPlanValidationError, parse_ml_plan

pytestmark = pytest.mark.unit

DOC_PAGE = "docs/source/ml/plan_reference.md"

CHANNEL = {
    "name": "accessibility",
    "biological_role": "accessibility",
    "sources": [
        {
            "modality": "conversion",
            "stage": "preprocess",
            "layer": "GpC_site_binary",
            "site_context": "GpC",
        }
    ],
}

DATASET = {
    "modalities": ["conversion"],
    "channel_policy": "single_modality",
    "channels": [CHANNEL],
    "labels": {"column": "activity", "classes": {"inactive": 0, "active": 1}},
}


def _plan(**overrides) -> dict:
    """The minimal plan exactly as documented, with targeted overrides."""
    document = {
        "schema_version": 1,
        "scope": {"kind": "experiment"},
        "datasets": {"accessibility": dict(DATASET)},
        "splits": {"by_replicate": {"strategy": "leave_one_group_out", "group_by": ["sample_id"]}},
        "models": {"nb": {"backend": "sklearn", "family": "bernoulli_nb"}},
        "jobs": {
            "train_nb": {
                "action": "train",
                "dataset": "accessibility",
                "split": "by_replicate",
                "models": ["nb"],
            }
        },
    }
    document.update(overrides)
    return document


def test_the_documented_minimal_plan_parses() -> None:
    plan = parse_ml_plan(_plan())

    assert plan.plan_hash
    assert list(plan.datasets) == ["accessibility"]
    assert list(plan.jobs) == ["train_nb"]


def test_scope_uses_set_not_set_name(tmp_path) -> None:
    # The dataclass field is set_name; the plan key is set. Documented because
    # the mismatch is invisible until the parser rejects it.
    parsed = parse_ml_plan(_plan(scope={"kind": "project", "set": "nkg2a_active_vs_inactive"}))
    assert parsed.scope.set_name == "nkg2a_active_vs_inactive"

    with pytest.raises(MLPlanValidationError, match="unknown fields"):
        parse_ml_plan(_plan(scope={"kind": "project", "set_name": "x"}))


def test_sklearn_declares_family_and_torch_declares_recipe() -> None:
    with pytest.raises(MLPlanValidationError, match="sklearn models require 'family'"):
        parse_ml_plan(_plan(models={"nb": {"backend": "sklearn", "recipe": "bernoulli_nb"}}))

    torch_plan = _plan(
        models={"cnn": {"backend": "torch", "recipe": "residual_dilated_cnn"}},
        jobs={
            "train_cnn": {
                "action": "train",
                "dataset": "accessibility",
                "split": "by_replicate",
                "models": ["cnn"],
            }
        },
    )
    assert parse_ml_plan(torch_plan).models["cnn"].recipe == "residual_dilated_cnn"

    torch_plan["models"]["cnn"] = {"backend": "torch", "family": "residual_dilated_cnn"}
    with pytest.raises(MLPlanValidationError, match="torch models require 'recipe'"):
        parse_ml_plan(torch_plan)


def test_evaluation_roles_accept_only_natural_balancing() -> None:
    assert (
        parse_ml_plan(_plan(balancing={"weighted": {"train": {"method": "class_weight"}}}))
        .balancing["weighted"]
        .train.method
        == "class_weight"
    )

    with pytest.raises(MLPlanValidationError):
        parse_ml_plan(_plan(balancing={"bad": {"validation": {"method": "downsample"}}}))


def test_explain_jobs_require_at_least_one_method() -> None:
    jobs = {
        "train_nb": {
            "action": "train",
            "dataset": "accessibility",
            "split": "by_replicate",
            "models": ["nb"],
        },
        "explain_nb": {
            "action": "explain",
            "dataset": "accessibility",
            "model": "nb",
            "source_job": "train_nb",
        },
    }
    with pytest.raises(MLPlanValidationError, match="at least one explain method"):
        parse_ml_plan(_plan(jobs=jobs))

    jobs["explain_nb"]["explain"] = ["permutation_importance"]
    assert parse_ml_plan(_plan(jobs=jobs)).jobs["explain_nb"].explain == ("permutation_importance",)


def test_unknown_keys_are_rejected_rather_than_ignored() -> None:
    # The page promises typos fail loudly. If this ever starts passing, the
    # promise is false.
    with pytest.raises(MLPlanValidationError, match="unknown fields"):
        parse_ml_plan(_plan(unexpected_key=True))


def test_the_documentation_page_still_exists() -> None:
    from pathlib import Path

    assert Path(DOC_PAGE).is_file(), (
        f"{DOC_PAGE} is gone; these tests pin rules stated there and should move with it"
    )

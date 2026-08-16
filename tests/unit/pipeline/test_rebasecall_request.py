from __future__ import annotations

import json

import pandas as pd
import pytest

from smftools.pipeline.rebasecall_request import (
    RebasecallRequestError,
    load_rebasecall_request,
    parse_selection_predicate,
    rebasecall_request_from_dict,
)

pytestmark = pytest.mark.unit


def _request(**overrides):
    payload = {
        "schema_version": 1,
        "name": "publication-2026",
        "source": {"raw_generation": "raw-a", "preprocess_generation": "pre-a"},
        "selection": {
            "mode": "qc",
            "predicate": {
                "all": [
                    {"column": "passes_read_qc", "op": "eq", "value": True},
                    {"column": "passes_dedup", "op": "eq", "value": True},
                ]
            },
        },
        "basecall": {"model": "hac@latest"},
        "signal": {"materialize": False},
        "downstream": {"target": "full"},
        "promotion": {"activate": False},
    }
    payload.update(overrides)
    return payload


def test_request_is_strict_versioned_and_relocation_independent(tmp_path):
    first = _request(
        signal={
            "materialize": True,
            "relocations": [
                {
                    "source_id": "source-a",
                    "sha256": "a" * 64,
                    "path": "relocated/reads.pod5",
                }
            ],
        }
    )
    second = json.loads(json.dumps(first))

    parsed_first = rebasecall_request_from_dict(first, base_directory=tmp_path / "one")
    parsed_second = rebasecall_request_from_dict(second, base_directory=tmp_path / "two")

    assert parsed_first.request_id == parsed_second.request_id
    assert parsed_first.signal.relocations[0].path != parsed_second.signal.relocations[0].path
    assert parsed_first.source.raw_generation == "raw-a"
    assert parsed_first.source.preprocess_generation == "pre-a"
    assert parsed_first.selection.predicate is not None
    assert parsed_first.selection.predicate.columns == ("passes_dedup", "passes_read_qc")
    assert parsed_first.basecall.read_splitting == "preserve"
    assert parsed_first.basecall.min_qscore == 0.0
    assert parsed_first.to_json() == parsed_first.to_json()


def test_set_like_request_fields_are_canonicalized(tmp_path):
    first = _request(
        selection={"mode": "ids", "id_kind": "read_id", "ids": ["b", "a"]},
        signal={
            "relocations": [
                {"source_id": "b", "sha256": "B" * 64, "path": "b.pod5"},
                {"source_id": "a", "sha256": "A" * 64, "path": "a.pod5"},
            ]
        },
    )
    second = _request(
        selection={"mode": "ids", "id_kind": "read_id", "ids": ["a", "b"]},
        signal={
            "relocations": [
                {"source_id": "a", "sha256": "a" * 64, "path": "a.pod5"},
                {"source_id": "b", "sha256": "b" * 64, "path": "b.pod5"},
            ]
        },
    )

    parsed_first = rebasecall_request_from_dict(first, base_directory=tmp_path)
    parsed_second = rebasecall_request_from_dict(second, base_directory=tmp_path)

    assert parsed_first.to_dict() == parsed_second.to_dict()
    assert parsed_first.request_id == parsed_second.request_id


def test_json_and_yaml_load_to_the_same_canonical_request(tmp_path):
    import yaml

    payload = _request()
    json_path = tmp_path / "request.json"
    yaml_path = tmp_path / "request.yaml"
    json_path.write_text(json.dumps(payload), encoding="utf-8")
    yaml_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    from_json = load_rebasecall_request(json_path)
    from_yaml = load_rebasecall_request(yaml_path)

    assert from_json.to_dict() == from_yaml.to_dict()
    assert from_json.request_id == from_yaml.request_id


def test_predicate_evaluates_nested_safe_operators_and_missing_policy():
    predicate = parse_selection_predicate(
        {
            "all": [
                {"column": "passes_read_qc", "op": "eq", "value": True},
                {
                    "not": {
                        "column": "is_duplicate",
                        "op": "eq",
                        "value": True,
                        "missing": "false",
                    }
                },
            ]
        }
    )
    frame = pd.DataFrame(
        {
            "passes_read_qc": [True, True, False],
            "is_duplicate": [False, None, False],
        },
        index=["a", "b", "c"],
    ).astype({"passes_read_qc": "boolean", "is_duplicate": "boolean"})

    assert predicate.evaluate(frame).tolist() == [True, True, False]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.update(schema_version=2), "schema_version"),
        (lambda payload: payload.update(schema_version="1"), "must be an integer"),
        (lambda payload: payload.update(name=123), "must be a string"),
        (
            lambda payload: payload.update(basecall={"model": "hac", "min_qscore": "10"}),
            "must be numeric",
        ),
        (lambda payload: payload.update(unexpected=True), "unknown field"),
        (
            lambda payload: payload.update(promotion={"activate": True}),
            "promotion.activate must be false",
        ),
        (
            lambda payload: payload.update(
                selection={
                    "mode": "qc",
                    "predicate": {"column": "arbitrary", "op": "eq", "value": True},
                }
            ),
            "not allowlisted",
        ),
        (
            lambda payload: payload.update(
                selection={
                    "mode": "ids",
                    "id_kind": "read_id",
                    "ids": ["read-a", "read-a"],
                }
            ),
            "contains duplicates",
        ),
        (
            lambda payload: payload.update(
                selection={
                    "mode": "qc",
                    "predicate": {
                        "column": "passes_read_qc",
                        "op": "in",
                        "value": [[True]],
                    },
                }
            ),
            "list of scalar values",
        ),
    ],
)
def test_request_rejects_unsafe_or_ambiguous_shapes(mutation, message):
    payload = _request()
    mutation(payload)

    with pytest.raises(RebasecallRequestError, match=message):
        rebasecall_request_from_dict(payload)


def test_missing_predicate_column_fails_by_default():
    predicate = parse_selection_predicate(
        {"column": "passes_variant_qc", "op": "eq", "value": True}
    )

    with pytest.raises(RebasecallRequestError, match="unavailable"):
        predicate.evaluate(pd.DataFrame({"passes_qc": [True]}))

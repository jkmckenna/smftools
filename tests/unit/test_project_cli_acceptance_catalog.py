"""Keep the project-CLI acceptance catalog traceable to real coverage.

The catalog claims that each PCLI item and each property PCLI-04 must prove is
either automated or deliberately deferred. Claims nobody checks decay as tests
are renamed, so every piece of cited evidence is resolved back to a test that
still exists.
"""

import json
from collections import Counter
from pathlib import Path

CATALOG_PATH = Path("tests/acceptance/project_cli_criteria.json")
EXPECTED_CATEGORY_COUNTS = {
    "finding": 1,
    "item": 4,
    "property": 12,
    "gap": 1,
}
STATUSES = {"automated", "deferred"}


def test_project_cli_acceptance_catalog_is_complete_and_traceable():
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    criteria = catalog["criteria"]
    identifiers = [entry["id"] for entry in criteria]

    assert catalog["schema_version"] == 1
    assert len(criteria) == sum(EXPECTED_CATEGORY_COUNTS.values())
    assert len(identifiers) == len(set(identifiers))
    assert Counter(identifier.split(".", 1)[0] for identifier in identifiers) == Counter(
        EXPECTED_CATEGORY_COUNTS
    )

    for entry in criteria:
        assert entry["status"] in STATUSES, entry["id"]
        assert entry["title"], entry["id"]
        if entry["status"] == "automated":
            assert entry["evidence"], entry["id"]
        else:
            assert entry.get("owner"), entry["id"]
            assert entry.get("reason"), entry["id"]
        for reference in entry["evidence"]:
            relative_path, separator, symbol = reference.partition("::")
            assert separator and symbol, reference
            evidence_path = Path(relative_path)
            assert evidence_path.is_file(), reference
            assert f"def {symbol}(" in evidence_path.read_text(encoding="utf-8"), reference


def test_every_pcli_item_and_required_property_appears():
    """The catalog covers the lane, not just the parts that went smoothly."""
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    identifiers = {entry["id"] for entry in catalog["criteria"]}

    assert {f"item.pcli_{index:02d}" for index in range(1, 5)} <= identifiers
    # The properties PCLI-04 names explicitly.
    assert {
        "property.plan_run_validate",
        "property.relocation",
        "property.source_mutation",
        "property.structured_failure",
        "property.force_behavior",
        "property.named_set_selection",
        "property.duplicate_bare_read_identity",
    } <= identifiers

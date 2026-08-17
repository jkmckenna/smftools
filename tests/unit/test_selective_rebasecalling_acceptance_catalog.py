"""Keep the selective re-basecalling acceptance catalog traceable to real coverage.

The catalog claims that every audit finding, implementation item, minimum
scenario, and validation profile of the re-basecalling program is either
automated or deliberately deferred to a named owner. A claim nobody checks
decays into fiction as tests are renamed or removed, so every piece of cited
evidence is resolved back to a test that still exists.
"""

import json
from collections import Counter
from pathlib import Path

CATALOG_PATH = Path("tests/acceptance/selective_rebasecalling_criteria.json")
EXPECTED_CATEGORY_COUNTS = {
    # One entry per audit finding (SRB-C1..M4), one per delivered implementation
    # item (SRB-01a..09), one per minimum scenario the plan lists, and one per
    # validation profile that runs outside ordinary CI.
    "finding": 12,
    "item": 18,
    "scenario": 13,
    "profile": 2,
}
STATUSES = {"automated", "deferred", "withdrawn"}


def test_selective_rebasecalling_catalog_is_complete_and_traceable():
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
            # `reason` is the vocabulary of a deferment. An automated entry that
            # carries one is either mislabelled or excusing a gap it claims not
            # to have; context that outlives the deferment goes in `note`.
            assert "reason" not in entry, entry["id"]
        else:
            # A deferment names who owns it and why, so it stays a decision
            # rather than an absence nobody remembers making.
            assert entry.get("owner"), entry["id"]
            assert entry.get("reason"), entry["id"]
            assert not entry["evidence"], entry["id"]
        if "note" in entry:
            assert isinstance(entry["note"], str) and entry["note"], entry["id"]
        for reference in entry["evidence"]:
            relative_path, separator, symbol = reference.partition("::")
            assert separator and symbol, reference
            evidence_path = Path(relative_path)
            assert evidence_path.is_file(), reference
            assert f"def {symbol}(" in evidence_path.read_text(encoding="utf-8"), reference


def test_every_audit_finding_and_delivered_item_appears_exactly_once():
    """The catalog must cover the program, not a convenient subset of it."""
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    identifiers = {entry["id"] for entry in catalog["criteria"]}

    expected_findings = {
        f"finding.srb_{name.lower()}"
        for name in ("C1", "C2", "H1", "H2", "H3", "H4", "H5", "H6", "M1", "M2", "M3", "M4")
    }
    expected_items = {
        f"item.srb_{name}"
        for name in (
            "01a",
            "01b",
            "02a",
            "02b",
            "03a",
            "03b",
            "04a",
            "04b",
            "05a",
            "05b",
            "06a",
            "06b",
            "06c",
            "07a",
            "07b",
            "08a",
            "08b",
            "09",
        )
    }

    assert expected_findings <= identifiers
    assert expected_items <= identifiers


def test_deferred_entries_stay_the_exception_and_name_what_blocks_them():
    """A catalog that defers most of itself is a plan, not evidence."""
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    criteria = catalog["criteria"]
    deferred = [entry for entry in criteria if entry["status"] == "deferred"]

    assert len(deferred) < len(criteria) // 4
    for entry in deferred:
        # A reason that does not say what is missing cannot be acted on later.
        assert len(entry["reason"]) > 60, entry["id"]

"""Keep the input/alignment acceptance catalog traceable to real coverage.

The catalog claims that each audit finding, implementation item, and acceptance
scenario of the input/alignment program is either automated, deliberately
deferred, or withdrawn. A claim nobody checks decays into fiction as tests are
renamed or removed, so every piece of cited evidence is resolved back to a test
that still exists.
"""

import json
from collections import Counter
from pathlib import Path

CATALOG_PATH = Path("tests/acceptance/input_alignment_criteria.json")
EXPECTED_CATEGORY_COUNTS = {
    # One entry per audit finding (IAR-C1..M6), one per implementation item
    # (IAR-01..15), one per audit acceptance scenario, and one per pre-IAR-15
    # ledger finding (D1..D6).
    "finding": 14,
    "item": 15,
    "scenario": 15,
    "ledger": 6,
}
STATUSES = {"automated", "deferred", "withdrawn"}


def test_input_alignment_acceptance_catalog_is_complete_and_traceable():
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
        elif entry["status"] == "deferred":
            # A deferment names who owns it and why, so it stays a decision
            # rather than an absence nobody remembers making.
            assert entry.get("owner"), entry["id"]
            assert entry.get("reason"), entry["id"]
        else:
            assert entry.get("reason"), entry["id"]
        if "note" in entry:
            assert isinstance(entry["note"], str) and entry["note"], entry["id"]
        for reference in entry["evidence"]:
            relative_path, separator, symbol = reference.partition("::")
            assert separator and symbol, reference
            evidence_path = Path(relative_path)
            assert evidence_path.is_file(), reference
            assert f"def {symbol}(" in evidence_path.read_text(encoding="utf-8"), reference


def test_every_implementation_item_and_audit_finding_appears_once():
    """The catalog must cover the program, not just the parts that went well."""
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    identifiers = {entry["id"] for entry in catalog["criteria"]}

    expected_findings = {f"finding.iar_{name}" for name in ("c1", "c2")}
    expected_findings |= {f"finding.iar_h{index}" for index in range(1, 7)}
    expected_findings |= {f"finding.iar_m{index}" for index in range(1, 7)}
    expected_items = {f"item.iar_{index:02d}" for index in range(1, 16)}
    expected_ledger = {f"ledger.d{index}" for index in range(1, 7)}

    assert expected_findings <= identifiers
    assert expected_items <= identifiers
    assert expected_ledger <= identifiers

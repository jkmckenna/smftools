import json
from collections import Counter
from pathlib import Path

CATALOG_PATH = Path("tests/acceptance/partitioned_pipeline_criteria.json")
EXPECTED_CATEGORY_COUNTS = {
    "correctness": 7,
    "genome": 8,
    "resources": 8,
    "portability": 9,
    "query": 4,
    "completion": 4,
}


def test_partitioned_pipeline_acceptance_catalog_is_complete_and_traceable():
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    criteria = catalog["criteria"]
    identifiers = [entry["id"] for entry in criteria]

    assert catalog["schema_version"] == 1
    assert len(criteria) == 40
    assert len(identifiers) == len(set(identifiers))
    assert Counter(identifier.split(".", 1)[0] for identifier in identifiers) == Counter(
        EXPECTED_CATEGORY_COUNTS
    )

    for entry in criteria:
        assert entry["status"] in {"automated", "deferred"}
        if entry["status"] == "deferred":
            assert entry.get("owner")
            assert entry.get("reason")
        else:
            assert entry["evidence"]
        for reference in entry["evidence"]:
            relative_path, separator, symbol = reference.partition("::")
            assert separator and symbol, reference
            evidence_path = Path(relative_path)
            assert evidence_path.is_file(), reference
            assert f"def {symbol}(" in evidence_path.read_text(encoding="utf-8"), reference

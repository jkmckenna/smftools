"""The mask contract stated in docs/source/ml/splits_and_masks.md, asserted.

That page carries a warning: seven mask kinds are declarable, the partition
reader produces four, and a declared mask of another kind is silently omitted
rather than rejected. Users are told to declare only the produced four.

If production is ever added for ``attention``, ``corruption``, or ``loss``, the
warning becomes wrong and misleading. This fails when that happens.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from smftools.machine_learning.contracts import MASK_KINDS
from smftools.machine_learning.data.partition_dataset import MLPartitionBatch

pytestmark = pytest.mark.unit

DOC_PAGE = Path("docs/source/ml/splits_and_masks.md")

DECLARABLE = {"observed", "availability", "design", "padding", "attention", "corruption", "loss"}
PRODUCED = {"observed", "availability", "design", "padding"}


def test_the_declarable_mask_vocabulary_is_unchanged() -> None:
    assert set(MASK_KINDS) == DECLARABLE


def test_the_batch_carries_exactly_the_produced_kinds() -> None:
    fields = {name for name in MLPartitionBatch.__dataclass_fields__ if name.endswith("_mask")}

    assert fields == {f"{kind}_mask" for kind in PRODUCED}


def test_declarable_but_unproduced_kinds_are_still_unproduced() -> None:
    # The documented warning exists because these three are accepted by the
    # schema and then silently dropped by mask_arrays. When one gains a
    # producer, update the page before deleting this.
    unproduced = DECLARABLE - PRODUCED

    assert unproduced == {"attention", "corruption", "loss"}
    for kind in unproduced:
        assert f"{kind}_mask" not in MLPartitionBatch.__dataclass_fields__


def test_the_page_documenting_this_still_exists_and_warns() -> None:
    assert DOC_PAGE.is_file(), f"{DOC_PAGE} is gone; this test pins claims made there"
    text = DOC_PAGE.read_text(encoding="utf-8")

    assert "silently omitted" in text, (
        "the page must keep warning that unproduced mask kinds are dropped without error"
    )

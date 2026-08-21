"""Row ordering for latent-clustered molecules (`EGL-28b`).

The ordering exists so a clustermap reads as solid Leiden blocks with real
structure inside each one. Two failure modes are silent rather than loud: an
order that is not a permutation quietly drops or duplicates molecules, and a
block that is not contiguous makes the annotation strips beside the heatmap
disagree with the rows they label.
"""

from __future__ import annotations

import numpy as np
import pytest

from smftools.tools.latent_ordering import (
    _numeric_aware_key,
    cluster_display_order,
    hierarchical_block_order,
    latent_row_order,
)

pytestmark = pytest.mark.unit


def _labelled_blobs(seed=0):
    """Three separated clusters; the middle one has two sub-blobs inside it."""
    rng = np.random.default_rng(seed)
    a = rng.normal([0, 0], 0.3, size=(30, 2))
    b1 = rng.normal([10, 0], 0.3, size=(15, 2))
    b2 = rng.normal([10, 5], 0.3, size=(15, 2))
    c = rng.normal([20, 0], 0.3, size=(30, 2))
    points = np.vstack([a, b1, b2, c])
    labels = np.array(["0"] * 30 + ["1"] * 30 + ["2"] * 30)
    return points, labels


# --- the invariants a clustermap depends on ---------------------------------


def test_order_is_a_permutation():
    """Anything else silently drops or duplicates molecules."""
    points, labels = _labelled_blobs()
    order, _blocks = latent_row_order(points, labels)
    assert sorted(order.tolist()) == list(range(len(labels)))


def test_blocks_are_contiguous_and_match_their_spans():
    """The spans drive the separators and annotation strips.

    If a reported span does not hold exactly one label, the strip beside the
    heatmap labels the wrong rows -- and it still renders.
    """
    points, labels = _labelled_blobs()
    order, blocks = latent_row_order(points, labels)
    reordered = labels[order]
    for label, start, stop in blocks:
        assert set(reordered[start:stop]) == {label}
    assert [stop - start for _label, start, stop in blocks] == [30, 30, 30]
    assert blocks[0][1] == 0
    assert blocks[-1][2] == len(labels)


def test_every_cluster_appears_exactly_once():
    points, labels = _labelled_blobs()
    _order, blocks = latent_row_order(points, labels)
    reported = [label for label, _start, _stop in blocks]
    assert sorted(reported) == sorted(set(labels))


# --- cluster ordering --------------------------------------------------------


def test_numeric_labels_sort_numerically():
    """Leiden labels are strings, so a plain sort gives 0, 1, 10, 11, 2, ...

    With up to 39 clusters measured on a real unit, lexicographic order is
    visibly scrambled in a figure whose point is contiguous, comparable blocks.
    """
    labels = [str(index) for index in range(12)]
    assert sorted(labels, key=_numeric_aware_key) == labels


def test_non_numeric_labels_still_order_deterministically():
    assert _numeric_aware_key("unassigned") > _numeric_aware_key("9")


def test_similar_clusters_are_placed_adjacently():
    """Adjacency in the figure should mean something.

    Two clusters near each other in the latent space must not be separated by a
    distant one just because of their label numbers.
    """
    points = np.vstack(
        [
            np.full((10, 2), [0.0, 0.0]),
            np.full((10, 2), [100.0, 0.0]),
            np.full((10, 2), [1.0, 0.0]),
        ]
    )
    labels = np.array(["0"] * 10 + ["1"] * 10 + ["2"] * 10)

    order = cluster_display_order(points, labels)

    assert abs(order.index("0") - order.index("2")) == 1, "the two near clusters must be adjacent"


def test_fewer_than_three_clusters_keeps_label_order():
    points = np.random.default_rng(0).normal(size=(20, 2))
    labels = np.array(["1"] * 10 + ["0"] * 10)
    assert cluster_display_order(points, labels) == ["0", "1"]


# --- within-block ordering ---------------------------------------------------


def test_within_a_block_similar_rows_end_up_together():
    """Ordering within a cluster is the reason to do this at all."""
    points, labels = _labelled_blobs()
    order, blocks = latent_row_order(points, labels)
    _label, start, stop = blocks[[b[0] for b in blocks].index("1")]

    # Cluster "1" holds two sub-blobs separated in the second coordinate.
    within = points[order[start:stop]]
    high = within[:, 1] > 2.5
    transitions = int(np.sum(high[1:] != high[:-1]))
    assert transitions == 1, "the two sub-blobs must form two runs, not interleave"


def test_tiny_blocks_are_returned_unchanged():
    assert hierarchical_block_order(np.zeros((2, 2))).tolist() == [0, 1]


def test_all_nan_block_does_not_raise():
    order = hierarchical_block_order(np.full((5, 2), np.nan))
    assert sorted(order.tolist()) == list(range(5))


# --- the size guard ----------------------------------------------------------


def test_large_blocks_fall_back_instead_of_linking():
    """`linkage` builds a condensed distance matrix: 1.6 GB at 20,000 rows.

    The spatial stage already hit this, where an uncapped linkage turned one
    PNG into a multi-minute stall. A Leiden cluster is bounded by nothing but
    the population, so the guard has to live here.
    """
    points = np.random.default_rng(0).normal(size=(200, 2))
    order = hierarchical_block_order(points, max_linkage_rows=50)
    assert sorted(order.tolist()) == list(range(200))


def test_the_fallback_is_deterministic():
    points = np.random.default_rng(0).normal(size=(200, 2))
    first = hierarchical_block_order(points, max_linkage_rows=50)
    second = hierarchical_block_order(points, max_linkage_rows=50)
    assert first.tolist() == second.tolist()


def test_the_fallback_still_reflects_structure():
    """Degrading to an arbitrary order would make the picture misleading.

    The projection order is one axis rather than a hierarchy, but two separated
    groups must still come out as two runs.
    """
    rng = np.random.default_rng(0)
    points = np.vstack([rng.normal(0, 0.2, size=(100, 2)), rng.normal(20, 0.2, size=(100, 2))])
    order = hierarchical_block_order(points, max_linkage_rows=50)

    is_far = (points[order][:, 0] > 10).astype(int)
    assert int(np.sum(is_far[1:] != is_far[:-1])) == 1


def test_ordering_is_reproducible_across_calls():
    points, labels = _labelled_blobs()
    first, _ = latent_row_order(points, labels)
    second, _ = latent_row_order(points, labels)
    assert first.tolist() == second.tolist()


# --- edges -------------------------------------------------------------------


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        latent_row_order(np.zeros((5, 2)), np.array(["0", "1"]))


def test_empty_input_returns_empty():
    order, blocks = latent_row_order(np.zeros((0, 2)), np.array([], dtype=str))
    assert order.size == 0
    assert blocks == []


def test_single_cluster_still_orders_within_it():
    points, _ = _labelled_blobs()
    labels = np.array(["0"] * len(points))
    order, blocks = latent_row_order(points, labels)
    assert len(blocks) == 1
    assert blocks[0] == ("0", 0, len(points))
    assert sorted(order.tolist()) == list(range(len(points)))

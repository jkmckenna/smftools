"""Reference-switching needs adjacent supporting sites (`F14`).

`segment_variant_calls` and `segment_sparse_variant_calls` interpolate a segment
class between consecutive informative sites, so a single discordant site opens a
segment of the other reference spanning every position up to the next site. With
no minimum support, that made `chimeric_variant_sites` useless: on the `241213`
pilot it flagged 100% of QC-passing reads (3,860 / 3,861) on the strength of 2-3
discordant bases out of ~2,300 self-matching ones.

`variant_chimera_min_adjacent_sites` requires a run of N consecutive informative
sites calling the other reference. Sites, not bases -- variant sites are sparse,
so a base-length floor would be a proxy for the wrong thing.

Note the deaminase chimera path has carried an equivalent floor
(`deaminase_chimera_min_events_per_span = 3`) all along; this closes the gap for
the variant caller.
"""

from __future__ import annotations

import numpy as np
import pytest

from smftools.preprocessing.variant_evidence import (
    SparseVariantCall,
    segment_sparse_variant_calls,
    segment_variant_calls,
)

pytestmark = pytest.mark.unit


def _sparse(classes, min_adjacent_sites, *, aligned_member_index=0):
    calls = [
        SparseVariantCall(position=index * 10, site_id=str(index), observed_base="A", call=call)
        for index, call in enumerate(classes)
    ]
    return segment_sparse_variant_calls(
        calls,
        span_start=0,
        span_end=len(classes) * 10,
        aligned_member_index=aligned_member_index,
        min_adjacent_sites=min_adjacent_sites,
    )


def _dense(classes, min_adjacent_sites, *, aligned_member_index=0):
    """Lay the same site classes out on a sparse genomic grid."""
    width = len(classes) * 10
    calls = np.zeros(width, dtype=int)
    for index, call in enumerate(classes):
        calls[index * 10] = call
    covered = np.ones(width, dtype=bool)
    return segment_variant_calls(
        calls,
        covered,
        aligned_member_index=aligned_member_index,
        min_adjacent_sites=min_adjacent_sites,
    )


# Two isolated discordant sites separated by concordant ones -- the shape that
# produced the pilot's 3,786 `multi_segment_mismatch` reads.
NOISE = [1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1]
# A genuine switch: four adjacent sites agreeing on the other reference.
REAL = [1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 1]


@pytest.mark.parametrize("segmenter", [_sparse, _dense], ids=["sparse", "dense"])
def test_isolated_sites_are_not_chimeric_at_two(segmenter):
    """The defect, at the threshold the user asked for."""
    result = segmenter(NOISE, 2)
    assert result.has_other_reference_segment is False
    assert result.other_reference_segment_type == "no_segment_mismatch"


@pytest.mark.parametrize("segmenter", [_sparse, _dense], ids=["sparse", "dense"])
def test_adjacent_run_is_still_chimeric(segmenter):
    """The guard against over-correcting into "nothing is ever chimeric"."""
    result = segmenter(REAL, 2)
    assert result.has_other_reference_segment is True
    assert result.other_reference_segment_type != "no_segment_mismatch"


@pytest.mark.parametrize("segmenter", [_sparse, _dense], ids=["sparse", "dense"])
def test_default_of_one_preserves_previous_behavior(segmenter):
    """Isolated sites still flag at the historical setting.

    Pinned so the change is opt-in at the library boundary and the old
    behavior stays reachable for comparison against existing generations.
    """
    result = segmenter(NOISE, 1)
    assert result.has_other_reference_segment is True
    assert result.other_reference_segment_type == "multi_segment_mismatch"


@pytest.mark.parametrize("segmenter", [_sparse, _dense], ids=["sparse", "dense"])
def test_exactly_two_adjacent_sites_qualify(segmenter):
    """The boundary is inclusive: two adjacent sites are a span of two."""
    classes = [1, 1, 2, 2, 1, 1, 1, 1]
    assert segmenter(classes, 2).has_other_reference_segment is True
    assert segmenter(classes, 3).has_other_reference_segment is False


@pytest.mark.parametrize("segmenter", [_sparse, _dense], ids=["sparse", "dense"])
def test_counts_stay_raw(segmenter):
    """Only the interpretation is gated; the statistics are not rewritten.

    `variant_other_base_count` is the evidence used to diagnose `F14` in the
    first place, so suppressing it along with the flag would remove the means
    of noticing the next such problem.
    """
    result = segmenter(NOISE, 2)
    assert result.has_other_reference_segment is False
    assert result.other_base_count > 0


@pytest.mark.parametrize("segmenter", [_sparse, _dense], ids=["sparse", "dense"])
def test_threshold_below_one_is_clamped(segmenter):
    """A nonsensical setting must not disable segmentation entirely."""
    assert segmenter(REAL, 0).has_other_reference_segment is True


@pytest.mark.parametrize("segmenter", [_sparse, _dense], ids=["sparse", "dense"])
def test_no_informative_sites_is_not_chimeric(segmenter):
    assert segmenter([1, 1, 1, 1], 2).has_other_reference_segment is False


def test_second_member_reads_are_symmetric():
    """A read aligned to member 2 must be judged against member 1, not member 2."""
    flipped = [2, 2, 2, 1, 2, 2, 2, 2, 1, 2, 2, 2]
    assert _sparse(flipped, 2, aligned_member_index=1).has_other_reference_segment is False
    real_flipped = [2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 2]
    assert _sparse(real_flipped, 2, aligned_member_index=1).has_other_reference_segment is True


def test_config_default_is_two():
    """The pipeline default is the floor, not the historical no-op."""
    from smftools.config.experiment_config import ExperimentConfig

    assert ExperimentConfig().variant_chimera_min_adjacent_sites == 2


def test_config_reads_the_override(tmp_path):
    """The knob is user-specified from the experiment config sheet."""
    from smftools.config.experiment_config import ExperimentConfig

    path = tmp_path / "experiment_config.csv"
    path.write_text(
        "variable,value,help,options,type\nvariant_chimera_min_adjacent_sites,4,,,int\n",
        encoding="utf-8",
    )
    cfg, _report = ExperimentConfig.from_csv(path)
    assert cfg.variant_chimera_min_adjacent_sites == 4

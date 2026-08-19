"""Unbarcoded reads must not become a sample (`F19`).

Dorado writes reads it could not assign to a barcode into a file named after the
run id and basecall model -- not "unclassified.bam". Two places assumed the
literal word: the caller filtering demux outputs, and the sidecar builder, which
fell back to the whole filename when its `barcode(\\d+)` pattern did not match.

The result was a run id travelling through the pipeline as a barcode. On the
`241213` pilot it was the largest "sample" in the experiment -- 85,436 of
268,489 sidecar reads, and 4,646 of 19,328 analysed reads -- appearing in every
per-sample plot and statistic as if it were a real library member.
"""

from __future__ import annotations

import pytest

from smftools.informatics.bam_functions import is_unclassified_bam_name

pytestmark = pytest.mark.unit

RUN_ID_NAME = "ba0e54a1-67ab-4f30-b7e7-15db5782e51f_dna_r10.4.1_e8.2_400bps_sup@v5.2.0"


def test_run_id_prefixed_output_is_unclassified():
    """The exact name that produced the phantom sample."""
    assert is_unclassified_bam_name(RUN_ID_NAME) is True


def test_literal_unclassified_is_still_recognised():
    """The spelling the original filter looked for must keep working."""
    assert is_unclassified_bam_name("unclassified") is True
    assert is_unclassified_bam_name("some_prefix_unclassified") is True


@pytest.mark.parametrize(
    "name",
    ["SQK-NBD114-24_barcode08", "barcode01", "BARCODE12", "SQK-RBK114-96_barcode24"],
)
def test_real_barcode_files_are_kept(name):
    assert is_unclassified_bam_name(name) is False


def test_custom_pool_names_are_kept():
    """Guard against over-correcting.

    The stem fallback exists for kits whose files are not named `barcodeNN`.
    Only the run-id shape is treated as unclassified, so those still work.
    """
    assert is_unclassified_bam_name("my_custom_pool") is False
    assert is_unclassified_bam_name("sampleA") is False


def test_matching_is_anchored_not_substring():
    """A uuid *inside* a name is not the dorado unclassified pattern."""
    assert is_unclassified_bam_name(f"barcode01_{RUN_ID_NAME}") is False


def test_bare_run_id_without_model_suffix():
    assert is_unclassified_bam_name("ba0e54a1-67ab-4f30-b7e7-15db5782e51f") is True

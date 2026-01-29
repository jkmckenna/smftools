from smftools.tools.sequence_alignment import align_sequences_with_mismatches


def test_align_sequences_with_substitution() -> None:
    aligned1, aligned2, mismatches = align_sequences_with_mismatches("ACGT", "ACCT")

    assert aligned1 == "ACGT"
    assert aligned2 == "ACCT"
    assert len(mismatches) == 1

    mismatch = mismatches[0]
    assert mismatch.event == "substitution"
    assert mismatch.seq1_pos == 2
    assert mismatch.seq1_base == "G"
    assert mismatch.seq2_pos == 2
    assert mismatch.seq2_base == "C"


def test_align_sequences_with_insertion() -> None:
    aligned1, aligned2, mismatches = align_sequences_with_mismatches("ACGT", "ACGTT")

    assert aligned1 == "ACGT-"
    assert aligned2 == "ACGTT"
    assert len(mismatches) == 1

    mismatch = mismatches[0]
    assert mismatch.event == "insertion"
    assert mismatch.seq1_pos is None
    assert mismatch.seq1_base is None
    assert mismatch.seq2_pos == 4
    assert mismatch.seq2_base == "T"


def test_align_sequences_ignore_n_mismatches() -> None:
    aligned1, aligned2, mismatches = align_sequences_with_mismatches("ACNT", "ACGT")

    assert aligned1 == "ACNT"
    assert aligned2 == "ACGT"
    assert mismatches == []

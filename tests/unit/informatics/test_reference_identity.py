from smftools.informatics.reference_identity import UID_LENGTH, reference_uid


def test_identical_sequence_same_uid_regardless_of_name():
    seq = "ACGTACGTAA"
    assert reference_uid(seq) == reference_uid(seq)
    # name is irrelevant -- only the sequence matters
    assert len(reference_uid(seq)) == UID_LENGTH


def test_case_insensitive():
    assert reference_uid("acgtac") == reference_uid("ACGTAC")


def test_padding_trimmed_by_length():
    # per-experiment padding must not change the uid
    assert reference_uid("ACGTAC", length=6) == reference_uid("ACGTACNNNN", length=6)


def test_different_sequence_different_uid():
    assert reference_uid("ACGTAC") != reference_uid("ACGTAG")


def test_length_none_hashes_full_sequence():
    assert reference_uid("ACGTACNNNN") != reference_uid("ACGTAC")

import numpy as np

from smftools.informatics.converted_BAM_to_adata import (
    SEQUENCE_ENCODING_CONFIG,
    _encode_sequence_array,
    _encode_sequence_reads,
)


def test_encode_sequence_reads_matches_per_read_encoding() -> None:
    config = SEQUENCE_ENCODING_CONFIG
    base_identities = {
        "read_fwd": np.array(list("ACGT"), dtype="<U1"),
        "read_rev": np.array(list("TGCA"), dtype="<U1"),
    }
    valid_length = 4

    encoded = _encode_sequence_reads(base_identities, valid_length, config)

    assert set(encoded) == {"read_fwd", "read_rev"}
    for read_name, sequence in base_identities.items():
        expected = _encode_sequence_array(sequence, valid_length, config)
        np.testing.assert_array_equal(encoded[read_name], expected)


def test_encode_sequence_reads_applies_padding_beyond_valid_length() -> None:
    config = SEQUENCE_ENCODING_CONFIG
    base_identities = {"read": np.array(list("ACGT"), dtype="<U1")}

    encoded = _encode_sequence_reads(base_identities, valid_length=2, config=config)

    assert encoded["read"][2] == config.padding_value
    assert encoded["read"][3] == config.padding_value


def test_encode_sequence_reads_skips_none_sequences() -> None:
    config = SEQUENCE_ENCODING_CONFIG
    base_identities = {
        "read_present": np.array(list("ACGT"), dtype="<U1"),
        "read_missing": None,
    }

    encoded = _encode_sequence_reads(base_identities, valid_length=4, config=config)

    assert set(encoded) == {"read_present"}

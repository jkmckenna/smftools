import numpy as np

from smftools.constants import (
    MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT,
    MODKIT_EXTRACT_SEQUENCE_PADDING_BASE,
)
from smftools.informatics.modkit_extract_to_adata import _encode_sequence_array


def test_encode_sequence_array_applies_padding() -> None:
    sequence = np.array(list("ACGTNX"), dtype="<U1")
    padding_value = MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT[MODKIT_EXTRACT_SEQUENCE_PADDING_BASE]

    encoded = _encode_sequence_array(
        sequence,
        valid_length=4,
        base_to_int=MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT,
        padding_value=padding_value,
    )

    assert encoded[:4].tolist() == [
        MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT["A"],
        MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT["C"],
        MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT["G"],
        MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT["T"],
    ]
    assert encoded[4:].tolist() == [padding_value, padding_value]


def test_encode_sequence_array_retains_unknown_as_n() -> None:
    sequence = np.array(list("ACGTNX"), dtype="<U1")
    padding_value = MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT[MODKIT_EXTRACT_SEQUENCE_PADDING_BASE]

    encoded = _encode_sequence_array(
        sequence,
        valid_length=6,
        base_to_int=MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT,
        padding_value=padding_value,
    )

    assert encoded.tolist() == [
        MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT["A"],
        MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT["C"],
        MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT["G"],
        MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT["T"],
        MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT["N"],
        MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT["N"],
    ]
    assert padding_value != MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT["N"]

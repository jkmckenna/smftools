from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np

from smftools.constants import (
    MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT,
    MODKIT_EXTRACT_SEQUENCE_INT_TO_BASE,
)


def encode_sequence_to_int(
    sequence: str | Iterable[str],
    *,
    base_to_int: Mapping[str, int] = MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT,
    unknown_base: str = "N",
) -> np.ndarray:
    """Encode a base sequence into integer values using constant mappings.

    Args:
        sequence: Sequence string or iterable of base characters.
        base_to_int: Mapping of base characters to integer encodings.
        unknown_base: Base to use when a character is not in the encoding map.

    Returns:
        np.ndarray: Integer-encoded sequence array.

    Raises:
        ValueError: If an unknown base is encountered and ``unknown_base`` is not mapped.
    """
    if unknown_base not in base_to_int:
        raise ValueError(f"Unknown base '{unknown_base}' not present in encoding map.")

    if isinstance(sequence, str):
        sequence_iter = sequence
    else:
        sequence_iter = list(sequence)

    fallback = base_to_int[unknown_base]
    encoded = np.fromiter(
        (base_to_int.get(base, fallback) for base in sequence_iter),
        dtype=np.int16,
        count=len(sequence_iter),
    )
    return encoded


def decode_int_sequence(
    encoded_sequence: Iterable[int] | np.ndarray,
    *,
    int_to_base: Mapping[int, str] = MODKIT_EXTRACT_SEQUENCE_INT_TO_BASE,
    unknown_base: str = "N",
) -> list[str]:
    """Decode integer-encoded bases into characters using constant mappings.

    Args:
        encoded_sequence: Iterable of integer-encoded bases.
        int_to_base: Mapping of integer encodings to base characters.
        unknown_base: Base to use when an integer is not in the decoding map.

    Returns:
        list[str]: Decoded base characters.

    Raises:
        ValueError: If ``unknown_base`` is not available for fallback.
    """
    if unknown_base not in int_to_base.values():
        raise ValueError(f"Unknown base '{unknown_base}' not present in decoding map.")

    fallback = unknown_base
    return [int_to_base.get(int(value), fallback) for value in encoded_sequence]


def phred_to_fastq_quality_string(quality_scores: Iterable[int]) -> str:
    """Encode integer Phred quality scores as a FASTQ Phred+33 quality string.

    Values are clamped to ``[0, 93]``, the printable-ASCII range for Phred+33
    encoding. This also maps the ragged store's ``-1`` missing-quality sentinel
    (see ``ragged_store.alignment_to_ragged_record``) to Phred 0 (``"!"``).

    Args:
        quality_scores: Iterable of per-base integer Phred quality scores.

    Returns:
        str: FASTQ quality line, one ASCII character per base.
    """
    return "".join(chr(min(max(int(value), 0), 93) + 33) for value in quality_scores)

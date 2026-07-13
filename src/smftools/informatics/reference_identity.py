"""Canonical, name-independent reference identity via sequence hashing.

Different experiments sometimes name the same locus differently across FASTAs. A
sequence hash gives a stable ``reference_uid`` that is identical whenever the
underlying (unpadded, uppercased) reference sequence is identical, regardless of the
FASTA record name -- so the project layer can harmonize references automatically.
Near-identical (not byte-identical) references are handled separately by manual
aliases in the project ``reference_registry.yaml``.
"""

from __future__ import annotations

import hashlib

UID_LENGTH = 16


def reference_uid(sequence: str, length: int | None = None) -> str:
    """Return a canonical sequence-identity hash for a reference.

    Args:
        sequence: The reference sequence (may include trailing padding).
        length: If given, the true unpadded length; the sequence is trimmed to it
            before hashing so per-experiment padding does not change the uid.

    Returns:
        A short hex digest (name-independent) identifying the sequence.
    """
    seq = str(sequence).upper()
    if length is not None:
        seq = seq[: int(length)]
    return hashlib.sha256(seq.encode("utf-8")).hexdigest()[:UID_LENGTH]

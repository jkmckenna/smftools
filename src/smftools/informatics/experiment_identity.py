"""Canonical human-readable experiment identity resolution."""

from __future__ import annotations

from collections.abc import Mapping


def _normalized_identity(value: object | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def resolve_experiment_id(
    candidates: Mapping[str, object | None],
    *,
    required: bool = False,
) -> str | None:
    """Resolve matching experiment-id candidates or report every conflict.

    Args:
        candidates: Identity values keyed by their user-facing source names.
            Empty and ``None`` values are ignored; every remaining value must
            match exactly.
        required: Raise when no candidate supplies an identity.

    Returns:
        The single normalized identity, or ``None`` when none was supplied and
        ``required`` is false.

    Raises:
        ValueError: Candidates conflict or a required identity is absent.
    """
    normalized = {
        str(source): value
        for source, raw_value in candidates.items()
        if (value := _normalized_identity(raw_value)) is not None
    }
    identities = set(normalized.values())
    if len(identities) > 1:
        details = ", ".join(f"{source}={value!r}" for source, value in normalized.items())
        raise ValueError(f"experiment identity mismatch: {details}")
    if not identities:
        if required:
            sources = ", ".join(str(source) for source in candidates)
            raise ValueError(f"experiment identity is required from one of: {sources}")
        return None
    return next(iter(identities))

"""Stable experiment and molecule identities for partitioned stores."""

from __future__ import annotations

import base64
import hashlib
import uuid

IDENTITY_SCHEMA_VERSION = 1
EXPERIMENT_UID_COLUMN = "experiment_uid"
MOLECULE_UID_COLUMN = "molecule_uid"
READ_ID_COLUMN = "read_id"


def new_experiment_uid() -> str:
    """Return a new opaque experiment identity suitable for persistent storage."""
    return str(uuid.uuid4())


def legacy_experiment_uid(project_identity: object, experiment_id: object) -> str:
    """Return a stable compatibility UID for a pre-identity registry entry."""
    return str(
        uuid.uuid5(uuid.NAMESPACE_URL, f"smftools:legacy:{project_identity!s}:{experiment_id!s}")
    )


def validate_experiment_uid(value: object) -> str:
    """Normalize and validate a persisted experiment UUID."""
    try:
        return str(uuid.UUID(str(value)))
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError(f"invalid experiment_uid: {value!r}") from exc


def molecule_uid(experiment_uid: object, read_id: object) -> str:
    """Return a deterministic compact identity for one experiment/read pair.

    The hash input is length-prefixed so arbitrary read identifiers cannot create
    delimiter ambiguities. A 128-bit SHA-256 prefix keeps indexes compact while
    retaining ample collision resistance; the unhashed primary key remains
    ``(experiment_uid, read_id)``.
    """
    experiment = validate_experiment_uid(experiment_uid).encode("utf-8")
    read = str(read_id).encode("utf-8")
    payload = len(experiment).to_bytes(4, "big") + experiment + len(read).to_bytes(8, "big") + read
    return hashlib.sha256(payload).hexdigest()[:32]


def pooled_obs_name(experiment_uid: object, read_id: object) -> str:
    """Encode a reversible project-wide observation name."""

    def _encode(value: str) -> str:
        return base64.urlsafe_b64encode(value.encode("utf-8")).decode("ascii").rstrip("=")

    return f"m1.{_encode(validate_experiment_uid(experiment_uid))}.{_encode(str(read_id))}"


def split_pooled_obs_name(value: object) -> tuple[str, str]:
    """Decode an observation name produced by :func:`pooled_obs_name`."""
    parts = str(value).split(".")
    if len(parts) != 3 or parts[0] != "m1":
        raise ValueError(f"invalid pooled molecule observation name: {value!r}")

    def _decode(component: str) -> str:
        padding = "=" * (-len(component) % 4)
        try:
            return base64.urlsafe_b64decode(component + padding).decode("utf-8")
        except (ValueError, UnicodeDecodeError) as exc:
            raise ValueError(f"invalid pooled molecule observation name: {value!r}") from exc

    return validate_experiment_uid(_decode(parts[1])), _decode(parts[2])

"""Stable experiment and molecule identities for partitioned stores."""

from __future__ import annotations

import base64
import hashlib
import re
import uuid

IDENTITY_SCHEMA_VERSION = 2
EXPERIMENT_UID_COLUMN = "experiment_uid"
MOLECULE_UID_COLUMN = "molecule_uid"
READ_ID_COLUMN = "read_id"
TEMPLATE_ID_COLUMN = "template_id"
SEGMENT_ID_COLUMN = "segment_id"
SEGMENT_UID_COLUMN = "segment_uid"
_MATE_SUFFIX_RE = re.compile(r"(?:/|[._-](?:R|read)?)([12])$", re.IGNORECASE)


def template_read_id(read_id: object) -> str:
    """Return the shared template name for a SAM query/segment name."""
    return _MATE_SUFFIX_RE.sub("", str(read_id))


def alignment_segment_id(read: object) -> str:
    """Return a unique primary-segment identity without changing BAM QNAME.

    Paired SAM records intentionally share a query name. Raw storage requires
    one unique row key per physical alignment segment, so R1/R2 receive a
    canonical ``/1`` or ``/2`` suffix while the unsuffixed value remains the
    template identity. Unpaired/long-read names are unchanged.
    """
    return alignment_segment_id_from_fields(
        getattr(read, "query_name"),
        paired=bool(getattr(read, "is_paired", False)),
        read1=bool(getattr(read, "is_read1", False)),
        read2=bool(getattr(read, "is_read2", False)),
    )


def alignment_segment_id_from_fields(
    query_name: object, *, paired: bool, read1: bool, read2: bool
) -> str:
    """Return a segment identity from decoded SAM name/flag fields."""
    query_name = str(query_name)
    if not paired:
        return query_name
    template = template_read_id(query_name)
    if bool(read1) == bool(read2):
        raise ValueError(f"paired alignment {query_name!r} must set exactly one of read1/read2")
    return f"{template}/{'1' if read1 else '2'}"


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


def _identity_digest(*values: object) -> str:
    """Return a delimiter-safe 128-bit digest for ordered identity components."""
    payload = bytearray()
    for value in values:
        encoded = str(value).encode("utf-8")
        payload.extend(len(encoded).to_bytes(8, "big"))
        payload.extend(encoded)
    return hashlib.sha256(payload).hexdigest()[:32]


def molecule_uid(experiment_uid: object, template_id: object) -> str:
    """Return a deterministic compact identity for one experiment/template pair.

    The hash input is length-prefixed so arbitrary read identifiers cannot create
    delimiter ambiguities. A 128-bit SHA-256 prefix keeps indexes compact while
    retaining ample collision resistance; the unhashed primary key remains
    ``(experiment_uid, read_id)``.
    """
    # Preserve the schema-v1 byte encoding so existing single-read molecule
    # identities remain stable when ``read_id`` becomes ``template_id``.
    experiment = validate_experiment_uid(experiment_uid).encode("utf-8")
    template = str(template_id).encode("utf-8")
    payload = (
        len(experiment).to_bytes(4, "big")
        + experiment
        + len(template).to_bytes(8, "big")
        + template
    )
    return hashlib.sha256(payload).hexdigest()[:32]


def segment_uid(experiment_uid: object, template_id: object, segment_id: object) -> str:
    """Return a deterministic identity for one physical segment of a molecule."""
    segment = str(segment_id).strip()
    if not segment:
        raise ValueError("segment_id must be nonempty")
    return _identity_digest(validate_experiment_uid(experiment_uid), template_id, segment)


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

"""Immutable reference-set identity and pure informative-site calculation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

from ..constants import (
    CONVERSION_BASE_SUBSTITUTIONS,
    VARIANT_INFORMATIVE_SITE_SCHEMA_VERSION,
    VARIANT_REFERENCE_SET_SCHEMA_VERSION,
)
from ..tools.sequence_alignment import AlignmentMismatch, align_sequences_with_mismatches

VARIANT_REFERENCE_ALGORITHM_VERSION = "1"
SUBSTITUTIONS_ONLY_POLICY = "disjoint_accepted_bases_substitutions_only"
EXCLUDED_INDEL_POLICY = "excluded"
_VALID_BASES = frozenset("ACGTN")
_COLUMN_SUFFIX = "_strand_FASTA_base"


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _stable_id(prefix: str, payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(encoded).hexdigest()}"


def _normalize_sequence(value: Sequence[Any] | str, *, field_name: str) -> str:
    if isinstance(value, str):
        sequence = value
    else:
        sequence = "".join(
            str(base)
            for base in value
            if base is not None and str(base).strip() not in {"", "nan", "<NA>"}
        )
    sequence = sequence.upper().replace(" ", "")
    if not sequence:
        raise ValueError(f"{field_name} must contain at least one base")
    unexpected = sorted(set(sequence).difference(_VALID_BASES))
    if unexpected:
        raise ValueError(f"{field_name} contains unsupported bases: {unexpected}")
    return sequence


def _normalize_legacy_sequence_source(
    value: Sequence[Any] | str,
    *,
    field_name: str,
) -> str:
    """Match legacy AnnData extraction by dropping padded ``N`` array entries."""
    if isinstance(value, str):
        return _normalize_sequence(value, field_name=field_name)
    unpadded = [
        base
        for base in value
        if base is not None and str(base).strip().upper() not in {"", "N", "NAN", "<NA>"}
    ]
    return _normalize_sequence(unpadded, field_name=field_name)


def normalize_legacy_variant_pair(
    values: Sequence[str | None] | None,
) -> tuple[str, str] | None:
    """Normalize the legacy two-column setting and reject partial/ambiguous shape."""
    if values is None:
        return None
    normalized = tuple(
        None if value is None or not str(value).strip() else str(value).strip() for value in values
    )
    if not normalized:
        return None
    if len(normalized) != 2:
        raise ValueError(
            "references_to_align_for_variant_annotation must contain exactly two members"
        )
    if normalized == (None, None):
        return None
    if any(value is None for value in normalized):
        raise ValueError(
            "references_to_align_for_variant_annotation must provide both members or neither"
        )
    if normalized[0] == normalized[1]:
        raise ValueError("variant reference members must be distinct")
    return str(normalized[0]), str(normalized[1])


def _source_aliases(source_id: str) -> frozenset[str]:
    aliases = {source_id}
    stem = source_id.removesuffix(_COLUMN_SUFFIX)
    aliases.add(stem)
    for strand in ("top", "bottom"):
        if stem.endswith(f"_{strand}"):
            aliases.add(stem.removesuffix(f"_{strand}"))
    return frozenset(aliases)


def _resolve_source_id(requested: str, sources: Mapping[str, Any]) -> str:
    if requested in sources:
        return requested
    matches = sorted(
        source_id for source_id in sources if requested in _source_aliases(str(source_id))
    )
    if not matches:
        raise ValueError(f"variant reference sequence source is missing: {requested!r}")
    if len(matches) > 1:
        raise ValueError(
            f"variant reference alias {requested!r} is ambiguous; candidates: {matches}"
        )
    return str(matches[0])


def _member_id(source_id: str) -> str:
    return source_id.removesuffix(_COLUMN_SUFFIX)


def _orientation(source_id: str) -> str:
    stem = _member_id(source_id)
    if stem.endswith("_bottom"):
        return "reverse_complement"
    return "forward"


@dataclass(frozen=True)
class VariantAlignmentScoring:
    """Versioned global-alignment scoring semantics."""

    match: int = 1
    mismatch: int = -1
    gap: int = -2
    ignore_n: bool = True

    def to_dict(self) -> dict[str, int | bool]:
        return {
            "match": int(self.match),
            "mismatch": int(self.mismatch),
            "gap": int(self.gap),
            "ignore_n": bool(self.ignore_n),
        }


@dataclass(frozen=True)
class VariantReferenceMember:
    """One canonical reference and the bases accepted when calling reads."""

    member_id: str
    sequence: str
    orientation: str = "forward"
    accepted_sequences: tuple[str, ...] = ()
    source_id: str | None = None
    aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        member_id = str(self.member_id).strip()
        if not member_id:
            raise ValueError("variant reference member_id must not be empty")
        sequence = _normalize_sequence(self.sequence, field_name=f"{member_id} sequence")
        orientation = str(self.orientation)
        if orientation not in {"forward", "reverse_complement"}:
            raise ValueError("variant reference orientation must be forward or reverse_complement")
        accepted = tuple(
            _normalize_sequence(value, field_name=f"{member_id} accepted sequence")
            for value in self.accepted_sequences
        )
        if not accepted:
            accepted = (sequence,)
        elif sequence not in accepted:
            accepted = (sequence, *accepted)
        if any(len(value) != len(sequence) for value in accepted):
            raise ValueError(
                f"accepted sequences for {member_id!r} must match canonical sequence length"
            )
        object.__setattr__(self, "member_id", member_id)
        object.__setattr__(self, "sequence", sequence)
        object.__setattr__(self, "orientation", orientation)
        object.__setattr__(self, "accepted_sequences", tuple(dict.fromkeys(accepted)))
        object.__setattr__(
            self,
            "aliases",
            tuple(sorted({str(alias) for alias in self.aliases if str(alias)})),
        )
        if self.source_id is not None:
            object.__setattr__(self, "source_id", str(self.source_id))

    @property
    def sequence_checksum(self) -> str:
        return _sha256_text(self.sequence)

    def accepted_bases(self, position: int) -> frozenset[str]:
        return frozenset(sequence[position] for sequence in self.accepted_sequences)

    def identity_dict(self) -> dict[str, Any]:
        """Return path- and display-label-independent scientific identity."""
        return {
            "sequence_checksum": self.sequence_checksum,
            "orientation": self.orientation,
            "accepted_sequence_checksums": [
                _sha256_text(sequence) for sequence in self.accepted_sequences
            ],
        }

    def to_dict(self) -> dict[str, Any]:
        payload = {
            **self.identity_dict(),
            "member_id": self.member_id,
            "sequence": self.sequence,
            "accepted_sequences": list(self.accepted_sequences),
            "aliases": list(self.aliases),
        }
        if self.source_id is not None:
            payload["source_id"] = self.source_id
        return payload


@dataclass(frozen=True)
class VariantReferenceSet:
    """Initial two-member reference contract with extensible list-based schema."""

    members: tuple[VariantReferenceMember, ...]
    scoring: VariantAlignmentScoring = field(default_factory=VariantAlignmentScoring)
    conversion_semantics: str = "none"
    informative_site_policy: str = SUBSTITUTIONS_ONLY_POLICY
    per_read_indel_policy: str = EXCLUDED_INDEL_POLICY
    algorithm_version: str = VARIANT_REFERENCE_ALGORITHM_VERSION
    schema_version: int = VARIANT_REFERENCE_SET_SCHEMA_VERSION

    def __post_init__(self) -> None:
        members = tuple(self.members)
        if len(members) != 2:
            raise ValueError(
                "the initial variant calling contract requires exactly two reference members"
            )
        member_ids = [member.member_id for member in members]
        if len(set(member_ids)) != len(member_ids):
            raise ValueError("variant reference member IDs must be unique")
        if not str(self.informative_site_policy).strip():
            raise ValueError("informative-site policy must not be empty")
        if self.per_read_indel_policy != EXCLUDED_INDEL_POLICY:
            raise ValueError(
                "per-read indels are explicitly excluded from the initial calling contract"
            )
        if int(self.schema_version) != VARIANT_REFERENCE_SET_SCHEMA_VERSION:
            raise ValueError("variant reference-set schema version is incompatible")
        object.__setattr__(self, "members", members)
        object.__setattr__(self, "conversion_semantics", str(self.conversion_semantics))
        object.__setattr__(self, "algorithm_version", str(self.algorithm_version))
        object.__setattr__(self, "schema_version", int(self.schema_version))

    @property
    def reference_set_id(self) -> str:
        return _stable_id("vrs", self.identity_dict())

    def member(self, member_id: str) -> VariantReferenceMember:
        try:
            return next(member for member in self.members if member.member_id == member_id)
        except StopIteration as exc:
            raise KeyError(f"unknown variant reference member: {member_id!r}") from exc

    def member_index(self, member_id: str) -> int:
        for index, member in enumerate(self.members):
            if member.member_id == member_id:
                return index
        raise KeyError(f"unknown variant reference member: {member_id!r}")

    def resolve_member_index(self, reference: str) -> int:
        """Resolve a canonical member ID or declared source alias."""
        reference = str(reference)
        matches = [
            index
            for index, member in enumerate(self.members)
            if reference == member.member_id or reference in member.aliases
        ]
        if not matches:
            raise KeyError(f"reference does not belong to variant reference set: {reference!r}")
        if len(matches) > 1:
            raise ValueError(
                f"reference alias is ambiguous in variant reference set: {reference!r}"
            )
        return matches[0]

    def identity_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "algorithm_version": self.algorithm_version,
            "members": [member.identity_dict() for member in self.members],
            "scoring": self.scoring.to_dict(),
            "conversion_semantics": self.conversion_semantics,
            "informative_site_policy": self.informative_site_policy,
            "per_read_indel_policy": self.per_read_indel_policy,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.identity_dict(),
            "reference_set_id": self.reference_set_id,
            "members": [member.to_dict() for member in self.members],
        }


@dataclass(frozen=True)
class VariantReferenceEvent:
    """One aligned substitution or explicitly non-callable indel event."""

    event_id: str
    event: str
    member_positions: tuple[int | None, int | None]
    member_bases: tuple[str | None, str | None]
    callable: bool
    exclusion_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event": self.event,
            "member_positions": list(self.member_positions),
            "member_bases": list(self.member_bases),
            "callable": self.callable,
            "exclusion_reason": self.exclusion_reason,
        }


@dataclass(frozen=True)
class InformativeVariantSite:
    """One callable substitution with disjoint accepted bases."""

    site_id: str
    member_positions: tuple[int, int]
    member_bases: tuple[str, str]
    accepted_bases: tuple[frozenset[str], frozenset[str]]
    schema_version: int = VARIANT_INFORMATIVE_SITE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.accepted_bases[0].intersection(self.accepted_bases[1]):
            raise ValueError("informative-site accepted base sets must be disjoint")
        if int(self.schema_version) != VARIANT_INFORMATIVE_SITE_SCHEMA_VERSION:
            raise ValueError("informative-site schema version is incompatible")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "site_id": self.site_id,
            "member_positions": list(self.member_positions),
            "member_bases": list(self.member_bases),
            "accepted_bases": [sorted(values) for values in self.accepted_bases],
        }


@dataclass(frozen=True)
class VariantInformativeSiteCatalog:
    """Pure reference alignment result for one immutable reference set."""

    reference_set_id: str
    aligned_sequences: tuple[str, str]
    events: tuple[VariantReferenceEvent, ...]
    informative_sites: tuple[InformativeVariantSite, ...]
    algorithm_version: str = VARIANT_REFERENCE_ALGORITHM_VERSION
    schema_version: int = VARIANT_INFORMATIVE_SITE_SCHEMA_VERSION
    # Which chemistry this catalog's acceptance assumes, e.g. "5mC:top".
    # Part of catalog_id: two strands over the same references yield different
    # informative-site sets and must not collide as one cached identity.
    conversion_semantics: str = "none"

    @property
    def catalog_id(self) -> str:
        return _stable_id(
            "vis",
            {
                "reference_set_id": self.reference_set_id,
                "algorithm_version": self.algorithm_version,
                "conversion_semantics": self.conversion_semantics,
                "events": [event.to_dict() for event in self.events],
                "informative_sites": [site.to_dict() for site in self.informative_sites],
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "algorithm_version": self.algorithm_version,
            "catalog_id": self.catalog_id,
            "reference_set_id": self.reference_set_id,
            "conversion_semantics": self.conversion_semantics,
            "aligned_sequences": list(self.aligned_sequences),
            "events": [event.to_dict() for event in self.events],
            "informative_sites": [site.to_dict() for site in self.informative_sites],
        }


def variant_reference_set_from_legacy(
    values: Sequence[str | None] | None,
    sequence_sources: Mapping[str, Sequence[Any] | str],
    *,
    converted_sequence_sources: Mapping[str, Sequence[Any] | str] | None = None,
    scoring: VariantAlignmentScoring | None = None,
    conversion_semantics: str = "none",
    informative_site_policy: str = SUBSTITUTIONS_ONLY_POLICY,
) -> VariantReferenceSet | None:
    """Resolve the legacy pair against explicit sources without path-based identity."""
    pair = normalize_legacy_variant_pair(values)
    if pair is None:
        return None
    converted_sequence_sources = dict(converted_sequence_sources or {})
    members: list[VariantReferenceMember] = []
    for requested in pair:
        source_id = _resolve_source_id(requested, sequence_sources)
        canonical = _normalize_legacy_sequence_source(
            sequence_sources[source_id],
            field_name=f"{source_id} sequence",
        )
        accepted = [canonical]
        if source_id in converted_sequence_sources:
            accepted.append(
                _normalize_legacy_sequence_source(
                    converted_sequence_sources[source_id],
                    field_name=f"{source_id} converted sequence",
                )
            )
        members.append(
            VariantReferenceMember(
                member_id=_member_id(source_id),
                sequence=canonical,
                orientation=_orientation(source_id),
                accepted_sequences=tuple(accepted),
                source_id=source_id,
                aliases=tuple(_source_aliases(source_id)),
            )
        )
    return VariantReferenceSet(
        members=tuple(members),
        scoring=scoring or VariantAlignmentScoring(),
        conversion_semantics=conversion_semantics,
        informative_site_policy=informative_site_policy,
    )


def _event_from_mismatch(
    mismatch: AlignmentMismatch,
    *,
    index: int,
) -> VariantReferenceEvent:
    callable_event = mismatch.event == "substitution"
    return VariantReferenceEvent(
        event_id=f"event-{index:06d}",
        event=mismatch.event,
        member_positions=(mismatch.seq1_pos, mismatch.seq2_pos),
        member_bases=(mismatch.seq1_base, mismatch.seq2_base),
        callable=callable_event,
        exclusion_reason=None if callable_event else "per_read_indel_calling_excluded",
    )


def conversion_substitutions_for_strand(
    modality: str | None,
    conversion_types: Sequence[str] | None,
    strand: str,
) -> tuple[tuple[str, str], ...]:
    """Base substitutions a read of ``strand`` could legitimately carry.

    Chemistry is chosen by the strand the molecule was converted on, which is
    the reference-strand assignment (`Strand` in obs, the suffix of
    `Reference_strand`) -- *not* the BAM reverse flag. Measured on the `241213`
    pilot: split by `Strand`, C/T sites miscall at 31.7% and G/A at 3.1%, a
    clean separation; split by `Read_mapping_direction` both fwd (21.6%) and rev
    (38.5%) are affected, so it does not discriminate.

    Returns an empty tuple for `direct` modality, which has no conversion
    chemistry, and for unknown strands, so callers get canonical-only
    acceptance rather than a silent guess.
    """
    normalized = str(modality or "").strip().lower()
    if normalized not in {"conversion", "deaminase"}:
        return ()
    strand_key = str(strand or "").strip().lower()
    if strand_key not in {"top", "bottom"}:
        return ()
    substitutions: list[tuple[str, str]] = []
    for modification in conversion_types or ():
        pair = CONVERSION_BASE_SUBSTITUTIONS.get((str(modification).strip(), strand_key))
        if pair is not None and pair not in substitutions:
            substitutions.append(pair)
    return tuple(substitutions)


def _accepted_with_conversion(
    member: VariantReferenceMember,
    position: int,
    substitutions: Sequence[tuple[str, str]],
) -> frozenset[str]:
    """Bases acceptable for ``member`` at ``position`` under a conversion.

    A reference C under `C->T` accepts both: the read shows C where the base
    resisted conversion and T where it did not. Widening acceptance is what
    makes a site non-disjoint and therefore excluded -- that exclusion is the
    entire mechanism, so it must be applied to the *reference* base, never to
    the observed read base.
    """
    accepted = set(member.accepted_bases(position))
    for source, destination in substitutions:
        if source in accepted:
            accepted.add(destination)
    return frozenset(accepted)


def calculate_variant_informative_sites(
    reference_set: VariantReferenceSet,
    *,
    conversion_substitutions: Sequence[tuple[str, str]] = (),
    conversion_semantics: str = "none",
) -> VariantInformativeSiteCatalog:
    """Align canonical members and return callable substitutions plus excluded indels.

    ``conversion_substitutions`` widens each member's accepted bases by the
    chemistry a read of one strand could carry, so a site the chemistry makes
    ambiguous is excluded rather than miscalled. The reference set itself is
    unchanged, which keeps ``reference_set_id`` stable across strands -- task
    planning groups on that id, and one catalog per strand must not look like
    two different reference sets.
    """
    if reference_set.informative_site_policy != SUBSTITUTIONS_ONLY_POLICY:
        raise ValueError(
            "informative-site calculation does not support policy "
            f"{reference_set.informative_site_policy!r}"
        )
    first, second = reference_set.members
    aligned_first, aligned_second, mismatches = align_sequences_with_mismatches(
        first.sequence,
        second.sequence,
        match_score=reference_set.scoring.match,
        mismatch_score=reference_set.scoring.mismatch,
        gap_score=reference_set.scoring.gap,
        ignore_n=reference_set.scoring.ignore_n,
    )
    events: list[VariantReferenceEvent] = []
    informative_sites: list[InformativeVariantSite] = []
    for index, mismatch in enumerate(mismatches):
        event = _event_from_mismatch(mismatch, index=index)
        if event.callable:
            assert mismatch.seq1_pos is not None
            assert mismatch.seq2_pos is not None
            assert mismatch.seq1_base is not None
            assert mismatch.seq2_base is not None
            accepted = (
                _accepted_with_conversion(first, mismatch.seq1_pos, conversion_substitutions),
                _accepted_with_conversion(second, mismatch.seq2_pos, conversion_substitutions),
            )
            if accepted[0].intersection(accepted[1]):
                event = replace(
                    event,
                    callable=False,
                    exclusion_reason="accepted_base_sets_overlap",
                )
            else:
                informative_sites.append(
                    InformativeVariantSite(
                        site_id=f"site-{len(informative_sites):06d}",
                        member_positions=(mismatch.seq1_pos, mismatch.seq2_pos),
                        member_bases=(mismatch.seq1_base, mismatch.seq2_base),
                        accepted_bases=accepted,
                    )
                )
        events.append(event)
    return VariantInformativeSiteCatalog(
        reference_set_id=reference_set.reference_set_id,
        aligned_sequences=(aligned_first, aligned_second),
        events=tuple(events),
        informative_sites=tuple(informative_sites),
        conversion_semantics=str(conversion_semantics),
    )

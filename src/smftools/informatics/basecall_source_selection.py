"""Pick which representation of a run's reads to ingest.

A run directory usually holds POD5 signal, a `fastq_pass` tree, and sometimes
BAMs from whichever model was run. Until `BCS-01` that directory could not be
used as ``input_data_path`` at all -- discovery found more than one recognized
kind and config loading refused -- so the practice was to point at one
subdirectory by hand.

What a user wants to express is a *model*, not a path. A source satisfies the
configuration when all three hold (`BCS-03`):

1. **The model matches.** A bare selector (``hac``) accepts any version of that
   family, newest winning; a fully qualified name must match exactly. The Dorado
   version never participates -- see :mod:`basecall_provenance`.
2. **Its capabilities suffice.** Model identity alone is not enough: a canonical
   FASTQ carries no MM/ML, which is fine for ``deaminase`` and ``conversion`` and
   disqualifying for ``direct``.
3. **Its bytes are reachable.** A source on a detached volume satisfies nothing.
   That state exists only because of `PSR-01`, and consulting it here is what
   stops selection choosing a path it cannot read.

Where several sources qualify, BAM beats FASTQ -- tags, read groups and any
existing alignment survive -- and `fastq_pass` beats everything under
`fastq_fail`, which is never selectable (`BCS-04`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

from smftools.logging_utils import get_logger

from .basecall_provenance import (
    BasecallProvenance,
    is_bare_selector,
    model_family,
    model_version,
    read_provenance,
)

logger = get_logger(__name__)

#: Preference among qualifying kinds. BAM preserves tags, read groups and any
#: alignment that already exists; FASTQ is sequence-only.
_KIND_PREFERENCE = {"bam": 0, "cram": 1, "fastq": 2}

#: Modalities that read base conversions from the sequence itself and therefore
#: need no modification tags.
_SEQUENCE_ONLY_MODALITIES = {"deaminase", "conversion"}

#: Path components whose contents are never selectable: reads the instrument
#: already judged as failing quality.
_EXCLUDED_COMPONENTS = {"fastq_fail", "fail", "pod5_skip"}


@dataclass(frozen=True)
class CandidateSource:
    """One discovered representation of a run's reads, with why it did or did not qualify."""

    path: Path
    kind: str
    provenance: Optional[BasecallProvenance] = None
    qualifies: bool = False
    reason: str = ""

    def describe(self) -> str:
        """One clause naming this source and its verdict."""
        model = self.provenance.model if self.provenance and self.provenance.model else "unknown"
        verdict = "qualifies" if self.qualifies else self.reason
        return f"{self.path.name} ({self.kind}, model={model}): {verdict}"


@dataclass(frozen=True)
class SourceSelection:
    """The chosen source, or the decision to basecall, with every candidate's verdict."""

    kind: Optional[str] = None
    paths: tuple[Path, ...] = ()
    provenance: Optional[BasecallProvenance] = None
    must_basecall: bool = False
    candidates: tuple[CandidateSource, ...] = field(default_factory=tuple)

    @property
    def resolved(self) -> bool:
        """Whether reads were found; False means basecalling is required."""
        return bool(self.paths)


def is_excluded(path: Path) -> bool:
    """Whether a path lies under a directory whose reads are never selectable."""
    return any(part.lower() in _EXCLUDED_COMPONENTS for part in Path(path).parts)


def model_matches(selector: str, model: Optional[str]) -> bool:
    """Whether a source's model satisfies the configured selector.

    Args:
        selector: The configured model, bare (``hac``) or fully qualified.
        model: The model a source records, or None when it records none.

    Returns:
        bool: True when the source satisfies the request.
    """
    if not model:
        return False
    selector = str(selector).strip()
    if is_bare_selector(selector):
        return model_family(model) == selector
    return str(model) == selector


def capability_suffices(provenance: Optional[BasecallProvenance], kind: str, modality: str) -> bool:
    """Whether a source carries what the modality needs.

    ``direct`` reads modification probabilities from MM/ML tags, so a
    sequence-only source cannot serve it however well its model matches.
    """
    if str(modality).strip().lower() in _SEQUENCE_ONLY_MODALITIES:
        return True
    if kind == "fastq":
        return False
    return bool(provenance and provenance.carries_modifications)


def evaluate_candidate(
    path: Path,
    kind: str,
    *,
    model_selector: str,
    modality: str,
    reachable: bool = True,
) -> CandidateSource:
    """Judge one discovered source against the three selection rules."""
    path = Path(path)
    if is_excluded(path):
        return CandidateSource(path=path, kind=kind, reason="excluded directory")
    if not reachable:
        return CandidateSource(path=path, kind=kind, reason="on a detached volume")
    provenance = read_provenance(path, kind=kind)
    if provenance is None or not provenance.model:
        return CandidateSource(
            path=path, kind=kind, provenance=provenance, reason="records no basecall model"
        )
    if not model_matches(model_selector, provenance.model):
        return CandidateSource(
            path=path,
            kind=kind,
            provenance=provenance,
            reason=f"model {provenance.model} does not satisfy {model_selector!r}",
        )
    if not capability_suffices(provenance, kind, modality):
        return CandidateSource(
            path=path,
            kind=kind,
            provenance=provenance,
            reason=f"carries no modification tags, which modality {modality!r} requires",
        )
    return CandidateSource(path=path, kind=kind, provenance=provenance, qualifies=True)


def select_read_source(
    discovered: dict[str, Sequence[Path]],
    *,
    model_selector: str,
    modality: str,
    reachable: bool = True,
) -> SourceSelection:
    """Choose the representation to ingest, or decide that basecalling is required.

    Args:
        discovered: A :func:`~smftools.config.discover_input_files.discover_input_files`
            result, keyed ``<kind>_paths`` as that function returns it.
        model_selector: The configured basecalling model, bare or qualified.
        modality: The experiment's SMF modality.
        reachable: Whether the discovered paths can currently be read.

    Returns:
        SourceSelection: The chosen kind and paths, or ``must_basecall`` when
        nothing qualified and signal is present. Every candidate's verdict is
        carried so a refusal can say what was found and why each was rejected.
    """
    candidates: list[CandidateSource] = []
    for kind in sorted(_KIND_PREFERENCE, key=_KIND_PREFERENCE.get):
        for path in discovered.get(f"{kind}_paths") or ():
            candidates.append(
                evaluate_candidate(
                    path,
                    kind,
                    model_selector=model_selector,
                    modality=modality,
                    reachable=reachable,
                )
            )

    qualifying = [candidate for candidate in candidates if candidate.qualifies]
    if qualifying:
        best_kind = min(qualifying, key=lambda c: _KIND_PREFERENCE[c.kind]).kind
        chosen = [c for c in qualifying if c.kind == best_kind]
        # Newest model version wins within a kind, which is what makes a bare
        # selector mean "any version of this family, preferring the newest".
        newest = max(model_version(c.provenance.model) for c in chosen)
        chosen = [c for c in chosen if model_version(c.provenance.model) == newest]
        logger.info(
            "Selected %d %s source(s) for model %r (%s)",
            len(chosen),
            best_kind,
            model_selector,
            chosen[0].provenance.model,
        )
        return SourceSelection(
            kind=best_kind,
            paths=tuple(sorted(c.path for c in chosen)),
            provenance=chosen[0].provenance,
            candidates=tuple(candidates),
        )

    has_signal = bool(discovered.get("pod5_paths") or discovered.get("fast5_paths"))
    return SourceSelection(must_basecall=has_signal, candidates=tuple(candidates))


def describe_rejection(selection: SourceSelection, *, model_selector: str) -> str:
    """Explain why nothing qualified, naming each candidate and its verdict."""
    if not selection.candidates:
        return f"no read sources were discovered for model {model_selector!r}"
    verdicts = "; ".join(candidate.describe() for candidate in selection.candidates)
    return (
        f"no discovered source satisfies model {model_selector!r}, and no signal is "
        f"present to basecall from. Candidates: {verdicts}"
    )

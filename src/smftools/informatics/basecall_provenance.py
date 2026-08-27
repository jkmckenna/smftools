"""Read which basecalling model produced a set of reads.

An experiment's data directory usually holds several representations of the same
reads -- POD5 signal, a `fastq_pass` tree, sometimes BAMs from whichever model
was run -- and choosing between them means asking each one what produced it.
Three formats record that three different ways (`BCS-02`):

===========  ==========================================================
FASTQ        ``basecall_model_version_id=`` in the read header comment
BAM/CRAM     ``basecall_model=`` inside a read group's ``DS`` field
POD5         no derivative exists yet; the model is whatever is configured
===========  ==========================================================

One reader serves both selection and, later, the basecall stage's own outputs, so
a basecall smftools produced and one the instrument produced are interrogated by
identical code. Two readers would eventually disagree, and a disagreement here
silently changes which source an experiment ingests.

**The Dorado version is recorded and never gating.** A basecaller release that
leaves the model identity unchanged would otherwise force a re-basecall of every
archived run on each instrument software update, for reads the model itself says
are equivalent.
"""

from __future__ import annotations

import gzip
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from smftools.logging_utils import get_logger

logger = get_logger(__name__)

#: ``basecall_model_version_id=dna_r10.4.1_e8.2_400bps_hac@v5.0.0``
_FASTQ_MODEL = re.compile(r"basecall_model_version_id=(\S+)")
#: ``basecall_model=dna_r10.4.1_e8.2_400bps_hac@v5.0.0`` inside a read group ``DS``.
_BAM_MODEL = re.compile(r"basecall_model=(\S+)")
#: The trailing ``@v5.0.0`` of a fully qualified model name.
_MODEL_VERSION = re.compile(r"@v([0-9]+(?:\.[0-9]+)*)$")
#: The speed/accuracy family token: ``fast``, ``hac``, ``sup``.
_MODEL_FAMILY = re.compile(r"(?:^|_)(fast|hac|sup)(?:@|$)")


@dataclass(frozen=True)
class BasecallProvenance:
    """What produced one set of reads, as far as the reads themselves record it."""

    model: Optional[str] = None
    dorado_version: Optional[str] = None
    carries_modifications: bool = False

    @property
    def family(self) -> Optional[str]:
        """The speed/accuracy family (``fast``/``hac``/``sup``), if the name has one."""
        return model_family(self.model) if self.model else None

    @property
    def version(self) -> tuple[int, ...]:
        """The model version as a comparable tuple, empty when unversioned."""
        return model_version(self.model) if self.model else ()


def model_family(model_name: str) -> Optional[str]:
    """Return the speed/accuracy family token of a model name, or None."""
    match = _MODEL_FAMILY.search(str(model_name))
    return match.group(1) if match else None


def model_version(model_name: str) -> tuple[int, ...]:
    """Return a model name's trailing version as a comparable tuple.

    Mirrors ``dorado_model._model_version_key`` so the two orderings cannot drift.
    """
    match = _MODEL_VERSION.search(str(model_name))
    return () if match is None else tuple(int(part) for part in match.group(1).split("."))


def is_bare_selector(selector: str) -> bool:
    """Whether a configured model is a short family name rather than a full name.

    ``hac`` is bare and accepts any version of that family; a fully qualified
    ``dna_r10.4.1_e8.2_400bps_hac@v5.0.0`` demands an exact match.
    """
    text = str(selector).strip()
    return text in {"fast", "hac", "sup"}


def _open_text(path: Path):
    if str(path).lower().endswith((".gz", ".bgz")):
        return gzip.open(path, "rt", errors="replace")
    return open(path, "rt", errors="replace")


def read_fastq_provenance(path: str | Path) -> Optional[BasecallProvenance]:
    """Read the basecall model from a FASTQ's first read header.

    MinKNOW and Dorado both stamp ``basecall_model_version_id=`` into the header
    comment. Only the first record is read: every read in a file comes from one
    basecalling pass, so scanning further costs IO for an answer already known.

    Args:
        path: A FASTQ file, optionally gzip-compressed.

    Returns:
        BasecallProvenance or None: None when the file is unreadable or its
        header records no model, which is not an error -- plenty of FASTQ has no
        provenance and simply cannot satisfy a model-specific request.
    """
    path = Path(path)
    try:
        with _open_text(path) as handle:
            header = handle.readline()
    except OSError as exc:
        logger.debug("could not read FASTQ provenance from %s: %s", path, exc)
        return None
    if not header.startswith("@"):
        return None
    match = _FASTQ_MODEL.search(header)
    if match is None:
        return None
    # FASTQ is sequence-only: it cannot carry MM/ML however it was produced.
    return BasecallProvenance(model=match.group(1), carries_modifications=False)


def read_bam_provenance(path: str | Path) -> Optional[BasecallProvenance]:
    """Read the basecall model and modification capability from a BAM/CRAM header.

    The model lives in each read group's ``DS`` description, the Dorado version in
    the ``@PG`` program record. Modification capability is taken from the header's
    declared ``MM``/``ML`` support where present, falling back to the first read.

    Args:
        path: A BAM or CRAM file.

    Returns:
        BasecallProvenance or None: None when the file cannot be opened or
        declares no basecall model.
    """
    path = Path(path)
    try:
        import pysam
    except ImportError:  # pragma: no cover - pysam is an optional extra
        logger.debug("pysam unavailable; cannot read BAM provenance from %s", path)
        return None
    try:
        handle = pysam.AlignmentFile(str(path), "rb", check_sq=False)
    except (OSError, ValueError) as exc:
        logger.debug("could not open %s for provenance: %s", path, exc)
        return None
    with handle:
        header = handle.header.to_dict()
        models = set()
        for group in header.get("RG", []):
            match = _BAM_MODEL.search(str(group.get("DS", "")))
            if match:
                models.add(match.group(1))
        dorado_version = None
        for program in header.get("PG", []):
            if str(program.get("PN", program.get("ID", ""))).lower().startswith("dorado"):
                dorado_version = str(program.get("VN")) if program.get("VN") else None
                break
        if not models:
            return None
        if len(models) > 1:
            # Reads from several models in one file cannot answer "which model
            # produced this" with a single name, and guessing one would silently
            # mix chemistries.
            logger.warning(
                "%s declares %d basecall models (%s); treating its provenance as unknown",
                path,
                len(models),
                ", ".join(sorted(models)),
            )
            return None
        carries_modifications = _bam_carries_modifications(handle)
    return BasecallProvenance(
        model=models.pop(),
        dorado_version=dorado_version,
        carries_modifications=carries_modifications,
    )


def _bam_carries_modifications(handle) -> bool:
    """Whether the first readable record carries MM/ML modification tags."""
    try:
        for record in handle.head(1):
            return record.has_tag("MM") or record.has_tag("Mm")
    except (OSError, ValueError, StopIteration):
        return False
    return False


def read_provenance(path: str | Path, *, kind: str) -> Optional[BasecallProvenance]:
    """Read provenance from one source, dispatching on its discovered kind.

    Args:
        path: The source file.
        kind: A kind from ``discover_input_files`` (``fastq``, ``bam``, ``cram``,
            ``pod5``, ``fast5``, ...).

    Returns:
        BasecallProvenance or None: None for signal inputs, which have no
        derivative to interrogate, and for sources that record no model.
    """
    if kind == "fastq":
        return read_fastq_provenance(path)
    if kind in {"bam", "cram"}:
        return read_bam_provenance(path)
    return None

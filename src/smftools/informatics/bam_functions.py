from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
import time
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from math import ceil
from dataclasses import dataclass, field
from itertools import zip_longest
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
from tqdm import tqdm

from smftools.constants import MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT
from smftools.logging_utils import get_logger
from smftools.optional_imports import require

from ..readwrite import date_string, time_string

if TYPE_CHECKING:
    import pysam as pysam_types

try:
    import pysam
except Exception:
    pysam = None  # type: ignore

logger = get_logger(__name__)

_PROGRESS_RE = re.compile(r"Output records written:\s*(\d+)")
_EMPTY_RE = re.compile(r"^\s*$")

# Global cache for dorado version
_DORADO_VERSION_CACHE: Optional[Tuple[int, int, int]] = None


def _get_dorado_version() -> Optional[Tuple[int, int, int]]:
    """Get installed dorado version as (major, minor, patch) tuple, or None if not found.

    Returns
    -------
    tuple of (int, int, int) or None
        Version tuple like (1, 3, 1) for dorado 1.3.1, or None if dorado not found.
    """
    global _DORADO_VERSION_CACHE
    if _DORADO_VERSION_CACHE is not None:
        return _DORADO_VERSION_CACHE

    try:
        result = subprocess.run(["dorado", "--version"], capture_output=True, text=True, timeout=5)
        version_str = result.stdout.strip() or result.stderr.strip()
        # Parse "1.3.1" or "0.9.0+9dc15a85"
        match = re.match(r"(\d+)\.(\d+)\.(\d+)", version_str)
        if match:
            _DORADO_VERSION_CACHE = (int(match.group(1)), int(match.group(2)), int(match.group(3)))
            logger.info(
                f"Detected dorado version: {'.'.join(str(v) for v in _DORADO_VERSION_CACHE)}"
            )
            return _DORADO_VERSION_CACHE
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        logger.debug(f"Could not determine dorado version: {e}")

    return None


def _bam_has_barcode_info_tags(bam_path: str | Path, sample_size: int = 100) -> dict:
    """Check whether classified reads in a BAM have dorado bi/bv barcode scoring tags.

    Samples up to sample_size reads that carry a BC tag and returns a dict with boolean flags.

    Parameters
    ----------
    bam_path : str or Path
        Path to BAM file to check.
    sample_size : int, default 100
        Maximum number of BC-tagged reads to sample.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'has_bc': at least one read has a BC tag
        - 'has_bi': at least one classified read also has a bi tag
        - 'has_bv': at least one classified read also has a bv tag
    """
    pysam_mod = _require_pysam()
    has_bc = False
    has_bi = False
    has_bv = False
    checked = 0

    try:
        with pysam_mod.AlignmentFile(str(bam_path), "rb", check_sq=False) as bam:
            for read in bam:
                if read.has_tag("BC"):
                    has_bc = True
                    if read.has_tag("bi"):
                        has_bi = True
                    if read.has_tag("bv"):
                        has_bv = True
                    checked += 1
                    if checked >= sample_size:
                        break
    except Exception as e:
        logger.warning(f"Error checking BAM tags in {bam_path}: {e}")

    return {"has_bc": has_bc, "has_bi": has_bi, "has_bv": has_bv}


# ---------------------------------------------------------------------------
# Flanking-sequence configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class FlankingConfig:
    """Flanking sequences on each side of a barcode or UMI."""

    adapter_side: Optional[str] = None
    amplicon_side: Optional[str] = None
    adapter_pad: int = 5
    amplicon_pad: int = 5


@dataclass
class PerEndFlankingConfig:
    """Per-reference-end flanking configuration."""

    left_ref_end: Optional[FlankingConfig] = None
    right_ref_end: Optional[FlankingConfig] = None
    same_orientation: bool = False


@dataclass
class BarcodeKitConfig:
    """Full barcode kit configuration loaded from YAML."""

    name: Optional[str] = None
    barcodes: Dict[str, str] = field(default_factory=dict)
    barcode_length: int = 0
    flanking: Optional[PerEndFlankingConfig] = None
    barcode_ends: str = "both"
    barcode_max_edit_distance: int = 3
    barcode_composite_max_edits: int = 4
    barcode_min_separation: Optional[int] = None
    barcode_amplicon_gap_tolerance: int = 5


@dataclass
class UMIKitConfig:
    """UMI configuration loaded from YAML."""

    flanking: Optional[PerEndFlankingConfig] = None
    length: int = 0
    umi_ends: str = "both"
    umi_flank_mode: str = "adapter_only"
    adapter_max_edits: int = 0
    amplicon_max_edits: int = 0
    same_orientation: bool = False


_BAM_FLAG_BITS: Tuple[Tuple[int, str], ...] = (
    (0x1, "paired"),
    (0x2, "proper_pair"),
    (0x4, "unmapped"),
    (0x8, "mate_unmapped"),
    (0x10, "reverse"),
    (0x20, "mate_reverse"),
    (0x40, "read1"),
    (0x80, "read2"),
    (0x100, "secondary"),
    (0x200, "qc_fail"),
    (0x400, "duplicate"),
    (0x800, "supplementary"),
)


def _require_pysam() -> "pysam_types":
    """Return the pysam module or raise if unavailable."""
    if pysam is not None:
        return pysam
    return require("pysam", extra="pysam", purpose="samtools-compatible Python backend")


def _resolve_samtools_backend(backend: str | None) -> str:
    """Resolve backend choice for samtools-compatible operations.

    Args:
        backend: One of {"auto", "python", "cli"} (case-insensitive).

    Returns:
        Resolved backend string ("python" or "cli").
    """
    choice = (backend or "auto").strip().lower()
    if choice not in {"auto", "python", "cli"}:
        raise ValueError("samtools_backend must be one of: auto, python, cli")

    have_pysam = pysam is not None
    have_samtools = shutil.which("samtools") is not None

    if choice == "python":
        if not have_pysam:
            raise RuntimeError("samtools_backend=python requires pysam to be installed.")
        return "python"
    if choice == "cli":
        if not have_samtools:
            raise RuntimeError("samtools_backend=cli requires samtools in PATH.")
        return "cli"

    if have_samtools:
        return "cli"
    if have_pysam:
        return "python"
    raise RuntimeError("Neither pysam nor samtools is available in PATH.")


def _has_bam_index(bam_path: Path) -> bool:
    """Return True if the BAM index exists alongside the BAM."""
    return (
        bam_path.with_suffix(bam_path.suffix + ".bai").exists()
        or Path(str(bam_path) + ".bai").exists()
    )


def _ensure_bam_index(bam_path: Path, backend: str) -> None:
    """Ensure a BAM index exists, creating one if needed."""
    if _has_bam_index(bam_path):
        return
    if backend == "python":
        _index_bam_with_pysam(bam_path)
    else:
        _index_bam_with_samtools(bam_path)


def _parse_idxstats_output(output: str) -> Tuple[int, int, Dict[str, Tuple[int, float]]]:
    """Parse samtools idxstats output into counts and proportions."""
    aligned_reads_count = 0
    unaligned_reads_count = 0
    record_counts: Dict[str, int] = {}
    for line in output.splitlines():
        if not line.strip():
            continue
        ref, _length, mapped, unmapped = line.split("\t")[:4]
        if ref == "*":
            unaligned_reads_count += int(unmapped)
            continue
        mapped_count = int(mapped)
        aligned_reads_count += mapped_count
        record_counts[ref] = mapped_count

    proportions: Dict[str, Tuple[int, float]] = {}
    for ref, count in record_counts.items():
        proportion = count / aligned_reads_count if aligned_reads_count else 0.0
        proportions[ref] = (count, proportion)

    return aligned_reads_count, unaligned_reads_count, proportions




_COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def _reverse_complement(seq: str) -> str:
    """Return the reverse complement of a DNA sequence."""
    return seq.translate(_COMPLEMENT)[::-1]


def _find_flanking_sequence(
    seq: str,
    flanking_seq: str,
    search_window: int,
    search_from_start: bool,
    matcher: str = "edlib",
    max_edits: int = 2,
) -> Optional[Tuple[int, int]]:
    """Find a flanking sequence within a search window of a read end.

    Returns
    -------
    Optional[Tuple[int, int]]
        (start, end) positions of the match, or None if not found.
    """
    if not seq or not flanking_seq:
        return None

    seq_len = len(seq)
    flanking_upper = flanking_seq.upper()

    if matcher == "exact":
        matches = [(m.start(), m.end()) for m in re.finditer(re.escape(flanking_upper), seq)]
    else:
        edlib = require("edlib", extra="umi", purpose="fuzzy flanking sequence matching")
        result = edlib.align(flanking_upper, seq, mode="HW", task="locations", k=max(0, max_edits))
        locations = result.get("locations", []) if isinstance(result, dict) else []
        matches = []
        for loc in locations:
            if not isinstance(loc, (list, tuple)) or len(loc) != 2:
                continue
            start_i, end_i = int(loc[0]), int(loc[1])
            if start_i < 0 or end_i < start_i:
                continue
            matches.append((start_i, end_i + 1))

    best: Optional[Tuple[int, int]] = None
    best_distance: Optional[int] = None
    for start, end in matches:
        distance = start if search_from_start else (seq_len - end)
        if distance > search_window:
            continue
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best = (start, end)

    return best


def _build_composite_query(
    adapter_side: str,
    amplicon_side: str,
    target_length: int,
) -> Tuple[str, int, int]:
    """Build composite query ``adapter + N*length + amplicon`` and mask indices."""
    mask = "N" * target_length
    composite = adapter_side + mask + amplicon_side
    mask_start = len(adapter_side)
    mask_end = mask_start + target_length
    return composite, mask_start, mask_end


def _extract_mask_location(
    cigar: str,
    query_mask_start: int,
    query_mask_end: int,
) -> Optional[Tuple[int, int]]:
    """Map query mask positions to target coordinates using edlib CIGAR."""
    if not cigar:
        return None

    query_pos = 0
    target_pos = 0
    target_start = None
    target_end = None

    num = ""
    for ch in cigar:
        if ch.isdigit():
            num += ch
            continue
        if not num:
            return None
        length = int(num)
        num = ""

        if ch in {"=", "M", "X"}:
            for _ in range(length):
                if query_pos == query_mask_start:
                    target_start = target_pos
                if query_pos == query_mask_end:
                    target_end = target_pos
                    return target_start, target_end
                query_pos += 1
                target_pos += 1
        elif ch == "I":
            for _ in range(length):
                if query_pos == query_mask_start:
                    target_start = target_pos
                if query_pos == query_mask_end:
                    target_end = target_pos
                    return target_start, target_end
                query_pos += 1
        elif ch == "D":
            for _ in range(length):
                if query_pos == query_mask_start:
                    target_start = target_pos
                if query_pos == query_mask_end:
                    target_end = target_pos
                    return target_start, target_end
                target_pos += 1
        else:
            return None

    if query_pos == query_mask_end:
        target_end = target_pos
    if target_start is not None and target_end is not None:
        return target_start, target_end
    return None


def _composite_extract(
    read_sequence: str,
    adapter_side: str,
    amplicon_side: str,
    target_length: int,
    search_window: int,
    search_from_start: bool,
    max_edits: int,
    adapter_pad: int = 5,
    amplicon_pad: int = 5,
) -> Optional[Tuple[str, str, int, int]]:
    """Extract barcode using composite alignment against read-end window.

    Returns
    -------
    Optional[Tuple[str, str, int, int]]
        (barcode_seq, padded_region, start, end) or None if extraction fails.
        barcode_seq may be variable length due to indels in alignment.
        padded_region includes flanking context for padded scoring.
    """
    if not read_sequence:
        return None

    edlib = require("edlib", extra="umi", purpose="composite barcode extraction")
    seq = read_sequence.upper()
    seq_len = len(seq)

    composite, mask_start, mask_end = _build_composite_query(
        adapter_side.upper(),
        amplicon_side.upper(),
        target_length,
    )

    window_len = min(seq_len, search_window + len(composite))
    if search_from_start:
        window_start = 0
        window = seq[:window_len]
    else:
        window_start = max(0, seq_len - window_len)
        window = seq[window_start:]

    if not window:
        return None

    result = edlib.align(
        composite,
        window,
        mode="HW",
        task="path",
        k=max_edits,
        additionalEqualities=[("N", "A"), ("N", "C"), ("N", "G"), ("N", "T")],
    )
    if result.get("editDistance", -1) == -1:
        return None

    locations = result.get("locations") or []
    if not locations:
        return None
    cigar = result.get("cigar")
    if cigar is None:
        return None

    mask_loc = _extract_mask_location(cigar, mask_start, mask_end)
    if mask_loc is None:
        return None

    target_mask_start, target_mask_end = mask_loc
    aln_start = locations[0][0] if isinstance(locations[0], (list, tuple)) else locations[0]
    bc_start = window_start + aln_start + target_mask_start
    bc_end = window_start + aln_start + target_mask_end

    if bc_start < 0 or bc_end > seq_len or bc_end <= bc_start:
        return None

    # Extract core barcode (may be variable length due to indels)
    barcode_seq = seq[bc_start:bc_end]

    # Extract padded region with flanking context for padded scoring
    pad_start = max(0, bc_start - adapter_pad)
    pad_end = min(seq_len, bc_end + amplicon_pad)
    padded_region = seq[pad_start:pad_end]

    return barcode_seq, padded_region, bc_start, bc_end


def _extract_barcode_with_flanking(
    read_sequence: str,
    target_length: int,
    search_window: int,
    search_from_start: bool,
    flanking: FlankingConfig,
    adapter_matcher: str = "edlib",
    composite_max_edits: int = 4,
) -> Tuple[Optional[str], Optional[int], Optional[int], Optional[str]]:
    """Extract a target sequence (barcode or UMI) using flanking sequence detection.

    Behavior:
    - If both adapter_side and amplicon_side are provided, use composite alignment
      of ``adapter + N*target_length + amplicon`` within the end window.
    - If only one flank is provided, fall back to single-flank anchored extraction.

    Parameters
    ----------
    read_sequence : str
        The read sequence to search within.
    target_length : int
        Expected length of the target (barcode / UMI).
    search_window : int
        Maximum distance from the targeted read end to search for a flanking match.
    search_from_start : bool
        If True, search from the start of the read (left reference end on forward
        strand); otherwise search from the end.
    flanking : FlankingConfig
        Flanking sequences for adapter_side and/or amplicon_side.
    adapter_matcher : str
        Matching algorithm for flanking detection (``"exact"`` or ``"edlib"``).
    composite_max_edits : int
        Max edit distance allowed for composite or single-flank matching.

    Returns
    -------
    Tuple[Optional[str], Optional[int], Optional[int], Optional[str]]
        ``(extracted_sequence, start_pos, end_pos, padded_region)`` or
        ``(None, None, None, None)`` if extraction fails.
        padded_region is only available for composite extraction.
    """
    if not read_sequence:
        return None, None, None, None

    seq = read_sequence.upper()
    seq_len = len(seq)

    tgt_start: Optional[int] = None
    tgt_end: Optional[int] = None

    has_adapter = bool(flanking.adapter_side)
    has_amplicon = bool(flanking.amplicon_side)

    # Composite alignment when both flanks are available
    if has_adapter and has_amplicon:
        comp = _composite_extract(
            read_sequence=seq,
            adapter_side=flanking.adapter_side,
            amplicon_side=flanking.amplicon_side,
            target_length=target_length,
            search_window=search_window,
            search_from_start=search_from_start,
            max_edits=composite_max_edits,
            adapter_pad=flanking.adapter_pad,
            amplicon_pad=flanking.amplicon_pad,
        )
        if comp is not None:
            extracted, padded_region, tgt_start, tgt_end = comp
            return extracted, tgt_start, tgt_end, padded_region

    # Single-flank fallback
    if has_adapter:
        hit = _find_flanking_sequence(
            seq,
            flanking.adapter_side,
            search_window,
            search_from_start,
            matcher=adapter_matcher,
            max_edits=composite_max_edits,
        )
        if hit is not None:
            adapter_start, adapter_end = hit
            if search_from_start:
                tgt_start, tgt_end = adapter_end, adapter_end + target_length
            else:
                tgt_start, tgt_end = adapter_start - target_length, adapter_start
    elif has_amplicon:
        hit = _find_flanking_sequence(
            seq,
            flanking.amplicon_side,
            search_window,
            search_from_start,
            matcher=adapter_matcher,
            max_edits=composite_max_edits,
        )
        if hit is not None:
            amplicon_start, amplicon_end = hit
            if search_from_start:
                tgt_start, tgt_end = amplicon_start - target_length, amplicon_start
            else:
                tgt_start, tgt_end = amplicon_end, amplicon_end + target_length
    else:
        return None, None, None, None

    # Bounds check
    if tgt_start is None or tgt_end is None:
        return None, None, None, None
    if tgt_start < 0 or tgt_end > seq_len:
        return None, None, None, None

    extracted = seq[tgt_start:tgt_end]
    if len(extracted) != target_length:
        return None, None, None, None

    return extracted, tgt_start, tgt_end, None


def _extract_sequence_with_flanking(
    read_sequence: str,
    target_length: int,
    search_window: int,
    search_from_start: bool,
    flanking: FlankingConfig,
    flank_mode: str = "adapter_only",
    adapter_matcher: str = "edlib",
    adapter_max_edits: int = 2,
    amplicon_max_edits: int = 2,
    same_orientation: bool = False,
) -> Tuple[Optional[str], Optional[int], Optional[int]]:
    """Extract a target sequence (barcode or UMI) using flanking sequence detection."""
    if not read_sequence:
        return None, None, None

    seq = read_sequence.upper()
    seq_len = len(seq)

    if flank_mode == "adapter_only":
        if not flanking.adapter_side:
            return None, None, None
        hit = _find_flanking_sequence(
            seq,
            flanking.adapter_side,
            search_window,
            search_from_start,
            matcher=adapter_matcher,
            max_edits=adapter_max_edits,
        )
        if hit is None:
            return None, None, None
        adapter_start, adapter_end = hit
        if search_from_start or same_orientation:
            tgt_start, tgt_end = adapter_end, adapter_end + target_length
        else:
            tgt_start, tgt_end = adapter_start - target_length, adapter_start

    elif flank_mode == "amplicon_only":
        if not flanking.amplicon_side:
            return None, None, None
        hit = _find_flanking_sequence(
            seq,
            flanking.amplicon_side,
            search_window,
            search_from_start,
            matcher=adapter_matcher,
            max_edits=amplicon_max_edits,
        )
        if hit is None:
            return None, None, None
        amplicon_start, amplicon_end = hit
        if search_from_start or same_orientation:
            tgt_start, tgt_end = amplicon_start - target_length, amplicon_start
        else:
            tgt_start, tgt_end = amplicon_end, amplicon_end + target_length

    elif flank_mode == "both":
        if not flanking.adapter_side:
            return None, None, None
        hit = _find_flanking_sequence(
            seq,
            flanking.adapter_side,
            search_window,
            search_from_start,
            matcher=adapter_matcher,
            max_edits=adapter_max_edits,
        )
        if hit is None:
            return None, None, None
        adapter_start, adapter_end = hit
        if search_from_start or same_orientation:
            tgt_start, tgt_end = adapter_end, adapter_end + target_length
        else:
            tgt_start, tgt_end = adapter_start - target_length, adapter_start

        if flanking.amplicon_side:
            if search_from_start or same_orientation:
                region_start = tgt_end
                region_end = min(
                    tgt_end + len(flanking.amplicon_side) + amplicon_max_edits + 1,
                    seq_len,
                )
            else:
                region_end = tgt_start
                region_start = max(
                    tgt_start - len(flanking.amplicon_side) - amplicon_max_edits - 1,
                    0,
                )
            region = seq[region_start:region_end]
            if not region:
                return None, None, None
            amp_hit = _find_flanking_sequence(
                region,
                flanking.amplicon_side,
                search_window=len(region),
                search_from_start=True,
                matcher=adapter_matcher,
                max_edits=amplicon_max_edits,
            )
            if amp_hit is None:
                return None, None, None

    elif flank_mode == "either":
        # Try both, then amplicon_only, then adapter_only
        result = _extract_sequence_with_flanking(
            read_sequence,
            target_length,
            search_window,
            search_from_start,
            flanking,
            flank_mode="both",
            adapter_matcher=adapter_matcher,
            adapter_max_edits=adapter_max_edits,
            amplicon_max_edits=amplicon_max_edits,
            same_orientation=same_orientation,
        )
        if result[0] is not None:
            return result
        result = _extract_sequence_with_flanking(
            read_sequence,
            target_length,
            search_window,
            search_from_start,
            flanking,
            flank_mode="amplicon_only",
            adapter_matcher=adapter_matcher,
            adapter_max_edits=adapter_max_edits,
            amplicon_max_edits=amplicon_max_edits,
            same_orientation=same_orientation,
        )
        if result[0] is not None:
            return result
        return _extract_sequence_with_flanking(
            read_sequence,
            target_length,
            search_window,
            search_from_start,
            flanking,
            flank_mode="adapter_only",
            adapter_matcher=adapter_matcher,
            adapter_max_edits=adapter_max_edits,
            amplicon_max_edits=amplicon_max_edits,
            same_orientation=same_orientation,
        )
    elif flank_mode == "composite":
        if not (flanking.adapter_side and flanking.amplicon_side):
            return None, None, None
        composite_max_edits = max(0, int(max(adapter_max_edits, amplicon_max_edits)))
        comp = _composite_extract(
            read_sequence=seq,
            adapter_side=flanking.adapter_side,
            amplicon_side=flanking.amplicon_side,
            target_length=target_length,
            search_window=search_window,
            search_from_start=search_from_start,
            max_edits=composite_max_edits,
            adapter_pad=flanking.adapter_pad,
            amplicon_pad=flanking.amplicon_pad,
        )
        if comp is None:
            return None, None, None
        extracted, _, tgt_start, tgt_end = comp
        return extracted, tgt_start, tgt_end
    else:
        raise ValueError(
            "flank_mode must be one of: adapter_only, amplicon_only, both, either, composite. "
            f"Got: {flank_mode}"
        )

    if tgt_start < 0 or tgt_end > seq_len:
        return None, None, None

    extracted = seq[tgt_start:tgt_end]
    if len(extracted) != target_length:
        return None, None, None

    return extracted, tgt_start, tgt_end


def _extract_umis_for_reads(
    sequences: List[str],
    length: int,
    search_window: int,
    matcher: str,
    max_edits: int,
    umi_amplicon_max_edits: int,
    effective_flank_mode: str,
    flanking_candidates: List[Tuple[str, FlankingConfig]],
    configured_slots: List[int],
    check_start: bool,
    check_end: bool,
) -> List[Tuple[List[Optional[str]], List[Optional[str]]]]:
    """Extract UMIs for a list of read sequences.

    Returns a list of ``(umi_values, umi_positional)`` tuples, one per read.
    ``umi_values`` is ``[U1, U2]`` (biological) and ``umi_positional`` is ``[US, UE]`` (positional).
    """
    # Pre-compute RC'd flanking configs for read-end searches
    end_flanking_cache: List[FlankingConfig] = []
    for _slot, candidate in flanking_candidates:
        if candidate.adapter_side and candidate.amplicon_side:
            end_flanking_cache.append(FlankingConfig(
                adapter_side=_reverse_complement(candidate.amplicon_side),
                amplicon_side=_reverse_complement(candidate.adapter_side),
                adapter_pad=candidate.amplicon_pad,
                amplicon_pad=candidate.adapter_pad,
            ))
        elif candidate.adapter_side:
            end_flanking_cache.append(FlankingConfig(
                adapter_side=_reverse_complement(candidate.adapter_side),
                amplicon_side=None,
                adapter_pad=candidate.adapter_pad,
                amplicon_pad=candidate.amplicon_pad,
            ))
        elif candidate.amplicon_side:
            end_flanking_cache.append(FlankingConfig(
                adapter_side=None,
                amplicon_side=_reverse_complement(candidate.amplicon_side),
                adapter_pad=candidate.adapter_pad,
                amplicon_pad=candidate.amplicon_pad,
            ))
        else:
            end_flanking_cache.append(FlankingConfig(adapter_side=None, amplicon_side=None))

    results: List[Tuple[List[Optional[str]], List[Optional[str]]]] = []
    for sequence in sequences:
        umi_values: List[Optional[str]] = [None, None]
        umi_positional: List[Optional[str]] = [None, None]

        for read_end in ("start", "end"):
            if read_end == "start" and not check_start:
                continue
            if read_end == "end" and not check_end:
                continue
            search_from_start = read_end == "start"

            for i, (slot, candidate) in enumerate(flanking_candidates):
                end_flanking = candidate if search_from_start else end_flanking_cache[i]

                extracted, _, _ = _extract_sequence_with_flanking(
                    read_sequence=sequence,
                    target_length=length,
                    search_window=search_window,
                    search_from_start=search_from_start,
                    flanking=end_flanking,
                    flank_mode=effective_flank_mode,
                    adapter_matcher=matcher,
                    adapter_max_edits=max_edits,
                    amplicon_max_edits=umi_amplicon_max_edits,
                    same_orientation=False,
                )
                if extracted and read_end == "end":
                    extracted = _reverse_complement(extracted)

                if extracted:
                    idx = 0 if slot == "top" else 1
                    if umi_values[idx] is None:
                        umi_values[idx] = extracted
                    pos_idx = 0 if search_from_start else 1
                    if umi_positional[pos_idx] is None:
                        umi_positional[pos_idx] = extracted

            if configured_slots and all(
                umi_values[idx] is not None for idx in configured_slots
            ):
                break

        results.append((umi_values, umi_positional))
    return results


def _extract_umis_from_bam_range(
    bam_path: str,
    start_idx: int,
    end_idx: int,
    length: int,
    search_window: int,
    matcher: str,
    max_edits: int,
    umi_amplicon_max_edits: int,
    effective_flank_mode: str,
    flanking_candidates: List[Tuple[str, FlankingConfig]],
    configured_slots: List[int],
    check_start: bool,
    check_end: bool,
) -> List[Tuple[List[Optional[str]], List[Optional[str]]]]:
    """Worker: open BAM, read sequences in [start_idx, end_idx), extract UMIs.

    Each worker reads the BAM independently so no sequence data is pickled
    across process boundaries — only config (small dataclasses/scalars) is serialized.
    """
    import pysam  # import in child process to avoid pickling the module

    sequences: List[str] = []
    with pysam.AlignmentFile(bam_path, "rb") as bam:
        for i, read in enumerate(bam.fetch(until_eof=True)):
            if i < start_idx:
                continue
            if i >= end_idx:
                break
            sequences.append(read.query_sequence or "")

    return _extract_umis_for_reads(
        sequences,
        length=length,
        search_window=search_window,
        matcher=matcher,
        max_edits=max_edits,
        umi_amplicon_max_edits=umi_amplicon_max_edits,
        effective_flank_mode=effective_flank_mode,
        flanking_candidates=flanking_candidates,
        configured_slots=configured_slots,
        check_start=check_start,
        check_end=check_end,
    )


def annotate_umi_tags_in_bam(
    bam_path: str | Path,
    *,
    use_umi: bool,
    umi_kit_config: UMIKitConfig,
    umi_length: Any = None,
    umi_search_window: int = 200,
    umi_adapter_matcher: str = "edlib",
    umi_adapter_max_edits: int = 0,
    samtools_backend: str | None = "auto",
    umi_ends: Optional[str] = None,
    umi_flank_mode: Optional[str] = None,
    umi_amplicon_max_edits: int = 0,
    same_orientation: bool = False,
    threads: Optional[int] = None,
) -> Path:
    """Annotate aligned BAM reads with UMI tags before demultiplexing.

    Uses flanking-sequence-based extraction via ``_extract_sequence_with_flanking``.
    When ``threads`` > 1, UMI extraction is parallelized across CPU cores using
    multiprocessing while BAM I/O remains single-threaded.

    Tags written:
        US / UE  – positional: UMI extracted from read **start** / **end**
        U1 / U2  – biological: UMI from **top** (left ref end) / **bottom** (right ref end) config slot
        RX       – combined tag using U1-U2 ordering
    """
    input_bam = Path(bam_path)
    if not use_umi:
        return input_bam

    if umi_kit_config is None or umi_kit_config.flanking is None:
        raise ValueError(
            "umi_kit_config with flanking sequences is required for UMI annotation."
        )

    flanking_config = umi_kit_config.flanking
    length = umi_kit_config.length if umi_kit_config.length else int(umi_length or 0)
    effective_umi_ends = umi_ends or umi_kit_config.umi_ends or "both"
    effective_flank_mode = umi_flank_mode or umi_kit_config.umi_flank_mode or "adapter_only"
    if length <= 0:
        raise ValueError(
            "UMI length must be a positive integer when using flanking-based extraction."
        )

    search_window = max(0, int(umi_search_window))
    matcher = str(umi_adapter_matcher).strip().lower()
    if matcher not in {"exact", "edlib"}:
        raise ValueError("umi_adapter_matcher must be one of: exact, edlib")
    max_edits = max(0, int(umi_adapter_max_edits))
    if matcher == "edlib":
        require("edlib", extra="umi", purpose="fuzzy UMI adapter matching")
    backend_choice = _resolve_samtools_backend(samtools_backend)

    check_start = effective_umi_ends in ("both", "left_only", "read_start")
    check_end = effective_umi_ends in ("both", "right_only", "read_end")

    flanking_candidates: List[Tuple[str, FlankingConfig]] = []
    if flanking_config.left_ref_end is not None and (
        flanking_config.left_ref_end.adapter_side or flanking_config.left_ref_end.amplicon_side
    ):
        flanking_candidates.append(("top", flanking_config.left_ref_end))
    if flanking_config.right_ref_end is not None and (
        flanking_config.right_ref_end.adapter_side
        or flanking_config.right_ref_end.amplicon_side
    ):
        flanking_candidates.append(("bottom", flanking_config.right_ref_end))

    configured_slots = [0 if slot == "top" else 1 for slot, _ in flanking_candidates]
    configured_adapter_count = len(configured_slots)

    cpu_count = os.cpu_count() or 1
    num_workers = min(max(1, int(threads)), cpu_count) if threads else 1

    pysam_mod = _require_pysam()
    tmp_bam = input_bam.with_name(f"{input_bam.stem}.umi_tmp{input_bam.suffix}")

    # ── Count reads ─────────────────────────────────────────────────────
    total_reads = 0
    with pysam_mod.AlignmentFile(str(input_bam), "rb") as in_bam:
        for _ in tqdm(in_bam.fetch(until_eof=True), desc="UMI: counting reads", unit=" reads"):
            total_reads += 1

    # ── UMI extraction ──────────────────────────────────────────────────
    # Shared kwargs for extraction (small config only — no sequence data)
    extraction_kwargs = dict(
        length=length,
        search_window=search_window,
        matcher=matcher,
        max_edits=max_edits,
        umi_amplicon_max_edits=umi_amplicon_max_edits,
        effective_flank_mode=effective_flank_mode,
        flanking_candidates=flanking_candidates,
        configured_slots=configured_slots,
        check_start=check_start,
        check_end=check_end,
    )

    if num_workers <= 1 or total_reads == 0:
        # Single-process fast path – no IPC overhead
        sequences: List[str] = []
        with pysam_mod.AlignmentFile(str(input_bam), "rb") as in_bam:
            for read in in_bam.fetch(until_eof=True):
                sequences.append(read.query_sequence or "")
        all_results = _extract_umis_for_reads(sequences, **extraction_kwargs)
        del sequences
    else:
        chunk_size = max(1000, ceil(total_reads / num_workers))
        num_chunks = ceil(total_reads / chunk_size)
        actual_workers = min(num_workers, num_chunks)
        logger.info(
            "UMI extraction: %d reads across %d workers (chunk_size=%d)",
            total_reads,
            actual_workers,
            chunk_size,
        )
        bam_path_str = str(input_bam)
        with ProcessPoolExecutor(max_workers=actual_workers) as pool:
            futures = {}
            for chunk_idx in range(num_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min(start_idx + chunk_size, total_reads)
                future = pool.submit(
                    _extract_umis_from_bam_range,
                    bam_path_str,
                    start_idx,
                    end_idx,
                    **extraction_kwargs,
                )
                futures[future] = chunk_idx

            # Collect results in submission order, with progress per chunk
            chunk_results: Dict[int, List[Tuple[List[Optional[str]], List[Optional[str]]]]] = {}
            with tqdm(
                total=total_reads,
                desc=f"UMI: extracting ({actual_workers} workers)",
                unit=" reads",
            ) as pbar:
                for future in as_completed(futures):
                    idx = futures[future]
                    result = future.result()
                    chunk_results[idx] = result
                    pbar.update(len(result))

            all_results: List[Tuple[List[Optional[str]], List[Optional[str]]]] = []
            for chunk_idx in range(num_chunks):
                all_results.extend(chunk_results[chunk_idx])
            del chunk_results

    # ── Write BAM with tags ─────────────────────────────────────────────
    reads_with_any_umi = 0
    reads_with_all_umis = 0

    with (
        pysam_mod.AlignmentFile(str(input_bam), "rb") as in_bam,
        pysam_mod.AlignmentFile(str(tmp_bam), "wb", template=in_bam) as out_bam,
    ):
        for read_idx, read in enumerate(tqdm(
            in_bam.fetch(until_eof=True),
            total=total_reads,
            desc="UMI: writing tags",
            unit=" reads",
        )):
            umi_values, umi_positional = all_results[read_idx]

            present = [u for u in umi_values if u]
            if present:
                reads_with_any_umi += 1
            if configured_adapter_count and len(present) == configured_adapter_count:
                reads_with_all_umis += 1

            # Write positional tags (US/UE)
            us, ue = umi_positional[0], umi_positional[1]
            if us:
                read.set_tag("US", us, value_type="Z")
            if ue:
                read.set_tag("UE", ue, value_type="Z")

            # Write biological tags (U1/U2)
            u1, u2 = umi_values[0], umi_values[1]
            if u1:
                read.set_tag("U1", u1, value_type="Z")
            if u2:
                read.set_tag("U2", u2, value_type="Z")

            # Write combined tag (RX) using biological ordering
            if u1 and u2:
                read.set_tag("RX", f"{u1}-{u2}", value_type="Z")
            elif u1:
                read.set_tag("RX", u1, value_type="Z")
            elif u2:
                read.set_tag("RX", u2, value_type="Z")

            out_bam.write(read)

    del all_results

    tmp_bam.replace(input_bam)
    index_paths = (
        input_bam.with_suffix(input_bam.suffix + ".bai"),
        Path(str(input_bam) + ".bai"),
    )
    for idx_path in index_paths:
        if idx_path.exists():
            idx_path.unlink()
    if backend_choice == "python":
        _index_bam_with_pysam(input_bam)
    else:
        _index_bam_with_samtools(input_bam)

    logger.info(
        "UMI annotation complete for %s: total_reads=%d, reads_with_any_umi=%d, reads_with_all_umis=%d",
        input_bam,
        total_reads,
        reads_with_any_umi,
        reads_with_all_umis,
    )
    return input_bam


def _extract_barcode_adjacent_to_adapter_on_read_end(
    read_sequence: str,
    adapter_sequence: str,
    barcode_length: int,
    barcode_search_window: int,
    search_from_start: bool,
    adapter_matcher: str = "edlib",
    adapter_max_edits: int = 2,
) -> Tuple[Optional[str], Optional[int]]:
    """
    Extract barcode sequence adjacent to adapter, constrained to read end.

    Returns
    -------
    Tuple[Optional[str], Optional[int]]
        (barcode_sequence, adapter_start_position) or (None, None) if not found.
    """
    if not read_sequence or not adapter_sequence:
        return None, None

    seq = read_sequence.upper()
    adapter = adapter_sequence.upper()
    seq_len = len(seq)
    if seq_len == 0:
        return None, None

    matcher = str(adapter_matcher).strip().lower()
    if matcher not in {"exact", "edlib"}:
        raise ValueError("adapter_matcher must be one of: exact, edlib")

    if matcher == "exact":
        matches = [(m.start(), m.end()) for m in re.finditer(re.escape(adapter), seq)]
    else:
        edlib = require("edlib", extra="umi", purpose="fuzzy barcode adapter matching")
        result = edlib.align(adapter, seq, mode="HW", task="locations", k=max(0, adapter_max_edits))
        locations = result.get("locations", []) if isinstance(result, dict) else []
        matches = []
        for loc in locations:
            if not isinstance(loc, (list, tuple)) or len(loc) != 2:
                continue
            start_i, end_i = int(loc[0]), int(loc[1])
            if start_i < 0 or end_i < start_i:
                continue
            matches.append((start_i, end_i + 1))

    best: Optional[Tuple[int, int]] = None
    for start, end in matches:
        distance = start if search_from_start else (seq_len - end)
        if distance > barcode_search_window:
            continue
        if best is None or distance < best[0]:
            best = (distance, start)

    if best is None:
        return None, None

    adapter_start = best[1]
    adapter_end = adapter_start + len(adapter)
    if search_from_start:
        bc_start, bc_end = adapter_end, adapter_end + barcode_length
    else:
        bc_start, bc_end = adapter_start - barcode_length, adapter_start

    if bc_start < 0 or bc_end > seq_len:
        return None, None
    barcode = seq[bc_start:bc_end]
    return (barcode, adapter_start) if len(barcode) == barcode_length else (None, None)


def _match_barcode_to_references(
    extracted_barcode: str,
    barcode_references: Dict[str, str],
    max_edit_distance: int = 3,
    min_separation: Optional[int] = None,
    padded_region: Optional[str] = None,
) -> Tuple[Optional[str], Optional[int]]:
    """
    Match an extracted barcode sequence to reference barcodes.

    Parameters
    ----------
    extracted_barcode : str
        The extracted barcode sequence.
    barcode_references : Dict[str, str]
        Mapping of barcode names to barcode sequences.
    max_edit_distance : int
        Maximum edit distance to consider a match.
    min_separation : int, optional
        Minimum required distance to the second-best match. If the gap between
        best and second-best distances is smaller, the match is rejected.
    padded_region : str, optional
        Padded region. If provided, aligns fixed-length barcode reference against the
        variable-length padded region using HW semi-global alignment.

    Returns
    -------
    Tuple[Optional[str], Optional[int]]
        (best_barcode_name, edit_distance) or (None, None) if no match.
    """
    if not extracted_barcode or not barcode_references:
        return None, None

    extracted = extracted_barcode.upper()
    best_match: Optional[str] = None
    best_distance: Optional[int] = None
    second_best: Optional[int] = None

    try:
        edlib = require("edlib", extra="umi", purpose="barcode matching")
        use_edlib = True
    except Exception:
        use_edlib = False

    for bc_name, bc_seq in barcode_references.items():
        bc_seq = bc_seq.upper()

        if use_edlib:
            if padded_region is not None:
                # Padded scoring: align fixed-length barcode ref against
                # variable-length padded region using HW (infix) mode so the
                # barcode finds its best position within the padded window
                padded_upper = padded_region.upper()
                result = edlib.align(
                    bc_seq,  # Query: fixed-length reference barcode
                    padded_upper,  # Target: variable-length padded region
                    mode="HW",
                    task="distance",
                    k=max_edit_distance,
                )
            else:
                result = edlib.align(
                    extracted,  # Query: extracted barcode
                    bc_seq,  # Target: reference barcode
                    mode="NW",
                    task="distance",
                    k=max_edit_distance,
                )

            dist = result.get("editDistance", -1)
            if dist == -1:
                continue
        else:
            # Fallback to simple Hamming distance for same-length sequences
            if len(extracted) != len(bc_seq):
                continue
            dist = sum(1 for a, b in zip(extracted, bc_seq) if a != b)
            if dist > max_edit_distance:
                continue

        if best_distance is None or dist < best_distance:
            second_best = best_distance
            best_distance = dist
            best_match = bc_name
        elif second_best is None or dist < second_best:
            second_best = dist

    if best_match is None:
        return None, None

    if min_separation is not None and second_best is not None:
        if (second_best - best_distance) < int(min_separation):
            return None, None

    return best_match, best_distance


def _parse_flanking_config_from_dict(d: Dict[str, Any]) -> FlankingConfig:
    """Parse a FlankingConfig from a dictionary with adapter_side/amplicon_side keys."""
    adapter_side = d.get("adapter_side")
    amplicon_side = d.get("amplicon_side")
    if adapter_side is not None:
        adapter_side = str(adapter_side).strip().upper() or None
    if amplicon_side is not None:
        amplicon_side = str(amplicon_side).strip().upper() or None

    adapter_pad = int(d.get("adapter_pad", 5))
    amplicon_pad = int(d.get("amplicon_pad", 5))

    return FlankingConfig(
        adapter_side=adapter_side,
        amplicon_side=amplicon_side,
        adapter_pad=adapter_pad,
        amplicon_pad=amplicon_pad,
    )


def _parse_per_end_flanking(flanking_data: Dict[str, Any]) -> PerEndFlankingConfig:
    """Parse per-end flanking config from YAML flanking section.

    Supports both global flanking (adapter_side/amplicon_side at top level)
    and per-end flanking (left_ref_end/right_ref_end subsections).
    """
    per_end = PerEndFlankingConfig(
        same_orientation=bool(flanking_data.get("same_orientation", False)),
    )

    has_per_end = "left_ref_end" in flanking_data or "right_ref_end" in flanking_data

    if has_per_end:
        if "left_ref_end" in flanking_data and isinstance(flanking_data["left_ref_end"], dict):
            per_end.left_ref_end = _parse_flanking_config_from_dict(flanking_data["left_ref_end"])
        if "right_ref_end" in flanking_data and isinstance(flanking_data["right_ref_end"], dict):
            per_end.right_ref_end = _parse_flanking_config_from_dict(flanking_data["right_ref_end"])
    else:
        # Global flanking applies to both ends
        global_config = _parse_flanking_config_from_dict(flanking_data)
        per_end.left_ref_end = global_config
        per_end.right_ref_end = FlankingConfig(
            adapter_side=global_config.adapter_side,
            amplicon_side=global_config.amplicon_side,
        )

    return per_end


def _validate_barcode_sequences(
    references: Dict[str, str], yaml_path: Path
) -> Tuple[Dict[str, str], int]:
    """Validate and normalize barcode sequences. Returns (uppercased refs, barcode_length)."""
    if not references:
        raise ValueError(
            f"No valid barcode sequences found in {yaml_path}. "
            "Expected format: 'barcode_name: SEQUENCE' or 'barcodes: {name: seq}'"
        )

    valid_bases = set("ACGTN")
    for name, seq in references.items():
        if not isinstance(seq, str):
            raise ValueError(f"Barcode '{name}' has non-string value: {seq}")
        seq_upper = seq.upper()
        invalid_chars = set(seq_upper) - valid_bases
        if invalid_chars:
            raise ValueError(
                f"Barcode '{name}' contains invalid characters: {invalid_chars}. "
                "Only A, C, G, T, N are allowed."
            )

    references = {k: v.upper() for k, v in references.items()}

    lengths = {len(seq) for seq in references.values()}
    if len(lengths) > 1:
        raise ValueError(
            f"Barcodes have inconsistent lengths: {lengths}. "
            "All barcodes must have the same length."
        )

    barcode_length = lengths.pop()
    return references, barcode_length


def load_barcode_references_from_yaml(
    yaml_path: str | Path,
) -> Union[Tuple[Dict[str, str], int], BarcodeKitConfig]:
    """Load barcode reference sequences from a YAML file.

    Supports both the legacy format (flat dict or ``barcodes:`` key only)
    and the new extended format with flanking sequences and config parameters.

    Legacy format (returns ``Tuple[Dict[str, str], int]``)::

        barcode01: "ACGTACGT"
        barcode02: "TGCATGCA"

    New format (returns ``BarcodeKitConfig``)::

        name: SQK-NBD114-96
        flanking:
          adapter_side: AAGGTTAA
          amplicon_side: CAGCACCT
        barcode_ends: both
        barcode_max_edit_distance: 3
        barcode_composite_max_edits: 4
        barcodes:
          NB01: CACAAAGACACCGACAACTTTCTT
          NB02: ACAGACGACTACAAACGGAATCGA

    The new format is detected by the presence of a ``flanking`` key,
    ``top_flanking``/``bottom_flanking`` keys, ``barcode_ends`` key,
    or ``barcode_composite_max_edits`` key.

    Parameters
    ----------
    yaml_path : str | Path
        Path to the YAML file.

    Returns
    -------
    Tuple[Dict[str, str], int] | BarcodeKitConfig
        Legacy tuple for old-format files, or ``BarcodeKitConfig`` for new format.
    """
    import yaml

    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Barcode YAML file not found: {yaml_path}")

    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    if data is None:
        raise ValueError(f"Barcode YAML file is empty: {yaml_path}")

    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML structure in {yaml_path}. Expected a dictionary.")

    # Detect new format by presence of extended keys
    _new_format_keys = {
        "flanking",
        "top_flanking",
        "bottom_flanking",
        "barcode_ends",
        "barcode_composite_max_edits",
    }
    is_new_format = bool(_new_format_keys & set(data.keys()))

    # Extract barcodes dict
    if "barcodes" in data and isinstance(data["barcodes"], dict):
        references = data["barcodes"]
    else:
        # Flat structure: all string-valued keys are barcodes
        references = {k: v for k, v in data.items() if isinstance(v, str)}

    references, barcode_length = _validate_barcode_sequences(references, yaml_path)

    if not is_new_format:
        return references, barcode_length

    # Build BarcodeKitConfig for new format
    flanking = None
    if "top_flanking" in data or "bottom_flanking" in data:
        per_end = {
            "left_ref_end": data.get("top_flanking", {}),
            "right_ref_end": data.get("bottom_flanking", {}),
        }
        flanking = _parse_per_end_flanking(per_end)
    elif "flanking" in data and isinstance(data["flanking"], dict):
        flanking = _parse_per_end_flanking(data["flanking"])

    return BarcodeKitConfig(
        name=data.get("name"),
        barcodes=references,
        barcode_length=barcode_length,
        flanking=flanking,
        barcode_ends=str(data.get("barcode_ends", "both")).strip().lower(),
        barcode_max_edit_distance=int(data.get("barcode_max_edit_distance", 3)),
        barcode_composite_max_edits=int(data.get("barcode_composite_max_edits", 4)),
        barcode_min_separation=(
            None
            if data.get("barcode_min_separation", None) is None
            else int(data.get("barcode_min_separation"))
        ),
        barcode_amplicon_gap_tolerance=int(data.get("barcode_amplicon_gap_tolerance", 5)),
    )


def load_umi_config_from_yaml(yaml_path: str | Path) -> UMIKitConfig:
    """Load UMI configuration from a YAML file.

    The YAML file can contain a top-level ``umi:`` key::

        umi:
          flanking:
            adapter_side: GTACTGAC
            amplicon_side: AATTCCGG
          length: 12
          umi_ends: left_only
          umi_flank_mode: both
          adapter_max_edits: 1

    Or the same keys at the top level (no ``umi:`` wrapper).

    Parameters
    ----------
    yaml_path : str | Path
        Path to the YAML file.

    Returns
    -------
    UMIKitConfig
    """
    import yaml

    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"UMI YAML file not found: {yaml_path}")

    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    if data is None:
        raise ValueError(f"UMI YAML file is empty: {yaml_path}")

    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML structure in {yaml_path}. Expected a dictionary.")

    # Support nested umi: key
    if "umi" in data and isinstance(data["umi"], dict):
        data = data["umi"]

    flanking = None
    if "top_flanking" in data or "bottom_flanking" in data:
        per_end = {
            "left_ref_end": data.get("top_flanking", {}),
            "right_ref_end": data.get("bottom_flanking", {}),
        }
        flanking = _parse_per_end_flanking(per_end)
    elif "flanking" in data and isinstance(data["flanking"], dict):
        flanking = _parse_per_end_flanking(data["flanking"])

    same_orientation = bool(data.get("same_orientation", False))
    if flanking is not None:
        flanking.same_orientation = same_orientation

    return UMIKitConfig(
        flanking=flanking,
        length=int(data.get("length", 0)),
        umi_ends=str(data.get("umi_ends", "both")).strip().lower(),
        umi_flank_mode=str(data.get("umi_flank_mode", "adapter_only")).strip().lower(),
        adapter_max_edits=int(data.get("adapter_max_edits", 0)),
        amplicon_max_edits=int(data.get("amplicon_max_edits", 0)),
        same_orientation=same_orientation,
    )


def _build_flanking_from_adapters(adapters: List[Optional[str]]) -> PerEndFlankingConfig:
    """Convert legacy barcode_adapters / umi_adapters list to PerEndFlankingConfig.

    Legacy adapters are ``[left_ref_end_adapter, right_ref_end_adapter]`` and
    correspond to the ``adapter_side`` flanking sequence for each reference end.
    """
    left_adapter = adapters[0] if len(adapters) > 0 else None
    right_adapter = adapters[1] if len(adapters) > 1 else None

    return PerEndFlankingConfig(
        left_ref_end=FlankingConfig(adapter_side=left_adapter) if left_adapter else None,
        right_ref_end=FlankingConfig(adapter_side=right_adapter) if right_adapter else None,
    )


def resolve_barcode_config(yaml_config: BarcodeKitConfig, cfg: Any) -> Dict[str, Any]:
    """Resolve barcode configuration with priority: experiment_config > yaml > defaults.

    Parameters
    ----------
    yaml_config : BarcodeKitConfig
        Configuration loaded from YAML.
    cfg : Any
        Experiment configuration object (may have attributes for overrides).

    Returns
    -------
    Dict[str, Any]
        Resolved configuration dictionary with keys: barcode_ends,
        barcode_max_edit_distance, barcode_composite_max_edits, flanking.
    """

    def _get(attr: str, yaml_val: Any, default: Any) -> Any:
        cfg_val = getattr(cfg, attr, None)
        if cfg_val is not None:
            return cfg_val
        if yaml_val is not None:
            return yaml_val
        return default

    return {
        "barcode_ends": _get("barcode_ends", yaml_config.barcode_ends, "both"),
        "barcode_max_edit_distance": _get(
            "barcode_max_edit_distance", yaml_config.barcode_max_edit_distance, 3
        ),
        "barcode_composite_max_edits": _get(
            "barcode_composite_max_edits", yaml_config.barcode_composite_max_edits, 4
        ),
        "barcode_min_separation": _get(
            "barcode_min_separation", yaml_config.barcode_min_separation, None
        ),
        "barcode_amplicon_gap_tolerance": _get(
            "barcode_amplicon_gap_tolerance", yaml_config.barcode_amplicon_gap_tolerance, 5
        ),
        "flanking": yaml_config.flanking,
    }


def resolve_umi_config(umi_config: Optional[UMIKitConfig], cfg: Any) -> Dict[str, Any]:
    """Resolve UMI configuration with priority: experiment_config > yaml > defaults.

    Parameters
    ----------
    umi_config : UMIKitConfig or None
        Configuration loaded from UMI YAML.
    cfg : Any
        Experiment configuration object.

    Returns
    -------
    Dict[str, Any]
        Resolved UMI configuration dictionary.
    """

    def _get(attr: str, yaml_val: Any, default: Any) -> Any:
        cfg_val = getattr(cfg, attr, None)
        if cfg_val is not None:
            return cfg_val
        if yaml_val is not None:
            return yaml_val
        return default

    yaml_ends = umi_config.umi_ends if umi_config else None
    yaml_mode = umi_config.umi_flank_mode if umi_config else None
    yaml_max_edits = umi_config.adapter_max_edits if umi_config else None
    yaml_amplicon_max_edits = umi_config.amplicon_max_edits if umi_config else None
    yaml_flanking = umi_config.flanking if umi_config else None
    yaml_same_orientation = umi_config.same_orientation if umi_config else False
    if not yaml_same_orientation and yaml_flanking is not None:
        yaml_same_orientation = yaml_flanking.same_orientation

    return {
        "umi_ends": _get("umi_ends", yaml_ends, "both"),
        "umi_flank_mode": _get("umi_flank_mode", yaml_mode, "adapter_only"),
        "umi_adapter_max_edits": _get("umi_adapter_max_edits", yaml_max_edits, 0),
        "umi_amplicon_max_edits": _get("umi_amplicon_max_edits", yaml_amplicon_max_edits, 0),
        "flanking": yaml_flanking,
        "same_orientation": _get("same_orientation", yaml_same_orientation, False),
    }


def extract_and_assign_barcodes_in_bam(
    bam_path: str | Path,
    *,
    barcode_adapters: List[Optional[str]],
    barcode_references: Dict[str, str],
    barcode_length: Optional[int] = None,
    barcode_search_window: int = 200,
    barcode_max_edit_distance: int = 3,
    barcode_adapter_matcher: str = "edlib",
    barcode_composite_max_edits: int = 4,
    barcode_min_separation: Optional[int] = None,
    require_both_ends: bool = False,
    min_barcode_score: Optional[int] = None,
    samtools_backend: str | None = "auto",
    # New flanking parameters (optional; when provided, use flanking-based extraction)
    barcode_kit_config: Optional[BarcodeKitConfig] = None,
    barcode_ends: Optional[str] = None,
    barcode_amplicon_gap_tolerance: int = 5,
) -> Path:
    """Extract barcodes from reads and assign best-matching barcode from reference set.

    This function extracts barcode sequences adjacent to adapter sequences at read ends,
    matches them against a reference barcode set, and writes BAM tags for:

    - BC: Assigned barcode name (or "unclassified")
    - B1: Read-start match edit distance (if found)
    - B2: Read-end match edit distance (if found)
    - B5: Read-start barcode name (if found)
    - B6: Read-end barcode name (if found)
    - BM: Match type ("both", "read_start_only", "read_end_only", "mismatch", "unclassified")

    When ``barcode_kit_config`` with flanking sequences is provided, extraction uses
    ``_extract_sequence_with_flanking`` instead of
    ``_extract_barcode_adjacent_to_adapter_on_read_end``.

    Parameters
    ----------
    bam_path : str or Path
        Path to the input BAM file (will be modified in place).
    barcode_adapters : List[Optional[str]]
        Two-element list of adapter sequences: [left_adapter, right_adapter].
        Either can be None to skip that end.
    barcode_references : Dict[str, str]
        Mapping of barcode names to barcode sequences.
    barcode_length : int, optional
        Expected length of barcode sequences. If None, derived from barcode_references.
    barcode_search_window : int
        Maximum distance from read end to search for adapter (default 200).
    barcode_max_edit_distance : int
        Maximum edit distance to consider a barcode match (default 3).
    barcode_adapter_matcher : str
        Adapter matching method: "exact" or "edlib" (default "edlib").
    barcode_composite_max_edits : int
        Maximum edit distance for composite or single-flank matching (default 4).
    barcode_min_separation : int, optional
        Minimum required distance to the second-best match.
    require_both_ends : bool
        If True, only assign barcode if both ends match the same barcode.
    min_barcode_score : int, optional
        Minimum edit distance threshold.
    samtools_backend : str or None
        Backend for BAM indexing.
    barcode_kit_config : BarcodeKitConfig, optional
        Full barcode kit config with flanking sequences. When provided with
        flanking data, enables flanking-based extraction.
    barcode_ends : str, optional
        Which read ends to check: "both", "read_start", "read_end",
        "left_only", "right_only".
    barcode_amplicon_gap_tolerance : int
        Allowed gap/overlap (bp) between amplicon and barcode in amplicon-only extraction.

    Returns
    -------
    Path
        Path to the modified BAM file.
    """
    input_bam = Path(bam_path)

    # Derive barcode_length from references if not provided
    if barcode_length is None:
        if not barcode_references:
            raise ValueError("barcode_references is empty; cannot derive barcode_length")
        lengths = {len(seq) for seq in barcode_references.values()}
        if len(lengths) > 1:
            raise ValueError(
                f"Barcodes have inconsistent lengths: {lengths}. "
                "All barcodes must have the same length."
            )
        barcode_length = lengths.pop()

    # Determine if we should use flanking-based extraction
    use_flanking = barcode_kit_config is not None and barcode_kit_config.flanking is not None
    flanking_config = barcode_kit_config.flanking if use_flanking else None
    effective_barcode_ends = barcode_ends or (
        barcode_kit_config.barcode_ends if barcode_kit_config else "both"
    )

    # Build legacy adapter list or determine which ends to check
    if not use_flanking:
        # Legacy path: validate adapters
        if barcode_adapters is None:
            barcode_adapters = [None, None]
        elif not isinstance(barcode_adapters, (list, tuple)) or len(barcode_adapters) != 2:
            raise ValueError(
                "barcode_adapters must be a two-element list: [left_adapter, right_adapter]"
            )

        adapters: List[Optional[str]] = []
        for adapter in barcode_adapters:
            if adapter is None:
                adapters.append(None)
            else:
                val = str(adapter).strip().upper()
                adapters.append(val if val and val.lower() != "none" else None)

        if all(a is None for a in adapters):
            logger.warning("No barcode adapters provided; skipping barcode extraction")
            return input_bam

    if not barcode_references:
        raise ValueError("barcode_references must be provided with at least one barcode")

    # Normalize barcode references
    bc_refs = {name: seq.upper() for name, seq in barcode_references.items()}

    matcher = str(barcode_adapter_matcher).strip().lower()
    if matcher not in {"exact", "edlib"}:
        raise ValueError("barcode_adapter_matcher must be one of: exact, edlib")
    if matcher == "edlib":
        require("edlib", extra="umi", purpose="fuzzy barcode adapter matching")

    composite_max_edits = (
        0 if barcode_composite_max_edits is None else int(barcode_composite_max_edits)
    )

    # Determine which ends to process (read_start/read_end)
    check_start = effective_barcode_ends in ("both", "left_only", "read_start")
    check_end = effective_barcode_ends in ("both", "right_only", "read_end")

    backend_choice = _resolve_samtools_backend(samtools_backend)
    pysam_mod = _require_pysam()
    tmp_bam = input_bam.with_name(f"{input_bam.stem}.bc_tmp{input_bam.suffix}")

    # Statistics
    total_reads = 0
    reads_both_ends = 0
    reads_start_only = 0
    reads_end_only = 0
    reads_unclassified = 0
    reads_mismatch_ends = 0

    with (
        pysam_mod.AlignmentFile(str(input_bam), "rb") as in_bam,
        pysam_mod.AlignmentFile(str(tmp_bam), "wb", template=in_bam) as out_bam,
    ):
        for read in in_bam.fetch(until_eof=True):
            total_reads += 1
            sequence = read.query_sequence or ""

            bc_matches: List[Tuple[Optional[str], Optional[int]]] = [
                (None, None),
                (None, None),
            ]
            extracted_start_seq: Optional[str] = None
            extracted_end_seq: Optional[str] = None
            padded_region: Optional[str] = None

            for i, read_end in enumerate(["start", "end"]):
                if i == 0 and not check_start:
                    continue
                if i == 1 and not check_end:
                    continue

                search_from_start = read_end == "start"

                extracted_bc: Optional[str] = None
                padded_region = None

                if use_flanking:
                    flanking_candidates: List[FlankingConfig] = []
                    if flanking_config is not None:
                        if flanking_config.left_ref_end is not None and (
                            flanking_config.left_ref_end.adapter_side
                            or flanking_config.left_ref_end.amplicon_side
                        ):
                            flanking_candidates.append(flanking_config.left_ref_end)
                        if (
                            flanking_config.right_ref_end is not None
                            and flanking_config.right_ref_end not in flanking_candidates
                            and (
                                flanking_config.right_ref_end.adapter_side
                                or flanking_config.right_ref_end.amplicon_side
                            )
                        ):
                            flanking_candidates.append(flanking_config.right_ref_end)

                    for candidate in flanking_candidates:
                        end_flanking = candidate
                        if read_end == "end":
                            if candidate.adapter_side and candidate.amplicon_side:
                                end_flanking = FlankingConfig(
                                    adapter_side=_reverse_complement(candidate.amplicon_side),
                                    amplicon_side=_reverse_complement(candidate.adapter_side),
                                    adapter_pad=candidate.amplicon_pad,
                                    amplicon_pad=candidate.adapter_pad,
                                )
                            elif candidate.adapter_side:
                                end_flanking = FlankingConfig(
                                    adapter_side=_reverse_complement(candidate.adapter_side),
                                    amplicon_side=None,
                                    adapter_pad=candidate.adapter_pad,
                                    amplicon_pad=candidate.amplicon_pad,
                                )
                            elif candidate.amplicon_side:
                                end_flanking = FlankingConfig(
                                    adapter_side=None,
                                    amplicon_side=_reverse_complement(candidate.amplicon_side),
                                    adapter_pad=candidate.adapter_pad,
                                    amplicon_pad=candidate.amplicon_pad,
                                )
                            else:
                                end_flanking = FlankingConfig(
                                    adapter_side=None,
                                    amplicon_side=None,
                                )

                        extracted_bc, _, _, padded_region = _extract_barcode_with_flanking(
                            read_sequence=sequence,
                            target_length=barcode_length,
                            search_window=barcode_search_window,
                            search_from_start=search_from_start,
                            flanking=end_flanking,
                            adapter_matcher=matcher,
                            composite_max_edits=composite_max_edits,
                        )

                        if extracted_bc and read_end == "end":
                            extracted_bc = _reverse_complement(extracted_bc)
                            if padded_region is not None:
                                padded_region = _reverse_complement(padded_region)

                        if extracted_bc:
                            break
                else:
                    # Legacy path
                    adapter = adapters[i] if i < len(adapters) else None
                    if adapter is None:
                        continue
                    search_adapter = _reverse_complement(adapter) if read_end == "end" else adapter
                    extracted_bc, _ = _extract_barcode_adjacent_to_adapter_on_read_end(
                        read_sequence=sequence,
                        adapter_sequence=search_adapter,
                        barcode_length=barcode_length,
                        barcode_search_window=barcode_search_window,
                        search_from_start=search_from_start,
                        adapter_matcher=matcher,
                        adapter_max_edits=composite_max_edits,
                    )
                    if extracted_bc and read_end == "end":
                        extracted_bc = _reverse_complement(extracted_bc)

                if extracted_bc:
                    if read_end == "start":
                        extracted_start_seq = extracted_bc
                    else:
                        extracted_end_seq = extracted_bc
                    match_name, match_dist = _match_barcode_to_references(
                        extracted_bc,
                        bc_refs,
                        max_edit_distance=barcode_max_edit_distance,
                        min_separation=barcode_min_separation,
                        padded_region=padded_region,
                    )
                    if match_name is not None:
                        if min_barcode_score is None or match_dist <= min_barcode_score:
                            bc_matches[i] = (match_name, match_dist)

            left_match, left_dist = bc_matches[0]
            right_match, right_dist = bc_matches[1]

            # Determine match type and final barcode assignment
            if left_match and right_match:
                if left_match == right_match:
                    match_type = "both"
                    assigned_bc = left_match
                    reads_both_ends += 1
                else:
                    match_type = "mismatch"
                    assigned_bc = "unclassified"
                    reads_mismatch_ends += 1
                    reads_unclassified += 1
            elif left_match:
                match_type = "read_start_only"
                reads_start_only += 1
                assigned_bc = "unclassified" if require_both_ends else left_match
                if require_both_ends:
                    reads_unclassified += 1
            elif right_match:
                match_type = "read_end_only"
                reads_end_only += 1
                assigned_bc = "unclassified" if require_both_ends else right_match
                if require_both_ends:
                    reads_unclassified += 1
            else:
                match_type = "unclassified"
                assigned_bc = "unclassified"
                reads_unclassified += 1

            # Write tags
            read.set_tag("BC", assigned_bc, value_type="Z")
            read.set_tag("BM", match_type, value_type="Z")

            if extracted_start_seq:
                read.set_tag("B3", extracted_start_seq, value_type="Z")
            if extracted_end_seq:
                read.set_tag("B4", extracted_end_seq, value_type="Z")

            if left_match is not None:
                read.set_tag("B1", left_dist, value_type="i")
                read.set_tag("B5", left_match, value_type="Z")
            if right_match is not None:
                read.set_tag("B2", right_dist, value_type="i")
                read.set_tag("B6", right_match, value_type="Z")

            out_bam.write(read)

    # Replace original BAM and re-index
    tmp_bam.replace(input_bam)
    index_paths = (
        input_bam.with_suffix(input_bam.suffix + ".bai"),
        Path(str(input_bam) + ".bai"),
    )
    for idx_path in index_paths:
        if idx_path.exists():
            idx_path.unlink()
    if backend_choice == "python":
        _index_bam_with_pysam(input_bam)
    else:
        _index_bam_with_samtools(input_bam)

    logger.info(
        "Barcode extraction complete for %s:\n"
        "  total_reads=%d\n"
        "  both_ends=%d (%.1f%%)\n"
        "  read_start_only=%d (%.1f%%)\n"
        "  read_end_only=%d (%.1f%%)\n"
        "  mismatch_ends=%d (%.1f%%)\n"
        "  unclassified=%d (%.1f%%)",
        input_bam,
        total_reads,
        reads_both_ends,
        100 * reads_both_ends / max(1, total_reads),
        reads_start_only,
        100 * reads_start_only / max(1, total_reads),
        reads_end_only,
        100 * reads_end_only / max(1, total_reads),
        reads_mismatch_ends,
        100 * reads_mismatch_ends / max(1, total_reads),
        reads_unclassified,
        100 * reads_unclassified / max(1, total_reads),
    )

    return input_bam


def _stream_dorado_logs(stderr_iter) -> None:
    """Stream dorado stderr and emit structured log messages.

    Args:
        stderr_iter: Iterable of stderr lines.
    """
    last_n: int | None = None

    for raw in stderr_iter:
        line = raw.rstrip("\n")
        if _EMPTY_RE.match(line):
            continue

        m = _PROGRESS_RE.search(line)
        if m:
            n = int(m.group(1))
            logger.debug("[dorado] Output records written: %d", n)
            last_n = n
            continue

        logger.info("[dorado] %s", line)

    if last_n is not None:
        logger.info("[dorado] Final output records written: %d", last_n)


def _bam_to_fastq_with_pysam(bam_path: Union[str, Path], fastq_path: Union[str, Path]) -> None:
    """
    Minimal BAM->FASTQ using pysam. Writes unmapped or unaligned reads as-is.
    """
    bam_path = str(bam_path)
    fastq_path = str(fastq_path)

    logger.debug(f"Converting BAM to FASTQ using _bam_to_fastq_with_pysam")

    pysam_mod = _require_pysam()
    with (
        pysam_mod.AlignmentFile(bam_path, "rb", check_sq=False) as bam,
        open(fastq_path, "w", encoding="utf-8") as fq,
    ):
        for r in bam.fetch(until_eof=True):
            # Optionally skip secondary/supplementary:
            # if r.is_secondary or r.is_supplementary:
            #     continue

            name = r.query_name or ""
            seq = r.query_sequence or ""

            # Get numeric qualities; may be None
            q = r.query_qualities

            if q is None:
                # fallback: fill with low quality ("!")
                qual_str = "!" * len(seq)
            else:
                # q is an array/list of ints (Phred scores).
                # Convert to FASTQ string with Phred+33 encoding,
                # clamping to sane range [0, 93] to stay in printable ASCII.
                qual_str = "".join(chr(min(max(int(qv), 0), 93) + 33) for qv in q)

            fq.write(f"@{name}\n{seq}\n+\n{qual_str}\n")


def _sort_bam_with_pysam(
    in_bam: Union[str, Path], out_bam: Union[str, Path], threads: Optional[int] = None
) -> None:
    """Sort a BAM file using pysam.

    Args:
        in_bam: Input BAM path.
        out_bam: Output BAM path.
        threads: Optional thread count.
    """
    logger.debug(f"Sorting BAM using _sort_bam_with_pysam")
    in_bam, out_bam = str(in_bam), str(out_bam)
    args = []
    if threads:
        args += ["-@", str(threads)]
    args += ["-o", out_bam, in_bam]
    pysam_mod = _require_pysam()
    pysam_mod.sort(*args)


def _index_bam_with_pysam(bam_path: Union[str, Path], threads: Optional[int] = None) -> None:
    """Index a BAM file using pysam.

    Args:
        bam_path: BAM path to index.
        threads: Optional thread count.
    """
    bam_path = str(bam_path)
    logger.debug(f"Indexing BAM using _index_bam_with_pysam")
    pysam_mod = _require_pysam()
    # pysam.index supports samtools-style args
    if threads:
        pysam_mod.index("-@", str(threads), bam_path)
    else:
        pysam_mod.index(bam_path)


def _bam_to_fastq_with_samtools(bam_path: Union[str, Path], fastq_path: Union[str, Path]) -> None:
    """Convert BAM to FASTQ using samtools."""
    if not shutil.which("samtools"):
        raise RuntimeError("samtools is required but not available in PATH.")
    cmd = ["samtools", "fastq", str(bam_path)]
    logger.debug("Converting BAM to FASTQ using samtools: %s", " ".join(cmd))
    with open(fastq_path, "w", encoding="utf-8") as fq:
        cp = subprocess.run(cmd, stdout=fq, stderr=subprocess.PIPE, text=True)
    if cp.returncode != 0:
        raise RuntimeError(f"samtools fastq failed (exit {cp.returncode}):\n{cp.stderr}")


def _sort_bam_with_samtools(
    in_bam: Union[str, Path], out_bam: Union[str, Path], threads: Optional[int] = None
) -> None:
    """Sort a BAM file using samtools."""
    if not shutil.which("samtools"):
        raise RuntimeError("samtools is required but not available in PATH.")
    cmd = ["samtools", "sort", "-o", str(out_bam)]
    if threads:
        cmd += ["-@", str(threads)]
    cmd.append(str(in_bam))
    logger.debug("Sorting BAM using samtools: %s", " ".join(cmd))
    cp = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if cp.returncode != 0:
        raise RuntimeError(f"samtools sort failed (exit {cp.returncode}):\n{cp.stderr}")


def _index_bam_with_samtools(bam_path: Union[str, Path], threads: Optional[int] = None) -> None:
    """Index a BAM file using samtools."""
    if not shutil.which("samtools"):
        raise RuntimeError("samtools is required but not available in PATH.")
    cmd = ["samtools", "index"]
    if threads:
        cmd += ["-@", str(threads)]
    cmd.append(str(bam_path))
    logger.debug("Indexing BAM using samtools: %s", " ".join(cmd))
    cp = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if cp.returncode != 0:
        raise RuntimeError(f"samtools index failed (exit {cp.returncode}):\n{cp.stderr}")


def align_and_sort_BAM(
    fasta,
    input,
    output,
    cfg,
):
    """
    A wrapper for running dorado aligner and samtools functions

    Parameters:
        fasta (str): File path to the reference genome to align to.
        input (str): File path to the basecalled file to align. Works for .bam and .fastq files
        cfg: The configuration object

    Returns:
        None
            The function writes out files for: 1) An aligned BAM, 2) and aligned_sorted BAM, 3) an index file for the aligned_sorted BAM, 4) A bed file for the aligned_sorted BAM, 5) A text file containing read names in the aligned_sorted BAM
    """
    logger.debug("Aligning and sorting BAM using align_and_sort_BAM")
    input_basename = input.name
    input_suffix = input.suffix
    input_as_fastq = input.with_name(input.stem + ".fastq")

    aligned_BAM = output.parent / output.stem
    aligned_output = aligned_BAM.with_suffix(cfg.bam_suffix)

    aligned_sorted_BAM = aligned_BAM.with_name(aligned_BAM.stem + "_sorted")
    aligned_sorted_output = aligned_sorted_BAM.with_suffix(cfg.bam_suffix)

    if cfg.threads:
        threads = str(cfg.threads)
    else:
        threads = None

    samtools_backend = _resolve_samtools_backend(getattr(cfg, "samtools_backend", "auto"))

    if cfg.aligner == "minimap2":
        if not cfg.align_from_bam:
            logger.debug(f"Converting BAM to FASTQ: {input}")
            if samtools_backend == "python":
                _bam_to_fastq_with_pysam(input, input_as_fastq)
            else:
                _bam_to_fastq_with_samtools(input, input_as_fastq)
            logger.debug(f"Aligning FASTQ to Reference: {input_as_fastq}")
            mm_input = input_as_fastq
        else:
            logger.debug(f"Aligning BAM to Reference: {input}")
            mm_input = input

        if threads:
            minimap_command = (
                ["minimap2"] + cfg.aligner_args + ["-t", threads, str(fasta), str(mm_input)]
            )
        else:
            minimap_command = ["minimap2"] + cfg.aligner_args + [str(fasta), str(mm_input)]

        with open(aligned_output, "wb") as out:
            proc = subprocess.Popen(
                minimap_command,
                stdout=out,
                stderr=subprocess.PIPE,
                text=True,
            )

            assert proc.stderr is not None
            for line in proc.stderr:
                logger.info("[minimap2] %s", line.rstrip())

            ret = proc.wait()
            if ret != 0:
                raise RuntimeError(f"minimap2 failed with exit code {ret}")

        if not cfg.align_from_bam:
            os.remove(input_as_fastq)

    elif cfg.aligner == "dorado":
        # Run dorado aligner
        print(f"Aligning BAM to Reference: {input}")
        if threads:
            alignment_command = (
                ["dorado", "aligner", "-t", threads] + cfg.aligner_args + [str(fasta), str(input)]
            )
        else:
            alignment_command = ["dorado", "aligner"] + cfg.aligner_args + [str(fasta), str(input)]

        with open(aligned_output, "wb") as out:
            proc = subprocess.Popen(
                alignment_command,
                stdout=out,
                stderr=subprocess.PIPE,
                text=True,
            )

            assert proc.stderr is not None
            _stream_dorado_logs(proc.stderr)
            ret = proc.wait()

            if ret != 0:
                raise RuntimeError(f"dorado failed with exit code {ret}")
    else:
        logger.error(f"Aligner not recognized: {cfg.aligner}. Choose from minimap2 and dorado")
        return

    # --- Sort & Index ---
    logger.debug(f"Sorting: {aligned_output} -> {aligned_sorted_output}")
    if samtools_backend == "python":
        _sort_bam_with_pysam(aligned_output, aligned_sorted_output, threads=threads)
    else:
        _sort_bam_with_samtools(aligned_output, aligned_sorted_output, threads=threads)

    logger.debug(f"Indexing: {aligned_sorted_output}")
    if samtools_backend == "python":
        _index_bam_with_pysam(aligned_sorted_output, threads=threads)
    else:
        _index_bam_with_samtools(aligned_sorted_output, threads=threads)


def bam_qc(
    bam_files: Iterable[str | Path],
    bam_qc_dir: str | Path,
    threads: Optional[int],
    modality: str,
    stats: bool = True,
    flagstats: bool = True,
    idxstats: bool = True,
    samtools_backend: str | None = "auto",
) -> None:
    """
    QC for BAM/CRAMs: stats, flagstat, idxstats.
    Prefers pysam; falls back to `samtools` if needed.
    Runs BAMs in parallel (up to `threads`, default serial).
    """
    import subprocess

    logger.debug("Performing BAM QC using bam_qc")

    backend_choice = _resolve_samtools_backend(samtools_backend)
    have_pysam = backend_choice == "python"
    pysam_mod = _require_pysam() if have_pysam else None

    bam_qc_dir = Path(bam_qc_dir)
    bam_qc_dir.mkdir(parents=True, exist_ok=True)

    bam_paths = [Path(b) for b in bam_files]

    def _has_index(p: Path) -> bool:
        """Return True if a BAM/CRAM index exists for the path."""
        suf = p.suffix.lower()
        if suf == ".bam":
            return p.with_suffix(p.suffix + ".bai").exists() or Path(str(p) + ".bai").exists()
        if suf == ".cram":
            return Path(str(p) + ".crai").exists()
        return False

    def _ensure_index(p: Path) -> None:
        """Ensure a BAM/CRAM index exists, creating one if needed."""
        if _has_index(p):
            return
        if have_pysam:
            assert pysam_mod is not None
            pysam_mod.index(str(p))  # supports BAM & CRAM
        else:
            cmd = ["samtools", "index", str(p)]
            # capture text so errors are readable; raise on failure
            cp = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            if cp.returncode != 0:
                raise RuntimeError(f"samtools index failed (exit {cp.returncode}):\n{cp.stderr}")

    def _run_samtools_to_file(cmd: list[str], out_path: Path, bam: Path, tag: str) -> int:
        """
        Stream stderr to logger; write stdout to out_path; return rc; raise with stderr tail on failure.
        """
        last_err = deque(maxlen=80)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w") as fh:
            proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.PIPE, text=True)
            assert proc.stderr is not None
            for line in proc.stderr:
                line = line.rstrip()
                if line:
                    last_err.append(line)
                    logger.debug("[%s][%s] %s", tag, bam.name, line)
            rc = proc.wait()

        if rc != 0:
            tail = "\n".join(last_err)
            raise RuntimeError(f"{tag} failed for {bam} (exit {rc}). Stderr tail:\n{tail}")
        return rc

    def _run_one(bam: Path) -> tuple[Path, list[tuple[str, int]]]:
        """Run stats/flagstat/idxstats for a single BAM.

        Args:
            bam: Path to the BAM file.

        Returns:
            Tuple of (bam_path, list of (stage, return_code)).
        """
        import subprocess

        results: list[tuple[str, int]] = []
        base = bam.stem  # e.g. sample.bam -> sample
        out_stats = bam_qc_dir / f"{base}_stats.txt"
        out_flag = bam_qc_dir / f"{base}_flagstat.txt"
        out_idx = bam_qc_dir / f"{base}_idxstats.txt"

        # Make sure index exists (idxstats requires; stats/flagstat usually don't, but indexing is cheap/useful)
        try:
            _ensure_index(bam)
        except Exception as e:
            # Still attempt stats/flagstat if requested; idxstats may fail later if index is required.
            logger.warning("Indexing failed for %s: %s", bam, e)

        # --- stats ---
        if stats:
            if have_pysam:
                assert pysam_mod is not None
                if not hasattr(pysam_mod, "stats"):
                    raise RuntimeError("pysam.stats is unavailable in this pysam build.")
                txt = pysam_mod.stats(str(bam))
                out_stats.write_text(txt)
                results.append(("stats(pysam)", 0))
            else:
                cmd = ["samtools", "stats", str(bam)]
                rc = _run_samtools_to_file(cmd, out_stats, bam, "samtools stats")
                results.append(("stats(samtools)", rc))

        # --- flagstat ---
        if flagstats:
            if have_pysam:
                assert pysam_mod is not None
                if not hasattr(pysam_mod, "flagstat"):
                    raise RuntimeError("pysam.flagstat is unavailable in this pysam build.")
                txt = pysam_mod.flagstat(str(bam))
                out_flag.write_text(txt)
                results.append(("flagstat(pysam)", 0))
            else:
                cmd = ["samtools", "flagstat", str(bam)]
                rc = _run_samtools_to_file(cmd, out_flag, bam, "samtools flagstat")
                results.append(("flagstat(samtools)", rc))

        # --- idxstats ---
        if idxstats:
            if have_pysam:
                assert pysam_mod is not None
                if not hasattr(pysam_mod, "idxstats"):
                    raise RuntimeError("pysam.idxstats is unavailable in this pysam build.")
                txt = pysam_mod.idxstats(str(bam))
                out_idx.write_text(txt)
                results.append(("idxstats(pysam)", 0))
            else:
                cmd = ["samtools", "idxstats", str(bam)]
                rc = _run_samtools_to_file(cmd, out_idx, bam, "samtools idxstats")
                results.append(("idxstats(samtools)", rc))

        return bam, results

    max_workers = int(threads) if threads and int(threads) > 0 else 1

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_run_one, b) for b in bam_paths]
        for fut in as_completed(futs):
            try:
                bam, res = fut.result()
                summary = ", ".join(f"{name}:{rc}" for name, rc in res) or "no-op"
                logger.info("[qc] %s: %s", bam.name, summary)
            except Exception as e:
                logger.exception("QC failed: %s", e)

    if modality not in {"conversion", "direct", "deaminase"}:
        logger.warning("Unknown modality '%s', continuing.", modality)

    logger.info("QC processing completed.")


def concatenate_fastqs_to_bam(
    fastq_files: List[Union[str, Tuple[str, str], Path, Tuple[Path, Path]]],
    output_bam: Union[str, Path],
    barcode_tag: str = "BC",
    barcode_map: Optional[Dict[Union[str, Path], str]] = None,
    add_read_group: bool = True,
    rg_sample_field: Optional[str] = None,
    progress: bool = True,
    auto_pair: bool = True,
    gzip_suffixes: Tuple[str, ...] = (".gz", ".gzip"),
    samtools_backend: str | None = "auto",
) -> Dict[str, Any]:
    """
    Concatenate FASTQ(s) into an **unaligned** BAM. Supports single-end and paired-end.

    Parameters
    ----------
    fastq_files : list[Path|str] or list[(Path|str, Path|str)]
        Either explicit pairs (R1,R2) or a flat list of FASTQs (auto-paired if auto_pair=True).
    output_bam : Path|str
        Output BAM path (parent directory will be created).
    barcode_tag : str
        SAM tag used to store barcode on each read (default 'BC').
    barcode_map : dict or None
        Optional mapping {path: barcode} to override automatic filename-based barcode extraction.
    add_read_group : bool
        If True, add @RG header lines (ID = barcode) and set each read's RG tag.
    rg_sample_field : str or None
        If set, include SM=<value> in @RG.
    progress : bool
        Show tqdm progress bars.
    auto_pair : bool
        Auto-pair R1/R2 based on filename patterns if given a flat list.
    gzip_suffixes : tuple[str, ...]
        Suffixes treated as gzip-compressed FASTQ files.
    samtools_backend : str | None
        Backend selection for samtools-compatible operations (auto|python|cli).

    Returns
    -------
    dict
      {'total_reads','per_file','paired_pairs_written','singletons_written','barcodes'}
    """

    # ---------- helpers (Pathlib-only) ----------
    def _strip_fastq_ext(p: Path) -> str:
        """
        Remove common FASTQ multi-suffixes; return stem-like name.
        """
        name = p.name
        lowers = name.lower()
        gzip_exts = tuple(s.lower() for s in gzip_suffixes)
        for ext in (
            *(f".fastq{suf}" for suf in gzip_exts),
            *(f".fq{suf}" for suf in gzip_exts),
            ".fastq.bz2",
            ".fq.bz2",
            ".fastq.xz",
            ".fq.xz",
            ".fastq",
            ".fq",
        ):
            if lowers.endswith(ext):
                return name[: -len(ext)]
        return p.stem  # fallback: remove last suffix only

    def _extract_barcode_from_filename(p: Path) -> str:
        """Extract a barcode token from a FASTQ filename.

        Args:
            p: FASTQ path.

        Returns:
            Barcode token string.
        """
        stem = _strip_fastq_ext(p)
        if "_" in stem:
            token = stem.split("_")[-1]
            if token:
                return token
        return stem

    def _classify_read_token(stem: str) -> Tuple[Optional[str], Optional[int]]:
        """Classify a FASTQ filename stem into (prefix, read_number).

        Args:
            stem: Filename stem.

        Returns:
            Tuple of (prefix, read_number) or (None, None) if not matched.
        """
        # return (prefix, readnum) if matches; else (None, None)
        patterns = [
            r"(?i)(.*?)[._-]r?([12])$",  # prefix_R1 / prefix.r2 / prefix-1
            r"(?i)(.*?)[._-]read[_-]?([12])$",  # prefix_read1
        ]
        for pat in patterns:
            m = re.match(pat, stem)
            if m:
                return m.group(1), int(m.group(2))
        return None, None

    def _pair_by_filename(paths: List[Path]) -> Tuple[List[Tuple[Path, Path]], List[Path]]:
        """Pair FASTQ files based on filename conventions.

        Args:
            paths: FASTQ paths to pair.

        Returns:
            Tuple of (paired list, leftover list).
        """
        pref_map: Dict[str, Dict[int, Path]] = {}
        unpaired: List[Path] = []
        for pth in paths:
            stem = _strip_fastq_ext(pth)
            pref, num = _classify_read_token(stem)
            if pref is None:
                unpaired.append(pth)
            else:
                entry = pref_map.setdefault(pref, {})
                entry[num] = pth
        pairs: List[Tuple[Path, Path]] = []
        leftovers: List[Path] = []
        for d in pref_map.values():
            if 1 in d and 2 in d:
                pairs.append((d[1], d[2]))
            else:
                leftovers.extend(d.values())
        leftovers.extend(unpaired)
        return pairs, leftovers

    def _fastq_iter(p: Path):
        """Yield FASTQ records using pysam.FastxFile.

        Args:
            p: FASTQ path.

        Yields:
            Pysam Fastx records.
        """
        # pysam.FastxFile handles compressed extensions transparently
        pysam_mod = _require_pysam()
        with pysam_mod.FastxFile(str(p)) as fx:
            for rec in fx:
                yield rec  # rec.name, rec.sequence, rec.quality

    def _fastq_iter_plain(p: Path) -> Iterable[Tuple[str, str, str]]:
        """Yield FASTQ records from plain-text parsing.

        Args:
            p: FASTQ path.

        Yields:
            Tuple of (name, sequence, quality).
        """
        import bz2
        import gzip
        import lzma

        lowers = p.name.lower()
        if any(lowers.endswith(suf) for suf in (s.lower() for s in gzip_suffixes)):
            handle = gzip.open(p, "rt", encoding="utf-8")
        elif lowers.endswith(".bz2"):
            handle = bz2.open(p, "rt", encoding="utf-8")
        elif lowers.endswith(".xz"):
            handle = lzma.open(p, "rt", encoding="utf-8")
        else:
            handle = p.open("r", encoding="utf-8")

        with handle as fh:
            while True:
                header = fh.readline()
                if not header:
                    break
                seq = fh.readline()
                fh.readline()
                qual = fh.readline()
                if not qual:
                    break
                name = header.strip()
                if name.startswith("@"):
                    name = name[1:]
                name = name.split()[0]
                yield name, seq.strip(), qual.strip()

    def _make_unaligned_segment(
        name: str,
        seq: str,
        qual: Optional[str],
        bc: str,
        read1: bool,
        read2: bool,
    ) -> pysam.AlignedSegment:
        """Construct an unaligned pysam.AlignedSegment.

        Args:
            name: Read name.
            seq: Read sequence.
            qual: FASTQ quality string.
            bc: Barcode string.
            read1: Whether this is read 1.
            read2: Whether this is read 2.

        Returns:
            Unaligned pysam.AlignedSegment.
        """
        pysam_mod = _require_pysam()
        a = pysam_mod.AlignedSegment()
        a.query_name = name
        a.query_sequence = seq
        if qual is not None:
            a.query_qualities = pysam_mod.qualitystring_to_array(qual)
        a.is_unmapped = True
        a.is_paired = read1 or read2
        a.is_read1 = read1
        a.is_read2 = read2
        a.mate_is_unmapped = a.is_paired
        a.reference_id = -1
        a.reference_start = -1
        a.next_reference_id = -1
        a.next_reference_start = -1
        a.template_length = 0
        a.set_tag(barcode_tag, str(bc), value_type="Z")
        if add_read_group:
            a.set_tag("RG", str(bc), value_type="Z")
        return a

    def _write_sam_line(
        handle,
        name: str,
        seq: str,
        qual: str,
        bc: str,
        *,
        read1: bool,
        read2: bool,
        add_read_group: bool,
    ) -> None:
        """Write a single unaligned SAM record to a text stream."""
        if read1:
            flag = 77
        elif read2:
            flag = 141
        else:
            flag = 4
        tags = [f"{barcode_tag}:Z:{bc}"]
        if add_read_group:
            tags.append(f"RG:Z:{bc}")
        tag_str = "\t".join(tags)
        if not qual:
            qual = "*"
        line = "\t".join(
            [
                name,
                str(flag),
                "*",
                "0",
                "0",
                "*",
                "*",
                "0",
                "0",
                seq,
                qual,
                tag_str,
            ]
        )
        handle.write(f"{line}\n")

    # ---------- normalize inputs to Path ----------
    def _to_path_pair(x) -> Tuple[Path, Path]:
        """Convert a tuple of path-like objects to Path instances."""
        a, b = x
        return Path(a), Path(b)

    explicit_pairs: List[Tuple[Path, Path]] = []
    singles: List[Path] = []

    if not isinstance(fastq_files, (list, tuple)):
        raise ValueError("fastq_files must be a list of paths or list of (R1,R2) tuples.")

    if all(isinstance(x, (list, tuple)) and len(x) == 2 for x in fastq_files):
        explicit_pairs = [_to_path_pair(x) for x in fastq_files]
    else:
        flat_paths = [Path(x) for x in fastq_files if x is not None]
        if auto_pair:
            explicit_pairs, leftovers = _pair_by_filename(flat_paths)
            singles = leftovers
        else:
            singles = flat_paths

    output_bam = Path(output_bam)
    output_bam.parent.mkdir(parents=True, exist_ok=True)

    # ---------- barcodes ----------
    barcode_map = {Path(k): v for k, v in (barcode_map or {}).items()}
    per_path_barcode: Dict[Path, str] = {}
    barcodes_in_order: List[str] = []

    for r1, r2 in explicit_pairs:
        bc = barcode_map.get(r1) or barcode_map.get(r2) or _extract_barcode_from_filename(r1)
        per_path_barcode[r1] = bc
        per_path_barcode[r2] = bc
        if bc not in barcodes_in_order:
            barcodes_in_order.append(bc)
    for pth in singles:
        bc = barcode_map.get(pth) or _extract_barcode_from_filename(pth)
        per_path_barcode[pth] = bc
        if bc not in barcodes_in_order:
            barcodes_in_order.append(bc)

    # ---------- BAM header ----------
    header = {"HD": {"VN": "1.6", "SO": "unknown"}, "SQ": []}
    if add_read_group:
        header["RG"] = [
            {"ID": bc, **({"SM": rg_sample_field} if rg_sample_field else {})}
            for bc in barcodes_in_order
        ]
    header.setdefault("PG", []).append(
        {"ID": "concat-fastq", "PN": "concatenate_fastqs_to_bam", "VN": "1"}
    )

    # ---------- counters ----------
    per_file_counts: Dict[Path, int] = {}
    total_written = 0
    paired_pairs_written = 0
    singletons_written = 0

    # ---------- write BAM ----------
    backend_choice = _resolve_samtools_backend(samtools_backend)
    if backend_choice == "python":
        pysam_mod = _require_pysam()
        bam_out_ctx = pysam_mod.AlignmentFile(str(output_bam), "wb", header=header)
    else:
        cmd = ["samtools", "view", "-b", "-o", str(output_bam), "-"]
        logger.debug("Writing BAM using samtools: %s", " ".join(cmd))
        bam_out_ctx = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
        )
        assert bam_out_ctx.stdin is not None
        header_lines = ["@HD\tVN:1.6\tSO:unknown"]
        if add_read_group:
            for bc in barcodes_in_order:
                rg_fields = [f"ID:{bc}"]
                if rg_sample_field:
                    rg_fields.append(f"SM:{rg_sample_field}")
                rg_body = "\t".join(rg_fields)
                header_lines.append(f"@RG\t{rg_body}")
        header_lines.append("@PG\tID:concat-fastq\tPN:concatenate_fastqs_to_bam\tVN:1")
        bam_out_ctx.stdin.write("\n".join(header_lines) + "\n")

    try:
        # Paired
        it_pairs = explicit_pairs
        if progress and it_pairs:
            it_pairs = tqdm(it_pairs, desc="Paired FASTQ→BAM")
        for r1_path, r2_path in it_pairs:
            if not (r1_path.exists() and r2_path.exists()):
                raise FileNotFoundError(f"Paired file missing: {r1_path} or {r2_path}")
            bc = per_path_barcode.get(r1_path) or per_path_barcode.get(r2_path) or "barcode"

            if backend_choice == "python":
                it1 = _fastq_iter(r1_path)
                it2 = _fastq_iter(r2_path)
            else:
                it1 = _fastq_iter_plain(r1_path)
                it2 = _fastq_iter_plain(r2_path)

            for rec1, rec2 in zip_longest(it1, it2, fillvalue=None):

                def _clean(n: Optional[str]) -> Optional[str]:
                    """Normalize FASTQ read names by trimming read suffixes."""
                    if n is None:
                        return None
                    return re.sub(r"(?:/1$|/2$|\s[12]$)", "", n)

                name = (
                    _clean(getattr(rec1, "name", None) if backend_choice == "python" else rec1[0])
                    if rec1 is not None
                    else None
                )
                if name is None:
                    name = (
                        _clean(
                            getattr(rec2, "name", None) if backend_choice == "python" else rec2[0]
                        )
                        if rec2 is not None
                        else None
                    )
                if name is None:
                    name = (
                        getattr(rec1, "name", None)
                        if backend_choice == "python" and rec1 is not None
                        else (rec1[0] if rec1 is not None else None)
                    )
                if name is None:
                    name = (
                        getattr(rec2, "name", None)
                        if backend_choice == "python" and rec2 is not None
                        else (rec2[0] if rec2 is not None else None)
                    )

                if rec1 is not None:
                    if backend_choice == "python":
                        a1 = _make_unaligned_segment(
                            name, rec1.sequence, rec1.quality, bc, read1=True, read2=False
                        )
                        bam_out_ctx.write(a1)
                    else:
                        _write_sam_line(
                            bam_out_ctx.stdin,
                            name,
                            rec1[1],
                            rec1[2],
                            bc,
                            read1=True,
                            read2=False,
                            add_read_group=add_read_group,
                        )
                    per_file_counts[r1_path] = per_file_counts.get(r1_path, 0) + 1
                    total_written += 1
                if rec2 is not None:
                    if backend_choice == "python":
                        a2 = _make_unaligned_segment(
                            name, rec2.sequence, rec2.quality, bc, read1=False, read2=True
                        )
                        bam_out_ctx.write(a2)
                    else:
                        _write_sam_line(
                            bam_out_ctx.stdin,
                            name,
                            rec2[1],
                            rec2[2],
                            bc,
                            read1=False,
                            read2=True,
                            add_read_group=add_read_group,
                        )
                    per_file_counts[r2_path] = per_file_counts.get(r2_path, 0) + 1
                    total_written += 1

                if rec1 is not None and rec2 is not None:
                    paired_pairs_written += 1
                else:
                    if rec1 is not None:
                        singletons_written += 1
                    if rec2 is not None:
                        singletons_written += 1

        # Singles
        it_singles = singles
        if progress and it_singles:
            it_singles = tqdm(it_singles, desc="Single FASTQ→BAM")
        for pth in it_singles:
            if not pth.exists():
                raise FileNotFoundError(pth)
            bc = per_path_barcode.get(pth, "barcode")
            if backend_choice == "python":
                iterator = _fastq_iter(pth)
            else:
                iterator = _fastq_iter_plain(pth)
            for rec in iterator:
                if backend_choice == "python":
                    a = _make_unaligned_segment(
                        rec.name, rec.sequence, rec.quality, bc, read1=False, read2=False
                    )
                    bam_out_ctx.write(a)
                else:
                    _write_sam_line(
                        bam_out_ctx.stdin,
                        rec[0],
                        rec[1],
                        rec[2],
                        bc,
                        read1=False,
                        read2=False,
                        add_read_group=add_read_group,
                    )
                per_file_counts[pth] = per_file_counts.get(pth, 0) + 1
                total_written += 1
                singletons_written += 1
    finally:
        if backend_choice == "python":
            bam_out_ctx.close()
        else:
            if bam_out_ctx.stdin is not None:
                bam_out_ctx.stdin.close()
            rc = bam_out_ctx.wait()
            if rc != 0:
                stderr = bam_out_ctx.stderr.read() if bam_out_ctx.stderr else ""
                raise RuntimeError(f"samtools view failed (exit {rc}):\n{stderr}")

    return {
        "total_reads": total_written,
        "per_file": {str(k): v for k, v in per_file_counts.items()},
        "paired_pairs_written": paired_pairs_written,
        "singletons_written": singletons_written,
        "barcodes": barcodes_in_order,
    }


def count_aligned_reads(bam_file, samtools_backend: str | None = "auto"):
    """
    Counts the number of aligned reads in a bam file that map to each reference record.

    Parameters:
        bam_file (str): A string representing the path to an aligned BAM file.

    Returns:
       aligned_reads_count (int): The total number or reads aligned in the BAM.
       unaligned_reads_count (int): The total number of reads not aligned in the BAM.
       record_counts (dict): A dictionary keyed by reference record instance that points toa tuple containing the total reads mapped to the record and the fraction of mapped reads which map to the record.

    """
    logger.info("Counting aligned reads in BAM > {}".format(bam_file.name))
    backend_choice = _resolve_samtools_backend(samtools_backend)
    aligned_reads_count = 0
    unaligned_reads_count = 0

    if backend_choice == "python":
        pysam_mod = _require_pysam()
        record_counts = defaultdict(int)
        with pysam_mod.AlignmentFile(str(bam_file), "rb") as bam:
            total_reads = bam.mapped + bam.unmapped
            # Iterate over reads to get the total mapped read counts and the reads that map to each reference
            for read in bam:
                if read.is_unmapped:
                    unaligned_reads_count += 1
                else:
                    aligned_reads_count += 1
                    record_counts[read.reference_name] += (
                        1  # Automatically increments if key exists, adds if not
                    )

            # reformat the dictionary to contain read counts mapped to the reference, as well as the proportion of mapped reads in reference
            for reference in record_counts:
                proportion_mapped_reads_in_record = record_counts[reference] / aligned_reads_count
                record_counts[reference] = (
                    record_counts[reference],
                    proportion_mapped_reads_in_record,
                )
        return aligned_reads_count, unaligned_reads_count, dict(record_counts)

    bam_path = Path(bam_file)
    _ensure_bam_index(bam_path, backend_choice)
    cmd = ["samtools", "idxstats", str(bam_path)]
    cp = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if cp.returncode != 0:
        raise RuntimeError(f"samtools idxstats failed (exit {cp.returncode}):\n{cp.stderr}")
    return _parse_idxstats_output(cp.stdout)


def annotate_demux_type_from_bi_tag(
    bam_path: str | Path, output_path: Optional[str | Path] = None, threshold: float = 0.65
) -> Path:
    """Annotate reads with a BM tag based on dorado bi per-end barcode scores.

    The bi tag is a float array of 7 elements written by dorado >= 1.3.1:

    - bi[0]: overall barcode score
    - bi[1-2]: top barcode position/length
    - bi[3]: **top (front) barcode score**
    - bi[4-5]: bottom barcode position/length
    - bi[6]: **bottom (rear) barcode score**

    Classification logic:

    - Both bi[3] and bi[6] > threshold → "both"
    - Only bi[3] > threshold → "read_start_only"
    - Only bi[6] > threshold → "read_end_only"
    - Has BC but no bi tag → "unknown"
    - No BC tag → "unclassified"

    Parameters
    ----------
    bam_path : str or Path
        Path to input BAM file.
    output_path : str or Path, optional
        Path to output BAM file. If None, overwrites input in-place (via a temporary file).
    threshold : float, default 0.0
        Minimum per-end score to consider a barcode match.

    Returns
    -------
    Path
        Path to the output BAM file.
    """
    pysam_mod = _require_pysam()
    bam_path = Path(bam_path)

    if output_path is None:
        tmp_path = bam_path.with_suffix(".bm_tmp.bam")
    else:
        tmp_path = Path(output_path)

    counts = {
        "both": 0,
        "read_start_only": 0,
        "read_end_only": 0,
        "unknown": 0,
        "unclassified": 0,
    }

    with pysam_mod.AlignmentFile(str(bam_path), "rb", check_sq=False) as inbam:
        with pysam_mod.AlignmentFile(str(tmp_path), "wb", header=inbam.header) as outbam:
            for read in inbam:
                if not read.has_tag("BC"):
                    bm_value = "unclassified"
                elif not read.has_tag("bi"):
                    bm_value = "unknown"
                else:
                    bi = read.get_tag("bi")
                    top_score = bi[3] if len(bi) > 3 else -1.0
                    bottom_score = bi[6] if len(bi) > 6 else -1.0

                    if top_score > threshold and bottom_score > threshold:
                        bm_value = "both"
                    elif top_score > threshold:
                        bm_value = "read_start_only"
                    elif bottom_score > threshold:
                        bm_value = "read_end_only"
                    else:
                        bm_value = "unknown"

                read.set_tag("BM", bm_value, value_type="Z")
                outbam.write(read)
                counts[bm_value] += 1

    if output_path is None:
        tmp_path.rename(bam_path)
        result_path = bam_path
    else:
        result_path = tmp_path

    # Re-index after rewriting the BAM
    pysam_mod.index(str(result_path))

    logger.info(
        "BM tag annotation complete for %s: %s",
        result_path.name,
        ", ".join(f"{k}={v}" for k, v in counts.items()),
    )
    return result_path


def demux_and_index_BAM(
    aligned_sorted_BAM,
    split_dir,
    bam_suffix,
    barcode_kit,
    barcode_both_ends,
    trim,
    threads,
    no_classify=False,
    file_prefix=None,
):
    """Split an input BAM by barcode and index the outputs.

    Parameters
    ----------
    aligned_sorted_BAM : Path
        Path to the aligned, sorted BAM input.
    split_dir : Path
        Directory to write demultiplexed BAMs.
    bam_suffix : str
        Suffix to add to BAM filenames (e.g., \".bam\").
    barcode_kit : str
        Name of the barcoding kit to pass to dorado.
    barcode_both_ends : bool
        Whether to require both ends to be barcoded.
    trim : bool
        Whether to trim barcodes after demultiplexing.
    threads : int
        Number of threads to use.
    no_classify : bool, default False
        When True, use ``--no-classify`` to split by existing BC tags without
        re-classifying barcodes. Ignores ``barcode_kit`` and ``barcode_both_ends``.
    file_prefix : str or None, default None
        Optional prefix for output BAM filenames. If None, defaults to
        ``\"de\"``/``\"se\"`` based on ``barcode_both_ends`` (legacy behavior).

    Returns
    -------
    list[Path]
        List of split BAM file paths.
    """

    input_bam = aligned_sorted_BAM.with_suffix(bam_suffix)

    # Build command based on mode
    if no_classify:
        command = ["dorado", "demux", "--no-classify"]
    else:
        command = ["dorado", "demux", "--kit-name", barcode_kit]
        if barcode_both_ends:
            command.append("--barcode-both-ends")

    if not trim:
        command.append("--no-trim")
    if threads:
        command += ["-t", str(threads)]

    command += ["--emit-summary", "--sort-bam", "--output-dir", str(split_dir)]
    command.append(str(input_bam))

    logger.info("Running dorado demux: %s", " ".join(command))

    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert proc.stderr is not None
    _stream_dorado_logs(proc.stderr)
    rc = proc.wait()

    if rc != 0:
        raise RuntimeError(f"dorado demux failed with exit code {rc}")

    bam_files = sorted(
        p for p in split_dir.glob(f"*{bam_suffix}") if p.is_file() and p.suffix == bam_suffix
    )
    if not bam_files:
        # dorado demux can nest BAMs under run/sample directories
        nested_bams = sorted(
            p for p in split_dir.rglob(f"*{bam_suffix}") if p.is_file() and p.suffix == bam_suffix
        )
        if nested_bams:
            logger.info(
                "Flattening %d demuxed BAMs from nested directories into %s",
                len(nested_bams),
                split_dir,
            )
            flattened = []
            for bam in nested_bams:
                target = split_dir / bam.name
                if target.exists():
                    logger.warning("Target BAM already exists, skipping move: %s", target)
                    continue
                shutil.move(str(bam), str(target))
                bai = bam.with_suffix(bam_suffix + ".bai")
                if bai.exists():
                    target_bai = target.with_suffix(bam_suffix + ".bai")
                    if target_bai.exists():
                        logger.warning("Target BAI already exists, skipping move: %s", target_bai)
                    else:
                        shutil.move(str(bai), str(target_bai))
                flattened.append(target)

            # Remove empty nested directories
            for root, dirs, _files in os.walk(split_dir, topdown=False):
                for d in dirs:
                    path = Path(root) / d
                    if path == split_dir:
                        continue
                    try:
                        if not any(path.iterdir()):
                            path.rmdir()
                    except OSError:
                        pass

            bam_files = sorted(flattened)
        else:
            bam_files = []

    if not bam_files:
        raise FileNotFoundError(f"No BAM files found in {split_dir} with suffix {bam_suffix}")

    # ---- Optional renaming with prefix ----
    renamed_bams = []
    # Use file_prefix if provided, otherwise default to legacy se/de prefix
    if file_prefix is None:
        prefix = "de" if barcode_both_ends else "se"
    else:
        prefix = file_prefix

    for bam in bam_files:
        bam = Path(bam)
        bai = bam.with_suffix(bam_suffix + ".bai")  # dorado's sorting produces .bam.bai

        if prefix:
            new_name = f"{prefix}_{bam.name}"
        else:
            new_name = bam.name

        new_bam = bam.with_name(new_name)
        bam.rename(new_bam)

        # rename index if exists
        if bai.exists():
            new_bai = new_bam.with_suffix(bam_suffix + ".bai")
            bai.rename(new_bai)

        renamed_bams.append(new_bam)

    return renamed_bams


def extract_base_identities(
    bam_file,
    record,
    positions,
    max_reference_length,
    sequence,
    samtools_backend: str | None = "auto",
):
    """
    Efficiently extracts base identities from mapped reads with reference coordinates.

    Parameters:
        bam_file (str): Path to the BAM file.
        record (str): Name of the reference record.
        positions (list): Positions to extract (0-based).
        max_reference_length (int): Maximum reference length for padding.
        sequence (str): The sequence of the record fasta

    Returns:
        dict: Base identities from forward mapped reads.
        dict: Base identities from reverse mapped reads.
        dict: Mismatch counts per read.
        dict: Mismatch trends per read.
        dict: Integer-encoded mismatch bases per read.
        dict: Base quality scores per read aligned to reference positions.
        dict: Read span masks per read (1 within span, 0 outside).
    """
    logger.debug("Extracting nucleotide identities for each read using extract_base_identities")
    timestamp = time.strftime("[%Y-%m-%d %H:%M:%S]")

    positions = set(positions)
    fwd_base_identities = defaultdict(lambda: np.full(max_reference_length, "N", dtype="<U1"))
    rev_base_identities = defaultdict(lambda: np.full(max_reference_length, "N", dtype="<U1"))
    mismatch_counts_per_read = defaultdict(lambda: defaultdict(Counter))
    mismatch_base_identities = defaultdict(
        lambda: np.full(
            max_reference_length,
            MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT["N"],
            dtype=np.int16,
        )
    )
    base_quality_scores = defaultdict(lambda: np.full(max_reference_length, -1, dtype=np.int16))
    read_span_masks = defaultdict(lambda: np.zeros(max_reference_length, dtype=np.int8))

    backend_choice = _resolve_samtools_backend(samtools_backend)
    ref_seq = sequence.upper()
    sequence_length = len(sequence)

    def _encode_mismatch_base(base: str) -> int:
        return MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT.get(
            base.upper(), MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT["N"]
        )

    if backend_choice == "python":
        logger.debug("Extracting base identities using python")
        pysam_mod = _require_pysam()
        # print(f"{timestamp} Reading reads from {chromosome} BAM file: {bam_file}")
        with pysam_mod.AlignmentFile(str(bam_file), "rb") as bam:
            total_reads = bam.mapped
            for read in bam.fetch(record):
                if not read.is_mapped:
                    continue  # Skip unmapped reads

                read_name = read.query_name
                query_sequence = read.query_sequence
                query_qualities = read.query_qualities or []
                base_dict = rev_base_identities if read.is_reverse else fwd_base_identities

                # Init arrays for each read in each dict
                mismatch_base_identities[read_name]
                base_quality_scores[read_name]
                read_span_masks[read_name]

                if read.reference_start is not None and read.reference_end is not None:
                    span_end = min(read.reference_end, max_reference_length)
                    read_span_masks[read_name][read.reference_start : span_end] = 1

                # Use get_aligned_pairs directly with positions filtering
                aligned_pairs = read.get_aligned_pairs(matches_only=True)

                for read_position, reference_position in aligned_pairs:
                    if reference_position is None or read_position is None:
                        continue
                    read_base = query_sequence[read_position]
                    ref_base = ref_seq[reference_position]
                    if reference_position in positions:
                        base_dict[read_name][reference_position] = read_base
                        if read_position < len(query_qualities):
                            base_quality_scores[read_name][reference_position] = query_qualities[
                                read_position
                            ]

                    # Track mismatches (excluding Ns)
                    if read_base != ref_base and read_base != "N" and ref_base != "N":
                        mismatch_counts_per_read[read_name][ref_base][read_base] += 1
                        mismatch_base_identities[read_name][reference_position] = (
                            _encode_mismatch_base(read_base)
                        )
    else:
        bam_path = Path(bam_file)
        logger.debug("Extracting base identities using samtools")
        _ensure_bam_index(bam_path, backend_choice)

        def _iter_aligned_pairs(cigar: str, start: int) -> Iterable[Tuple[int, int]]:
            qpos = 0
            rpos = start
            for length_str, op in re.findall(r"(\d+)([MIDNSHP=XB])", cigar):
                length = int(length_str)
                if op in {"M", "=", "X"}:
                    for _ in range(length):
                        yield qpos, rpos
                        qpos += 1
                        rpos += 1
                elif op in {"I", "S"}:
                    qpos += length
                elif op in {"D", "N"}:
                    rpos += length
                elif op in {"H", "P"}:
                    continue

        def _reference_span_from_cigar(cigar: str) -> int:
            span = 0
            for length_str, op in re.findall(r"(\d+)([MIDNSHP=XB])", cigar):
                if op in {"M", "D", "N", "=", "X"}:
                    span += int(length_str)
            return span

        cmd = ["samtools", "view", "-F", "4", str(bam_path), record]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        assert proc.stdout is not None
        for line in proc.stdout:
            if not line.strip() or line.startswith("@"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 11:
                continue
            read_name = fields[0]
            flag = int(fields[1])
            pos = int(fields[3])
            cigar = fields[5]
            query_sequence = fields[9]
            qual_string = fields[10]
            if cigar == "*" or query_sequence == "*":
                continue
            base_dict = rev_base_identities if (flag & 16) else fwd_base_identities
            mismatch_base_identities[read_name]
            base_quality_scores[read_name]
            read_span_masks[read_name]
            qualities = (
                [ord(ch) - 33 for ch in qual_string] if qual_string and qual_string != "*" else []
            )
            ref_start = pos - 1
            ref_end = ref_start + _reference_span_from_cigar(cigar)
            span_end = min(ref_end, max_reference_length)
            if ref_start < max_reference_length:
                read_span_masks[read_name][ref_start:span_end] = 1
            for read_pos, ref_pos in _iter_aligned_pairs(cigar, pos - 1):
                if read_pos >= len(query_sequence) or ref_pos >= len(ref_seq):
                    continue
                read_base = query_sequence[read_pos]
                ref_base = ref_seq[ref_pos]
                if ref_pos in positions:
                    base_dict[read_name][ref_pos] = read_base
                    if read_pos < len(qualities):
                        base_quality_scores[read_name][ref_pos] = qualities[read_pos]
                if read_base != ref_base and read_base != "N" and ref_base != "N":
                    mismatch_counts_per_read[read_name][ref_base][read_base] += 1
                    mismatch_base_identities[read_name][ref_pos] = _encode_mismatch_base(read_base)
        rc = proc.wait()
        if rc != 0:
            stderr = proc.stderr.read() if proc.stderr else ""
            raise RuntimeError(f"samtools view failed (exit {rc}):\n{stderr}")

    # Determine C→T vs G→A dominance per read
    mismatch_trend_per_read = {}
    for read_name, ref_dict in mismatch_counts_per_read.items():
        c_to_t = ref_dict.get("C", {}).get("T", 0)
        g_to_a = ref_dict.get("G", {}).get("A", 0)

        if abs(c_to_t - g_to_a) < 0.01 and c_to_t > 0:
            mismatch_trend_per_read[read_name] = "equal"
        elif c_to_t > g_to_a:
            mismatch_trend_per_read[read_name] = "C->T"
        elif g_to_a > c_to_t:
            mismatch_trend_per_read[read_name] = "G->A"
        else:
            mismatch_trend_per_read[read_name] = "none"

    if sequence_length < max_reference_length:
        padding_value = MODKIT_EXTRACT_SEQUENCE_BASE_TO_INT["PAD"]
        for mismatch_values in mismatch_base_identities.values():
            mismatch_values[sequence_length:] = padding_value

    return (
        dict(fwd_base_identities),
        dict(rev_base_identities),
        dict(mismatch_counts_per_read),
        mismatch_trend_per_read,
        dict(mismatch_base_identities),
        dict(base_quality_scores),
        dict(read_span_masks),
    )


def extract_read_features_from_bam(
    bam_file_path: str | Path, samtools_backend: str | None = "auto"
) -> Dict[str, List[float]]:
    """Extract read metrics from a BAM file.

    Args:
        bam_file_path: Path to the BAM file.
        samtools_backend: Backend selection for samtools-compatible operations (auto|python|cli).

    Returns:
        Mapping of read name to [read_length, read_median_qscore, reference_length,
        mapped_length, mapping_quality, reference_start, reference_end].
    """
    logger.debug(
        "Extracting read metrics from BAM using extract_read_features_from_bam: %s",
        bam_file_path,
    )
    backend_choice = _resolve_samtools_backend(samtools_backend)
    read_metrics: Dict[str, List[float]] = {}

    if backend_choice == "python":
        pysam_mod = _require_pysam()
        with pysam_mod.AlignmentFile(str(bam_file_path), "rb") as bam_file:
            reference_lengths = dict(zip(bam_file.references, bam_file.lengths))
            for read in bam_file:
                if read.is_unmapped:
                    continue
                read_quality = read.query_qualities
                if read_quality is None:
                    median_read_quality = float("nan")
                else:
                    median_read_quality = float(np.median(read_quality))
                reference_length = reference_lengths.get(read.reference_name, float("nan"))
                mapped_length = sum(end - start for start, end in read.get_blocks())
                mapping_quality = float(read.mapping_quality)
                reference_start = float(read.reference_start)
                reference_end = float(read.reference_end)
                read_metrics[read.query_name] = [
                    float(read.query_length),
                    median_read_quality,
                    float(reference_length),
                    float(mapped_length),
                    mapping_quality,
                    reference_start,
                    reference_end,
                ]
        return read_metrics

    bam_path = Path(bam_file_path)

    def _parse_reference_lengths(header_text: str) -> Dict[str, int]:
        ref_lengths: Dict[str, int] = {}
        for line in header_text.splitlines():
            if not line.startswith("@SQ"):
                continue
            fields = line.split("\t")
            name = None
            length = None
            for field in fields[1:]:
                if field.startswith("SN:"):
                    name = field.split(":", 1)[1]
                elif field.startswith("LN:"):
                    length = int(field.split(":", 1)[1])
            if name is not None and length is not None:
                ref_lengths[name] = length
        return ref_lengths

    def _mapped_length_from_cigar(cigar: str) -> int:
        mapped = 0
        for length_str, op in re.findall(r"(\d+)([MIDNSHP=XB])", cigar):
            length = int(length_str)
            if op in {"M", "=", "X"}:
                mapped += length
        return mapped

    def _reference_span_from_cigar(cigar: str) -> int:
        reference_span = 0
        for length_str, op in re.findall(r"(\d+)([MIDNSHP=XB])", cigar):
            length = int(length_str)
            if op in {"M", "D", "N", "=", "X"}:
                reference_span += length
        return reference_span

    header_cp = subprocess.run(
        ["samtools", "view", "-H", str(bam_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if header_cp.returncode != 0:
        raise RuntimeError(
            f"samtools view -H failed (exit {header_cp.returncode}):\n{header_cp.stderr}"
        )
    reference_lengths = _parse_reference_lengths(header_cp.stdout)

    proc = subprocess.Popen(
        ["samtools", "view", "-F", "4", str(bam_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        if not line.strip() or line.startswith("@"):
            continue
        fields = line.rstrip("\n").split("\t")
        if len(fields) < 11:
            continue
        read_name = fields[0]
        reference_name = fields[2]
        mapping_quality = float(fields[4])
        cigar = fields[5]
        reference_start = float(int(fields[3]) - 1)
        sequence = fields[9]
        quality = fields[10]
        if sequence == "*":
            read_length = float("nan")
        else:
            read_length = float(len(sequence))
        if quality == "*" or not quality:
            median_read_quality = float("nan")
        else:
            phreds = [ord(char) - 33 for char in quality]
            median_read_quality = float(np.median(phreds))
        reference_length = float(reference_lengths.get(reference_name, float("nan")))
        mapped_length = float(_mapped_length_from_cigar(cigar)) if cigar != "*" else 0.0
        if cigar != "*":
            reference_end = float(reference_start + _reference_span_from_cigar(cigar))
        else:
            reference_end = float("nan")
        read_metrics[read_name] = [
            read_length,
            median_read_quality,
            reference_length,
            mapped_length,
            mapping_quality,
            reference_start,
            reference_end,
        ]

    rc = proc.wait()
    if rc != 0:
        stderr = proc.stderr.read() if proc.stderr else ""
        raise RuntimeError(f"samtools view failed (exit {rc}):\n{stderr}")

    return read_metrics


def extract_read_tags_from_bam(
    bam_file_path: str | Path,
    tag_names: Iterable[str] | None = None,
    include_flags: bool = True,
    include_cigar: bool = True,
    samtools_backend: str | None = "auto",
) -> Dict[str, Dict[str, object]]:
    """Extract per-read tag metadata from a BAM file.

    Args:
        bam_file_path: Path to the BAM file.
        tag_names: Iterable of BAM tag names to extract (e.g., ["NM", "MD", "MM", "ML"]).
            If None, only flags/cigar are populated.
        include_flags: Whether to include a list of flag names for each read.
        include_cigar: Whether to include the CIGAR string for each read.
        samtools_backend: Backend selection for samtools-compatible operations (auto|python|cli).

    Returns:
        Mapping of read name to a dict of extracted tag values.
    """
    backend_choice = _resolve_samtools_backend(samtools_backend)
    tag_names_list = [tag.upper() for tag in tag_names] if tag_names else []
    read_tags: Dict[str, Dict[str, object]] = {}

    def _decode_flags(flag: int) -> list[str]:
        return [name for bit, name in _BAM_FLAG_BITS if flag & bit]

    if backend_choice == "python":
        pysam_mod = _require_pysam()
        with pysam_mod.AlignmentFile(str(bam_file_path), "rb") as bam_file:
            for read in bam_file.fetch(until_eof=True):
                if not read.query_name:
                    continue
                tag_map: Dict[str, object] = {}
                if include_cigar:
                    tag_map["CIGAR"] = read.cigarstring
                if include_flags:
                    tag_map["FLAGS"] = _decode_flags(read.flag)
                for tag in tag_names_list:
                    try:
                        tag_map[tag] = read.get_tag(tag)
                    except Exception:
                        tag_map[tag] = None
                read_tags[read.query_name] = tag_map
    else:
        cmd = ["samtools", "view", "-F", "4", str(bam_file_path)]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        assert proc.stdout is not None
        for line in proc.stdout:
            if not line.strip() or line.startswith("@"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 11:
                continue
            read_name = fields[0]
            flag = int(fields[1])
            cigar = fields[5]
            tag_map: Dict[str, object] = {}
            if include_cigar:
                tag_map["CIGAR"] = cigar
            if include_flags:
                tag_map["FLAGS"] = _decode_flags(flag)
            if tag_names_list:
                raw_tags = fields[11:]
                parsed_tags: Dict[str, str] = {}
                for raw_tag in raw_tags:
                    parts = raw_tag.split(":", 2)
                    if len(parts) == 3:
                        tag_name, _tag_type, value = parts
                        parsed_tags[tag_name.upper()] = value
                for tag in tag_names_list:
                    tag_map[tag] = parsed_tags.get(tag)
            read_tags[read_name] = tag_map
        rc = proc.wait()
        if rc != 0:
            stderr = proc.stderr.read() if proc.stderr else ""
            raise RuntimeError(f"samtools view failed (exit {rc}):\n{stderr}")

    return read_tags


def find_secondary_supplementary_read_names(
    bam_file_path: str | Path,
    read_names: Iterable[str],
    samtools_backend: str | None = "auto",
) -> tuple[set[str], set[str]]:
    """Find read names with secondary or supplementary alignments in a BAM.

    Args:
        bam_file_path: Path to the BAM file to scan.
        read_names: Iterable of read names to check.
        samtools_backend: Backend selection for samtools-compatible operations (auto|python|cli).

    Returns:
        Tuple of (secondary_read_names, supplementary_read_names).
    """
    target_names = set(read_names)
    if not target_names:
        return set(), set()

    secondary_reads: set[str] = set()
    supplementary_reads: set[str] = set()
    backend_choice = _resolve_samtools_backend(samtools_backend)

    if backend_choice == "python":
        pysam_mod = _require_pysam()
        with pysam_mod.AlignmentFile(str(bam_file_path), "rb") as bam_file:
            for read in bam_file.fetch(until_eof=True):
                if not read.query_name or read.query_name not in target_names:
                    continue
                if read.is_secondary:
                    secondary_reads.add(read.query_name)
                if read.is_supplementary:
                    supplementary_reads.add(read.query_name)
    else:

        def _collect(flag: int) -> set[str]:
            cmd = ["samtools", "view", "-f", str(flag), str(bam_file_path)]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            assert proc.stdout is not None
            hits: set[str] = set()
            for line in proc.stdout:
                if not line.strip() or line.startswith("@"):
                    continue
                read_name = line.split("\t", 1)[0]
                if read_name in target_names:
                    hits.add(read_name)
            rc = proc.wait()
            if rc != 0:
                stderr = proc.stderr.read() if proc.stderr else ""
                raise RuntimeError(f"samtools view failed (exit {rc}):\n{stderr}")
            return hits

        secondary_reads = _collect(0x100)
        supplementary_reads = _collect(0x800)

    return secondary_reads, supplementary_reads


def extract_secondary_supplementary_alignment_spans(
    bam_file_path: str | Path,
    read_names: Iterable[str],
    samtools_backend: str | None = "auto",
) -> tuple[
    dict[str, list[tuple[float, float, float]]], dict[str, list[tuple[float, float, float]]]
]:
    """Extract reference/read span data for secondary/supplementary alignments.

    Args:
        bam_file_path: Path to the BAM file to scan.
        read_names: Iterable of read names to check.
        samtools_backend: Backend selection for samtools-compatible operations (auto|python|cli).

    Returns:
        Tuple of (secondary_spans, supplementary_spans) where each mapping contains
        read names mapped to lists of (reference_start, reference_end, read_span).
    """
    target_names = set(read_names)
    if not target_names:
        return {}, {}

    secondary_spans: dict[str, list[tuple[float, float, float]]] = {}
    supplementary_spans: dict[str, list[tuple[float, float, float]]] = {}
    backend_choice = _resolve_samtools_backend(samtools_backend)

    if backend_choice == "python":
        pysam_mod = _require_pysam()
        with pysam_mod.AlignmentFile(str(bam_file_path), "rb") as bam_file:
            for read in bam_file.fetch(until_eof=True):
                if not read.query_name or read.query_name not in target_names:
                    continue
                if not (read.is_secondary or read.is_supplementary):
                    continue
                reference_start = (
                    float(read.reference_start)
                    if read.reference_start is not None
                    else float("nan")
                )
                reference_end = (
                    float(read.reference_end) if read.reference_end is not None else float("nan")
                )
                read_span = (
                    float(read.query_alignment_length)
                    if read.query_alignment_length is not None
                    else float("nan")
                )
                if read.is_secondary:
                    secondary_spans.setdefault(read.query_name, []).append(
                        (reference_start, reference_end, read_span)
                    )
                if read.is_supplementary:
                    supplementary_spans.setdefault(read.query_name, []).append(
                        (reference_start, reference_end, read_span)
                    )
        return secondary_spans, supplementary_spans

    def _mapped_length_from_cigar(cigar: str) -> int:
        mapped = 0
        for length_str, op in re.findall(r"(\d+)([MIDNSHP=XB])", cigar):
            length = int(length_str)
            if op in {"M", "=", "X"}:
                mapped += length
        return mapped

    def _reference_span_from_cigar(cigar: str) -> int:
        reference_span = 0
        for length_str, op in re.findall(r"(\d+)([MIDNSHP=XB])", cigar):
            length = int(length_str)
            if op in {"M", "D", "N", "=", "X"}:
                reference_span += length
        return reference_span

    def _collect(flag: int) -> dict[str, list[tuple[float, float, float]]]:
        cmd = ["samtools", "view", "-f", str(flag), str(bam_file_path)]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        assert proc.stdout is not None
        spans: dict[str, list[tuple[float, float, float]]] = {}
        for line in proc.stdout:
            if not line.strip() or line.startswith("@"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 11:
                continue
            read_name = fields[0]
            if read_name not in target_names:
                continue
            cigar = fields[5]
            reference_start = float(int(fields[3]) - 1)
            if cigar != "*":
                reference_end = float(reference_start + _reference_span_from_cigar(cigar))
                read_span = float(_mapped_length_from_cigar(cigar))
            else:
                reference_end = float("nan")
                read_span = float("nan")
            spans.setdefault(read_name, []).append((reference_start, reference_end, read_span))
        rc = proc.wait()
        if rc != 0:
            stderr = proc.stderr.read() if proc.stderr else ""
            raise RuntimeError(f"samtools view failed (exit {rc}):\n{stderr}")
        return spans

    secondary_spans = _collect(0x100)
    supplementary_spans = _collect(0x800)

    return secondary_spans, supplementary_spans


def extract_readnames_from_bam(aligned_BAM, samtools_backend: str | None = "auto"):
    """
    Takes a BAM and writes out a txt file containing read names from the BAM

    Parameters:
        aligned_BAM (str): Path to an input aligned_BAM to extract read names from.

    Returns:
        None

    """
    # Make a text file of reads for the BAM
    backend_choice = _resolve_samtools_backend(samtools_backend)
    txt_output = aligned_BAM.split(".bam")[0] + "_read_names.txt"

    if backend_choice == "python":
        pysam_mod = _require_pysam()
        with (
            pysam_mod.AlignmentFile(aligned_BAM, "rb") as bam,
            open(txt_output, "w", encoding="utf-8") as output_file,
        ):
            for read in bam:
                output_file.write(f"{read.query_name}\n")
        return

    samtools_view = subprocess.Popen(
        ["samtools", "view", aligned_BAM], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    assert samtools_view.stdout is not None
    with open(txt_output, "w", encoding="utf-8") as output_file:
        for line in samtools_view.stdout:
            if not line.strip():
                continue
            qname = line.split("\t", 1)[0]
            output_file.write(f"{qname}\n")
    rc = samtools_view.wait()
    if rc != 0:
        stderr = samtools_view.stderr.read() if samtools_view.stderr else ""
        raise RuntimeError(f"samtools view failed (exit {rc}):\n{stderr}")


def separate_bam_by_bc(
    input_bam, output_prefix, bam_suffix, split_dir, samtools_backend: str | None = "auto"
):
    """
    Separates an input BAM file on the BC SAM tag values.

    Parameters:
        input_bam (str): File path to the BAM file to split.
        output_prefix (str): A prefix to append to the output BAM.
        bam_suffix (str): A suffix to add to the bam file.
        split_dir (str): String indicating path to directory to split BAMs into

    Returns:
        None
            Writes out split BAM files.
    """
    logger.debug("Demultiplexing BAM based on the BC tag")
    bam_base = input_bam.name
    bam_base_minus_suffix = input_bam.stem

    backend_choice = _resolve_samtools_backend(samtools_backend)

    if backend_choice == "python":
        pysam_mod = _require_pysam()
        # Open the input BAM file for reading
        with pysam_mod.AlignmentFile(str(input_bam), "rb") as bam:
            # Create a dictionary to store output BAM files
            output_files = {}
            # Iterate over each read in the BAM file
            for read in bam:
                try:
                    # Get the barcode tag value
                    bc_tag = read.get_tag("BC", with_value_type=True)[0]
                    # bc_tag = read.get_tag("BC", with_value_type=True)[0].split('barcode')[1]
                    # Open the output BAM file corresponding to the barcode
                    if bc_tag not in output_files:
                        output_path = (
                            split_dir
                            / f"{output_prefix}_{bam_base_minus_suffix}_{bc_tag}{bam_suffix}"
                        )
                        output_files[bc_tag] = pysam_mod.AlignmentFile(
                            str(output_path), "wb", header=bam.header
                        )
                    # Write the read to the corresponding output BAM file
                    output_files[bc_tag].write(read)
                except KeyError:
                    logger.warning(f"BC tag not present for read: {read.query_name}")
        # Close all output BAM files
        for output_file in output_files.values():
            output_file.close()
        return

    def _collect_bc_tags() -> set[str]:
        bc_tags: set[str] = set()
        proc = subprocess.Popen(
            ["samtools", "view", str(input_bam)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            if not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            for tag in fields[11:]:
                if tag.startswith("BC:"):
                    bc_tags.add(tag.split(":", 2)[2])
                    break
        rc = proc.wait()
        if rc != 0:
            stderr = proc.stderr.read() if proc.stderr else ""
            raise RuntimeError(f"samtools view failed (exit {rc}):\n{stderr}")
        return bc_tags

    bc_tags = _collect_bc_tags()
    if not bc_tags:
        logger.warning("No BC tags found in %s", input_bam)
        return

    for bc_tag in bc_tags:
        output_path = split_dir / f"{output_prefix}_{bam_base_minus_suffix}_{bc_tag}{bam_suffix}"
        cmd = ["samtools", "view", "-b", "-d", f"BC:{bc_tag}", "-o", str(output_path)]
        cmd.append(str(input_bam))
        cp = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if cp.returncode != 0:
            raise RuntimeError(
                f"samtools view failed for BC={bc_tag} (exit {cp.returncode}):\n{cp.stderr}"
            )


def split_and_index_BAM(
    aligned_sorted_BAM, split_dir, bam_suffix, samtools_backend: str | None = "auto"
):
    """
    A wrapper function for splitting BAMS and indexing them.
    Parameters:
        aligned_sorted_BAM (str): A string representing the file path of the aligned_sorted BAM file.
        split_dir (str): A string representing the file path to the directory to split the BAMs into.
        bam_suffix (str): A suffix to add to the bam file.

    Returns:
        None
            Splits an input BAM file on barcode value and makes a BAM index file.
    """
    logger.debug("Demultiplexing and indexing BAMS based on BC tag using split_and_index_BAM")
    aligned_sorted_BAM = Path(aligned_sorted_BAM)
    split_dir = Path(split_dir)
    aligned_sorted_output = aligned_sorted_BAM.with_suffix(bam_suffix)
    file_prefix = date_string()
    separate_bam_by_bc(
        aligned_sorted_output,
        file_prefix,
        bam_suffix,
        split_dir,
        samtools_backend=samtools_backend,
    )
    # Make a BAM index file for the BAMs in that directory
    bam_pattern = "*" + bam_suffix
    bam_files = glob.glob(str(split_dir / bam_pattern))
    bam_files = [Path(bam) for bam in bam_files if ".bai" not in str(bam)]
    backend_choice = _resolve_samtools_backend(samtools_backend)
    for input_file in bam_files:
        if backend_choice == "python":
            _index_bam_with_pysam(input_file)
        else:
            _index_bam_with_samtools(input_file)

    return bam_files

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Any, Tuple, Union, Optional
import re
from itertools import zip_longest

from smftools.optional_imports import require

pysam = require("pysam", extra="informatics", purpose="archived fastq to BAM")
from tqdm import tqdm


def concatenate_fastqs_to_bam(
    fastq_files: List[Union[str, Tuple[str, str], Path, Tuple[Path, Path]]],
    output_bam: Union[str, Path],
    barcode_tag: str = "BC",
    barcode_map: Optional[Dict[Union[str, Path], str]] = None,
    add_read_group: bool = True,
    rg_sample_field: Optional[str] = None,
    progress: bool = True,
    auto_pair: bool = True,
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
        for ext in (".fastq.gz", ".fq.gz", ".fastq.bz2", ".fq.bz2", ".fastq.xz", ".fq.xz", ".fastq", ".fq"):
            if lowers.endswith(ext):
                return name[: -len(ext)]
        return p.stem  # fallback: remove last suffix only

    def _extract_barcode_from_filename(p: Path) -> str:
        """Extract a barcode token from a FASTQ filename."""
        stem = _strip_fastq_ext(p)
        if "_" in stem:
            token = stem.split("_")[-1]
            if token:
                return token
        return stem

    def _classify_read_token(stem: str) -> Tuple[Optional[str], Optional[int]]:
        """Classify a FASTQ filename stem into (prefix, read_number)."""
        # return (prefix, readnum) if matches; else (None, None)
        patterns = [
            r"(?i)(.*?)[._-]r?([12])$",        # prefix_R1 / prefix.r2 / prefix-1
            r"(?i)(.*?)[._-]read[_-]?([12])$", # prefix_read1
        ]
        for pat in patterns:
            m = re.match(pat, stem)
            if m:
                return m.group(1), int(m.group(2))
        return None, None

    def _pair_by_filename(paths: List[Path]) -> Tuple[List[Tuple[Path, Path]], List[Path]]:
        """Pair FASTQ files based on filename conventions."""
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
        """Yield FASTQ records using pysam.FastxFile."""
        # pysam.FastxFile handles compressed extensions transparently
        with pysam.FastxFile(str(p)) as fx:
            for rec in fx:
                yield rec  # rec.name, rec.sequence, rec.quality

    def _make_unaligned_segment(
        name: str,
        seq: str,
        qual: Optional[str],
        bc: str,
        read1: bool,
        read2: bool,
    ) -> pysam.AlignedSegment:
        """Construct an unaligned pysam.AlignedSegment."""
        a = pysam.AlignedSegment()
        a.query_name = name
        a.query_sequence = seq
        if qual is not None:
            a.query_qualities = pysam.qualitystring_to_array(qual)
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
        header["RG"] = [{"ID": bc, **({"SM": rg_sample_field} if rg_sample_field else {})} for bc in barcodes_in_order]
    header.setdefault("PG", []).append(
        {"ID": "concat-fastq", "PN": "concatenate_fastqs_to_bam", "VN": "1"}
    )

    # ---------- counters ----------
    per_file_counts: Dict[Path, int] = {}
    total_written = 0
    paired_pairs_written = 0
    singletons_written = 0

    # ---------- write BAM ----------
    with pysam.AlignmentFile(str(output_bam), "wb", header=header) as bam_out:
        # Paired
        it_pairs = explicit_pairs
        if progress and it_pairs:
            it_pairs = tqdm(it_pairs, desc="Paired FASTQ→BAM")
        for r1_path, r2_path in it_pairs:
            if not (r1_path.exists() and r2_path.exists()):
                raise FileNotFoundError(f"Paired file missing: {r1_path} or {r2_path}")
            bc = per_path_barcode.get(r1_path) or per_path_barcode.get(r2_path) or "barcode"

            it1 = _fastq_iter(r1_path)
            it2 = _fastq_iter(r2_path)

            for rec1, rec2 in zip_longest(it1, it2, fillvalue=None):
                def _clean(n: Optional[str]) -> Optional[str]:
                    """Normalize FASTQ read names by trimming read suffixes."""
                    if n is None:
                        return None
                    return re.sub(r"(?:/1$|/2$|\s[12]$)", "", n)

                name = (
                    _clean(getattr(rec1, "name", None))
                    or _clean(getattr(rec2, "name", None))
                    or getattr(rec1, "name", None)
                    or getattr(rec2, "name", None)
                )

                if rec1 is not None:
                    a1 = _make_unaligned_segment(name, rec1.sequence, rec1.quality, bc, read1=True, read2=False)
                    bam_out.write(a1)
                    per_file_counts[r1_path] = per_file_counts.get(r1_path, 0) + 1
                    total_written += 1
                if rec2 is not None:
                    a2 = _make_unaligned_segment(name, rec2.sequence, rec2.quality, bc, read1=False, read2=True)
                    bam_out.write(a2)
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
            for rec in _fastq_iter(pth):
                a = _make_unaligned_segment(rec.name, rec.sequence, rec.quality, bc, read1=False, read2=False)
                bam_out.write(a)
                per_file_counts[pth] = per_file_counts.get(pth, 0) + 1
                total_written += 1
                singletons_written += 1

    return {
        "total_reads": total_written,
        "per_file": {str(k): v for k, v in per_file_counts.items()},
        "paired_pairs_written": paired_pairs_written,
        "singletons_written": singletons_written,
        "barcodes": barcodes_in_order,
    }

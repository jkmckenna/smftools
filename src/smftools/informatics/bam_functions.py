from __future__ import annotations

from pathlib import Path
import os
import subprocess
import glob
import time
from typing import Dict, List, Any, Tuple, Union, Optional, Iterable
import re
from itertools import zip_longest
import pysam

import numpy as np
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm
from collections import defaultdict, Counter

from ..readwrite import make_dirs, time_string, date_string


def _bam_to_fastq_with_pysam(bam_path: Union[str, Path], fastq_path: Union[str, Path]) -> None:
    """
    Minimal BAM->FASTQ using pysam. Writes unmapped or unaligned reads as-is.
    """
    bam_path = str(bam_path)
    fastq_path = str(fastq_path)
    with (
        pysam.AlignmentFile(bam_path, "rb", check_sq=False) as bam,
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
    in_bam, out_bam = str(in_bam), str(out_bam)
    args = []
    if threads:
        args += ["-@", str(threads)]
    args += ["-o", out_bam, in_bam]
    pysam.sort(*args)


def _index_bam_with_pysam(bam_path: Union[str, Path], threads: Optional[int] = None) -> None:
    bam_path = str(bam_path)
    # pysam.index supports samtools-style args
    if threads:
        pysam.index("-@", str(threads), bam_path)
    else:
        pysam.index(bam_path)


def align_and_sort_BAM(
    fasta,
    input,
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
    input_basename = input.name
    input_suffix = input.suffix
    input_as_fastq = input.with_name(input.stem + ".fastq")

    output_path_minus_suffix = cfg.output_directory / input.stem

    aligned_BAM = output_path_minus_suffix.with_name(output_path_minus_suffix.stem + "_aligned")
    aligned_output = aligned_BAM.with_suffix(cfg.bam_suffix)
    aligned_sorted_BAM = aligned_BAM.with_name(aligned_BAM.stem + "_sorted")
    aligned_sorted_output = aligned_sorted_BAM.with_suffix(cfg.bam_suffix)

    if cfg.threads:
        threads = str(cfg.threads)
    else:
        threads = None

    if cfg.aligner == "minimap2":
        if not cfg.align_from_bam:
            print(f"Converting BAM to FASTQ: {input}")
            _bam_to_fastq_with_pysam(input, input_as_fastq)
            print(f"Aligning FASTQ to Reference: {input_as_fastq}")
            mm_input = input_as_fastq
        else:
            print(f"Aligning BAM to Reference: {input}")
            mm_input = input

        if threads:
            minimap_command = (
                ["minimap2"] + cfg.aligner_args + ["-t", threads, str(fasta), str(mm_input)]
            )
        else:
            minimap_command = ["minimap2"] + cfg.aligner_args + [str(fasta), str(mm_input)]
        subprocess.run(minimap_command, stdout=open(aligned_output, "wb"))

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
        subprocess.run(alignment_command, stdout=open(aligned_output, "wb"))

    else:
        print(f"Aligner not recognized: {cfg.aligner}. Choose from minimap2 and dorado")
        return

    # --- Sort & Index with pysam ---
    print(f"[pysam] Sorting: {aligned_output} -> {aligned_sorted_output}")
    _sort_bam_with_pysam(aligned_output, aligned_sorted_output, threads=threads)

    print(f"[pysam] Indexing: {aligned_sorted_output}")
    _index_bam_with_pysam(aligned_sorted_output, threads=threads)


def bam_qc(
    bam_files: Iterable[str | Path],
    bam_qc_dir: str | Path,
    threads: Optional[int],
    modality: str,
    stats: bool = True,
    flagstats: bool = True,
    idxstats: bool = True,
) -> None:
    """
    QC for BAM/CRAMs: stats, flagstat, idxstats.
    Prefers pysam; falls back to `samtools` if needed.
    Runs BAMs in parallel (up to `threads`, default serial).
    """
    import subprocess
    import shutil

    # Try to import pysam once
    try:
        import pysam

        HAVE_PYSAM = True
    except Exception:
        HAVE_PYSAM = False

    bam_qc_dir = Path(bam_qc_dir)
    bam_qc_dir.mkdir(parents=True, exist_ok=True)

    bam_files = [Path(b) for b in bam_files]

    def _has_index(p: Path) -> bool:
        if p.suffix.lower() == ".bam":
            bai = p.with_suffix(p.suffix + ".bai")
            bai_alt = Path(str(p) + ".bai")
            return bai.exists() or bai_alt.exists()
        if p.suffix.lower() == ".cram":
            crai = Path(str(p) + ".crai")
            return crai.exists()
        return False

    def _ensure_index(p: Path) -> None:
        if _has_index(p):
            return
        if HAVE_PYSAM:
            # pysam.index supports both BAM & CRAM
            pysam.index(str(p))
        else:
            cmd = ["samtools", "index", str(p)]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def _run_one(bam: Path) -> Tuple[Path, List[Tuple[str, int]]]:
        # outputs + return (file, [(task_name, returncode)])
        results: List[Tuple[str, int]] = []
        base = bam.stem  # filename without .bam
        out_stats = bam_qc_dir / f"{base}_stats.txt"
        out_flag = bam_qc_dir / f"{base}_flagstat.txt"
        out_idx = bam_qc_dir / f"{base}_idxstats.txt"

        # Make sure index exists (samtools stats/flagstat don’t require, idxstats does)
        try:
            _ensure_index(bam)
        except Exception as e:
            # Still attempt stats/flagstat if requested
            print(f"[warn] Indexing failed for {bam}: {e}")

        # Choose runner per task
        def run_stats():
            if not stats:
                return
            if HAVE_PYSAM and hasattr(pysam, "stats"):
                txt = pysam.stats(str(bam))
                out_stats.write_text(txt)
                results.append(("stats(pysam)", 0))
            else:
                cmd = ["samtools", "stats", str(bam)]
                with open(out_stats, "w") as fh:
                    cp = subprocess.run(cmd, stdout=fh, stderr=subprocess.PIPE)
                results.append(("stats(samtools)", cp.returncode))
                if cp.returncode != 0:
                    raise RuntimeError(cp.stderr.decode(errors="replace"))

        def run_flagstat():
            if not flagstats:
                return
            if HAVE_PYSAM and hasattr(pysam, "flagstat"):
                txt = pysam.flagstat(str(bam))
                out_flag.write_text(txt)
                results.append(("flagstat(pysam)", 0))
            else:
                cmd = ["samtools", "flagstat", str(bam)]
                with open(out_flag, "w") as fh:
                    cp = subprocess.run(cmd, stdout=fh, stderr=subprocess.PIPE)
                results.append(("flagstat(samtools)", cp.returncode))
                if cp.returncode != 0:
                    raise RuntimeError(cp.stderr.decode(errors="replace"))

        def run_idxstats():
            if not idxstats:
                return
            if HAVE_PYSAM and hasattr(pysam, "idxstats"):
                txt = pysam.idxstats(str(bam))
                out_idx.write_text(txt)
                results.append(("idxstats(pysam)", 0))
            else:
                cmd = ["samtools", "idxstats", str(bam)]
                with open(out_idx, "w") as fh:
                    cp = subprocess.run(cmd, stdout=fh, stderr=subprocess.PIPE)
                results.append(("idxstats(samtools)", cp.returncode))
                if cp.returncode != 0:
                    raise RuntimeError(cp.stderr.decode(errors="replace"))

        # Sanity: ensure samtools exists if pysam missing
        if not HAVE_PYSAM:
            if not shutil.which("samtools"):
                raise RuntimeError("Neither pysam nor samtools is available in PATH.")

        # Execute tasks (serial per file; parallelized across files)
        run_stats()
        run_flagstat()
        run_idxstats()
        return bam, results

    # Parallel across BAMs
    max_workers = int(threads) if threads and int(threads) > 0 else 1
    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for b in bam_files:
            futures.append(ex.submit(_run_one, b))

        for fut in as_completed(futures):
            try:
                bam, res = fut.result()
                summary = ", ".join(f"{name}:{rc}" for name, rc in res) or "no-op"
                print(f"[qc] {bam.name}: {summary}")
            except Exception as e:
                print(f"[error] QC failed: {e}")

    # Placeholders to keep your signature stable
    if modality not in {"conversion", "direct"}:
        print(f"[warn] Unknown modality '{modality}', continuing.")

    print("QC processing completed.")


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
        for ext in (
            ".fastq.gz",
            ".fq.gz",
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
        stem = _strip_fastq_ext(p)
        if "_" in stem:
            token = stem.split("_")[-1]
            if token:
                return token
        return stem

    def _classify_read_token(stem: str) -> Tuple[Optional[str], Optional[int]]:
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
                    a1 = _make_unaligned_segment(
                        name, rec1.sequence, rec1.quality, bc, read1=True, read2=False
                    )
                    bam_out.write(a1)
                    per_file_counts[r1_path] = per_file_counts.get(r1_path, 0) + 1
                    total_written += 1
                if rec2 is not None:
                    a2 = _make_unaligned_segment(
                        name, rec2.sequence, rec2.quality, bc, read1=False, read2=True
                    )
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
                a = _make_unaligned_segment(
                    rec.name, rec.sequence, rec.quality, bc, read1=False, read2=False
                )
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


def count_aligned_reads(bam_file):
    """
    Counts the number of aligned reads in a bam file that map to each reference record.

    Parameters:
        bam_file (str): A string representing the path to an aligned BAM file.

    Returns:
       aligned_reads_count (int): The total number or reads aligned in the BAM.
       unaligned_reads_count (int): The total number of reads not aligned in the BAM.
       record_counts (dict): A dictionary keyed by reference record instance that points toa tuple containing the total reads mapped to the record and the fraction of mapped reads which map to the record.

    """
    print("{0}: Counting aligned reads in BAM > {1}".format(time_string(), bam_file))
    aligned_reads_count = 0
    unaligned_reads_count = 0
    # Make a dictionary, keyed by the reference_name of reference chromosome that points to an integer number of read counts mapped to the chromosome, as well as the proportion of mapped reads in that chromosome
    record_counts = defaultdict(int)

    with pysam.AlignmentFile(str(bam_file), "rb") as bam:
        total_reads = bam.mapped + bam.unmapped
        # Iterate over reads to get the total mapped read counts and the reads that map to each reference
        for read in tqdm(bam, desc="Counting aligned reads in BAM", total=total_reads):
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
            record_counts[reference] = (record_counts[reference], proportion_mapped_reads_in_record)

    return aligned_reads_count, unaligned_reads_count, dict(record_counts)


def demux_and_index_BAM(
    aligned_sorted_BAM, split_dir, bam_suffix, barcode_kit, barcode_both_ends, trim, threads
):
    """
    A wrapper function for splitting BAMS and indexing them.
    Parameters:
        aligned_sorted_BAM (str): A string representing the file path of the aligned_sorted BAM file.
        split_dir (str): A string representing the file path to the directory to split the BAMs into.
        bam_suffix (str): A suffix to add to the bam file.
        barcode_kit (str): Name of barcoding kit.
        barcode_both_ends (bool): Whether to require both ends to be barcoded.
        trim (bool): Whether to trim off barcodes after demultiplexing.
        threads (int): Number of threads to use.

    Returns:
        bam_files (list): List of split BAM file path strings
            Splits an input BAM file on barcode value and makes a BAM index file.
    """
    input_bam = aligned_sorted_BAM.with_suffix(bam_suffix)
    command = ["dorado", "demux", "--kit-name", barcode_kit]
    if barcode_both_ends:
        command.append("--barcode-both-ends")
    if not trim:
        command.append("--no-trim")
    if threads:
        command += ["-t", str(threads)]
    else:
        pass
    command += ["--emit-summary", "--sort-bam", "--output-dir", str(split_dir)]
    command.append(str(input_bam))
    command_string = " ".join(command)
    print(f"Running: {command_string}")
    subprocess.run(command)

    bam_files = sorted(
        p for p in split_dir.glob(f"*{bam_suffix}") if p.is_file() and p.suffix == bam_suffix
    )

    if not bam_files:
        raise FileNotFoundError(f"No BAM files found in {split_dir} with suffix {bam_suffix}")

    # ---- Optional renaming with prefix ----
    renamed_bams = []
    prefix = "de" if barcode_both_ends else "se"

    for bam in bam_files:
        bam = Path(bam)
        bai = bam.with_suffix(bam_suffix + ".bai")  # dorado’s sorting produces .bam.bai

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


def extract_base_identities(bam_file, chromosome, positions, max_reference_length, sequence):
    """
    Efficiently extracts base identities from mapped reads with reference coordinates.

    Parameters:
        bam_file (str): Path to the BAM file.
        chromosome (str): Name of the reference chromosome.
        positions (list): Positions to extract (0-based).
        max_reference_length (int): Maximum reference length for padding.
        sequence (str): The sequence of the record fasta

    Returns:
        dict: Base identities from forward mapped reads.
        dict: Base identities from reverse mapped reads.
    """
    timestamp = time.strftime("[%Y-%m-%d %H:%M:%S]")

    positions = set(positions)
    fwd_base_identities = defaultdict(lambda: np.full(max_reference_length, "N", dtype="<U1"))
    rev_base_identities = defaultdict(lambda: np.full(max_reference_length, "N", dtype="<U1"))
    mismatch_counts_per_read = defaultdict(lambda: defaultdict(Counter))

    # print(f"{timestamp} Reading reads from {chromosome} BAM file: {bam_file}")
    with pysam.AlignmentFile(str(bam_file), "rb") as bam:
        total_reads = bam.mapped
        ref_seq = sequence.upper()
        for read in bam.fetch(chromosome):
            if not read.is_mapped:
                continue  # Skip unmapped reads

            read_name = read.query_name
            query_sequence = read.query_sequence
            base_dict = rev_base_identities if read.is_reverse else fwd_base_identities

            # Use get_aligned_pairs directly with positions filtering
            aligned_pairs = read.get_aligned_pairs(matches_only=True)

            for read_position, reference_position in aligned_pairs:
                if reference_position in positions:
                    read_base = query_sequence[read_position]
                    ref_base = ref_seq[reference_position]

                    base_dict[read_name][reference_position] = read_base

                # Track mismatches (excluding Ns)
                if read_base != ref_base and read_base != "N" and ref_base != "N":
                    mismatch_counts_per_read[read_name][ref_base][read_base] += 1

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

    return (
        dict(fwd_base_identities),
        dict(rev_base_identities),
        dict(mismatch_counts_per_read),
        mismatch_trend_per_read,
    )


def extract_read_features_from_bam(bam_file_path):
    """
    Make a dict of reads from a bam that points to a list of read metrics: read length, read median Q-score, reference length, mapped length, mapping quality
    Params:
        bam_file_path (str):
    Returns:
        read_metrics (dict)
    """
    # Open the BAM file
    print(f"Extracting read features from BAM: {bam_file_path}")
    with pysam.AlignmentFile(bam_file_path, "rb") as bam_file:
        read_metrics = {}
        reference_lengths = bam_file.lengths  # List of lengths for each reference (chromosome)
        for read in bam_file:
            # Skip unmapped reads
            if read.is_unmapped:
                continue
            # Extract the read metrics
            read_quality = read.query_qualities
            median_read_quality = np.median(read_quality)
            # Extract the reference (chromosome) name and its length
            reference_name = read.reference_name
            reference_index = bam_file.references.index(reference_name)
            reference_length = reference_lengths[reference_index]
            mapped_length = sum(end - start for start, end in read.get_blocks())
            mapping_quality = read.mapping_quality  # Phred-scaled MAPQ
            read_metrics[read.query_name] = [
                read.query_length,
                median_read_quality,
                reference_length,
                mapped_length,
                mapping_quality,
            ]

    return read_metrics


def extract_readnames_from_bam(aligned_BAM):
    """
    Takes a BAM and writes out a txt file containing read names from the BAM

    Parameters:
        aligned_BAM (str): Path to an input aligned_BAM to extract read names from.

    Returns:
        None

    """
    import subprocess

    # Make a text file of reads for the BAM
    txt_output = aligned_BAM.split(".bam")[0] + "_read_names.txt"
    samtools_view = subprocess.Popen(["samtools", "view", aligned_BAM], stdout=subprocess.PIPE)
    with open(txt_output, "w") as output_file:
        cut_process = subprocess.Popen(
            ["cut", "-f1"], stdin=samtools_view.stdout, stdout=output_file
        )
    samtools_view.stdout.close()
    cut_process.wait()
    samtools_view.wait()


def separate_bam_by_bc(input_bam, output_prefix, bam_suffix, split_dir):
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
    bam_base = input_bam.name
    bam_base_minus_suffix = input_bam.stem

    # Open the input BAM file for reading
    with pysam.AlignmentFile(str(input_bam), "rb") as bam:
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
                        split_dir / f"{output_prefix}_{bam_base_minus_suffix}_{bc_tag}{bam_suffix}"
                    )
                    output_files[bc_tag] = pysam.AlignmentFile(
                        str(output_path), "wb", header=bam.header
                    )
                # Write the read to the corresponding output BAM file
                output_files[bc_tag].write(read)
            except KeyError:
                print(f"BC tag not present for read: {read.query_name}")
    # Close all output BAM files
    for output_file in output_files.values():
        output_file.close()


def split_and_index_BAM(aligned_sorted_BAM, split_dir, bam_suffix):
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
    aligned_sorted_output = aligned_sorted_BAM + bam_suffix
    file_prefix = date_string()
    separate_bam_by_bc(aligned_sorted_output, file_prefix, bam_suffix, split_dir)
    # Make a BAM index file for the BAMs in that directory
    bam_pattern = "*" + bam_suffix
    bam_files = glob.glob(split_dir / bam_pattern)
    bam_files = [str(bam) for bam in bam_files if ".bai" not in str(bam)]
    for input_file in bam_files:
        pysam.index(input_file)

    return bam_files

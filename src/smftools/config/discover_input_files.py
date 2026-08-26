from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Union

from smftools.constants import BAM_SUFFIX

POD5_EXTS = {".pod5", ".p5"}
FAST5_EXTS = {".fast5", ".f5"}
FASTQ_EXTS = {
    ".fastq",
    ".fq",
    ".fastq.gz",
    ".fq.gz",
    ".fastq.bz2",
    ".fq.bz2",
    ".fastq.xz",
    ".fq.xz",
    ".fastq.zst",
    ".fq.zst",
}
H5AD_EXTS = {".h5ad", ".h5"}
COMPRESSED_EXTS = {".gz", ".bz2", ".xz", ".zst"}


def extension_key(path: Path) -> str:
    """Return a robust extension key, folding one compressor suffix into the one before it.

    Examples:
        ``a.fastq.gz`` -> ``.fastq.gz``; ``a.fq.xz`` -> ``.fq.xz``; ``a.bam`` -> ``.bam``;
        ``a`` -> ``""``.

    Args:
        path: The file path to key.

    Returns:
        str: The lowercased extension key.
    """
    suff = [s.lower() for s in Path(path).suffixes]
    if not suff:
        return ""
    if suff[-1] in COMPRESSED_EXTS and len(suff) >= 2:
        return suff[-2] + suff[-1]
    return suff[-1]


def input_kind_for_path(path: Path, *, bam_suffix: str = BAM_SUFFIX) -> str:
    """Classify one input file by extension.

    Shared by directory discovery and by offline identity restoration, which has
    only recorded paths to work from. Keeping one implementation is what stops the
    two from disagreeing and silently moving a stage's config hash (`PSR-01`).

    Args:
        path: The file to classify.
        bam_suffix: The configured BAM suffix.

    Returns:
        str: One of ``pod5``, ``fast5``, ``fastq``, ``h5ad``, ``bam``, ``sam``,
        ``cram``, or ``other``.
    """
    if not bam_suffix.startswith("."):
        bam_suffix = "." + bam_suffix
    key = extension_key(path)
    if key in POD5_EXTS:
        return "pod5"
    if key in FAST5_EXTS:
        return "fast5"
    if key in FASTQ_EXTS:
        return "fastq"
    if key in H5AD_EXTS:
        return "h5ad"
    if key == bam_suffix.lower():
        return "bam"
    if key == ".sam":
        return "sam"
    if key == ".cram":
        return "cram"
    return "other"


def discover_input_files(
    input_data_path: Union[str, Path],
    bam_suffix: str = BAM_SUFFIX,
    recursive: bool = False,
    follow_symlinks: bool = False,
) -> Dict[str, Any]:
    """
    Discover input files under `input_data_path`.

    Returns a dict with:
      - pod5_paths, fast5_paths, fastq_paths, bam_paths, sam_paths, cram_paths,
        h5ad_paths, other_paths (lists of Path)
      - one ``input_is_*`` boolean for each recognized input kind
      - all_files_searched (int)

    Behavior:
      - If `input_data_path` is a file, returns that single file categorized.
      - If a directory, scans immediate children (recursive=False) or entire tree (recursive=True).
      - Handles multi-suffix files like .fastq.gz, .fq.xz, etc.
    """
    p = Path(input_data_path).resolve()

    # normalize bam suffix with a leading dot and lower
    if not bam_suffix.startswith("."):
        bam_suffix = "." + bam_suffix
    bam_suffix = bam_suffix.lower()

    pod5_paths: List[Path] = []
    fast5_paths: List[Path] = []
    fastq_paths: List[Path] = []
    bam_paths: List[Path] = []
    sam_paths: List[Path] = []
    cram_paths: List[Path] = []
    h5ad_paths: List[Path] = []
    other_paths: List[Path] = []

    buckets = {
        "pod5": pod5_paths,
        "fast5": fast5_paths,
        "fastq": fastq_paths,
        "h5ad": h5ad_paths,
        "bam": bam_paths,
        "sam": sam_paths,
        "cram": cram_paths,
        "other": other_paths,
    }

    def categorize_file(fp: Path) -> None:
        buckets[input_kind_for_path(fp, bam_suffix=bam_suffix)].append(fp)

    if not p.exists():
        raise FileNotFoundError(f"input_data_path does not exist: {input_data_path}")

    total_searched = 0

    if p.is_file():
        total_searched = 1
        categorize_file(p)
    else:
        # Directory scan
        if recursive:
            # Python 3.12+ supports follow_symlinks in glob/rglob. Fallback for older versions.
            try:
                iterator = p.rglob("*", follow_symlinks=follow_symlinks)  # type: ignore[call-arg]
            except TypeError:
                iterator = p.rglob("*")  # follow_symlinks not supported
        else:
            iterator = p.iterdir()

        for fp in iterator:
            if not fp.is_file():
                continue
            total_searched += 1
            categorize_file(fp)

    return {
        "pod5_paths": sorted(pod5_paths),
        "fast5_paths": sorted(fast5_paths),
        "fastq_paths": sorted(fastq_paths),
        "bam_paths": sorted(bam_paths),
        "sam_paths": sorted(sam_paths),
        "cram_paths": sorted(cram_paths),
        "h5ad_paths": sorted(h5ad_paths),
        "other_paths": sorted(other_paths),
        "input_is_pod5": len(pod5_paths) > 0,
        "input_is_fast5": len(fast5_paths) > 0,
        "input_is_fastq": len(fastq_paths) > 0,
        "input_is_bam": len(bam_paths) > 0,
        "input_is_sam": len(sam_paths) > 0,
        "input_is_cram": len(cram_paths) > 0,
        "input_is_h5ad": len(h5ad_paths) > 0,
        "all_files_searched": total_searched,
    }

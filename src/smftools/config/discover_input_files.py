from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Union

from smftools.constants import BAM_SUFFIX


def discover_input_files(
    input_data_path: Union[str, Path],
    bam_suffix: str = BAM_SUFFIX,
    recursive: bool = False,
    follow_symlinks: bool = False,
) -> Dict[str, Any]:
    """
    Discover input files under `input_data_path`.

    Returns a dict with:
      - pod5_paths, fast5_paths, fastq_paths, bam_paths, other_paths (lists of Path)
      - input_is_pod5, input_is_fast5, input_is_fastq, input_is_bam (bools)
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

    # Sets of canonical extension keys we’ll compare against
    pod5_exts = {".pod5", ".p5"}
    fast5_exts = {".fast5", ".f5"}
    fastq_exts = {
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
    h5ad_exts = {".h5ad", ".h5"}
    compressed_exts = {".gz", ".bz2", ".xz", ".zst"}

    def ext_key(pp: Path) -> str:
        """
        A robust extension key: last suffix, or last two if the final one is a compressor (.gz/.bz2/.xz/.zst).
        Examples:
          a.fastq.gz -> ".fastq.gz"
          a.fq.xz    -> ".fq.xz"
          a.bam      -> ".bam"
          a          -> ""
        """
        suff = [s.lower() for s in pp.suffixes]
        if not suff:
            return ""
        if suff[-1] in compressed_exts and len(suff) >= 2:
            return suff[-2] + suff[-1]
        return suff[-1]

    pod5_paths: List[Path] = []
    fast5_paths: List[Path] = []
    fastq_paths: List[Path] = []
    bam_paths: List[Path] = []
    h5ad_paths: List[Path] = []
    other_paths: List[Path] = []

    def categorize_file(fp: Path) -> None:
        key = ext_key(fp)
        if key in pod5_exts:
            pod5_paths.append(fp)
        elif key in fast5_exts:
            fast5_paths.append(fp)
        elif key in fastq_exts:
            fastq_paths.append(fp)
        elif key in h5ad_exts:
            h5ad_paths.append(fp)
        elif key == bam_suffix:
            bam_paths.append(fp)
        else:
            other_paths.append(fp)

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
        "h5ad_paths": sorted(h5ad_paths),
        "other_paths": sorted(other_paths),
        "input_is_pod5": len(pod5_paths) > 0,
        "input_is_fast5": len(fast5_paths) > 0,
        "input_is_fastq": len(fastq_paths) > 0,
        "input_is_bam": len(bam_paths) > 0,
        "input_is_h5ad": len(h5ad_paths) > 0,
        "all_files_searched": total_searched,
    }

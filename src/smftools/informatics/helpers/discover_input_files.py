from pathlib import Path
from typing import Dict, List, Any, Tuple

def discover_input_files(
    input_data_path: str,
    bam_suffix: str = ".bam",
    recursive: bool = False,
    follow_symlinks: bool = False,
) -> Dict[str, Any]:
    """
    Discover input files under `input_data_path`.

    Returns a dict with:
      - pod5_paths, fast5_paths, fastq_paths, bam_paths (lists of str)
      - input_is_pod5, input_is_fast5, input_is_fastq, input_is_bam (bools)
      - all_files_searched (int)
    Behavior:
      - If `input_data_path` is a file, returns that single file categorized.
      - If it is a directory, scans either immediate children (recursive=False)
        or entire tree (recursive=True). Uses Path.suffixes to detect .fastq.gz etc.
    """
    p = Path(input_data_path)
    pod5_exts = {".pod5", ".p5"}
    fast5_exts = {".fast5", ".f5"}
    fastq_exts = {".fastq", ".fq", ".fastq.gz", ".fq.gz", ".fastq.xz", ".fq.xz"}
    # normalize bam suffix with leading dot
    if not bam_suffix.startswith("."):
        bam_suffix = "." + bam_suffix
    bam_suffix = bam_suffix.lower()

    pod5_paths: List[str] = []
    fast5_paths: List[str] = []
    fastq_paths: List[str] = []
    bam_paths: List[str] = []
    other_paths: List[str] = []

    def _file_ext_key(pp: Path) -> str:
        # join suffixes to handle .fastq.gz
        return "".join(pp.suffixes).lower() if pp.suffixes else pp.suffix.lower()

    if p.exists() and p.is_file():
        ext_key = _file_ext_key(p)
        if ext_key in pod5_exts:
            pod5_paths.append(str(p))
        elif ext_key in fast5_exts:
            fast5_paths.append(str(p))
        elif ext_key in fastq_exts:
            fastq_paths.append(str(p))
        elif ext_key == bam_suffix:
            bam_paths.append(str(p))
        else:
            other_paths.append(str(p))
        total_searched = 1
    elif p.exists() and p.is_dir():
        if recursive:
            iterator = p.rglob("*")
        else:
            iterator = p.iterdir()
        total_searched = 0
        for fp in iterator:
            if not fp.is_file():
                continue
            total_searched += 1
            ext_key = _file_ext_key(fp)
            if ext_key in pod5_exts:
                pod5_paths.append(str(fp))
            elif ext_key in fast5_exts:
                fast5_paths.append(str(fp))
            elif ext_key in fastq_exts:
                fastq_paths.append(str(fp))
            elif ext_key == bam_suffix:
                bam_paths.append(str(fp))
            else:
                # additional heuristic: check filename contains extension fragments (.pod5 etc)
                name = fp.name.lower()
                if any(e in name for e in pod5_exts):
                    pod5_paths.append(str(fp))
                elif any(e in name for e in fast5_exts):
                    fast5_paths.append(str(fp))
                elif any(e in name for e in [".fastq", ".fq"]):
                    fastq_paths.append(str(fp))
                elif name.endswith(bam_suffix):
                    bam_paths.append(str(fp))
                else:
                    other_paths.append(str(fp))
    else:
        raise FileNotFoundError(f"input_data_path does not exist: {input_data_path}")

    return {
        "pod5_paths": sorted(pod5_paths),
        "fast5_paths": sorted(fast5_paths),
        "fastq_paths": sorted(fastq_paths),
        "bam_paths": sorted(bam_paths),
        "other_paths": sorted(other_paths),
        "input_is_pod5": len(pod5_paths) > 0,
        "input_is_fast5": len(fast5_paths) > 0,
        "input_is_fastq": len(fastq_paths) > 0,
        "input_is_bam": len(bam_paths) > 0,
        "all_files_searched": total_searched,
    }

from __future__ import annotations

import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable, Optional, Tuple, List

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
        out_idx  = bam_qc_dir / f"{base}_idxstats.txt"

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

# def bam_qc(bam_files, bam_qc_dir, threads, modality, stats=True, flagstats=True, idxstats=True):
#     """
#     Performs QC on BAM files by running samtools stats, flagstat, and idxstats.
    
#     Parameters:
#     - bam_files: List of BAM file paths.
#     - bam_qc_dir: Directory to save QC reports.
#     - threads: Number threads to use.
#     - modality: 'conversion' or 'direct' (affects processing mode).
#     - stats: Run `samtools stats` if True.
#     - flagstats: Run `samtools flagstat` if True.
#     - idxstats: Run `samtools idxstats` if True.
#     """
#     import os
#     import subprocess
    
#     # Ensure the QC output directory exists
#     os.makedirs(bam_qc_dir, exist_ok=True)

#     if threads:
#         threads = str(threads)
#     else:
#         pass

#     for bam in bam_files:
#         bam_name = os.path.basename(bam).replace(".bam", "")  # Extract filename without extension

#         # Run samtools QC commands based on selected options
#         if stats:
#             stats_out = os.path.join(bam_qc_dir, f"{bam_name}_stats.txt")
#             if threads:
#                 command = ["samtools", "stats", "-@", threads, bam]
#             else: 
#                 command = ["samtools", "stats", bam]
#             print(f"Running: {' '.join(command)} > {stats_out}")
#             with open(stats_out, "w") as out_file:
#                 subprocess.run(command, stdout=out_file)

#         if flagstats:
#             flagstats_out = os.path.join(bam_qc_dir, f"{bam_name}_flagstat.txt")
#             if threads:
#                 command = ["samtools", "flagstat", "-@", threads, bam]
#             else:
#                 command = ["samtools", "flagstat", bam]
#             print(f"Running: {' '.join(command)} > {flagstats_out}")
#             with open(flagstats_out, "w") as out_file:
#                 subprocess.run(command, stdout=out_file)

#         if idxstats:
#             idxstats_out = os.path.join(bam_qc_dir, f"{bam_name}_idxstats.txt")
#             if threads:
#                 command = ["samtools", "idxstats", "-@", threads, bam]
#             else:
#                 command = ["samtools", "idxstats", bam]
#             print(f"Running: {' '.join(command)} > {idxstats_out}")
#             with open(idxstats_out, "w") as out_file:
#                 subprocess.run(command, stdout=out_file)

#         if modality == 'conversion':
#             pass
#         elif modality == 'direct':
#             pass

#     print("QC processing completed.")   
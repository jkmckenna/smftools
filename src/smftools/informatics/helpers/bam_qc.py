## bam_qc

def bam_qc(bam_files, bam_qc_dir, threads, modality, stats=True, flagstats=True, idxstats=True):
    """
    Performs QC on BAM files by running samtools stats, flagstat, and idxstats.
    
    Parameters:
    - bam_files: List of BAM file paths.
    - bam_qc_dir: Directory to save QC reports.
    - threads: Number threads to use.
    - modality: 'conversion' or 'direct' (affects processing mode).
    - stats: Run `samtools stats` if True.
    - flagstats: Run `samtools flagstat` if True.
    - idxstats: Run `samtools idxstats` if True.
    """
    import os
    import subprocess
    
    # Ensure the QC output directory exists
    os.makedirs(bam_qc_dir, exist_ok=True)

    if threads:
        threads = str(threads)
    else:
        pass

    for bam in bam_files:
        bam_name = os.path.basename(bam).replace(".bam", "")  # Extract filename without extension

        # Run samtools QC commands based on selected options
        if stats:
            stats_out = os.path.join(bam_qc_dir, f"{bam_name}_stats.txt")
            if threads:
                command = ["samtools", "stats", "-@", threads, bam]
            else: 
                command = ["samtools", "stats", bam]
            print(f"Running: {' '.join(command)} > {stats_out}")
            with open(stats_out, "w") as out_file:
                subprocess.run(command, stdout=out_file)

        if flagstats:
            flagstats_out = os.path.join(bam_qc_dir, f"{bam_name}_flagstat.txt")
            if threads:
                command = ["samtools", "flagstat", "-@", threads, bam]
            else:
                command = ["samtools", "flagstat", bam]
            print(f"Running: {' '.join(command)} > {flagstats_out}")
            with open(flagstats_out, "w") as out_file:
                subprocess.run(command, stdout=out_file)

        if idxstats:
            idxstats_out = os.path.join(bam_qc_dir, f"{bam_name}_idxstats.txt")
            if threads:
                command = ["samtools", "idxstats", "-@", threads, bam]
            else:
                command = ["samtools", "idxstats", bam]
            print(f"Running: {' '.join(command)} > {idxstats_out}")
            with open(idxstats_out, "w") as out_file:
                subprocess.run(command, stdout=out_file)

        if modality == 'conversion':
            pass
        elif modality == 'direct':
            pass

    print("QC processing completed.")   
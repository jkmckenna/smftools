def aligned_BAM_to_bed(aligned_BAM, out_dir, fasta, make_bigwigs, threads=None):
    """
    Takes an aligned BAM as input and writes a BED file of reads as output.
    Bed columns are: Record name, start position, end position, read length, read name.

    Parameters:
        aligned_BAM (str): Path to an input aligned_BAM to extract to a BED file.
        out_dir (str): Directory to output files.
        fasta (str): File path to the reference genome.
        make_bigwigs (bool): Whether to generate bigwig files.
        threads (int): Number of threads to use.

    Returns:
        None
    """
    import subprocess
    import os
    import concurrent.futures
    from concurrent.futures import ProcessPoolExecutor
    from .bed_to_bigwig import bed_to_bigwig
    from . import make_dirs
    from .plot_read_length_and_coverage_histograms import plot_read_length_and_coverage_histograms

    threads = threads or os.cpu_count()  # Use max available cores if not specified

    # Create necessary directories
    plotting_dir = os.path.join(out_dir, "bed_cov_histograms")
    bed_dir = os.path.join(out_dir, "beds")
    make_dirs([plotting_dir, bed_dir])

    bed_output = os.path.join(bed_dir, os.path.basename(aligned_BAM).replace(".bam", "_bed.bed"))

    print(f"Creating BED from BAM: {aligned_BAM} using {threads} threads...")

    # Convert BAM to BED format
    with open(bed_output, "w") as output_file:
        samtools_view = subprocess.Popen(["samtools", "view", "-@", str(threads), aligned_BAM], stdout=subprocess.PIPE)
        awk_process = subprocess.Popen(
            ["awk", '{print $3 "\t" $4 "\t" $4+length($10)-1 "\t" length($10)-1 "\t" $1}'],
            stdin=samtools_view.stdout,
            stdout=output_file
        )

    samtools_view.stdout.close()
    awk_process.wait()
    samtools_view.wait()

    print(f"BED file created: {bed_output}")

    def split_bed(bed):
        """Splits BED into aligned and unaligned reads."""
        aligned = bed.replace(".bed", "_aligned.bed")
        unaligned = bed.replace(".bed", "_unaligned.bed")

        with open(bed, "r") as infile, open(aligned, "w") as aligned_out, open(unaligned, "w") as unaligned_out:
            for line in infile:
                (unaligned_out if line.startswith("*") else aligned_out).write(line)

        os.remove(bed)
        return aligned

    print(f"Splitting BED: {bed_output}")
    aligned_bed = split_bed(bed_output)

    with ProcessPoolExecutor() as executor:  # Use processes instead of threads
        futures = []
        futures.append(executor.submit(plot_read_length_and_coverage_histograms, aligned_bed, plotting_dir))
        if make_bigwigs:
            futures.append(executor.submit(bed_to_bigwig, fasta, aligned_bed))

        # Wait for all tasks to complete
        concurrent.futures.wait(futures)

    print("Processing completed successfully.")
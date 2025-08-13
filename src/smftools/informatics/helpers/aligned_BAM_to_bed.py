def aligned_BAM_to_bed(aligned_BAM, out_dir, fasta, make_bigwigs, threads=None):
    """
    Takes an aligned BAM as input and writes a BED file of reads as output.
    Bed columns are: Record name, start position, end position, read length, read name, mapping quality, read quality.

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
    import pysam
    import numpy as np
    import concurrent.futures
    from concurrent.futures import ProcessPoolExecutor
    from .bed_to_bigwig import bed_to_bigwig
    from . import make_dirs
    from .plot_bed_histograms import plot_bed_histograms

    threads = threads or os.cpu_count()  # Use max available cores if not specified

    # Create necessary directories
    plotting_dir = os.path.join(out_dir, "bed_cov_histograms")
    bed_dir = os.path.join(out_dir, "beds")
    make_dirs([plotting_dir, bed_dir])

    bed_output = os.path.join(bed_dir, os.path.basename(aligned_BAM).replace(".bam", "_bed.bed"))

    print(f"Creating BED-like file from BAM (with MAPQ and avg base quality): {aligned_BAM}")

    with pysam.AlignmentFile(aligned_BAM, "rb") as bam, open(bed_output, "w") as out:
        for read in bam.fetch(until_eof=True):
            if read.is_unmapped:
                chrom = "*"
                start1 = 1
                rl = read.query_length or 0
                mapq = 0
            else:
                chrom = bam.get_reference_name(read.reference_id)
                # pysam reference_start is 0-based → +1 for 1-based SAM-like start
                start1 = int(read.reference_start) + 1
                rl = read.query_length or 0
                mapq = int(read.mapping_quality)

            # End position in 1-based inclusive coords
            end1 = start1 + (rl or 0) - 1

            qname = read.query_name
            quals = read.query_qualities
            if quals is None or rl == 0:
                avg_q = float("nan")
            else:
                avg_q = float(np.mean(quals))

            out.write(f"{chrom}\t{start1}\t{end1}\t{rl}\t{qname}\t{mapq}\t{avg_q:.3f}\n")

    print(f"BED-like file created: {bed_output}")

    def split_bed(bed):
        """Splits into aligned and unaligned reads (chrom == '*')."""
        aligned = bed.replace(".bed", "_aligned.bed")
        unaligned = bed.replace(".bed", "_unaligned.bed")
        with open(bed, "r") as infile, open(aligned, "w") as aligned_out, open(unaligned, "w") as unaligned_out:
            for line in infile:
                (unaligned_out if line.startswith("*\t") else aligned_out).write(line)
        os.remove(bed)
        return aligned

    print(f"Splitting: {bed_output}")
    aligned_bed = split_bed(bed_output)

    with ProcessPoolExecutor() as executor:
        futures = []
        futures.append(executor.submit(plot_bed_histograms, aligned_bed, plotting_dir, fasta))
        if make_bigwigs:
            futures.append(executor.submit(bed_to_bigwig, fasta, aligned_bed))
        concurrent.futures.wait(futures)

    print("Processing completed successfully.")